const videoInput = document.getElementById('videoInput');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const statusBox = document.getElementById('status');
const results = document.getElementById('results');
const totalShotsEl = document.getElementById('totalShots');
const totalMakesEl = document.getElementById('totalMakes');
const totalPctEl = document.getElementById('totalPct');
const zoneTable = document.getElementById('zoneTable');
const shotTable = document.getElementById('shotTable');
const warningsEl = document.getElementById('warnings');
const courtScrollHintEl = document.getElementById('courtScrollHint');
const rimScrollHintEl = document.getElementById('rimScrollHint');
const clipViewerEl = document.getElementById('clipViewer');
const clipVideoEl = document.getElementById('clipVideo');
const clipMetaEl = document.getElementById('clipMeta');
const courtCalibrationEl = document.getElementById('courtCalibration');
const courtCalibImage = document.getElementById('courtCalibImage');
const courtCalibCanvas = document.getElementById('courtCalibCanvas');
const courtConfirmBtn = document.getElementById('courtConfirm');
const courtResetBtn = document.getElementById('courtReset');
const courtDismissBtn = document.getElementById('courtDismiss');
const netZoneCalibrationEl = document.getElementById('rimCalibration');
const netZoneCalibImage = document.getElementById('rimCalibImage');
const netZoneCalibCanvas = document.getElementById('rimCalibCanvas');
const netZoneConfirmBtn = document.getElementById('rimConfirm');
const netZoneDismissBtn = document.getElementById('rimDismiss');
const netZoneHintEl = document.getElementById('rimCalibHint');
const resetCalibrationBtn = document.getElementById('resetCalibrationBtn');
const collectTrainingDataEl = document.getElementById('collectTrainingData');
let statusPoll = null;
let selectedFile = null;
let courtSelection = [];
let courtCalibrationMeta = null;
let netZoneSelection = null;
let netZoneCalibrationMeta = null;
let calibrationState = null;
let currentResults = null;
const SHOTLAB_CAMERA_MODE = 'behind_basket';
const SHOTLAB_CAMERA_POSITION_OVERRIDE = 'in_front_of_shooter';

const ZONE_LABELS = {
    left: 'Left',
    center: 'Center',
    right: 'Right',
    unknown: 'Unknown'
};

const ZONE_ALIASES = {
    left_corner_3: 'left',
    left_baseline_2: 'left',
    left_wing_2: 'left',
    left_wing_3: 'left',
    left_corner: 'left',
    left_wing: 'left',
    baseline_left: 'left',
    right_corner_3: 'right',
    right_baseline_2: 'right',
    right_wing_2: 'right',
    right_wing_3: 'right',
    right_corner: 'right',
    right_wing: 'right',
    baseline_right: 'right',
    restricted_area: 'center',
    paint: 'center',
    mid_range: 'center',
    top_of_key_3: 'center',
    top_of_key: 'center'
};

function normalizeZoneId(zone) {
    if (!zone) return 'unknown';
    const key = String(zone).trim().toLowerCase();
    if (key === 'left' || key === 'center' || key === 'right' || key === 'unknown') {
        return key;
    }
    return ZONE_ALIASES[key] || key;
}

function normalizeZoneStats(zoneStats = {}) {
    const merged = {};
    Object.entries(zoneStats || {}).forEach(([rawZone, rawStats]) => {
        const zone = normalizeZoneId(rawZone);
        const stats = rawStats || {};
        const attempts = Number(stats.attempts) || 0;
        const makes = Number(stats.makes) || 0;
        const percentage = Number(stats.percentage);
        const score = Number(stats.shotsync_score);
        if (!merged[zone]) {
            merged[zone] = {
                attempts: 0,
                makes: 0,
                benchmark: null,
                benchmark_std: null,
                _score_weighted_sum: 0,
                _score_weighted_count: 0
            };
        }
        merged[zone].attempts += attempts;
        merged[zone].makes += makes;
        if (merged[zone].benchmark === null && stats.benchmark !== undefined && stats.benchmark !== null) {
            merged[zone].benchmark = Number(stats.benchmark);
        }
        if (merged[zone].benchmark_std === null && stats.benchmark_std !== undefined && stats.benchmark_std !== null) {
            merged[zone].benchmark_std = Number(stats.benchmark_std);
        }
        if (Number.isFinite(score)) {
            const weight = attempts > 0 ? attempts : (Number.isFinite(percentage) ? 1 : 0);
            if (weight > 0) {
                merged[zone]._score_weighted_sum += score * weight;
                merged[zone]._score_weighted_count += weight;
            }
        }
    });

    const normalized = {};
    Object.entries(merged).forEach(([zone, entry]) => {
        const attempts = entry.attempts;
        const makes = entry.makes;
        normalized[zone] = {
            attempts,
            makes,
            percentage: attempts > 0 ? (makes / attempts) * 100.0 : 0.0,
            shotsync_score: entry._score_weighted_count > 0
                ? entry._score_weighted_sum / entry._score_weighted_count
                : null,
            benchmark: entry.benchmark,
            benchmark_std: entry.benchmark_std
        };
    });
    return normalized;
}

const COURT_LANDMARKS = [
    { key: 'basket', label: 'Basket Center', required: true },
    { key: 'free_throw', label: 'Free Throw Line', required: true },
    { key: 'left_wing', label: 'Shooter-Left 3PT Shoulder', required: true },
    { key: 'right_wing', label: 'Shooter-Right 3PT Shoulder', required: true },
    { key: 'baseline_left', label: 'Shooter-Left Baseline Corner', required: false },
    { key: 'baseline_right', label: 'Shooter-Right Baseline Corner', required: false },
    { key: 'top_of_key', label: 'Top of Key', required: false }
];

const LANDMARK_HELP = {
    basket: {
        title: 'Basket center',
        desc: 'Click the middle of the hoop ring (center of the circle).',
        tip: 'Not the net or the pole.'
    },
    free_throw: {
        title: 'Free throw line center',
        desc: 'Click the middle of the free throw line (15 ft from the hoop).',
        tip: 'It is a straight line across the lane.'
    },
    left_wing: {
        title: 'Shooter-left 3PT shoulder',
        desc: 'Click where the shooter-left 3-point arc meets the straight side segment.',
        tip: 'Shooter-left means player left while facing the hoop.'
    },
    right_wing: {
        title: 'Shooter-right 3PT shoulder',
        desc: 'Click where the shooter-right 3-point arc meets the straight side segment.',
        tip: 'Shooter-right means player right while facing the hoop.'
    },
    baseline_left: {
        title: 'Shooter-left baseline corner',
        desc: 'Click the baseline corner on the shooter-left side (corner three spot).',
        tip: 'This can appear on the right side of your screen in sideline views.'
    },
    baseline_right: {
        title: 'Shooter-right baseline corner',
        desc: 'Click the baseline corner on the shooter-right side (corner three spot).',
        tip: 'This can appear on the left side of your screen in sideline views.'
    },
    top_of_key: {
        title: 'Top of key',
        desc: 'Click the top of the 3PT arc, straight above the hoop.',
        tip: 'Optional, improves accuracy.'
    }
};

const COURT_COLORS = {
    basket: '#ff4d4d',
    free_throw: '#4dff8f',
    left_wing: '#4da3ff',
    right_wing: '#ff7ad1',
    baseline_left: '#a37dff',
    baseline_right: '#ffb74d',
    top_of_key: '#ffd54d'
};

const COURT_MARKER_LABELS = {
    basket: 'B',
    free_throw: 'FT',
    left_wing: 'SLW',
    right_wing: 'SRW',
    baseline_left: 'SBL',
    baseline_right: 'SBR',
    top_of_key: 'TOK'
};

const CAMERA_POSITION_OPTIONS = [
    { value: 'behind_shooter', label: 'Behind shooter' },
    { value: 'sideline_right', label: "Right side angle (camera on shooter's right)" },
    { value: 'sideline_left', label: "Left side angle (camera on shooter's left)" },
    { value: 'auto', label: 'Use automatic estimate' }
];

function normalizeCameraPositionOverride(value) {
    const normalized = String(value || '').trim().toLowerCase();
    if (!normalized) return null;
    const allowed = new Set(['behind_shooter', 'sideline_right', 'sideline_left', 'auto']);
    return allowed.has(normalized) ? normalized : null;
}

function cameraPositionLabel(value) {
    const normalized = normalizeCameraPositionOverride(value);
    const lookup = {
        behind_shooter: 'Behind Shooter',
        sideline_right: 'Right Sideline',
        sideline_left: 'Left Sideline',
        auto: 'Automatic Estimate'
    };
    return normalized ? (lookup[normalized] || normalized) : 'Not selected';
}

function swapLeftRightLandmarks(landmarks) {
    if (!landmarks) return;
    const pairs = [
        ['left_wing', 'right_wing'],
        ['baseline_left', 'baseline_right']
    ];
    pairs.forEach(([leftKey, rightKey]) => {
        const left = landmarks[leftKey] ? { ...landmarks[leftKey] } : null;
        const right = landmarks[rightKey] ? { ...landmarks[rightKey] } : null;
        if (right) {
            landmarks[leftKey] = right;
        } else {
            delete landmarks[leftKey];
        }
        if (left) {
            landmarks[rightKey] = left;
        } else {
            delete landmarks[rightKey];
        }
    });
}

function estimateCameraPositionPreview(landmarks, frameWidth, frameHeight) {
    if (!landmarks || !frameWidth || !frameHeight || !landmarks.basket) {
        return { position: 'unknown', confidence: 0 };
    }

    const basket = landmarks.basket;
    const basketXRatio = basket.x / frameWidth;
    const baselineLeft = landmarks.baseline_left;
    const baselineRight = landmarks.baseline_right;

    if (baselineLeft && baselineRight) {
        const baselineXSpan = Math.abs(baselineRight.x - baselineLeft.x);
        const baselineYDiff = Math.abs(baselineRight.y - baselineLeft.y);
        const baselineXSpanRatio = baselineXSpan / frameWidth;
        if (baselineXSpanRatio > 0.4 && baselineYDiff > 100) {
            if (baselineRight.y > baselineLeft.y) {
                return { position: 'sideline_right', confidence: 0.85 };
            }
            return { position: 'sideline_left', confidence: 0.85 };
        }
    }

    const leftWing = landmarks.left_wing;
    const rightWing = landmarks.right_wing;
    if (leftWing && rightWing) {
        const leftDist = basket.x - leftWing.x;
        const rightDist = rightWing.x - basket.x;
        const isCentered = basketXRatio > 0.35 && basketXRatio < 0.65;
        const wingsOnOppositeSides = leftDist > 0 && rightDist > 0;
        const wingRatio = rightDist > 0 ? (leftDist / rightDist) : 0;
        const wingsSymmetric = wingsOnOppositeSides && wingRatio > 0.5 && wingRatio < 2.0;
        if (isCentered && wingsSymmetric) {
            return { position: 'behind_shooter', confidence: 0.8 };
        }
    }

    if (basketXRatio < 0.3) {
        return { position: 'sideline_right', confidence: 0.6 };
    }
    if (basketXRatio > 0.7) {
        return { position: 'sideline_left', confidence: 0.6 };
    }
    return { position: 'unknown', confidence: 0.4 };
}

function updateCourtOrientationPreview() {
    if (!calibrationState || calibrationState.mode !== 'court') return;
    const previewEl = calibrationState.modal.querySelector('#landmark-orientation');
    if (!previewEl) return;

    const manualOverride = normalizeCameraPositionOverride(calibrationState.cameraPositionOverride);
    if (manualOverride && manualOverride !== 'auto') {
        previewEl.classList.add('ready');
        previewEl.innerHTML = `<strong>Camera side preview:</strong> <span class="orientation-badge">${cameraPositionLabel(manualOverride)}</span> (manual override selected).`;
        return;
    }

    const prediction = estimateCameraPositionPreview(
        calibrationState.landmarks,
        calibrationState.canvas.width,
        calibrationState.canvas.height
    );
    const position = prediction.position;
    const confidencePct = Math.round((prediction.confidence || 0) * 100);

    if (position === 'unknown') {
        previewEl.classList.remove('ready');
        previewEl.innerHTML = 'Camera side preview: add basket + baseline corners to verify side labeling.';
        return;
    }

    const label = position === 'sideline_right'
        ? 'Right Sideline'
        : position === 'sideline_left'
            ? 'Left Sideline'
            : position === 'behind_shooter'
                ? 'Behind Shooter'
                : 'Unknown';
    previewEl.classList.add('ready');
    previewEl.innerHTML = `<strong>Camera side preview:</strong> <span class="orientation-badge">${label}</span> (${confidencePct}% conf). If this looks flipped, use "Swap Shooter Left/Right".`;
}

videoInput.addEventListener('change', () => {
    if (videoInput.files && videoInput.files[0]) {
        selectedFile = videoInput.files[0];
        fileName.textContent = selectedFile.name;
        analyzeBtn.disabled = false;
        courtSelection = [];
        courtCalibrationMeta = null;
        netZoneSelection = null;
        netZoneCalibrationMeta = null;
        hideClipViewer();
        hideCourtCalibration();
        hideNetZoneCalibration();
    } else {
        selectedFile = null;
        fileName.textContent = 'No file selected';
        analyzeBtn.disabled = true;
    }
});

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    runAnalysis();
});

if (resetCalibrationBtn) {
    resetCalibrationBtn.addEventListener('click', async () => {
        const confirmReset = window.confirm(
            'Clear any saved calibration data?'
        );
        if (!confirmReset) return;
        try {
            await fetch('/api/reset_calibration', { method: 'POST' });
            setStatus('Calibration data cleared.');
        } catch (error) {
            setStatus('Unable to clear calibration.');
        }
    });
}


function setStatus(message) {
    statusBox.textContent = message;
    statusBox.classList.remove('hidden');
}

function showProgress(message) {
    setStatus(message);
    statusBox.classList.add('loading');
}

function hideProgress() {
    statusBox.classList.remove('loading');
}

function startStatusPoll() {
    stopStatusPoll();
    statusPoll = setInterval(async () => {
        try {
            const response = await fetch('/api/shotlab_status');
            if (!response.ok) return;
            const data = await response.json();
            if (!data || !data.stage) return;
            const pct = data.progress !== undefined ? Math.round(data.progress * 100) : null;
            const detail = data.detail ? ` — ${data.detail}` : '';
            const message = pct !== null
                ? `Processing (${data.stage}) ${pct}%${detail}`
                : `Processing (${data.stage})${detail}`;
            setStatus(message);
        } catch (e) {
            // ignore polling errors
        }
    }, 2000);
}

function stopStatusPoll() {
    if (statusPoll) {
        clearInterval(statusPoll);
        statusPoll = null;
    }
}

function showCalibrationUI(frameBase64, type, autoCalibration) {
    const existing = document.getElementById('calibration-modal');
    if (existing) {
        existing.remove();
    }

    const mode = type === 'court' ? 'court' : 'net_zone';
    const autoNetZone = autoCalibration && autoCalibration.net_zone;
    const autoCourtLandmarks = autoCalibration && autoCalibration.landmarks;
    const autoConfidence = autoCalibration && typeof autoCalibration.confidence === 'number'
        ? autoCalibration.confidence
        : null;
    const autoMarkup = (autoNetZone || autoCourtLandmarks)
        ? `<div class="calibration-auto">Auto-detected ${mode === 'court' ? 'basket' : 'net zone'}${autoConfidence !== null ? ` (${Math.round(autoConfidence * 100)}% conf)` : ''}. Confirm or adjust.</div>`
        : '';

    const steps = [
        'Top-left corner at rim height',
        'Bottom-right corner at net bottom'
    ];
    const stepMarkup = steps.map((label, idx) => `
        <div class="instruction-step ${idx === 0 ? 'active' : ''}" id="calib-step-${idx}">
            <span class="step-number">${idx + 1}</span>
            <p>${label}</p>
        </div>
    `).join('');

    const landmarkMarkup = COURT_LANDMARKS.map((landmark) => `
        <button class="landmark-select ${landmark.required ? 'required' : ''}" data-landmark="${landmark.key}">
            <span>${landmark.label}</span>
            ${landmark.required ? '<span class="required-pill">Required</span>' : '<span class="optional-pill">Optional</span>'}
            <span class="landmark-check" data-check="${landmark.key}"></span>
        </button>
    `).join('');

    const presetCameraOverride = normalizeCameraPositionOverride(autoCalibration?.camera_position_override || '');
    const cameraSelectionMarkup = mode === 'court'
        ? `<div class="camera-position-panel"><label for="camera-position-select">Confirm camera angle (required)</label><select id="camera-position-select" class="camera-position-select"><option value="">Select camera angle...</option>${CAMERA_POSITION_OPTIONS.map((opt) => `<option value="${opt.value}"${presetCameraOverride === opt.value ? ' selected' : ''}>${opt.label}</option>`).join('')}</select><p class="camera-position-note">Use this to lock side orientation. If landmarks look mirrored, click "Swap Shooter Left/Right".</p></div>`
        : '';

    const instructionsMarkup = mode === 'court'
        ? `<div class="landmark-grid">${landmarkMarkup}</div>${cameraSelectionMarkup}<div class="landmark-tools"><button class="btn-secondary" id="calibration-swap-sides" type="button">Swap Shooter Left/Right</button><div class="landmark-orientation" id="landmark-orientation">Camera side preview: add landmarks to verify side labeling.</div></div><div class="landmark-status" id="landmark-status"></div>`
        : `<div class="calibration-instructions">${stepMarkup}</div>`;

    const modal = document.createElement('div');
    modal.id = 'calibration-modal';
    modal.className = 'calibration-modal';
    modal.innerHTML = `
        <div class="calibration-content">
            <div class="calibration-header">
                <h2>${mode === 'court' ? 'Quick setup — map the court' : 'Quick setup — mark the basket'}</h2>
                <p class="calibration-subtitle">Pick each landmark on the image. Required: basket + free throw + both shooter-side 3PT shoulders + at least one baseline corner.</p>
            </div>
            ${autoMarkup}
            ${mode === 'court' ? `
                <div class="calibration-help">
                    <div class="help-card">
                        <h3>What to click</h3>
                        <ul class="help-list">
                            <li><strong>Basket center:</strong> middle of the rim circle.</li>
                            <li><strong>Free throw:</strong> middle of the free throw line.</li>
                            <li><strong>Shooter-left/right 3PT shoulder:</strong> where the 3-point arc meets the straight side segment (outside/on the 3PT line).</li>
                            <li><strong>Shooter-left/right baseline corner:</strong> court corner by the baseline (corner three spot).</li>
                        </ul>
                        <p class="help-note">Use shooter perspective: left/right as the player sees while facing the hoop, not screen-left/screen-right.</p>
                    </div>
                    <div class="help-card">
                        <h3>Quick diagram</h3>
                        <div class="court-mini">
                            <svg viewBox="0 0 320 200" role="img" aria-label="Half court landmark guide">
                                <rect x="20" y="16" width="280" height="164" rx="12" ry="12" fill="none" stroke="currentColor" stroke-width="2"/>
                                <line x1="20" y1="180" x2="300" y2="180" stroke="currentColor" stroke-width="2"/>

                                <line x1="58" y1="180" x2="58" y2="112" stroke="currentColor" stroke-width="2" opacity="0.75"/>
                                <line x1="262" y1="180" x2="262" y2="112" stroke="currentColor" stroke-width="2" opacity="0.75"/>
                                <path d="M 58 112 A 102 102 0 0 1 262 112" fill="none" stroke="currentColor" stroke-width="2" opacity="0.75"/>

                                <rect x="116" y="76" width="88" height="104" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.5"/>
                                <line x1="110" y1="112" x2="210" y2="112" stroke="currentColor" stroke-width="1.5" opacity="0.9"/>
                                <circle cx="160" cy="66" r="20" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.65"/>
                                <circle cx="160" cy="30" r="9" fill="none" stroke="currentColor" stroke-width="2"/>

                                <circle cx="160" cy="30" r="3.5" fill="currentColor"/>
                                <circle cx="160" cy="112" r="3.5" fill="currentColor"/>
                                <circle cx="58" cy="112" r="3.5" fill="currentColor"/>
                                <circle cx="262" cy="112" r="3.5" fill="currentColor"/>
                                <circle cx="36" cy="168" r="3.5" fill="currentColor"/>
                                <circle cx="284" cy="168" r="3.5" fill="currentColor"/>

                                <text x="172" y="33" font-size="10" fill="currentColor">B</text>
                                <text x="170" y="115" font-size="10" fill="currentColor">FT</text>
                                <text x="36" y="106" font-size="9" fill="currentColor">SLW</text>
                                <text x="258" y="106" font-size="9" fill="currentColor">SRW</text>
                                <text x="18" y="182" font-size="9" fill="currentColor">SBL</text>
                                <text x="278" y="182" font-size="9" fill="currentColor">SBR</text>
                            </svg>
                        </div>
                        <p class="help-note">B = basket, FT = free throw center, SLW/SRW = shooter-left/right 3PT shoulder, SBL/SBR = shooter-left/right baseline corner.</p>
                    </div>
                </div>
                ${instructionsMarkup}
                <div class="landmark-hint" id="landmark-hint">Select a landmark to see a quick tip.</div>
            ` : instructionsMarkup}
            <div class="canvas-container">
                <canvas id="calibration-canvas"></canvas>
                <div class="helper-overlay" id="calibration-overlay">
                    <div class="crosshair"></div>
                </div>
            </div>
            <div class="calibration-actions">
                <button class="btn-secondary" id="calibration-reset">${mode === 'court' ? 'Reset Landmarks' : 'Start Over'}</button>
                <button class="btn-primary" id="calibration-use" disabled>Use This Setting</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    const canvas = modal.querySelector('#calibration-canvas');
    const ctx = canvas.getContext('2d');
    const useBtn = modal.querySelector('#calibration-use');
    const resetBtn = modal.querySelector('#calibration-reset');
    if (autoNetZone && resetBtn) {
        resetBtn.textContent = 'Adjust manually';
    }

    const img = new Image();
    img.onload = () => {
        const maxWidth = 900;
        const scale = Math.min(1, maxWidth / img.width);
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        calibrationState = {
            modal,
            canvas,
            ctx,
            img,
            points: [],
            landmarks: {},
            activeLandmark: null,
            cameraPositionOverride: presetCameraOverride || '',
            scale,
            mode,
            source: autoNetZone || autoCourtLandmarks ? 'auto' : 'manual',
            autoSuggested: Boolean(autoNetZone || autoCourtLandmarks),
            autoCalibration
        };
        canvas.addEventListener('click', handleCalibrationClick);
        canvas.addEventListener('mousemove', handleCalibrationHover);

        if (mode === 'net_zone' && autoNetZone) {
            const x1 = autoNetZone.x1_norm * canvas.width;
            const y1 = autoNetZone.y1_norm * canvas.height;
            const x2 = autoNetZone.x2_norm * canvas.width;
            const y2 = autoNetZone.y2_norm * canvas.height;
            calibrationState.points = [
                { x: x1, y: y1 },
                { x: x2, y: y2 }
            ];
            redrawCalibrationCanvas();
            updateCalibrationSteps();
        }

        if (mode === 'court' && autoCourtLandmarks && autoCourtLandmarks.basket) {
            const basket = autoCourtLandmarks.basket;
            calibrationState.landmarks.basket = {
                x: basket.x_norm * canvas.width,
                y: basket.y_norm * canvas.height
            };
            updateCalibrationSteps();
            redrawCalibrationCanvas();
        }
    };
    if (frameBase64 && frameBase64.startsWith('data:')) {
        img.src = frameBase64;
    } else {
        img.src = `data:image/jpeg;base64,${frameBase64}`;
    }

    if (mode === 'court') {
        const buttons = modal.querySelectorAll('.landmark-select');
        const swapBtn = modal.querySelector('#calibration-swap-sides');
        const cameraSelect = modal.querySelector('#camera-position-select');
        buttons.forEach((btn) => {
            btn.addEventListener('click', () => {
                const key = btn.dataset.landmark;
                calibrationState.activeLandmark = key;
                calibrationState.source = 'manual';
                buttons.forEach((b) => b.classList.remove('active'));
                btn.classList.add('active');
                const hint = modal.querySelector('#landmark-hint');
                if (hint && LANDMARK_HELP[key]) {
                    hint.innerHTML = `<strong>${LANDMARK_HELP[key].title}:</strong> ${LANDMARK_HELP[key].desc} <span>${LANDMARK_HELP[key].tip}</span>`;
                }
            });
        });
        if (swapBtn) {
            swapBtn.addEventListener('click', () => {
                swapLeftRightLandmarks(calibrationState.landmarks);
                calibrationState.activeLandmark = null;
                calibrationState.source = 'manual';
                buttons.forEach((btn) => btn.classList.remove('active'));
                redrawCalibrationCanvas();
                updateCalibrationSteps();
            });
        }
        if (cameraSelect) {
            cameraSelect.addEventListener('change', () => {
                calibrationState.cameraPositionOverride = normalizeCameraPositionOverride(cameraSelect.value) || '';
                updateCalibrationSteps();
            });
        }
    }

    resetBtn.addEventListener('click', resetCalibrationUI);
    useBtn.addEventListener('click', submitCalibration);
}

function handleCalibrationHover(event) {
    if (!calibrationState) return;
    const overlay = calibrationState.modal.querySelector('#calibration-overlay');
    if (!overlay) return;
    const rect = calibrationState.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    overlay.style.left = `${x}px`;
    overlay.style.top = `${y}px`;
}

function handleCalibrationClick(event) {
    if (!calibrationState) return;
    const rect = calibrationState.canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (calibrationState.canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (calibrationState.canvas.height / rect.height);

    if (calibrationState.mode === 'court') {
        if (!calibrationState.activeLandmark) {
            alert('Select a landmark button first.');
            return;
        }
        calibrationState.landmarks[calibrationState.activeLandmark] = { x, y };
        calibrationState.activeLandmark = null;
        const buttons = calibrationState.modal.querySelectorAll('.landmark-select');
        buttons.forEach((btn) => btn.classList.remove('active'));
        redrawCalibrationCanvas();
        updateCalibrationSteps();
        return;
    }

    const maxPoints = 2;
    if (calibrationState.points.length >= maxPoints) return;
    calibrationState.points.push({ x, y });
    redrawCalibrationCanvas();
    updateCalibrationSteps();
}

function redrawCalibrationCanvas() {
    if (!calibrationState) return;
    const { ctx, canvas, img, points, mode, landmarks } = calibrationState;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(111, 255, 143, 0.95)';
    ctx.fillStyle = 'rgba(111, 255, 143, 0.9)';

    if (mode === 'court') {
        Object.entries(landmarks || {}).forEach(([name, pos]) => {
            const color = COURT_COLORS[name] || '#6fff8f';
            ctx.strokeStyle = color;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = '#101010';
            ctx.font = 'bold 12px Space Grotesk, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(COURT_MARKER_LABELS[name] || name.replace('_', ' ').toUpperCase(), pos.x, pos.y - 14);
            ctx.fillStyle = color;
        });
        return;
    }

    if (mode === 'net_zone') {
        if (points[0]) {
            ctx.beginPath();
            ctx.arc(points[0].x, points[0].y, 7, 0, Math.PI * 2);
            ctx.fill();
            ctx.font = 'bold 14px Space Grotesk, sans-serif';
            ctx.fillText('RIM', points[0].x + 12, points[0].y - 8);
        }
        if (points.length >= 2) {
            const p1 = points[0];
            const p2 = points[1];
            const left = Math.min(p1.x, p2.x);
            const top = Math.min(p1.y, p2.y);
            const width = Math.abs(p2.x - p1.x);
            const height = Math.abs(p2.y - p1.y);
            ctx.strokeRect(left, top, width, height);
            ctx.fillStyle = 'rgba(111, 255, 143, 0.12)';
            ctx.fillRect(left, top, width, height);
            ctx.fillStyle = 'rgba(111, 255, 143, 0.9)';
            ctx.fillText('NET', left + width - 36, top + height + 18);
        }
        return;
    }

    if (mode === 'court') {
        if (points.length >= 2) {
            ctx.beginPath();
            points.forEach((pt, idx) => {
                if (idx === 0) {
                    ctx.moveTo(pt.x, pt.y);
                } else {
                    ctx.lineTo(pt.x, pt.y);
                }
            });
            ctx.stroke();
        }
        points.forEach((pt, idx) => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#101010';
            ctx.font = 'bold 12px Space Grotesk, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(idx + 1), pt.x, pt.y + 0.5);
            ctx.fillStyle = 'rgba(111, 255, 143, 0.9)';
        });
    }
}

function updateCalibrationSteps() {
    if (!calibrationState) return;
    if (calibrationState.mode === 'court') {
        const buttons = calibrationState.modal.querySelectorAll('.landmark-select');
        buttons.forEach((btn) => {
            const key = btn.dataset.landmark;
            const check = calibrationState.modal.querySelector(`[data-check=\"${key}\"]`);
            if (calibrationState.landmarks[key]) {
                btn.classList.add('selected');
                if (check) check.textContent = '✓';
            } else {
                btn.classList.remove('selected');
                if (check) check.textContent = '';
            }
        });
        const statusEl = calibrationState.modal.querySelector('#landmark-status');
        const landmarksReady = isCourtReady(calibrationState.landmarks);
        const cameraReady = Boolean(normalizeCameraPositionOverride(calibrationState.cameraPositionOverride));
        const ready = landmarksReady && cameraReady;
        if (statusEl) {
            const count = Object.keys(calibrationState.landmarks || {}).length;
            if (ready) {
                statusEl.textContent = `Ready: ${count} landmark(s) selected, camera angle confirmed (${cameraPositionLabel(calibrationState.cameraPositionOverride)}).`;
            } else if (!landmarksReady) {
                statusEl.textContent = `Need basket, free throw, both 3PT shoulders, and one baseline corner (${count} selected).`;
            } else {
                statusEl.textContent = 'Confirm camera angle to continue.';
            }
            statusEl.classList.toggle('ready', ready);
        }
        const useBtn = calibrationState.modal.querySelector('#calibration-use');
        if (useBtn) {
            useBtn.disabled = !ready;
            useBtn.classList.toggle('pulse', ready);
        }
        updateCourtOrientationPreview();
        return;
    }

    const steps = calibrationState.modal.querySelectorAll('.instruction-step');
    steps.forEach((step, idx) => {
        step.classList.remove('active', 'complete');
        if (idx < calibrationState.points.length) {
            step.classList.add('complete');
        } else if (idx === calibrationState.points.length) {
            step.classList.add('active');
        }
    });
    const useBtn = calibrationState.modal.querySelector('#calibration-use');
    const maxPoints = 2;
    if (useBtn) {
        useBtn.disabled = calibrationState.points.length < maxPoints;
        useBtn.classList.toggle('pulse', calibrationState.points.length >= maxPoints);
    }
}

function isCourtReady(landmarks) {
    if (!landmarks) return false;
    if (!landmarks.basket || !landmarks.free_throw) return false;
    if (!landmarks.left_wing || !landmarks.right_wing) return false;
    if (!landmarks.baseline_left && !landmarks.baseline_right) return false;
    return Object.keys(landmarks).length >= 5;
}

function resetCalibrationUI() {
    if (!calibrationState) return;
    const { ctx, canvas, img } = calibrationState;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    calibrationState.points = [];
    calibrationState.landmarks = {};
    calibrationState.source = 'manual';
    if (calibrationState.mode === 'court' && calibrationState.autoCalibration?.landmarks?.basket) {
        const basket = calibrationState.autoCalibration.landmarks.basket;
        calibrationState.landmarks.basket = {
            x: basket.x_norm * canvas.width,
            y: basket.y_norm * canvas.height
        };
        calibrationState.source = 'auto';
    }
    redrawCalibrationCanvas();
    updateCalibrationSteps();
}

async function submitCalibration() {
    if (!calibrationState) return;
    if (calibrationState.mode === 'net_zone' && calibrationState.points.length < 2) return;
    const canvas = calibrationState.canvas;
    if (calibrationState.mode === 'court') {
        if (!isCourtReady(calibrationState.landmarks)) return;
    } else {
        const maxPoints = 2;
        if (calibrationState.points.length < maxPoints) return;
    }

    showProgress('Saving calibration...');
    try {
        let response;
        if (calibrationState.mode === 'court') {
            const landmarks = {};
            Object.entries(calibrationState.landmarks).forEach(([key, pt]) => {
                landmarks[key] = {
                    x_norm: pt.x / canvas.width,
                    y_norm: pt.y / canvas.height
                };
            });
            response = await fetch('/api/save_court_calibration', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    landmarks,
                    camera_position_override: normalizeCameraPositionOverride(calibrationState.cameraPositionOverride),
                    source: calibrationState.source,
                    auto_landmarks: calibrationState.autoCalibration?.landmarks || null
                })
            });
        } else {
            throw new Error('Manual basket/net calibration is no longer supported.');
        }
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Calibration failed');
        }
        if (calibrationState.modal) {
            calibrationState.modal.remove();
        }
        calibrationState = null;
        showProgress('Processing with saved calibration...');
        runAnalysis({ confirmCourtReuse: true });
    } catch (error) {
        hideProgress();
        setStatus(`Calibration failed: ${error.message}`);
    }
}

async function runAnalysis(options = {}) {
    if (!selectedFile) return;
    const opts = (options && typeof options === 'object') ? options : {};
    setStatus('Uploading and analyzing... This can take a minute. Use behind-basket camera facing the shooter.');
    showProgress('Analyzing your shots...');
    analyzeBtn.disabled = true;
    if (courtConfirmBtn) courtConfirmBtn.disabled = true;
    if (netZoneConfirmBtn) netZoneConfirmBtn.disabled = true;
    hideClipViewer();
    startStatusPoll();

    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('camera_mode', SHOTLAB_CAMERA_MODE);
    formData.append('camera_position_override', SHOTLAB_CAMERA_POSITION_OVERRIDE);
    formData.append('confirm_court_reuse', 'true');
    const allowTraining = collectTrainingDataEl ? collectTrainingDataEl.checked : true;
    formData.append('collect_training_data', allowTraining ? 'true' : 'false');
    if (opts.forceCourtRecalibration) {
        formData.append('force_court_recalibration', 'true');
    }

    try {
        const response = await fetch('/api/process_shotlab_v4', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Analysis failed');
        }
        if (data?.calibration?.court_required) {
            const existingWarnings = Array.isArray(data.warnings) ? data.warnings.slice() : [];
            existingWarnings.push('court_transform_unavailable');
            data.warnings = existingWarnings;
        }
        renderResults(data);
        setStatus('Analysis complete.');
    } catch (error) {
        setStatus(`Error: ${error.message}`);
    } finally {
        stopStatusPoll();
        hideProgress();
        analyzeBtn.disabled = false;
    }
}

function renderResults(data) {
    const shots = Array.isArray(data.shots)
        ? data.shots.map((shot) => ({
            ...shot,
            zone: normalizeZoneId(shot.zone || shot.shooting_zone || 'unknown')
        }))
        : [];
    const inputZoneStats = data.zone_stats || {};
    const normalizedZoneStats = normalizeZoneStats(inputZoneStats);
    const fallbackZoneStats = Object.keys(normalizedZoneStats).length > 0
        ? normalizedZoneStats
        : buildZoneStatsFromShots(shots, {});

    currentResults = {
        ...data,
        zone_stats: fallbackZoneStats,
        shots
    };
    results.classList.remove('hidden');

    const totalAttempts = currentResults.total_attempts || 0;
    const totalMakes = currentResults.total_makes || 0;
    const pct = totalAttempts > 0 ? (totalMakes / totalAttempts) * 100 : 0;

    totalShotsEl.textContent = totalAttempts;
    totalMakesEl.textContent = totalMakes;
    totalPctEl.textContent = `${pct.toFixed(1)}%`;

    renderHeatmap(currentResults.zone_stats || {});

    zoneTable.innerHTML = '';
    const zones = Object.entries(currentResults.zone_stats || {}).sort((a, b) => b[1].attempts - a[1].attempts);
    if (zones.length === 0) {
        zoneTable.innerHTML = '<tr><td colspan="5">No zone data available.</td></tr>';
    } else {
        zones.forEach(([zone, stats]) => {
            const zoneLabel = ZONE_LABELS[zone] || zone;
            const attempts = stats.attempts || 0;
            const makes = stats.makes || 0;
            const completion = stats.percentage !== undefined ? stats.percentage.toFixed(1) : '0.0';
            const score = stats.shotsync_score !== null && stats.shotsync_score !== undefined
                ? stats.shotsync_score.toFixed(1)
                : '—';
            const benchmark = stats.benchmark !== null && stats.benchmark !== undefined
                ? stats.benchmark.toFixed(1)
                : '—';

            zoneTable.insertAdjacentHTML('beforeend', `
                <tr>
                    <td>${zoneLabel}</td>
                    <td>${attempts}</td>
                    <td>${makes}</td>
                    <td>${completion}%</td>
                    <td>${score} <span class="muted">(bench ${benchmark})</span></td>
                </tr>
            `);
        });
    }

    shotTable.innerHTML = '';
    const shotsForTable = currentResults.shots || [];
    if (shotsForTable.length === 0) {
        shotTable.innerHTML = '<tr><td colspan="4">No shots detected.</td></tr>';
        hideClipViewer();
    } else {
        shotsForTable.forEach((shot) => {
            const zoneId = normalizeZoneId(shot.zone || shot.shooting_zone || 'unknown');
            const zoneLabel = ZONE_LABELS[zoneId] || zoneId || 'Unknown';
            const result = shot.outcome === 'make' ? 'Make' : shot.outcome === 'miss' ? 'Miss' : 'Unknown';
            const badgeClass = shot.outcome === 'make' ? 'make' : shot.outcome === 'miss' ? 'miss' : 'unknown';
            const score = shot.shotsync_score !== null && shot.shotsync_score !== undefined
                ? shot.shotsync_score.toFixed(1)
                : '—';
            const clipUrlAttr = shot.clip_url ? `data-clip-url="${shot.clip_url}"` : '';
            const clipStartAttr = Number.isFinite(shot.clip_start) ? `data-clip-start="${shot.clip_start.toFixed(3)}"` : '';
            const clipEndAttr = Number.isFinite(shot.clip_end) ? `data-clip-end="${shot.clip_end.toFixed(3)}"` : '';
            const videoUrlAttr = shot.video_url ? `data-video-url="${shot.video_url}"` : '';
            const shotMeta = `${zoneLabel} • ${result}`;

            shotTable.insertAdjacentHTML('beforeend', `
                <tr class="shot-row" ${clipUrlAttr} ${clipStartAttr} ${clipEndAttr} ${videoUrlAttr}
                    data-shot-meta="${shotMeta}" data-shot-index="${shot.shot_number - 1}">
                    <td>${shot.shot_number}</td>
                    <td>${zoneLabel}</td>
                    <td><span class="badge ${badgeClass}">${result}</span></td>
                    <td>${score}</td>
                    <td><button class="btn-remove" data-shot-index="${shot.shot_number - 1}">Remove</button></td>
                </tr>
            `);
        });

        // Wire click handlers after rows are rendered.
        const rows = shotTable.querySelectorAll('tr.shot-row');
        rows.forEach((row) => {
            row.addEventListener('click', () => {
                const clipUrl = row.getAttribute('data-clip-url');
                const videoUrl = row.getAttribute('data-video-url');
                const clipStart = parseFloat(row.getAttribute('data-clip-start'));
                const clipEnd = parseFloat(row.getAttribute('data-clip-end'));
                const meta = row.getAttribute('data-shot-meta') || 'Shot replay';
                if (clipUrl || videoUrl) {
                    showClipViewer({ clipUrl, videoUrl, clipStart, clipEnd }, meta);
                }
            });
        });

        const removeButtons = shotTable.querySelectorAll('.btn-remove');
        removeButtons.forEach((btn) => {
            btn.addEventListener('click', (event) => {
                event.stopPropagation();
                const index = Number(btn.getAttribute('data-shot-index'));
                if (Number.isFinite(index)) {
                    removeShotAt(index);
                }
            });
        });
    }

    const warningLabels = {
        auto_rim_detection_failed: 'Automatic rim detection failed (YOLO could not find the rim)',
        court_transform_unavailable: 'Court mapping unavailable (zone mapping disabled for this video)'
    };
    const warningText = (Array.isArray(data.warnings) ? data.warnings : []).map((w) => warningLabels[w] || w);
    const shotsWithNoScore = shotsForTable.filter((shot) => shot.shotsync_score === null || shot.shotsync_score === undefined).length;
    if (shotsForTable.length > 0 && shotsWithNoScore > 0) {
        warningText.push(
            `Form score unavailable on ${shotsWithNoScore}/${shotsForTable.length} shot(s). This can happen when the shooter is occluded by the pole/backboard; make/miss detection still runs.`
        );
    }
    if (warningText.length > 0) {
        warningsEl.classList.remove('hidden');
        warningsEl.textContent = `Notes: ${warningText.join(' ')}`;
    } else {
        warningsEl.classList.add('hidden');
        warningsEl.textContent = '';
    }

}

function buildZoneStatsFromShots(shots, baseZoneStats = {}) {
    const normalizedBase = normalizeZoneStats(baseZoneStats || {});
    const stats = {};
    shots.forEach((shot) => {
        const zone = normalizeZoneId(shot.zone || shot.shooting_zone || 'unknown');
        if (!stats[zone]) {
            stats[zone] = {
                attempts: 0,
                makes: 0,
                score_total: 0,
                score_count: 0,
                benchmark: normalizedBase[zone]?.benchmark ?? null,
                benchmark_std: normalizedBase[zone]?.benchmark_std ?? null
            };
        }
        stats[zone].attempts += 1;
        if (shot.outcome === 'make') {
            stats[zone].makes += 1;
        }
        if (shot.shotsync_score !== null && shot.shotsync_score !== undefined) {
            stats[zone].score_total += shot.shotsync_score;
            stats[zone].score_count += 1;
        }
    });

    const finalStats = {};
    Object.entries(stats).forEach(([zone, entry]) => {
        const attempts = entry.attempts;
        const makes = entry.makes;
        finalStats[zone] = {
            attempts,
            makes,
            percentage: attempts > 0 ? (makes / attempts) * 100.0 : 0.0,
            shotsync_score: entry.score_count > 0 ? entry.score_total / entry.score_count : null,
            benchmark: entry.benchmark,
            benchmark_std: entry.benchmark_std
        };
    });
    return finalStats;
}

function removeShotAt(index) {
    if (!currentResults || !Array.isArray(currentResults.shots)) return;
    if (index < 0 || index >= currentResults.shots.length) return;
    currentResults.shots.splice(index, 1);
    currentResults.shots.forEach((shot, idx) => {
        shot.shot_number = idx + 1;
    });
    currentResults.total_attempts = currentResults.shots.length;
    currentResults.total_makes = currentResults.shots.filter((shot) => shot.outcome === 'make').length;
    currentResults.zone_stats = buildZoneStatsFromShots(currentResults.shots, currentResults.zone_stats || {});
    renderResults(currentResults);
}

function showClipViewer({ clipUrl, videoUrl, clipStart, clipEnd }, metaText) {
    clipViewerEl.classList.remove('hidden');
    clipMetaEl.textContent = `${metaText} — loading clip...`;
    clipVideoEl.onerror = null;
    clipVideoEl.onloadeddata = null;
    clipVideoEl.onloadedmetadata = null;
    clipVideoEl.ontimeupdate = null;

    let fallbackStage = 0;
    clipVideoEl.onerror = () => {
        if (videoUrl && fallbackStage === 0) {
            fallbackStage = 1;
            clipMetaEl.textContent = `${metaText} — loading full video segment...`;
            clipVideoEl.src = videoUrl;
            clipVideoEl.load();
            return;
        }
        clipMetaEl.textContent = `${metaText} — clip still generating. Try again in a moment.`;
    };
    clipVideoEl.onloadeddata = () => {
        clipMetaEl.textContent = metaText;
    };
    clipVideoEl.onloadedmetadata = () => {
        if (videoUrl && Number.isFinite(clipStart)) {
            clipVideoEl.currentTime = clipStart;
            clipVideoEl.play().catch(() => {});
        }
    };
    clipVideoEl.ontimeupdate = () => {
        if (videoUrl && Number.isFinite(clipEnd) && clipVideoEl.currentTime >= clipEnd) {
            clipVideoEl.pause();
        }
    };

    if (videoUrl && Number.isFinite(clipStart)) {
        clipVideoEl.src = videoUrl;
    } else if (clipUrl) {
        clipVideoEl.src = clipUrl;
    }
    clipVideoEl.load();
}

function hideClipViewer() {
    clipViewerEl.classList.add('hidden');
    clipVideoEl.removeAttribute('src');
    clipMetaEl.textContent = 'Select a shot to view the clip.';
}

function syncCanvasToImage(imageEl, canvasEl) {
    if (!imageEl || !canvasEl) return;
    const rect = imageEl.getBoundingClientRect();
    canvasEl.width = Math.max(1, Math.round(rect.width));
    canvasEl.height = Math.max(1, Math.round(rect.height));
}

function handleCourtCalibration(data) {
    hideCourtCalibration();
    if (courtScrollHintEl) {
        courtScrollHintEl.classList.add('hidden');
    }
}

function hideCourtCalibration() {
    if (!courtCalibrationEl || !courtCalibImage || !courtCalibCanvas) return;
    courtCalibrationEl.classList.add('hidden');
    courtCalibImage.removeAttribute('src');
    const ctx = courtCalibCanvas.getContext('2d');
    ctx.clearRect(0, 0, courtCalibCanvas.width || 1, courtCalibCanvas.height || 1);
}

function drawCourtMarkers() {
    const ctx = courtCalibCanvas.getContext('2d');
    ctx.clearRect(0, 0, courtCalibCanvas.width, courtCalibCanvas.height);
    if (!courtSelection.length) return;

    ctx.strokeStyle = '#ff6b3d';
    ctx.fillStyle = '#ff6b3d';
    ctx.lineWidth = 3;

    // Draw connecting lines in click order to help the user see the polygon.
    ctx.beginPath();
    courtSelection.forEach((pt, idx) => {
        const x = pt.x_norm * courtCalibCanvas.width;
        const y = pt.y_norm * courtCalibCanvas.height;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw numbered markers for each point.
    courtSelection.forEach((pt, idx) => {
        const x = pt.x_norm * courtCalibCanvas.width;
        const y = pt.y_norm * courtCalibCanvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#121212';
        ctx.font = '12px Space Grotesk, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(idx + 1), x, y + 0.5);
        ctx.fillStyle = '#ff6b3d';
    });
}

function handleNetZoneCalibration(data) {
    hideNetZoneCalibration();
    if (rimScrollHintEl) {
        rimScrollHintEl.classList.add('hidden');
    }
}

function hideNetZoneCalibration() {
    if (!netZoneCalibrationEl || !netZoneCalibImage || !netZoneCalibCanvas) return;
    netZoneCalibrationEl.classList.add('hidden');
    netZoneCalibImage.removeAttribute('src');
    const ctx = netZoneCalibCanvas.getContext('2d');
    ctx.clearRect(0, 0, netZoneCalibCanvas.width || 1, netZoneCalibCanvas.height || 1);
}

function setNetZoneHint(message) {
    if (netZoneHintEl) {
        netZoneHintEl.textContent = message;
    }
}

function isNetZoneReady() {
    return Boolean(netZoneSelection && netZoneSelection.x1_norm !== undefined);
}

function drawNetZoneMarker() {
    const ctx = netZoneCalibCanvas.getContext('2d');
    ctx.clearRect(0, 0, netZoneCalibCanvas.width, netZoneCalibCanvas.height);
    if (!netZoneSelection || !netZoneSelection.topLeft) return;

    const tlx = netZoneSelection.topLeft.x_norm * netZoneCalibCanvas.width;
    const tly = netZoneSelection.topLeft.y_norm * netZoneCalibCanvas.height;
    ctx.fillStyle = '#ff6b3d';
    ctx.strokeStyle = '#ff6b3d';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(tlx, tly, 6, 0, Math.PI * 2);
    ctx.fill();

    if (!netZoneSelection.bottomRight) return;
    const brx = netZoneSelection.bottomRight.x_norm * netZoneCalibCanvas.width;
    const bry = netZoneSelection.bottomRight.y_norm * netZoneCalibCanvas.height;
    const x1 = Math.min(tlx, brx);
    const y1 = Math.min(tly, bry);
    const x2 = Math.max(tlx, brx);
    const y2 = Math.max(tly, bry);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = '#ff6b3d';
    ctx.font = '14px Space Grotesk, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('RIM', x1 + 6, y1 - 6);
    ctx.textBaseline = 'top';
    ctx.textAlign = 'right';
    ctx.fillText('NET', x2 - 6, y2 + 6);
}

if (netZoneCalibCanvas) netZoneCalibCanvas.addEventListener('click', (event) => {
    if (!netZoneCalibrationMeta) return;
    const rect = netZoneCalibCanvas.getBoundingClientRect();
    const x = Math.min(Math.max(0, event.clientX - rect.left), rect.width);
    const y = Math.min(Math.max(0, event.clientY - rect.top), rect.height);
    const point = {
        x_norm: rect.width > 0 ? x / rect.width : 0.5,
        y_norm: rect.height > 0 ? y / rect.height : 0.5
    };

    if (!netZoneSelection || netZoneSelection.step >= 3) {
        netZoneSelection = { step: 1, topLeft: null, bottomRight: null };
    }

    if (netZoneSelection.step === 1) {
        netZoneSelection.topLeft = point;
        netZoneSelection.bottomRight = null;
        netZoneSelection.step = 2;
        setNetZoneHint('Now click the BOTTOM-RIGHT corner at net bottom (right edge).');
    } else if (netZoneSelection.step === 2) {
        netZoneSelection.bottomRight = point;
        const x1 = Math.min(netZoneSelection.topLeft.x_norm, point.x_norm);
        const y1 = Math.min(netZoneSelection.topLeft.y_norm, point.y_norm);
        const x2 = Math.max(netZoneSelection.topLeft.x_norm, point.x_norm);
        const y2 = Math.max(netZoneSelection.topLeft.y_norm, point.y_norm);
        netZoneSelection.x1_norm = x1;
        netZoneSelection.y1_norm = y1;
        netZoneSelection.x2_norm = x2;
        netZoneSelection.y2_norm = y2;
        netZoneSelection.step = 3;
        setNetZoneHint('Net zone marked. Click "Re-run with Net Zone" to continue.');
        if (netZoneConfirmBtn) netZoneConfirmBtn.disabled = false;
    }

    drawNetZoneMarker();
});

if (netZoneConfirmBtn) netZoneConfirmBtn.addEventListener('click', () => {
    if (!isNetZoneReady()) return;
    runAnalysis(netZoneSelection);
});

if (netZoneDismissBtn) netZoneDismissBtn.addEventListener('click', () => {
    netZoneSelection = null;
    if (netZoneConfirmBtn) netZoneConfirmBtn.disabled = true;
    hideNetZoneCalibration();
});

if (courtCalibCanvas) courtCalibCanvas.addEventListener('click', (event) => {
    if (!courtCalibrationMeta) return;
    const rect = courtCalibCanvas.getBoundingClientRect();
    const x = Math.min(Math.max(0, event.clientX - rect.left), rect.width);
    const y = Math.min(Math.max(0, event.clientY - rect.top), rect.height);
    const point = {
        x_norm: rect.width > 0 ? x / rect.width : 0.5,
        y_norm: rect.height > 0 ? y / rect.height : 0.5
    };
    if (courtSelection.length >= 4) {
        courtSelection = [point];
    } else {
        courtSelection.push(point);
    }
    drawCourtMarkers();
    if (courtConfirmBtn) courtConfirmBtn.disabled = courtSelection.length !== 4;
});

if (courtConfirmBtn) courtConfirmBtn.addEventListener('click', () => {
    if (courtSelection.length !== 4) return;
    runAnalysis();
});

if (courtResetBtn) courtResetBtn.addEventListener('click', () => {
    courtSelection = [];
    if (courtConfirmBtn) courtConfirmBtn.disabled = true;
    drawCourtMarkers();
});

if (courtDismissBtn) courtDismissBtn.addEventListener('click', () => {
    courtSelection = [];
    if (courtConfirmBtn) courtConfirmBtn.disabled = true;
    hideCourtCalibration();
});

window.addEventListener('resize', () => {
    if (courtCalibrationEl && !courtCalibrationEl.classList.contains('hidden')) {
        syncCanvasToImage(courtCalibImage, courtCalibCanvas);
        drawCourtMarkers();
    }
    if (netZoneCalibrationEl && !netZoneCalibrationEl.classList.contains('hidden')) {
        syncCanvasToImage(netZoneCalibImage, netZoneCalibCanvas);
        drawNetZoneMarker();
    }
});

function renderHeatmap(zoneStats) {
    const zones = document.querySelectorAll('.heatmap [data-zone]');
    zones.forEach((shape) => {
        const zone = shape.dataset.zone;
        const stats = zoneStats[zone];
        let fill = '#1c1c1c';
        let titleText = `${ZONE_LABELS[zone] || zone}: no data`;

        if (stats && stats.attempts > 0) {
            const pct = stats.percentage || 0;
            const hue = Math.max(0, Math.min(120, (pct / 100) * 120));
            fill = `hsl(${hue}, 65%, 45%)`;
            titleText = `${ZONE_LABELS[zone] || zone}: ${pct.toFixed(1)}% (${stats.makes}/${stats.attempts})`;
        }

        shape.style.fill = fill;
        let title = shape.querySelector('title');
        if (!title) {
            title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            shape.appendChild(title);
        }
        title.textContent = titleText;
    });
}
