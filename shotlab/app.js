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
const clipViewerEl = document.getElementById('clipViewer');
const clipVideoEl = document.getElementById('clipVideo');
const clipMetaEl = document.getElementById('clipMeta');
const courtCalibrationEl = document.getElementById('courtCalibration');
const courtCalibImage = document.getElementById('courtCalibImage');
const courtCalibCanvas = document.getElementById('courtCalibCanvas');
const courtConfirmBtn = document.getElementById('courtConfirm');
const courtResetBtn = document.getElementById('courtReset');
const courtDismissBtn = document.getElementById('courtDismiss');
const rimCalibrationEl = document.getElementById('rimCalibration');
const rimCalibImage = document.getElementById('rimCalibImage');
const rimCalibCanvas = document.getElementById('rimCalibCanvas');
const rimConfirmBtn = document.getElementById('rimConfirm');
const rimDismissBtn = document.getElementById('rimDismiss');
let statusPoll = null;
let selectedFile = null;
let courtSelection = [];
let courtCalibrationMeta = null;
let rimSelection = null;
let rimCalibrationMeta = null;

const ZONE_LABELS = {
    restricted_area: 'Restricted Area',
    paint: 'Paint',
    mid_range: 'Mid Range',
    left_corner_3: 'Left Corner 3',
    right_corner_3: 'Right Corner 3',
    left_wing_3: 'Left Wing 3',
    right_wing_3: 'Right Wing 3',
    top_of_key_3: 'Top of Key 3',
    left_baseline_2: 'Left Baseline 2',
    right_baseline_2: 'Right Baseline 2',
    left_wing_2: 'Left Wing 2',
    right_wing_2: 'Right Wing 2',
    unknown: 'Unknown'
};

videoInput.addEventListener('change', () => {
    if (videoInput.files && videoInput.files[0]) {
        selectedFile = videoInput.files[0];
        fileName.textContent = selectedFile.name;
        analyzeBtn.disabled = false;
        courtSelection = [];
        courtCalibrationMeta = null;
        rimSelection = null;
        rimCalibrationMeta = null;
        hideClipViewer();
        hideCourtCalibration();
        hideRimCalibration();
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

function setStatus(message) {
    statusBox.textContent = message;
    statusBox.classList.remove('hidden');
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

async function runAnalysis(rimOverride = null) {
    if (!selectedFile) return;
    setStatus('Uploading and analyzing... This can take a minute.');
    analyzeBtn.disabled = true;
    courtConfirmBtn.disabled = true;
    rimConfirmBtn.disabled = true;
    hideClipViewer();
    startStatusPoll();

    const formData = new FormData();
    formData.append('video', selectedFile);
    const courtOverride = courtSelection.length === 4 ? courtSelection : null;
    if (courtOverride) {
        formData.append('court_points', JSON.stringify(courtOverride));
    }
    if (rimOverride && rimOverride.x_norm !== undefined && rimOverride.y_norm !== undefined) {
        formData.append('rim_x_norm', String(rimOverride.x_norm));
        formData.append('rim_y_norm', String(rimOverride.y_norm));
        formData.append('rim_r_norm', String(rimOverride.r_norm ?? 0.03));
    } else if (rimSelection && rimSelection.x_norm !== undefined) {
        formData.append('rim_x_norm', String(rimSelection.x_norm));
        formData.append('rim_y_norm', String(rimSelection.y_norm));
        formData.append('rim_r_norm', String(rimSelection.r_norm ?? 0.03));
    }

    try {
        const response = await fetch('/api/process_shotlab_session', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Analysis failed');
        }
        renderResults(data);
        setStatus('Analysis complete.');
    } catch (error) {
        setStatus(`Error: ${error.message}`);
    } finally {
        stopStatusPoll();
        analyzeBtn.disabled = false;
        courtConfirmBtn.disabled = courtSelection.length !== 4;
        rimConfirmBtn.disabled = rimSelection === null;
    }
}

function renderResults(data) {
    results.classList.remove('hidden');

    const totalAttempts = data.total_attempts || 0;
    const totalMakes = data.total_makes || 0;
    const pct = totalAttempts > 0 ? (totalMakes / totalAttempts) * 100 : 0;

    totalShotsEl.textContent = totalAttempts;
    totalMakesEl.textContent = totalMakes;
    totalPctEl.textContent = `${pct.toFixed(1)}%`;

    renderHeatmap(data.zone_stats || {});
    handleCourtCalibration(data);
    handleRimCalibration(data);

    zoneTable.innerHTML = '';
    const zones = Object.entries(data.zone_stats || {}).sort((a, b) => b[1].attempts - a[1].attempts);
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
    const shots = data.shots || [];
    if (shots.length === 0) {
        shotTable.innerHTML = '<tr><td colspan="4">No shots detected.</td></tr>';
        hideClipViewer();
    } else {
        shots.forEach((shot) => {
            const zoneLabel = ZONE_LABELS[shot.zone] || shot.zone || 'Unknown';
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
                <tr class="shot-row" ${clipUrlAttr} ${clipStartAttr} ${clipEndAttr} ${videoUrlAttr} data-shot-meta="${shotMeta}">
                    <td>${shot.shot_number}</td>
                    <td>${zoneLabel}</td>
                    <td><span class="badge ${badgeClass}">${result}</span></td>
                    <td>${score}</td>
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
    }

    if (data.warnings && data.warnings.length) {
        warningsEl.classList.remove('hidden');
        warningsEl.textContent = `Warnings: ${data.warnings.join(', ')}`;
    } else {
        warningsEl.classList.add('hidden');
        warningsEl.textContent = '';
    }

    if (data.debug) {
        const parts = [];
        const shots = data.debug.shots || {};
        const pose = data.debug.pose || {};
        const ball = data.debug.ball || {};
        const court = data.debug.court || {};
        const rim = data.debug.rim || {};
        const shotDet = (pose.shot_detection || {});

        if (shots.shot_attempts !== undefined) {
            parts.push(`shots=${shots.shot_attempts}`);
        }
        if (shots.outcome_reasons) {
            const reasons = Object.entries(shots.outcome_reasons)
                .map(([key, value]) => `${key}:${value}`)
                .join(',');
            if (reasons) {
                parts.push(`outcomes=${reasons}`);
            }
        }
        if (pose.state_counts) {
            const pre = pose.state_counts.pre_shot || 0;
            const follow = pose.state_counts.follow_through || 0;
            parts.push(`pose(pre=${pre}, follow=${follow})`);
        }
        if (shotDet.follow_entries !== undefined) {
            const vel = shotDet.velocity_candidates || 0;
            const repl = shotDet.replaced_close_candidate || 0;
            const seg = shotDet.follow_segments || 0;
            const allowFollow = shotDet.allow_follow_only ? 1 : 0;
            parts.push(`shotDet(follow=${shotDet.follow_entries}, seg=${seg}, vel=${vel}, followOnly=${shotDet.follow_only_used || 0}, allowFollow=${allowFollow}, replaced=${repl})`);
        }
        if (shotDet.ball_filter) {
            const kept = shotDet.ball_filter.kept ?? 0;
            const dropped = shotDet.ball_filter.dropped ?? 0;
            parts.push(`ballFilter(kept=${kept}, dropped=${dropped})`);
        }
        if (shotDet.dedupe) {
            const kept = shotDet.dedupe.kept ?? 0;
            const dropped = shotDet.dedupe.dropped ?? 0;
            parts.push(`dedupe(kept=${kept}, dropped=${dropped})`);
        }
        if (ball.tracks !== undefined) {
            parts.push(`ball(tracks=${ball.tracks}, timeouts=${ball.timeouts || 0})`);
        }
        if (court.court_transform_available === false) {
            parts.push('court_transform=missing');
        }
        if (rim.available === false) {
            parts.push('rim=missing');
        }
        if (data.calibration && data.calibration.court_required) {
            parts.push('court=needs_clicks');
        }
        if (data.calibration && data.calibration.rim_required) {
            parts.push('rim=needs_click');
        }

        if (parts.length) {
            warningsEl.classList.remove('hidden');
            const prefix = warningsEl.textContent ? `${warningsEl.textContent} | ` : '';
            warningsEl.textContent = `${prefix}Debug: ${parts.join(' | ')}`;
        }
    }
}

function showClipViewer({ clipUrl, videoUrl, clipStart, clipEnd }, metaText) {
    clipViewerEl.classList.remove('hidden');
    clipMetaEl.textContent = `${metaText} — loading clip...`;
    clipVideoEl.onerror = null;
    clipVideoEl.onloadeddata = null;
    clipVideoEl.onloadedmetadata = null;
    clipVideoEl.ontimeupdate = null;

    let triedFallback = false;
    clipVideoEl.onerror = () => {
        if (!triedFallback && clipUrl && clipVideoEl.src !== clipUrl) {
            triedFallback = true;
            clipMetaEl.textContent = `${metaText} — loading clip...`;
            clipVideoEl.src = clipUrl;
            clipVideoEl.load();
            return;
        }
        clipMetaEl.textContent = `${metaText} — clip is still generating. Try again in a moment.`;
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
    const rect = imageEl.getBoundingClientRect();
    canvasEl.width = Math.max(1, Math.round(rect.width));
    canvasEl.height = Math.max(1, Math.round(rect.height));
}

function handleCourtCalibration(data) {
    const calibration = data.calibration || {};
    const courtCal = calibration.court;
    const needsCourt = Boolean(calibration.court_required && courtCal && courtCal.court_frame);
    if (!needsCourt) {
        hideCourtCalibration();
        if (courtScrollHintEl) {
            courtScrollHintEl.classList.add('hidden');
        }
        return;
    }
    if (courtScrollHintEl) {
        courtScrollHintEl.classList.remove('hidden');
    }
    courtCalibrationMeta = courtCal;
    courtSelection = [];
    courtConfirmBtn.disabled = true;
    courtCalibImage.src = `data:image/jpeg;base64,${courtCal.court_frame}`;
    courtCalibImage.onload = () => {
        syncCanvasToImage(courtCalibImage, courtCalibCanvas);
        drawCourtMarkers();
    };
    courtCalibrationEl.classList.remove('hidden');
}

function hideCourtCalibration() {
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

function handleRimCalibration(data) {
    const calibration = data.calibration || {};
    const rimCal = calibration.rim;
    const needsRim = Boolean(calibration.rim_required && rimCal && rimCal.rim_frame);
    if (!needsRim) {
        hideRimCalibration();
        return;
    }
    rimCalibrationMeta = rimCal;
    rimSelection = null;
    rimConfirmBtn.disabled = true;
    rimCalibImage.src = `data:image/jpeg;base64,${rimCal.rim_frame}`;
    rimCalibImage.onload = () => {
        syncCanvasToImage(rimCalibImage, rimCalibCanvas);
        drawRimMarker();
    };
    rimCalibrationEl.classList.remove('hidden');
}

function hideRimCalibration() {
    rimCalibrationEl.classList.add('hidden');
    rimCalibImage.removeAttribute('src');
    const ctx = rimCalibCanvas.getContext('2d');
    ctx.clearRect(0, 0, rimCalibCanvas.width || 1, rimCalibCanvas.height || 1);
}

function drawRimMarker() {
    const ctx = rimCalibCanvas.getContext('2d');
    ctx.clearRect(0, 0, rimCalibCanvas.width, rimCalibCanvas.height);
    if (!rimSelection) return;
    const x = rimSelection.x_norm * rimCalibCanvas.width;
    const y = rimSelection.y_norm * rimCalibCanvas.height;
    const r = (rimSelection.r_norm || 0.03) * Math.min(rimCalibCanvas.width, rimCalibCanvas.height);
    ctx.strokeStyle = '#ff6b3d';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(8, r), 0, Math.PI * 2);
    ctx.stroke();
}

rimCalibCanvas.addEventListener('click', (event) => {
    if (!rimCalibrationMeta) return;
    const rect = rimCalibCanvas.getBoundingClientRect();
    const x = Math.min(Math.max(0, event.clientX - rect.left), rect.width);
    const y = Math.min(Math.max(0, event.clientY - rect.top), rect.height);
    rimSelection = {
        x_norm: rect.width > 0 ? x / rect.width : 0.5,
        y_norm: rect.height > 0 ? y / rect.height : 0.5,
        r_norm: 0.035
    };
    drawRimMarker();
    rimConfirmBtn.disabled = false;
});

rimConfirmBtn.addEventListener('click', () => {
    if (!rimSelection) return;
    runAnalysis(rimSelection);
});

rimDismissBtn.addEventListener('click', () => {
    rimSelection = null;
    rimConfirmBtn.disabled = true;
    hideRimCalibration();
});

courtCalibCanvas.addEventListener('click', (event) => {
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
    courtConfirmBtn.disabled = courtSelection.length !== 4;
});

courtConfirmBtn.addEventListener('click', () => {
    if (courtSelection.length !== 4) return;
    runAnalysis();
});

courtResetBtn.addEventListener('click', () => {
    courtSelection = [];
    courtConfirmBtn.disabled = true;
    drawCourtMarkers();
});

courtDismissBtn.addEventListener('click', () => {
    courtSelection = [];
    courtConfirmBtn.disabled = true;
    hideCourtCalibration();
});

window.addEventListener('resize', () => {
    if (!courtCalibrationEl.classList.contains('hidden')) {
        syncCanvasToImage(courtCalibImage, courtCalibCanvas);
        drawCourtMarkers();
    }
    if (!rimCalibrationEl.classList.contains('hidden')) {
        syncCanvasToImage(rimCalibImage, rimCalibCanvas);
        drawRimMarker();
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
