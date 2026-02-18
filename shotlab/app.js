(() => {
  const CHECK_CONFIG = [
    { key: "elbow_alignment", label: "Elbow Alignment", max: 20 },
    { key: "release_height", label: "Release Height", max: 20 },
    { key: "follow_through", label: "Follow-Through", max: 20 },
    { key: "base_and_balance", label: "Base and Balance", max: 15 },
    { key: "shoulder_alignment", label: "Shoulder Alignment", max: 15 },
    { key: "guide_hand", label: "Guide Hand", max: 10 },
  ];
  const RADAR_AXIS_ORDER = [
    { key: "elbow_alignment", label: "Elbow Alignment", max: 20 },
    { key: "release_height", label: "Release Height", max: 20 },
    { key: "follow_through", label: "Follow-Through", max: 20 },
    { key: "guide_hand", label: "Guide Hand", max: 10 },
    { key: "base_and_balance", label: "Base & Balance", max: 15 },
    { key: "shoulder_alignment", label: "Shoulder Alignment", max: 15 },
  ];
  const CHECK_PLAYBOOK = {
    elbow_alignment: {
      why: "A flared elbow pushes the ball sideways, reducing accuracy on longer shots",
      try: "Practice with your elbow touching a wall at setup. If it pulls away, you're flaring.",
    },
    release_height: {
      why: "A low release creates a flat arc - less margin for error at the rim and easier to block",
      try: "Hold the ball at your set point - it should be above your eyebrow. Shoot 10 free throws focusing only on getting the ball above your forehead before releasing.",
    },
    follow_through: {
      why: "Without follow-through, you lose backspin and distance control - the #1 factor in consistency",
      try: "After every shot, freeze with your arm extended and count to 2. Your wrist should be relaxed, fingers pointing at the rim.",
    },
    base_and_balance: {
      why: "Without knee bend, you're shooting with arms only - less power, more strain on longer shots",
      try: "Bend your knees until you feel your quads engage. Shoot 10 shots focusing only on pushing UP through your legs.",
    },
    shoulder_alignment: {
      why: "Tilted shoulders aim the ball offline - you'll miss left or right consistently",
      try: "Face the basket with your feet and shoulders before every shot. Practice squaring up on catch-and-shoot.",
    },
    guide_hand: {
      why: "A pushing guide hand adds sidespin, causing the ball to curve away from the basket",
      try: "Put a small piece of tape on your guide hand palm. If it touches the ball at release, your guide hand is pushing.",
    },
  };
  const MINI_RADAR_LABELS = {
    elbow_alignment: "Elbow",
    release_height: "Release",
    follow_through: "Follow",
    guide_hand: "Guide",
    base_and_balance: "Base",
    shoulder_alignment: "Shoulder",
  };
  const CHECK_GRADE_MEASURED = {
    elbow_alignment: {
      A: "Your elbow stayed tucked under the ball - textbook form.",
      B: "Your elbow drifted slightly outward at release. Close to good but room to tighten up.",
      C: "Your elbow is flaring out noticeably, pushing the ball off-line.",
      DF: "Your elbow is significantly flared. This is the #1 thing to fix - it's pulling every shot sideways.",
    },
    release_height: {
      A: "You released well above your forehead with full arm extension - great arc.",
      B: "Your release is at forehead height. A bit higher would give you more arc and margin.",
      C: "Your release is around shoulder height - too low for a consistent arc.",
      DF: "You're releasing from chest/shoulder level. This creates a flat shot that's easy to block and has no margin at the rim.",
    },
    follow_through: {
      A: "Your arm stayed extended toward the basket after release - full follow-through with good wrist snap.",
      B: "You held your follow-through briefly but pulled back a bit early. Try holding 1 second longer.",
      C: "Your arm started to drop right after release. You're losing backspin and distance control.",
      DF: "You pulled your hand back immediately after the shot. Without follow-through, every shot is a guess on distance.",
    },
    base_and_balance: {
      A: "Good athletic stance - knees bent, balanced over your feet, power coming from your legs.",
      B: "Decent base but could bend your knees a bit more for extra power on longer shots.",
      C: "Legs are too straight - you're shooting mostly with arms. Or you're leaning to one side.",
      DF: "Almost no knee bend and/or significant lean. You're using all arms with no leg power. This kills range and consistency.",
    },
    shoulder_alignment: {
      A: "Shoulders are square to the basket - great alignment.",
      B: "Slight shoulder tilt. Not a major issue but worth keeping level.",
      C: "Noticeable shoulder tilt - this aims the ball offline on longer shots.",
      DF: "Significant shoulder drop on one side. You're consistently aiming offline before you even release.",
    },
    guide_hand: {
      A: "Guide hand stayed still at release - clean support without pushing.",
      B: "Slight guide hand movement. Mostly clean but watch for it on pressure shots.",
      C: "Guide hand is pushing the ball at release, adding unwanted sidespin.",
      DF: "Guide hand is significantly interfering with the shot - it's adding spin and pulling the ball off target every time.",
    },
  };

  const ZONE_ORDER = ["left", "center", "right", "unknown"];

  const STATUS_COLORS = {
    good: "#4CAF50",
    needs_work: "#FFC107",
    poor: "#F44336",
    unavailable: "#666666",
  };

  const SKELETON_CONNECTIONS = [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"],
  ];

  const SEGMENT_CHECK_MAP = {
    "right_shoulder-right_elbow": "elbow_alignment",
    "right_elbow-right_wrist": "elbow_alignment",
    "left_shoulder-left_elbow": "guide_hand",
    "left_elbow-left_wrist": "guide_hand",
    "left_shoulder-right_shoulder": "shoulder_alignment",
    "left_hip-left_knee": "base_and_balance",
    "left_knee-left_ankle": "base_and_balance",
    "right_hip-right_knee": "base_and_balance",
    "right_knee-right_ankle": "base_and_balance",
    "left_hip-right_hip": "base_and_balance",
  };

  const state = {
    file: null,
    fileObjectUrl: null,
    videoName: null,
    payload: null,
    shots: [],
    sessionSummary: {},
    corrections: [],
    saveStatusTimer: null,
    sortBy: "shot_order",
    statusPollTimer: null,
    progressValue: 0,
    progressHighWater: 0,
    progressStageText: "Analyzing your shots...",
    skeletonPlayers: {},
    comparisonController: null,
    expandedTrafficLight: null,
  };

  const el = {
    uploadScreen: document.getElementById("uploadScreen"),
    loadingScreen: document.getElementById("loadingScreen"),
    resultsScreen: document.getElementById("resultsScreen"),
    uploadForm: document.getElementById("uploadForm"),
    videoFile: document.getElementById("videoFile"),
    dropZone: document.getElementById("dropZone"),
    fileName: document.getElementById("fileName"),
    uploadError: document.getElementById("uploadError"),
    analyzeBtn: document.getElementById("analyzeBtn"),
    progressFill: document.getElementById("progressFill"),
    progressPercent: document.getElementById("progressPercent"),
    progressText: document.getElementById("progressText"),
    newSessionBtn: document.getElementById("newSessionBtn"),
    shotsyncGauge: document.getElementById("shotsyncGauge"),
    gaugeValue: document.getElementById("gaugeValue"),
    gaugeScore: document.getElementById("gaugeScore"),
    gaugeLabel: document.getElementById("gaugeLabel"),
    quickStats: document.getElementById("quickStats"),
    topImprovementTitle: document.getElementById("topImprovementTitle"),
    topImprovementText: document.getElementById("topImprovementText"),
    strengthTitle: document.getElementById("strengthTitle"),
    strengthText: document.getElementById("strengthText"),
    makeVsMissText: document.getElementById("makeVsMissText"),
    checkBreakdown: document.getElementById("checkBreakdown"),
    shotDots: document.getElementById("shotDots"),
    zoneStats: document.getElementById("zoneStats"),
    zoneTips: document.getElementById("zoneTips"),
    comparisonSection: document.getElementById("comparisonSection"),
    comparisonContent: document.getElementById("comparisonContent"),
    shotSort: document.getElementById("shotSort"),
    shotList: document.getElementById("shotList"),
    overallConsistencyText: document.getElementById("overallConsistencyText"),
    zoneSummaryText: document.getElementById("zoneSummaryText"),
    correctionsCard: document.getElementById("correctionsCard"),
    correctionsCount: document.getElementById("correctionsCount"),
    saveCorrectionsBtn: document.getElementById("saveCorrectionsBtn"),
    saveCorrectionsStatus: document.getElementById("saveCorrectionsStatus"),
  };

  function isNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function asNumber(value) {
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function prettyCheckName(key) {
    const found = CHECK_CONFIG.find((item) => item.key === key);
    return found ? found.label : String(key || "Unknown Check");
  }

  function statusClass(status) {
    const normalized = String(status || "unavailable").toLowerCase();
    if (["good", "needs_work", "poor", "unavailable"].includes(normalized)) {
      return normalized;
    }
    return "unavailable";
  }

  function statusLabel(status) {
    const normalized = statusClass(status);
    if (normalized === "needs_work") {
      return "Needs Work";
    }
    if (normalized === "unavailable") {
      return "Unavailable";
    }
    return normalized.charAt(0).toUpperCase() + normalized.slice(1);
  }

  function scoreTier(score) {
    if (!isNumber(score)) return { letter: "-", label: "No Score", color: "#7c91a6", numeric: null };
    const numeric = clamp(score, 0, 100);
    if (numeric >= 80) return { letter: "A", label: "Excellent", color: "#4CAF50", numeric };
    if (numeric >= 60) return { letter: "B", label: "Good", color: "#2196F3", numeric };
    if (numeric >= 40) return { letter: "C", label: "Average", color: "#FFC107", numeric };
    if (numeric >= 20) return { letter: "D", label: "Needs Work", color: "#FF9800", numeric };
    return { letter: "F", label: "Poor", color: "#F44336", numeric };
  }

  function gradeFromRatio(ratio) {
    if (!isNumber(ratio)) return { letter: "C", color: "#FFC107", label: "Average" };
    if (ratio >= 0.75) return { letter: "A", color: "#4CAF50", label: "Excellent" };
    if (ratio >= 0.5) return { letter: "B", color: "#2196F3", label: "Good" };
    return { letter: "C", color: "#FFC107", label: "Needs Work" };
  }

  function checkGradeFromPoints(points, maxPoints) {
    if (!isNumber(points) || !isNumber(maxPoints) || maxPoints <= 0) {
      return gradeFromRatio(0.5);
    }
    return gradeFromRatio(points / maxPoints);
  }

  function formatFormGrade(score) {
    const grade = scoreTier(asNumber(score));
    if (!isNumber(grade.numeric)) return "Form: - (--)";
    return `Form: ${grade.letter} (${Math.round(grade.numeric)})`;
  }

  function makeGradeSpan(score) {
    const grade = scoreTier(asNumber(score));
    const letter = grade.letter || "-";
    const number = isNumber(grade.numeric) ? Math.round(grade.numeric) : "--";
    return `<span class="form-grade-wrap">Form: <span class="form-grade-letter" style="color:${grade.color};">${letter}</span> <span class="form-grade-score">(${number})</span></span>`;
  }

  function formatPct(value) {
    const num = asNumber(value);
    return num === null ? "0.0%" : `${num.toFixed(1)}%`;
  }

  function formatScore(value) {
    const num = asNumber(value);
    return num === null ? "--" : `${Math.round(num)}`;
  }

  function formatScorePrecise(value) {
    const num = asNumber(value);
    return num === null ? "--" : num.toFixed(1);
  }

  function parseShots(payload) {
    const sourceShots = payload?.shots_analysis || payload?.shots || [];
    if (!Array.isArray(sourceShots)) {
      return [];
    }

    return sourceShots.map((shot, index) => {
      const shotNum = asNumber(shot?.shot_num ?? shot?.shot_number ?? index + 1) ?? (index + 1);
      const clipStart = asNumber(shot?.clip_start_time ?? shot?.clip_start);
      const clipEnd = asNumber(shot?.clip_end_time ?? shot?.clip_end);
      const rawOutcome = String(shot?.outcome || "unknown").toLowerCase();
      const normalizedOutcome = rawOutcome === "make" || rawOutcome === "miss" ? rawOutcome : "unknown";
      return {
        ...shot,
        shot_num: shotNum,
        outcome: normalizedOutcome,
        original_outcome: normalizedOutcome,
        manual_edited: false,
        zone: String(shot?.zone || shot?.shooting_zone || "unknown").toLowerCase(),
        shotsync_score: asNumber(shot?.shotsync_score),
        shooter_x: asNumber(shot?.shooter_x),
        shooter_y: asNumber(shot?.shooter_y),
        clip_start_time: clipStart,
        clip_end_time: clipEnd,
        skeleton_frames: Array.isArray(shot?.skeleton_frames) ? shot.skeleton_frames : [],
      };
    });
  }

  function summarizeCheckFromShots(checkKey, shots) {
    const summary = {
      good: 0,
      needs_work: 0,
      poor: 0,
      unavailable: 0,
      avg_points: null,
      makes_avg: null,
      misses_avg: null,
    };

    const points = [];
    const makePoints = [];
    const missPoints = [];

    for (const shot of shots) {
      const check = shot?.coaching?.[checkKey];
      if (!check || typeof check !== "object") continue;

      const status = statusClass(check.status);
      if (status === "good" || status === "needs_work" || status === "poor" || status === "unavailable") {
        summary[status] += 1;
      }

      const pointValue = asNumber(check.points);
      if (pointValue !== null) {
        points.push(pointValue);
        if (shot.outcome === "make") makePoints.push(pointValue);
        if (shot.outcome === "miss") missPoints.push(pointValue);
      }
    }

    if (points.length) {
      summary.avg_points = Number((points.reduce((a, b) => a + b, 0) / points.length).toFixed(1));
    }
    if (makePoints.length) {
      summary.makes_avg = Number((makePoints.reduce((a, b) => a + b, 0) / makePoints.length).toFixed(1));
    }
    if (missPoints.length) {
      summary.misses_avg = Number((missPoints.reduce((a, b) => a + b, 0) / missPoints.length).toFixed(1));
    }

    return summary;
  }

  function hasPopulatedPerCheckSummary(perCheckSummary) {
    if (!perCheckSummary || typeof perCheckSummary !== "object") {
      return false;
    }
    return CHECK_CONFIG.some((check) => {
      const entry = perCheckSummary[check.key];
      if (!entry || typeof entry !== "object") return false;
      return (asNumber(entry.good) ?? 0) > 0
        || (asNumber(entry.needs_work) ?? 0) > 0
        || (asNumber(entry.poor) ?? 0) > 0
        || (asNumber(entry.avg_points) ?? 0) > 0
        || (asNumber(entry.makes_avg) ?? 0) > 0
        || (asNumber(entry.misses_avg) ?? 0) > 0;
    });
  }

  function hasSessionCoachingData(sessionCoaching) {
    if (!sessionCoaching || typeof sessionCoaching !== "object") {
      return false;
    }

    if (sessionCoaching.make_vs_miss_insight || sessionCoaching.overall_consistency) {
      return true;
    }

    const sessionScore = asNumber(sessionCoaching.session_shotsync_score);
    if (sessionScore !== null && sessionScore > 0) {
      return true;
    }

    return hasPopulatedPerCheckSummary(sessionCoaching.per_check_summary);
  }

  function deriveSessionCoachingFromShots(shots, baselineSummary) {
    const perCheckSummary = {};
    const issueCounts = [];
    const goodCounts = [];

    for (const check of CHECK_CONFIG) {
      const summary = summarizeCheckFromShots(check.key, shots);
      perCheckSummary[check.key] = summary;
      issueCounts.push({ key: check.key, count: summary.needs_work + summary.poor });
      goodCounts.push({ key: check.key, count: summary.good });
    }

    const bestIssue = issueCounts.slice().sort((a, b) => b.count - a.count)[0];
    const bestStrength = goodCounts.slice().sort((a, b) => b.count - a.count)[0];

    let topImprovement = baselineSummary?.top_improvement || {};
    if (!topImprovement?.check && bestIssue && bestIssue.count > 0) {
      topImprovement = {
        check: bestIssue.key,
        message: `${prettyCheckName(bestIssue.key)} is your biggest area for improvement based on this session.`,
      };
    }

    let strength = baselineSummary?.strength || {};
    if (!strength?.check && bestStrength && bestStrength.count > 0) {
      strength = {
        check: bestStrength.key,
        message: `${prettyCheckName(bestStrength.key)} is currently your strongest check.`,
      };
    }

    let biggestGap = null;
    for (const check of CHECK_CONFIG) {
      const entry = perCheckSummary[check.key] || {};
      const makesAvg = asNumber(entry.makes_avg);
      const missesAvg = asNumber(entry.misses_avg);
      if (makesAvg === null || missesAvg === null) continue;
      const gap = Math.abs(makesAvg - missesAvg);
      if (!biggestGap || gap > biggestGap.gap) {
        biggestGap = { key: check.key, makesAvg, missesAvg, gap };
      }
    }

    const makesScores = shots
      .filter((shot) => shot.outcome === "make")
      .map((shot) => asNumber(shot.shotsync_score))
      .filter((score) => score !== null);
    const missesScores = shots
      .filter((shot) => shot.outcome === "miss")
      .map((shot) => asNumber(shot.shotsync_score))
      .filter((score) => score !== null);
    const makesAvgScore = makesScores.length
      ? Number((makesScores.reduce((a, b) => a + b, 0) / makesScores.length).toFixed(1))
      : null;
    const missesAvgScore = missesScores.length
      ? Number((missesScores.reduce((a, b) => a + b, 0) / missesScores.length).toFixed(1))
      : null;

    const generatedMakeVsMiss = biggestGap
      ? `Biggest make-vs-miss difference is ${prettyCheckName(biggestGap.key)} (${biggestGap.makesAvg.toFixed(1)} on makes vs ${biggestGap.missesAvg.toFixed(1)} on misses).`
      : null;

    return {
      ...baselineSummary,
      session_shotsync_score: asNumber(baselineSummary?.session_shotsync_score) ?? averageScore(shots),
      makes_avg_score: asNumber(baselineSummary?.makes_avg_score) ?? makesAvgScore,
      misses_avg_score: asNumber(baselineSummary?.misses_avg_score) ?? missesAvgScore,
      top_improvement: topImprovement,
      strength,
      make_vs_miss_insight: baselineSummary?.make_vs_miss_insight || generatedMakeVsMiss,
      per_check_summary: hasPopulatedPerCheckSummary(baselineSummary?.per_check_summary)
        ? baselineSummary.per_check_summary
        : perCheckSummary,
      overall_consistency: baselineSummary?.overall_consistency || "Keep collecting behind-basket shots to improve consistency tracking.",
      zone_breakdown: baselineSummary?.zone_breakdown || {},
    };
  }

  function resolveSessionCoaching(payload, summary, shots) {
    const candidates = [
      summary?.session_coaching,
      payload?.session_coaching,
      payload?.coaching_summary,
    ];

    for (const candidate of candidates) {
      if (hasSessionCoachingData(candidate)) {
        return deriveSessionCoachingFromShots(shots, candidate);
      }
    }

    return deriveSessionCoachingFromShots(shots, {});
  }

  function getSessionSummary(payload, shots) {
    const summary = payload?.session_summary || {};
    const makes = asNumber(summary?.makes ?? payload?.total_makes);
    const total = asNumber(summary?.total_shots ?? payload?.total_attempts ?? shots.length);
    const misses = asNumber(summary?.misses);
    const shootingPercentage = asNumber(summary?.shooting_percentage ?? payload?.shooting_percentage);

    return {
      ...summary,
      makes: makes ?? shots.filter((shot) => shot.outcome === "make").length,
      total_shots: total ?? shots.length,
      misses: misses ?? Math.max(0, (total ?? shots.length) - (makes ?? 0)),
      shooting_percentage:
        shootingPercentage ??
        ((total && total > 0) ? ((100 * (makes ?? 0)) / total) : 0),
      session_coaching: resolveSessionCoaching(payload, summary, shots),
    };
  }

  function averageScore(shots) {
    const values = shots
      .map((shot) => asNumber(shot?.shotsync_score))
      .filter((score) => score !== null);
    if (!values.length) {
      return null;
    }
    return values.reduce((sum, score) => sum + score, 0) / values.length;
  }

  function refreshSessionTotals() {
    const total = state.shots.length;
    const makes = state.shots.filter((shot) => shot.outcome === "make").length;
    const misses = Math.max(0, total - makes);
    const pct = total > 0 ? (100 * makes) / total : 0;
    state.sessionSummary = {
      ...state.sessionSummary,
      total_shots: total,
      makes,
      misses,
      shooting_percentage: Number(pct.toFixed(1)),
    };
  }

  function showScreen(screen) {
    el.uploadScreen.hidden = screen !== "upload";
    el.loadingScreen.hidden = screen !== "loading";
    el.resultsScreen.hidden = screen !== "results";
  }

  function clearUploadError() {
    el.uploadError.hidden = true;
    el.uploadError.textContent = "";
  }

  function showUploadError(message) {
    el.uploadError.hidden = false;
    el.uploadError.textContent = message;
  }

  function setSelectedFile(file) {
    state.file = file || null;
    if (state.fileObjectUrl) {
      URL.revokeObjectURL(state.fileObjectUrl);
      state.fileObjectUrl = null;
    }
    if (file) {
      state.fileObjectUrl = URL.createObjectURL(file);
      const mb = (file.size / (1024 * 1024)).toFixed(1);
      if (el.fileName) el.fileName.textContent = `${file.name} (${mb} MB)`;
    } else {
      if (el.fileName) el.fileName.textContent = "No file selected";
    }
  }

  function updateProgress(value, text) {
    state.progressHighWater = Math.max(state.progressHighWater, clamp(value, 0, 100));
    state.progressValue = state.progressHighWater;
    el.progressFill.style.width = `${state.progressValue.toFixed(1)}%`;
    el.loadingScreen.querySelector(".progress-track")?.setAttribute("aria-valuenow", `${Math.round(state.progressValue)}`);
    if (el.progressPercent) {
      el.progressPercent.textContent = `${Math.round(state.progressValue)}%`;
    }
    if (text) {
      state.progressStageText = String(text);
    }
    const pct = Math.round(state.progressValue);
    if (pct >= 100) {
      el.progressText.textContent = "Analysis complete.";
      return;
    }
    el.progressText.textContent = "Analyzing your shots...";
  }

  function progressDriftStep(currentPct) {
    if (currentPct < 40) return 1.2;
    if (currentPct < 70) return 0.7;
    if (currentPct < 90) return 0.35;
    return 0.15;
  }

  function startLoadingIndicators() {
    state.progressValue = 0;
    state.progressHighWater = 0;
    state.progressStageText = "Uploading video...";
    updateProgress(4, "Uploading video...");

    state.statusPollTimer = window.setInterval(async () => {
      try {
        const response = await fetch("/api/shotlab_status", { cache: "no-store" });
        if (!response.ok) return;
        const status = await response.json();
        const progressRaw = asNumber(status?.progress);
        const progressFromStatus = progressRaw !== null
          ? (progressRaw <= 1 ? progressRaw * 100 : progressRaw)
          : null;
        const message = status?.message || status?.stage_message || status?.phase || "Analyzing your shots...";
        if (progressFromStatus !== null) {
          const backendProgress = clamp(progressFromStatus, 0, 97);
          const driftedProgress = Math.min(96.8, state.progressValue + progressDriftStep(state.progressValue));
          const monotonicProgress = Math.max(backendProgress, driftedProgress);
          updateProgress(monotonicProgress, message);
        } else {
          const fallbackProgress = Math.min(96.8, state.progressValue + progressDriftStep(state.progressValue));
          updateProgress(fallbackProgress, message);
        }
      } catch (_err) {
        // Keep UI smooth even if polling fails.
      }
    }, 1400);
  }

  function stopLoadingIndicators() {
    if (state.statusPollTimer) {
      clearInterval(state.statusPollTimer);
      state.statusPollTimer = null;
    }
    updateProgress(100, "Analysis complete.");
  }

  function renderGauge(score) {
    const safeScore = isNumber(score) ? clamp(score, 0, 100) : null;
    const tier = scoreTier(safeScore);

    el.shotsyncGauge.style.setProperty("--value", `${safeScore ?? 0}`);
    el.shotsyncGauge.style.setProperty("--gauge-color", tier.color);
    el.gaugeValue.textContent = tier.letter;
    if (el.gaugeScore) {
      el.gaugeScore.textContent = safeScore === null ? "--/100" : `${Math.round(safeScore)}/100`;
    }
    el.gaugeLabel.textContent = tier.label;
    el.gaugeLabel.style.color = tier.color;
  }

  /* ─── Change 2: Visual Shot Strip ─── */

  function renderShotStrip() {
    const totalShots = state.shots.length;
    const makes = state.shots.filter((s) => s.outcome === "make").length;
    const pct = totalShots > 0 ? (100 * makes / totalShots) : 0;

    const dots = state.shots
      .slice()
      .sort((a, b) => a.shot_num - b.shot_num)
      .map((shot) => {
        const cls = shot.outcome === "make" ? "strip-dot make" : (shot.outcome === "miss" ? "strip-dot miss" : "strip-dot unknown");
        return `<button type="button" class="strip-dot-btn ${cls}" data-shot-num="${shot.shot_num}" title="Shot ${shot.shot_num}: ${shot.outcome}"></button>`;
      })
      .join("");

    el.quickStats.innerHTML = `
      <div class="shot-strip-row">${dots}</div>
      <div class="shot-strip-label">${totalShots} shots \u2022 ${pct.toFixed(1)}% shooting</div>
    `;

    el.quickStats.querySelectorAll(".strip-dot-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        const shotNum = asNumber(btn.getAttribute("data-shot-num"));
        if (shotNum === null) return;
        const card = el.shotList.querySelector(`[data-shot-card="${shotNum}"]`);
        if (card) {
          card.scrollIntoView({ behavior: "smooth", block: "center" });
          card.open = true;
        }
      });
    });
  }

  function renderSessionHeader() {
    const summary = state.sessionSummary;
    const sessionCoaching = summary.session_coaching || {};

    const backendSessionScore = asNumber(sessionCoaching.session_shotsync_score);
    const fallbackSessionScore = averageScore(state.shots);
    const sessionScore = (backendSessionScore !== null && (backendSessionScore > 0 || fallbackSessionScore === null))
      ? backendSessionScore
      : fallbackSessionScore;
    renderGauge(sessionScore);

    renderShotStrip();
  }

  function titleFromCheck(checkKey, fallbackTitle) {
    if (!checkKey) {
      return fallbackTitle;
    }
    return prettyCheckName(checkKey);
  }

  function gradeArticle(letter) {
    return ["A", "F"].includes(String(letter || "").toUpperCase()) ? "an" : "a";
  }

  function findPositiveMakeGaps(perCheckSummary) {
    if (!perCheckSummary || typeof perCheckSummary !== "object") return [];
    const items = [];
    CHECK_CONFIG.forEach((check) => {
      const entry = perCheckSummary?.[check.key] || {};
      const makesAvg = asNumber(entry.makes_avg);
      const missesAvg = asNumber(entry.misses_avg);
      const maxPoints = asNumber(check.max) ?? 20;
      if (makesAvg === null || missesAvg === null || !isNumber(maxPoints) || maxPoints <= 0) return;
      const rawGap = makesAvg - missesAvg;
      if (rawGap <= 0) return;
      const normalizedGap = rawGap / maxPoints;
      items.push({
        key: check.key,
        label: check.label,
        makesAvg,
        missesAvg,
        maxPoints,
        rawGap,
        normalizedGap,
      });
    });
    if (!items.length) return [];
    items.sort((a, b) => b.normalizedGap - a.normalizedGap || b.rawGap - a.rawGap);
    return items;
  }

  function findBestPositiveMakeGap(perCheckSummary) {
    const ranked = findPositiveMakeGaps(perCheckSummary);
    return ranked.length ? ranked[0] : null;
  }

  function buildMakeVsMissGradeText(coaching) {
    const makesAvg = asNumber(coaching?.makes_avg_score);
    const missesAvg = asNumber(coaching?.misses_avg_score);
    const makes = state.shots.filter((shot) => shot.outcome === "make").length;
    const misses = state.shots.filter((shot) => shot.outcome === "miss").length;
    const total = state.shots.length;

    if (total <= 1) {
      return "Keep shooting — more reps will unlock make-vs-miss comparisons.";
    }
    if (makes === 0) {
      return "Keep shooting — makes will come as your form improves.";
    }
    if (misses === 0) {
      return "All makes! Try some longer shots to challenge your form.";
    }

    const gap = findBestPositiveMakeGap(coaching?.per_check_summary);
    if (makesAvg !== null && missesAvg !== null) {
      const makeTier = scoreTier(makesAvg);
      const missTier = scoreTier(missesAvg);
      const lead = `Your makes average ${gradeArticle(makeTier.letter)} ${makeTier.letter} (${Math.round(makeTier.numeric)}) vs your misses at ${gradeArticle(missTier.letter)} ${missTier.letter} (${Math.round(missTier.numeric)}).`;
      if (gap?.label) {
        return `${lead} The biggest difference is ${gap.label}.`;
      }
      return `${lead} Your form is similar on makes and misses. Focus on checks rated C or below.`;
    }

    if (gap?.label) {
      const makeTier = scoreTier(gap.makesAvg);
      const missTier = scoreTier(gap.missesAvg);
      return `Your makes and misses differ most in ${gap.label}: makes are ${makeTier.letter} (${Math.round(makeTier.numeric)}) vs misses ${missTier.letter} (${Math.round(missTier.numeric)}).`;
    }
    return "Your form is similar on makes and misses. Focus on checks rated C or below to improve overall.";
  }

  function sourceCandidatesForShot(shot) {
    return [
      String(shot?.clip_url || "").trim(),
      String(shot?.video_url || "").trim(),
      String(state.payload?.video_url || "").trim(),
      String(state.fileObjectUrl || "").trim(),
    ].filter((item, idx, arr) => item && arr.indexOf(item) === idx);
  }

  function attachRangedClipController(videoEl, shot, options = {}) {
    const candidates = sourceCandidatesForShot(shot);
    const onUnavailable = typeof options.onUnavailable === "function" ? options.onUnavailable : null;
    if (!videoEl || !candidates.length) {
      if (onUnavailable) onUnavailable();
      return null;
    }
    const clipUrl = String(shot?.clip_url || "").trim();
    const clipStart = asNumber(shot?.clip_start_time) ?? 0;
    const clipEnd = asNumber(shot?.clip_end_time);
    const releaseTime = asNumber(shot?.release_time);
    let sourceIdx = 0;
    let usingClipAsset = false;
    let range = null;
    let unavailable = false;
    let pendingPlay = false;
    let pendingSeekStart = true;
    let loadGuardTimer = null;

    const computeRange = (duration) => {
      const safeDuration = isNumber(duration) ? Math.max(0.2, duration) : 0.2;
      if (usingClipAsset) {
        const end = safeDuration;
        const releaseOffset = releaseTime !== null
          ? clamp(releaseTime - clipStart, 0, end)
          : Math.min(end * 0.5, end);
        return { start: 0, end, releaseOffset };
      }
      const inferredStart = (isNumber(clipStart) && clipStart >= 0) ? clipStart : 0;
      const startTooLarge = inferredStart > (safeDuration + 0.05);
      let start = startTooLarge ? 0 : inferredStart;
      start = clamp(start, 0, Math.max(0, safeDuration - 0.2));

      const rawEnd = (isNumber(clipEnd) && clipEnd > start) ? clipEnd : null;
      let end = rawEnd ?? safeDuration;
      end = clamp(end, start + 0.2, safeDuration);
      if ((end - start) < 0.2) {
        start = Math.max(0, safeDuration - 0.2);
        end = safeDuration;
      }
      const releaseOffset = releaseTime !== null
        ? clamp(releaseTime - start, 0, Math.max(0.01, end - start))
        : Math.max(0.01, (end - start) * 0.5);
      return { start, end, releaseOffset };
    };

    const seekStart = () => {
      if (!range) {
        pendingSeekStart = true;
        return;
      }
      pendingSeekStart = false;
      try { videoEl.currentTime = range.start; } catch (_e) {}
    };

    const onTimeUpdate = () => {
      if (!range) return;
      if ((asNumber(videoEl.currentTime) ?? 0) >= range.end) {
        videoEl.pause();
      }
    };

    const loadSource = (idx) => {
      if (idx < 0 || idx >= candidates.length) {
        unavailable = true;
        if (onUnavailable) onUnavailable();
        return;
      }
      sourceIdx = idx;
      usingClipAsset = (candidates[idx] === clipUrl && clipUrl.length > 0);
      range = null;
      if (loadGuardTimer) {
        clearTimeout(loadGuardTimer);
        loadGuardTimer = null;
      }
      videoEl.src = candidates[idx];
      videoEl.load();
      loadGuardTimer = window.setTimeout(() => {
        if (range || unavailable) return;
        if (sourceIdx < candidates.length - 1) {
          loadSource(sourceIdx + 1);
        } else {
          unavailable = true;
          if (onUnavailable) onUnavailable();
        }
      }, 2600);
    };

    const onLoadedMetadata = () => {
      if (loadGuardTimer) {
        clearTimeout(loadGuardTimer);
        loadGuardTimer = null;
      }
      const duration = asNumber(videoEl.duration);
      if ((duration === null || duration <= 0.05) && sourceIdx < candidates.length - 1) {
        loadSource(sourceIdx + 1);
        return;
      }
      range = computeRange(duration);
      if (pendingSeekStart || !options.autoplay) {
        seekStart();
      }
      if (options.autoplay || pendingPlay) {
        pendingPlay = false;
        videoEl.play().catch(() => {});
      }
    };

    const onError = () => {
      if (loadGuardTimer) {
        clearTimeout(loadGuardTimer);
        loadGuardTimer = null;
      }
      if (sourceIdx < candidates.length - 1) {
        loadSource(sourceIdx + 1);
        return;
      }
      unavailable = true;
      if (onUnavailable) onUnavailable();
    };

    videoEl.addEventListener("timeupdate", onTimeUpdate);
    videoEl.addEventListener("loadedmetadata", onLoadedMetadata);
    videoEl.addEventListener("error", onError);
    loadSource(0);

    return {
      isReady() {
        return Boolean(range) && !unavailable;
      },
      seekStart,
      play() {
        if (unavailable) return;
        if (!range) {
          pendingPlay = true;
          return;
        }
        pendingPlay = false;
        videoEl.play().catch(() => {});
      },
      pause() {
        pendingPlay = false;
        videoEl.pause();
      },
      playFromStart() {
        if (unavailable) return;
        pendingSeekStart = true;
        pendingPlay = true;
        if (!range) return;
        seekStart();
        pendingPlay = false;
        videoEl.play().catch(() => {});
      },
      getReleaseOffset() {
        return range ? range.releaseOffset : 0;
      },
      getReleaseMediaTime() {
        return range ? (range.start + range.releaseOffset) : null;
      },
      getEndTime() {
        return range ? range.end : null;
      },
      getCurrentTime() {
        return asNumber(videoEl.currentTime) ?? 0;
      },
      destroy() {
        if (loadGuardTimer) {
          clearTimeout(loadGuardTimer);
          loadGuardTimer = null;
        }
        videoEl.removeEventListener("timeupdate", onTimeUpdate);
        videoEl.removeEventListener("loadedmetadata", onLoadedMetadata);
        videoEl.removeEventListener("error", onError);
        try { videoEl.pause(); } catch (_e) {}
      },
    };
  }

  function bestComparisonShots() {
    const makes = state.shots
      .filter((shot) => shot.outcome === "make")
      .slice()
      .sort((a, b) => (asNumber(b.shotsync_score) ?? -1) - (asNumber(a.shotsync_score) ?? -1));
    const misses = state.shots
      .filter((shot) => shot.outcome === "miss")
      .slice()
      .sort((a, b) => (asNumber(a.shotsync_score) ?? 101) - (asNumber(b.shotsync_score) ?? 101));
    return {
      makeShot: makes[0] || null,
      missShot: misses[0] || null,
      makeCount: makes.length,
      missCount: misses.length,
    };
  }

  function buildComparisonInsightText(coaching) {
    const backendText = String(coaching?.make_vs_miss_insight || "").trim();
    const generated = buildMakeVsMissGradeText(coaching);
    if (!backendText) return generated;
    if (/\d/.test(backendText)) return generated;
    return `${generated} ${backendText}`.trim();
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function gradeFromCheckPoints(points, maxPoints) {
    if (!isNumber(points) || !isNumber(maxPoints) || maxPoints <= 0) {
      return scoreTier(null);
    }
    return scoreTier((points / maxPoints) * 100);
  }

  function defaultImprovementTip(checkKey) {
    if (checkKey === "follow_through") return "Try holding your arm up after release.";
    if (checkKey === "release_height") return "Try releasing higher with full extension.";
    if (checkKey === "elbow_alignment") return "Try keeping your elbow stacked under the ball.";
    if (checkKey === "guide_hand") return "Try using your guide hand only for balance.";
    if (checkKey === "base_and_balance") return "Try staying balanced through your landing.";
    if (checkKey === "shoulder_alignment") return "Try keeping shoulders square to the rim.";
    return "Try repeating the make mechanics in your next rep.";
  }

  function buildComparisonCalloutData(coaching, makeShot, missShot, selectedKey = null) {
    const rankedGaps = findPositiveMakeGaps(coaching?.per_check_summary);
    const positiveGap = selectedKey
      ? (rankedGaps.find((item) => item.key === selectedKey) || rankedGaps[0] || null)
      : (rankedGaps[0] || null);
    const checkKey = positiveGap?.key || "follow_through";
    const checkLabel = positiveGap?.label || prettyCheckName(checkKey);
    const checkCfg = CHECK_CONFIG.find((check) => check.key === checkKey) || { max: 20, label: checkLabel };
    const maxPoints = asNumber(checkCfg.max) ?? 20;

    const makeEntry = (makeShot?.coaching && typeof makeShot.coaching === "object")
      ? (makeShot.coaching[checkKey] || {})
      : {};
    const missEntry = (missShot?.coaching && typeof missShot.coaching === "object")
      ? (missShot.coaching[checkKey] || {})
      : {};

    const makePoints = asNumber(positiveGap?.makesAvg ?? makeEntry?.points);
    const missPoints = asNumber(positiveGap?.missesAvg ?? missEntry?.points);
    const makeTier = gradeFromCheckPoints(makePoints, maxPoints);
    const missTier = gradeFromCheckPoints(missPoints, maxPoints);

    const makeDesc = compactTip(makeEntry?.feedback, checkLabel, makeEntry?.status);
    const missDesc = compactTip(missEntry?.feedback, checkLabel, missEntry?.status);
    const tip = missEntry?.feedback
      ? compactTip(missEntry.feedback, checkLabel, "needs_work")
      : (makeEntry?.feedback
        ? compactTip(makeEntry.feedback, checkLabel, "good")
        : defaultImprovementTip(checkKey));

    if (!positiveGap) {
      return {
        key: checkKey,
        label: "FORM TREND",
        makeTier,
        missTier,
        makeDesc: "Makes and misses are graded similarly.",
        missDesc: "No clear single check separates outcomes.",
        tip: "Your form is similar on makes and misses. Focus on the checks rated C or below to improve overall.",
        hasPositiveGap: false,
        differences: [],
      };
    }

    return {
      key: checkKey,
      label: checkLabel,
      makeTier,
      missTier,
      makeDesc,
      missDesc,
      tip,
      hasPositiveGap: true,
      differences: rankedGaps.map((item) => {
        const makeTierForItem = gradeFromCheckPoints(item.makesAvg, item.maxPoints);
        const missTierForItem = gradeFromCheckPoints(item.missesAvg, item.maxPoints);
        return {
          key: item.key,
          label: item.label,
          makeTier: makeTierForItem,
          missTier: missTierForItem,
        };
      }),
    };
  }

  function destroyComparisonController() {
    if (state.comparisonController && typeof state.comparisonController.destroy === "function") {
      state.comparisonController.destroy();
    }
    state.comparisonController = null;
  }

  function renderMakeVsMissComparison() {
    destroyComparisonController();
    if (!el.comparisonSection || !el.comparisonContent) return;

    const coaching = state.sessionSummary.session_coaching || {};
    const { makeShot, missShot, makeCount, missCount } = bestComparisonShots();
    const total = state.shots.length;
    if (total <= 1 || makeCount === 0 || missCount === 0 || !makeShot || !missShot) {
      el.comparisonSection.hidden = true;
      el.comparisonContent.innerHTML = "";
      return;
    }

    el.comparisonSection.hidden = false;
    const makeForm = makeGradeSpan(makeShot.shotsync_score);
    const missForm = makeGradeSpan(missShot.shotsync_score);
    const insightText = buildComparisonInsightText(coaching);
    const callout = buildComparisonCalloutData(coaching, makeShot, missShot);
    let selectedDiffKey = callout.key;
    const makeGradeNum = isNumber(callout.makeTier.numeric) ? Math.round(callout.makeTier.numeric) : "--";
    const missGradeNum = isNumber(callout.missTier.numeric) ? Math.round(callout.missTier.numeric) : "--";
    const diffTabsHtml = (Array.isArray(callout.differences) && callout.differences.length > 1)
      ? `<div class="comparison-diff-tabs" id="comparisonDiffTabs">${
        callout.differences.map((diff) => {
          const makeNum = isNumber(diff.makeTier?.numeric) ? Math.round(diff.makeTier.numeric) : "--";
          const missNum = isNumber(diff.missTier?.numeric) ? Math.round(diff.missTier.numeric) : "--";
          return `<button type="button" class="comparison-diff-tab ${diff.key === selectedDiffKey ? "active" : ""}" data-diff-key="${escapeHtml(diff.key)}">${escapeHtml(diff.label)} <span class="diff-inline-grade">${escapeHtml(diff.makeTier.letter)}(${escapeHtml(makeNum)}) vs ${escapeHtml(diff.missTier.letter)}(${escapeHtml(missNum)})</span></button>`;
        }).join("")
      }</div>`
      : "";
    const makeSources = sourceCandidatesForShot(makeShot);
    const missSources = sourceCandidatesForShot(missShot);
    const hasAnyVideo = makeSources.length > 0 || missSources.length > 0;
    const hasBothVideo = makeSources.length > 0 && missSources.length > 0;
    if (!hasAnyVideo) {
      el.comparisonContent.innerHTML = `<div class="comparison-no-video">${escapeHtml(insightText)}</div>`;
      return;
    }
    el.comparisonContent.innerHTML = `
      <div class="comparison-grid comparison-grid-guided">
        <div class="comparison-col">
          <div class="comparison-head">Your Make</div>
          <div class="comparison-form">${makeForm}</div>
          <div class="comparison-video-wrap make" data-side="make">
            <video class="comparison-video make" controls muted playsinline preload="metadata"></video>
            <div class="comparison-release-corner">📍 Release</div>
            <div class="comparison-release-marker"></div>
          </div>
        </div>
        <div class="comparison-col">
          <div class="comparison-head">Your Miss</div>
          <div class="comparison-form">${missForm}</div>
          <div class="comparison-video-wrap miss" data-side="miss">
            <video class="comparison-video miss" controls muted playsinline preload="metadata"></video>
            <div class="comparison-release-corner">📍 Release</div>
            <div class="comparison-release-marker"></div>
          </div>
        </div>
      </div>
      <div class="comparison-callout-slot comparison-callout-row">
        <div
          class="comparison-callout"
          id="comparisonDiffCard"
          style="--make-grade-color:${escapeHtml(callout.makeTier.color)};--miss-grade-color:${escapeHtml(callout.missTier.color)};"
          hidden
        >
          <div class="comparison-callout-title">BIGGEST DIFFERENCE:</div>
          ${diffTabsHtml}
          <div class="comparison-callout-line"><strong id="comparisonDiffName">${escapeHtml(callout.label)}</strong></div>
          <div class="comparison-callout-line" id="comparisonDiffScoreLine">
            <span class="comparison-callout-line-head make">Make:</span>
            ${escapeHtml(callout.makeTier.letter)} (${escapeHtml(makeGradeNum)})
            <span class="comparison-callout-line-head miss" style="margin-left:8px;">vs Miss:</span>
            ${escapeHtml(callout.missTier.letter)} (${escapeHtml(missGradeNum)})
          </div>
          <div class="comparison-callout-tip" id="comparisonDiffTip">${
            escapeHtml(
              callout.hasPositiveGap
                ? `Your ${callout.label} is ${callout.makeTier.letter} on makes but ${callout.missTier.letter} on misses. ${callout.tip}`
                : callout.tip
            )
          }</div>
          <div class="comparison-callout-actions">
            <button type="button" id="comparisonContinueBtn" class="comparison-continue-btn">▶ Resume Both</button>
          </div>
        </div>
      </div>
      ${hasAnyVideo ? `<div class="comparison-actions">
        <button type="button" id="playBothBtn" class="play-both-btn" ${hasBothVideo ? "" : "disabled"}>▶ Play Both Side by Side</button>
        <button type="button" id="comparisonReplayBtn" class="play-both-btn ghost" hidden>↻ Replay</button>
      </div>` : ""}
      <div class="comparison-insight">${escapeHtml(insightText)}</div>
    `;

    const makeVideo = el.comparisonContent.querySelector(".comparison-video.make");
    const missVideo = el.comparisonContent.querySelector(".comparison-video.miss");
    const makeWrap = el.comparisonContent.querySelector('.comparison-video-wrap[data-side="make"]');
    const missWrap = el.comparisonContent.querySelector('.comparison-video-wrap[data-side="miss"]');
    const playBothBtn = el.comparisonContent.querySelector("#playBothBtn");
    const replayBtn = el.comparisonContent.querySelector("#comparisonReplayBtn");
    const diffCard = el.comparisonContent.querySelector("#comparisonDiffCard");
    const continueBtn = el.comparisonContent.querySelector("#comparisonContinueBtn");
    const diffNameEl = el.comparisonContent.querySelector("#comparisonDiffName");
    const diffScoreEl = el.comparisonContent.querySelector("#comparisonDiffScoreLine");
    const diffTipEl = el.comparisonContent.querySelector("#comparisonDiffTip");
    const diffTabs = Array.from(el.comparisonContent.querySelectorAll(".comparison-diff-tab"));
    if (!makeVideo || !missVideo || !makeWrap || !missWrap) return;

    let delayTimer = null;
    let runStarted = false;
    let runCompleted = false;
    let releaseFrozen = false;
    let releaseReached = { make: false, miss: false };
    let clipEnded = { make: false, miss: false };

    const resetRunState = () => {
      runStarted = false;
      runCompleted = false;
      releaseFrozen = false;
      releaseReached = { make: false, miss: false };
      clipEnded = { make: false, miss: false };
    };

    const clearTimers = () => {
      if (delayTimer) {
        clearTimeout(delayTimer);
        delayTimer = null;
      }
    };

    const setReleaseFocus = (active) => {
      makeWrap.classList.toggle("release-focus", active);
      missWrap.classList.toggle("release-focus", active);
    };

    const showCallout = (visible, animate = false) => {
      if (!diffCard) return;
      diffCard.hidden = !visible;
      diffCard.classList.toggle("visible", visible);
      if (animate && visible) {
        diffCard.classList.remove("animate-in");
        void diffCard.offsetWidth;
        diffCard.classList.add("animate-in");
      }
      if (continueBtn) {
        continueBtn.hidden = !visible;
      }
    };

    const renderSelectedDifference = (diffKey) => {
      const next = buildComparisonCalloutData(coaching, makeShot, missShot, diffKey);
      selectedDiffKey = next.key;
      if (diffNameEl) diffNameEl.textContent = next.label;
      if (diffScoreEl) {
        const nextMakeNum = isNumber(next.makeTier.numeric) ? Math.round(next.makeTier.numeric) : "--";
        const nextMissNum = isNumber(next.missTier.numeric) ? Math.round(next.missTier.numeric) : "--";
        diffScoreEl.innerHTML = `<span class="comparison-callout-line-head make">Make:</span> ${escapeHtml(next.makeTier.letter)} (${escapeHtml(nextMakeNum)}) <span class="comparison-callout-line-head miss" style="margin-left:8px;">vs Miss:</span> ${escapeHtml(next.missTier.letter)} (${escapeHtml(nextMissNum)})`;
      }
      if (diffTipEl) {
        diffTipEl.textContent = next.hasPositiveGap
          ? `Your ${next.label} is ${next.makeTier.letter} on makes but ${next.missTier.letter} on misses. ${next.tip}`
          : next.tip;
      }
      if (diffCard) {
        diffCard.style.setProperty("--make-grade-color", next.makeTier.color);
        diffCard.style.setProperty("--miss-grade-color", next.missTier.color);
      }
      diffTabs.forEach((btn) => {
        btn.classList.toggle("active", btn.getAttribute("data-diff-key") === selectedDiffKey);
      });
    };

    const updateActionButtons = () => {
      if (!playBothBtn) return;
      if (releaseFrozen) {
        playBothBtn.hidden = true;
        if (replayBtn) replayBtn.hidden = true;
        return;
      }
      playBothBtn.hidden = false;
      if (runCompleted) {
        playBothBtn.textContent = "▶ Play Both Side by Side";
        if (replayBtn) replayBtn.hidden = false;
        return;
      }
      if (!makeVideo.paused || !missVideo.paused) {
        playBothBtn.textContent = "❚❚ Pause Both";
      } else if (runStarted) {
        playBothBtn.textContent = "▶ Resume Both";
      } else {
        playBothBtn.textContent = "▶ Play Both Side by Side";
      }
      if (replayBtn) replayBtn.hidden = true;
    };

    const makeCtrl = (makeSources.length > 0) ? attachRangedClipController(makeVideo, makeShot, {
      onUnavailable: () => {
        makeVideo.hidden = true;
        makeWrap.classList.add("comparison-video-unavailable");
      },
    }) : null;
    if (makeSources.length === 0) {
      makeVideo.hidden = true;
      makeWrap.classList.add("comparison-video-unavailable");
    }
    const missCtrl = (missSources.length > 0) ? attachRangedClipController(missVideo, missShot, {
      onUnavailable: () => {
        missVideo.hidden = true;
        missWrap.classList.add("comparison-video-unavailable");
      },
    }) : null;
    if (missSources.length === 0) {
      missVideo.hidden = true;
      missWrap.classList.add("comparison-video-unavailable");
    }

    const pauseBoth = () => {
      if (delayTimer) {
        clearTimeout(delayTimer);
        delayTimer = null;
      }
      makeCtrl?.pause();
      missCtrl?.pause();
      updateActionButtons();
    };

    const resumeBoth = () => {
      if (!makeCtrl?.isReady() || !missCtrl?.isReady()) return;
      makeCtrl.play();
      missCtrl.play();
      updateActionButtons();
    };

    const finishRun = () => {
      if (runCompleted) return;
      runCompleted = true;
      runStarted = false;
      releaseFrozen = false;
      clearTimers();
      makeCtrl?.pause();
      missCtrl?.pause();
      setReleaseFocus(false);
      showCallout(true, false);
      if (continueBtn) continueBtn.hidden = true;
      updateActionButtons();
    };

    const resumeAfterFreeze = () => {
      if (!releaseFrozen) return;
      releaseFrozen = false;
      if (continueBtn) continueBtn.hidden = true;
      setReleaseFocus(false);
      resumeBoth();
    };

    const freezeAtRelease = () => {
      if (releaseFrozen || runCompleted || !runStarted) return;
      releaseFrozen = true;
      makeCtrl?.pause();
      missCtrl?.pause();
      setReleaseFocus(true);
      showCallout(true, true);
      updateActionButtons();
    };

    const onVideoTick = () => {
      if (!runStarted || runCompleted) return;

      const makeRelease = makeCtrl?.getReleaseMediaTime();
      const missRelease = missCtrl?.getReleaseMediaTime();
      const makeEnd = makeCtrl?.getEndTime();
      const missEnd = missCtrl?.getEndTime();
      const makeNow = makeCtrl?.getCurrentTime() ?? 0;
      const missNow = missCtrl?.getCurrentTime() ?? 0;

      if (makeRelease !== null && makeRelease !== undefined && makeNow >= (makeRelease - 0.03)) {
        releaseReached.make = true;
      }
      if (missRelease !== null && missRelease !== undefined && missNow >= (missRelease - 0.03)) {
        releaseReached.miss = true;
      }
      if (!releaseFrozen && releaseReached.make && releaseReached.miss) {
        freezeAtRelease();
      }

      if (makeEnd !== null && makeEnd !== undefined && makeNow >= (makeEnd - 0.03)) {
        clipEnded.make = true;
      }
      if (missEnd !== null && missEnd !== undefined && missNow >= (missEnd - 0.03)) {
        clipEnded.miss = true;
      }
      if (clipEnded.make && clipEnded.miss) {
        finishRun();
      }
    };

    const playBothAligned = () => {
      if (!makeCtrl?.isReady() || !missCtrl?.isReady()) return;
      clearTimers();
      resetRunState();
      showCallout(false, false);
      if (continueBtn) continueBtn.hidden = true;
      setReleaseFocus(false);
      if (replayBtn) replayBtn.hidden = true;

      const makeOffset = makeCtrl.getReleaseOffset();
      const missOffset = missCtrl.getReleaseOffset();
      const delayMs = Math.abs(makeOffset - missOffset) * 1000;

      makeCtrl.seekStart();
      missCtrl.seekStart();
      runStarted = true;
      if (makeOffset >= missOffset) {
        makeCtrl.play();
        missCtrl.pause();
        delayTimer = window.setTimeout(() => {
          missCtrl.play();
          delayTimer = null;
        }, delayMs);
      } else {
        missCtrl.play();
        makeCtrl.pause();
        delayTimer = window.setTimeout(() => {
          makeCtrl.play();
          delayTimer = null;
        }, delayMs);
      }
      updateActionButtons();
    };

    showCallout(false, false);
    renderSelectedDifference(selectedDiffKey);
    makeVideo.addEventListener("timeupdate", onVideoTick);
    missVideo.addEventListener("timeupdate", onVideoTick);
    makeVideo.addEventListener("ended", onVideoTick);
    missVideo.addEventListener("ended", onVideoTick);

    if (playBothBtn) {
      playBothBtn.addEventListener("click", (event) => {
        event.preventDefault();
        if (runCompleted) {
          playBothAligned();
          return;
        }
        if (releaseFrozen) {
          return;
        }
        const anyPlaying = (!makeVideo.paused || !missVideo.paused);
        if (anyPlaying) {
          pauseBoth();
          return;
        }
        if (runStarted) {
          resumeBoth();
          return;
        }
        playBothAligned();
      });
    }
    if (continueBtn) {
      continueBtn.addEventListener("click", (event) => {
        event.preventDefault();
        resumeAfterFreeze();
      });
    }
    diffTabs.forEach((btn) => {
      btn.addEventListener("click", (event) => {
        event.preventDefault();
        const diffKey = String(btn.getAttribute("data-diff-key") || "").trim();
        if (!diffKey) return;
        renderSelectedDifference(diffKey);
      });
    });
    if (replayBtn) {
      replayBtn.addEventListener("click", (event) => {
        event.preventDefault();
        playBothAligned();
      });
    }
    updateActionButtons();

    state.comparisonController = {
      destroy() {
        clearTimers();
        makeVideo.removeEventListener("timeupdate", onVideoTick);
        missVideo.removeEventListener("timeupdate", onVideoTick);
        makeVideo.removeEventListener("ended", onVideoTick);
        missVideo.removeEventListener("ended", onVideoTick);
        makeCtrl?.destroy();
        missCtrl?.destroy();
      },
    };
  }

  function getCheckSummaryForInsights(checkKey, coaching) {
    const checkCfg = CHECK_CONFIG.find((item) => item.key === checkKey);
    if (!checkCfg) return null;

    const entry = coaching?.per_check_summary?.[checkKey] || {};
    let good = asNumber(entry.good) ?? 0;
    let needsWork = asNumber(entry.needs_work) ?? 0;
    let poor = asNumber(entry.poor) ?? 0;
    const unavailable = asNumber(entry.unavailable) ?? 0;
    let total = good + needsWork + poor + unavailable;

    if (total <= 0) {
      state.shots.forEach((shot) => {
        const check = shot?.coaching?.[checkKey];
        if (!check || typeof check !== "object") return;
        const status = statusClass(check.status);
        if (status === "good") good += 1;
        else if (status === "needs_work") needsWork += 1;
        else if (status === "poor") poor += 1;
      });
      total = good + needsWork + poor;
    }

    let avgPoints = asNumber(entry.avg_points);
    if (avgPoints === null && total > 0) {
      avgPoints = avgPointsFromShots(checkKey);
    }
    if (avgPoints === null) {
      avgPoints = (good > 0 && needsWork === 0 && poor === 0)
        ? checkCfg.max
        : ((needsWork > 0 && poor === 0) ? (checkCfg.max * 0.62) : (checkCfg.max * 0.35));
    }

    const grade = scoreTier((avgPoints / Math.max(1, checkCfg.max)) * 100);
    return {
      key: checkKey,
      label: checkCfg.label,
      max: checkCfg.max,
      avgPoints,
      grade,
      total,
      good,
      needsWork,
      poor,
      issueCount: needsWork + poor,
    };
  }

  function pickTopImprovementCheck(coaching) {
    const preferred = String(coaching?.top_improvement?.check || "").trim();
    if (preferred) {
      const preferredSummary = getCheckSummaryForInsights(preferred, coaching);
      if (preferredSummary) return preferredSummary;
    }
    const summaries = CHECK_CONFIG
      .map((check) => getCheckSummaryForInsights(check.key, coaching))
      .filter(Boolean);
    if (!summaries.length) return null;
    summaries.sort((a, b) => (
      b.issueCount - a.issueCount
      || a.grade.numeric - b.grade.numeric
      || a.label.localeCompare(b.label)
    ));
    return summaries[0];
  }

  function pickStrengthCheck(coaching) {
    const preferred = String(coaching?.strength?.check || "").trim();
    if (preferred) {
      const preferredSummary = getCheckSummaryForInsights(preferred, coaching);
      if (preferredSummary) return preferredSummary;
    }
    const summaries = CHECK_CONFIG
      .map((check) => getCheckSummaryForInsights(check.key, coaching))
      .filter(Boolean);
    if (!summaries.length) return null;
    summaries.sort((a, b) => (
      b.good - a.good
      || b.grade.numeric - a.grade.numeric
      || a.label.localeCompare(b.label)
    ));
    return summaries[0];
  }

  function buildImprovementPotentialPhrase(sessionScore, checkSummary) {
    const score = asNumber(sessionScore);
    if (score === null || !checkSummary) return "";
    const gain = Math.max(0, checkSummary.max - checkSummary.avgPoints);
    if (gain < 10) return "";
    const current = scoreTier(score);
    const projected = scoreTier(score + gain);
    if (!current.letter || !projected.letter || current.letter === projected.letter) {
      return `Fixing this check could add about ${Math.round(gain)} points to your Form Score.`;
    }
    return `Fixing this check alone could move your Form Score from ${current.letter} to ${projected.letter}.`;
  }

  function buildPrescriptiveTopInsight(coaching) {
    const top = pickTopImprovementCheck(coaching);
    if (!top) {
      return {
        title: "Form Consistency",
        text: "Collect more tracked shots to identify your biggest improvement opportunity.",
      };
    }
    const measured = measuredTextForGrade(top.key, top.grade.letter);
    const playbook = playbookForCheck(top.key, top.label);
    const sessionScore = asNumber(coaching?.session_shotsync_score) ?? averageScore(state.shots);
    const potentialPhrase = buildImprovementPotentialPhrase(sessionScore, top);
    const needsPhrase = top.total > 0
      ? `Your ${top.label.toLowerCase()} needed correction on ${top.issueCount} of ${top.total} shots this session.`
      : `${top.label} is currently your biggest opportunity this session.`;
    return {
      title: `${top.label.toUpperCase()} (${top.grade.letter})`,
      text: `${measured} ${needsPhrase} ${potentialPhrase} NEXT SESSION FOCUS: ${playbook.try}`,
    };
  }

  function buildPrescriptiveStrengthInsight(coaching) {
    const strength = pickStrengthCheck(coaching);
    if (!strength) {
      return {
        title: "Session Trend",
        text: "Form trend is unavailable for this run.",
      };
    }
    const measured = measuredTextForGrade(strength.key, strength.grade.letter);
    const reliablePhrase = strength.total > 0
      ? `Great job - ${strength.label.toLowerCase()} stayed solid on ${strength.good} of ${strength.total} shots.`
      : `${strength.label} is currently your strongest check.`;
    return {
      title: `${strength.label.toUpperCase()} (${strength.grade.letter})`,
      text: `${measured} ${reliablePhrase} Keep it up.`,
    };
  }

  function buildPrescriptiveMakeVsMissInsight(coaching) {
    const makes = state.shots.filter((shot) => shot.outcome === "make").length;
    const misses = state.shots.filter((shot) => shot.outcome === "miss").length;
    const total = state.shots.length;
    if (total <= 1) {
      return "Keep shooting - more reps will unlock make-vs-miss comparisons.";
    }
    if (makes === 0) {
      return "Keep shooting - makes will come as your form improves.";
    }
    if (misses === 0) {
      return "All makes! Try some longer shots to challenge your form.";
    }

    const bestGap = findBestPositiveMakeGap(coaching?.per_check_summary);
    const makesAvg = asNumber(coaching?.makes_avg_score);
    const missesAvg = asNumber(coaching?.misses_avg_score);
    const smallSample = total < 15;
    if (!bestGap || (smallSample && makesAvg !== null && missesAvg !== null && makesAvg <= missesAvg)) {
      return `With only ${total} shots, the make/miss comparison needs more data. Shoot a longer session (15+ shots) for a clearer pattern.`;
    }

    const makeTier = gradeFromCheckPoints(bestGap.makesAvg, bestGap.maxPoints);
    const missTier = gradeFromCheckPoints(bestGap.missesAvg, bestGap.maxPoints);
    const cue = comparisonFocusCue(bestGap.key);
    return `Your makes and misses differ most in ${bestGap.label} - ${makeTier.letter} on makes vs ${missTier.letter} on misses. ${cue}`;
  }

  function renderInsights() {
    const coaching = state.sessionSummary.session_coaching || {};
    const hasData = hasSessionCoachingData(coaching);

    if (!hasData) {
      const scoredShots = state.shots
        .map((shot) => asNumber(shot.shotsync_score))
        .filter((score) => score !== null);
      const spread = scoredShots.length
        ? Math.max(...scoredShots) - Math.min(...scoredShots)
        : null;

      el.topImprovementTitle.textContent = "Form Consistency";
      el.topImprovementText.textContent = spread !== null
        ? `Your form spread is ${spread.toFixed(1)} points. Focus on repeating the same mechanics every rep.`
        : "Collect more tracked shots to estimate consistency.";
      el.strengthTitle.textContent = "Session Trend";
      el.strengthText.textContent = averageScore(state.shots) !== null
        ? `Current session form is ${formatFormGrade(averageScore(state.shots)).replace("Form: ", "")}.`
        : "Form trend is unavailable for this run.";
      el.makeVsMissText.textContent = buildPrescriptiveMakeVsMissInsight({
        makes_avg_score: averageScore(state.shots.filter((shot) => shot.outcome === "make")),
        misses_avg_score: averageScore(state.shots.filter((shot) => shot.outcome === "miss")),
        per_check_summary: {},
      });
      return;
    }

    const topInsight = buildPrescriptiveTopInsight(coaching);
    const strengthInsight = buildPrescriptiveStrengthInsight(coaching);

    el.topImprovementTitle.textContent = topInsight.title;
    el.topImprovementText.textContent = topInsight.text;
    el.strengthTitle.textContent = strengthInsight.title;
    el.strengthText.textContent = strengthInsight.text;
    el.makeVsMissText.textContent = buildPrescriptiveMakeVsMissInsight(coaching);
  }

  function dominantStatus(summaryEntry, avgPoints, maxPoints) {
    if (summaryEntry) {
      const counts = [
        { status: "good", count: asNumber(summaryEntry.good) ?? 0, rank: 0 },
        { status: "needs_work", count: asNumber(summaryEntry.needs_work) ?? 0, rank: 1 },
        { status: "poor", count: asNumber(summaryEntry.poor) ?? 0, rank: 2 },
      ];
      counts.sort((a, b) => {
        if (b.count !== a.count) return b.count - a.count;
        return b.rank - a.rank;
      });
      if (counts[0].count > 0) {
        return counts[0].status;
      }
    }

    if (!isNumber(avgPoints) || !isNumber(maxPoints) || maxPoints <= 0) {
      return "unavailable";
    }
    const pct = avgPoints / maxPoints;
    if (pct >= 0.8) return "good";
    if (pct >= 0.4) return "needs_work";
    return "poor";
  }

  function avgPointsFromShots(checkKey) {
    const values = state.shots
      .map((shot) => asNumber(shot?.coaching?.[checkKey]?.points))
      .filter((value) => value !== null);
    if (!values.length) return null;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }

  function checkSampleCountFromShots(checkKey) {
    return state.shots.filter((shot) => {
      const check = shot?.coaching?.[checkKey];
      if (!check || typeof check !== "object") return false;
      const points = asNumber(check.points);
      const status = statusClass(check.status);
      return points !== null || status !== "unavailable";
    }).length;
  }

  /* ─── Radar Form Chart ─── */

  function getCheckAggregate(checkKey, checkMax) {
    const coaching = state.sessionSummary.session_coaching || {};
    const summary = coaching.per_check_summary || {};
    const entry = summary?.[checkKey] || null;
    const summaryCount = (asNumber(entry?.good) ?? 0)
      + (asNumber(entry?.needs_work) ?? 0)
      + (asNumber(entry?.poor) ?? 0)
      + (asNumber(entry?.unavailable) ?? 0);
    const sampleCount = checkSampleCountFromShots(checkKey);
    const avgPoints = sampleCount > 0
      ? avgPointsFromShots(checkKey)
      : (summaryCount > 0 ? asNumber(entry?.avg_points) : null);
    const makesAvg = summaryCount > 0 ? asNumber(entry?.makes_avg) : null;
    const missesAvg = summaryCount > 0 ? asNumber(entry?.misses_avg) : null;
    const status = dominantStatus(entry, avgPoints, checkMax);
    return { entry, avgPoints, makesAvg, missesAvg, status };
  }

  function hexToRgba(hex, alpha) {
    const value = String(hex || "").replace("#", "");
    if (value.length !== 6) return `rgba(255,255,255,${alpha})`;
    const r = parseInt(value.slice(0, 2), 16);
    const g = parseInt(value.slice(2, 4), 16);
    const b = parseInt(value.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function radarColorForScore(score) {
    const numeric = asNumber(score);
    if (numeric === null) return "#7c91a6";
    return "#FF9800";
  }

  function polarPoint(cx, cy, radius, angle) {
    return {
      x: cx + (Math.cos(angle) * radius),
      y: cy + (Math.sin(angle) * radius),
    };
  }

  function labelText(label) {
    if (typeof label === "string") return label;
    if (label && typeof label === "object") {
      if (typeof label.name === "string") return label.name;
      if (typeof label.label === "string") return label.label;
      if (typeof label.text === "string") return label.text;
    }
    return String(label || "");
  }

  function drawRadarPolygon(ctx, points, options = {}) {
    if (!points.length) return;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
    if (options.fillStyle) {
      ctx.fillStyle = options.fillStyle;
      ctx.fill();
    }
    if (options.strokeStyle) {
      ctx.strokeStyle = options.strokeStyle;
      ctx.lineWidth = options.lineWidth || 1;
      if (options.dash) ctx.setLineDash(options.dash);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawRadarChart(canvas, axes, sessionScore) {
    const dpr = window.devicePixelRatio || 1;
    const box = canvas.getBoundingClientRect();
    const size = Math.max(220, Math.floor(Math.min(box.width || 320, 340)));
    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const cx = size / 2;
    const cy = size / 2;
    const radius = size * 0.31;
    const labelRadius = (radius * 1.28) + 20;
    const n = axes.length;
    const start = -Math.PI / 2;
    const angleStep = (Math.PI * 2) / n;

    const axisMeta = axes.map((axis, index) => {
      const angle = start + (index * angleStep);
      const outer = polarPoint(cx, cy, radius, angle);
      const rawLabelPos = polarPoint(cx, cy, labelRadius, angle);
      const labelPos = {
        x: clamp(rawLabelPos.x, 52, size - 52),
        y: clamp(rawLabelPos.y, 26, size - 26),
      };
      const ratio = axis.avgPoints === null ? 0 : clamp(axis.avgPoints / axis.max, 0, 1);
      const grade = gradeFromRatio(ratio);
      return { ...axis, angle, outer, labelPos, ratio, grade };
    });

    for (const scale of [0.25, 0.5, 0.75]) {
      const pts = axisMeta.map((axis) => polarPoint(cx, cy, radius * scale, axis.angle));
      drawRadarPolygon(ctx, pts, { strokeStyle: "#333333", lineWidth: 1 });
    }

    axisMeta.forEach((axis) => {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(axis.outer.x, axis.outer.y);
      ctx.strokeStyle = "#2f2f2f";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    });

    drawRadarPolygon(
      ctx,
      axisMeta.map((axis) => axis.outer),
      { strokeStyle: "#666666", lineWidth: 1.2, dash: [5, 4] },
    );

    const scoreColor = radarColorForScore(sessionScore);
    const actualPoints = axisMeta.map((axis) => {
      return polarPoint(cx, cy, radius * axis.ratio, axis.angle);
    });
    drawRadarPolygon(ctx, actualPoints, {
      fillStyle: hexToRgba(scoreColor, 0.2),
      strokeStyle: scoreColor,
      lineWidth: 2,
    });

    const canDrawOverlays = axisMeta.every((axis) => (
      isNumber(axis.makesAvg) && axis.makesAvg > 0 && isNumber(axis.missesAvg) && axis.missesAvg > 0
    ));
    if (canDrawOverlays) {
      const makesPts = axisMeta.map((axis) => {
        const ratio = clamp(axis.makesAvg / axis.max, 0, 1);
        return polarPoint(cx, cy, radius * ratio, axis.angle);
      });
      drawRadarPolygon(ctx, makesPts, { strokeStyle: "#4CAF50", lineWidth: 1.6, dash: [6, 4] });
      const missesPts = axisMeta.map((axis) => {
        const ratio = clamp(axis.missesAvg / axis.max, 0, 1);
        return polarPoint(cx, cy, radius * ratio, axis.angle);
      });
      drawRadarPolygon(ctx, missesPts, { strokeStyle: "#F44336", lineWidth: 1.6, dash: [6, 4] });
    }

    axisMeta.forEach((axis, idx) => {
      const pt = actualPoints[idx];
      const color = STATUS_COLORS[axis.status] || STATUS_COLORS.unavailable;
      ctx.save();
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4.2, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = "#101010";
      ctx.stroke();
      ctx.restore();
    });

    return { axisMeta, center: { x: cx, y: cy }, radius, overlaysVisible: canDrawOverlays };
  }

  function buildRadarSummary(axes) {
    const rated = axes
      .filter((axis) => axis.avgPoints !== null)
      .map((axis) => ({ ...axis, ratio: clamp(axis.avgPoints / axis.max, 0, 1), labelText: labelText(axis.label) }));
    if (rated.length < 2) {
      return "Collect more form-tracked shots to compare strengths and weaknesses.";
    }
    rated.sort((a, b) => b.ratio - a.ratio);
    const strongest = rated[0];
    const weakest = rated[rated.length - 1];
    return `Your form is strongest in ${strongest.labelText} and weakest in ${weakest.labelText}.`;
  }

  function renderCheckBreakdown() {
    const coaching = state.sessionSummary.session_coaching || {};
    const sessionScore = asNumber(coaching.session_shotsync_score) ?? averageScore(state.shots);
    const axes = RADAR_AXIS_ORDER.map((axis) => {
      const agg = getCheckAggregate(axis.key, axis.max);
      return {
        ...axis,
        ...agg,
      };
    });

    el.checkBreakdown.innerHTML = `
      <div class="radar-block">
        <div class="radar-title">YOUR FORM PROFILE</div>
        <div class="radar-canvas-wrap">
          <canvas id="formRadarCanvas" class="form-radar-canvas" width="320" height="320" aria-label="Form checks radar chart"></canvas>
          <div id="formRadarLabels" class="radar-label-layer"></div>
        </div>
        <div id="radarLegend" class="radar-legend"></div>
        <div class="radar-footer">
          <div class="radar-score-line">Session Form: <strong>${formatFormGrade(sessionScore).replace("Form: ", "")}</strong></div>
          <div class="radar-summary-line">${buildRadarSummary(axes)}</div>
        </div>
      </div>
      <div class="tl-expanded" id="tlExpanded" hidden></div>
    `;

    const canvas = document.getElementById("formRadarCanvas");
    const labelsLayer = document.getElementById("formRadarLabels");
    if (!canvas || !labelsLayer) return;

    const legend = document.getElementById("radarLegend");
    const radar = drawRadarChart(canvas, axes, sessionScore);
    if (!radar) return;
    if (legend) {
      legend.innerHTML = `
        <div class="radar-legend-row">
          <span class="radar-legend-item"><span class="legend-line solid session"></span> Session Average</span>
        </div>
        <div class="radar-legend-row">
          ${radar.overlaysVisible
            ? `<span class="radar-legend-item"><span class="legend-line dashed makes"></span> Makes</span>
               <span class="radar-legend-item"><span class="legend-line dashed misses"></span> Misses</span>`
            : `<span class="radar-legend-item muted">Makes/Misses overlays unavailable for this session.</span>`
          }
        </div>
      `;
    }

    const axisLabelHtml = radar.axisMeta.map((axis) => {
      const axisLabel = labelText(axis.label);
      return (
      `<button type="button" class="radar-axis-label" data-check-key="${axis.key}" style="left:${axis.labelPos.x}px;top:${axis.labelPos.y}px;" title="${axisLabel}">
        ${axisLabel}
      </button>`
      );
    }).join("");
    const axisGradeHtml = radar.axisMeta.map((axis) => {
      return `<button type="button" class="radar-axis-grade" data-check-key="${axis.key}" style="left:${axis.outer.x}px;top:${axis.outer.y}px;background:${axis.grade.color};" title="${labelText(axis.label)}: ${axis.grade.letter}">
        ${axis.grade.letter}
      </button>`;
    }).join("");
    labelsLayer.innerHTML = `${axisLabelHtml}${axisGradeHtml}`;

    labelsLayer.querySelectorAll(".radar-axis-label").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        toggleTrafficLightDetail(btn.getAttribute("data-check-key"));
      });
    });
    labelsLayer.querySelectorAll(".radar-axis-grade").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        toggleTrafficLightDetail(btn.getAttribute("data-check-key"));
      });
    });

    canvas.addEventListener("click", (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const dx = x - radar.center.x;
      const dy = y - radar.center.y;
      const dist = Math.sqrt((dx * dx) + (dy * dy));
      if (dist < 18 || dist > radar.radius * 1.35) return;
      let best = null;
      radar.axisMeta.forEach((axis) => {
        const ddx = x - axis.outer.x;
        const ddy = y - axis.outer.y;
        const d2 = (ddx * ddx) + (ddy * ddy);
        if (!best || d2 < best.d2) {
          best = { d2, key: axis.key };
        }
      });
      if (best && best.d2 <= 2800) {
        toggleTrafficLightDetail(best.key);
      }
    });
  }

  function toggleTrafficLightDetail(checkKey) {
    const expandedEl = document.getElementById("tlExpanded");
    if (!expandedEl) return;

    if (state.expandedTrafficLight === checkKey) {
      expandedEl.hidden = true;
      state.expandedTrafficLight = null;
      return;
    }

    state.expandedTrafficLight = checkKey;
    const check = CHECK_CONFIG.find((c) => c.key === checkKey);
    if (!check) return;

    const coaching = state.sessionSummary.session_coaching || {};
    const summary = coaching.per_check_summary || {};
    const entry = summary?.[checkKey] || null;
    const summaryCount = (asNumber(entry?.good) ?? 0)
      + (asNumber(entry?.needs_work) ?? 0)
      + (asNumber(entry?.poor) ?? 0)
      + (asNumber(entry?.unavailable) ?? 0);
    const sampleCount = checkSampleCountFromShots(checkKey);
    const avgPoints = sampleCount > 0
      ? avgPointsFromShots(checkKey)
      : (summaryCount > 0 ? asNumber(entry?.avg_points) : null);
    const makesAvg = summaryCount > 0 ? asNumber(entry?.makes_avg) : null;
    const missesAvg = summaryCount > 0 ? asNumber(entry?.misses_avg) : null;
    const status = dominantStatus(entry, avgPoints, check.max);

    let feedbackText = "";
    for (const shot of state.shots) {
      const shotCheck = shot?.coaching?.[checkKey];
      if (shotCheck?.feedback) {
        feedbackText = shotCheck.feedback;
        break;
      }
    }

    expandedEl.innerHTML = `
      <div class="tl-detail-card">
        <div class="tl-detail-head">
          <span class="tl-detail-name">${check.label}</span>
          <span class="badge ${statusClass(status)}">${statusLabel(status)}</span>
        </div>
        <div class="tl-detail-score">${formatScorePrecise(avgPoints)}/${check.max}</div>
        <div class="tl-detail-compare">Makes avg ${formatScorePrecise(makesAvg)} \u2022 Misses avg ${formatScorePrecise(missesAvg)}</div>
        ${feedbackText ? `<p class="tl-detail-feedback">${feedbackText}</p>` : ""}
      </div>
    `;
    expandedEl.hidden = false;
  }

  function fallbackCoordByZone(zone, shotNum) {
    const normalized = String(zone || "unknown").toLowerCase();
    const seed = Math.sin((shotNum || 1) * 12.9898) * 43758.5453;
    const jitter = (seed - Math.floor(seed)) * 0.12 - 0.06;
    if (normalized === "left") return { x: clamp(0.24 + jitter, 0.08, 0.42), y: 0.72 };
    if (normalized === "right") return { x: clamp(0.76 + jitter, 0.58, 0.92), y: 0.72 };
    return { x: clamp(0.5 + jitter, 0.35, 0.65), y: 0.72 };
  }

  function normalizeZoneKey(zone) {
    const key = String(zone || "").trim().toLowerCase();
    if (key === "left" || key === "center" || key === "right" || key === "unknown") return key;
    return "unknown";
  }

  function computeZoneStats() {
    const grouped = {};
    for (const shot of state.shots) {
      const zone = shot.zone || "unknown";
      if (!grouped[zone]) {
        grouped[zone] = { attempts: 0, makes: 0, total: 0, count: 0 };
      }
      grouped[zone].attempts += 1;
      if (shot.outcome === "make") grouped[zone].makes += 1;
      if (isNumber(shot.shotsync_score)) {
        grouped[zone].total += shot.shotsync_score;
        grouped[zone].count += 1;
      }
    }

    const stats = {};
    for (const [zone, value] of Object.entries(grouped)) {
      stats[zone] = {
        attempts: value.attempts,
        makes: value.makes,
        avg_shotsync_score: value.count > 0 ? Number((value.total / value.count).toFixed(1)) : null,
      };
    }
    state.sessionSummary = { ...state.sessionSummary, zone_stats: stats };
    return stats;
  }

  function renderShotChart() {
    el.shotDots.innerHTML = "";

    for (const shot of state.shots) {
      const fallback = fallbackCoordByZone(shot.zone, shot.shot_num);
      const nx = shot.shooter_x !== null ? clamp(shot.shooter_x, 0, 1) : fallback.x;
      const ny = shot.shooter_y !== null ? clamp(shot.shooter_y, 0, 1) : fallback.y;

      const x = 8 + (84 * nx);
      const y = 8 + (104 * ny);
      const score = shot.shotsync_score;
      const radius = 2.6 + (isNumber(score) ? (score / 100) * 3.2 : 1.4);
      const cls = shot.outcome === "make" ? "dot-make" : (shot.outcome === "miss" ? "dot-miss" : "dot-unknown");

      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", x.toFixed(2));
      dot.setAttribute("cy", y.toFixed(2));
      dot.setAttribute("r", radius.toFixed(2));
      dot.setAttribute("class", cls);

      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      const tier = scoreTier(shot.shotsync_score);
      const scoreText = isNumber(tier.numeric) ? `${tier.letter} (${Math.round(tier.numeric)})` : "-";
      title.textContent = `Shot ${shot.shot_num}: ${shot.outcome} | ${shot.zone} | Form ${scoreText}`;
      dot.appendChild(title);
      el.shotDots.appendChild(dot);
    }

    const zoneStats = computeZoneStats();
    const visibleZones = ["left", "center", "right"];
    const statsHtml = [];
    for (const zone of visibleZones) {
      const stat = zoneStats?.[zone];
      const attempts = asNumber(stat?.attempts) ?? 0;
      const makes = asNumber(stat?.makes) ?? 0;
      const pct = attempts > 0 ? (100 * makes) / attempts : 0;
      const avgTier = scoreTier(stat?.avg_shotsync_score);
      const avgText = attempts > 0 && isNumber(avgTier.numeric) ? `${avgTier.letter} (${formatScorePrecise(avgTier.numeric)})` : "-";
      statsHtml.push(
        `<div class="zone-stat-chip" data-zone="${zone}"><strong>${zone.toUpperCase()}</strong> ${makes}/${attempts} (${pct.toFixed(1)}%) avg ${avgText}</div>`
      );
    }
    const unknownStat = zoneStats?.unknown;
    if ((asNumber(unknownStat?.attempts) ?? 0) > 0) {
      const attempts = asNumber(unknownStat?.attempts) ?? 0;
      const makes = asNumber(unknownStat?.makes) ?? 0;
      const pct = attempts > 0 ? (100 * makes) / attempts : 0;
      const avgTier = scoreTier(unknownStat?.avg_shotsync_score);
      const avgText = isNumber(avgTier.numeric) ? `${avgTier.letter} (${formatScorePrecise(avgTier.numeric)})` : "-";
      statsHtml.push(
        `<div class="zone-stat-chip zone-extra" data-zone="unknown"><strong>UNKNOWN</strong> ${makes}/${attempts} (${pct.toFixed(1)}%) avg ${avgText}</div>`
      );
    }
    el.zoneStats.innerHTML = statsHtml.length
      ? statsHtml.join("")
      : '<div class="empty-state">No zone stats available.</div>';

    const zoneBreakdownRaw = state.sessionSummary?.session_coaching?.zone_breakdown || {};
    const zoneBreakdown = {};
    Object.entries(zoneBreakdownRaw).forEach(([zone, text]) => {
      zoneBreakdown[normalizeZoneKey(zone)] = String(text || "");
    });

    const tipHtml = visibleZones.map((zone) => {
      const tip = String(zoneBreakdown?.[zone] || "").trim();
      const content = tip || "No zone-specific tip yet.";
      return `<div class="zone-tip-chip ${tip ? "" : "empty"}" data-zone="${zone}"><strong>${zone.toUpperCase()}:</strong> ${escapeHtml(content)}</div>`;
    });
    const unknownTip = String(zoneBreakdown?.unknown || "").trim();
    if (unknownTip) {
      tipHtml.push(`<div class="zone-tip-chip zone-extra" data-zone="unknown"><strong>UNKNOWN:</strong> ${escapeHtml(unknownTip)}</div>`);
    }

    el.zoneTips.innerHTML = tipHtml.length
      ? tipHtml.join("")
      : '<div class="empty-state">No zone-specific coaching tips yet.</div>';
  }

  function sortedShots() {
    const shots = [...state.shots];
    const nullScore = -9999;

    switch (state.sortBy) {
      case "score_desc":
        shots.sort((a, b) => (b.shotsync_score ?? nullScore) - (a.shotsync_score ?? nullScore));
        break;
      case "score_asc":
        shots.sort((a, b) => (a.shotsync_score ?? 9999) - (b.shotsync_score ?? 9999));
        break;
      case "outcome":
        shots.sort((a, b) => {
          const av = a.outcome === "make" ? 0 : 1;
          const bv = b.outcome === "make" ? 0 : 1;
          if (av !== bv) return av - bv;
          return a.shot_num - b.shot_num;
        });
        break;
      case "shot_order":
      default:
        shots.sort((a, b) => a.shot_num - b.shot_num);
        break;
    }
    return shots;
  }

  /* ─── Change 3: Skeleton Animation ─── */

  function getCheckStatusForShot(shot, checkKey) {
    const check = shot?.coaching?.[checkKey];
    if (!check || typeof check !== "object") return "unavailable";
    return statusClass(check.status);
  }

  function getSegmentColor(jointA, jointB, shot) {
    const segKey = `${jointA}-${jointB}`;
    const segKeyReverse = `${jointB}-${jointA}`;
    const checkKey = SEGMENT_CHECK_MAP[segKey] || SEGMENT_CHECK_MAP[segKeyReverse];
    if (!checkKey) return "rgba(255,255,255,0.5)";
    const status = getCheckStatusForShot(shot, checkKey);
    return STATUS_COLORS[status] || "rgba(255,255,255,0.5)";
  }

  function getSegmentWidth(jointA, jointB) {
    const segKey = `${jointA}-${jointB}`;
    const segKeyReverse = `${jointB}-${jointA}`;
    const checkKey = SEGMENT_CHECK_MAP[segKey] || SEGMENT_CHECK_MAP[segKeyReverse];
    return checkKey ? 3 : 2;
  }

  const PHASE_DEFS = [
    { key: "setup", label: "SETUP", checks: ["base_and_balance"] },
    { key: "release", label: "RELEASE", checks: ["elbow_alignment", "release_height", "shoulder_alignment"] },
    { key: "follow_through", label: "FOLLOW-THROUGH", checks: ["follow_through", "guide_hand"] },
  ];

  function statusRank(status) {
    const normalized = statusClass(status);
    if (normalized === "poor") return 3;
    if (normalized === "needs_work") return 2;
    if (normalized === "good") return 1;
    return 0;
  }

  function worstStatus(statuses) {
    if (!statuses.length) return "needs_work";
    let worst = "unavailable";
    let rank = -1;
    statuses.forEach((status) => {
      const current = statusRank(status);
      if (current > rank) {
        rank = current;
        worst = statusClass(status);
      }
    });
    return worst === "unavailable" ? "needs_work" : worst;
  }

  function phaseGradeFromRatio(ratio) {
    if (ratio >= 0.75) return "A";
    if (ratio >= 0.5) return "B";
    return "C";
  }

  function compactTip(text, fallbackLabel, status) {
    const raw = String(text || "").replace(/\s+/g, " ").trim();
    if (raw) {
      const sentence = raw.split(/[.!?]/)[0].trim();
      const tip = sentence || raw;
      return tip.length > 86 ? `${tip.slice(0, 83)}...` : tip;
    }
    if (statusClass(status) === "good") return `${fallbackLabel} looks solid`;
    if (statusClass(status) === "poor") return `${fallbackLabel} needs attention`;
    return `${fallbackLabel} can improve`;
  }

  function readCheckMetric(check, maxPoints) {
    const angle = asNumber(check?.actual_angle ?? check?.angle ?? check?.measured_angle ?? check?.value);
    if (angle !== null) return `Angle ${Math.round(angle)}°`;
    const points = asNumber(check?.points);
    if (points !== null && isNumber(maxPoints)) return `${formatScorePrecise(points)}/${maxPoints} pts`;
    return "No metric";
  }

  function buildShotPhaseData(shot) {
    return PHASE_DEFS.map((phase) => {
      const detailChecks = phase.checks.map((checkKey) => {
        const config = CHECK_CONFIG.find((item) => item.key === checkKey) || { key: checkKey, label: prettyCheckName(checkKey), max: 20 };
        const entry = shot?.coaching?.[checkKey] || {};
        const status = statusClass(entry?.status);
        const points = asNumber(entry?.points);
        const maxPoints = asNumber(config.max) ?? 20;
        const ratio = points !== null
          ? clamp(points / Math.max(1, maxPoints), 0, 1)
          : (status === "good" ? 0.85 : (status === "poor" ? 0.3 : 0.6));
        return {
          key: checkKey,
          label: config.label,
          status,
          feedback: compactTip(entry?.feedback, config.label, status),
          points,
          maxPoints,
          ratio,
          metric: readCheckMetric(entry, maxPoints),
        };
      });
      const statuses = detailChecks.map((item) => item.status);
      const avgRatio = detailChecks.length
        ? detailChecks.reduce((sum, item) => sum + item.ratio, 0) / detailChecks.length
        : 0.6;
      const status = worstStatus(statuses);
      const color = STATUS_COLORS[status] || STATUS_COLORS.needs_work;
      return {
        ...phase,
        checks: detailChecks,
        status,
        color,
        grade: phaseGradeFromRatio(avgRatio),
        avgRatio,
      };
    });
  }

  function checkLetterGrade(entry, checkConfig) {
    const maxPoints = asNumber(checkConfig?.max) ?? 20;
    const points = asNumber(entry?.points);
    if (isNumber(points) && maxPoints > 0) {
      return scoreTier((points / maxPoints) * 100);
    }
    const status = statusClass(entry?.status);
    if (status === "good") return scoreTier(82);
    if (status === "needs_work") return scoreTier(52);
    if (status === "poor") return scoreTier(18);
    return scoreTier(null);
  }

  function shotCheckRatio(entry, maxPoints) {
    const points = asNumber(entry?.points);
    if (points !== null && maxPoints > 0) {
      return clamp(points / maxPoints, 0, 1);
    }
    const status = statusClass(entry?.status);
    if (status === "good") return 0.82;
    if (status === "needs_work") return 0.58;
    if (status === "poor") return 0.28;
    return 0.45;
  }

  function playbookForCheck(checkKey, label, fallbackTip) {
    const playbook = CHECK_PLAYBOOK[checkKey];
    if (playbook) return playbook;
    return {
      why: `${label} directly impacts your shot consistency and accuracy.`,
      try: fallbackTip || `Focus on ${label.toLowerCase()} in your next 10 shots.`,
    };
  }

  function measuredTextForGrade(checkKey, gradeLetter) {
    const map = CHECK_GRADE_MEASURED[checkKey];
    if (!map) return "This check needs more samples before we can grade it confidently.";
    const letter = String(gradeLetter || "").toUpperCase();
    if (letter === "-" || letter === "") {
      return "This check did not have enough tracked data on this shot.";
    }
    if (letter === "A") return map.A;
    if (letter === "B") return map.B;
    if (letter === "C") return map.C;
    return map.DF;
  }

  function comparisonFocusCue(checkKey) {
    if (checkKey === "base_and_balance") return "Focus on feeling your legs engage before every shot.";
    if (checkKey === "follow_through") return "Focus on holding your arm up after release on every rep.";
    if (checkKey === "release_height") return "Focus on getting the ball above your forehead before release.";
    if (checkKey === "elbow_alignment") return "Focus on keeping your elbow stacked under the ball.";
    if (checkKey === "guide_hand") return "Focus on keeping your guide hand quiet at release.";
    if (checkKey === "shoulder_alignment") return "Focus on keeping your shoulders level and square before each shot.";
    return "Focus on repeating your strongest mechanics each rep.";
  }

  function drawMiniShotRadar(canvas, shot) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const box = canvas.getBoundingClientRect();
    const size = Math.max(130, Math.floor(Math.min(box.width || 150, 170)));
    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const cx = size / 2;
    const cy = size / 2;
    const radius = size * 0.30;
    const n = RADAR_AXIS_ORDER.length;
    const start = -Math.PI / 2;
    const step = (Math.PI * 2) / n;
    const score = scoreTier(asNumber(shot?.shotsync_score));
    const axes = RADAR_AXIS_ORDER.map((axis, index) => {
      const angle = start + (index * step);
      const entry = shot?.coaching?.[axis.key] || {};
      const ratio = shotCheckRatio(entry, axis.max);
      const status = statusClass(entry?.status);
      return {
        key: axis.key,
        angle,
        ratio,
        status,
        outer: polarPoint(cx, cy, radius, angle),
        point: polarPoint(cx, cy, radius * ratio, angle),
      };
    });

    [0.25, 0.5, 0.75].forEach((scale) => {
      const grid = axes.map((axis) => polarPoint(cx, cy, radius * scale, axis.angle));
      drawRadarPolygon(ctx, grid, { strokeStyle: "#2f2f2f", lineWidth: 1 });
    });

    drawRadarPolygon(
      ctx,
      axes.map((axis) => axis.outer),
      { strokeStyle: "rgba(180,180,180,0.45)", lineWidth: 1, dash: [4, 3] },
    );

    ctx.save();
    ctx.fillStyle = "#8a8a8a";
    ctx.font = '10px "Work Sans", sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    axes.forEach((axis) => {
      const raw = polarPoint(cx, cy, radius * 1.42, axis.angle);
      const lx = clamp(raw.x, 18, size - 18);
      const ly = clamp(raw.y, 10, size - 10);
      const label = MINI_RADAR_LABELS[axis.key] || labelText(axis.key);
      ctx.fillText(label, lx, ly);
    });
    ctx.restore();

    const actual = axes.map((axis) => axis.point);
    drawRadarPolygon(ctx, actual, {
      fillStyle: hexToRgba(score.color, 0.22),
      strokeStyle: score.color,
      lineWidth: 1.8,
    });

    actual.forEach((pt, index) => {
      const dotColor = STATUS_COLORS[axes[index].status] || STATUS_COLORS.unavailable;
      ctx.save();
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = dotColor;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#0f0f0f";
      ctx.stroke();
      ctx.restore();
    });
  }

  function compactFeedbackRows(shot) {
    const coaching = (shot?.coaching && typeof shot.coaching === "object") ? shot.coaching : {};
    const rows = CHECK_CONFIG.map((check) => {
      const entry = coaching?.[check.key] || {};
      const status = statusClass(entry?.status);
      const statusOrder = (status === "good") ? 0 : (status === "needs_work" ? 1 : (status === "poor" ? 2 : 3));
      const grade = checkLetterGrade(entry, check);
      const tip = compactTip(entry?.feedback, check.label, status);
      const playbook = playbookForCheck(check.key, check.label, tip);
      return {
        key: check.key,
        label: check.label,
        status,
        statusOrder,
        tip,
        grade,
        measured: measuredTextForGrade(check.key, grade.letter),
        why: playbook.why,
        drill: playbook.try,
      };
    });
    rows.sort((a, b) => a.statusOrder - b.statusOrder || a.label.localeCompare(b.label));
    return rows;
  }

  function createAnnotatedStills(shot, container) {
    const feedbackRows = compactFeedbackRows(shot);
    const sources = sourceCandidatesForShot(shot);
    const hasClip = sources.length > 0;
    const feedbackHtml = feedbackRows.map((row) => (
      `<details class="shot-feedback-item ${row.status}" data-check-key="${escapeHtml(row.key)}">
        <summary class="shot-feedback-line ${row.status}">
          <span class="shot-feedback-icon ${row.status}" aria-hidden="true"></span>
          <span class="shot-feedback-main">${escapeHtml(row.label)} (${escapeHtml(row.grade.letter)}) - ${escapeHtml(row.tip)}</span>
          <span class="shot-feedback-meta">
            <span class="shot-feedback-grade ${row.status}" aria-label="Grade ${escapeHtml(row.grade.letter)}">${escapeHtml(row.grade.letter)}</span>
            <span class="shot-feedback-chevron" aria-hidden="true">▸</span>
          </span>
        </summary>
        <div class="shot-feedback-expand-wrap">
          <div class="shot-feedback-expand">
            <div class="shot-feedback-expand-title">WHAT WE MEASURED</div>
            <div class="shot-feedback-expand-text">${escapeHtml(row.measured)}</div>
            <div class="shot-feedback-expand-title">WHY IT MATTERS</div>
            <div class="shot-feedback-expand-text">${escapeHtml(row.why)}</div>
            <div class="shot-feedback-expand-title">TRY THIS</div>
            <div class="shot-feedback-expand-text">${escapeHtml(row.drill)}</div>
          </div>
        </div>
      </details>`
    )).join("");

    container.innerHTML = `
      <div class="shot-detail-compact">
        <div class="shot-detail-top">
          <div class="shot-mini-radar-wrap" aria-hidden="true">
            <canvas class="shot-mini-radar-canvas" width="150" height="150"></canvas>
          </div>
          <div class="shot-detail-top-actions">
            <button type="button" class="watch-shot-btn" ${hasClip ? "" : "disabled"}>
              ▶ WATCH YOUR SHOT
            </button>
            ${hasClip ? "" : `<div class="stills-empty">${escapeHtml(String(shot?.stills_message || "Clip not available for this shot."))}</div>`}
          </div>
        </div>
        <div class="shot-inline-video-wrap" hidden>
          <video class="shot-inline-video" controls muted playsinline preload="metadata"></video>
        </div>
        <div class="shot-feedback-lines">
          ${feedbackHtml}
        </div>
      </div>
    `;

    const radarCanvas = container.querySelector(".shot-mini-radar-canvas");
    drawMiniShotRadar(radarCanvas, shot);

    const button = container.querySelector(".watch-shot-btn");
    const wrap = container.querySelector(".shot-inline-video-wrap");
    const video = container.querySelector(".shot-inline-video");
    if (!button || !wrap || !video) {
      return { destroy() { container.innerHTML = ""; } };
    }

    let controller = null;
    let unavailable = !hasClip;
    let open = false;
    if (hasClip) {
      controller = attachRangedClipController(video, shot, {
        onUnavailable: () => {
          unavailable = true;
          button.disabled = true;
          wrap.hidden = true;
          button.textContent = "CLIP UNAVAILABLE";
        },
      });
      if (!controller) {
        unavailable = true;
      }
    }

    const onWatchClick = (event) => {
      event.preventDefault();
      if (unavailable || !controller) return;
      if (!open) {
        wrap.hidden = false;
        controller.playFromStart();
        button.textContent = "❚❚ HIDE SHOT";
        open = true;
        return;
      }
      controller.pause();
      wrap.hidden = true;
      button.textContent = "▶ WATCH YOUR SHOT";
      open = false;
    };
    button.addEventListener("click", onWatchClick);

    return {
      destroy() {
        button.removeEventListener("click", onWatchClick);
        controller?.destroy();
        container.innerHTML = "";
      },
    };
  }

  function buildMetricBars(vectors) {
    const wrap = document.createElement("div");
    wrap.className = "metric-wrap";
    if (!vectors || !vectors.length) return wrap;

    for (const vec of vectors) {
      if (vec.actual_angle == null || !vec.ideal_range) continue;

      const row = document.createElement("div");
      row.className = "metric-row";

      const lbl = document.createElement("span");
      lbl.className = "metric-label";
      lbl.textContent = vec.check_label;
      row.appendChild(lbl);

      const bar = document.createElement("div");
      bar.className = "metric-bar";

      const scaleMin = 0;
      const scaleMax = 180;
      const idealLow = vec.ideal_range[0];
      const idealHigh = vec.ideal_range[1];
      const actual = vec.actual_angle;

      const pctLow = ((idealLow - scaleMin) / (scaleMax - scaleMin)) * 100;
      const pctHigh = ((idealHigh - scaleMin) / (scaleMax - scaleMin)) * 100;
      const pctActual = Math.min(100, Math.max(0, ((actual - scaleMin) / (scaleMax - scaleMin)) * 100));

      const idealZone = document.createElement("div");
      idealZone.className = "metric-ideal";
      idealZone.style.left = pctLow + "%";
      idealZone.style.width = (pctHigh - pctLow) + "%";
      bar.appendChild(idealZone);

      const marker = document.createElement("div");
      marker.className = "metric-marker";
      marker.style.left = pctActual + "%";
      const color = STATUS_COLORS[vec.status] || STATUS_COLORS.needs_work;
      marker.style.backgroundColor = color;
      bar.appendChild(marker);

      row.appendChild(bar);

      const val = document.createElement("span");
      val.className = "metric-val";
      val.textContent = Math.round(actual) + "\u00b0";
      row.appendChild(val);

      wrap.appendChild(row);
    }

    return wrap;
  }

  function openStillLightbox(still) {
    const overlay = document.createElement("div");
    overlay.className = "still-lightbox";

    const inner = document.createElement("div");
    inner.className = "still-lightbox-inner";

    const closeBtn = document.createElement("button");
    closeBtn.className = "still-lightbox-close";
    closeBtn.textContent = "\u00D7";
    closeBtn.addEventListener("click", (e) => { e.stopPropagation(); overlay.remove(); });

    const label = document.createElement("div");
    label.className = "still-lightbox-label";
    label.textContent = still.label;

    const img = document.createElement("img");
    img.src = still.frame_url;
    img.alt = `${still.label} frame`;

    inner.appendChild(closeBtn);
    inner.appendChild(label);
    inner.appendChild(img);
    inner.appendChild(buildMetricBars(still.vectors));
    overlay.appendChild(inner);

    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });

    document.addEventListener("keydown", function escHandler(e) {
      if (e.key === "Escape") {
        overlay.remove();
        document.removeEventListener("keydown", escHandler);
      }
    });

    document.body.appendChild(overlay);
  }

  /* ─── Shot card rendering with all changes ─── */

  function renderShotMiniTrafficLights(shot) {
    const phaseData = buildShotPhaseData(shot);
    return phaseData.map((phase) => {
      return `<span class="mini-phase-dot ${phase.status}" style="background:${phase.color};" title="${phase.label}: ${phase.grade} (${statusLabel(phase.status)})"></span>`;
    }).join("");
  }

  function getWorstCheckFeedback(shot) {
    const priority = ["poor", "needs_work", "good"];
    for (const targetStatus of priority) {
      for (const check of CHECK_CONFIG) {
        const shotCheck = shot?.coaching?.[check.key];
        if (!shotCheck) continue;
        if (statusClass(shotCheck.status) === targetStatus && shotCheck.feedback) {
          return shotCheck.feedback;
        }
      }
    }
    return null;
  }

  function renderShotCheck(shot, config) {
    const check = shot?.coaching?.[config.key] || {};
    const status = statusClass(check.status);
    const color = STATUS_COLORS[status] || STATUS_COLORS.unavailable;

    return `
      <div class="shot-check-mini">
        <span class="mini-tl" style="background:${color};"></span>
        <span class="shot-check-mini-label">${config.label}: ${statusLabel(status)}</span>
      </div>
    `;
  }

  function renderCoachingHighlights(shot) {
    const coaching = shot?.coaching;
    if (!coaching || typeof coaching !== "object") {
      return '<div class="coaching-highlight muted">Form data not available for this shot.</div>';
    }

    const issues = [];
    const strengths = [];
    for (const check of CHECK_CONFIG) {
      const entry = coaching[check.key];
      if (!entry || typeof entry !== "object") continue;
      const s = statusClass(entry.status);
      if (s === "poor" || s === "needs_work") {
        issues.push({ ...check, entry, statusCls: s });
      } else if (s === "good") {
        strengths.push({ ...check, entry, statusCls: s });
      }
    }

    if (!issues.length && !strengths.length) {
      return '<div class="coaching-highlight muted">Form data not available for this shot.</div>';
    }

    const cards = [];
    for (const item of issues.slice(0, 2)) {
      const fb = item.entry.feedback || `${item.label} needs improvement.`;
      cards.push(`<div class="coaching-highlight ${item.statusCls}"><span class="ch-name">${item.label}</span><span class="ch-feedback">${fb}</span></div>`);
    }
    for (const item of strengths.slice(0, 2)) {
      const fb = item.entry.feedback || `${item.label} looks good.`;
      cards.push(`<div class="coaching-highlight good"><span class="ch-name">${item.label}</span><span class="ch-feedback">${fb}</span></div>`);
    }
    return cards.join("");
  }

  function pushCorrection(shot, originalOutcome, correctedOutcome) {
    state.corrections.push({
      shot_num: shot.shot_num,
      original_outcome: originalOutcome,
      corrected_outcome: correctedOutcome,
      release_frame: asNumber(shot.release_frame),
      video_name: state.videoName || state.payload?.video_name || state.file?.name || "unknown_video",
      timestamp: new Date().toISOString(),
    });
  }

  function clearSaveStatusTimer() {
    if (state.saveStatusTimer) {
      clearTimeout(state.saveStatusTimer);
      state.saveStatusTimer = null;
    }
  }

  function setSaveStatus(text, mode = "neutral") {
    if (!el.saveCorrectionsStatus) return;
    clearSaveStatusTimer();
    el.saveCorrectionsStatus.textContent = text || "";
    el.saveCorrectionsStatus.classList.remove("ok", "error");
    if (mode === "ok") el.saveCorrectionsStatus.classList.add("ok");
    if (mode === "error") el.saveCorrectionsStatus.classList.add("error");
    if (text) {
      state.saveStatusTimer = window.setTimeout(() => {
        if (el.saveCorrectionsStatus) {
          el.saveCorrectionsStatus.textContent = "";
          el.saveCorrectionsStatus.classList.remove("ok", "error");
        }
      }, 2200);
    }
  }

  function renderCorrectionControls() {
    if (!el.correctionsCard || !el.correctionsCount || !el.saveCorrectionsBtn) {
      return;
    }
    const count = state.corrections.length;
    el.correctionsCard.hidden = count === 0;
    if (count === 0) {
      setSaveStatus("");
      return;
    }
    el.correctionsCount.textContent = `${count} manual correction${count === 1 ? "" : "s"} ready to save`;
  }

  function destroyAllSkeletonPlayers() {
    for (const key of Object.keys(state.skeletonPlayers)) {
      state.skeletonPlayers[key]?.destroy();
    }
    state.skeletonPlayers = {};
  }

  function renderAfterShotEdit() {
    refreshSessionTotals();
    renderSessionHeader();
    renderInsights();
    renderCheckBreakdown();
    renderShotList();
    renderMakeVsMissComparison();
    renderSessionSummary();
    renderShotChart();
    renderCorrectionControls();
  }

  function flipShotOutcome(shotNum) {
    const shot = state.shots.find((item) => item.shot_num === shotNum);
    if (!shot) return;
    const originalOutcome = shot.outcome;
    const correctedOutcome = shot.outcome === "make" ? "miss" : "make";
    shot.outcome = correctedOutcome;
    shot.manual_edited = true;
    pushCorrection(shot, originalOutcome, correctedOutcome);
    renderAfterShotEdit();
    const flashed = el.shotList.querySelector(`[data-shot-num="${shotNum}"].outcome-toggle`);
    if (flashed) {
      flashed.classList.add("flash");
      window.setTimeout(() => flashed.classList.remove("flash"), 420);
    }
  }

  function removeShot(shotNum) {
    const index = state.shots.findIndex((item) => item.shot_num === shotNum);
    if (index < 0) return;
    const shot = state.shots[index];
    pushCorrection(shot, shot.outcome, "removed");
    state.shots.splice(index, 1);
    renderAfterShotEdit();
  }

  function attachShotActionHandlers() {
    const outcomeButtons = el.shotList.querySelectorAll(".outcome-toggle");
    outcomeButtons.forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const shotNum = asNumber(button.getAttribute("data-shot-num"));
        if (shotNum === null) return;
        flipShotOutcome(shotNum);
      });
    });

    const removeButtons = el.shotList.querySelectorAll(".remove-shot-btn");
    removeButtons.forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        const shotNum = asNumber(button.getAttribute("data-shot-num"));
        if (shotNum === null) return;
        removeShot(shotNum);
      });
    });

    const removeXButtons = el.shotList.querySelectorAll(".remove-x-btn");
    removeXButtons.forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const shotNum = asNumber(button.getAttribute("data-shot-num"));
        if (shotNum === null) return;
        removeShot(shotNum);
      });
    });

    el.shotList.querySelectorAll("details.shot-card").forEach((details) => {
      details.addEventListener("toggle", () => {
        const shotNum = asNumber(details.getAttribute("data-shot-card"));
        if (shotNum === null) return;

        if (state.skeletonPlayers[shotNum]) {
          state.skeletonPlayers[shotNum].destroy();
          delete state.skeletonPlayers[shotNum];
        }

        if (details.open) {
          const container = details.querySelector(".annotated-stills-mount");
          const shot = state.shots.find((s) => s.shot_num === shotNum);
          if (container && shot) {
            const player = createAnnotatedStills(shot, container);
            if (player) {
              state.skeletonPlayers[shotNum] = player;
            }
          }
        } else {
          if (state.skeletonPlayers[shotNum]) {
            state.skeletonPlayers[shotNum].destroy();
            delete state.skeletonPlayers[shotNum];
          }
        }
      });
    });
  }

  function renderShotList() {
    destroyAllSkeletonPlayers();

    if (!state.shots.length) {
      el.shotList.innerHTML = '<div class="empty-state">No shot data available.</div>';
      return;
    }

    const shots = sortedShots();
    const html = shots.map((shot) => {
      const outcome = shot.outcome === "make" ? "make" : (shot.outcome === "miss" ? "miss" : "unknown");
      const zone = (shot.zone || "unknown").toUpperCase();
      const manualTag = shot.manual_edited ? '<span class="manual-tag">\u270E Manual</span>' : "";
      const miniPhaseHtml = renderShotMiniTrafficLights(shot);
      const scoreHtml = makeGradeSpan(shot.shotsync_score);

      return `
        <details class="shot-card" data-shot-card="${shot.shot_num}">
          <summary>
            <span class="shot-num">Shot ${shot.shot_num}</span>
            <button
              type="button"
              class="outcome-pill outcome-toggle ${outcome}"
              data-shot-num="${shot.shot_num}"
              title="Tap to change make/miss"
              aria-label="Flip shot ${shot.shot_num} outcome"
            >
              ${outcome.toUpperCase()} \u21C4 ${manualTag}
            </button>
            <span class="shot-zone">${zone}</span>
            <span class="shot-phase-mini" aria-label="Shot phase grades">${miniPhaseHtml}</span>
            <span class="shot-score">${scoreHtml}</span>
            <button type="button" class="remove-x-btn" data-shot-num="${shot.shot_num}" title="Remove shot" aria-label="Remove shot ${shot.shot_num}">\u00D7</button>
          </summary>
          <div class="shot-detail shot-detail-layout">
            <div class="annotated-stills-mount"></div>
          </div>
          <div class="shot-actions-bar">
            <button type="button" class="remove-shot-btn" data-shot-num="${shot.shot_num}">REMOVE SHOT</button>
          </div>
        </details>
      `;
    });

    el.shotList.innerHTML = html.join("");
    attachShotActionHandlers();
    showCorrectionTooltip();
  }

  /* ─── Change 4: Make/Miss Correction Tooltip ─── */

  function showCorrectionTooltip() {
    if (localStorage.getItem("shotlab_correction_tooltip_shown") === "1") return;
    if (!state.shots.length) return;

    const firstPill = el.shotList.querySelector(".outcome-toggle");
    if (!firstPill) return;

    const tooltip = document.createElement("div");
    tooltip.className = "correction-tooltip";
    tooltip.textContent = "Tap to correct make/miss";

    firstPill.parentElement.style.position = "relative";
    firstPill.parentElement.appendChild(tooltip);

    const dismiss = () => {
      tooltip.remove();
      localStorage.setItem("shotlab_correction_tooltip_shown", "1");
      document.removeEventListener("click", dismiss);
    };

    window.setTimeout(() => {
      document.addEventListener("click", dismiss);
    }, 100);
  }

  function renderSessionSummary() {
    const coaching = state.sessionSummary.session_coaching || {};
    if (!hasSessionCoachingData(coaching)) {
      el.overallConsistencyText.textContent = "Coaching summary is not available for this session.";
      el.zoneSummaryText.textContent = "Zone coaching tips will appear once coaching metrics are present.";
      return;
    }
    el.overallConsistencyText.textContent = coaching.overall_consistency || "No consistency summary available.";

    const zoneBreakdown = coaching.zone_breakdown || {};
    const zoneText = Object.entries(zoneBreakdown)
      .map(([zone, text]) => `${String(zone).toUpperCase()}: ${String(text)}`)
      .join(" ");

    el.zoneSummaryText.textContent = zoneText || "No zone breakdown text available.";
  }

  function renderDashboard(payload) {
    state.shots = parseShots(payload);
    state.sessionSummary = getSessionSummary(payload, state.shots);
    state.videoName = String(payload?.video_name || payload?.video_path || state.file?.name || "unknown_video");
    state.corrections = [];
    state.expandedTrafficLight = null;

    refreshSessionTotals();
    renderSessionHeader();
    renderInsights();
    renderCheckBreakdown();
    renderShotList();
    renderMakeVsMissComparison();
    renderSessionSummary();
    renderShotChart();
    renderCorrectionControls();
  }

  async function saveCorrections() {
    if (!state.corrections.length) return;
    if (!el.saveCorrectionsBtn) return;

    el.saveCorrectionsBtn.disabled = true;
    setSaveStatus("Saving...", "neutral");

    try {
      const response = await fetch("/api/save_corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_name: state.videoName || state.file?.name || "unknown_video",
          corrections: state.corrections,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload?.success === false) {
        throw new Error(payload?.error || `Save failed (${response.status})`);
      }
      setSaveStatus("Saved \u2713", "ok");
    } catch (error) {
      setSaveStatus(`Save failed: ${error.message}`, "error");
    } finally {
      el.saveCorrectionsBtn.disabled = false;
    }
  }

  async function submitSession(event) {
    event.preventDefault();
    clearUploadError();

    if (!state.file) {
      showUploadError("Please select a video before analyzing.");
      return;
    }

    showScreen("loading");
    el.analyzeBtn.disabled = true;
    startLoadingIndicators();

    const formData = new FormData();
    formData.append("video", state.file);
    formData.append("camera_mode", "behind_basket");

    try {
      const response = await fetch("/api/process_shotlab_session", {
        method: "POST",
        body: formData,
      });

      let payload = null;
      try {
        payload = await response.json();
      } catch (_err) {
        payload = null;
      }

      if (!response.ok || !payload?.success) {
        const message = payload?.error || `Request failed with status ${response.status}`;
        throw new Error(message);
      }

      state.payload = payload;
      renderDashboard(payload);
      showScreen("results");
    } catch (error) {
      showScreen("upload");
      showUploadError(`Analysis failed: ${error.message}`);
    } finally {
      stopLoadingIndicators();
      el.analyzeBtn.disabled = false;
    }
  }

  function resetSession() {
    destroyComparisonController();
    state.payload = null;
    state.videoName = null;
    state.shots = [];
    state.sessionSummary = {};
    state.corrections = [];
    state.sortBy = "shot_order";
    state.progressValue = 0;
    state.progressHighWater = 0;
    state.expandedTrafficLight = null;
    destroyAllSkeletonPlayers();

    if (el.uploadForm) {
      el.uploadForm.reset();
    }
    setSelectedFile(null);
    clearUploadError();
    el.shotSort.value = "shot_order";
    renderCorrectionControls();
    showScreen("upload");
  }

  function bindEvents() {
    if (!el.uploadForm || !el.videoFile || !el.dropZone) {
      console.error("ShotLab: Missing required elements (uploadForm, videoFile, dropZone)");
      return;
    }
    el.uploadForm.addEventListener("submit", (e) => {
      e.preventDefault();
      e.stopPropagation();
      submitSession(e);
    }, true);

    el.videoFile.addEventListener("change", () => {
      /* Defer to next tick: some browsers don't populate files until after change fires */
      setTimeout(() => {
        const file = el.videoFile.files?.[0] || null;
        setSelectedFile(file);
        clearUploadError();
      }, 0);
    });

    /* Click: file input overlays drop zone and receives clicks directly */

    ["dragenter", "dragover"].forEach((eventName) => {
      el.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        el.dropZone.classList.add("dragging");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      el.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        el.dropZone.classList.remove("dragging");
      });
    });

    el.dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      const file = event.dataTransfer?.files?.[0] || null;
      if (!file) return;
      setSelectedFile(file);
      const transfer = new DataTransfer();
      transfer.items.add(file);
      el.videoFile.files = transfer.files;
      clearUploadError();
    });

    if (el.shotSort) {
      el.shotSort.addEventListener("change", () => {
        state.sortBy = el.shotSort.value;
        renderShotList();
      });
    }
    if (el.newSessionBtn) {
      el.newSessionBtn.addEventListener("click", resetSession);
    }
    if (el.saveCorrectionsBtn) {
      el.saveCorrectionsBtn.addEventListener("click", saveCorrections);
    }
  }

  function init() {
    if (typeof window !== "undefined" && window.location.search.includes("video=")) {
      const url = new URL(window.location.href);
      url.searchParams.delete("video");
      window.history.replaceState({}, "", url.toString());
    }
    bindEvents();
    showScreen("upload");
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
