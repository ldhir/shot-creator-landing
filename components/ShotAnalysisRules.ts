// ============================================================================
// SHOT ANALYSIS RULE ENGINE
// Roots Basketball — NBA Player Match & Ideal Form Analysis
// ============================================================================

export type Tier = 'ideal' | 'good' | 'weakness';
export type Direction = 'more' | 'less' | 'higher' | 'lower' | 'shorter' | 'longer';

// ======================== SHARED TYPES ========================

export interface MetricResult {
  metric: string;
  tier: Tier;
  value: number;
  feedback: string;
  delta?: number;
  direction?: Direction;
  isDegraded?: boolean;
}

export interface IdealFormResult {
  strengths: MetricResult[];
  workOn: MetricResult[];
  quickTips: string[];
  consistencyScore: number;
}

// ======================== NBA MATCH TYPES ========================

export interface NBAMatchMetrics {
  trunkLean?: number;
  kneeBend?: number;
  wristSnap?: number;
  releaseHeight?: number;
  footAlignment?: number;
}

export interface QuickFix {
  metric: string;
  title: string;
  description: string;
  reason: 'weakness' | 'degraded' | 'inconsistent';
}

export interface NBAMatchResult {
  sharedTraits: MetricResult[];
  keyDifferences: MetricResult[];
  quickFixes: QuickFix[];
}

export interface PreviousSessionTiers {
  [metric: string]: Tier;
}

// ======================== THRESHOLDS ========================

interface SimpleThreshold {
  kind: 'lower-better' | 'higher-better';
  ideal: number;
  good: number;
}

interface RangeThreshold {
  kind: 'range';
  idealMin: number;
  idealMax: number;
  goodMin: number;
  goodMax: number;
}

type Threshold = SimpleThreshold | RangeThreshold;

const THRESHOLDS: Record<string, Threshold> = {
  elbowFlare:     { kind: 'lower-better', ideal: 3, good: 7 },
  trunkLean:      { kind: 'lower-better', ideal: 3, good: 7 },
  kneeBend:       { kind: 'range', idealMin: 115, idealMax: 125, goodMin: 110, goodMax: 130 },
  elbowExtension: { kind: 'range', idealMin: 90, idealMax: 100, goodMin: 80, goodMax: 110 },
  wristSnap:      { kind: 'higher-better', ideal: 90, good: 75 },
  rhythm:         { kind: 'lower-better', ideal: 0, good: 50 },  // special: negative = reversed
  footAlignment:  { kind: 'lower-better', ideal: 3, good: 7 },
  footStance:     { kind: 'range', idealMin: 90, idealMax: 110, goodMin: 80, goodMax: 120 },
  releaseHeight:  { kind: 'higher-better', ideal: 15, good: 10 },
  jumpHeight:     { kind: 'higher-better', ideal: -5, good: -15 },
};

// ======================== FEEDBACK ========================

const FEEDBACK: Record<string, { weakness: string; weaknessAlt?: string }> = {
  elbowFlare:     { weakness: "Your elbow is drifting out — this causes left/right misses" },
  trunkLean:      { weakness: "You're leaning forward — stay tall through your release" },
  kneeBend:       { weakness: "Your legs aren't loaded — you're losing power",
                    weaknessAlt: "Too deep — losing explosiveness" },
  elbowExtension: { weakness: "Your arm isn't extending fully — shorter range, less arc",
                    weaknessAlt: "Overextending — losing control" },
  wristSnap:      { weakness: "Your wrist isn't snapping through — affects spin and touch" },
  rhythm:         { weakness: "Your timing is off — elbow is firing before your legs" },
  footAlignment:  { weakness: "Your feet aren't squared — you're fading left/right" },
  footStance:     { weakness: "Your base is too narrow — affects balance",
                    weaknessAlt: "Your base is too wide — affects balance" },
  releaseHeight:  { weakness: "You're releasing low — easier to block, flatter shot" },
  jumpHeight:     { weakness: "Your jump height is dropping — could be fatigue" },
};

const QUICK_FIXES: Record<string, { title: string; description: string }> = {
  elbowFlare:     { title: "Straighten your elbow path",  description: "Your elbow is drifting out — this causes left/right misses" },
  trunkLean:      { title: "Stay more upright",           description: "You're leaning forward — stay tall through your release" },
  kneeBend:       { title: "Deepen your knee bend",       description: "Your legs aren't loaded — you're losing power" },
  elbowExtension: { title: "Extend your arm fully",       description: "Your arm isn't extending fully — shorter range, less arc" },
  wristSnap:      { title: "Snap your wrist fully",       description: "Your wrist isn't snapping through — affects spin and touch" },
  rhythm:         { title: "Sync your timing",             description: "Your timing is off — elbow is firing before your legs" },
  footAlignment:  { title: "Square your feet",             description: "Your feet aren't squared — you're fading left/right" },
  footStance:     { title: "Fix your base width",          description: "Your base is too narrow/wide — affects balance" },
  releaseHeight:  { title: "Get your release higher",      description: "You're releasing low — easier to block, flatter shot" },
  jumpHeight:     { title: "Load your legs",               description: "Your jump height is inconsistent — focus on a consistent leg load each rep" },
};

// ======================== CLASSIFICATION ========================

/**
 * Classify a single metric. Returns tier + appropriate feedback.
 *
 * Handles three shapes:
 *  - lower-better  (elbowFlare, trunkLean, footAlignment, rhythm)
 *  - higher-better  (wristSnap, releaseHeight, jumpHeight)
 *  - range / bidirectional (kneeBend, elbowExtension, footStance)
 *
 * Rhythm has extra logic: negative value = reversed sequence → always weakness.
 */
export function classifyMetric(metric: string, value: number): MetricResult {
  const t = THRESHOLDS[metric];
  const fb = FEEDBACK[metric];
  if (!t || !fb) {
    return { metric, tier: 'weakness', value, feedback: '' };
  }

  // --- Rhythm special case ---
  if (metric === 'rhythm') {
    if (value < 0) {
      return { metric, tier: 'weakness', value, feedback: fb.weakness };
    }
    const st = t as SimpleThreshold;
    if (value <= st.ideal) return { metric, tier: 'ideal', value, feedback: '' };
    if (value <= st.good)  return { metric, tier: 'good',  value, feedback: '' };
    return { metric, tier: 'weakness', value, feedback: fb.weakness };
  }

  // --- Range / bidirectional ---
  if (t.kind === 'range') {
    const rt = t as RangeThreshold;
    if (value >= rt.idealMin && value <= rt.idealMax) {
      return { metric, tier: 'ideal', value, feedback: '' };
    }
    if (value >= rt.goodMin && value <= rt.goodMax) {
      return { metric, tier: 'good', value, feedback: '' };
    }
    // Weakness — pick direction-specific feedback
    const isHigh = value > rt.idealMax;
    const feedback = isHigh && fb.weaknessAlt ? fb.weaknessAlt : fb.weakness;
    return { metric, tier: 'weakness', value, feedback };
  }

  // --- Simple thresholds ---
  const st = t as SimpleThreshold;
  if (st.kind === 'lower-better') {
    if (value <= st.ideal) return { metric, tier: 'ideal', value, feedback: '' };
    if (value <= st.good)  return { metric, tier: 'good',  value, feedback: '' };
    return { metric, tier: 'weakness', value, feedback: fb.weakness };
  }

  // higher-better
  if (value >= st.ideal) return { metric, tier: 'ideal', value, feedback: '' };
  if (value >= st.good)  return { metric, tier: 'good',  value, feedback: '' };
  return { metric, tier: 'weakness', value, feedback: fb.weakness };
}

// ======================== SEVERITY (for sorting) ========================

/**
 * How far a value is from the ideal range, normalized 0–1.
 * Used to sort workOn by "worst first."
 */
export function calculateSeverity(metric: string, value: number): number {
  const t = THRESHOLDS[metric];
  if (!t) return 0;

  let idealMin: number;
  let idealMax: number;

  if (t.kind === 'range') {
    idealMin = (t as RangeThreshold).idealMin;
    idealMax = (t as RangeThreshold).idealMax;
  } else if (t.kind === 'lower-better') {
    idealMin = -Infinity;
    idealMax = (t as SimpleThreshold).ideal;
  } else {
    // higher-better
    idealMin = (t as SimpleThreshold).ideal;
    idealMax = Infinity;
  }

  if (value >= idealMin && value <= idealMax) return 0;

  const distance = value < idealMin ? idealMin - value : value - idealMax;

  // Practical full-range per metric for normalisation
  const ranges: Record<string, number> = {
    elbowFlare: 30, trunkLean: 30, footAlignment: 30,
    kneeBend: 180, elbowExtension: 180,
    footStance: 200,
    wristSnap: 100, releaseHeight: 100, jumpHeight: 100,
    rhythm: 200,
  };

  return Math.min(1, distance / (ranges[metric] ?? 100));
}

// ======================== FATIGUE DETECTION ========================

function avg(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/**
 * Returns fatigue tips based on jump-height trend within the session
 * and comparison to career baseline.
 */
export function detectFatigue(jumpHeights: number[], careerAverage?: number): string[] {
  const tips: string[] = [];

  if (jumpHeights.length >= 20) {
    const first10 = avg(jumpHeights.slice(0, 10));
    const last10  = avg(jumpHeights.slice(-10));
    if (first10 > 0 && (first10 - last10) / first10 > 0.10) {
      tips.push('Your jump height dropped late session — fatigue?');
    }
  }

  if (careerAverage && careerAverage > 0) {
    const sessionAvg = avg(jumpHeights);
    if ((careerAverage - sessionAvg) / careerAverage > 0.10) {
      tips.push("Your legs aren't as loaded today — recovery day?");
    }
  }

  return tips;
}

// ======================== CONSISTENCY ========================

/**
 * Converts per-metric standard deviations into a 0–100 consistency score.
 * Low variance → high score.
 */
export function calculateConsistencyScore(metricStdDevs: Record<string, number>): number {
  const entries = Object.entries(metricStdDevs);
  if (entries.length === 0) return 0;

  // Max expected std-dev per metric (anything above → score 0 for that metric)
  const caps: Record<string, number> = {
    elbowFlare: 10, trunkLean: 10, footAlignment: 10, releaseHeight: 10,
    kneeBend: 20, elbowExtension: 20, wristSnap: 20, jumpHeight: 20,
    footStance: 30, rhythm: 100,
  };

  let total = 0;
  for (const [metric, sd] of entries) {
    const cap = caps[metric] ?? 20;
    total += Math.max(0, 1 - Math.min(sd / cap, 1)) * 100;
  }

  return Math.round(total / entries.length);
}

// ======================== MAIN ENTRY POINT ========================

/**
 * Classify a full session of shooting metrics.
 *
 * @param userMetrics     Session-average values keyed by metric name.
 *                        Missing keys are silently skipped.
 * @param jumpHeights     Optional array of every jump-height value in the session
 *                        (used for fatigue detection).
 * @param careerJumpAvg   Optional career-average jump height for baseline comparison.
 * @param metricStdDevs   Optional per-metric standard deviations for consistency scoring.
 */
export function classifyIdealForm(
  userMetrics: Record<string, number>,
  jumpHeights?: number[],
  careerJumpAvg?: number,
  metricStdDevs?: Record<string, number>,
): IdealFormResult {
  const strengths: MetricResult[]  = [];
  const weaknesses: MetricResult[] = [];
  const quickTips: string[]        = [];

  for (const [metric, value] of Object.entries(userMetrics)) {
    if (value == null || !THRESHOLDS[metric]) continue;

    const result = classifyMetric(metric, value);

    if (result.tier === 'ideal') {
      strengths.push(result);
    } else if (result.tier === 'weakness') {
      weaknesses.push(result);
    }
  }

  // Sort by severity (worst first), cap at 3
  weaknesses.sort((a, b) =>
    calculateSeverity(b.metric, b.value) - calculateSeverity(a.metric, a.value)
  );
  const workOn = weaknesses.slice(0, 3);

  // Quick tips from weakness quick-fix descriptions
  for (const item of workOn) {
    const fix = QUICK_FIXES[item.metric];
    if (fix) quickTips.push(fix.description);
  }

  // Fatigue tips
  if (jumpHeights && jumpHeights.length > 0) {
    quickTips.push(...detectFatigue(jumpHeights, careerJumpAvg));
  }

  const consistencyScore = metricStdDevs
    ? calculateConsistencyScore(metricStdDevs)
    : 0;

  return { strengths, workOn, quickTips, consistencyScore };
}

// ======================== NBA PLAYER MATCH ========================

const NBA_MATCH_THRESHOLDS = {
  trunkLean:     { ideal: 2, good: 5 },
  kneeBend:      { ideal: 5, good: 10 },
  wristSnap:     { ideal: 5, good: 15 },
  releaseHeight: { ideal: 5, good: 10 },
  footAlignment: { ideal: 3, good: 7 },
} as const;

const NBA_FEEDBACK_TEMPLATES = {
  trunkLean: (direction: Direction, playerName: string) =>
    `${direction === 'more' ? 'More' : 'Less'} forward lean than ${playerName}'s upright form`,
  kneeBend: (direction: Direction, _playerName: string) =>
    `${direction === 'more' ? 'More' : 'Less'} knee flexion at release point`,
  wristSnap: (direction: Direction, playerName: string) =>
    `Follow-through ${direction === 'shorter' ? 'shorter' : 'longer'} than ${playerName}'s finish`,
  releaseHeight: (direction: Direction, playerName: string) =>
    `Release point ${direction === 'higher' ? 'higher' : 'lower'} than ${playerName}'s`,
  footAlignment: (direction: Direction, playerName: string) =>
    `Stance ${direction === 'more' ? 'more' : 'less'} squared than ${playerName}'s base`,
};

const NBA_QUICK_FIX_TEMPLATES: Record<string, { title: string; description: (playerName: string) => string }> = {
  trunkLean: {
    title: 'Stay more upright',
    description: (playerName) => `Reduce forward trunk lean to match ${playerName}'s vertical posture`,
  },
  kneeBend: {
    title: 'Deepen your knee bend',
    description: (_playerName) => 'More knee flexion generates power and consistency at release',
  },
  wristSnap: {
    title: 'Snap your wrist fully',
    description: (_playerName) => 'Complete follow-through with a full wrist snap for better rotation',
  },
  releaseHeight: {
    title: 'Get your release higher',
    description: (playerName) => `Higher release point matches ${playerName}'s form and improves arc`,
  },
  footAlignment: {
    title: 'Square your feet',
    description: (playerName) => `Align your stance to the basket like ${playerName}'s base`,
  },
};

function classifyByDistance(delta: number, thresholds: { ideal: number; good: number }): Tier {
  const absDelta = Math.abs(delta);
  if (absDelta <= thresholds.ideal) return 'ideal';
  if (absDelta <= thresholds.good) return 'good';
  return 'weakness';
}

function getDirection(delta: number, metric: string): Direction {
  switch (metric) {
    case 'trunkLean':
    case 'kneeBend':
    case 'footAlignment':
      return delta > 0 ? 'more' : 'less';
    case 'wristSnap':
      return delta > 0 ? 'longer' : 'shorter';
    case 'releaseHeight':
      return delta > 0 ? 'higher' : 'lower';
    default:
      return delta > 0 ? 'more' : 'less';
  }
}

function hasDegraded(currentTier: Tier, previousTier: Tier | undefined): boolean {
  if (!previousTier) return false;
  const tierOrder: Tier[] = ['ideal', 'good', 'weakness'];
  return tierOrder.indexOf(currentTier) > tierOrder.indexOf(previousTier);
}

/**
 * NBA Player Match Classification
 * Compares user metrics to matched NBA player's metrics
 */
export function classifyNBAMatch(
  userMetrics: NBAMatchMetrics,
  playerMetrics: NBAMatchMetrics,
  playerName: string,
  previousSessionTiers?: PreviousSessionTiers
): NBAMatchResult {
  const sharedTraits: MetricResult[] = [];
  const keyDifferences: MetricResult[] = [];
  const quickFixes: QuickFix[] = [];

  const metrics: (keyof NBAMatchMetrics)[] = [
    'trunkLean', 'kneeBend', 'wristSnap', 'releaseHeight', 'footAlignment',
  ];

  for (const metric of metrics) {
    const userValue = userMetrics[metric];
    const playerValue = playerMetrics[metric];
    if (userValue === undefined || playerValue === undefined) continue;

    const delta = userValue - playerValue;
    const tier = classifyByDistance(delta, NBA_MATCH_THRESHOLDS[metric]);
    const direction = getDirection(delta, metric);
    const feedback = NBA_FEEDBACK_TEMPLATES[metric](direction, playerName);
    const isDegraded = hasDegraded(tier, previousSessionTiers?.[metric]);

    const result: MetricResult = { metric, tier, value: userValue, feedback, delta, direction, isDegraded };

    if (tier === 'ideal') {
      sharedTraits.push(result);
    } else {
      keyDifferences.push(result);
    }

    const shouldSurfaceQuickFix = tier === 'weakness' || (tier === 'good' && isDegraded);
    if (shouldSurfaceQuickFix && NBA_QUICK_FIX_TEMPLATES[metric]) {
      const template = NBA_QUICK_FIX_TEMPLATES[metric];
      quickFixes.push({
        metric,
        title: template.title,
        description: template.description(playerName),
        reason: tier === 'weakness' ? 'weakness' : 'degraded',
      });
    }
  }

  keyDifferences.sort((a, b) => {
    const tierOrder = { weakness: 0, good: 1, ideal: 2 };
    return tierOrder[a.tier] - tierOrder[b.tier];
  });

  return { sharedTraits, keyDifferences, quickFixes };
}

// ======================== UI HELPERS ========================

export function getTierColor(tier: Tier): string {
  switch (tier) {
    case 'ideal':    return '#4ade80'; // green
    case 'good':     return '#fbbf24'; // yellow
    case 'weakness': return '#f87171'; // red
    default:         return '#a0a0a0';
  }
}

export function getTierLabel(tier: Tier): string {
  switch (tier) {
    case 'ideal':    return 'Strength';
    case 'good':     return 'Good';
    case 'weakness': return 'Work On';
    default:         return 'Unknown';
  }
}

// Re-export constants for consumers that need raw data
export { THRESHOLDS, FEEDBACK, QUICK_FIXES };
