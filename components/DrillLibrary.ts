// ============================================================================
// DRILL LIBRARY & SELECTION LOGIC
// Roots Basketball - Targeted Drills for Shot Improvement
// ============================================================================

// ======================== TYPES ========================

export interface Drill {
  id: string;
  name: string;
  description: string;
  targetMetrics: string[];
  reps: string;
  coachingCue: string;
  videoUrl?: string;
}

export interface WorkOnItem {
  title: string;
  description: string;
  drillName: string;
  drillTip: string;
}

export interface WeaknessInput {
  metric: string;
  feedback: string;
}

// ======================== DRILL DATABASE ========================

export const drills: Drill[] = [
  // Elbow Flare
  {
    id: 'wall-alignment',
    name: 'Wall Alignment',
    description: 'Stand with shooting elbow touching wall, practice release without elbow leaving wall',
    targetMetrics: ['elbowFlare'],
    reps: '3x10',
    coachingCue: 'Keep your elbow brushing the wall on every rep',
  },
  {
    id: 'elbow-in-holds',
    name: 'Elbow-In Holds',
    description: 'Hold ball at set point, partner checks elbow alignment, hold 3 sec',
    targetMetrics: ['elbowFlare'],
    reps: '10',
    coachingCue: 'Freeze at the top — elbow should point at the rim',
  },
  {
    id: 'one-hand-form-shooting',
    name: 'One-Hand Form Shooting',
    description: 'Close range, shooting hand only, focus on straight elbow path',
    targetMetrics: ['elbowFlare'],
    reps: '20',
    coachingCue: 'Your elbow, wrist, and finger should make one straight line to the basket',
  },

  // Trunk Lean
  {
    id: 'pause-and-shoot',
    name: 'Pause-and-Shoot',
    description: 'Catch, pause 2 sec at set point with upright posture, then release',
    targetMetrics: ['trunkLean'],
    reps: '20',
    coachingCue: 'Feel your weight centered over your hips at the pause',
  },
  {
    id: 'back-to-wall-shooting',
    name: 'Back-to-Wall Shooting',
    description: 'Heels 6 inches from wall, shoot without touching wall with back',
    targetMetrics: ['trunkLean'],
    reps: '15',
    coachingCue: "If you feel the wall, you're leaning",
  },
  {
    id: 'chair-drill',
    name: 'Chair Drill',
    description: 'Start seated, stand and shoot in one motion, forces vertical rise',
    targetMetrics: ['trunkLean', 'kneeBend'],
    reps: '15',
    coachingCue: 'Drive straight up — the chair keeps you honest',
  },

  // Knee Bend
  {
    id: 'squat-and-shoot',
    name: 'Squat-and-Shoot',
    description: 'Exaggerated knee bend, pause at bottom, explode into shot',
    targetMetrics: ['kneeBend'],
    reps: '3x10',
    coachingCue: 'Sink until your thighs burn, then explode',
  },
  {
    id: 'legs-only-shooting',
    name: 'Legs-Only Shooting',
    description: 'Close range, minimal arm motion, power comes entirely from legs',
    targetMetrics: ['kneeBend'],
    reps: '15',
    coachingCue: 'Your legs do 80% of the work here',
  },
  {
    id: 'jump-stop-shooting',
    name: 'Jump Stop Shooting',
    description: 'Catch with jump stop, feel the load in your legs, then shoot',
    targetMetrics: ['kneeBend', 'rhythm'],
    reps: '20',
    coachingCue: 'Land loaded, then go — no wasted motion',
  },

  // Elbow Extension
  {
    id: 'form-shooting-no-jump',
    name: 'Form Shooting (no jump)',
    description: 'Close range, full arm extension, no jump, focus on arm path',
    targetMetrics: ['elbowExtension'],
    reps: '20',
    coachingCue: 'Reach for the rim like putting something on a high shelf',
  },
  {
    id: 'reach-and-hold',
    name: 'Reach-and-Hold',
    description: 'Shoot normally, freeze follow-through for 3 sec with arm fully extended',
    targetMetrics: ['elbowExtension'],
    reps: '15',
    coachingCue: 'Lock out your elbow — hold until the ball hits the ground',
  },
  {
    id: 'guide-hand-release',
    name: 'Guide Hand Release',
    description: 'Shoot with guide hand falling away early, forces full shooting arm extension',
    targetMetrics: ['elbowExtension', 'wristSnap'],
    reps: '15',
    coachingCue: 'Guide hand off early, shooting arm does all the work',
  },

  // Wrist Snap
  {
    id: 'wrist-snap-drill',
    name: 'Wrist Snap Drill',
    description: 'Ball on palm, snap wrist to self-pass with backspin, catch, repeat',
    targetMetrics: ['wristSnap'],
    reps: '3x20',
    coachingCue: 'Fingers point down at the finish, ball spins back to you',
  },
  {
    id: 'reach-for-rim-finish',
    name: 'Reach-for-Rim Finish',
    description: 'Shoot and hold follow-through, fingers pointing at rim',
    targetMetrics: ['wristSnap'],
    reps: '20',
    coachingCue: 'Finish with your hand in the cookie jar',
  },
  {
    id: 'one-hand-flick',
    name: 'One-Hand Flick',
    description: 'Close range, one hand, wrist snap only, ball should have high arc',
    targetMetrics: ['wristSnap'],
    reps: '20',
    coachingCue: 'All wrist, no arm — feel the snap',
  },

  // Rhythm
  {
    id: 'rhythm-shooting',
    name: 'Rhythm Shooting',
    description: 'Catch, one-two step, shoot in continuous motion, no pause',
    targetMetrics: ['rhythm'],
    reps: '20',
    coachingCue: "Say 'catch-dip-shoot' out loud to feel the flow",
  },
  {
    id: '1-2-step-footwork',
    name: '1-2 Step Footwork',
    description: 'Focus on footwork timing, catch with inside foot first, then shoot',
    targetMetrics: ['rhythm', 'footAlignment'],
    reps: '3x10',
    coachingCue: 'Inside foot, outside foot, up — same every time',
  },
  {
    id: 'dip-and-drive',
    name: 'Dip-and-Drive',
    description: 'Exaggerate the dip, feel the gather, then explode into shot',
    targetMetrics: ['rhythm'],
    reps: '15',
    coachingCue: 'Low to high, smooth and fast',
  },

  // Foot Alignment
  {
    id: 'catch-and-square',
    name: 'Catch-and-Square',
    description: 'Partner passes from different angles, catch and square feet before shooting',
    targetMetrics: ['footAlignment'],
    reps: '3x10',
    coachingCue: 'Toes to the rim before the ball touches your hands',
  },
  {
    id: 'cone-target-stance',
    name: 'Cone Target Stance',
    description: 'Place cones where feet should land, catch and hit the marks every rep',
    targetMetrics: ['footAlignment', 'footStance'],
    reps: '20',
    coachingCue: 'Hit your spots — same base every time',
  },
  {
    id: 'feet-alignment-drill',
    name: 'Feet Alignment Drill',
    description: 'Tape line on floor pointing at rim, align feet to tape before every shot',
    targetMetrics: ['footAlignment'],
    reps: '20',
    coachingCue: "The tape doesn't lie",
  },
  {
    id: 'mirror-stance-check',
    name: 'Mirror Stance Check',
    description: 'Practice in front of mirror, check foot width and alignment at set point',
    targetMetrics: ['footAlignment', 'footStance'],
    reps: '10',
    coachingCue: 'Look down, feet should match shoulders',
  },

  // Foot Stance
  {
    id: 'shoulder-width-holds',
    name: 'Shoulder-Width Holds',
    description: 'Set feet at shoulder width, partner checks, hold and shoot',
    targetMetrics: ['footStance'],
    reps: '15',
    coachingCue: 'Find your base, own your base',
  },
  {
    id: 'narrow-to-wide-adjustment',
    name: 'Narrow-to-Wide Adjustment',
    description: 'Start too narrow, adjust to proper width, then shoot',
    targetMetrics: ['footStance'],
    reps: '3x10',
    coachingCue: 'Feel the difference — stable base, stable shot',
  },

  // Release Height
  {
    id: 'high-point-target',
    name: 'High-Point Target',
    description: 'Hang a target above set point, touch it before every release',
    targetMetrics: ['releaseHeight'],
    reps: '20',
    coachingCue: "If you don't touch it, the rep doesn't count",
  },
  {
    id: 'rainbow-shooting',
    name: 'Rainbow Shooting',
    description: 'Aim for high arc, ball should drop straight down through net',
    targetMetrics: ['releaseHeight'],
    reps: '20',
    coachingCue: "Shoot it over an imaginary defender's hand",
  },
  {
    id: 'set-point-freeze',
    name: 'Set-Point Freeze',
    description: 'Catch, bring ball to set point, freeze 2 sec, then release',
    targetMetrics: ['releaseHeight'],
    reps: '15',
    coachingCue: 'Find your spot, hold it, then go',
  },
];

// ======================== DRILL PRIORITY MAPPING ========================

export const drillPriority: Record<string, { primary: string[]; secondary: string[] }> = {
  elbowFlare: {
    primary: ['wall-alignment', 'elbow-in-holds'],
    secondary: ['one-hand-form-shooting'],
  },
  trunkLean: {
    primary: ['pause-and-shoot', 'back-to-wall-shooting'],
    secondary: ['chair-drill'],
  },
  kneeBend: {
    primary: ['squat-and-shoot', 'legs-only-shooting'],
    secondary: ['jump-stop-shooting', 'chair-drill'],
  },
  elbowExtension: {
    primary: ['form-shooting-no-jump', 'reach-and-hold'],
    secondary: ['guide-hand-release'],
  },
  wristSnap: {
    primary: ['wrist-snap-drill', 'reach-for-rim-finish'],
    secondary: ['one-hand-flick', 'guide-hand-release'],
  },
  rhythm: {
    primary: ['rhythm-shooting', '1-2-step-footwork'],
    secondary: ['dip-and-drive', 'jump-stop-shooting'],
  },
  footAlignment: {
    primary: ['catch-and-square', 'feet-alignment-drill'],
    secondary: ['cone-target-stance', '1-2-step-footwork'],
  },
  footStance: {
    primary: ['shoulder-width-holds', 'narrow-to-wide-adjustment'],
    secondary: ['cone-target-stance', 'mirror-stance-check'],
  },
  releaseHeight: {
    primary: ['high-point-target', 'rainbow-shooting'],
    secondary: ['set-point-freeze'],
  },
  jumpHeight: {
    primary: ['legs-only-shooting', 'squat-and-shoot'],
    secondary: ['jump-stop-shooting'],
  },
};

// ======================== QUICK FIX TEMPLATES ========================

const quickFixTemplates: Record<string, { title: string; description: string }> = {
  elbowFlare: {
    title: 'Straighten your elbow path',
    description: 'Your elbow is drifting out — this causes left/right misses',
  },
  trunkLean: {
    title: 'Stay more upright',
    description: "You're leaning forward — stay tall through your release",
  },
  kneeBend: {
    title: 'Deepen your knee bend',
    description: "Your legs aren't loaded — you're losing power",
  },
  elbowExtension: {
    title: 'Extend your arm fully',
    description: "Your arm isn't extending fully — shorter range, less arc",
  },
  wristSnap: {
    title: 'Snap your wrist fully',
    description: "Your wrist isn't snapping through — affects spin and touch",
  },
  rhythm: {
    title: 'Sync your timing',
    description: 'Your timing is off — elbow is firing before your legs',
  },
  footAlignment: {
    title: 'Square your feet',
    description: "Your feet aren't squared — you're fading left/right",
  },
  footStance: {
    title: 'Fix your base width',
    description: 'Your base is too narrow/wide — affects balance',
  },
  releaseHeight: {
    title: 'Get your release higher',
    description: "You're releasing low — easier to block, flatter shot",
  },
  jumpHeight: {
    title: 'Load your legs',
    description: 'Your jump height is inconsistent — focus on a consistent leg load each rep',
  },
};

// ======================== DRILL LOOKUP FUNCTIONS ========================

/**
 * Get a drill by its ID
 */
export function getDrillById(id: string): Drill | undefined {
  return drills.find((drill) => drill.id === id);
}

/**
 * Get drills for a specific metric
 * Returns primary drills first, then secondary if needed
 */
export function getDrillsForMetric(metric: string, count: number = 2): Drill[] {
  const priority = drillPriority[metric];
  if (!priority) {
    return [];
  }

  const result: Drill[] = [];
  const allDrillIds = [...priority.primary, ...priority.secondary];

  for (const drillId of allDrillIds) {
    if (result.length >= count) break;

    const drill = getDrillById(drillId);
    if (drill) {
      result.push(drill);
    }
  }

  return result;
}

/**
 * Get drills for multiple weaknesses
 * Returns max 2 drills per weakness, deduplicates, sorts by coverage
 */
export function getDrillsForWeaknesses(weaknesses: string[]): Drill[] {
  if (weaknesses.length === 0) {
    return [];
  }

  // Track which drills we've selected and how many weaknesses they address
  const drillScores: Map<string, { drill: Drill; weaknessCount: number; weaknesses: Set<string> }> = new Map();

  // For each weakness, get up to 2 drills (prioritizing primary)
  for (const metric of weaknesses) {
    const priority = drillPriority[metric];
    if (!priority) continue;

    const drillIds = [...priority.primary.slice(0, 2), ...priority.secondary.slice(0, 1)];
    let addedForMetric = 0;

    for (const drillId of drillIds) {
      if (addedForMetric >= 2) break;

      const drill = getDrillById(drillId);
      if (!drill) continue;

      const existing = drillScores.get(drillId);
      if (existing) {
        // Drill already selected - check if it addresses this weakness too
        if (!existing.weaknesses.has(metric) && drill.targetMetrics.includes(metric)) {
          existing.weaknessCount++;
          existing.weaknesses.add(metric);
        }
      } else {
        // New drill
        const addressedWeaknesses = new Set<string>();
        for (const w of weaknesses) {
          if (drill.targetMetrics.includes(w)) {
            addressedWeaknesses.add(w);
          }
        }
        drillScores.set(drillId, {
          drill,
          weaknessCount: addressedWeaknesses.size,
          weaknesses: addressedWeaknesses,
        });
        addedForMetric++;
      }
    }
  }

  // Convert to array and sort by weakness count (multi-target drills first)
  const sortedDrills = Array.from(drillScores.values())
    .sort((a, b) => b.weaknessCount - a.weaknessCount)
    .map((entry) => entry.drill);

  return sortedDrills;
}

/**
 * Get WorkOn items for UI display
 * Takes weaknesses from rule engine and returns formatted items with drills
 */
export function getWorkOnItems(weaknesses: WeaknessInput[]): WorkOnItem[] {
  const workOnItems: WorkOnItem[] = [];

  // Take first 3 weaknesses (already sorted by severity)
  const topWeaknesses = weaknesses.slice(0, 3);

  for (const weakness of topWeaknesses) {
    const { metric } = weakness;

    // Get quick fix template
    const template = quickFixTemplates[metric];
    if (!template) continue;

    // Get primary drill for this metric
    const priority = drillPriority[metric];
    if (!priority || priority.primary.length === 0) continue;

    const drill = getDrillById(priority.primary[0]);
    if (!drill) continue;

    workOnItems.push({
      title: template.title,
      description: template.description,
      drillName: drill.name,
      drillTip: drill.coachingCue,
    });
  }

  return workOnItems;
}

// ======================== ADDITIONAL UTILITIES ========================

/**
 * Get all drills that target a specific metric
 */
export function getAllDrillsForMetric(metric: string): Drill[] {
  return drills.filter((drill) => drill.targetMetrics.includes(metric));
}

/**
 * Get drills that target multiple metrics (combo drills)
 */
export function getMultiTargetDrills(): Drill[] {
  return drills.filter((drill) => drill.targetMetrics.length > 1);
}

/**
 * Get a random drill for a metric (for variety)
 */
export function getRandomDrillForMetric(metric: string): Drill | undefined {
  const metricssDrills = getAllDrillsForMetric(metric);
  if (metricssDrills.length === 0) return undefined;
  const randomIndex = Math.floor(Math.random() * metricssDrills.length);
  return metricssDrills[randomIndex];
}

/**
 * Build a practice session from weaknesses
 * Returns a structured session with warm-up, main drills, and cool-down
 */
export interface PracticeSession {
  warmUp: Drill[];
  mainDrills: Drill[];
  totalReps: string;
  estimatedTime: string;
}

export function buildPracticeSession(weaknesses: string[]): PracticeSession {
  const mainDrills = getDrillsForWeaknesses(weaknesses);

  // Add form shooting as warm-up if not already included
  const warmUp: Drill[] = [];
  const formShooting = getDrillById('one-hand-form-shooting');
  if (formShooting && !mainDrills.some((d) => d.id === 'one-hand-form-shooting')) {
    warmUp.push(formShooting);
  }

  // Calculate total reps (rough estimate)
  const repCounts = mainDrills.map((d) => {
    const match = d.reps.match(/(\d+)/);
    return match ? parseInt(match[1], 10) : 15;
  });
  const totalRepCount = repCounts.reduce((sum, r) => sum + r, 0);

  // Estimate time (roughly 30 seconds per rep including reset)
  const estimatedMinutes = Math.ceil((totalRepCount * 0.5) + 5); // +5 for warm-up

  return {
    warmUp,
    mainDrills,
    totalReps: `~${totalRepCount} shots`,
    estimatedTime: `${estimatedMinutes} mins`,
  };
}

/**
 * Get drill recommendations with priority scores
 */
export interface DrillRecommendation {
  drill: Drill;
  priority: 'high' | 'medium' | 'low';
  targetedWeaknesses: string[];
}

export function getDrillRecommendations(
  weaknesses: string[],
  maxRecommendations: number = 5
): DrillRecommendation[] {
  const recommendations: DrillRecommendation[] = [];
  const usedDrillIds = new Set<string>();

  // First pass: primary drills for each weakness (high priority)
  for (const metric of weaknesses) {
    const priority = drillPriority[metric];
    if (!priority) continue;

    for (const drillId of priority.primary) {
      if (usedDrillIds.has(drillId)) continue;
      if (recommendations.length >= maxRecommendations) break;

      const drill = getDrillById(drillId);
      if (!drill) continue;

      const targetedWeaknesses = weaknesses.filter((w) => drill.targetMetrics.includes(w));

      recommendations.push({
        drill,
        priority: 'high',
        targetedWeaknesses,
      });
      usedDrillIds.add(drillId);
    }
  }

  // Second pass: secondary drills (medium priority)
  if (recommendations.length < maxRecommendations) {
    for (const metric of weaknesses) {
      const priority = drillPriority[metric];
      if (!priority) continue;

      for (const drillId of priority.secondary) {
        if (usedDrillIds.has(drillId)) continue;
        if (recommendations.length >= maxRecommendations) break;

        const drill = getDrillById(drillId);
        if (!drill) continue;

        const targetedWeaknesses = weaknesses.filter((w) => drill.targetMetrics.includes(w));

        recommendations.push({
          drill,
          priority: 'medium',
          targetedWeaknesses,
        });
        usedDrillIds.add(drillId);
      }
    }
  }

  // Sort: multi-target drills first, then by priority
  recommendations.sort((a, b) => {
    // First by number of weaknesses addressed
    if (b.targetedWeaknesses.length !== a.targetedWeaknesses.length) {
      return b.targetedWeaknesses.length - a.targetedWeaknesses.length;
    }
    // Then by priority
    const priorityOrder = { high: 0, medium: 1, low: 2 };
    return priorityOrder[a.priority] - priorityOrder[b.priority];
  });

  return recommendations;
}
