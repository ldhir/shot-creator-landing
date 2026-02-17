# Benchmark Metrics Extraction Guide

## Overview

The benchmark video processing system now automatically extracts and stores detailed metrics for each frame. These metrics enable multi-metric similarity comparison with non-linear penalty functions.

## What's New

When processing benchmark videos (using `processVideoForBenchmark()` or recording benchmarks), the system now extracts and stores the following metrics for each frame:

### Extracted Metrics

1. **wrist_snap** - Angle between forearm and vertical (degrees)
   - Measures how much the wrist is snapped forward
   - Range: 0-180° (0° = no snap, higher = more snap)

2. **elbow_extension** - Angle at elbow joint (degrees)
   - Measures elbow bend/extension
   - Range: 0-180° (180° = fully extended)

3. **knee_bend** - Angle at knee joint (degrees)
   - Measures knee flexion
   - Range: 0-180° (lower = more bend)

4. **elbow_flare** - Deviation from ideal elbow position (degrees)
   - Measures how much elbow deviates from 90° perpendicular to shoulder line
   - Range: 0-90° (0° = perfect alignment, higher = more flare)

5. **trunk_lean** - Forward/backward lean of torso (degrees)
   - Positive = forward lean, negative = backward lean
   - Range: typically -30° to +30°

6. **foot_alignment** - Difference between foot and shoulder alignment (degrees)
   - Measures how aligned feet are with shoulders
   - Range: typically -45° to +45° (0° = perfectly aligned)

7. **shoulder_angle** - Shoulder alignment in x-z plane (degrees)
   - Used for calculating foot alignment

8. **foot_angle** - Foot alignment in x-z plane (degrees)
   - Used for calculating foot alignment

9. **release_height** - Wrist height relative to ground (normalized units)
   - Negative value = above ground
   - Can be converted to actual inches using user height

## Data Structure

Each frame in the benchmark data now includes a `metrics` object:

```javascript
{
    state: "pre_shot",
    time: 0.5,
    elbow_angle: 120.5,
    wrist_angle: 150.2,
    arm_angle: 45.8,
    landmarks: [[x, y, z], ...], // 33 normalized landmarks
    metrics: {
        wrist_snap: 25.3,
        elbow_extension: 165.2,
        knee_bend: 95.5,
        elbow_flare: 8.2,
        trunk_lean: 5.1,
        foot_alignment: -3.2,
        shoulder_angle: 2.1,
        foot_angle: -1.1,
        release_height: -0.45  // relative to ground
    }
}
```

## How It Works

### During Video Processing

When you process a benchmark video using `processVideoForBenchmark()`:

1. The video is processed frame by frame
2. For each frame, pose landmarks are extracted using MediaPipe
3. Landmarks are normalized to align shoulders with x-axis
4. Detailed metrics are extracted from the normalized landmarks
5. Metrics are stored in the `metrics` field of each frame

### Metric Extraction Function

The `extractDetailedMetricsFromLandmarks()` function:
- Takes normalized landmarks (33 points) as input
- Calculates angles and measurements using 3D geometry
- Returns a metrics object with all calculated values
- Handles missing landmarks gracefully (returns null for unavailable metrics)

## Usage Examples

### Processing a New Benchmark Video

```javascript
// Process a video file and extract all metrics
const videoFile = document.getElementById('videoInput').files[0];
const poseData = await processVideoForBenchmark(videoFile, 'curry');

// poseData now includes metrics for each frame
console.log(poseData[0].metrics); // See first frame's metrics
```

### Accessing Metrics in Comparison

```javascript
// When comparing shots, you can access metrics:
const benchmarkFrame = benchmarkData[10];
const userFrame = userPoseData[10];

// Compare specific metrics
const wristSnapDiff = Math.abs(
    benchmarkFrame.metrics.wrist_snap - 
    userFrame.metrics.wrist_snap
);

const elbowExtDiff = Math.abs(
    benchmarkFrame.metrics.elbow_extension - 
    userFrame.metrics.elbow_extension
);
```

### Calculating Release Height in Inches

```javascript
// Release height is stored as normalized units
// To convert to inches, you need the user's height
function calculateReleaseHeightInches(metrics, userHeightInches) {
    if (!metrics.release_height) return null;
    
    // The release_height is relative to the person's height
    // You'll need to scale it based on the actual person height
    // This is a simplified calculation - adjust based on your coordinate system
    return Math.abs(metrics.release_height) * userHeightInches;
}
```

## Next Steps

With these metrics extracted, you can now:

1. **Implement Multi-Metric Similarity**: Use the weighted, non-linear similarity calculation discussed earlier
2. **Compare Specific Metrics**: Focus on individual aspects of form (e.g., just wrist snap or elbow extension)
3. **Generate Detailed Feedback**: Provide specific feedback on each metric
4. **Track Improvements**: Monitor changes in specific metrics over time

## Notes

- Metrics are calculated from normalized landmarks, ensuring consistent measurements regardless of camera angle
- Some metrics may be `null` if required landmarks aren't detected (low visibility, occlusion, etc.)
- The `release_height` is stored as a relative value and needs user height to convert to actual inches
- All angle measurements are in degrees
- The system automatically extracts these metrics for both benchmark and user recordings

## Compatibility

- Existing benchmark files without `metrics` will still work (metrics will be `null`)
- New benchmarks will automatically include metrics
- When re-processing old videos, metrics will be extracted and added
