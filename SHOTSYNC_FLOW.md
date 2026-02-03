# ShotSync Function Flow - Complete Process

## Overview
This document traces the complete flow of functions when a user uploads a video and uses ShotSync to extract landmarks, calculate angles, and compare with benchmarks.

---

## 1. VIDEO UPLOAD PHASE

### Entry Point: User Clicks "UPLOAD VIDEO"
**File:** `shotsync/index.html`  
**Line:** ~10580

**Function:** `userVideoUpload.addEventListener('change', ...)`
- User selects video file
- Creates blob URL: `URL.createObjectURL(file)`
- Sets video source: `video.src = url`
- Waits for video to load (`loadedmetadata`, `loadeddata` events)

**Next:** Calls `onLoaded()` callback

---

## 2. VIDEO LOADED & ANALYSIS OPTIONS

### Function: `onLoaded()` (inline async function)
**File:** `shotsync/index.html`  
**Line:** ~10600

**Actions:**
1. Calls `showAnalysisOptions()` - Shows analysis type selection UI
2. Displays filename
3. **Automatically calls:** `await generateLoop(true)` - Extracts angles silently

---

## 3. LANDMARK EXTRACTION PHASE

### Function: `generateLoop(silent = false)`
**File:** `shotsync/index.html`  
**Line:** ~12999

**Purpose:** Process video frame-by-frame to extract pose landmarks and calculate angles

**Process:**
1. **Initialize MediaPipe Pose:**
   ```javascript
   const pose = new Pose({
       locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
   });
   pose.setOptions({
       modelComplexity: 2,
       smoothLandmarks: true,
       minDetectionConfidence: 0.7,
       minTrackingConfidence: 0.7
   });
   ```

2. **Process Each Frame:**
   - Loop through video frames at 30 FPS
   - For each frame:
     - Draw frame to canvas: `ctx.drawImage(video, 0, 0, ...)`
     - Send to MediaPipe: `await pose.send({ image: processCanvas })`

3. **Receive Landmarks via Callback:**
   - `pose.onResults((results) => { ... })` - Called for each frame
   - Extracts landmarks: `results.poseLandmarks` (33 points)
   - Converts to normalized format:
     ```javascript
     const landmarks = results.poseLandmarks.map((lm, i) => ({
         x: lm.x,      // Normalized 0-1
         y: lm.y,      // Normalized 0-1
         z: lm.z || 0, // Normalized depth
         visibility: lm.visibility || 1
     }));
     ```

4. **Shot Stage Detection:**
   - Analyzes landmark positions to detect:
     - `pre_shot`: Wrist below shoulder, above hip, hands together
     - `follow_through`: Wrist above shoulder
     - `neutral`: Everything else
   - Uses state machine to track shot sequences

5. **Extract Angles:**
   - Calls `extractAnglesFromLandmarks(landmarks)` for each frame
   - Stores in `loop3DAngles[]` array

6. **Store Data:**
   - `loop3DPoses[]` - Array of landmark arrays (one per frame)
   - `loop3DStages[]` - Array of stage strings ('pre_shot', 'follow_through', 'neutral')
   - `loop3DAngles[]` - Array of angle objects (one per frame)

---

## 4. ANGLE EXTRACTION FUNCTION

### Function: `extractAnglesFromLandmarks(landmarks)`
**File:** `shotsync/index.html`  
**Line:** ~12788

**Purpose:** Calculate all joint angles from landmark positions

**Calculates:**
1. **Knee Bend:**
   - Right: `calculateAngle(hip[24], knee[26], ankle[28])`
   - Left: `calculateAngle(hip[23], knee[25], ankle[27])`

2. **Elbow Extension:**
   - Right: `calculateAngle(shoulder[12], elbow[14], wrist[16])`
   - Left: `calculateAngle(shoulder[11], elbow[13], wrist[15])`

3. **Wrist Snap:**
   - `calculateAngle(finger[20], wrist[16], elbow[14])`

4. **Elbow Flare:**
   - Uses law of cosines
   - Points: elbow[14], shoulder[12], point below shoulder
   - Formula: `arccos((a² + b² - c²) / (2ab))`

5. **Trunk Lean:**
   - Uses law of cosines
   - Points: shoulder[12], hip[24], knee[26]
   - Angle at hip

6. **Foot Angle:**
   - `atan2(deltaZ, deltaX)` where:
     - `deltaX = ankle[28].x - ankle[27].x`
     - `deltaZ = ankle[28].z - ankle[27].z`

7. **Shoulder Angle:**
   - `atan2(deltaZ, deltaX)` where:
     - `deltaX = shoulder[12].x - shoulder[11].x`
     - `deltaZ = shoulder[12].z - shoulder[11].z`

8. **Foot Alignment:**
   - `foot_angle - shoulder_angle`

9. **Release Height:**
   - Calculated at release frame (max wrist height)
   - Uses eye-to-ankle distance from pre-shot frame as reference

**Returns:** Object with all calculated angles

---

## 5. HELPER FUNCTION: `calculateAngle(point1, point2, point3)`
**File:** `shotsync/index.html`  
**Line:** ~12759

**Purpose:** Calculate angle between three 3D points using dot product

**Formula:**
```javascript
// Vector from point2 to point1
v1 = point1 - point2
// Vector from point2 to point3
v2 = point3 - point2
// Dot product
dot = v1 · v2
// Magnitudes
mag1 = |v1|, mag2 = |v2|
// Angle
angle = arccos(dot / (mag1 × mag2)) × 180/π
```

---

## 6. USER SELECTS ANALYSIS TYPE

### User Clicks "Player Match" or "Ideal Form"
**File:** `shotsync/index.html`  
**Line:** ~10740 (Player Match), ~10989 (Ideal Form)

**Player Match Flow:**
1. Checks if angles extracted: `if (!angles || angles.length === 0)`
2. If not extracted, calls `generateLoop(true)` again
3. Then calls `runPlayerMatchComparison()`

**Ideal Form Flow:**
1. Calls `calculateIdealFormMetrics()` directly

---

## 7. PLAYER MATCH COMPARISON FLOW

### Function: `runPlayerMatchComparison()`
**File:** `shotsync/index.html`  
**Line:** ~10784

**Process:**

1. **Convert Data Format:**
   - Function: `convertToUserData()`
   - Converts `loop3DPoses`, `loop3DStages`, `loop3DAngles` to format expected by comparison
   - Creates array of frame objects:
     ```javascript
     {
         state: 'pre_shot' | 'follow_through' | 'neutral',
         time: frameIndex * (1/30),
         landmarks: [...], // Array of 33 landmarks
         metrics: {
             right_elbow: ...,
             wrist_snap: ...,
             elbow_flare: ...,
             // ... all angles
         }
     }
     ```

2. **Call Comparison Function:**
   - `window.compareWithAllBenchmarks(userData)`
   - This is defined in `tool/app.js`
   - Compares user data with all benchmark files

---

## 8. COMPARISON FUNCTION (in tool/app.js)

### Function: `compareWithAllBenchmarks(userData)`
**File:** `tool/app.js`  
**Line:** ~3199

**Process:**

1. **Load All Benchmark Files:**
   - Gets list of benchmark files: `getAllBenchmarkFiles()`
   - For each benchmark:
     - Loads benchmark data: `loadBenchmarkFromFile(playerId)`
     - Calls `compareDetailedMetrics(userData, benchmarkData)`

2. **Extract Metrics:**
   - Calls `extractMetricsFromData(userData)` - Extracts user metrics
   - Calls `extractMetricsFromData(benchmarkData)` - Extracts benchmark metrics
   - This function:
     - Finds key frames (pre_shot, follow_through, release point)
     - Extracts metrics at specific frames
     - Calculates release height using new method
     - Stores coordinate values for display

3. **Compare Metrics:**
   - Function: `compareDetailedMetrics(userData, benchmarkData)`
   - Calculates similarity scores for each metric
   - Uses weighted, non-linear similarity scoring
   - Returns comparison object with:
     - `overallScore`: Overall similarity percentage
     - `metricScores`: Individual metric scores
     - `userMetrics`: User's extracted metrics
     - `benchmarkMetrics`: Benchmark metrics
     - `sharedTraits`: High similarity metrics
     - `differences`: Low similarity metrics

4. **Sort and Return:**
   - Sorts matches by overall score
   - Returns top 5 matches

---

## 9. METRIC EXTRACTION FUNCTION

### Function: `extractMetricsFromData(data)`
**File:** `tool/app.js`  
**Line:** ~2749

**Purpose:** Extract averaged metrics from shot data

**Process:**

1. **Find Key Frames:**
   - First pre_shot frame (valid sequence: pre_shot → follow_through)
   - First follow_through frame
   - Release point frame (minimum wrist Y = highest wrist)

2. **Extract Metrics at Specific Frames:**
   - **Pre-shot frame:**
     - `elbow_flare`
     - `knee_bend`
     - `trunk_lean`
   
   - **Elbow Extension:**
     - `pre_shot_elbow_angle - follow_through_elbow_angle`
   
   - **Release point:**
     - `wrist_snap`
     - `release_height` (recalculated using new method)
   
   - **Follow-through frame:**
     - `foot_angle`
     - `shoulder_angle`
     - `foot_alignment`

3. **Release Height Calculation:**
   - Gets eye and ankle Y coordinates from first pre-shot frame
   - Gets wrist Y coordinate from release frame
   - Calculates using scale factor based on 95% of user height
   - Stores coordinate values for display

4. **Return Metrics Object:**
   ```javascript
   {
       release_height: ...,
       wrist_snap: ...,
       elbow_extension: ...,
       foot_angle: ...,
       shoulder_angle: ...,
       foot_alignment: ...,
       elbow_flare: ...,
       knee_bend: ...,
       trunk_lean: ...,
       release_height_coords: { ... }, // For display
       foot_angle_coords: { ... }      // For display
   }
   ```

---

## 10. IDEAL FORM METRICS CALCULATION

### Function: `calculateIdealFormMetrics()`
**File:** `shotsync/index.html`  
**Line:** ~11942

**Purpose:** Calculate Ideal Form metrics from extracted angles

**Process:**

1. **Find Key Frames:**
   - Release frame (minimum wrist Y)
   - First pre-shot frame
   - Follow-through frame

2. **Calculate Metrics:**
   - Uses `loop3DAngles[]` array
   - Calculates release height using new method
   - Stores coordinate values globally: `window.releaseHeightCoords`

3. **Update UI:**
   - Calls `updateIdealFormUI(metrics)`
   - Displays metrics in Ideal Form results section

---

## 11. DISPLAY RESULTS

### Function: `updatePlayerMatchUI(matches)`
**File:** `shotsync/index.html`  
**Line:** ~11405

**Purpose:** Display comparison results in UI

**Process:**

1. **Display Top Matches:**
   - Shows top 5 player matches
   - Displays similarity scores
   - Shows expandable details for each match

2. **Display Metric Details:**
   - When user clicks expand button
   - Shows user value vs benchmark value
   - Displays coordinate values (for release_height, foot_angle)
   - Shows calculation descriptions

3. **Display Shared Traits & Differences:**
   - Shared traits: Metrics with >85% similarity
   - Differences: Metrics with <70% similarity

---

## DATA STRUCTURES

### Key Global Variables:
- `loop3DPoses[]` - Array of landmark arrays (33 points per frame)
- `loop3DStages[]` - Array of stage strings
- `loop3DAngles[]` - Array of angle objects per frame
- `window.releaseHeightCoords` - Coordinate values for release height
- `window.topPlayerMatches[]` - Top 5 comparison matches

### Landmark Format:
```javascript
{
    x: 0.0-1.0,        // Normalized X coordinate
    y: 0.0-1.0,        // Normalized Y coordinate (0 = top, 1 = bottom)
    z: float,          // Normalized depth
    visibility: 0.0-1.0 // Confidence score
}
```

### Angle Object Format:
```javascript
{
    right_knee: degrees,
    left_knee: degrees,
    right_elbow: degrees,
    left_elbow: degrees,
    wrist_snap: degrees,
    elbow_flare: degrees,
    trunk_lean: degrees,
    foot_angle: degrees,
    shoulder_angle: degrees,
    foot_alignment: degrees,
    release_height: null, // Calculated separately
    foot_angle_coords: { ... }, // For display
    release_height_coords: { ... } // For display
}
```

---

## SUMMARY FLOW

```
1. User uploads video
   ↓
2. Video loads → showAnalysisOptions()
   ↓
3. generateLoop(true) - Extract landmarks & angles
   ↓
4. MediaPipe Pose processes each frame
   ↓
5. pose.onResults() receives landmarks
   ↓
6. extractAnglesFromLandmarks() calculates angles
   ↓
7. Data stored in loop3DPoses[], loop3DStages[], loop3DAngles[]
   ↓
8. User clicks "Player Match" or "Ideal Form"
   ↓
9a. Player Match:
    → runPlayerMatchComparison()
    → convertToUserData()
    → compareWithAllBenchmarks()
    → extractMetricsFromData() (for user & benchmarks)
    → compareDetailedMetrics()
    → updatePlayerMatchUI()
    
9b. Ideal Form:
    → calculateIdealFormMetrics()
    → updateIdealFormUI()
```

---

## KEY FUNCTIONS REFERENCE

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `generateLoop()` | shotsync/index.html | ~12999 | Process video, extract landmarks |
| `extractAnglesFromLandmarks()` | shotsync/index.html | ~12788 | Calculate all joint angles |
| `calculateAngle()` | shotsync/index.html | ~12759 | Calculate angle between 3 points |
| `extractMetricsFromData()` | tool/app.js | ~2749 | Extract metrics from frame data |
| `compareDetailedMetrics()` | tool/app.js | ~2520 | Compare user vs benchmark metrics |
| `compareWithAllBenchmarks()` | tool/app.js | ~3199 | Compare with all benchmark files |
| `calculateIdealFormMetrics()` | shotsync/index.html | ~11942 | Calculate Ideal Form metrics |
| `runPlayerMatchComparison()` | shotsync/index.html | ~10784 | Run player match comparison |
| `updatePlayerMatchUI()` | shotsync/index.html | ~11405 | Display comparison results |
