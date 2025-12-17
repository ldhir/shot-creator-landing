# How Makes and Misses Are Calculated

## Overview
The algorithm uses **three different methods** to determine if a shot was a make or miss. If **any** of these methods indicates a make, the shot is counted as a **MAKE**. Otherwise, it's a **MISS**.

## Step 1: Shot Detection
Before scoring, the algorithm first detects that a shot occurred:
- **"Up" detection**: Ball is detected in the area above/around the backboard
- **"Down" detection**: Ball is detected below the hoop
- **Shot attempt**: Only counted if ball goes from "up" → "down" in that order

## Step 2: Scoring (Three Methods)

### Method 1: Direct Rim Passage Check
**What it does**: Checks if the ball's center passed through the rim area during the shot.

**How it works**:
1. Defines a rim area:
   - Horizontal: Rim center ± 0.5 × rim width
   - Vertical: From 0.3×rim_height above rim to 0.5×rim_height below rim
2. Checks if any ball position during the shot was inside this area
3. If yes → **MAKE**

**Example**: If ball center was at (x=500, y=300) and rim center is (x=500, y=310), and ball passed through the rim zone → MAKE

---

### Method 2: Trajectory Prediction (Original Method, Improved)
**What it does**: Predicts where the ball would cross the rim height using linear regression.

**How it works**:
1. Finds all ball positions that are:
   - Within 2× hoop width horizontally from rim center
   - Both above and below the rim
2. Uses **linear regression** (line fitting) on these points to create a trajectory
3. Calculates where this line would cross the rim height
4. Checks if predicted crossing point is within 0.6× rim width of rim center
5. If yes → **MAKE**

**Example**:
- Ball positions: (480, 250), (490, 280), (500, 310), (510, 340)
- Fits a line through these points
- Predicts: at rim height (y=310), ball would be at x=500
- Rim center is at x=500 → **MAKE**

---

### Method 3: Proximity Check
**What it does**: Simple check - if ball got very close to rim center, it's likely a make.

**How it works**:
1. Finds the minimum distance from any ball position to the rim center
2. If minimum distance < 0.4 × rim width → **MAKE**

**Example**: 
- Rim center: (500, 310), rim width: 50 pixels
- Ball got as close as 15 pixels from rim center
- 15 < (0.4 × 50 = 20) → **MAKE**

---

## Important Details

### Rim Position Calculation
- **Rim center X**: Hoop center X coordinate
- **Rim center Y**: Hoop center Y - 0.5 × hoop height (rim is at top of hoop)
- **Rim width**: Uses the detected hoop width

### Ball Position Data
Each ball position contains:
- `(x, y)`: Center coordinates
- `frame_number`: When it was detected
- `width, height`: Ball bounding box size
- `confidence`: Detection confidence

### Why Three Methods?
1. **Method 1** catches obvious makes (ball clearly went through)
2. **Method 2** handles cases where ball detection is intermittent (uses trajectory)
3. **Method 3** catches makes where ball was very close but detection missed the exact rim passage

### Current Behavior
- If **ANY** method returns True → **MAKE**
- If **ALL** methods return False → **MISS**

---

## Potential Issues

### Why Shots Might Be Misclassified

1. **Insufficient ball detections**: 
   - If ball is only detected 1-2 times during shot, trajectory prediction fails
   - **Fix**: Need at least 2-3 ball positions for scoring

2. **Ball detection gaps**:
   - If ball disappears mid-flight, trajectory might be wrong
   - **Fix**: Method 1 and 3 help compensate

3. **Rim position accuracy**:
   - If hoop detection is off, rim center calculation is wrong
   - **Fix**: Hoop detection needs to be stable

4. **Timing issues**:
   - If "up" and "down" are detected too early/late, wrong ball positions are used
   - **Fix**: Current code filters to only use positions between up_frame and down_frame

---

## Visualization

```
Rim Area (Method 1):
     ┌─────────────┐
     │             │  ← 0.3×height above rim
─────┼─────────────┼─────  ← Rim center Y
     │   RIM       │
     │   AREA      │
     │             │  ← 0.5×height below rim
     └─────────────┘
   ← 0.5×width → ← 0.5×width →

Trajectory (Method 2):
     Ball positions: ●───●───●───●
                          │
                          │ (predicted line)
                          ↓
                    ┌─────┼─────┐
                    │     ●     │  ← Predicted crossing point
                    └───────────┘
                    (rim center)
```

---

## Code Location
- **Improved scoring**: `shot_detector_web_simple.py` → `improved_score()` method (line ~236)
- **Original scoring**: `utils.py` → `score()` function (line 16)
- **Shot detection**: `shot_detector_web_simple.py` → `shot_detection()` method (line ~169)

