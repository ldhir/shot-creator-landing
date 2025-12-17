# Yellow Triangle Coordinate Calculation Logic

## Overview
The yellow triangle is formed by drawing lines from the rim's bottom corners to the player's shoulder center. This creates a visual triangle that helps calculate the player's position relative to the hoop.

## Triangle Vertices
1. **Rim Bottom-Left Corner**: `(rim_bottom_left_x, rim_bottom_left_y)`
2. **Rim Bottom-Right Corner**: `(rim_bottom_right_x, rim_bottom_right_y)`
3. **Shoulder Center**: `(shoulder_x, shoulder_y)` - midpoint between left and right shoulders

## Visual Representation
```
        Shoulder Center (shoulder_x, shoulder_y)
              /\
             /  \
            /    \
           /      \
          /        \
         /          \
        /            \
       /              \
      /                \
     /                  \
    /____________________\
Rim Bottom-Left    Rim Bottom-Right
```

## Code Location
File: `shot_detector_web_simple.py`
Lines: 352-385

## Step-by-Step Calculation

### 1. Draw Yellow Lines (Visual)
```python
# Draw lines from rim bottom corners to shoulder center
cv2.line(self.frame, self.rim_bottom_left, 
        (int(shoulder_x), int(shoulder_y)), (0, 255, 255), 2)  # Yellow color
cv2.line(self.frame, self.rim_bottom_right, 
        (int(shoulder_x), int(shoulder_y)), (0, 255, 255), 2)  # Yellow color
```

### 2. Calculate Distances (Triangle Sides)
```python
# Distance from rim bottom-left corner to shoulder
dist_left = math.sqrt(
    (shoulder_x - self.rim_bottom_left[0])**2 + 
    (shoulder_y - self.rim_bottom_left[1])**2
)

# Distance from rim bottom-right corner to shoulder
dist_right = math.sqrt(
    (shoulder_x - self.rim_bottom_right[0])**2 + 
    (shoulder_y - self.rim_bottom_right[1])**2
)

# Use average distance for more accuracy
distance_pixels = (dist_left + dist_right) / 2.0
```

### 3. Convert Pixels to Real-World Distance
```python
# Use rim width as reference scale
# Rim is 1.5 feet wide (standard basketball rim diameter)
pixels_per_foot = rim_width_pixels / 1.5

# Convert pixel distance to feet
distance_feet = distance_pixels / pixels_per_foot

# Store for shot detection
self.player_distance_from_rim_feet = distance_feet
```

### 4. Calculate Lateral Position
```python
# Rim bottom center (midpoint between corners)
rim_bottom_center_x = (rim_bottom_left_x + rim_bottom_right_x) / 2
rim_bottom_center_y = (rim_bottom_left_y + rim_bottom_right_y) / 2

# Lateral offset from rim center to ball release position
dx_pixels = player_x - rim_bottom_center_x

# Convert to feet using rim width scale
lateral_distance_feet = abs(dx_pixels) / pixels_per_foot
```

### 5. Map to Court Coordinates (Normalized 0-1)
```python
# X coordinate: Lateral position
# Normalize: -25 feet (left) to +25 feet (right) -> 0 to 1
if dx_pixels >= 0:
    normalized_x = 0.5 + min(lateral_distance_feet / 25.0, 0.5)
else:
    normalized_x = 0.5 - min(lateral_distance_feet / 25.0, 0.5)

normalized_x = max(0.0, min(1.0, normalized_x))

# Y coordinate: Distance from hoop
# Normalize: 0 feet (hoop) to 23.75 feet (3pt line) -> 0 to 1
court_y = min(distance_feet / 23.75, 1.0)
court_y = max(0.0, min(1.0, court_y))
```

## Key Assumptions
1. **Rim Width**: 1.5 feet (standard basketball rim diameter)
2. **Court Width**: 50 feet total (25 feet each side from center)
3. **3-Point Line**: 23.75 feet from hoop
4. **Scale Reference**: Rim width in pixels is used to convert all measurements to feet

## Current Status
**Note**: This yellow triangle method is currently stored (`player_distance_from_rim_feet`) but the actual shot detection uses the **MediaPipe 3D + Rim Scaling** method instead (lines 624-692). The yellow triangle calculation serves as a fallback method.

## Why Use Triangle Method?
- **Visual Reference**: The yellow lines provide visual feedback
- **Simple Geometry**: Uses basic distance calculations
- **Rim as Scale**: Uses known rim width (1.5 feet) as a scale reference
- **Camera-Agnostic**: Works regardless of camera angle (as long as rim width is detected)

## Limitations
- Requires both rim corners and shoulders to be visible
- Assumes rim width is accurately detected
- 2D pixel-based calculation (doesn't account for depth/perspective as well as 3D methods)

