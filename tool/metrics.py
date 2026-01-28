"""
Centralized metric calculations for shot analysis.

All angle/metric logic lives here. Ported from:
  - shotsync/index.html  lines 10400-10577  (client-side JS)
  - tool/app.py           lines 137-158      (server-side Python)

Every function expects landmarks as a list/array where each element is
an (x, y, z) triple (or an object with .x, .y, .z). Indices follow
the MediaPipe Pose model (33 landmarks).
"""

import math
import numpy as np


# ======================== LANDMARK INDICES (MediaPipe Pose) ========================

LANDMARKS = {
    'nose':             0,
    'left_eye_inner':   1,
    'left_eye':         2,
    'left_eye_outer':   3,
    'right_eye_inner':  4,
    'right_eye':        5,
    'right_eye_outer':  6,
    'left_ear':         7,
    'right_ear':        8,
    'mouth_left':       9,
    'mouth_right':      10,
    'left_shoulder':    11,
    'right_shoulder':   12,
    'left_elbow':       13,
    'right_elbow':      14,
    'left_wrist':       15,
    'right_wrist':      16,
    'left_pinky':       17,
    'right_pinky':      18,
    'left_index':       19,
    'right_index':      20,
    'left_thumb':       21,
    'right_thumb':      22,
    'left_hip':         23,
    'right_hip':        24,
    'left_knee':        25,
    'right_knee':       26,
    'left_ankle':       27,
    'right_ankle':      28,
    'left_heel':        29,
    'right_heel':       30,
    'left_foot_index':  31,
    'right_foot_index': 32,
}


# ======================== LOW-LEVEL HELPERS ========================

def _to_array(point):
    """Convert a landmark to a numpy array.

    Accepts:
      - numpy array (returned as-is)
      - list/tuple of [x, y, z] or [x, y]
      - object with .x, .y, .z attributes (raw MediaPipe landmark)

    Returns np.ndarray or None if the point is None or all-NaN.
    """
    if point is None:
        return None
    if isinstance(point, np.ndarray):
        if np.all(np.isnan(point)):
            return None
        return point
    if hasattr(point, 'x') and hasattr(point, 'y'):
        # Raw MediaPipe landmark object
        return np.array([point.x, point.y, getattr(point, 'z', 0.0)])
    if isinstance(point, (list, tuple)):
        arr = np.array(point, dtype=float)
        if np.all(np.isnan(arr)):
            return None
        return arr
    return None


def _get_xy(point):
    """Extract (x, y) from a landmark as plain floats.

    Works with numpy arrays, lists, and MediaPipe landmark objects.
    Returns (x, y) tuple or None.
    """
    if point is None:
        return None
    if isinstance(point, np.ndarray):
        if len(point) < 2 or np.isnan(point[0]) or np.isnan(point[1]):
            return None
        return float(point[0]), float(point[1])
    if hasattr(point, 'x') and hasattr(point, 'y'):
        return float(point.x), float(point.y)
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        x, y = float(point[0]), float(point[1])
        if math.isnan(x) or math.isnan(y):
            return None
        return x, y
    return None


def _distance(a, b):
    """Euclidean distance between two landmark points.

    Returns float or None if either point is invalid.
    """
    a = _to_array(a)
    b = _to_array(b)
    if a is None or b is None:
        return None
    # Use only the dimensions present in both
    min_len = min(len(a), len(b))
    return float(np.linalg.norm(a[:min_len] - b[:min_len]))


# ======================== CORE ANGLE CALCULATION ========================

def calculate_angle(a, b, c):
    """Compute the angle (in degrees) at point *b* formed by rays b→a and b→c.

    This is the 3D dot-product formula:
        angle = arccos( (BA · BC) / (|BA| × |BC|) )

    Ported from:
      - tool/app.py:147  ``calculate_3d_angle``
      - shotsync/index.html  ``calculateAngle``

    Parameters
    ----------
    a, b, c : array-like, MediaPipe landmark, or None
        Three points. *b* is the vertex.

    Returns
    -------
    float or None
        Angle in degrees [0, 180], or None if any point is invalid.
    """
    a = _to_array(a)
    b = _to_array(b)
    c = _to_array(c)
    if a is None or b is None or c is None:
        return None

    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-5:
        return None
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


# ======================== INDIVIDUAL METRIC FUNCTIONS ========================

def calculate_elbow_flare(landmarks):
    """Elbow flare angle — how far the shooting elbow drifts out from the body.

    Measures the angle at the right shoulder between the right elbow and
    right hip.  Higher value = more flare.

    Ported from shotsync/index.html:10479
        calculateAngle(lm[14], lm[12], lm[24])

    Landmarks used:
        14 (right elbow) → 12 (right shoulder) → 24 (right hip)

    Returns
    -------
    float or None
        Angle in degrees, or None if landmarks are missing.
    """
    right_elbow    = landmarks[LANDMARKS['right_elbow']]
    right_shoulder = landmarks[LANDMARKS['right_shoulder']]
    right_hip      = landmarks[LANDMARKS['right_hip']]
    return calculate_angle(right_elbow, right_shoulder, right_hip)


def calculate_trunk_lean(landmarks):
    """Forward/backward lean of the torso relative to vertical.

    Uses atan2 of the horizontal displacement (shoulder − hip) over the
    vertical displacement (hip − shoulder) so that 0° = perfectly upright,
    positive = leaning forward, negative = leaning backward.

    Ported from shotsync/index.html:10483-10489
        Math.atan2(shoulderX - hipX, hipY - shoulderY) * (180 / PI)

    Landmarks used:
        12 (right shoulder), 24 (right hip)

    Returns
    -------
    float or None
        Angle in degrees, or None if landmarks are missing.
    """
    shoulder_xy = _get_xy(landmarks[LANDMARKS['right_shoulder']])
    hip_xy      = _get_xy(landmarks[LANDMARKS['right_hip']])
    if shoulder_xy is None or hip_xy is None:
        return None
    shoulder_x, shoulder_y = shoulder_xy
    hip_x, hip_y = hip_xy
    return float(math.atan2(shoulder_x - hip_x, hip_y - shoulder_y) * (180.0 / math.pi))


def calculate_knee_bend(landmarks):
    """Right knee joint angle.

    Measures the angle at the right knee between the hip and ankle.
    Lower value = deeper bend (more loaded legs).

    Ported from shotsync/index.html:10493
        calculateAngle(lm[24], lm[26], lm[28])

    Landmarks used:
        24 (right hip) → 26 (right knee) → 28 (right ankle)

    Returns
    -------
    float or None
        Angle in degrees [0, 180], or None if landmarks are missing.
    """
    right_hip   = landmarks[LANDMARKS['right_hip']]
    right_knee  = landmarks[LANDMARKS['right_knee']]
    right_ankle = landmarks[LANDMARKS['right_ankle']]
    return calculate_angle(right_hip, right_knee, right_ankle)


def calculate_elbow_extension(landmarks):
    """How straight the shooting arm is at release.

    Measures the angle at the right elbow between the shoulder and wrist.
    180° = fully extended.

    Ported from shotsync/index.html:10503
        calculateAngle(lm[12], lm[14], lm[16])

    Landmarks used:
        12 (right shoulder) → 14 (right elbow) → 16 (right wrist)

    Returns
    -------
    float or None
        Angle in degrees [0, 180], or None if landmarks are missing.
    """
    right_shoulder = landmarks[LANDMARKS['right_shoulder']]
    right_elbow    = landmarks[LANDMARKS['right_elbow']]
    right_wrist    = landmarks[LANDMARKS['right_wrist']]
    return calculate_angle(right_shoulder, right_elbow, right_wrist)


def calculate_wrist_snap(pre_shot_wrist_angle, follow_through_wrist_angle):
    """Change in wrist angle between pre-shot and follow-through.

    A larger value means more wrist snap (more backspin on the ball).
    Returns the absolute difference.

    Ported from shotsync/index.html:10547-10550
        Math.abs(followThroughWrist - preShotWrist)

    Also matches tool/shot_stage_tf.py:1290
        user_wrist_snap = release_wrist - follow_through_wrist

    Parameters
    ----------
    pre_shot_wrist_angle : float or None
        Wrist angle (elbow→wrist→index) in the last pre-shot frame.
    follow_through_wrist_angle : float or None
        Wrist angle in the first follow-through frame.

    Returns
    -------
    float or None
        Absolute angle change in degrees, or None if either input is None.
    """
    if pre_shot_wrist_angle is None or follow_through_wrist_angle is None:
        return None
    return abs(follow_through_wrist_angle - pre_shot_wrist_angle)


def calculate_foot_alignment(landmarks):
    """How much the feet are rotated relative to the shoulders.

    Computes the angle of the ankle line (left ankle → right ankle) and
    subtracts the angle of the shoulder line (left shoulder → right shoulder).
    0° = feet perfectly squared to shoulders.

    Ported from shotsync/index.html:10519-10528
        ankleAngle  = atan2(lm[28].y - lm[27].y, lm[28].x - lm[27].x)
        shoulderAngle = atan2(lm[12].y - lm[11].y, lm[12].x - lm[11].x)
        footAlignment = ankleAngle - shoulderAngle

    Landmarks used:
        27 (left ankle), 28 (right ankle), 11 (left shoulder), 12 (right shoulder)

    Returns
    -------
    float or None
        Angle difference in degrees, or None if landmarks are missing.
    """
    left_ankle_xy  = _get_xy(landmarks[LANDMARKS['left_ankle']])
    right_ankle_xy = _get_xy(landmarks[LANDMARKS['right_ankle']])
    left_shoulder_xy  = _get_xy(landmarks[LANDMARKS['left_shoulder']])
    right_shoulder_xy = _get_xy(landmarks[LANDMARKS['right_shoulder']])

    if any(v is None for v in [left_ankle_xy, right_ankle_xy, left_shoulder_xy, right_shoulder_xy]):
        return None

    ankle_angle = math.atan2(
        right_ankle_xy[1] - left_ankle_xy[1],
        right_ankle_xy[0] - left_ankle_xy[0],
    ) * (180.0 / math.pi)

    shoulder_angle = math.atan2(
        right_shoulder_xy[1] - left_shoulder_xy[1],
        right_shoulder_xy[0] - left_shoulder_xy[0],
    ) * (180.0 / math.pi)

    return float(ankle_angle - shoulder_angle)


def calculate_foot_stance(landmarks):
    """Foot stance width as a ratio of ankle distance to shoulder distance.

    A value of 1.0 means ankles are exactly shoulder-width apart.

    Ported from shotsync/index.html:10531-10536
        ankleDistance / shoulderDistance

    Landmarks used:
        27 (left ankle), 28 (right ankle), 11 (left shoulder), 12 (right shoulder)

    Returns
    -------
    float or None
        Ratio, or None if landmarks are missing or shoulder distance is zero.
    """
    ankle_dist   = _distance(landmarks[LANDMARKS['left_ankle']],
                             landmarks[LANDMARKS['right_ankle']])
    shoulder_dist = _distance(landmarks[LANDMARKS['left_shoulder']],
                              landmarks[LANDMARKS['right_shoulder']])
    if ankle_dist is None or shoulder_dist is None or shoulder_dist < 1e-5:
        return None
    return float(ankle_dist / shoulder_dist)


def calculate_release_height(landmarks):
    """Wrist Y position at release, inverted so higher = better.

    In MediaPipe normalized coordinates, Y=0 is top of frame and Y=1 is
    bottom, so we return (1 - wrist_y).

    Ported from shotsync/index.html:10513-10515
        1 - wristY

    Landmarks used:
        16 (right wrist)

    Returns
    -------
    float or None
        Value in [0, 1] where higher = wrist is physically higher in frame.
        None if the landmark is missing.
    """
    wrist_xy = _get_xy(landmarks[LANDMARKS['right_wrist']])
    if wrist_xy is None:
        return None
    return float(1.0 - wrist_xy[1])


# ======================== ORIGINAL app.py ANGLES ========================
# These three are already computed in tool/app.py:231-240 and sent to the
# frontend as elbow_angle, wrist_angle, arm_angle.  They are kept here so
# every angle calculation lives in one place.

def calculate_elbow_angle(landmarks):
    """Shooting arm elbow bend (same as app.py elbow_angle).

    Ported from tool/app.py:236
        calculate_3d_angle(right_shoulder, right_elbow, right_wrist)

    Landmarks used:
        12 (right shoulder) → 14 (right elbow) → 16 (right wrist)

    Note: This is the same geometric calculation as ``calculate_elbow_extension``
    (both measure shoulder→elbow→wrist). They are kept as separate functions
    because the original codebase uses them in different contexts: app.py
    records ``elbow_angle`` per-frame during video processing, while
    shotsync/index.html computes ``elbowExtension`` as the max across frames.

    Returns
    -------
    float or None
    """
    right_shoulder = landmarks[LANDMARKS['right_shoulder']]
    right_elbow    = landmarks[LANDMARKS['right_elbow']]
    right_wrist    = landmarks[LANDMARKS['right_wrist']]
    return calculate_angle(right_shoulder, right_elbow, right_wrist)


def calculate_wrist_angle(landmarks):
    """Wrist flexion angle (same as app.py wrist_angle).

    Ported from tool/app.py:237-238
        calculate_3d_angle(right_elbow, right_wrist, right_index)

    Landmarks used:
        14 (right elbow) → 16 (right wrist) → 20 (right index finger)

    Returns
    -------
    float or None
    """
    right_elbow = landmarks[LANDMARKS['right_elbow']]
    right_wrist = landmarks[LANDMARKS['right_wrist']]
    right_index = landmarks[LANDMARKS['right_index']]
    return calculate_angle(right_elbow, right_wrist, right_index)


def calculate_arm_angle(landmarks):
    """Arm raise angle across the shoulders (same as app.py arm_angle).

    Ported from tool/app.py:239-240
        calculate_3d_angle(left_shoulder, right_shoulder, right_elbow)

    Landmarks used:
        11 (left shoulder) → 12 (right shoulder) → 14 (right elbow)

    Returns
    -------
    float or None
    """
    left_shoulder  = landmarks[LANDMARKS['left_shoulder']]
    right_shoulder = landmarks[LANDMARKS['right_shoulder']]
    right_elbow    = landmarks[LANDMARKS['right_elbow']]
    return calculate_angle(left_shoulder, right_shoulder, right_elbow)


# ======================== BATCH: ALL SINGLE-FRAME METRICS ========================

def calculate_all_single_frame_metrics(landmarks):
    """Compute every metric that can be derived from a single frame of landmarks.

    Parameters
    ----------
    landmarks : list
        A list of at least 29 landmark points (indexed 0-28). Each element
        can be a numpy array [x, y, z], a list/tuple, or a MediaPipe
        landmark object with .x, .y, .z attributes.

    Returns
    -------
    dict
        Keys are snake_case metric names, values are floats or None.
        ``wrist_snap`` is always None here because it requires two frames.

    Example
    -------
    >>> m = calculate_all_single_frame_metrics(frame.landmarks)
    >>> m['elbow_flare']   # 42.3 degrees
    >>> m['trunk_lean']    # 2.1 degrees
    """
    return {
        # New metrics (ported from shotsync/index.html)
        'elbow_flare':      calculate_elbow_flare(landmarks),
        'trunk_lean':       calculate_trunk_lean(landmarks),
        'knee_bend':        calculate_knee_bend(landmarks),
        'elbow_extension':  calculate_elbow_extension(landmarks),
        'release_height':   calculate_release_height(landmarks),
        'foot_alignment':   calculate_foot_alignment(landmarks),
        'foot_stance':      calculate_foot_stance(landmarks),

        # wrist_snap requires two frames (pre-shot and follow-through),
        # so it cannot be computed from a single frame.
        'wrist_snap':       None,

        # Original app.py angles (kept for backwards compatibility)
        'elbow_angle':      calculate_elbow_angle(landmarks),
        'wrist_angle':      calculate_wrist_angle(landmarks),
        'arm_angle':        calculate_arm_angle(landmarks),
    }
