from flask import Flask, request, jsonify, send_from_directory, send_file, Response, abort
from flask_cors import CORS
import cv2
import numpy as np
import time
import base64
import io
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
import boto3
from botocore.exceptions import ClientError
import requests
from collections import defaultdict
import uuid
import mimetypes
import re
import bisect

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install it with: pip install mediapipe")

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: fastdtw not available. Install it with: pip install fastdtw")

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("Warning: roboflow not available. Install it with: pip install roboflow")

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("Warning: supervision not available. Install it with: pip install supervision")

# Direct integration of AI-Basketball-Shot-Detection-Tracker
# Use it EXACTLY as it runs in the other window - import the Flask app and detector directly

try:
    import sys
    import threading
    # Add AI-Basketball-Shot-Detection-Tracker folder to path
    tracker_folder = os.path.join(os.path.dirname(__file__), 'AI-Basketball-Shot-Detection-Tracker')
    if os.path.exists(tracker_folder):
        sys.path.insert(0, tracker_folder)
        print("=" * 60)
        print("AI-Basketball-Shot-Detection-Tracker Direct Integration")
        print("=" * 60)
        print(f"✓ Found AI-Basketball-Shot-Detection-Tracker folder at: {tracker_folder}")
        
        # Import the original Flask app and detector - use EXACTLY as written
        import shot_detector_web_simple as tracker_module
        from shot_detector_web_simple import ShotDetectorWeb
        
        print("✓ Successfully imported from AI-Basketball-Shot-Detection-Tracker")
        print("=" * 60)
        
        YOLOV8_AVAILABLE = True
        
    else:
        YOLOV8_AVAILABLE = False
        print(f"✗ Warning: AI-Basketball-Shot-Detection-Tracker folder not found at {tracker_folder}")
            
except ImportError as e:
    YOLOV8_AVAILABLE = False
    print(f"Warning: Could not import from AI-Basketball-Shot-Detection-Tracker: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    YOLOV8_AVAILABLE = False
    print(f"Warning: Error importing AI-Basketball-Shot-Detection-Tracker: {e}")
    import traceback
    traceback.print_exc()

# Use the detector from the original module - initialize it the SAME way as original (line 1832)
# We'll initialize it when needed, but use the original module's detector variable

# Configure Flask to serve from project root for landing page
from pathlib import Path
project_root = Path(__file__).parent.parent  # Go up from tool/ to project root

app = Flask(__name__, static_folder='static')  # Default static folder for tool
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# Configure upload folder (EXACT COPY from original shot_detector_web_simple.py)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Note: Root static files will be handled after route definitions to avoid conflicts

# ====================== CONFIG / CONSTANTS ======================

# Email Configuration - Brevo (formerly Sendinblue) - EASIER SETUP!
# Set these environment variables:
BREVO_API_KEY = os.environ.get('BREVO_API_KEY', '')
BREVO_SENDER_EMAIL = os.environ.get('BREVO_SENDER_EMAIL', 'shotsyncbasketball@gmail.com')
BREVO_SENDER_NAME = os.environ.get('BREVO_SENDER_NAME', 'Shot Sync')

# Fallback to AWS SES if Brevo not configured
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')
FROM_EMAIL = os.environ.get('FROM_EMAIL', 'shotsyncbasketball@gmail.com')

# Initialize SES client (fallback)
ses_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and not BREVO_API_KEY:
    try:
        ses_client = boto3.client(
            'ses',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    except Exception as e:
        print(f"Warning: Could not initialize AWS SES client: {e}")
        ses_client = None

if MEDIAPIPE_AVAILABLE:
    mp_pose = mp.solutions.pose
    POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)
    pose = mp_pose.Pose(
        model_complexity=2,
        static_image_mode=False,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
else:
    mp_pose = None
    POSE_CONNECTIONS = []
    pose = None

# ====================== SHOTLAB MODEL CONFIG ======================

ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', '')

SHOTLAB_WORKSPACE = os.environ.get('SHOTLAB_WORKSPACE', '')
SHOTLAB_COURT_PROJECT = os.environ.get('SHOTLAB_COURT_PROJECT', 'basketball-court-detection-2')
SHOTLAB_COURT_VERSION = int(os.environ.get('SHOTLAB_COURT_VERSION', '0'))

SHOTLAB_BALL_WORKSPACE = os.environ.get('SHOTLAB_BALL_WORKSPACE', SHOTLAB_WORKSPACE)
SHOTLAB_BALL_PROJECT = os.environ.get('SHOTLAB_BALL_PROJECT', 'basketball-detection')
SHOTLAB_BALL_VERSION = int(os.environ.get('SHOTLAB_BALL_VERSION', '0'))

SHOTLAB_RIM_WORKSPACE = os.environ.get('SHOTLAB_RIM_WORKSPACE', SHOTLAB_WORKSPACE)
SHOTLAB_RIM_PROJECT = os.environ.get('SHOTLAB_RIM_PROJECT', 'rim-detection')
SHOTLAB_RIM_VERSION = int(os.environ.get('SHOTLAB_RIM_VERSION', '0'))

SHOTLAB_FRAME_STRIDE = int(os.environ.get('SHOTLAB_FRAME_STRIDE', '2'))
SHOTLAB_BALL_CONFIDENCE = int(os.environ.get('SHOTLAB_BALL_CONFIDENCE', '30'))
SHOTLAB_COURT_CONFIDENCE = int(os.environ.get('SHOTLAB_COURT_CONFIDENCE', '40'))
SHOTLAB_RIM_CONFIDENCE = int(os.environ.get('SHOTLAB_RIM_CONFIDENCE', '50'))
SHOTLAB_REQUEST_TIMEOUT = float(os.environ.get('SHOTLAB_REQUEST_TIMEOUT', '20'))
SHOTLAB_BALL_TIMEOUT_LIMIT = int(os.environ.get('SHOTLAB_BALL_TIMEOUT_LIMIT', '5'))
SHOTLAB_SHOT_MIN_GAP_SECONDS = float(os.environ.get('SHOTLAB_SHOT_MIN_GAP_SECONDS', '1.0'))
SHOTLAB_SHOT_MAX_DURATION_SECONDS = float(os.environ.get('SHOTLAB_SHOT_MAX_DURATION_SECONDS', '3.5'))
SHOTLAB_PRE_FOLLOW_MAX_GAP_SECONDS = float(os.environ.get('SHOTLAB_PRE_FOLLOW_MAX_GAP_SECONDS', '2.5'))
SHOTLAB_RELEASE_SEARCH_WINDOW_SECONDS = float(os.environ.get('SHOTLAB_RELEASE_SEARCH_WINDOW_SECONDS', '0.75'))
SHOTLAB_BALL_STRIDE = int(os.environ.get('SHOTLAB_BALL_STRIDE', str(SHOTLAB_FRAME_STRIDE)))
SHOTLAB_BALL_WINDOW_BEFORE_SECONDS = float(os.environ.get('SHOTLAB_BALL_WINDOW_BEFORE_SECONDS', '1.2'))
SHOTLAB_BALL_WINDOW_AFTER_SECONDS = float(os.environ.get('SHOTLAB_BALL_WINDOW_AFTER_SECONDS', '3.0'))
SHOTLAB_OUTCOME_WINDOW_AFTER_SECONDS = float(os.environ.get('SHOTLAB_OUTCOME_WINDOW_AFTER_SECONDS', '2.5'))
SHOTLAB_BALL_VALIDATE_PRE_SECONDS = float(os.environ.get('SHOTLAB_BALL_VALIDATE_PRE_SECONDS', '0.25'))
SHOTLAB_BALL_VALIDATE_POST_SECONDS = float(os.environ.get('SHOTLAB_BALL_VALIDATE_POST_SECONDS', '1.2'))
SHOTLAB_BALL_VALIDATE_MIN_POINTS = int(os.environ.get('SHOTLAB_BALL_VALIDATE_MIN_POINTS', '3'))
SHOTLAB_BALL_VALIDATE_REQUIRE_ARC = os.environ.get('SHOTLAB_BALL_VALIDATE_REQUIRE_ARC', '1') != '0'
SHOTLAB_BALL_VALIDATE_HAND_WINDOW_SECONDS = float(os.environ.get('SHOTLAB_BALL_VALIDATE_HAND_WINDOW_SECONDS', '0.2'))
SHOTLAB_BALL_VALIDATE_MAX_HAND_DIST_PX = float(os.environ.get('SHOTLAB_BALL_VALIDATE_MAX_HAND_DIST_PX', '140'))
SHOTLAB_SHOT_DEDUPE_SECONDS = float(os.environ.get('SHOTLAB_SHOT_DEDUPE_SECONDS', '2.0'))
SHOTLAB_REQUIRE_PRE_SHOT = os.environ.get('SHOTLAB_REQUIRE_PRE_SHOT', '0') != '0'
SHOTLAB_CLIP_OVERLAY = os.environ.get('SHOTLAB_CLIP_OVERLAY', '1') != '0'
SHOTLAB_CLIP_TRAIL_FRAMES = int(os.environ.get('SHOTLAB_CLIP_TRAIL_FRAMES', '14'))
SHOTLAB_CALIBRATION_SAMPLES = int(os.environ.get('SHOTLAB_CALIBRATION_SAMPLES', '18'))
SHOTLAB_CALIBRATION_MIN_KEYPOINTS = int(os.environ.get('SHOTLAB_CALIBRATION_MIN_KEYPOINTS', '4'))
SHOTLAB_RIM_MIN_CONFIDENCE = float(os.environ.get('SHOTLAB_RIM_MIN_CONFIDENCE', '0.15'))
SHOTLAB_ALLOW_FOLLOW_ONLY = os.environ.get('SHOTLAB_ALLOW_FOLLOW_ONLY', '0') != '0'
SHOTLAB_FOLLOW_ONLY_LOOKBACK_SECONDS = float(os.environ.get('SHOTLAB_FOLLOW_ONLY_LOOKBACK_SECONDS', '1.5'))
SHOTLAB_FOLLOW_SEGMENT_GAP_SECONDS = float(os.environ.get('SHOTLAB_FOLLOW_SEGMENT_GAP_SECONDS', '0.3'))
SHOTLAB_CLIP_BEFORE_SECONDS = float(os.environ.get('SHOTLAB_CLIP_BEFORE_SECONDS', '1.0'))
SHOTLAB_CLIP_AFTER_SECONDS = float(os.environ.get('SHOTLAB_CLIP_AFTER_SECONDS', '3.0'))
SHOTLAB_CLIP_MAX_DIM = int(os.environ.get('SHOTLAB_CLIP_MAX_DIM', '720'))
SHOTLAB_CLIP_PREGENERATE = os.environ.get('SHOTLAB_CLIP_PREGENERATE', '1') != '0'
SHOTLAB_USE_VELOCITY_CANDIDATES = os.environ.get('SHOTLAB_USE_VELOCITY_CANDIDATES', '1') != '0'

SHOTLAB_DEFAULT_BENCHMARK = float(os.environ.get('SHOTLAB_DEFAULT_BENCHMARK', '82'))
SHOTLAB_DEFAULT_STD = float(os.environ.get('SHOTLAB_DEFAULT_STD', '6'))
SHOTLAB_BENCHMARKS_FILE = os.environ.get(
    'SHOTLAB_BENCHMARKS_FILE',
    str(Path(__file__).parent / 'shotlab_benchmarks.json')
)
SHOTLAB_ZONE_BENCHMARKS = {
    'restricted_area': float(os.environ.get('SHOTLAB_BENCHMARK_RESTRICTED', '88')),
    'paint': float(os.environ.get('SHOTLAB_BENCHMARK_PAINT', '86')),
    'mid_range': float(os.environ.get('SHOTLAB_BENCHMARK_MID', '82')),
    'left_corner_3': float(os.environ.get('SHOTLAB_BENCHMARK_LC3', '78')),
    'right_corner_3': float(os.environ.get('SHOTLAB_BENCHMARK_RC3', '78')),
    'left_wing_3': float(os.environ.get('SHOTLAB_BENCHMARK_LW3', '80')),
    'right_wing_3': float(os.environ.get('SHOTLAB_BENCHMARK_RW3', '80')),
    'top_of_key_3': float(os.environ.get('SHOTLAB_BENCHMARK_TOK3', '79')),
    'left_baseline_2': float(os.environ.get('SHOTLAB_BENCHMARK_LB2', '83')),
    'right_baseline_2': float(os.environ.get('SHOTLAB_BENCHMARK_RB2', '83')),
    'left_wing_2': float(os.environ.get('SHOTLAB_BENCHMARK_LW2', '84')),
    'right_wing_2': float(os.environ.get('SHOTLAB_BENCHMARK_RW2', '84')),
    'unknown': float(os.environ.get('SHOTLAB_BENCHMARK_UNKNOWN', '80'))
}

shotlab_models_initialized = False
court_model = None
ball_model = None
rim_model = None
ball_tracker = None
shotlab_benchmark_cache = None
shotlab_status = {
    'stage': 'idle',
    'detail': '',
    'progress': 0.0,
    'updated_at': 0.0
}
roboflow_requests_patched = False
shotlab_sessions = {}

# ====================== POSE & ANGLE UTILS ======================

def get_3d_point(landmarks, index, width, height):
    """Extract (x, y, z) from Mediapipe landmarks. Returns None if not visible enough."""
    if index >= len(landmarks) or landmarks[index].visibility < 0.5:
        return None
    return np.array([
        landmarks[index].x * width,
        landmarks[index].y * height,
        landmarks[index].z
    ])

def calculate_3d_angle(a, b, c):
    """Compute angle at b formed by points a->b->c in 3D."""
    if a is None or b is None or c is None:
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-5:
        return None
    cosine_angle = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_arm_state(landmarks, width, height):
    """
    'pre_shot' => wrists close together below shoulders
    'follow_through' => wrist above shoulder
    'neutral' => default
    """
    def get_point(index, min_vis=0.35):
        if index >= len(landmarks) or landmarks[index].visibility < min_vis:
            return None
        return np.array([
            landmarks[index].x * width,
            landmarks[index].y * height,
            landmarks[index].z
        ])

    right_shoulder = get_point(12)
    right_elbow    = get_point(14)
    right_wrist    = get_point(16)
    left_shoulder  = get_point(11)
    left_elbow     = get_point(13)
    left_wrist     = get_point(15)
    left_hip       = get_point(23)
    right_hip      = get_point(24)

    if (right_wrist is not None and left_wrist is not None and
        left_hip is not None and right_hip is not None and
        (right_shoulder is not None or left_shoulder is not None)):
        waist_y = (left_hip[1] + right_hip[1]) / 2.0
        avg_wrist_y = (right_wrist[1] + left_wrist[1]) / 2.0
        dist_wrists = np.linalg.norm(right_wrist - left_wrist)
        shoulder_y = None
        if right_shoulder is not None and left_shoulder is not None:
            shoulder_y = min(right_shoulder[1], left_shoulder[1])
        elif right_shoulder is not None:
            shoulder_y = right_shoulder[1]
        elif left_shoulder is not None:
            shoulder_y = left_shoulder[1]
        if shoulder_y is not None and (dist_wrists < 0.14 * width and avg_wrist_y < waist_y
            and avg_wrist_y > shoulder_y + 0.03 * height):
            return "pre_shot"

    if right_wrist is not None and right_shoulder is not None:
        if right_shoulder[1] - right_wrist[1] > 0.05 * height:
            return "follow_through"
    if left_wrist is not None and left_shoulder is not None:
        if left_shoulder[1] - left_wrist[1] > 0.05 * height:
            return "follow_through"

    if right_shoulder is not None and right_wrist is not None:
        if right_wrist[1] > right_shoulder[1]:
            return "neutral"
    if left_shoulder is not None and left_wrist is not None:
        if left_wrist[1] > left_shoulder[1]:
            return "neutral"

    return "neutral"

# ====================== PROCESS VIDEO FRAMES ======================

def process_video_frames(frames_data):
    """
    Process video frames from base64 encoded images.
    Returns: shot_angles, landmark_frames
    """
    if not MEDIAPIPE_AVAILABLE:
        return [], []
    
    shot_angles = []
    landmark_frames = []
    
    previous_stage = "neutral"
    start_time = None
    last_print_time = time.time()
    recording_active = False
    seen_follow_through = False
    
    for frame_data in frames_data:
        # Decode base64 image
        frame_bytes = base64.b64decode(frame_data['image'].split(',')[1])
        nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        h, w, _ = frame.shape
        current_time = frame_data.get('timestamp', time.time())
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            state = get_arm_state(landmarks, w, h)
            
            # Compute angles
            right_shoulder = get_3d_point(landmarks, 12, w, h)
            right_elbow    = get_3d_point(landmarks, 14, w, h)
            right_wrist    = get_3d_point(landmarks, 16, w, h)
            
            elbow_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
            wrist_angle = calculate_3d_angle(right_elbow, right_wrist,
                                             get_3d_point(landmarks, 20, w, h))
            arm_angle   = calculate_3d_angle(get_3d_point(landmarks, 11, w, h),
                                             right_shoulder, right_elbow)
            
            # Store full body landmarks
            frame_landmarks_3d = np.full((33,3), np.nan, dtype=np.float32)
            for i in range(33):
                p = get_3d_point(landmarks, i, w, h)
                if p is not None:
                    frame_landmarks_3d[i] = p
            
            # Stage transitions
            if state != previous_stage:
                if state == "pre_shot" and not recording_active:
                    recording_active = True
                    seen_follow_through = False
                    start_time = current_time
                    shot_angles = []
                    landmark_frames = []
                    last_print_time = start_time
                elif state == "neutral" and recording_active and not seen_follow_through:
                    recording_active = False
                    seen_follow_through = False
                    start_time = None
                    shot_angles = []
                    landmark_frames = []
                elif state == "follow_through" and recording_active:
                    seen_follow_through = True
                elif state == "pre_shot" and recording_active and seen_follow_through:
                    elapsed = current_time - start_time
                    landmark_frames.append({
                        'time': float(elapsed),
                        'landmarks': frame_landmarks_3d.tolist()
                    })
                    break  # Shot completed
                
                previous_stage = state
            
            # Record frames while actively recording
            if recording_active:
                elapsed = current_time - start_time if start_time else 0.0
                landmark_frames.append({
                    'time': float(elapsed),
                    'landmarks': frame_landmarks_3d.tolist()
                })
                
                if state in ["pre_shot", "follow_through"]:
                    if current_time - last_print_time >= 0.1:
                        shot_angles.append({
                            'state': state,
                            'time': float(elapsed),
                            'elbow_angle': float(elbow_angle) if elbow_angle else None,
                            'wrist_angle': float(wrist_angle) if wrist_angle else None,
                            'arm_angle': float(arm_angle) if arm_angle else None
                        })
                        last_print_time = current_time
    
    return shot_angles, landmark_frames

# ====================== FORM ANALYSIS ======================

def compute_overall_form(e, w, a):
    angles = []
    if e is not None:
        angles.append(e)
    if w is not None:
        angles.append(w)
    if a is not None:
        angles.append(a)
    if len(angles) == 0:
        return None
    return sum(angles) / len(angles)

def extract_form_series(shot_data):
    times = []
    form_vals = []
    for entry in shot_data:
        t = entry['time']
        e = entry.get('elbow_angle')
        w = entry.get('wrist_angle')
        a = entry.get('arm_angle')
        measure = compute_overall_form(e, w, a)
        if measure is not None:
            times.append(t)
            form_vals.append(measure)
    return np.array(times), np.array(form_vals)

def compute_user_closeness(bench_form, user_form, path):
    alpha = 2.0
    user_map = {}
    for (i, j) in path:
        user_map.setdefault(j, []).append(i)
    
    user_closeness = np.zeros_like(user_form)
    for j in range(len(user_form)):
        if j in user_map:
            i_list = user_map[j]
            i_mid = i_list[len(i_list)//2]
            diff = abs(float(user_form[j]) - float(bench_form[i_mid]))
            score = max(0.0, min(100.0, 100.0 - alpha * diff))
        else:
            score = 100.0
        user_closeness[j] = score
    return user_closeness

def generate_feedback(benchmark_data, user_data, bench_times, user_times, user_closeness):
    """Generate actionable feedback comparing user shot to benchmark."""
    feedback = []
    
    avg_closeness = np.mean(user_closeness) if len(user_closeness) > 0 else 0
    feedback.append(f"Overall Score: {avg_closeness:.1f}%")
    
    if avg_closeness >= 90:
        feedback.append("Excellent form! Your shot closely matches the benchmark.")
    elif avg_closeness >= 75:
        feedback.append("Good form with room for improvement.")
    elif avg_closeness >= 60:
        feedback.append("Your form needs work. Focus on key areas below.")
    else:
        feedback.append("Significant differences detected. Review the specific feedback below.")
    
    return feedback

# ====================== SHOTLAB HELPERS ======================

def initialize_shotlab_models():
    """Initialize Roboflow models and tracking utilities for ShotLab."""
    global shotlab_models_initialized, court_model, ball_model, rim_model, ball_tracker
    if shotlab_models_initialized:
        return True, None
    if not ROBOFLOW_AVAILABLE:
        return False, "roboflow not available. Install it with: pip install roboflow"
    if not ROBOFLOW_API_KEY:
        return False, "ROBOFLOW_API_KEY not set"

    patch_roboflow_requests_timeout(SHOTLAB_REQUEST_TIMEOUT)

    def _resolve_version(project, requested_version, label):
        if requested_version and requested_version > 0:
            return requested_version
        try:
            versions = project.versions()
            if not versions:
                return None
            candidates = []
            for v in versions:
                if hasattr(v, 'version'):
                    candidates.append(int(v.version))
                elif isinstance(v, dict) and 'version' in v:
                    candidates.append(int(v['version']))
                elif isinstance(v, (int, str)) and str(v).isdigit():
                    candidates.append(int(v))
            return max(candidates) if candidates else None
        except Exception as e:
            print(f"ShotLab version resolve error ({label}): {e}")
            return None

    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)

        def _get_workspace(name, label):
            if name:
                return rf.workspace(name), None
            if hasattr(rf, "current_workspace") and rf.current_workspace:
                return rf.workspace(rf.current_workspace), None
            return None, f"Roboflow workspace not set for {label}. Set SHOTLAB_WORKSPACE or SHOTLAB_{label}_WORKSPACE."

        court_ws, err = _get_workspace(SHOTLAB_WORKSPACE, "COURT")
        if err:
            return False, err
        ball_ws, err = _get_workspace(SHOTLAB_BALL_WORKSPACE, "BALL")
        if err:
            return False, err
        rim_ws, err = _get_workspace(SHOTLAB_RIM_WORKSPACE, "RIM")
        if err:
            return False, err

        court_project = court_ws.project(SHOTLAB_COURT_PROJECT)
        ball_project = ball_ws.project(SHOTLAB_BALL_PROJECT)
        rim_project = rim_ws.project(SHOTLAB_RIM_PROJECT)

        court_version = _resolve_version(court_project, SHOTLAB_COURT_VERSION, "court")
        ball_version = _resolve_version(ball_project, SHOTLAB_BALL_VERSION, "ball")
        rim_version = _resolve_version(rim_project, SHOTLAB_RIM_VERSION, "rim")

        if not court_version:
            return False, "Could not determine court model version. Set SHOTLAB_COURT_VERSION."
        if not ball_version:
            return False, "Could not determine ball model version. Set SHOTLAB_BALL_VERSION."
        if not rim_version:
            return False, "Could not determine rim model version. Set SHOTLAB_RIM_VERSION."

        court_model = court_project.version(court_version).model
        ball_model = ball_project.version(ball_version).model
        rim_model = rim_project.version(rim_version).model

        if SUPERVISION_AVAILABLE:
            ball_tracker = sv.ByteTrack()

        shotlab_models_initialized = True
        return True, None
    except Exception as e:
        return False, str(e)

def detect_court_keypoints(frame):
    """Detect court keypoints via Roboflow model."""
    if court_model is None:
        return []
    try:
        result = court_model.predict(frame, confidence=SHOTLAB_COURT_CONFIDENCE).json()
        predictions = result.get('predictions', [])
        keypoints = []
        for prediction in predictions:
            if 'x' in prediction and 'y' in prediction:
                keypoints.append({
                    'class': prediction.get('class', ''),
                    'x': prediction['x'],
                    'y': prediction['y']
                })
        return keypoints
    except Exception as e:
        print(f"ShotLab court keypoint detection error: {e}")
        return []

def encode_frame_base64(frame):
    if frame is None:
        return None
    ok, buffer = cv2.imencode('.jpg', frame)
    if not ok:
        return None
    return base64.b64encode(buffer).decode('utf-8')

def sample_video_frames(video_path, num_samples):
    """Sample frames uniformly across a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return [], total_frames
    count = max(1, min(num_samples, total_frames))
    indices = np.linspace(0, total_frames - 1, count, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((int(idx), frame.copy()))
    cap.release()
    return frames, total_frames

def _find_keypoint_by_class(keypoints, class_names):
    for name in class_names:
        for kp in keypoints:
            if kp.get('class') == name:
                return kp
    return None

def _extract_court_corners(keypoints):
    """Extract baseline/opposite baseline corners in a stable order."""
    if not keypoints or len(keypoints) < 4:
        return None

    class_candidates = {
        'baseline_left': ['baseline_left', 'baseline-left', 'left_baseline', 'baseline_l', 'left_baseline_corner'],
        'baseline_right': ['baseline_right', 'baseline-right', 'right_baseline', 'baseline_r', 'right_baseline_corner'],
        'opposite_baseline_left': ['opposite_baseline_left', 'far_baseline_left', 'top_baseline_left', 'opposite-left'],
        'opposite_baseline_right': ['opposite_baseline_right', 'far_baseline_right', 'top_baseline_right', 'opposite-right']
    }

    bl = _find_keypoint_by_class(keypoints, class_candidates['baseline_left'])
    br = _find_keypoint_by_class(keypoints, class_candidates['baseline_right'])
    obl = _find_keypoint_by_class(keypoints, class_candidates['opposite_baseline_left'])
    obr = _find_keypoint_by_class(keypoints, class_candidates['opposite_baseline_right'])

    if bl and br and obl and obr:
        return np.float32([
            [bl['x'], bl['y']],
            [br['x'], br['y']],
            [obl['x'], obl['y']],
            [obr['x'], obr['y']]
        ])

    # Fallback: infer corners by y (baseline is typically closer to camera)
    pts = np.array([[kp['x'], kp['y']] for kp in keypoints], dtype=np.float32)
    if len(pts) < 4:
        return None
    order_y = np.argsort(pts[:, 1])
    top_two = pts[order_y[:2]]
    bottom_two = pts[order_y[-2:]]
    top_two = top_two[np.argsort(top_two[:, 0])]
    bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]
    return np.float32([
        [bottom_two[0][0], bottom_two[0][1]],
        [bottom_two[1][0], bottom_two[1][1]],
        [top_two[0][0], top_two[0][1]],
        [top_two[1][0], top_two[1][1]]
    ])

def get_perspective_transform(keypoints):
    """Compute perspective transform matrix from detected keypoints."""
    detected_points = _extract_court_corners(keypoints)
    if detected_points is None:
        return None
    court_points_real = np.float32([
        [0, 0],
        [50, 0],
        [0, 47],
        [50, 47]
    ])
    return cv2.getPerspectiveTransform(detected_points, court_points_real)

def detect_court_transform_multi(video_path, num_samples=SHOTLAB_CALIBRATION_SAMPLES):
    """Detect court keypoints across multiple frames and pick the best transform."""
    frames, total_frames = sample_video_frames(video_path, num_samples)
    best = None
    for frame_idx, frame in frames:
        keypoints = detect_court_keypoints(frame)
        transform = get_perspective_transform(keypoints)
        score = len(keypoints) + (10 if transform is not None else 0)
        if best is None or score > best['score']:
            best = {
                'frame_idx': frame_idx,
                'frame': frame,
                'keypoints': keypoints,
                'transform': transform,
                'score': score
            }
    if best is None:
        return None, [], None, {
            'samples': 0,
            'total_frames': total_frames,
            'best_keypoints': 0,
            'transform_available': False
        }
        return None, [], None, debug, None
    preview = encode_frame_base64(best['frame'])
    debug = {
        'samples': len(frames),
        'total_frames': total_frames,
        'best_frame': best['frame_idx'],
        'best_keypoints': len(best['keypoints']),
        'transform_available': best['transform'] is not None,
        'best_shape': list(best['frame'].shape[:2])
    }
    return best['transform'], best['keypoints'], preview, debug, best['frame'].shape

def manual_court_transform_from_request(form, frame_shape):
    """Compute a court transform from manual corner clicks (normalized coords)."""
    if frame_shape is None:
        return None
    try:
        points_json = form.get('court_points')
        if not points_json:
            return None
        points = json.loads(points_json)
        if not isinstance(points, list) or len(points) < 4:
            return None
        h, w = frame_shape[:2]
        detected_points = []
        for pt in points[:4]:
            x_norm = float(pt.get('x_norm'))
            y_norm = float(pt.get('y_norm'))
            detected_points.append([x_norm * w, y_norm * h])
        detected_points = np.float32(detected_points)
        court_points_real = np.float32([
            [0, 0],
            [50, 0],
            [0, 47],
            [50, 47]
        ])
        return cv2.getPerspectiveTransform(detected_points, court_points_real)
    except Exception:
        return None

def player_to_court_position(player_xy, transform_matrix):
    if player_xy is None or transform_matrix is None:
        return None
    point = np.array([[[player_xy[0], player_xy[1]]]], dtype=np.float32)
    court_pos = cv2.perspectiveTransform(point, transform_matrix)
    return court_pos[0][0]

def classify_shot_zone(court_x, court_y):
    """Classify shot into standard basketball zones (NBA half court)."""
    basket_x, basket_y = 25, 47
    distance = np.sqrt((court_x - basket_x) ** 2 + (court_y - basket_y) ** 2)
    is_three_pointer = distance > 23.75

    if distance < 5:
        return 'restricted_area'
    if court_x < 15:
        return 'left_corner_3' if is_three_pointer else 'left_baseline_2'
    if court_x > 35:
        return 'right_corner_3' if is_three_pointer else 'right_baseline_2'
    if 15 <= court_x <= 20:
        return 'left_wing_3' if is_three_pointer else 'left_wing_2'
    if 30 <= court_x <= 35:
        return 'right_wing_3' if is_three_pointer else 'right_wing_2'
    if is_three_pointer:
        return 'top_of_key_3'
    if distance > 15:
        return 'mid_range'
    return 'paint'

def load_shotlab_benchmarks():
    global shotlab_benchmark_cache
    if shotlab_benchmark_cache is not None:
        return shotlab_benchmark_cache
    try:
        if SHOTLAB_BENCHMARKS_FILE and os.path.exists(SHOTLAB_BENCHMARKS_FILE):
            with open(SHOTLAB_BENCHMARKS_FILE, 'r') as f:
                shotlab_benchmark_cache = json.load(f)
                return shotlab_benchmark_cache
    except Exception as e:
        print(f"ShotLab benchmark load error: {e}")
    shotlab_benchmark_cache = None
    return None

def get_zone_benchmark(zone):
    data = load_shotlab_benchmarks()
    if data and isinstance(data, dict):
        default = data.get('default', {})
        zones = data.get('zones', {})
        if zone in zones:
            entry = zones.get(zone, {})
            return float(entry.get('mean', SHOTLAB_DEFAULT_BENCHMARK)), float(entry.get('std', SHOTLAB_DEFAULT_STD))
        if 'unknown' in zones:
            entry = zones.get('unknown', {})
            return float(entry.get('mean', SHOTLAB_DEFAULT_BENCHMARK)), float(entry.get('std', SHOTLAB_DEFAULT_STD))
        if default:
            return float(default.get('mean', SHOTLAB_DEFAULT_BENCHMARK)), float(default.get('std', SHOTLAB_DEFAULT_STD))
    mean = SHOTLAB_ZONE_BENCHMARKS.get(zone, SHOTLAB_DEFAULT_BENCHMARK)
    return float(mean), SHOTLAB_DEFAULT_STD

def compute_shotsync_score(form_score, benchmark_mean, benchmark_std):
    """Convert form score (angle-based) into 0-100 similarity vs benchmark."""
    if form_score is None or benchmark_mean is None:
        return None
    std = benchmark_std if benchmark_std and benchmark_std > 0 else SHOTLAB_DEFAULT_STD
    diff = abs(float(form_score) - float(benchmark_mean))
    z = diff / std
    score = max(0.0, min(100.0, 100.0 - z * 20.0))
    return score

def update_shotlab_status(stage, detail=None, progress=None):
    shotlab_status['stage'] = stage
    if detail is not None:
        shotlab_status['detail'] = detail
    if progress is not None:
        shotlab_status['progress'] = float(progress)
    shotlab_status['updated_at'] = time.time()

def patch_roboflow_requests_timeout(timeout_seconds):
    """Patch Roboflow requests.post to apply a default timeout."""
    global roboflow_requests_patched
    if roboflow_requests_patched:
        return
    try:
        import roboflow.models.object_detection as rf_obj
        import roboflow.models.keypoint_detection as rf_kp
        import roboflow.models.instance_segmentation as rf_is
        import roboflow.models.semantic_segmentation as rf_ss
        import roboflow.models.classification as rf_cl
        import roboflow.models.inference as rf_inf

        modules = [rf_obj, rf_kp, rf_is, rf_ss, rf_cl, rf_inf]
        for module in modules:
            if hasattr(module, "requests"):
                original_post = module.requests.post
                def post_with_timeout(*args, _orig=original_post, **kwargs):
                    kwargs.setdefault("timeout", timeout_seconds)
                    return _orig(*args, **kwargs)
                module.requests.post = post_with_timeout

        roboflow_requests_patched = True
    except Exception as e:
        print(f"Roboflow timeout patch failed: {e}")

def get_player_foot_position_from_landmarks(landmarks, width, height):
    left_ankle = get_3d_point(landmarks, 27, width, height)
    right_ankle = get_3d_point(landmarks, 28, width, height)
    if left_ankle is None or right_ankle is None:
        return None
    return [
        float((left_ankle[0] + right_ankle[0]) / 2.0),
        float((left_ankle[1] + right_ankle[1]) / 2.0)
    ]

def get_right_wrist_point(landmarks, width, height):
    wrist = get_3d_point(landmarks, 16, width, height)
    if wrist is None:
        return None
    return [float(wrist[0]), float(wrist[1])]

def compute_release_angles(landmarks, width, height):
    right_shoulder = get_3d_point(landmarks, 12, width, height)
    right_elbow = get_3d_point(landmarks, 14, width, height)
    right_wrist = get_3d_point(landmarks, 16, width, height)
    right_index = get_3d_point(landmarks, 20, width, height)
    left_shoulder = get_3d_point(landmarks, 11, width, height)

    elbow_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
    wrist_angle = calculate_3d_angle(right_elbow, right_wrist, right_index)
    arm_angle = calculate_3d_angle(left_shoulder, right_shoulder, right_elbow)
    form_score = compute_overall_form(elbow_angle, wrist_angle, arm_angle)

    return {
        'elbow_angle': float(elbow_angle) if elbow_angle is not None else None,
        'wrist_angle': float(wrist_angle) if wrist_angle is not None else None,
        'arm_angle': float(arm_angle) if arm_angle is not None else None,
        'form_score': float(form_score) if form_score is not None else None
    }

def process_video_for_pose(video_path, frame_stride=1, progress_callback=None):
    """Process video for pose landmarks and shot state per frame."""
    if not MEDIAPIPE_AVAILABLE:
        return [], 0.0

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    pose_frames = []
    last_progress_update = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_stride > 1 and frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        h, w, _ = frame.shape
        timestamp = frame_idx / fps if fps else 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            state = get_arm_state(landmarks, w, h)
            foot_pos = get_player_foot_position_from_landmarks(landmarks, w, h)
            angles = compute_release_angles(landmarks, w, h)
            right_wrist = get_right_wrist_point(landmarks, w, h)
        else:
            state = "neutral"
            foot_pos = None
            angles = None
            right_wrist = None

        pose_frames.append({
            'frame_idx': frame_idx,
            'timestamp': float(timestamp),
            'state': state,
            'foot_pos': foot_pos,
            'angles': angles,
            'right_wrist': right_wrist
        })
        frame_idx += 1

        if progress_callback and total_frames > 0:
            progress = min(1.0, frame_idx / total_frames)
            if frame_idx % max(1, 10 * frame_stride) == 0 and progress != last_progress_update:
                last_progress_update = progress
                progress_callback(frame_idx, total_frames, progress)

    cap.release()
    return pose_frames, fps

def summarize_pose_states(pose_frames):
    counts = defaultdict(int)
    transitions = defaultdict(int)
    last_state = None
    for entry in pose_frames:
        state = entry.get('state', 'unknown')
        counts[state] += 1
        if last_state is not None and state != last_state:
            transitions[f"{last_state}->{state}"] += 1
        last_state = state
    return {
        'state_counts': dict(counts),
        'transitions': dict(transitions),
        'frames_with_pose': int(sum(counts.values()))
    }

def detect_shot_attempts_from_pose(pose_frames, fps):
    """Detect multiple shot attempts using state transitions with fallbacks and debug stats."""
    debug = {
        'follow_entries': 0,
        'pre_shot_entries': 0,
        'follow_segments': 0,
        'skipped_min_gap': 0,
        'skipped_no_pre_shot': 0,
        'follow_only_used': 0,
        'velocity_candidates': 0,
        'release_candidates': 0,
        'replaced_close_candidate': 0,
        'allow_follow_only': False
    }
    if not pose_frames or not fps:
        return [], debug

    min_gap_frames = max(1, int(SHOTLAB_SHOT_MIN_GAP_SECONDS * fps))
    max_duration_frames = max(1, int(SHOTLAB_SHOT_MAX_DURATION_SECONDS * fps))
    pre_follow_gap_frames = max(1, int(SHOTLAB_PRE_FOLLOW_MAX_GAP_SECONDS * fps))
    min_release_to_end_frames = max(1, int(0.3 * fps))
    follow_only_lookback_frames = max(1, int(SHOTLAB_FOLLOW_ONLY_LOOKBACK_SECONDS * fps))

    pre_shot_entries = [e for e in pose_frames if e.get('state') == 'pre_shot']
    follow_entries = [e for e in pose_frames if e.get('state') == 'follow_through']
    debug['pre_shot_entries'] = len(pre_shot_entries)
    debug['follow_entries'] = len(follow_entries)

    wrist_entries = [e for e in pose_frames if e.get('right_wrist') is not None]
    velocity_threshold = None
    wrist_velocity_map = {}
    if len(wrist_entries) >= 5:
        velocities = []
        for i in range(1, len(wrist_entries)):
            prev = wrist_entries[i - 1]
            curr = wrist_entries[i]
            dt = max(1e-4, curr['timestamp'] - prev['timestamp'])
            vy_up = -(curr['right_wrist'][1] - prev['right_wrist'][1]) / dt
            velocities.append(vy_up)
            wrist_velocity_map[curr['frame_idx']] = float(vy_up)
        if velocities:
            v_arr = np.array(velocities, dtype=np.float32)
            v_pos = v_arr[v_arr > 0]
            if len(v_pos) > 0:
                velocity_threshold = float(max(100.0, np.percentile(v_pos, 90) * 0.85))

    pre_shot_frames = sorted(e['frame_idx'] for e in pre_shot_entries)

    def has_recent_pre_shot(frame_idx):
        if not pre_shot_frames:
            return False
        pos = bisect.bisect_right(pre_shot_frames, frame_idx)
        if pos == 0:
            return False
        return (frame_idx - pre_shot_frames[pos - 1]) <= pre_follow_gap_frames

    def build_velocity_candidates(allow_follow_only=False):
        if len(wrist_entries) < 5 or velocity_threshold is None:
            return []
        candidates = []
        for i in range(1, len(wrist_entries) - 1):
            prev = wrist_entries[i - 1]
            curr = wrist_entries[i]
            nxt = wrist_entries[i + 1]
            dt_prev = max(1e-4, curr['timestamp'] - prev['timestamp'])
            dt_next = max(1e-4, nxt['timestamp'] - curr['timestamp'])
            vy_prev = -(curr['right_wrist'][1] - prev['right_wrist'][1]) / dt_prev
            vy_next = -(nxt['right_wrist'][1] - curr['right_wrist'][1]) / dt_next
            if vy_prev >= velocity_threshold and vy_prev >= vy_next:
                if SHOTLAB_REQUIRE_PRE_SHOT and pre_shot_frames and not allow_follow_only and not has_recent_pre_shot(curr['frame_idx']):
                    continue
                candidates.append({
                    'frame_idx': curr['frame_idx'],
                    'timestamp': curr['timestamp'],
                    'score': float(vy_prev / velocity_threshold),
                    'source': 'velocity'
                })
        return candidates

    def build_follow_segments():
        if not follow_entries:
            return []
        segment_gap = max(1, int(SHOTLAB_FOLLOW_SEGMENT_GAP_SECONDS * fps))
        segments = []
        current = [follow_entries[0]]
        for entry in follow_entries[1:]:
            if entry['frame_idx'] - current[-1]['frame_idx'] <= segment_gap:
                current.append(entry)
            else:
                segments.append(current)
                current = [entry]
        if current:
            segments.append(current)
        return segments

    follow_segments = build_follow_segments()
    debug['follow_segments'] = len(follow_segments)
    allow_follow_only = SHOTLAB_ALLOW_FOLLOW_ONLY and len(pre_shot_entries) < max(6, int(0.03 * len(pose_frames)))
    follow_candidates = []
    for segment in follow_segments:
        best = segment[0]
        best_score = 1.0
        if wrist_velocity_map:
            max_v = None
            for entry in segment:
                v = wrist_velocity_map.get(entry['frame_idx'])
                if v is None:
                    continue
                if max_v is None or v > max_v:
                    max_v = v
                    best = entry
            if max_v is not None and velocity_threshold:
                best_score = float(max_v / velocity_threshold)
        if SHOTLAB_REQUIRE_PRE_SHOT and pre_shot_frames and not allow_follow_only and not has_recent_pre_shot(best['frame_idx']):
            debug['skipped_no_pre_shot'] += 1
            continue
        follow_candidates.append({
            'frame_idx': best['frame_idx'],
            'timestamp': best.get('timestamp', 0.0),
            'score': best_score,
            'source': 'follow'
        })

    velocity_candidates = build_velocity_candidates(allow_follow_only=allow_follow_only) if SHOTLAB_USE_VELOCITY_CANDIDATES else []
    debug['velocity_candidates'] = len(velocity_candidates)

    release_candidates = follow_candidates + velocity_candidates
    if not release_candidates:
        return [], debug

    release_candidates.sort(key=lambda c: (c['frame_idx'], -c['score']))
    debug['release_candidates'] = len(release_candidates)

    follow_gaps = [
        follow_entries[i]['frame_idx'] - follow_entries[i - 1]['frame_idx']
        for i in range(1, len(follow_entries))
        if follow_entries[i]['frame_idx'] > follow_entries[i - 1]['frame_idx']
    ]
    if follow_gaps:
        median_gap = int(np.median(np.array(follow_gaps, dtype=np.float32)))
        adaptive_gap = max(int(0.3 * fps), int(median_gap * 0.7))
        min_gap_frames = max(1, min(min_gap_frames, adaptive_gap))

    debug['allow_follow_only'] = allow_follow_only

    def build_shot_from_release(release_frame, release_time):
        # Find the nearest preceding pre_shot within the allowed gap.
        candidates = [
            e for e in pre_shot_entries
            if 0 <= release_frame - e['frame_idx'] <= pre_follow_gap_frames
        ]
        if candidates:
            start_entry = max(candidates, key=lambda e: e['frame_idx'])
            start_frame = start_entry['frame_idx']
            start_time = start_entry.get('timestamp', 0.0)
        elif allow_follow_only:
            start_frame = max(0, release_frame - follow_only_lookback_frames)
            start_time = release_time
            debug['follow_only_used'] += 1
        else:
            debug['skipped_no_pre_shot'] += 1
            return None

        end_frame = release_frame + max_duration_frames
        for entry in pose_frames:
            if entry['frame_idx'] <= release_frame:
                continue
            if entry['frame_idx'] - release_frame < min_release_to_end_frames:
                continue
            if entry.get('state') in ('neutral', 'pre_shot'):
                end_frame = entry['frame_idx']
                break
        end_frame = min(end_frame, pose_frames[-1]['frame_idx'])
        return {
            'start_frame': start_frame,
            'release_frame': release_frame,
            'end_frame': end_frame,
            'timestamp': release_time if release_time is not None else start_time
        }

    shots = []
    last_release_frame = -min_gap_frames
    last_score = 0.0

    for cand in release_candidates:
        release_frame = cand['frame_idx']
        release_time = cand.get('timestamp', 0.0)
        score = float(cand.get('score', 1.0))
        gap = release_frame - last_release_frame

        if shots and gap < min_gap_frames:
            # Allow replacement if this candidate is much stronger and not too close.
            if gap > int(0.35 * min_gap_frames) and score > last_score * 1.35:
                replacement = build_shot_from_release(release_frame, release_time)
                if replacement is not None:
                    shots[-1] = replacement
                    last_release_frame = release_frame
                    last_score = score
                    debug['replaced_close_candidate'] += 1
                continue
            debug['skipped_min_gap'] += 1
            continue

        shot = build_shot_from_release(release_frame, release_time)
        if shot is None:
            continue
        shots.append(shot)
        last_release_frame = release_frame
        last_score = score

    debug['shots_detected'] = len(shots)
    return shots, debug

def get_nearest_pose_entry(pose_frames, frame_idx):
    if not pose_frames:
        return None
    nearest = min(pose_frames, key=lambda e: abs(e['frame_idx'] - frame_idx))
    return nearest

def get_pose_entry_with_angles(pose_frames, frame_idx, max_frame_gap):
    """Find the closest pose entry with angle data near a target frame."""
    best = None
    best_gap = None
    for entry in pose_frames:
        if not entry.get('angles'):
            continue
        gap = abs(entry['frame_idx'] - frame_idx)
        if gap > max_frame_gap:
            continue
        if best is None or gap < best_gap:
            best = entry
            best_gap = gap
    return best

def merge_frame_windows(windows):
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: (w[0], w[1]))
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

def build_ball_windows_from_shots(shot_attempts, total_frames, fps):
    if not shot_attempts or total_frames <= 0 or not fps:
        return []
    before = max(0, int(SHOTLAB_BALL_WINDOW_BEFORE_SECONDS * fps))
    after = max(0, int(SHOTLAB_BALL_WINDOW_AFTER_SECONDS * fps))
    last_frame = max(0, total_frames - 1)
    windows = []
    for shot in shot_attempts:
        release = int(shot.get('release_frame', shot.get('start_frame', 0)))
        end_base = int(max(release, shot.get('end_frame', release)))
        start = max(0, release - before)
        end = min(last_frame, end_base + after)
        if end >= start:
            windows.append((start, end))
    return merge_frame_windows(windows)

def process_video_for_ball_tracking(video_path, frame_stride=1, shot_attempts=None, fps=None, progress_callback=None):
    """Track basketball positions across the video."""
    if ball_model is None:
        return [], {'processed_frames': 0, 'timeouts': 0, 'errors': 0, 'total_frames': 0}

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tracks = []
    last_progress_update = -1
    timeout_count = 0
    error_count = 0
    processed_frames = 0
    window_frames_processed = 0

    # Only run ball detection near detected shot attempts to reduce cost.
    shot_windows = build_ball_windows_from_shots(shot_attempts or [], total_frames, fps)
    if not shot_windows and total_frames > 0:
        shot_windows = [(0, max(0, total_frames - 1))]
    total_window_frames = sum((end - start + 1) for start, end in shot_windows) if shot_windows else 0

    local_tracker = sv.ByteTrack() if SUPERVISION_AVAILABLE else None

    for window_idx, (win_start, win_end) in enumerate(shot_windows):
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, win_start)
        frame_idx = win_start
        while frame_idx <= win_end:
            ret, frame = cap.read()
            if not ret:
                break
            window_frames_processed += 1
            if frame_stride > 1 and frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            processed_frames += 1
            try:
                result = ball_model.predict(frame, confidence=SHOTLAB_BALL_CONFIDENCE).json()
            except requests.exceptions.Timeout:
                timeout_count += 1
                progress = (window_frames_processed / total_window_frames) if total_window_frames else 0.0
                update_shotlab_status('ball', f'Ball inference timeout {timeout_count}', 0.5 + progress * 0.35)
                if timeout_count >= SHOTLAB_BALL_TIMEOUT_LIMIT:
                    print("ShotLab ball tracking aborted due to repeated timeouts.")
                    break
                frame_idx += 1
                continue
            except Exception as e:
                error_count += 1
                print(f"ShotLab ball detection error at frame {frame_idx}: {e}")
                frame_idx += 1
                continue

            tracked = False
            if SUPERVISION_AVAILABLE and local_tracker is not None:
                try:
                    detections = sv.Detections.from_roboflow(result)
                    detections = local_tracker.update_with_detections(detections)
                    if len(detections) > 0:
                        confs = detections.confidence if detections.confidence is not None else None
                        idx = int(np.argmax(confs)) if confs is not None else 0
                        x1, y1, x2, y2 = detections.xyxy[idx]
                        tracks.append({
                            'frame': frame_idx,
                            'x': float((x1 + x2) / 2.0),
                            'y': float((y1 + y2) / 2.0),
                            'width': float(x2 - x1),
                            'height': float(y2 - y1),
                            'confidence': float(confs[idx]) if confs is not None else None,
                            'tracker_id': int(detections.tracker_id[idx]) if detections.tracker_id is not None else None
                        })
                        tracked = True
                except Exception:
                    tracked = False

            if not tracked:
                predictions = result.get('predictions', [])
                if predictions:
                    best = max(predictions, key=lambda p: p.get('confidence', 0))
                    tracks.append({
                        'frame': frame_idx,
                        'x': float(best.get('x', 0)),
                        'y': float(best.get('y', 0)),
                        'width': float(best.get('width', 0)),
                        'height': float(best.get('height', 0)),
                        'confidence': float(best.get('confidence', 0)),
                        'tracker_id': None
                    })

            frame_idx += 1

            if progress_callback and total_window_frames > 0:
                progress = min(1.0, window_frames_processed / total_window_frames)
                if processed_frames % max(1, 10 * frame_stride) == 0 and progress != last_progress_update:
                    last_progress_update = progress
                    progress_callback(window_frames_processed, total_window_frames, progress)

    cap.release()
    return tracks, {
        'processed_frames': processed_frames,
        'timeouts': timeout_count,
        'errors': error_count,
        'total_frames': total_frames,
        'tracks': len(tracks),
        'windows': shot_windows,
        'window_frames': total_window_frames,
        'window_count': len(shot_windows),
        'stride': frame_stride
    }

def filter_shots_by_ball_tracks(shot_attempts, ball_tracks, fps, pose_frames=None):
    """Filter shot attempts using ball track consistency to reduce false positives."""
    debug = {
        'kept': 0,
        'dropped': 0,
        'dropped_reasons': defaultdict(int)
    }
    if not shot_attempts:
        return [], debug
    if not ball_tracks or not fps:
        debug['dropped'] = len(shot_attempts)
        debug['dropped_reasons']['no_ball_tracks'] += len(shot_attempts)
        return [], debug

    tracks_by_frame = sorted(ball_tracks, key=lambda t: t['frame'])
    frames = [t['frame'] for t in tracks_by_frame]
    pre = max(0, int(SHOTLAB_BALL_VALIDATE_PRE_SECONDS * fps))
    post = max(1, int(SHOTLAB_BALL_VALIDATE_POST_SECONDS * fps))
    min_points = max(1, SHOTLAB_BALL_VALIDATE_MIN_POINTS)
    hand_window = max(0, int(SHOTLAB_BALL_VALIDATE_HAND_WINDOW_SECONDS * fps))
    hand_dist = float(SHOTLAB_BALL_VALIDATE_MAX_HAND_DIST_PX)

    filtered = []
    for shot in shot_attempts:
        release = int(shot.get('release_frame', shot.get('start_frame', 0)))
        window_start = max(0, release - pre)
        window_end = release + post
        start_idx = bisect.bisect_left(frames, window_start)
        end_idx = bisect.bisect_right(frames, window_end)
        window_tracks = tracks_by_frame[start_idx:end_idx]
        if len(window_tracks) < min_points:
            debug['dropped'] += 1
            debug['dropped_reasons']['ball_points_insufficient'] += 1
            continue

        if SHOTLAB_BALL_VALIDATE_REQUIRE_ARC:
            up_moves = 0
            down_moves = 0
            for i in range(1, len(window_tracks)):
                dy = window_tracks[i]['y'] - window_tracks[i - 1]['y']
                if dy < -1:
                    up_moves += 1
                elif dy > 1:
                    down_moves += 1
            if up_moves == 0 or down_moves == 0:
                debug['dropped'] += 1
                debug['dropped_reasons']['ball_no_arc'] += 1
                continue

        closest_release_frame = None
        if pose_frames is not None and hand_window >= 0:
            wrist_entry = get_nearest_pose_entry(pose_frames, release)
            wrist = wrist_entry.get('right_wrist') if wrist_entry else None
            if wrist:
                hand_start = max(0, release - hand_window)
                hand_end = release + hand_window
                hand_start_idx = bisect.bisect_left(frames, hand_start)
                hand_end_idx = bisect.bisect_right(frames, hand_end)
                hand_tracks = tracks_by_frame[hand_start_idx:hand_end_idx]
                if not hand_tracks:
                    debug['dropped'] += 1
                    debug['dropped_reasons']['ball_far_from_hand'] += 1
                    continue
                min_dist = None
                min_frame = None
                for track in hand_tracks:
                    dx = track['x'] - wrist[0]
                    dy = track['y'] - wrist[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        min_frame = track['frame']
                if min_dist is None or min_dist > hand_dist:
                    debug['dropped'] += 1
                    debug['dropped_reasons']['ball_far_from_hand'] += 1
                    continue
                closest_release_frame = min_frame

        shot_entry = dict(shot)
        if closest_release_frame is not None:
            shot_entry['release_frame'] = int(closest_release_frame)
            if fps:
                shot_entry['timestamp'] = float(closest_release_frame / fps)
        filtered.append(shot_entry)

    debug['kept'] = len(filtered)
    return filtered, debug

def dedupe_shots_by_release(shots, fps, min_gap_seconds=SHOTLAB_SHOT_DEDUPE_SECONDS):
    """Remove near-duplicate shots based on release frame proximity."""
    debug = {
        'kept': 0,
        'dropped': 0
    }
    if not shots or not fps:
        debug['kept'] = len(shots) if shots else 0
        return shots or [], debug

    min_gap_frames = max(1, int(min_gap_seconds * fps))
    ordered = sorted(shots, key=lambda s: int(s.get('release_frame', s.get('start_frame', 0))))
    deduped = []
    for shot in ordered:
        release = int(shot.get('release_frame', shot.get('start_frame', 0)))
        if not deduped:
            deduped.append(shot)
            continue
        prev_release = int(deduped[-1].get('release_frame', deduped[-1].get('start_frame', 0)))
        if release - prev_release < min_gap_frames:
            debug['dropped'] += 1
            continue
        deduped.append(shot)

    debug['kept'] = len(deduped)
    return deduped, debug

def detect_rim(frame):
    """Detect rim position in a given frame."""
    if rim_model is None:
        return None
    try:
        result = rim_model.predict(frame, confidence=SHOTLAB_RIM_CONFIDENCE).json()
        predictions = result.get('predictions', [])
        if not predictions:
            return None
        best = max(predictions, key=lambda p: p.get('confidence', 0))
        return {
            'x': float(best.get('x', 0)),
            'y': float(best.get('y', 0)),
            'width': float(best.get('width', 0)),
            'height': float(best.get('height', 0)),
            'confidence': float(best.get('confidence', 0))
        }
    except Exception as e:
        print(f"ShotLab rim detection error: {e}")
        return None

def sample_frames_at_indices(video_path, indices):
    cap = cv2.VideoCapture(video_path)
    frames = {}
    for idx in sorted(set(int(i) for i in indices if i is not None and i >= 0)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames[idx] = frame.copy()
    cap.release()
    return frames

def build_rim_sample_indices(total_frames, shot_attempts, fps, num_samples):
    if total_frames <= 0:
        return []
    indices = []
    if shot_attempts and fps:
        offsets = [
            0,
            int(0.25 * fps),
            int(0.5 * fps),
            int(0.9 * fps),
            -int(0.25 * fps)
        ]
        for shot in shot_attempts:
            release = int(shot.get('release_frame', shot.get('start_frame', 0)))
            for off in offsets:
                idx = release + off
                if 0 <= idx < total_frames:
                    indices.append(idx)
    if not indices:
        count = max(1, min(num_samples, total_frames))
        indices = np.linspace(0, total_frames - 1, count, dtype=int).tolist()
    if len(indices) > num_samples:
        step = max(1, len(indices) // num_samples)
        indices = indices[::step][:num_samples]
    return indices

def detect_rim_multi(video_path, shot_attempts, fps, num_samples=SHOTLAB_CALIBRATION_SAMPLES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    indices = build_rim_sample_indices(total_frames, shot_attempts, fps, num_samples)
    frames = sample_frames_at_indices(video_path, indices)

    best = None
    for idx, frame in frames.items():
        rim = detect_rim(frame)
        confidence = rim.get('confidence', 0) if rim else 0
        score = confidence + (rim.get('width', 0) * rim.get('height', 0) / 10000.0 if rim else 0)
        if rim and (best is None or score > best['score']):
            best = {
                'frame_idx': idx,
                'rim': rim,
                'frame': frame,
                'score': score
            }

    preview_frame = best['frame'] if best else (next(iter(frames.values())) if frames else None)
    preview = encode_frame_base64(preview_frame) if preview_frame is not None else None
    preview_shape = preview_frame.shape if preview_frame is not None else None

    rim_position = None
    rim_confidence = 0.0
    rim_source = 'auto'
    if best and best['rim'].get('confidence', 0) >= SHOTLAB_RIM_MIN_CONFIDENCE:
        rim_position = dict(best['rim'])
        rim_confidence = rim_position.get('confidence', 0.0)
    elif best:
        rim_position = None
        rim_confidence = best['rim'].get('confidence', 0.0)

    debug = {
        'samples': len(frames),
        'total_frames': total_frames,
        'best_frame': best['frame_idx'] if best else None,
        'best_confidence': rim_confidence,
        'rim_available': rim_position is not None
    }
    calibration = None
    if preview is not None and preview_shape is not None:
        h, w = preview_shape[:2]
        calibration = {
            'rim_frame': preview,
            'width': int(w),
            'height': int(h),
            'frame_idx': best['frame_idx'] if best else None
        }
        if rim_position is not None:
            calibration['rim_detected'] = {
                'x': rim_position['x'],
                'y': rim_position['y'],
                'width': rim_position.get('width', 0),
                'height': rim_position.get('height', 0)
            }

    return rim_position, debug, calibration, rim_source

def manual_rim_from_request(form, frame_shape):
    """Convert manual rim selection (normalized coords) into rim position."""
    if frame_shape is None:
        return None
    try:
        x_norm = form.get('rim_x_norm')
        y_norm = form.get('rim_y_norm')
        r_norm = form.get('rim_r_norm')
        if x_norm is None or y_norm is None:
            return None
        x_norm = float(x_norm)
        y_norm = float(y_norm)
        r_norm = float(r_norm) if r_norm is not None else 0.03
        h, w = frame_shape[:2]
        radius_px = max(8.0, r_norm * min(w, h))
        return {
            'x': float(x_norm * w),
            'y': float(y_norm * h),
            'width': float(radius_px * 2.0),
            'height': float(radius_px * 2.0),
            'confidence': 1.0,
            'source': 'manual'
        }
    except Exception:
        return None

def analyze_shot_outcome_with_debug(ball_tracks, rim_position, shot_frame_start, shot_frame_end):
    """Determine make/miss based on ball trajectory and provide debug info."""
    if not ball_tracks:
        return 'unknown', {'reason': 'no_ball_tracks', 'trajectory_points': 0}
    if not rim_position:
        return 'unknown', {'reason': 'no_rim_position', 'trajectory_points': 0}

    trajectory = [
        track for track in ball_tracks
        if shot_frame_start <= track['frame'] <= shot_frame_end
    ]
    if len(trajectory) < 2:
        return 'unknown', {'reason': 'insufficient_ball_points', 'trajectory_points': len(trajectory)}

    rim_center_x = rim_position['x']
    rim_center_y = rim_position['y']
    rim_width = rim_position.get('width', 0)
    rim_height = rim_position.get('height', 0)
    rim_radius_x = rim_width / 2.0 if rim_width else 20.0
    rim_radius_y = rim_height / 2.0 if rim_height else 18.0

    avg_ball_radius = None
    if trajectory:
        sizes = [
            (pt.get('width') or 0) / 2.0
            for pt in trajectory
            if pt.get('width') is not None
        ]
        if sizes:
            avg_ball_radius = float(np.median(np.array(sizes, dtype=np.float32)))
    if avg_ball_radius is None or avg_ball_radius <= 0:
        avg_ball_radius = 6.0

    x_tolerance = rim_radius_x * 0.9 + avg_ball_radius * 0.5
    y_margin = max(4.0, rim_radius_y * 0.2)
    inner_x = rim_radius_x * 0.85 + avg_ball_radius * 0.4
    inner_y = rim_radius_y * 0.85 + avg_ball_radius * 0.4

    ordered = sorted(trajectory, key=lambda t: t['frame'])
    saw_above = False
    entered = False
    entry_frame = None

    for pt in ordered:
        if pt['y'] < rim_center_y - y_margin:
            saw_above = True
        inside = (
            abs(pt['x'] - rim_center_x) <= inner_x and
            abs(pt['y'] - rim_center_y) <= inner_y
        )
        if inside and saw_above:
            entered = True
            entry_frame = pt['frame']
        if entered and pt['y'] > rim_center_y + y_margin and abs(pt['x'] - rim_center_x) <= x_tolerance:
            if entry_frame is None or pt['frame'] > entry_frame:
                return 'make', {
                    'reason': 'ball_entered_and_exited_below',
                    'trajectory_points': len(trajectory)
                }

    # Fallback make: clear center-line crossing while descending.
    for i in range(len(ordered) - 1):
        curr = ordered[i]
        next_pt = ordered[i + 1]
        dy = next_pt['y'] - curr['y']
        if dy <= 0:
            continue
        if curr['y'] <= rim_center_y - y_margin and next_pt['y'] >= rim_center_y + y_margin:
            t = (rim_center_y - curr['y']) / dy if dy != 0 else 0.0
            x_at = curr['x'] + t * (next_pt['x'] - curr['x'])
            if abs(x_at - rim_center_x) <= x_tolerance * 1.05:
                return 'make', {
                    'reason': 'ball_crossed_center_descending',
                    'trajectory_points': len(trajectory)
                }

    # Anything that doesn't show a clear enter+exit below is treated as miss.
    return 'miss', {'reason': 'no_clear_make_signal', 'trajectory_points': len(trajectory)}

def analyze_shot_outcome(ball_tracks, rim_position, shot_frame_start, shot_frame_end):
    outcome, _ = analyze_shot_outcome_with_debug(ball_tracks, rim_position, shot_frame_start, shot_frame_end)
    return outcome

def fetch_frames_by_index(video_path, frame_indices):
    """Fetch specific frames from a video in ascending order."""
    if not frame_indices:
        return {}
    cap = cv2.VideoCapture(video_path)
    frames = {}
    targets = sorted(set(frame_indices))
    target_set = set(targets)
    current_idx = 0
    target_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in target_set:
            frames[current_idx] = frame.copy()
            target_idx += 1
            if target_idx >= len(targets):
                break
        current_idx += 1

    cap.release()
    return frames

def register_shotlab_session(video_path, fps, shot_attempts, rim_position=None, ball_tracks=None):
    """Register a session so we can serve per-shot clips on demand."""
    session_id = uuid.uuid4().hex[:12]
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
    except Exception:
        total_frames = 0
    shotlab_sessions[session_id] = {
        'video_path': video_path,
        'fps': float(fps or 0.0),
        'shots': shot_attempts,
        'total_frames': total_frames,
        'rim_position': rim_position,
        'ball_tracks': ball_tracks,
        'ball_tracks_by_frame': None,
        'created_at': time.time()
    }
    return session_id

def extract_shot_clip(session_id, shot_index):
    """Extract (or reuse) a short clip around a detected shot."""
    session = shotlab_sessions.get(session_id)
    if session is None:
        raise ValueError('Invalid session_id')
    shots = session.get('shots') or []
    if shot_index < 0 or shot_index >= len(shots):
        raise ValueError('Invalid shot_index')

    video_path = session['video_path']
    fps = session.get('fps') or 30.0
    shot = shots[shot_index]
    before_frames = max(1, int(SHOTLAB_CLIP_BEFORE_SECONDS * fps))
    after_frames = max(1, int(SHOTLAB_CLIP_AFTER_SECONDS * fps))
    release_frame = int(shot.get('release_frame', shot.get('start_frame', 0)))
    start_frame = max(0, release_frame - before_frames)
    end_frame = release_frame + after_frames

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or session.get('total_frames') or 0)
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)

    clip_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'shotlab_clips')
    os.makedirs(clip_dir, exist_ok=True)
    clip_path = os.path.join(clip_dir, f'{session_id}_shot{shot_index + 1}.mp4')
    if os.path.exists(clip_path):
        cap.release()
        return clip_path

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    out_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError('Unable to read video dimensions')

    out_width, out_height = width, height
    scale = 1.0
    if SHOTLAB_CLIP_MAX_DIM and SHOTLAB_CLIP_MAX_DIM > 0:
        max_dim = max(width, height)
        if max_dim > SHOTLAB_CLIP_MAX_DIM:
            scale = SHOTLAB_CLIP_MAX_DIM / float(max_dim)
            out_width = max(2, int(width * scale))
            out_height = max(2, int(height * scale))
            # Many codecs prefer even dimensions.
            if out_width % 2 == 1:
                out_width -= 1
            if out_height % 2 == 1:
                out_height -= 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(clip_path, fourcc, out_fps, (out_width, out_height))
    if not writer.isOpened():
        cap.release()
        raise ValueError('Unable to create clip writer')

    rim_position = session.get('rim_position') if SHOTLAB_CLIP_OVERLAY else None
    ball_tracks = session.get('ball_tracks') if SHOTLAB_CLIP_OVERLAY else None
    ball_map = None
    if SHOTLAB_CLIP_OVERLAY and ball_tracks:
        ball_map = session.get('ball_tracks_by_frame')
        if ball_map is None:
            ball_map = {}
            for track in ball_tracks:
                frame_id = int(track.get('frame', -1))
                if frame_id < 0:
                    continue
                existing = ball_map.get(frame_id)
                if existing is None:
                    ball_map[frame_id] = track
                else:
                    conf_new = track.get('confidence', 0) or 0
                    conf_old = existing.get('confidence', 0) or 0
                    if conf_new >= conf_old:
                        ball_map[frame_id] = track
            session['ball_tracks_by_frame'] = ball_map

    trail = []
    trail_max = max(1, int(SHOTLAB_CLIP_TRAIL_FRAMES))

    current = start_frame
    while current <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

        if SHOTLAB_CLIP_OVERLAY:
            if rim_position is not None:
                rim_x = int(rim_position.get('x', 0) * scale)
                rim_y = int(rim_position.get('y', 0) * scale)
                rim_w = int((rim_position.get('width', 0) or 0) * scale)
                rim_h = int((rim_position.get('height', 0) or 0) * scale)
                if rim_w > 0 and rim_h > 0:
                    cv2.ellipse(frame, (rim_x, rim_y), (max(1, rim_w // 2), max(1, rim_h // 2)), 0, 0, 360, (0, 255, 255), 2)
                else:
                    cv2.circle(frame, (rim_x, rim_y), max(6, int(10 * scale)), (0, 255, 255), 2)
            if ball_map is not None:
                track = ball_map.get(current)
                if track is not None:
                    bx = int(track.get('x', 0) * scale)
                    by = int(track.get('y', 0) * scale)
                    trail.append((bx, by))
                    if len(trail) > trail_max:
                        trail.pop(0)
                if len(trail) >= 2:
                    for i in range(1, len(trail)):
                        cv2.line(frame, trail[i - 1], trail[i], (0, 200, 255), 2)
                if trail:
                    cv2.circle(frame, trail[-1], max(3, int(6 * scale)), (255, 80, 80), -1)

        writer.write(frame)
        current += 1

    writer.release()
    cap.release()
    return clip_path

def pre_generate_shot_clips(session_id):
    """Pre-generate all shot clips so replay clicks are fast."""
    session = shotlab_sessions.get(session_id)
    if session is None:
        return
    shots = session.get('shots') or []
    for idx in range(len(shots)):
        try:
            extract_shot_clip(session_id, idx)
        except Exception as e:
            print(f"ShotLab clip pregen error (shot {idx + 1}): {e}")

def pre_generate_shot_clips_async(session_id):
    if not session_id:
        return
    thread = threading.Thread(target=pre_generate_shot_clips, args=(session_id,), daemon=True)
    thread.start()

# ====================== API ENDPOINTS ======================

@app.route('/')
def index():
    """Serve the landing page from project root"""
    return send_from_directory(str(project_root), 'index.html')

@app.route('/tool')
@app.route('/tool/')
def tool_index():
    """Serve the ShotSync application"""
    # Serve index.html from the tool directory
    tool_dir = Path(__file__).parent
    return send_from_directory(str(tool_dir), 'index.html')

@app.route('/tool/<path:filename>')
def tool_static(filename):
    """Serve static files for the tool"""
    # Serve files from the tool directory (style.css, app.js, etc.)
    tool_dir = Path(__file__).parent
    # First try tool directory, then try static subdirectory
    file_path = tool_dir / filename
    if file_path.exists():
        return send_from_directory(str(tool_dir), filename)
    # Fallback to static subdirectory
    static_dir = tool_dir / 'static'
    if (static_dir / filename).exists():
        return send_from_directory(str(static_dir), filename)
    from flask import abort
    abort(404)

@app.route('/shotlab')
@app.route('/shotlab/')
def shotlab_index():
    """Serve the ShotLab page."""
    shotlab_dir = project_root / 'shotlab'
    return send_from_directory(str(shotlab_dir), 'index.html')

@app.route('/shotlab/<path:filename>')
def shotlab_static(filename):
    """Serve static files for ShotLab."""
    shotlab_dir = project_root / 'shotlab'
    file_path = shotlab_dir / filename
    if file_path.exists():
        return send_from_directory(str(shotlab_dir), filename)
    from flask import abort
    abort(404)

@app.route('/api/process_shot', methods=['POST'])
def process_shot():
    """Process a single shot (benchmark or user) from video frames."""
    try:
        data = request.json
        frames_data = data.get('frames', [])
        
        if not frames_data:
            return jsonify({'error': 'No frames provided'}), 400
        
        shot_angles, landmark_frames = process_video_frames(frames_data)
        
        return jsonify({
            'success': True,
            'shot_angles': shot_angles,
            'landmark_frames': landmark_frames
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare_shots', methods=['POST'])
def compare_shots():
    """Compare benchmark and user shots."""
    try:
        data = request.json
        benchmark_data = data.get('benchmark', {})
        user_data = data.get('user', {})
        
        bench_angles = benchmark_data.get('shot_angles', [])
        user_angles = user_data.get('shot_angles', [])
        
        if not bench_angles or not user_angles:
            return jsonify({'error': 'Missing shot data'}), 400
        
        # Extract form series
        bench_times, bench_form = extract_form_series(bench_angles)
        user_times, user_form = extract_form_series(user_angles)
        
        if len(bench_form) < 2 or len(user_form) < 2:
            return jsonify({'error': 'Insufficient data for comparison'}), 400
        
        # Run DTW
        if not DTW_AVAILABLE:
            return jsonify({'error': 'fastdtw library not available. Install with: pip install fastdtw'}), 500
        
        dist, path = fastdtw(bench_form.reshape(-1,1), user_form.reshape(-1,1), dist=euclidean)
        
        # Compute closeness
        user_closeness = compute_user_closeness(bench_form, user_form, path)
        
        # Generate feedback
        feedback = generate_feedback(bench_angles, user_angles, bench_times, user_times, user_closeness)
        
        # Key events
        def key_events(times):
            if len(times) == 0:
                return []
            idxs = [0, len(times)//3, len(times)//2, (2*len(times))//3, len(times)-1]
            names = ["Start", "Ball Set", "Elbow Above Shoulder", "Release", "Follow Through"]
            return [{'name': names[k], 'time': float(times[idxs[k]])} for k in range(len(idxs))]
        
        bench_events = key_events(bench_times)
        user_events = key_events(user_times)
        
        # Align times
        if bench_events and user_events:
            start_offset = bench_events[0]['time'] - user_events[0]['time']
            user_times_aligned = (user_times + start_offset).tolist()
        else:
            user_times_aligned = user_times.tolist()
        
        return jsonify({
            'success': True,
            'dtw_distance': float(dist),
            'user_closeness': user_closeness.tolist(),
            'bench_times': bench_times.tolist(),
            'user_times': user_times_aligned,
            'bench_events': bench_events,
            'user_events': user_events,
            'feedback': feedback,
            'benchmark_landmarks': benchmark_data.get('landmark_frames', []),
            'user_landmarks': user_data.get('landmark_frames', [])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shotlab_status')
def shotlab_status_endpoint():
    """Get current ShotLab processing status."""
    return jsonify(shotlab_status)

@app.route('/api/shotlab_clip')
def shotlab_clip_endpoint():
    """Serve a short clip for a detected shot."""
    try:
        session_id = request.args.get('session_id', '').strip()
        shot_index = int(request.args.get('shot_index', '-1'))
        clip_path = extract_shot_clip(session_id, shot_index)
        return send_file(clip_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/shotlab_video')
def shotlab_video_endpoint():
    """Serve the original ShotLab video (supports range requests)."""
    session_id = request.args.get('session_id', '').strip()
    session = shotlab_sessions.get(session_id)
    if session is None:
        abort(404)
    video_path = session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        abort(404)

    mime = mimetypes.guess_type(video_path)[0] or 'video/mp4'
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(video_path, mimetype=mime, conditional=True)

    size = os.path.getsize(video_path)
    match = re.match(r'bytes=(\d+)-(\d*)', range_header)
    if not match:
        return send_file(video_path, mimetype=mime, conditional=True)

    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else size - 1
    end = min(end, size - 1)
    if start > end:
        return send_file(video_path, mimetype=mime, conditional=True)

    length = end - start + 1
    with open(video_path, 'rb') as video_file:
        video_file.seek(start)
        data = video_file.read(length)

    response = Response(data, 206, mimetype=mime, direct_passthrough=True)
    response.headers.add('Content-Range', f'bytes {start}-{end}/{size}')
    response.headers.add('Accept-Ranges', 'bytes')
    response.headers.add('Content-Length', str(length))
    return response

@app.route('/api/process_shotlab_session', methods=['POST'])
def process_shotlab_session():
    """Process full shooting session video for ShotLab analysis."""
    try:
        update_shotlab_status('starting', 'Initializing ShotLab...', 0.0)
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video provided'}), 400
        if not MEDIAPIPE_AVAILABLE:
            return jsonify({'success': False, 'error': 'MediaPipe not available on server'}), 500

        ok, error = initialize_shotlab_models()
        if not ok:
            update_shotlab_status('error', error, 0.0)
            return jsonify({'success': False, 'error': error}), 500

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        filename = video_file.filename
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in app.config['ALLOWED_EXTENSIONS']:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        from werkzeug.utils import secure_filename
        safe_name = secure_filename(filename)
        timestamp = str(int(time.time()))
        save_name = f"shotlab_{timestamp}_{safe_name}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        update_shotlab_status('saving', 'Saving uploaded video...', 0.02)
        video_file.save(video_path)

        warnings = []

        # Read a reference frame for sizing/manual calibration.
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        if not ret or first_frame is None:
            update_shotlab_status('error', 'Unable to read video', 0.0)
            return jsonify({'success': False, 'error': 'Unable to read video'}), 500

        manual_rim = None

        # Detect court across multiple frames for robustness.
        update_shotlab_status('court', 'Detecting court across frames...', 0.06)
        transform_matrix, court_keypoints, court_preview, court_auto_debug, court_shape = detect_court_transform_multi(video_path)
        if transform_matrix is None:
            # Fallback to first frame if multi-frame detection fails.
            fallback_keypoints = detect_court_keypoints(first_frame)
            fallback_transform = get_perspective_transform(fallback_keypoints)
            if fallback_transform is not None:
                transform_matrix = fallback_transform
                court_keypoints = fallback_keypoints
                court_auto_debug['fallback_first_frame'] = True
            else:
                court_auto_debug['fallback_first_frame'] = False

        manual_court_shape = court_shape if court_shape is not None else first_frame.shape
        manual_court_transform = manual_court_transform_from_request(request.form or {}, manual_court_shape)
        court_source = 'auto'
        if manual_court_transform is not None:
            transform_matrix = manual_court_transform
            court_source = 'manual'
            court_auto_debug['manual_override'] = True
        if transform_matrix is None:
            warnings.append('court_transform_unavailable')

        court_calibration = None
        if court_preview is not None and manual_court_shape is not None:
            h_c, w_c = manual_court_shape[:2]
            court_calibration = {
                'court_frame': court_preview,
                'width': int(w_c),
                'height': int(h_c),
                'frame_idx': court_auto_debug.get('best_frame')
            }

        court_debug = {
            'court_keypoints': len(court_keypoints),
            'court_transform_available': transform_matrix is not None,
            'auto': court_auto_debug,
            'preview_available': court_preview is not None,
            'source': court_source
        }

        # If we still can't map the court, stop here to save model credits.
        if transform_matrix is None:
            update_shotlab_status('complete', 'Court calibration required', 1.0)
            return jsonify({
                'success': True,
                'shots': [],
                'zone_stats': {},
                'total_attempts': 0,
                'total_makes': 0,
                'warnings': warnings,
                'debug': {
                    'court': court_debug,
                    'pose': {},
                    'ball': {},
                    'rim': {}
                },
                'calibration': {
                    'rim': None,
                    'rim_required': False,
                    'court': court_calibration,
                    'court_required': True
                },
                'analysis_skipped': 'court_missing'
            })

        # Process pose and ball tracking
        update_shotlab_status('pose', 'Running pose detection...', 0.1)
        pose_frames, fps = process_video_for_pose(
            video_path,
            frame_stride=SHOTLAB_FRAME_STRIDE,
            progress_callback=lambda i, t, p: update_shotlab_status('pose', f'Pose frames {i}/{t}', 0.1 + p * 0.35)
        )
        shot_attempts, shot_detection_debug = detect_shot_attempts_from_pose(pose_frames, fps)
        pose_debug = summarize_pose_states(pose_frames)
        pose_debug.update({
            'fps': float(fps or 0.0),
            'frame_stride': SHOTLAB_FRAME_STRIDE,
            'shot_attempts_detected': len(shot_attempts),
            'shot_detection': shot_detection_debug
        })

        if not shot_attempts:
            update_shotlab_status('complete', 'No shots detected', 1.0)
            manual_rim_no_shots = manual_rim_from_request(request.form or {}, first_frame.shape)
            ball_debug = {
                'processed_frames': 0,
                'timeouts': 0,
                'errors': 0,
                'total_frames': 0,
                'tracks': 0,
                'windows': [],
                'window_frames': 0,
                'window_count': 0,
                'stride': SHOTLAB_BALL_STRIDE,
                'skipped': 'no_shot_attempts'
            }
            rim_debug = {
                'source': 'manual' if manual_rim_no_shots is not None else 'auto',
                'auto': {'samples': 0, 'rim_available': False},
                'available': manual_rim_no_shots is not None
            }
            return jsonify({
                'success': True,
                'shots': [],
                'zone_stats': {},
                'total_attempts': 0,
                'total_makes': 0,
                'warnings': warnings,
                'debug': {
                    'court': court_debug,
                    'pose': pose_debug,
                    'ball': ball_debug,
                    'rim': rim_debug
                },
                'calibration': {
                    'rim': None,
                    'rim_required': False,
                    'court': court_calibration,
                    'court_required': transform_matrix is None
                },
                'session_id': session_id
            })

        update_shotlab_status('rim', 'Detecting rim across frames...', 0.48)
        rim_position, rim_auto_debug, rim_calibration, rim_source = detect_rim_multi(
            video_path,
            shot_attempts,
            fps
        )
        rim_shape = None
        if rim_calibration and rim_calibration.get('height') and rim_calibration.get('width'):
            rim_shape = (int(rim_calibration['height']), int(rim_calibration['width']), 3)
        manual_rim = manual_rim_from_request(request.form or {}, rim_shape or first_frame.shape)
        if manual_rim is not None:
            rim_position = manual_rim
            rim_source = 'manual'
            rim_auto_debug['manual_override'] = True
        elif rim_position is None:
            warnings.append('rim_detection_unavailable_auto')
            if rim_calibration is None:
                rim_calibration = {
                    'rim_frame': encode_frame_base64(first_frame),
                    'width': int(first_frame.shape[1]),
                    'height': int(first_frame.shape[0]),
                    'frame_idx': 0
                }
            update_shotlab_status('complete', 'Rim calibration required', 1.0)
            return jsonify({
                'success': True,
                'shots': [],
                'zone_stats': {},
                'total_attempts': 0,
                'total_makes': 0,
                'warnings': warnings,
                'debug': {
                    'court': court_debug,
                    'pose': pose_debug,
                    'ball': {},
                    'rim': rim_auto_debug
                },
                'calibration': {
                    'rim': rim_calibration,
                    'rim_required': True,
                    'court': court_calibration,
                    'court_required': False
                },
                'analysis_skipped': 'rim_missing'
            })

        update_shotlab_status('ball', f'Tracking ball near {len(shot_attempts)} shots...', 0.5)
        ball_tracks, ball_debug = process_video_for_ball_tracking(
            video_path,
            frame_stride=SHOTLAB_BALL_STRIDE,
            shot_attempts=shot_attempts,
            fps=fps,
            progress_callback=lambda i, t, p: update_shotlab_status('ball', f'Ball window frames {i}/{t}', 0.5 + p * 0.35)
        )

        filtered_shots, shot_filter_debug = filter_shots_by_ball_tracks(shot_attempts, ball_tracks, fps, pose_frames)
        if shot_filter_debug:
            shot_detection_debug['ball_filter'] = {
                'kept': shot_filter_debug.get('kept', 0),
                'dropped': shot_filter_debug.get('dropped', 0),
                'reasons': dict(shot_filter_debug.get('dropped_reasons', {}))
            }
        deduped_shots, dedupe_debug = dedupe_shots_by_release(filtered_shots, fps)
        if dedupe_debug:
            shot_detection_debug['dedupe'] = {
                'kept': dedupe_debug.get('kept', 0),
                'dropped': dedupe_debug.get('dropped', 0)
            }
        shot_attempts = deduped_shots

        session_id = register_shotlab_session(video_path, fps, shot_attempts, rim_position, ball_tracks) if shot_attempts else None
        if session_id and SHOTLAB_CLIP_PREGENERATE:
            pre_generate_shot_clips_async(session_id)

        # Fetch release frames for rim detection fallback
        update_shotlab_status('analyze', 'Analyzing shots...', 0.86)
        release_frames = fetch_frames_by_index(
            video_path,
            [shot['release_frame'] for shot in shot_attempts if shot.get('release_frame') is not None]
        )

        shots_analysis = []
        zone_stats = defaultdict(lambda: {
            'attempts': 0,
            'makes': 0,
            'score_total': 0.0,
            'score_count': 0,
            'benchmark_mean': SHOTLAB_DEFAULT_BENCHMARK,
            'benchmark_std': SHOTLAB_DEFAULT_STD
        })
        release_search_gap_frames = max(1, int(SHOTLAB_RELEASE_SEARCH_WINDOW_SECONDS * fps))
        outcome_reasons = defaultdict(int)
        unknown_zones = 0
        shots_with_scores = 0
        clip_before_frames = max(0, int(SHOTLAB_CLIP_BEFORE_SECONDS * fps))
        clip_after_frames = max(0, int(SHOTLAB_CLIP_AFTER_SECONDS * fps))
        total_frames = shotlab_sessions.get(session_id, {}).get('total_frames', 0) if session_id else 0
        video_url = f'/api/shotlab_video?session_id={session_id}' if session_id else None

        for idx, shot in enumerate(shot_attempts):
            release_frame = shot.get('release_frame', shot['start_frame'])
            pose_entry = get_pose_entry_with_angles(pose_frames, release_frame, release_search_gap_frames)
            if pose_entry is None:
                pose_entry = get_nearest_pose_entry(pose_frames, release_frame)
            foot_pos = pose_entry.get('foot_pos') if pose_entry else None
            court_pos = player_to_court_position(foot_pos, transform_matrix) if foot_pos else None

            zone = 'unknown'
            if court_pos is not None:
                zone = classify_shot_zone(float(court_pos[0]), float(court_pos[1]))
            else:
                unknown_zones += 1

            shot_rim = rim_position
            if shot_rim is None and release_frame in release_frames:
                shot_rim = detect_rim(release_frames[release_frame])

            outcome_window_extra = int(SHOTLAB_OUTCOME_WINDOW_AFTER_SECONDS * fps) if fps else 0
            outcome_end_frame = release_frame + max(0, outcome_window_extra)
            outcome, outcome_debug = analyze_shot_outcome_with_debug(
                ball_tracks,
                shot_rim,
                shot['start_frame'],
                outcome_end_frame
            )
            outcome_reasons[outcome_debug.get('reason', 'unknown')] += 1

            form_score = None
            if pose_entry and pose_entry.get('angles'):
                form_score = pose_entry['angles'].get('form_score')

            benchmark_mean, benchmark_std = get_zone_benchmark(zone)
            zone_stats[zone]['benchmark_mean'] = benchmark_mean
            zone_stats[zone]['benchmark_std'] = benchmark_std
            shotsync_score = compute_shotsync_score(form_score, benchmark_mean, benchmark_std) if form_score is not None else None

            zone_stats[zone]['attempts'] += 1
            if outcome == 'make':
                zone_stats[zone]['makes'] += 1
            if shotsync_score is not None:
                zone_stats[zone]['score_total'] += shotsync_score
                zone_stats[zone]['score_count'] += 1
                shots_with_scores += 1

            clip_release_frame = int(shot.get('release_frame', shot.get('start_frame', 0)))
            clip_start_frame = max(0, clip_release_frame - clip_before_frames)
            clip_end_frame = clip_release_frame + clip_after_frames
            if total_frames > 0:
                clip_end_frame = min(clip_end_frame, total_frames - 1)
            clip_start = (clip_start_frame / fps) if fps else None
            clip_end = (clip_end_frame / fps) if fps else None

            shots_analysis.append({
                'shot_number': idx + 1,
                'timestamp': shot['timestamp'],
                'zone': zone,
                'court_position': {'x': float(court_pos[0]), 'y': float(court_pos[1])} if court_pos is not None else None,
                'outcome': outcome,
                'form_score': form_score,
                'shotsync_score': shotsync_score,
                'clip_url': f'/api/shotlab_clip?session_id={session_id}&shot_index={idx}' if session_id else None,
                'clip_start': clip_start,
                'clip_end': clip_end,
                'video_url': video_url,
                'debug': {
                    'outcome_reason': outcome_debug.get('reason'),
                    'trajectory_points': outcome_debug.get('trajectory_points', 0),
                    'release_frame': release_frame,
                    'start_frame': shot['start_frame'],
                    'end_frame': shot['end_frame']
                }
            })

        zone_percentages = {}
        for zone, stats in zone_stats.items():
            attempts = stats['attempts']
            makes = stats['makes']
            zone_percentages[zone] = {
                'attempts': attempts,
                'makes': makes,
                'percentage': (makes / attempts * 100.0) if attempts > 0 else 0.0,
                'shotsync_score': (stats['score_total'] / stats['score_count']) if stats['score_count'] > 0 else None,
                'benchmark': stats.get('benchmark_mean', SHOTLAB_DEFAULT_BENCHMARK),
                'benchmark_std': stats.get('benchmark_std', SHOTLAB_DEFAULT_STD)
            }

        total_makes = sum(1 for s in shots_analysis if s['outcome'] == 'make')
        debug_summary = {
            'court': court_debug,
            'pose': pose_debug,
            'ball': ball_debug,
            'rim': {
                'source': rim_source,
                'auto': rim_auto_debug,
                'available': rim_position is not None
            },
            'shots': {
                'shot_attempts': len(shot_attempts),
                'shots_with_scores': shots_with_scores,
                'unknown_zones': unknown_zones,
                'outcome_reasons': dict(outcome_reasons)
            }
        }

        update_shotlab_status('complete', 'Analysis complete', 1.0)
        return jsonify({
            'success': True,
            'shots': shots_analysis,
            'zone_stats': zone_percentages,
            'total_attempts': len(shots_analysis),
            'total_makes': total_makes,
            'warnings': warnings,
            'debug': debug_summary,
            'calibration': {
                'rim': rim_calibration,
                'rim_required': rim_position is None,
                'court': court_calibration,
                'court_required': transform_matrix is None
            },
            'video_url': video_url,
            'session_id': session_id
        })

    except Exception as e:
        update_shotlab_status('error', str(e), 0.0)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/send_email', methods=['POST'])
def send_email():
    """Send analysis results via email."""
    try:
        data = request.json
        firstName = data.get('firstName', '').strip()
        lastName = data.get('lastName', '').strip()
        email = data.get('email', '').strip()
        chartImage = data.get('chartImage', '')
        overallScore = data.get('overallScore', '--')
        feedback = data.get('feedback', '')
        
        # Validate inputs
        if not firstName or not lastName or not email:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Use Brevo if configured, otherwise fall back to AWS SES
        use_brevo = bool(BREVO_API_KEY)
        
        if not use_brevo and not ses_client:
            return jsonify({
                'success': False, 
                'error': 'Email service not configured. Please set BREVO_API_KEY (recommended) or AWS SES credentials.'
            }), 500
        
        # Create HTML body
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: white; border-radius: 10px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h1 style="color: #667eea; margin-top: 0; text-align: center;">🏀 Shot Sync</h1>
                    <h2 style="color: #333; text-align: center; font-size: 24px;">Thank You for Using Shot Sync!</h2>
                    
                    <p style="font-size: 16px;">Hi {firstName},</p>
                    <p>We're excited to share your personalized shot analysis results with you. Here's everything you need to improve your basketball shooting form:</p>
                    
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 10px; margin: 30px 0; text-align: center;">
                        <h2 style="color: white; margin: 0; font-size: 36px;">Overall Score: {overallScore}%</h2>
                    </div>
                    
                    <h3 style="color: #333; margin-top: 30px;">Your Shot Analysis Graph</h3>
                    <p style="color: #666; font-size: 14px;">This graph shows how your shot form compares to the benchmark throughout your shooting motion.</p>
                    <img src="data:image/png;base64,{chartImage.split(',')[1] if ',' in chartImage else chartImage}" alt="Shot Analysis Chart" style="max-width: 100%; height: auto; border-radius: 10px; margin: 20px 0; border: 2px solid #e0e0e0;">
                    
                    <h3 style="color: #333; margin-top: 30px;">Feedback & Recommendations</h3>
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #667eea;">
                        <pre style="white-space: pre-wrap; font-family: Arial, sans-serif; margin: 0; color: #333;">{feedback}</pre>
                    </div>
                    
                    <div style="background: #f0f7ff; padding: 20px; border-radius: 10px; margin: 30px 0; text-align: center;">
                        <p style="margin: 0; font-size: 16px; color: #333;"><strong>Keep practicing!</strong> Every shot is an opportunity to improve.</p>
                    </div>
                    
                    <p style="margin-top: 30px; color: #666; font-size: 14px;">We hope this analysis helps you take your game to the next level!</p>
                    <p style="margin-top: 20px;">Best regards,<br><strong>The Shot Sync Team</strong></p>
                    
                    <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 30px 0;">
                    <p style="text-align: center; color: #999; font-size: 12px;">Shot Sync - Your Basketball Shot Analysis Partner</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Send email via Brevo (preferred) or AWS SES
        try:
            if use_brevo:
                # Send via Brevo API
                brevo_url = 'https://api.brevo.com/v3/smtp/email'
                headers = {
                    'accept': 'application/json',
                    'api-key': BREVO_API_KEY,
                    'content-type': 'application/json'
                }
                
                payload = {
                    'sender': {
                        'name': BREVO_SENDER_NAME,
                        'email': BREVO_SENDER_EMAIL
                    },
                    'to': [{'email': email, 'name': f'{firstName} {lastName}'}],
                    'subject': f'Your Shot Sync Analysis Results - {firstName}!',
                    'htmlContent': html_body
                }
                
                response = requests.post(brevo_url, json=payload, headers=headers)
                
                if response.status_code == 201:
                    return jsonify({
                        'success': True,
                        'message': 'Email sent successfully via Brevo',
                        'message_id': response.json().get('messageId', '')
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Brevo API error: {response.status_code} - {response.text}'
                    }), 500
            else:
                # Send via AWS SES (fallback)
                response = ses_client.send_email(
                    Source=FROM_EMAIL,
                    Destination={'ToAddresses': [email]},
                    Message={
                        'Subject': {'Data': f'Your Shot Sync Analysis Results - {firstName}!', 'Charset': 'UTF-8'},
                        'Body': {
                            'Html': {'Data': html_body, 'Charset': 'UTF-8'}
                        }
                    }
                )
                
                return jsonify({
                    'success': True,
                    'message': 'Email sent successfully via AWS SES',
                    'message_id': response['MessageId']
                })
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            return jsonify({
                'success': False,
                'error': f'AWS SES error ({error_code}): {error_message}'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to send email: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ====================== SHOT TRACKER DETECTION ======================

    # Using only YOLOv8 tracker state from AI-Basketball-Shot-Detection-Tracker

# Global detector instance - initialized the same way as original

# Frame queue for feeding frames to ShotDetectorWeb
frame_queue = None
frame_queue_lock = threading.Lock()

class FrameFeeder:
    """Custom VideoCapture-like class that feeds frames from API to ShotDetectorWeb"""
    def __init__(self):
        self.frame_queue = []
        self.lock = threading.Lock()
        self.current_frame = None
    
    def read(self):
        """Read a frame from the queue - matches cv2.VideoCapture.read() interface"""
        with self.lock:
            if self.frame_queue:
                frame = self.frame_queue.pop(0)
                self.current_frame = frame.copy()  # Keep a copy
                return True, frame
            elif self.current_frame is not None:
                # Return last frame if queue is empty (keep processing same frame)
                return True, self.current_frame.copy()
            else:
                # Return black frame if no frames available (detector will wait)
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
    
    def set(self, prop, value):
        """Dummy method to match cv2.VideoCapture interface"""
        pass
    
    def release(self):
        """Release resources"""
        with self.lock:
            self.frame_queue = []
            self.current_frame = None
    
    def push_frame(self, frame):
        """Push a frame into the queue from API"""
        with self.lock:
            # Keep only last 5 frames to avoid memory issues
            if len(self.frame_queue) > 5:
                self.frame_queue.pop(0)
            self.frame_queue.append(frame.copy())
            self.current_frame = frame

def initialize_shot_detector():
    """Initialize detector EXACTLY as original does (line 1832: detector = ShotDetectorWeb(...))"""
    global frame_queue
    if not YOLOV8_AVAILABLE:
        return False
    
    # Use the original module's detector variable - initialize it the SAME way
    if hasattr(tracker_module, 'detector') and tracker_module.detector is not None:
        return True  # Already initialized
    
    try:
        # Create frame feeder that acts like VideoCapture
        frame_queue = FrameFeeder()
        
        # Monkey-patch sys.exit to prevent it from killing Flask
        import sys
        original_exit = sys.exit
        def safe_exit(code=0):
            if code != 0:
                raise Exception(f"ShotDetectorWeb tried to exit with code {code}")
            original_exit(code)
        sys.exit = safe_exit
        
        try:
            # Initialize EXACTLY as original (line 1832): ShotDetectorWeb(video_path=video_path, use_webcam=False)
            import tempfile
            import cv2
            
            dummy_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            dummy_video.close()
            
            # Create minimal valid video file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(dummy_video.name, fourcc, 20.0, (640, 480))
            if out.isOpened():
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                for _ in range(10):
                    out.write(frame)
                out.release()
            
            # Initialize EXACTLY as original does - use the original module's detector variable
            # BUT: Stop the processing thread first, replace cap, then restart
            tracker_module.detector = ShotDetectorWeb(video_path=dummy_video.name, use_webcam=False)
            
            # Stop the processing thread temporarily
            tracker_module.detector.running = False
            if tracker_module.detector.processing_thread.is_alive():
                tracker_module.detector.processing_thread.join(timeout=1.0)
            
            # Replace VideoCapture with frame feeder
            original_cap = tracker_module.detector.cap
            tracker_module.detector.cap = frame_queue
            if original_cap:
                original_cap.release()  # Release the dummy video capture
            
            # Restart the processing thread - it will now read from frame_queue
            tracker_module.detector.running = True
            tracker_module.detector.processing_thread = threading.Thread(target=tracker_module.detector.process_video, daemon=True)
            tracker_module.detector.processing_thread.start()
            
            print("✓ ShotDetectorWeb initialized EXACTLY as original - using original module's detector")
            
            # Clean up dummy file
            try:
                if os.path.exists(dummy_video.name):
                    os.unlink(dummy_video.name)
            except:
                pass
            
            return True
        finally:
            # Restore original sys.exit
            sys.exit = original_exit
                
    except Exception as e:
        print(f"✗ Warning: Could not initialize ShotDetectorWeb: {e}")
        import traceback
        traceback.print_exc()
        return False

# DON'T initialize on import - initialize lazily when first needed
# This prevents sys.exit from killing Flask during startup

def detect_ball_hsv_fallback(frame):
    """
    HSV-based color detection for basketball (fallback when YOLO fails)
    Detects orange/red basketball colors
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    
    # Orange range: Hue 10-30, Saturation > 50%, Value > 40%
    # Red range: Hue 0-10 or 170-180, Saturation > 60%, Value > 50%
    orange_lower = np.array([10, 128, 102])  # HSV: [10, 50%, 40%]
    orange_upper = np.array([30, 255, 255])
    
    red_lower1 = np.array([0, 153, 128])      # HSV: [0, 60%, 50%]
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 153, 128])
    red_upper2 = np.array([180, 255, 255])
    
    # Create masks for orange and red
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    ball_mask = cv2.bitwise_or(orange_mask, cv2.bitwise_or(red_mask1, red_mask2))
    
    # Find contours
    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (likely the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Filter by size (reasonable ball size)
        if 50 < area < 5000:  # Ball should be between 50-5000 pixels
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            if 5 < radius < 120:  # Reasonable radius
                return True, int(x), int(y), int(radius)
    
    return False, 0, 0, 0

def detect_rim_brightness_fallback(frame):
    """
    Brightness-based detection for rim (fallback when YOLO fails)
    Detects bright white/metallic rim
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame.shape[:2]
    
    # Look for bright regions in upper portion of frame (rim is usually at top)
    roi_top = int(height * 0.1)
    roi_bottom = int(height * 0.6)
    roi = gray[roi_top:roi_bottom, :]
    
    # Threshold for bright white/metallic (value > 70% of 255 = 178)
    _, bright_mask = cv2.threshold(roi, 178, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest bright contour in upper region
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Filter by size (rim should be reasonably sized)
        if 200 < area < 20000:  # Rim should be between 200-20000 pixels
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            y += roi_top  # Adjust y coordinate for ROI offset
            if 10 < radius < 150:  # Reasonable radius
                return True, int(x), int(y), int(radius)
    
    return False, 0, 0, 0

def detect_ball_and_rim_yolo8(frame):
    """
    Use the original detector EXACTLY as it works in the other window
    Use tracker_module.detector.get_frame() and get_stats() - same as original routes
    """
    # Check if tracker_module is available
    if not YOLOV8_AVAILABLE or 'tracker_module' not in globals():
        return None
    
    # Lazy initialization - only initialize when first needed
    if not hasattr(tracker_module, 'detector') or tracker_module.detector is None:
        initialize_shot_detector()
    
    if not hasattr(tracker_module, 'detector') or tracker_module.detector is None or frame_queue is None:
        return None
    
    try:
        detector = tracker_module.detector
        
        # Feed frame into detector (via frame feeder)
        # The detector's process_video() background thread reads from cap.read() continuously
        frame_queue.push_frame(frame)
        
        # Wait for detector to process the frame
        # The detector's process_video() loop will read this frame and process it
        # IMPORTANT: The detector needs time to:
        # 1. Read frame from FrameFeeder
        # 2. Run YOLO detection
        # 3. Draw visualizations (yellow lines, corner rectangles, shot counter)
        # 4. Store in current_frame
        import time
        initial_frame_count = detector.frame_count
        # Wait longer for processing - detector needs time to draw visualizations
        for _ in range(100):  # Check 100 times over 1 second
            time.sleep(0.01)
            if detector.frame_count > initial_frame_count:
                # Frame was processed, but wait a bit more for visualizations to be drawn
                time.sleep(0.05)  # Extra 50ms for visualization drawing
                break  # Frame was processed
        
        # Use the original detector EXACTLY as original routes do:
        # - tracker_module.detector.get_frame() (line 1770 in original)
        # - tracker_module.detector.get_stats() (line 1781 in original)
        
        # Get processed frame with all visualizations (exactly as original route does)
        # The detector's process_video() loop has already drawn:
        # - Yellow lines from rim to player (cv2.line with color (0, 255, 255))
        # - Corner rectangles around ball and rim (cvzone.cornerRect)
        # - Shot counter and make/miss overlay (display_score method)
        frame_bytes = detector.get_frame()
        if frame_bytes:
            # Decode to get the frame with all visualizations
            nparr = np.frombuffer(frame_bytes, np.uint8)
            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if processed_frame is None:
                # Fallback if decoding fails
                processed_frame = frame.copy()
        else:
            # If get_frame() returns None, use original frame (detector hasn't processed yet)
            processed_frame = frame.copy()
        
        # Get stats using detector's get_stats() method (exactly as original route line 1781)
        stats = detector.get_stats()
        
        # Extract detection info from detector state (same as original)
        ball_detected = len(detector.ball_pos) > 0
        ball_x, ball_y, ball_radius = 0, 0, 0
        ball_box = None
        if detector.ball_pos:
            latest_ball = detector.ball_pos[-1]
            ball_x, ball_y = latest_ball[0]
            ball_radius = max(latest_ball[2], latest_ball[3]) // 2
            ball_box = [ball_x - latest_ball[2]//2, ball_y - latest_ball[3]//2,
                       ball_x + latest_ball[2]//2, ball_y + latest_ball[3]//2]
        
        rim_detected = len(detector.hoop_pos) > 0
        rim_x, rim_y, rim_radius = 0, 0, 0
        rim_box = None
        if detector.hoop_pos:
            latest_rim = detector.hoop_pos[-1]
            rim_x, rim_y = latest_rim[0]
            rim_radius = max(latest_rim[2], latest_rim[3]) // 2
            rim_box = [rim_x - latest_rim[2]//2, rim_y - latest_rim[3]//2,
                      rim_x + latest_rim[2]//2, rim_y + latest_rim[3]//2]
        
        # Get shot result from overlay_text (exactly as original)
        shot_result = None
        if detector.overlay_text == "Make":
                                    shot_result = 'make'
        elif detector.overlay_text == "Miss":
                                    shot_result = 'miss'
                                
        # Encode processed frame
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'ball_detected': ball_detected,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'ball_radius': ball_radius,
            'ball_box': ball_box,
            'rim_detected': rim_detected,
            'rim_x': rim_x,
            'rim_y': rim_y,
            'rim_radius': rim_radius,
            'rim_box': rim_box,
            'shot_result': shot_result,
            'makes': stats['makes'],
            'attempts': stats['attempts'],
            'processed_frame': frame_base64
        }
    except Exception as e:
        print(f"ShotDetectorWeb detection error: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_ball_and_rim(frame):
    """
    Use ShotDetectorWeb class directly from AI-Basketball-Shot-Detection-Tracker
    This is the primary and only detection method - uses the original code without modifications
    """
    # Use original detector (AI-Basketball-Shot-Detection-Tracker) - same as original
    if YOLOV8_AVAILABLE and hasattr(tracker_module, 'detector') and tracker_module.detector is not None:
        result = detect_ball_and_rim_yolo8(frame)
        if result:
            return result
    
    # If ShotDetectorWeb not available, return empty detection
        return {
        'ball_detected': False,
        'ball_x': 0, 'ball_y': 0, 'ball_radius': 0,
        'rim_detected': False,
        'rim_x': 0, 'rim_y': 0, 'rim_radius': 0,
        'shot_result': None,
        'makes': 0,
        'attempts': 0
    }
    
    # Using only AI-Basketball-Shot-Detection-Tracker (YOLOv8)

# Shot detection is now handled inside detect_ball_and_rim_yolo8 (AI-Basketball-Shot-Detection-Tracker approach)

# Use ORIGINAL routes from AI-Basketball-Shot-Detection-Tracker - NO MODIFICATIONS
# The original code has these routes defined in tracker_module.app
# We'll proxy them to our app
if YOLOV8_AVAILABLE and tracker_module:
    # Get the original Flask app from the module
    original_app = tracker_module.app
    
    # Register original routes by copying them
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """ORIGINAL upload route - EXACT COPY from shot_detector_web_simple.py"""
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if file and tracker_module.allowed_file(file.filename):
            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                
                # Initialize detector if it's None, or switch video if already running
                # EXACT COPY from original shot_detector_web_simple.py line 1792-1832
                if tracker_module.detector is None:
                    # Monkey-patch sys.exit during initialization
                    import sys
                    original_exit = sys.exit
                    def safe_exit(code=0):
                        if code != 0:
                            raise Exception(f"ShotDetectorWeb tried to exit with code {code}")
                        original_exit(code)
                    sys.exit = safe_exit
                    try:
                        tracker_module.detector = ShotDetectorWeb(video_path=filepath, use_webcam=False)
                        print(f"✓ ShotDetectorWeb initialized with uploaded video: {filepath}")
                    finally:
                        sys.exit = original_exit  # Restore sys.exit
                else:
                    if tracker_module.detector.switch_video(filepath):
                        print(f"✓ ShotDetectorWeb switched video to: {filepath}")
                    else:
                        return jsonify({'success': False, 'message': 'Failed to load video'}), 500
                
                return jsonify({'success': True, 'message': 'Video uploaded and processing started'})
            except Exception as e:
                print(f"Error during video upload or detector initialization: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'message': f'Error saving file or initializing detector: {str(e)}'}), 500
        else:
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, or WEBM'}), 400
    
    @app.route('/api/video_feed')
    def video_feed():
        """ORIGINAL video_feed route - EXACT COPY"""
        if tracker_module.detector is None:
            # Return placeholder if detector not initialized
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Upload a video to start", (100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', placeholder)
            from flask import Response
            return Response(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n', 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        def generate():
            while True:
                frame = tracker_module.detector.get_frame()
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    time.sleep(0.1)
        from flask import Response
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/api/stats')
    def stats():
        """ORIGINAL stats route - EXACT COPY"""
        if tracker_module.detector is None:
            return jsonify({'makes': 0, 'attempts': 0, 'overlay_text': 'Waiting...', 'overlay_color': [0, 0, 0], 'heatmap': []})
        return jsonify(tracker_module.detector.get_stats())

@app.route('/api/detect_shot', methods=['POST'])
def detect_shot():
    """Compatibility endpoint - uses ORIGINAL detector EXACTLY as original code"""
    try:
        # Check if original detector exists - it's initialized when video is uploaded
        if not YOLOV8_AVAILABLE or not tracker_module:
            return jsonify({'error': 'AI-Basketball-Shot-Detection-Tracker not available'}), 500
        
        # Use original detector variable - EXACTLY as original code does
        if not hasattr(tracker_module, 'detector') or tracker_module.detector is None:
            return jsonify({'error': 'Detector not initialized. Upload a video first.'}), 500
        
        # Use original detector - NO MODIFICATIONS
        detector = tracker_module.detector
        
        # Use original get_frame() - EXACT COPY from original /video_feed route
        frame_bytes = detector.get_frame()
        if not frame_bytes:
            return jsonify({'error': 'No frame available'}), 500
        
        # Use original get_stats() - EXACT COPY from original /stats route  
        stats = detector.get_stats()
        
        # Extract info from detector state (same as original code does)
        ball_detected = len(detector.ball_pos) > 0
        ball_x, ball_y, ball_radius = 0, 0, 0
        ball_box = None
        if detector.ball_pos:
            latest_ball = detector.ball_pos[-1]
            ball_x, ball_y = latest_ball[0]
            ball_radius = max(latest_ball[2], latest_ball[3]) // 2
            ball_box = [ball_x - latest_ball[2]//2, ball_y - latest_ball[3]//2,
                       ball_x + latest_ball[2]//2, ball_y + latest_ball[3]//2]
        
        rim_detected = len(detector.hoop_pos) > 0
        rim_x, rim_y, rim_radius = 0, 0, 0
        rim_box = None
        if detector.hoop_pos:
            latest_rim = detector.hoop_pos[-1]
            rim_x, rim_y = latest_rim[0]
            rim_radius = max(latest_rim[2], latest_rim[3]) // 2
            rim_box = [rim_x - latest_rim[2]//2, rim_y - latest_rim[3]//2,
                      rim_x + latest_rim[2]//2, rim_y + latest_rim[3]//2]
        
        shot_result = None
        if detector.overlay_text == "Make":
                            shot_result = 'make'
        elif detector.overlay_text == "Miss":
                            shot_result = 'miss'
                        
        # Frame is already JPEG bytes from get_frame() - encode to base64
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'tracker': 'AI-Basketball-Shot-Detection-Tracker (YOLOv8)',
            'ball_detected': ball_detected,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'ball_radius': ball_radius,
            'ball_box': ball_box,
            'rim_detected': rim_detected,
            'rim_x': rim_x,
            'rim_y': rim_y,
            'rim_radius': rim_radius,
            'rim_box': rim_box,
            'shot_result': shot_result,
            'makes': stats['makes'],
            'attempts': stats['attempts'],
            'processed_frame': frame_base64  # Frame with ALL original visualizations
        })
    except Exception as e:
        print(f"Error in detect_shot: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset_shot_tracker', methods=['POST'])
def reset_shot_tracker():
    """Reset shot tracker state - use original detector"""
    if hasattr(tracker_module, 'detector') and tracker_module.detector:
        detector = tracker_module.detector
        detector.ball_pos = []
        detector.hoop_pos = []
        detector.frame_count = 0
        detector.makes = 0
        detector.attempts = 0
        detector.up = False
        detector.down = False
        detector.up_frame = 0
        detector.down_frame = 0
        detector.overlay_text = "Waiting..."
        detector.overlay_color = (0, 0, 0)
        detector.shot_heatmap = []
    return jsonify({'success': True})

@app.route('/api/get_shot_heatmap_data', methods=['POST'])
def get_shot_heatmap_data():
    """Get shot locations for heatmap visualization from ShotDetectorWeb."""
    try:
        data = request.json or {}
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Get heatmap data from original detector (same as original)
        if hasattr(tracker_module, 'detector') and tracker_module.detector:
            # Use original detector's get_stats() method which includes heatmap (same as original)
            stats = tracker_module.detector.get_stats()
            heatmap_data = stats.get('heatmap', [])
            
            # Convert to format expected by frontend
            # ShotDetectorWeb stores: {'x': normalized_x, 'y': court_y, 'is_make': is_make}
            shots = []
            for idx, shot in enumerate(heatmap_data):
                shots.append({
                    'x': shot.get('x', 0.5),
                    'y': shot.get('y', 0.5),
                    'timestamp': time.time() - (len(heatmap_data) - idx) * 60,  # Approximate timestamps
                    'result': 'make' if shot.get('is_make', False) else 'miss'
                })
            
            # Filter by date range if provided
            filtered_shots = shots
            if start_date and end_date:
                try:
                    start_ts = time.mktime(time.strptime(start_date, '%Y-%m-%d'))
                    end_ts = time.mktime(time.strptime(end_date, '%Y-%m-%d')) + 86400
                    filtered_shots = [
                        shot for shot in shots
                        if start_ts <= shot['timestamp'] <= end_ts
                    ]
                except:
                    # If date parsing fails, return all shots
                    filtered_shots = shots
        else:
            filtered_shots = []
        
        return jsonify({
            'success': True,
            'shots': filtered_shots
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Serve root static files (CSS, JS, images for landing page) - must be after API routes
@app.route('/<path:filename>')
def root_static(filename):
    """Serve static files from project root (for landing page)"""
    # Only serve specific file types to avoid conflicts with API routes
    static_extensions = ('.css', '.js', '.png', '.jpg', '.jpeg', '.webp', '.svg', '.mov', '.mp4', '.ico')
    if filename.endswith(static_extensions):
        try:
            return send_from_directory(str(project_root), filename)
        except:
            from flask import abort
            abort(404)
    else:
        from flask import abort
        abort(404)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
