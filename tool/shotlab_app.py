from flask import Flask, request, jsonify, send_from_directory, send_file, Response, abort, session
from flask_cors import CORS
import cv2
import numpy as np
import time
import base64
import io
import json
import smtplib
import math
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.exceptions import ClientError
import requests
from collections import defaultdict
import uuid
import mimetypes
import re
import bisect
from ultralytics import YOLO
import supervision as sv
from config import Config
from storage.s3_storage import ShotLabS3Storage
from detection.court_calibration import CourtCalibrator, COURT_LANDMARKS
from outcome_classifier import IntelligentOutcomeClassifier, estimate_camera_position
from yolo_detector import (
    YOLODetector,
    get_ball_position_at_frame,
    get_stable_rim_position,
    interpolate_ball_positions,
)
from shotlab_v5_pipeline import (
    candidate_centered_ball_windows,
    compute_dynamic_clip_window,
    compute_video_signature,
    refine_release_frame,
    resolve_court_reuse_gate,
    run_v5_candidate_pipeline,
    should_run_pose_refinement,
)
from shot_validity_model import ShotValidityModel, apply_shot_validity_filter

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

SUPERVISION_AVAILABLE = True

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("Warning: Roboflow not available. Install it with: pip install roboflow")

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
logger = logging.getLogger("shotlab_app")

# Initialize detection and storage lazily to avoid startup delays.
print("Initializing ShotLab detection models...")
shotlab_detector = None
yolo_detector = YOLODetector()
training_storage = None
training_executor = ThreadPoolExecutor(max_workers=2)
face_cascade = None
shot_validity_model = None
rtmpose_estimator_instance = None
rtmpose_estimator_error = None


# Configure upload folder (EXACT COPY from original shot_detector_web_simple.py)
app.config['UPLOAD_FOLDER'] = str(project_root / 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.secret_key = os.environ.get('SHOTLAB_SECRET_KEY', os.environ.get('FLASK_SECRET_KEY', 'shotlab-dev-secret'))

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

SHOTLAB_FRAME_STRIDE = int(os.environ.get('SHOTLAB_FRAME_STRIDE', '3'))
SHOTLAB_BALL_CONFIDENCE = float(os.environ.get('SHOTLAB_BALL_CONFIDENCE', str(Config.BALL_CONFIDENCE)))
SHOTLAB_COURT_CONFIDENCE = int(os.environ.get('SHOTLAB_COURT_CONFIDENCE', '40'))
SHOTLAB_RIM_CONFIDENCE = float(os.environ.get('SHOTLAB_RIM_CONFIDENCE', str(Config.RIM_CONFIDENCE)))
SHOTLAB_REQUEST_TIMEOUT = float(os.environ.get('SHOTLAB_REQUEST_TIMEOUT', '20'))
SHOTLAB_BALL_TIMEOUT_LIMIT = int(os.environ.get('SHOTLAB_BALL_TIMEOUT_LIMIT', '5'))
SHOTLAB_SHOT_MIN_GAP_SECONDS = float(os.environ.get('SHOTLAB_SHOT_MIN_GAP_SECONDS', '1.0'))
SHOTLAB_SHOT_MAX_DURATION_SECONDS = float(os.environ.get('SHOTLAB_SHOT_MAX_DURATION_SECONDS', '3.5'))
SHOTLAB_PRE_FOLLOW_MAX_GAP_SECONDS = float(os.environ.get('SHOTLAB_PRE_FOLLOW_MAX_GAP_SECONDS', '2.5'))
SHOTLAB_NEUTRAL_GRACE_SECONDS = float(os.environ.get('SHOTLAB_NEUTRAL_GRACE_SECONDS', '0.35'))
SHOTLAB_REQUIRE_JUMP = os.environ.get('SHOTLAB_REQUIRE_JUMP', '1') != '0'
SHOTLAB_JUMP_MIN_FRAC = float(os.environ.get('MIN_JUMP_HEIGHT', os.environ.get('SHOTLAB_JUMP_MIN_FRAC', '0.012')))
SHOTLAB_JUMP_MIN_FRAMES = int(os.environ.get('SHOTLAB_JUMP_MIN_FRAMES', '3'))
SHOTLAB_MIN_SHOT_DURATION_FRAMES = int(os.environ.get('MIN_SHOT_DURATION_FRAMES', '0'))
SHOTLAB_RELEASE_SEARCH_WINDOW_SECONDS = float(os.environ.get('SHOTLAB_RELEASE_SEARCH_WINDOW_SECONDS', '0.75'))
SHOTLAB_BALL_STRIDE = int(os.environ.get('SHOTLAB_BALL_STRIDE', '1'))
SHOTLAB_BALL_WINDOW_BEFORE_SECONDS = float(os.environ.get('SHOTLAB_BALL_WINDOW_BEFORE_SECONDS', '1.2'))
SHOTLAB_BALL_WINDOW_AFTER_SECONDS = float(os.environ.get('SHOTLAB_BALL_WINDOW_AFTER_SECONDS', '3.0'))
SHOTLAB_OUTCOME_WINDOW_AFTER_SECONDS = float(os.environ.get('SHOTLAB_OUTCOME_WINDOW_AFTER_SECONDS', '2.5'))
SHOTLAB_BALL_VALIDATE_PRE_SECONDS = float(os.environ.get('SHOTLAB_BALL_VALIDATE_PRE_SECONDS', '0.25'))
SHOTLAB_BALL_VALIDATE_POST_SECONDS = float(os.environ.get('SHOTLAB_BALL_VALIDATE_POST_SECONDS', '1.2'))
SHOTLAB_BALL_VALIDATE_MIN_POINTS = int(os.environ.get('SHOTLAB_BALL_VALIDATE_MIN_POINTS', '3'))
SHOTLAB_BALL_VALIDATE_REQUIRE_ARC = os.environ.get('SHOTLAB_BALL_VALIDATE_REQUIRE_ARC', '1') != '0'
SHOTLAB_BALL_VALIDATE_HAND_WINDOW_SECONDS = float(os.environ.get('SHOTLAB_BALL_VALIDATE_HAND_WINDOW_SECONDS', '0.2'))
SHOTLAB_BALL_VALIDATE_MAX_HAND_DIST_PX = float(os.environ.get('SHOTLAB_BALL_VALIDATE_MAX_HAND_DIST_PX', '140'))
SHOTLAB_BALL_COVERAGE_MIN = float(os.environ.get('SHOTLAB_BALL_COVERAGE_MIN', '0.30'))
SHOTLAB_DEBUG_BALL = os.environ.get('SHOTLAB_DEBUG_BALL', '0') == '1'
SHOTLAB_DEBUG_CALIBRATION = os.environ.get('SHOTLAB_DEBUG_CALIBRATION', '0') == '1'
SHOTLAB_DEBUG_REJECTED_SHOTS = os.environ.get('SHOTLAB_DEBUG_REJECTED_SHOTS', '0') == '1'
SHOTLAB_DROP_JUMP_LOW = os.environ.get('SHOTLAB_DROP_JUMP_LOW', '0') == '1'
SHOTLAB_DROP_BALL_NOT_UPWARD = os.environ.get('SHOTLAB_DROP_BALL_NOT_UPWARD', '0') == '1'
SHOTLAB_FILTER_REBOUNDS = os.environ.get('SHOTLAB_FILTER_REBOUNDS', '1') != '0'
SHOTLAB_FILTER_DRIBBLES = os.environ.get('SHOTLAB_FILTER_DRIBBLES', '1') != '0'
SHOTLAB_REBOUND_CONFIDENCE_THRESHOLD = float(os.environ.get('SHOTLAB_REBOUND_CONFIDENCE_THRESHOLD', '0.6'))
SHOTLAB_DRIBBLE_CONFIDENCE_THRESHOLD = float(os.environ.get('SHOTLAB_DRIBBLE_CONFIDENCE_THRESHOLD', '0.6'))
SHOTLAB_TRAJECTORY_VALIDATION_THRESHOLD = float(os.environ.get('SHOTLAB_TRAJECTORY_VALIDATION_THRESHOLD', '0.5'))
SHOTLAB_BALL_FLIGHT_HAND_DIST_PX = float(os.environ.get('SHOTLAB_BALL_FLIGHT_HAND_DIST_PX', '350'))
SHOTLAB_BALL_FLIGHT_SEPARATION_DIST_PX = float(os.environ.get('SHOTLAB_BALL_FLIGHT_SEPARATION_DIST_PX', '80'))
SHOTLAB_BALL_FLIGHT_SEPARATION_MIN_FRAMES = int(os.environ.get('SHOTLAB_BALL_FLIGHT_SEPARATION_MIN_FRAMES', '1'))
SHOTLAB_BALL_FLIGHT_TOWARD_RIM_RATIO = float(os.environ.get('SHOTLAB_BALL_FLIGHT_TOWARD_RIM_RATIO', '0.8'))
SHOTLAB_BALL_FLIGHT_SIZE_RATIO_MIN = float(os.environ.get('SHOTLAB_BALL_FLIGHT_SIZE_RATIO_MIN', '1.4'))
SHOTLAB_BALL_FLIGHT_NEAR_HAND_PRE_FRAMES = int(os.environ.get('SHOTLAB_BALL_FLIGHT_NEAR_HAND_PRE_FRAMES', '20'))
SHOTLAB_BALL_FLIGHT_NEAR_HAND_POST_FRAMES = int(os.environ.get('SHOTLAB_BALL_FLIGHT_NEAR_HAND_POST_FRAMES', '15'))
SHOTLAB_BALL_FLIGHT_POST_START_FRAMES = int(os.environ.get('SHOTLAB_BALL_FLIGHT_POST_START_FRAMES', '3'))
SHOTLAB_BALL_FLIGHT_POST_END_FRAMES = int(os.environ.get('SHOTLAB_BALL_FLIGHT_POST_END_FRAMES', '45'))
SHOTLAB_BALL_FLIGHT_DISAPPEARANCE_GAP_FRAMES = int(os.environ.get('SHOTLAB_BALL_FLIGHT_DISAPPEARANCE_GAP_FRAMES', '4'))
SHOTLAB_BALL_FLIGHT_BIG_MOVE_OVERRIDE_PX = float(os.environ.get('SHOTLAB_BALL_FLIGHT_BIG_MOVE_OVERRIDE_PX', '100'))
SHOTLAB_BALL_FLIGHT_MIN_DIRECTED_DISPLACEMENT_PX = float(
    os.environ.get('SHOTLAB_BALL_FLIGHT_MIN_DIRECTED_DISPLACEMENT_PX', '50')
)
SHOTLAB_BALL_FLIGHT_MIN_COVERAGE = float(os.environ.get('SHOTLAB_BALL_FLIGHT_MIN_COVERAGE', '0.30'))
SHOTLAB_NET_ZONE_PAD_X = float(os.environ.get('SHOTLAB_NET_ZONE_PAD_X', '0.2'))
SHOTLAB_NET_ZONE_PAD_Y = float(os.environ.get('SHOTLAB_NET_ZONE_PAD_Y', '0.1'))
SHOTLAB_NET_ZONE_PAD_MIN_PX = float(os.environ.get('SHOTLAB_NET_ZONE_PAD_MIN_PX', '4'))
SHOTLAB_USE_INTELLIGENT_OUTCOME = os.environ.get('SHOTLAB_USE_INTELLIGENT_OUTCOME', '1') != '0'
SHOTLAB_RIM_FALLBACK_WIDTH_RATIO = float(os.environ.get('SHOTLAB_RIM_FALLBACK_WIDTH_RATIO', '0.55'))
SHOTLAB_RIM_FALLBACK_HEIGHT_RATIO = float(os.environ.get('SHOTLAB_RIM_FALLBACK_HEIGHT_RATIO', '0.35'))
SHOTLAB_RIM_FALLBACK_Y_OFFSET_RATIO = float(os.environ.get('SHOTLAB_RIM_FALLBACK_Y_OFFSET_RATIO', '0.05'))
SHOTLAB_POSE_MIN_VISIBILITY = float(os.environ.get('POSE_LANDMARK_MIN_VISIBILITY', '0.3'))
SHOTLAB_POSE_STRIDE = int(os.environ.get('SHOTLAB_POSE_STRIDE', str(SHOTLAB_FRAME_STRIDE)))
SHOTLAB_POSE_DETECTION_STRIDE = int(os.environ.get('SHOTLAB_POSE_DETECTION_STRIDE', '2'))
SHOTLAB_DEBUG_POSE = os.environ.get('SHOTLAB_DEBUG_POSE', '0') == '1'
SHOTLAB_DEBUG_FORM_SCORE = os.environ.get('SHOTLAB_DEBUG_FORM_SCORE', '0') == '1'
SHOTLAB_INTERPOLATE_ANGLES = os.environ.get('SHOTLAB_INTERPOLATE_ANGLES', '0') == '1'
SHOTLAB_WRIST_FALLBACK_ANGLE = float(os.environ.get('SHOTLAB_WRIST_FALLBACK_ANGLE', '150.0'))
SHOTLAB_SHOT_DEDUPE_SECONDS = float(os.environ.get('SHOTLAB_SHOT_DEDUPE_SECONDS', '2.0'))
SHOTLAB_REQUIRE_PRE_SHOT = os.environ.get('SHOTLAB_REQUIRE_PRE_SHOT', '0') != '0'
SHOTLAB_CLIP_OVERLAY = os.environ.get('SHOTLAB_CLIP_OVERLAY', '1') != '0'
SHOTLAB_CLIP_TRAIL_FRAMES = int(os.environ.get('SHOTLAB_CLIP_TRAIL_FRAMES', '14'))
SHOTLAB_CALIBRATION_SAMPLES = int(os.environ.get('SHOTLAB_CALIBRATION_SAMPLES', '18'))
SHOTLAB_CALIBRATION_MIN_KEYPOINTS = int(os.environ.get('SHOTLAB_CALIBRATION_MIN_KEYPOINTS', '4'))
SHOTLAB_RIM_MIN_CONFIDENCE = float(os.environ.get('SHOTLAB_RIM_MIN_CONFIDENCE', str(Config.RIM_CONFIDENCE)))
SHOTLAB_ALLOW_FOLLOW_ONLY = os.environ.get('SHOTLAB_ALLOW_FOLLOW_ONLY', '0') != '0'
SHOTLAB_FOLLOW_ONLY_LOOKBACK_SECONDS = float(os.environ.get('SHOTLAB_FOLLOW_ONLY_LOOKBACK_SECONDS', '1.5'))
SHOTLAB_MIN_PRESHOT_ENTRIES_FOR_FULL_GATING = int(
    os.environ.get('SHOTLAB_MIN_PRESHOT_ENTRIES_FOR_FULL_GATING', '5')
)
SHOTLAB_POSE_WARMUP_FOLLOW_ONLY_SECONDS = float(
    os.environ.get('SHOTLAB_POSE_WARMUP_FOLLOW_ONLY_SECONDS', '3.0')
)
SHOTLAB_FOLLOW_SEGMENT_GAP_SECONDS = float(os.environ.get('SHOTLAB_FOLLOW_SEGMENT_GAP_SECONDS', '0.3'))
SHOTLAB_YOLO_LAUNCH_ENABLE = os.environ.get('SHOTLAB_YOLO_LAUNCH_ENABLE', '1') != '0'
SHOTLAB_YOLO_LAUNCH_MIN_TRACK_FRAMES = int(os.environ.get('SHOTLAB_YOLO_LAUNCH_MIN_TRACK_FRAMES', '10'))
SHOTLAB_YOLO_LAUNCH_MIN_CONFIDENCE = float(os.environ.get('SHOTLAB_YOLO_LAUNCH_MIN_CONFIDENCE', '0.30'))
SHOTLAB_YOLO_LAUNCH_WINDOW_FRAMES = int(os.environ.get('SHOTLAB_YOLO_LAUNCH_WINDOW_FRAMES', '20'))
SHOTLAB_YOLO_LAUNCH_UPWARD_DELTA_PX = float(os.environ.get('SHOTLAB_YOLO_LAUNCH_UPWARD_DELTA_PX', '3.0'))
SHOTLAB_YOLO_LAUNCH_SHRINK_RATIO = float(os.environ.get('SHOTLAB_YOLO_LAUNCH_SHRINK_RATIO', '0.90'))
SHOTLAB_YOLO_LAUNCH_MIN_UPWARD_COUNT = int(os.environ.get('SHOTLAB_YOLO_LAUNCH_MIN_UPWARD_COUNT', '5'))
SHOTLAB_YOLO_LAUNCH_MIN_SHRINK_COUNT = int(os.environ.get('SHOTLAB_YOLO_LAUNCH_MIN_SHRINK_COUNT', '4'))
SHOTLAB_YOLO_LAUNCH_COOLDOWN_SECONDS = float(os.environ.get('SHOTLAB_YOLO_LAUNCH_COOLDOWN_SECONDS', '2.8'))
SHOTLAB_YOLO_LAUNCH_MERGE_GAP_FRAMES = int(os.environ.get('SHOTLAB_YOLO_LAUNCH_MERGE_GAP_FRAMES', '70'))
SHOTLAB_YOLO_LAUNCH_MIN_START_FRAME = int(os.environ.get('SHOTLAB_YOLO_LAUNCH_MIN_START_FRAME', '45'))
SHOTLAB_YOLO_MAX_ACCEPTED_SHARE = float(os.environ.get('SHOTLAB_YOLO_MAX_ACCEPTED_SHARE', '0.35'))
SHOTLAB_EVENT_MODEL_ENABLE = os.environ.get('SHOTLAB_EVENT_MODEL_ENABLE', '1') != '0'
SHOTLAB_EVENT_MODEL_PATH = os.environ.get(
    'SHOTLAB_EVENT_MODEL_PATH',
    str(Path(__file__).parent / 'shotlab_event_detection_best.pt')
)
SHOTLAB_EVENT_SHOOT_CLASS_ID = int(os.environ.get('SHOTLAB_EVENT_SHOOT_CLASS_ID', '6'))
SHOTLAB_EVENT_MADE_CLASS_ID = int(os.environ.get('SHOTLAB_EVENT_MADE_CLASS_ID', '2'))
SHOTLAB_EVENT_CANDIDATE_CONFIDENCE = float(os.environ.get('SHOTLAB_EVENT_CANDIDATE_CONFIDENCE', '0.40'))
SHOTLAB_EVENT_GAP_CONFIDENCE = float(os.environ.get('SHOTLAB_EVENT_GAP_CONFIDENCE', '0.50'))
SHOTLAB_EVENT_GAP_MIN_FRAMES = int(os.environ.get('SHOTLAB_EVENT_GAP_MIN_FRAMES', '90'))
SHOTLAB_EVENT_GAP_SAMPLE_STRIDE = int(os.environ.get('SHOTLAB_EVENT_GAP_SAMPLE_STRIDE', '3'))
SHOTLAB_EVENT_MERGE_GAP_FRAMES = int(os.environ.get('SHOTLAB_EVENT_MERGE_GAP_FRAMES', '45'))
SHOTLAB_EVENT_BATCH_SIZE = int(os.environ.get('SHOTLAB_EVENT_BATCH_SIZE', '12'))
SHOTLAB_EVENT_MAX_FRAMES_PER_VIDEO = int(os.environ.get('SHOTLAB_EVENT_MAX_FRAMES_PER_VIDEO', '64'))
SHOTLAB_EVENT_MAX_CANDIDATE_FRAMES_PER_VIDEO = int(
    os.environ.get('SHOTLAB_EVENT_MAX_CANDIDATE_FRAMES_PER_VIDEO', '48')
)
SHOTLAB_EVENT_MAX_GAP_FRAMES_PER_VIDEO = int(os.environ.get('SHOTLAB_EVENT_MAX_GAP_FRAMES_PER_VIDEO', '16'))
SHOTLAB_EVENT_CANDIDATE_NEIGHBORHOOD_FRAMES = int(
    os.environ.get('SHOTLAB_EVENT_CANDIDATE_NEIGHBORHOOD_FRAMES', '2')
)
SHOTLAB_EVENT_CANDIDATE_NEIGHBOR_RADIUS_FRAMES = int(
    os.environ.get('SHOTLAB_EVENT_CANDIDATE_NEIGHBOR_RADIUS_FRAMES', '6')
)
SHOTLAB_EVENT_MAX_SECONDS_PER_VIDEO = float(os.environ.get('SHOTLAB_EVENT_MAX_SECONDS_PER_VIDEO', '15.0'))
SHOTLAB_EVENT_MADE_SHADOW_ENABLE = os.environ.get('SHOTLAB_EVENT_MADE_SHADOW_ENABLE', '0') != '0'
SHOTLAB_EVENT_MADE_SHADOW_WINDOW_FRAMES = int(os.environ.get('SHOTLAB_EVENT_MADE_SHADOW_WINDOW_FRAMES', '8'))
SHOTLAB_EVENT_MADE_SHADOW_STRIDE = int(os.environ.get('SHOTLAB_EVENT_MADE_SHADOW_STRIDE', '2'))
SHOTLAB_EVENT_MADE_SHADOW_MIN_CONFIDENCE = float(os.environ.get('SHOTLAB_EVENT_MADE_SHADOW_MIN_CONFIDENCE', '0.35'))
SHOTLAB_EVENT_MADE_SHADOW_MAX_FRAMES = int(os.environ.get('SHOTLAB_EVENT_MADE_SHADOW_MAX_FRAMES', '120'))
SHOTLAB_YOLO_CONFIRM_NEAR_SHOOTER_MAX_DIST_PX = float(
    os.environ.get('SHOTLAB_YOLO_CONFIRM_NEAR_SHOOTER_MAX_DIST_PX', '235')
)
SHOTLAB_YOLO_CONFIRM_NEAR_SHOOTER_MIN_FRAMES = int(
    os.environ.get('SHOTLAB_YOLO_CONFIRM_NEAR_SHOOTER_MIN_FRAMES', '1')
)
SHOTLAB_YOLO_CONFIRM_POSE_GAP_FRAMES = int(
    os.environ.get('SHOTLAB_YOLO_CONFIRM_POSE_GAP_FRAMES', '3')
)
SHOTLAB_SIDELINE_PRE_FOLLOW_GAP_MULT = float(
    os.environ.get('SHOTLAB_SIDELINE_PRE_FOLLOW_GAP_MULT', '1.5')
)
SHOTLAB_SIDELINE_MIN_GAP_MULT = float(
    os.environ.get('SHOTLAB_SIDELINE_MIN_GAP_MULT', '0.75')
)
SHOTLAB_SIDELINE_ALLOW_FOLLOW_ONLY = os.environ.get('SHOTLAB_SIDELINE_ALLOW_FOLLOW_ONLY', '1') != '0'
SHOTLAB_POSE_SECOND_CHANCE_ENABLE = os.environ.get('SHOTLAB_POSE_SECOND_CHANCE_ENABLE', '1') != '0'
SHOTLAB_POSE_SECOND_CHANCE_MAX = int(os.environ.get('SHOTLAB_POSE_SECOND_CHANCE_MAX', '1'))
SHOTLAB_YOLO_FRAME_SKIP = int(os.environ.get('SHOTLAB_YOLO_FRAME_SKIP', '2'))
# Behind-basket MVP overrides (only intentionally exposed knobs for this mode).
SHOTLAB_BEHIND_BASKET_YOLO_FRAME_SKIP = int(os.environ.get('SHOTLAB_BEHIND_BASKET_YOLO_FRAME_SKIP', '2'))
SHOTLAB_BEHIND_BASKET_MIN_GAP_SECONDS = float(os.environ.get('SHOTLAB_BEHIND_BASKET_MIN_GAP_SECONDS', '1.5'))
SHOTLAB_BEHIND_BASKET_THREE_RIM_MULT = float(os.environ.get('SHOTLAB_BEHIND_BASKET_THREE_RIM_MULT', '2.5'))
SHOTLAB_RTMPOSE_MODEL_PATH = os.environ.get(
    'SHOTLAB_RTMPOSE_MODEL_PATH',
    str(Path(__file__).parent / 'rtmpose_m_coco.onnx')
)
SHOTLAB_RTMPOSE_AUTO_DOWNLOAD = os.environ.get('SHOTLAB_RTMPOSE_AUTO_DOWNLOAD', '1') != '0'

# Behind-basket MVP constants (kept code-level for simplicity).
BB_CENTER_BAND_RATIO = 0.12
BB_POSE_FALLBACK_MIN_CANDIDATES = 3
BB_POSE_FALLBACK_STRIDE = 3
BB_MAX_BALL_FIRST_RECOVERY_CANDIDATES = 10
BB_FORM_POSE_WINDOW_BEFORE_FRAMES = 30
BB_FORM_POSE_WINDOW_AFTER_FRAMES = 45
BB_FORM_SCORING_MIN_VISIBILITY = 0.25
BB_RTMPOSE_MIN_PLAYER_WIDTH_RATIO = 0.04
BB_RTMPOSE_MIN_PLAYER_BBOX_ASPECT_RATIO = 1.0
BB_RTMPOSE_MAX_PLAYER_BBOX_ASPECT_RATIO = 4.0
BB_RTMPOSE_STATIC_BBOX_LOOKBACK_FRAMES = 30
BB_RTMPOSE_STATIC_CENTER_MIN_MOTION_RATIO = 0.03
BB_RTMPOSE_PLAYER_FRAME_SEARCH_GAP = int(os.environ.get('BB_RTMPOSE_PLAYER_FRAME_SEARCH_GAP', '3'))
BB_SKELETON_KEYPOINT_INDICES = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}
BB_SKELETON_WINDOW_FRAMES = 25
BB_MIN_CLIP_AFTER_SECONDS = 4.2
BB_CLASSIFICATION_AFTER_FRAMES = 120
BB_SHOOT_FRAME_ANCHOR_MIN_CONF = 0.55
BB_SHOOT_CONFIRMED_REBOUND_KEEP_CONFIDENCE = 0.60
BB_LAYUP_RIM_DISTANCE_MULT = 1.6
SHOTLAB_FORM_SCORE_CALIBRATION_MEAN = float(os.environ.get('SHOTLAB_FORM_SCORE_CALIBRATION_MEAN', '97.024'))
SHOTLAB_FORM_SCORE_CALIBRATION_STD = float(os.environ.get('SHOTLAB_FORM_SCORE_CALIBRATION_STD', '21.839'))
SHOTLAB_FORM_SCORE_CALIBRATION_SCALE = float(os.environ.get('SHOTLAB_FORM_SCORE_CALIBRATION_SCALE', '15.0'))
SHOTLAB_ACTIVE_REGION_ENABLE = os.environ.get('SHOTLAB_ACTIVE_REGION_ENABLE', '1') != '0'
SHOTLAB_ACTIVE_REGION_MIN_GAP_FRAMES = int(os.environ.get('SHOTLAB_ACTIVE_REGION_MIN_GAP_FRAMES', '30'))
SHOTLAB_ACTIVE_REGION_MIN_LENGTH_FRAMES = int(os.environ.get('SHOTLAB_ACTIVE_REGION_MIN_LENGTH_FRAMES', '15'))
SHOTLAB_ACTIVE_REGION_PAD_BEFORE = int(os.environ.get('SHOTLAB_ACTIVE_REGION_PAD_BEFORE', '30'))
SHOTLAB_ACTIVE_REGION_PAD_AFTER = int(os.environ.get('SHOTLAB_ACTIVE_REGION_PAD_AFTER', '60'))
SHOTLAB_ACTIVE_REGION_MIN_BALL_MOTION_PX = float(os.environ.get('SHOTLAB_ACTIVE_REGION_MIN_BALL_MOTION_PX', '4.0'))
SHOTLAB_POSE_REFINE_WINDOW_BEFORE_FRAMES = int(os.environ.get('SHOTLAB_POSE_REFINE_WINDOW_BEFORE_FRAMES', '30'))
SHOTLAB_POSE_REFINE_WINDOW_AFTER_FRAMES = int(os.environ.get('SHOTLAB_POSE_REFINE_WINDOW_AFTER_FRAMES', '30'))
SHOTLAB_POSE_REFINE_ENABLE = os.environ.get('SHOTLAB_POSE_REFINE_ENABLE', '1') != '0'
SHOTLAB_UPWARD_ARC_MIN_APEX_DELTA_PX = float(os.environ.get('SHOTLAB_UPWARD_ARC_MIN_APEX_DELTA_PX', '55'))
SHOTLAB_UPWARD_ARC_HIGH_START_RATIO = float(os.environ.get('SHOTLAB_UPWARD_ARC_HIGH_START_RATIO', '0.45'))
SHOTLAB_UPWARD_ARC_MID_FLIGHT_FALL_DELTA_PX = float(os.environ.get('SHOTLAB_UPWARD_ARC_MID_FLIGHT_FALL_DELTA_PX', '100'))
SHOTLAB_CLIP_BEFORE_SECONDS = float(os.environ.get('SHOTLAB_CLIP_BEFORE_SECONDS', '1.0'))
SHOTLAB_CLIP_AFTER_SECONDS = float(os.environ.get('SHOTLAB_CLIP_AFTER_SECONDS', '3.0'))
SHOTLAB_CLIP_MAX_DIM = int(os.environ.get('SHOTLAB_CLIP_MAX_DIM', '720'))
SHOTLAB_CLIP_PREGENERATE = os.environ.get('SHOTLAB_CLIP_PREGENERATE', '1') != '0'
SHOTLAB_CLIP_DEFER = os.environ.get('SHOTLAB_CLIP_DEFER', '1') != '0'
SHOTLAB_USE_VELOCITY_CANDIDATES = os.environ.get('SHOTLAB_USE_VELOCITY_CANDIDATES', '1') != '0'
SHOTLAB_SHOT_DETECTION_MODE = os.environ.get('SHOTLAB_SHOT_DETECTION_MODE', 'state_machine')

_EVENT_ANGLE_KEYS = ('back', 'front', 'leftside', 'rightside', 'unknown')
_DEFAULT_EVENT_CANDIDATE_CONF_BY_ANGLE = {
    'back': 0.38,
    'front': 0.40,
    'leftside': 0.46,
    'rightside': 0.44,
    'unknown': 0.45,
}
_DEFAULT_EVENT_GAP_ENABLE_BY_ANGLE = {
    'back': True,
    'front': True,
    'leftside': False,
    'rightside': False,
    'unknown': False,
}


def _normalize_event_angle_key(value):
    raw = str(value or '').strip().lower().replace('-', '_').replace(' ', '_')
    alias_map = {
        'behind_shooter': 'back',
        'in_front_of_shooter': 'front',
        'behind_basket': 'front',
        'behind': 'back',
        'back': 'back',
        'front': 'front',
        'front_shooter': 'front',
        'sideline_left': 'leftside',
        'leftside': 'leftside',
        'left_side': 'leftside',
        'left': 'leftside',
        'sideline_right': 'rightside',
        'rightside': 'rightside',
        'right_side': 'rightside',
        'right': 'rightside',
        'unknown': 'unknown',
    }
    return alias_map.get(raw, 'unknown')


def _camera_position_to_event_angle(camera_position):
    return _normalize_event_angle_key(camera_position)


def _normalize_event_camera_angle_override(value):
    raw = str(value or '').strip()
    if not raw:
        return None
    normalized = _normalize_event_angle_key(raw)
    return normalized if normalized in _EVENT_ANGLE_KEYS else None


def _parse_event_angle_float_map(raw, default_map):
    parsed = dict(default_map or {})
    if not raw:
        return parsed

    value = str(raw).strip()
    data = None
    if value.startswith('{') and value.endswith('}'):
        try:
            data = json.loads(value)
        except Exception:
            data = None
    if isinstance(data, dict):
        items = data.items()
    else:
        items = []
        for chunk in value.split(','):
            token = chunk.strip()
            if not token:
                continue
            if '=' in token:
                k, v = token.split('=', 1)
            elif ':' in token:
                k, v = token.split(':', 1)
            else:
                continue
            items.append((k.strip(), v.strip()))

    for key_raw, val_raw in items:
        key = _normalize_event_angle_key(key_raw)
        try:
            parsed[key] = float(val_raw)
        except Exception:
            continue

    for key in _EVENT_ANGLE_KEYS:
        if key not in parsed:
            parsed[key] = float(default_map.get(key, SHOTLAB_EVENT_CANDIDATE_CONFIDENCE))

    return parsed


def _parse_event_angle_bool_map(raw, default_map):
    parsed = dict(default_map or {})
    if not raw:
        return parsed

    value = str(raw).strip()
    data = None
    if value.startswith('{') and value.endswith('}'):
        try:
            data = json.loads(value)
        except Exception:
            data = None
    if isinstance(data, dict):
        items = data.items()
    else:
        items = []
        for chunk in value.split(','):
            token = chunk.strip()
            if not token:
                continue
            if '=' in token:
                k, v = token.split('=', 1)
            elif ':' in token:
                k, v = token.split(':', 1)
            else:
                continue
            items.append((k.strip(), v.strip()))

    for key_raw, val_raw in items:
        key = _normalize_event_angle_key(key_raw)
        val_s = str(val_raw).strip().lower()
        parsed[key] = val_s in ('1', 'true', 'yes', 'y', 'on')

    for key in _EVENT_ANGLE_KEYS:
        if key not in parsed:
            parsed[key] = bool(default_map.get(key, True))

    return parsed


SHOTLAB_EVENT_CANDIDATE_CONF_BY_ANGLE = _parse_event_angle_float_map(
    os.environ.get('SHOTLAB_EVENT_CANDIDATE_CONF_BY_ANGLE', ''),
    _DEFAULT_EVENT_CANDIDATE_CONF_BY_ANGLE
)
_default_gap_conf_by_angle = {
    key: min(0.65, float(val) + 0.10)
    for key, val in SHOTLAB_EVENT_CANDIDATE_CONF_BY_ANGLE.items()
}
SHOTLAB_EVENT_GAP_CONF_BY_ANGLE = _parse_event_angle_float_map(
    os.environ.get('SHOTLAB_EVENT_GAP_CONF_BY_ANGLE', ''),
    _default_gap_conf_by_angle
)
SHOTLAB_EVENT_ENABLE_GAP_FILL_BY_ANGLE = _parse_event_angle_bool_map(
    os.environ.get('SHOTLAB_EVENT_ENABLE_GAP_FILL_BY_ANGLE', ''),
    _DEFAULT_EVENT_GAP_ENABLE_BY_ANGLE
)

SHOTLAB_DEFAULT_BENCHMARK = float(os.environ.get('SHOTLAB_DEFAULT_BENCHMARK', '125'))
SHOTLAB_DEFAULT_STD = float(os.environ.get('SHOTLAB_DEFAULT_STD', '15'))
SHOTLAB_BENCHMARKS_FILE = os.environ.get(
    'SHOTLAB_BENCHMARKS_FILE',
    str(Path(__file__).parent / 'shotlab_benchmarks.json')
)
SHOTLAB_V5_SHOT_VALIDITY_ENABLE = os.environ.get('SHOTLAB_V5_SHOT_VALIDITY_ENABLE', '1') != '0'
SHOTLAB_V5_SHOT_VALIDITY_THRESHOLD = float(os.environ.get('SHOTLAB_V5_SHOT_VALIDITY_THRESHOLD', '0.23'))
SHOTLAB_V5_SHOT_VALIDITY_MAX_DROP_SHARE = float(os.environ.get('SHOTLAB_V5_SHOT_VALIDITY_MAX_DROP_SHARE', '0.20'))
SHOTLAB_V5_SHOT_VALIDITY_MODEL_PATH = os.environ.get(
    'SHOTLAB_V5_SHOT_VALIDITY_MODEL_PATH',
    str(Path(__file__).parent.parent / 'models' / 'shotlab_v5' / 'shot_validity_gbdt.joblib')
)
SHOTLAB_ZONE_BENCHMARKS = {
    'restricted_area': float(os.environ.get('SHOTLAB_BENCHMARK_RESTRICTED', '125')),
    'paint': float(os.environ.get('SHOTLAB_BENCHMARK_PAINT', '125')),
    'mid_range': float(os.environ.get('SHOTLAB_BENCHMARK_MID', '125')),
    'left_corner_3': float(os.environ.get('SHOTLAB_BENCHMARK_LC3', '125')),
    'right_corner_3': float(os.environ.get('SHOTLAB_BENCHMARK_RC3', '125')),
    'left_wing_3': float(os.environ.get('SHOTLAB_BENCHMARK_LW3', '125')),
    'right_wing_3': float(os.environ.get('SHOTLAB_BENCHMARK_RW3', '125')),
    'top_of_key_3': float(os.environ.get('SHOTLAB_BENCHMARK_TOK3', '125')),
    'left_baseline_2': float(os.environ.get('SHOTLAB_BENCHMARK_LB2', '125')),
    'right_baseline_2': float(os.environ.get('SHOTLAB_BENCHMARK_RB2', '125')),
    'left_wing_2': float(os.environ.get('SHOTLAB_BENCHMARK_LW2', '125')),
    'right_wing_2': float(os.environ.get('SHOTLAB_BENCHMARK_RW2', '125')),
    'unknown': float(os.environ.get('SHOTLAB_BENCHMARK_UNKNOWN', '125'))
}

shotlab_models_initialized = False
shotlab_benchmark_cache = None
shotlab_status = {
    'stage': 'idle',
    'detail': '',
    'progress': 0.0,
    'updated_at': 0.0
}
shotlab_build_id = os.environ.get('SHOTLAB_BUILD_ID', f"shotlab_app_{int(time.time())}")
shotlab_sessions = {}

# ====================== DETECTION + STORAGE HELPERS ======================

class ShotLabDetectionModel:
    def __init__(self):
        self.model_type = None
        self.model = None
        self.model_label = None
        self.device = None
        self.model_path = None
        self.event_model = None
        self.event_model_label = None
        self.event_model_path = None
        self.event_model_classes = {}
        self._init_model()
        self._init_event_model()

    def _resolve_device(self):
        preferred = os.environ.get('SHOTLAB_DEVICE')
        if preferred:
            return preferred
        if not TORCH_AVAILABLE:
            return None
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _init_model(self):
        preferred = Config.BALL_MODEL_TYPE
        self.device = self._resolve_device()
        if preferred == 'custom' and self._init_custom():
            return
        if preferred in ('roboflow', 'auto'):
            if self._init_roboflow():
                return
        self._init_yolo()

    def _init_custom(self):
        if not Config.BASKETBALL_MODEL:
            return False
        model_path = Path(Config.BASKETBALL_MODEL)
        if not model_path.is_absolute():
            model_path = Path(__file__).parent.parent / model_path
        if not model_path.exists():
            return False
        try:
            self.model = YOLO(str(model_path))
            self.model_type = 'custom'
            self.model_label = f"custom:{model_path.name}"
            self.model_path = str(model_path)
            if self.device and TORCH_AVAILABLE:
                self.model.to(self.device)
            if self.device:
                print(f"ShotLab detector device: {self.device}")
            print(f"ShotLab detector: {self.model_label}")
            if hasattr(self.model, 'names'):
                print(f"ShotLab model classes: {self.model.names}")
            return True
        except Exception as exc:
            print(f"Warning: custom model init failed: {exc}")
            return False

    def _init_roboflow(self):
        if not ROBOFLOW_AVAILABLE or not Config.ROBOFLOW_API_KEY:
            return False
        try:
            rf = Roboflow(api_key=Config.ROBOFLOW_API_KEY)
            project = rf.workspace(Config.ROBOFLOW_WORKSPACE).project(Config.ROBOFLOW_PROJECT)
            self.model = project.version(Config.ROBOFLOW_VERSION).model
            self.model_type = 'roboflow'
            self.model_label = f"roboflow:{Config.ROBOFLOW_PROJECT}:{Config.ROBOFLOW_VERSION}"
            self.model_path = None
            if self.device:
                print(f"ShotLab detector device: {self.device}")
            print(f"ShotLab detector: {self.model_label}")
            return True
        except Exception as exc:
            print(f"Warning: Roboflow init failed, falling back to YOLO: {exc}")
            return False

    def _init_yolo(self):
        try:
            self.model = YOLO('yolo11m.pt')
            self.model_type = 'yolov11'
            self.model_label = 'yolo11m'
            self.model_path = 'yolo11m.pt'
            if self.device and TORCH_AVAILABLE:
                self.model.to(self.device)
            if self.device:
                print(f"ShotLab detector device: {self.device}")
            print("ShotLab detector: yolo11m")
            return True
        except Exception as exc:
            print(f"Warning: YOLOv11 init failed: {exc}")
        try:
            fallback_path = Path(__file__).parent.parent / 'yolov8s.pt'
            self.model = YOLO(str(fallback_path) if fallback_path.exists() else 'yolov8s.pt')
            self.model_type = 'yolov8'
            self.model_label = 'yolov8s'
            self.model_path = str(fallback_path) if fallback_path.exists() else 'yolov8s.pt'
            if self.device and TORCH_AVAILABLE:
                self.model.to(self.device)
            if self.device:
                print(f"ShotLab detector device: {self.device}")
            print("ShotLab detector: yolov8s")
            return True
        except Exception as exc:
            print(f"Warning: YOLOv8 init failed: {exc}")
        self.model = None
        self.model_type = None
        self.model_label = None
        return False

    def _resolve_event_model_path(self):
        raw = Path(SHOTLAB_EVENT_MODEL_PATH)
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(Path(__file__).parent / raw)
            candidates.append(Path(__file__).parent.parent / raw)
        for path in candidates:
            if path.exists():
                return path
        return candidates[0] if candidates else raw

    def _init_event_model(self):
        if not SHOTLAB_EVENT_MODEL_ENABLE:
            return False
        event_model_path = self._resolve_event_model_path()
        if not event_model_path.exists():
            print(f"Warning: ShotLab shoot event model not found at {event_model_path}")
            return False
        try:
            self.event_model = YOLO(str(event_model_path))
            self.event_model_label = f"event:{event_model_path.name}"
            self.event_model_path = str(event_model_path)
            if self.device and TORCH_AVAILABLE:
                self.event_model.to(self.device)
            if hasattr(self.event_model, 'names'):
                self.event_model_classes = dict(getattr(self.event_model, 'names') or {})
            print(f"ShotLab shoot event detector: {self.event_model_label}")
            if self.device:
                print(f"ShotLab shoot event detector device: {self.device}")
            return True
        except Exception as exc:
            print(f"Warning: ShotLab shoot event model init failed: {exc}")
            self.event_model = None
            self.event_model_label = None
            self.event_model_path = None
            self.event_model_classes = {}
            return False

    def detect_ball(self, frame):
        if self.model is None:
            return sv.Detections.empty()
        if self.model_type == 'roboflow':
            return self._detect_ball_roboflow(frame)
        return self._detect_ball_yolo(frame)

    def detect_rim(self, frame):
        if self.model is None:
            return None
        if self.model_type == 'roboflow':
            return self._detect_rim_roboflow(frame)
        return self._detect_rim_yolo(frame)

    def _detect_ball_roboflow(self, frame):
        try:
            result = self.model.predict(
                frame,
                confidence=Config.ROBOFLOW_CONFIDENCE,
                overlap=Config.ROBOFLOW_OVERLAP
            )
            predictions = result.predictions if result else []
        except Exception as exc:
            print(f"ShotLab Roboflow ball detection error: {exc}")
            return sv.Detections.empty()

        boxes = []
        confidences = []
        for pred in predictions:
            class_name = str(getattr(pred, 'class_name', '')).lower()
            if class_name not in Config.ROBOFLOW_BALL_CLASSES:
                continue
            x = float(pred.x)
            y = float(pred.y)
            w = float(pred.width)
            h = float(pred.height)
            boxes.append([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0])
            confidences.append(float(getattr(pred, 'confidence', 0.0)))

        detections = _detections_from_boxes(boxes, confidences)
        return _filter_ball_detections(detections)

    def _detect_ball_yolo(self, frame):
        class_ids = Config.BALL_CLASS_IDS if Config.BALL_CLASS_IDS else None
        if self.model_type == 'custom' and os.getenv('SHOTLAB_FORCE_CLASS_IDS', '0') != '1':
            class_ids = None
        try:
            results = self.model.predict(
                frame,
                imgsz=Config.BALL_IMG_SIZE,
                conf=Config.BALL_CONFIDENCE,
                iou=Config.BALL_IOU,
                classes=class_ids,
                device=self.device,
                verbose=False
            )
        except Exception as exc:
            print(f"ShotLab model prediction failed: {exc}")
            return sv.Detections.empty()
        detections = sv.Detections.from_ultralytics(results[0]) if results else sv.Detections.empty()
        if SHOTLAB_DEBUG_BALL:
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                self._no_detection_count = getattr(self, '_no_detection_count', 0) + 1
                if self._no_detection_count % 50 == 0:
                    print(f"⚠️  Still no detections after {self._no_detection_count} frames")
        return _filter_ball_detections(detections)

    def _detect_rim_roboflow(self, frame):
        try:
            result = self.model.predict(
                frame,
                confidence=Config.ROBOFLOW_CONFIDENCE,
                overlap=Config.ROBOFLOW_OVERLAP
            )
            predictions = result.predictions if result else []
        except Exception as exc:
            print(f"ShotLab Roboflow rim detection error: {exc}")
            return None

        rim_candidates = []
        for pred in predictions:
            class_name = str(getattr(pred, 'class_name', '')).lower()
            if class_name not in Config.ROBOFLOW_RIM_CLASSES:
                continue
            rim_candidates.append(pred)

        if not rim_candidates:
            return None
        best = max(rim_candidates, key=lambda p: float(getattr(p, 'confidence', 0.0)))
        return {
            'x': float(best.x),
            'y': float(best.y),
            'width': float(best.width),
            'height': float(best.height),
            'confidence': float(getattr(best, 'confidence', 0.0)),
            'source': 'roboflow'
        }

    def _detect_rim_yolo(self, frame):
        if not Config.RIM_CLASS_IDS:
            return None
        results = self.model.predict(
            frame,
            imgsz=Config.BALL_IMG_SIZE,
            conf=SHOTLAB_RIM_CONFIDENCE,
            iou=Config.BALL_IOU,
            classes=Config.RIM_CLASS_IDS,
            device=self.device,
            verbose=False
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        idx = int(np.argmax(confidences))
        x1, y1, x2, y2 = boxes[idx]
        return {
            'x': float((x1 + x2) / 2.0),
            'y': float((y1 + y2) / 2.0),
            'width': float(x2 - x1),
            'height': float(y2 - y1),
            'confidence': float(confidences[idx]),
            'source': self.model_type
        }


def _detections_from_boxes(boxes, confidences):
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int)
    )


def _filter_ball_detections(detections):
    if detections is None or len(detections) == 0:
        return sv.Detections.empty()
    valid_indices = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        if width < Config.BALL_MIN_SIZE or height < Config.BALL_MIN_SIZE:
            continue
        if width > Config.BALL_MAX_SIZE or height > Config.BALL_MAX_SIZE:
            continue
        aspect_ratio = width / height if height > 0 else 0.0
        if aspect_ratio < Config.BALL_ASPECT_MIN or aspect_ratio > Config.BALL_ASPECT_MAX:
            continue
        valid_indices.append(i)
    if not valid_indices:
        if SHOTLAB_DEBUG_BALL:
            print(f"⚠️  All {len(detections)} detections filtered out by size/aspect constraints.")
        return sv.Detections.empty()
    return detections[valid_indices]


def get_detection_model():
    global shotlab_detector
    if shotlab_detector is None:
        shotlab_detector = ShotLabDetectionModel()
    return shotlab_detector


def get_rtmpose_estimator():
    global rtmpose_estimator_instance, rtmpose_estimator_error
    if rtmpose_estimator_instance is not None:
        return rtmpose_estimator_instance
    if rtmpose_estimator_error:
        return None
    try:
        from rtmpose_estimator import RTMPoseEstimator  # noqa: WPS433
        estimator = RTMPoseEstimator(
            checkpoint_path=SHOTLAB_RTMPOSE_MODEL_PATH,
            auto_download=SHOTLAB_RTMPOSE_AUTO_DOWNLOAD,
            logger=logger,
        )
        if not bool(getattr(estimator, 'available', False)):
            rtmpose_estimator_error = str(getattr(estimator, 'last_error', 'RTMPose unavailable'))
            logger.warning("RTMPose unavailable: %s", rtmpose_estimator_error)
            return None
        rtmpose_estimator_instance = estimator
        logger.info("RTMPose form-scoring backend initialized (%s)", SHOTLAB_RTMPOSE_MODEL_PATH)
        return rtmpose_estimator_instance
    except Exception as exc:
        rtmpose_estimator_error = str(exc)
        logger.warning("RTMPose init failed; will fallback to MediaPipe: %s", exc)
        return None


def get_shot_validity_model():
    global shot_validity_model
    if shot_validity_model is None and SHOTLAB_V5_SHOT_VALIDITY_ENABLE:
        shot_validity_model = ShotValidityModel(model_path=SHOTLAB_V5_SHOT_VALIDITY_MODEL_PATH)
    return shot_validity_model


def get_training_storage():
    global training_storage
    if not Config.COLLECT_TRAINING_DATA:
        return None
    if training_storage is None:
        try:
            training_storage = ShotLabS3Storage(
                bucket_name=Config.S3_TRAINING_BUCKET,
                aws_access_key=Config.AWS_ACCESS_KEY_ID,
                aws_secret_key=Config.AWS_SECRET_ACCESS_KEY,
                region=Config.AWS_REGION
            )
        except Exception as exc:
            print(f"Warning: Could not initialize training storage: {exc}")
            training_storage = None
    return training_storage


def get_face_cascade():
    global face_cascade
    if face_cascade is not None:
        return face_cascade
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            face_cascade = None
    except Exception:
        face_cascade = None
    return face_cascade


def blur_faces(frame):
    if not Config.BLUR_FACES:
        return frame
    cascade = get_face_cascade()
    if cascade is None:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if faces is None or len(faces) == 0:
        return frame
    blurred = frame.copy()
    strength = max(3, int(Config.BLUR_FACES_STRENGTH))
    if strength % 2 == 0:
        strength += 1
    for (x, y, w, h) in faces:
        roi = blurred[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        blurred[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (strength, strength), 0)
    return blurred


class TrainingDataCollector:
    def __init__(self, session_id, video_metadata=None):
        self.session_id = session_id
        self.video_metadata = video_metadata or {}
        self.frame_save_counter = 0
        self.storage = get_training_storage()
        self.is_duplicate = False
        self.video_hash = None

    def should_save_frame(self, frame_idx, detections, metadata):
        if not self.storage or not Config.COLLECT_TRAINING_DATA:
            return False
        if self.is_duplicate:
            return False
        self.frame_save_counter += 1
        if Config.FRAME_SAVE_FREQUENCY > 0 and self.frame_save_counter % Config.FRAME_SAVE_FREQUENCY == 0:
            return True
        if metadata and metadata.get('shot_detected_by_pose') and (detections is None or len(detections) == 0):
            return True
        if detections is not None and len(detections) > 0:
            confs = detections.confidence if hasattr(detections, 'confidence') else None
            max_conf = float(np.max(confs)) if confs is not None and len(confs) > 0 else 0.0
            if Config.LOW_CONF_MIN < max_conf < Config.LOW_CONF_MAX:
                return True
        if metadata and metadata.get('is_release_point'):
            return True
        return False

    def save_frame_async(self, frame, detections, metadata, frame_idx):
        if not self.storage:
            return
        if self.is_duplicate:
            return
        payload = metadata.copy() if metadata else {}
        payload.update(self.video_metadata)
        if self.video_hash:
            payload['video_hash'] = self.video_hash
        frame_to_save = blur_faces(frame) if Config.BLUR_FACES else frame
        try:
            training_executor.submit(
                self.storage.save_training_frame,
                frame_to_save,
                detections,
                payload,
                self.session_id,
                frame_idx
            )
        except Exception as exc:
            print(f"Warning: Training data submit failed: {exc}")

# ====================== POSE & ANGLE UTILS ======================

def _shotlab_session_dir():
    return os.path.join(app.config['UPLOAD_FOLDER'], 'shotlab_sessions')

def _shotlab_session_path(session_id):
    return os.path.join(_shotlab_session_dir(), f'{session_id}.json')

def _save_shotlab_session(session_id, data):
    try:
        os.makedirs(_shotlab_session_dir(), exist_ok=True)
        payload = {
            'video_path': data.get('video_path'),
            'video_name': data.get('video_name'),
            'fps': data.get('fps'),
            'shots': data.get('shots'),
            'shots_analysis': data.get('shots_analysis'),
            'total_frames': data.get('total_frames'),
            'rim_position': data.get('rim_position'),
            'net_zone': data.get('net_zone'),
            'ball_tracks': data.get('ball_tracks'),
            'debug': data.get('debug')
        }
        with open(_shotlab_session_path(session_id), 'w') as handle:
            json.dump(payload, handle)
    except Exception as e:
        print(f"ShotLab session persist failed: {e}")

def _load_shotlab_session(session_id):
    try:
        path = _shotlab_session_path(session_id)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as handle:
            data = json.load(handle)
        data['ball_tracks_by_frame'] = None
        data['created_at'] = time.time()
        return data
    except Exception as e:
        print(f"ShotLab session load failed: {e}")
        return None

def build_video_signature(video_path, original_name=None):
    """Robust signature used for court-calibration reuse trust checks."""
    return compute_video_signature(video_path, original_name=original_name)


def _form_flag_true(form, key):
    value = None
    try:
        value = form.get(key)
    except Exception:
        value = None
    return str(value or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _normalize_camera_position_override(value):
    normalized = str(value or '').strip().lower()
    if not normalized:
        return None
    allowed = {'behind_shooter', 'sideline_left', 'sideline_right', 'in_front_of_shooter', 'auto'}
    return normalized if normalized in allowed else None


def _normalize_camera_mode(value):
    normalized = str(value or '').strip().lower().replace('-', '_').replace(' ', '_')
    if not normalized:
        return 'auto'
    if normalized in {'behind_basket', 'auto'}:
        return normalized
    return 'auto'


def _is_behind_basket_mode(camera_mode):
    return _normalize_camera_mode(camera_mode) == 'behind_basket'


def _apply_camera_position_override(camera_info, camera_position_override):
    info = dict(camera_info or {})
    override = _normalize_camera_position_override(camera_position_override)
    info['camera_position_override'] = override
    if override and override != 'auto':
        info['estimated_position'] = str(info.get('position', 'unknown') or 'unknown')
        info['estimated_confidence'] = float(info.get('confidence', 0.0) or 0.0)
        info['position'] = override
        info['confidence'] = max(float(info.get('confidence', 0.0) or 0.0), 0.99)
        info['override_applied'] = True
    else:
        info['override_applied'] = False
    return info


def _augment_court_calibration_payload(court_calibration, reuse_state, court_signature, camera_position_override=None):
    payload = dict(court_calibration or {})
    state = dict(reuse_state or {})
    payload['court_reuse_available'] = bool(state.get('court_reuse_available', False))
    payload['court_reuse_reason'] = str(state.get('court_reuse_reason', 'unknown'))
    payload['court_confirm_required'] = bool(state.get('court_confirm_required', False))
    payload['court_signature'] = str(court_signature or '')
    payload['camera_position_override'] = _normalize_camera_position_override(camera_position_override)
    return payload

def get_visibility_threshold(camera_position, landmark_name):
    """Get visibility threshold based on camera angle and landmark."""
    default_threshold = max(0.35, SHOTLAB_POSE_MIN_VISIBILITY)
    base_thresholds = {
        'shoulder': default_threshold,
        'elbow': default_threshold,
        'wrist': default_threshold,
        'hip': default_threshold
    }

    if camera_position and str(camera_position).startswith('sideline'):
        sideline_thresholds = {
            'shoulder': default_threshold,
            'elbow': 0.25,
            'wrist': 0.20,
            'hip': 0.40
        }
        return sideline_thresholds.get(landmark_name, 0.30)

    if camera_position == 'behind_shooter':
        behind_thresholds = {
            'shoulder': default_threshold,
            'elbow': 0.30,
            'wrist': 0.25,
            'hip': default_threshold
        }
        return behind_thresholds.get(landmark_name, 0.30)

    return base_thresholds.get(landmark_name, default_threshold)

def get_3d_point(landmarks, index, width, height, min_visibility=None):
    """Extract (x, y, z) from Mediapipe landmarks. Returns None if not visible enough."""
    if min_visibility is None:
        min_visibility = SHOTLAB_POSE_MIN_VISIBILITY
    if index >= len(landmarks) or landmarks[index].visibility < min_visibility:
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

def get_arm_state(landmarks, width, height, camera_position=None):
    """
    'pre_shot' => wrists close together below shoulders (or elbow fallback)
    'follow_through' => wrist/elbow above shoulder
    'neutral' => default
    """
    def get_point(index, min_vis):
        if index >= len(landmarks) or landmarks[index].visibility < min_vis:
            return None
        return np.array([
            landmarks[index].x * width,
            landmarks[index].y * height,
            landmarks[index].z
        ])

    shoulder_threshold = get_visibility_threshold(camera_position, 'shoulder')
    elbow_threshold = get_visibility_threshold(camera_position, 'elbow')
    wrist_threshold = get_visibility_threshold(camera_position, 'wrist')
    hip_threshold = get_visibility_threshold(camera_position, 'hip')

    right_shoulder = get_point(12, shoulder_threshold)
    right_elbow = get_point(14, elbow_threshold)
    right_wrist = get_point(16, wrist_threshold)
    left_shoulder = get_point(11, shoulder_threshold)
    left_elbow = get_point(13, elbow_threshold)
    left_wrist = get_point(15, wrist_threshold)
    left_hip = get_point(23, hip_threshold)
    right_hip = get_point(24, hip_threshold)

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

    def arm_state_for_side(shoulder, elbow, wrist):
        if shoulder is None:
            return None
        if wrist is not None:
            if shoulder[1] - wrist[1] > 0.05 * height:
                return "follow_through"
            if abs(wrist[1] - shoulder[1]) < 0.08 * height:
                return "pre_shot"
            if wrist[1] > shoulder[1] + 0.05 * height:
                return "neutral"
            return "neutral"
        if elbow is not None:
            if shoulder[1] - elbow[1] > 0.05 * height:
                return "follow_through"
            if elbow[1] < shoulder[1] + 0.05 * height:
                return "pre_shot"
            if elbow[1] > shoulder[1] + 0.08 * height:
                return "neutral"
            return "neutral"
        return None

    right_state = arm_state_for_side(right_shoulder, right_elbow, right_wrist)
    left_state = arm_state_for_side(left_shoulder, left_elbow, left_wrist)

    if right_state == "follow_through" or left_state == "follow_through":
        return "follow_through"
    if right_state == "pre_shot" or left_state == "pre_shot":
        return "pre_shot"
    if right_state == "neutral" or left_state == "neutral":
        return "neutral"

    return "neutral"

# ====================== PROCESS VIDEO FRAMES ======================

def process_video_frames(frames_data, camera_position=None):
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
            state = get_arm_state(landmarks, w, h, camera_position=camera_position)
            
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

def detect_basketball_local(frame):
    """
    Detect basketball in frame using local YOLOv8
    Returns: supervision Detections object
    """
    detector = get_detection_model()
    return detector.detect_ball(frame)

def debug_ball_detection_frame(frame, frame_idx, detector):
    print(f"\n=== BALL DETECTION DEBUG (Frame {frame_idx}) ===")
    print(f"Model path: {detector.model_path or 'N/A'}")
    print(f"Model loaded: {detector.model is not None}")
    print(f"Model type: {detector.model_type}")
    print(f"Model device: {detector.device}")
    if detector.model is None or detector.model_type == 'roboflow':
        print("Skipping raw YOLO debug (model not available or Roboflow).")
        return

    try:
        results = detector.model.predict(
            frame,
            imgsz=Config.BALL_IMG_SIZE,
            conf=0.01,
            iou=Config.BALL_IOU,
            classes=None,
            device=detector.device,
            verbose=True
        )
        raw_boxes = results[0].boxes if results else None
        raw_count = len(raw_boxes) if raw_boxes is not None else 0
        print(f"Raw detections: {raw_count}")
        if hasattr(detector.model, 'names'):
            print(f"Model classes: {detector.model.names}")

        if raw_boxes is not None and raw_count > 0:
            for i, box in enumerate(raw_boxes[:5]):
                cls_id = int(box.cls)
                cls_name = detector.model.names.get(cls_id, 'unknown') if hasattr(detector.model, 'names') else 'unknown'
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()
                print(f"  Detection {i}: class={cls_id} ({cls_name}), conf={conf:.3f}, bbox={bbox}")
        else:
            print("  ❌ NO DETECTIONS FOUND")
            print(f"  Frame shape: {frame.shape}")
            print(f"  Frame dtype: {frame.dtype}")
            print(f"  Frame range: [{frame.min()}, {frame.max()}]")
            debug_path = '/tmp/shotlab_debug_frame.jpg'
            cv2.imwrite(debug_path, frame)
            print(f"  Saved frame to {debug_path} for inspection")
    except Exception as exc:
        print(f"  ❌ DETECTION FAILED: {exc}")

def initialize_shotlab_models():
    """Initialize local models and tracking utilities for ShotLab."""
    global shotlab_models_initialized
    if shotlab_models_initialized:
        return True, None
    detector = get_detection_model()
    if detector.model is None:
        return False, 'No ball detection model available'
    get_shot_validity_model()
    shotlab_models_initialized = True
    return True, None

def detect_court_keypoints(frame):
    """Court detection - manual calibration only."""
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

def manual_court_transform_from_points(points, frame_shape):
    """Compute a court transform from normalized point list."""
    if frame_shape is None or not points or len(points) < 4:
        return None
    try:
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

def denormalize_landmarks(landmarks_norm, frame_shape):
    if not landmarks_norm or frame_shape is None:
        return None
    try:
        h, w = frame_shape[:2]
        landmarks_px = {}
        for name, payload in landmarks_norm.items():
            if not isinstance(payload, dict):
                continue
            x_norm = float(payload.get('x_norm'))
            y_norm = float(payload.get('y_norm'))
            landmarks_px[name] = (x_norm * w, y_norm * h)
        return landmarks_px if landmarks_px else None
    except Exception:
        return None

def manual_court_transform_from_landmarks(landmarks_norm, frame_shape):
    landmarks_px = denormalize_landmarks(landmarks_norm, frame_shape)
    if not landmarks_px:
        return None, None
    try:
        calibrator = CourtCalibrator()
        for name, pos in landmarks_px.items():
            calibrator.add_landmark(name, pos)
        if not calibrator.is_ready():
            return None, None
        result = calibrator.calibrate()
        meta = {
            'type': 'landmarks',
            'scale': result.get('scale'),
            'orientation': result.get('orientation'),
            'landmarks': landmarks_px
        }
        return result.get('homography'), meta
    except Exception:
        return None, None

def player_to_court_position(player_xy, transform_matrix, transform_meta=None):
    if player_xy is None or transform_matrix is None:
        return None
    point = np.array([[[player_xy[0], player_xy[1]]]], dtype=np.float32)
    court_pos = cv2.perspectiveTransform(point, transform_matrix)[0][0]
    if transform_meta and transform_meta.get('type') == 'landmarks':
        # Convert from basket-origin feet to legacy 50x47 court coords.
        court_pos = np.array([25.0 + court_pos[0], 47.0 - court_pos[1]], dtype=np.float32)
    return court_pos

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


def classify_shot_zone_behind_basket(anchor_xy, frame_width, frame_height=None, rim_position=None):
    """
    Lateral-only zone mapping for behind-basket camera mode.
    Returns:
      {
        'zone': one of left|center|right,
        'shooter_x': normalized shooter x in [0, 1],
        'shooter_y': normalized shooter y in [0, 1]
      }
    """
    frame_w = float(frame_width or 0.0)
    frame_h = float(frame_height or 0.0)
    if frame_w <= 1.0:
        frame_w = 1920.0
    if frame_h <= 1.0:
        frame_h = 1080.0

    # Fallback rim geometry: top-center-ish if no rim is available.
    rim_x = frame_w * 0.5
    rim_y = frame_h * 0.30
    if isinstance(rim_position, dict):
        try:
            if rim_position.get('x') is not None:
                rim_x = float(rim_position.get('x'))
            if rim_position.get('y') is not None:
                rim_y = float(rim_position.get('y'))
        except (TypeError, ValueError):
            pass
    elif isinstance(rim_position, (list, tuple)) and len(rim_position) >= 4:
        try:
            rx1, ry1, rx2, ry2 = [float(v) for v in rim_position[:4]]
            rim_x = (rx1 + rx2) * 0.5
            rim_y = (ry1 + ry2) * 0.5
        except (TypeError, ValueError):
            pass

    # Fallback anchor if missing: assume center lane.
    if anchor_xy is None or len(anchor_xy) < 2:
        shooter_x = frame_w * 0.5
        shooter_y = frame_h * 0.70
    else:
        shooter_x = float(anchor_xy[0])
        shooter_y = float(anchor_xy[1])

    center_half_width = BB_CENTER_BAND_RATIO * frame_w
    if shooter_x < (rim_x - center_half_width):
        zone = 'left'
    elif shooter_x > (rim_x + center_half_width):
        zone = 'right'
    else:
        zone = 'center'
    shooter_x_norm = max(0.0, min(1.0, float(shooter_x) / frame_w))
    shooter_y_norm = max(0.0, min(1.0, float(shooter_y) / frame_h))
    return {
        'zone': zone,
        'shooter_x': float(shooter_x_norm),
        'shooter_y': float(shooter_y_norm),
        'rim_x': float(rim_x),
        'rim_y': float(rim_y),
    }


def classify_shot_type_behind_basket(anchor_xy, rim_zone):
    """Flag layups near rim as a reporting attribute (not used for detection)."""
    if anchor_xy is None or len(anchor_xy) < 2:
        return 'jump_shot'
    zone = _normalize_zone_box(rim_zone)
    if zone is None:
        return 'jump_shot'
    rim_cx = float(zone['center_x'])
    rim_cy = float(zone['center_y'])
    rim_w = max(1.0, float(zone['width']))
    rim_h = max(1.0, float(zone['height']))
    shooter_x = float(anchor_xy[0])
    shooter_y = float(anchor_xy[1])
    dist_to_rim = math.hypot(shooter_x - rim_cx, shooter_y - rim_cy)
    near_vertical_band = (rim_cy - 1.0 * rim_h) <= shooter_y <= (rim_cy + 2.0 * rim_h)
    if dist_to_rim <= (BB_LAYUP_RIM_DISTANCE_MULT * rim_w) and near_vertical_band:
        return 'layup'
    return 'jump_shot'

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


def compute_calibrated_form_score(raw_form_score, benchmark_mean, benchmark_std, score_scale=15.0):
    """Map raw angle-based form values into a stable 0-100 band."""
    if raw_form_score is None:
        return None
    std = float(benchmark_std) if benchmark_std and float(benchmark_std) > 1e-6 else 1.0
    centered = 50.0 + ((float(raw_form_score) - float(benchmark_mean)) / std) * float(score_scale)
    return float(max(0.0, min(100.0, centered)))

def _summarize_form_visibility(window_pose_entries):
    keys = (
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
    )
    summary = {}
    for key in keys:
        values = []
        for entry in window_pose_entries:
            vis_map = entry.get('landmark_visibility') if isinstance(entry, dict) else None
            if not isinstance(vis_map, dict):
                continue
            value = vis_map.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if values:
            arr = np.array(values, dtype=np.float32)
            summary[key] = {
                'min': round(float(np.min(arr)), 3),
                'median': round(float(np.median(arr)), 3),
                'max': round(float(np.max(arr)), 3),
                'count': int(len(values)),
            }
        else:
            summary[key] = None
    return summary


def get_form_score_for_shot(
    pose_frames,
    release_frame,
    start_frame,
    end_frame,
    search_gap_frames,
    return_debug=False,
    form_visibility_thresholds=None,
):
    """Get a stable form score with staged windows around release."""
    release_frame_i = int(release_frame)
    thresholds = dict(form_visibility_thresholds or {})
    window_pose_entries = [
        e for e in (pose_frames or [])
        if int(start_frame) <= int(e.get('frame_idx', -1)) <= int(end_frame)
    ]
    landmark_visibility_by_frame = []
    for entry in window_pose_entries:
        vis_map = entry.get('landmark_visibility') if isinstance(entry, dict) else None
        if not isinstance(vis_map, dict):
            continue
        landmark_visibility_by_frame.append({
            'frame_idx': int(entry.get('frame_idx', -1)),
            'right_shoulder': vis_map.get('right_shoulder'),
            'right_elbow': vis_map.get('right_elbow'),
            'right_wrist': vis_map.get('right_wrist'),
            'left_shoulder': vis_map.get('left_shoulder'),
            'left_elbow': vis_map.get('left_elbow'),
            'left_wrist': vis_map.get('left_wrist'),
        })
    visibility_summary = _summarize_form_visibility(window_pose_entries)
    frames_with_angles = sum(1 for e in window_pose_entries if e.get('angles') is not None)
    frames_with_scores = sum(
        1
        for e in window_pose_entries
        if e.get('angles') is not None and e['angles'].get('form_score') is not None
    )

    def _scored_rows_in_window(window_start, window_end):
        rows = []
        for entry in (pose_frames or []):
            if not entry.get('angles'):
                continue
            score = entry['angles'].get('raw_form_score_before_scaling')
            if score is None:
                score = entry['angles'].get('form_score')
            if score is None:
                continue
            frame_idx = int(entry.get('frame_idx', 0))
            if frame_idx < int(window_start) or frame_idx > int(window_end):
                continue
            rows.append((frame_idx, float(score), dict(entry.get('angles') or {})))
        return rows

    def _scores_in_window(window_start, window_end):
        return [score for _, score, _ in _scored_rows_in_window(window_start, window_end)]

    def _nearest_score(scored_rows):
        if not scored_rows:
            return None
        ranked = sorted(
            scored_rows,
            key=lambda fs: (abs(int(fs[0]) - release_frame_i), int(fs[0]))
        )
        nearest = ranked[:5]
        values = np.array([float(v) for _, v, _ in nearest], dtype=np.float32)
        return float(np.median(values))

    def _select_representative_row(scored_rows):
        if not scored_rows:
            return None
        ranked = sorted(
            scored_rows,
            key=lambda fs: (abs(int(fs[0]) - release_frame_i), int(fs[0]))
        )
        return ranked[0]

    def _angle_snapshot(selected_row):
        if selected_row is None:
            return None
        frame_idx, raw_score, angles = selected_row
        keys = (
            'right_elbow_angle',
            'left_elbow_angle',
            'right_shoulder_angle',
            'left_shoulder_angle',
            'right_hip_angle',
            'left_hip_angle',
            'right_knee_angle',
            'left_knee_angle',
            'right_wrist_angle',
            'left_wrist_angle',
            'follow_through_extension',
            'shooting_arm',
        )
        payload = {'frame_idx': int(frame_idx), 'raw_form_score_before_scaling': float(raw_score)}
        for key in keys:
            value = angles.get(key)
            if isinstance(value, (int, float, np.floating)):
                payload[key] = float(value)
            else:
                payload[key] = value
        return payload

    def _finalize(score, strategy, reason='ok', selected_row=None):
        if return_debug:
            diagnostics = {
                'pose_frames_in_window': int(len(window_pose_entries)),
                'frames_with_angles_in_window': int(frames_with_angles),
                'frames_with_form_score_in_window': int(frames_with_scores),
                'landmark_visibility': visibility_summary,
                'landmark_visibility_by_frame': landmark_visibility_by_frame,
                'applied_thresholds': thresholds,
                'form_score_reason': str(reason),
                'raw_form_score_before_scaling': float(score) if score is not None else None,
                'selected_angles': _angle_snapshot(selected_row),
            }
            if selected_row is not None:
                diagnostics['selected_form_frame'] = int(selected_row[0])
            return score, strategy, diagnostics
        return score

    if SHOTLAB_DEBUG_FORM_SCORE:
        count = getattr(get_form_score_for_shot, '_debug_count', 0) + 1
        get_form_score_for_shot._debug_count = count
        if count <= 5:
            print(f"\n=== FORM SCORE RETRIEVAL (Shot {release_frame}) ===")
            print(f"Shot window: frames {start_frame}-{end_frame} ({end_frame - start_frame} frames)")
            print(f"Pose frames in window: {len(window_pose_entries)}")
            print(f"Frames with angles: {frames_with_angles} ({frames_with_angles/max(1,len(window_pose_entries))*100:.1f}%)")
            print(f"Frames with form_score: {frames_with_scores} ({frames_with_scores/max(1,len(window_pose_entries))*100:.1f}%)")
            if thresholds:
                print(f"Applied thresholds: {thresholds}")

    # Pass 1: release +/- 20 frames.
    pass1_rows = _scored_rows_in_window(int(release_frame) - 20, int(release_frame) + 20)
    pass1_scores = [score for _, score, _ in pass1_rows]
    if pass1_scores:
        score = float(np.median(np.array(pass1_scores, dtype=np.float32)))
        if SHOTLAB_DEBUG_FORM_SCORE and get_form_score_for_shot._debug_count <= 5:
            print(f"✅ Form score from release +/-20 window: {score:.1f}")
        return _finalize(score, 'release_pm20', reason='ok', selected_row=_select_representative_row(pass1_rows))

    # Pass 2: release -5 .. +30 (follow-through-friendly).
    pass2_rows = _scored_rows_in_window(int(release_frame) - 5, int(release_frame) + 30)
    pass2_scores = [score for _, score, _ in pass2_rows]
    if pass2_scores:
        score = float(np.median(np.array(pass2_scores, dtype=np.float32)))
        if SHOTLAB_DEBUG_FORM_SCORE and get_form_score_for_shot._debug_count <= 5:
            print(f"✅ Form score from release-5..+30 window: {score:.1f}")
        return _finalize(score, 'release_m5_p30', reason='ok', selected_row=_select_representative_row(pass2_rows))

    # Pass 3: shot-window nearest fallback (prefer temporally local scores).
    pass3_rows = _scored_rows_in_window(start_frame, end_frame)
    pass3_nearest = _nearest_score(pass3_rows)
    if pass3_nearest is not None:
        return _finalize(
            pass3_nearest,
            'shot_window_nearest',
            reason='ok',
            selected_row=_select_representative_row(pass3_rows),
        )

    if SHOTLAB_DEBUG_FORM_SCORE and get_form_score_for_shot._debug_count <= 5:
        print("❌ No form score found for shot")
    if not window_pose_entries:
        reason = 'no_pose_frames_in_window'
    elif frames_with_angles == 0:
        reason = 'low_landmark_visibility'
    else:
        reason = 'no_form_score_values'
    return _finalize(None, 'none_found', reason=reason)


def _extract_rim_contact_frame_from_vote_breakdown(vote_breakdown):
    if not isinstance(vote_breakdown, list):
        return None
    candidates = []
    for vote in vote_breakdown:
        details = (vote or {}).get('details') if isinstance(vote, dict) else None
        if not isinstance(details, dict):
            continue
        for key in ('rim_contact_frame', 'contact_frame'):
            value = details.get(key)
            if value is None:
                continue
            try:
                candidates.append(int(value))
            except (TypeError, ValueError):
                continue
    if not candidates:
        return None
    return int(min(candidates))

def filter_non_jump_shots(
    shot_attempts,
    pose_frames,
    ball_tracks,
    fps,
    rim_zone=None,
    net_zone=None,
    frame_width=None,
    frame_height=None,
    camera_position=None,
    debug=None
):
    """Filter shots using trajectory validation instead of jump/ball-upward checks."""
    if debug is None:
        debug = {}

    debug.setdefault('dropped_reasons', defaultdict(int))
    debug['input'] = len(shot_attempts)
    debug['trajectory_filter'] = {
        'input_count': len(shot_attempts),
        'valid_shots': 0,
        'invalid_shots': 0,
        'details': []
    }
    debug['dropped_shots'] = []

    if not shot_attempts:
        debug['kept'] = 0
        debug['dropped'] = 0
        debug['dropped_reasons'] = dict(debug['dropped_reasons'])
        return shot_attempts, debug

    ball_ordered, ball_frames = build_ball_track_index(ball_tracks) if ball_tracks else ([], [])
    filtered = []

    for shot in shot_attempts:
        start_frame = int(shot.get('start_frame', shot.get('release_frame', 0)))
        end_frame = int(shot.get('end_frame', shot.get('release_frame', start_frame)))
        release_frame = int(shot.get('release_frame', start_frame))

        shot_tracks = get_tracks_in_window(ball_ordered, ball_frames, start_frame, end_frame)
        shot_frames = list(range(start_frame, end_frame + 1))
        is_valid = validate_shot_trajectory(
            shot_frames=shot_frames,
            ball_tracks=shot_tracks,
            frame_width=frame_width,
            frame_height=frame_height,
            camera_position=camera_position
        )
        validation = getattr(validate_shot_trajectory, 'last_validation', None)
        if not validation:
            validation = {
                'is_valid_shot': bool(is_valid),
                'confidence': 0.0,
                'reason': 'validation_unavailable',
                'signals': {}
            }

        debug['trajectory_filter']['details'].append({
            'release_frame': release_frame,
            'is_valid': bool(validation.get('is_valid_shot', is_valid)),
            'confidence': validation['confidence'],
            'reason': validation['reason'],
            'signals': validation.get('signals', {})
        })
        signals = validation.get('signals', {})
        logger.info(
            (
                "Trajectory validation shot %s: valid=%s reason=%s "
                "xy_motion_ok=%s size_change_ok=%s disappearance_gap_ok=%s "
                "x_range_frac=%s y_range_frac=%s size_ratio=%s longest_mid_gap_frames=%s"
            ),
            release_frame,
            bool(validation.get('is_valid_shot', is_valid)),
            validation.get('reason'),
            signals.get('xy_motion_ok'),
            signals.get('size_change_ok'),
            signals.get('disappearance_gap_ok'),
            signals.get('x_range_frac'),
            signals.get('y_range_frac'),
            signals.get('size_ratio'),
            signals.get('longest_mid_gap_frames')
        )

        if bool(validation.get('is_valid_shot', is_valid)):
            debug['trajectory_filter']['valid_shots'] += 1
            filtered.append(shot)
        else:
            debug['trajectory_filter']['invalid_shots'] += 1
            debug['dropped_reasons']['trajectory_invalid'] += 1
            debug['dropped_shots'].append({
                'release_frame': release_frame,
                'reason': validation['reason']
            })
            logger.info(
                "Rejected shot %s after trajectory validation: %s",
                release_frame,
                validation['reason']
            )

    debug['kept'] = len(filtered)
    debug['dropped'] = max(0, len(shot_attempts) - len(filtered))
    debug['dropped_reasons'] = dict(debug['dropped_reasons'])
    return filtered, debug


def _normalize_zone_box(zone):
    if not zone:
        return None
    if 'x1' in zone and 'x2' in zone:
        x1 = float(zone.get('x1', 0))
        y1 = float(zone.get('y1', 0))
        x2 = float(zone.get('x2', 0))
        y2 = float(zone.get('y2', 0))
    elif 'x' in zone and 'width' in zone:
        x1 = float(zone.get('x', 0))
        y1 = float(zone.get('y', 0))
        x2 = x1 + float(zone.get('width', 0))
        y2 = y1 + float(zone.get('height', 0))
    else:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return {
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'width': max(1.0, x2 - x1),
        'height': max(1.0, y2 - y1),
        'center_x': (x1 + x2) / 2.0,
        'center_y': (y1 + y2) / 2.0
    }


def _track_xy(track):
    if 'x' in track and 'y' in track:
        return float(track['x']), float(track['y'])
    if 'center' in track and track['center'] is not None:
        return float(track['center'][0]), float(track['center'][1])
    return 0.0, 0.0


def _log_ball_trajectory(release_frame, tracks):
    print(f"=== BALL TRAJECTORY: Shot {release_frame} ===")
    print(f"  Total tracks: {len(tracks)}")
    if not tracks:
        return
    first = tracks[0]
    print(f"  Track keys: {list(first.keys())}")
    print(f"  First track raw: {first}")
    if len(tracks) <= 15:
        tracks_to_show = tracks
    else:
        mid = len(tracks) // 2
        tracks_to_show = tracks[:5] + tracks[mid - 2: mid + 3] + tracks[-5:]
    for t in tracks_to_show:
        x, y = _track_xy(t)
        size = (t.get('width', 0) or 0) * (t.get('height', 0) or 0)
        print(
            f"  frame={t.get('frame', '?'):4}, "
            f"x={x:6.1f}, "
            f"y={y:6.1f}, "
            f"size={size:6.0f}"
        )
    all_xy = [_track_xy(t) for t in tracks]
    all_x = [x for x, _ in all_xy]
    all_y = [y for _, y in all_xy]
    all_sizes = [
        (t.get('width', 0) or 0) * (t.get('height', 0) or 0)
        for t in tracks
        if (t.get('width', 0) or 0) > 0 and (t.get('height', 0) or 0) > 0
    ]
    print(f"  X range: {min(all_x):.0f} - {max(all_x):.0f}")
    print(f"  Y range: {min(all_y):.0f} - {max(all_y):.0f}")
    rim_excluded_points = sum(
        1 for t in tracks if int(t.get('rim_excluded_candidates', 0) or 0) > 0
    )
    if rim_excluded_points > 0:
        print(f"  Points with rim-like candidates excluded: {rim_excluded_points}/{len(tracks)}")
    if all_sizes:
        print(f"  Size range: {min(all_sizes):.0f} - {max(all_sizes):.0f}")
        min_size = min(all_sizes)
        size_tracks = [
            t for t in tracks
            if (t.get('width', 0) or 0) > 0 and (t.get('height', 0) or 0) > 0
        ]
        min_size_idx = all_sizes.index(min_size)
        min_size_track = size_tracks[min_size_idx]
        min_x, min_y = _track_xy(min_size_track)
        print(
            f"  Min size at: frame={min_size_track.get('frame', '?')}, "
            f"x={min_x:.0f}, "
            f"y={min_y:.0f}"
        )


def get_shot_trajectory_validation(shot_frames, ball_tracks, frame_width, frame_height, camera_position=None):
    """
    Validate whether trajectory resembles a jump shot.
    Conservative behavior: keep uncertain/borderline trajectories.
    """
    min_points = 5
    if not ball_tracks:
        return {
            'is_valid_shot': True,
            'confidence': 0.5,
            'reason': 'no_ball_tracks_keep',
            'signals': {'track_count': 0}
        }

    frame_set = set(int(f) for f in shot_frames) if shot_frames else None
    if frame_set:
        tracks = [t for t in ball_tracks if int(t.get('frame', -1)) in frame_set]
    else:
        tracks = list(ball_tracks)
    tracks = sorted(tracks, key=lambda t: int(t.get('frame', 0)))
    if len(tracks) < min_points:
        return {
            'is_valid_shot': True,
            'confidence': 0.5,
            'reason': 'insufficient_tracks_keep',
            'signals': {'track_count': len(tracks)}
        }

    width = float(frame_width or 0)
    height = float(frame_height or 0)
    if width <= 1 or height <= 1:
        return {
            'is_valid_shot': True,
            'confidence': 0.5,
            'reason': 'missing_frame_dimensions_keep',
            'signals': {'track_count': len(tracks)}
        }

    x_values = np.array([_track_xy(t)[0] for t in tracks], dtype=np.float32)
    y_values = np.array([_track_xy(t)[1] for t in tracks], dtype=np.float32)

    x_range = float(np.max(x_values) - np.min(x_values))
    y_range = float(np.max(y_values) - np.min(y_values))
    x_range_frac = x_range / width
    y_range_frac = y_range / height

    x_std = float(np.std(x_values))
    y_std = float(np.std(y_values))
    x_std_frac = x_std / width
    y_std_frac = y_std / height

    sizes = np.array([
        float(t.get('width', 0) or 0) * float(t.get('height', 0) or 0)
        for t in tracks
    ], dtype=np.float32)
    valid_sizes = sizes[sizes > 0]
    size_variation_frac = 0.0
    size_ratio = 1.0
    min_size_px = None
    max_size_px = None
    if valid_sizes.size >= 2:
        min_size = float(np.min(valid_sizes))
        max_size = float(np.max(valid_sizes))
        min_size_px = min_size
        max_size_px = max_size
        if max_size > 0:
            size_variation_frac = (max_size - min_size) / max_size
        if min_size > 0:
            size_ratio = max_size / min_size

    # Signal 1 (side-angle): meaningful XY movement.
    x_motion_threshold = 0.05
    y_motion_threshold = 0.10
    if camera_position == 'behind_shooter':
        x_motion_threshold = 0.03
        y_motion_threshold = 0.07
    xy_motion_ok = (x_range_frac > x_motion_threshold) or (y_range_frac > y_motion_threshold)
    # Signal 2 (behind-basket): depth motion via scale change.
    size_change_ok = size_ratio > 1.30

    # Signal 3: ball disappears in middle of window (net/occlusion event).
    track_frames = [int(t.get('frame', 0)) for t in tracks]
    frame_diffs = np.diff(np.array(track_frames, dtype=np.int32)) if len(track_frames) >= 2 else np.array([], dtype=np.int32)
    expected_step = int(np.median(frame_diffs)) if frame_diffs.size > 0 else 1
    expected_step = max(1, expected_step)
    shot_start = int(shot_frames[0]) if shot_frames else track_frames[0]
    shot_end = int(shot_frames[-1]) if shot_frames else track_frames[-1]
    shot_span = max(1, shot_end - shot_start + 1)
    mid_start = shot_start + int(shot_span * 0.20)
    mid_end = shot_end - int(shot_span * 0.20)
    longest_mid_gap_frames = 0
    for i in range(1, len(track_frames)):
        gap_frames = max(0, track_frames[i] - track_frames[i - 1] - expected_step)
        if gap_frames <= 0:
            continue
        gap_start = track_frames[i - 1] + expected_step
        gap_end = track_frames[i] - expected_step
        gap_mid = (gap_start + gap_end) / 2.0
        if mid_start <= gap_mid <= mid_end:
            longest_mid_gap_frames = max(longest_mid_gap_frames, gap_frames)
    disappearance_gap_ok = longest_mid_gap_frames >= 5

    triggered_signals = []
    if xy_motion_ok:
        triggered_signals.append('xy_motion')
    if size_change_ok:
        triggered_signals.append('size_change')
    if disappearance_gap_ok:
        triggered_signals.append('ball_disappearance')

    signals = {
        'track_count': len(tracks),
        'x_range_px': round(x_range, 2),
        'y_range_px': round(y_range, 2),
        'x_range_frac': round(x_range_frac, 4),
        'y_range_frac': round(y_range_frac, 4),
        'x_std_px': round(x_std, 2),
        'y_std_px': round(y_std, 2),
        'x_std_frac': round(x_std_frac, 5),
        'y_std_frac': round(y_std_frac, 5),
        'xy_motion_ok': xy_motion_ok,
        'x_motion_threshold': x_motion_threshold,
        'y_motion_threshold': y_motion_threshold,
        'camera_position': camera_position,
        'size_ratio': round(size_ratio, 3),
        'min_size_px': round(min_size_px, 2) if min_size_px is not None else None,
        'max_size_px': round(max_size_px, 2) if max_size_px is not None else None,
        'size_variation_frac': round(size_variation_frac, 4),
        'size_change_ok': size_change_ok,
        'disappearance_gap_ok': disappearance_gap_ok,
        'longest_mid_gap_frames': int(longest_mid_gap_frames),
        'expected_step': int(expected_step),
        'triggered_signals': triggered_signals
    }

    # Reject as minimal movement ONLY if all three motion signals fail.
    if not xy_motion_ok and not size_change_ok and not disappearance_gap_ok:
        return {
            'is_valid_shot': False,
            'confidence': 0.05,
            'reason': 'minimal_xy_movement',
            'signals': signals
        }

    return {
        'is_valid_shot': True,
        'confidence': 0.65 + 0.1 * (len(triggered_signals) - 1),
        'reason': f"trajectory_signal:{'+'.join(triggered_signals)}",
        'signals': signals
    }


def validate_shot_trajectory(shot_frames, ball_tracks, frame_width, frame_height, camera_position=None):
    """Return True when trajectory resembles a jump shot, else False."""
    validation = get_shot_trajectory_validation(
        shot_frames=shot_frames,
        ball_tracks=ball_tracks,
        frame_width=frame_width,
        frame_height=frame_height,
        camera_position=camera_position
    )
    validate_shot_trajectory.last_validation = validation
    return bool(validation.get('is_valid_shot', False))


def is_likely_rebound(ball_tracks, rim_zone, net_zone, camera_position=None):
    """Detect if ball movement pattern suggests a rebound rather than a shot."""
    if not ball_tracks or len(ball_tracks) < 5:
        return {'is_rebound': False, 'confidence': 0.0, 'reason': 'insufficient_tracks'}

    zone = _normalize_zone_box(rim_zone) or _normalize_zone_box(net_zone)
    if zone is None:
        return {'is_rebound': False, 'confidence': 0.0, 'reason': 'no_rim_or_net'}

    ordered = sorted(ball_tracks, key=lambda t: t.get('frame', 0))
    window = max(3, len(ordered) // 4)
    early_tracks = ordered[:window]
    late_tracks = ordered[-window:]

    early_positions = [_track_xy(t) for t in early_tracks]
    late_positions = [_track_xy(t) for t in late_tracks]

    early_x = sum(p[0] for p in early_positions) / len(early_positions)
    early_y = sum(p[1] for p in early_positions) / len(early_positions)
    late_x = sum(p[0] for p in late_positions) / len(late_positions)
    late_y = sum(p[1] for p in late_positions) / len(late_positions)

    early_sizes = [
        float(t.get('width', 0)) * float(t.get('height', 0))
        for t in early_tracks
        if float(t.get('width', 0) or 0) > 0 and float(t.get('height', 0) or 0) > 0
    ]
    late_sizes = [
        float(t.get('width', 0)) * float(t.get('height', 0))
        for t in late_tracks
        if float(t.get('width', 0) or 0) > 0 and float(t.get('height', 0) or 0) > 0
    ]

    if early_sizes and late_sizes:
        avg_early_size = sum(early_sizes) / len(early_sizes)
        avg_late_size = sum(late_sizes) / len(late_sizes)
        size_ratio = avg_late_size / avg_early_size if avg_early_size > 0 else 1.0
    else:
        size_ratio = 1.0

    rim_center_x = zone['center_x']
    rim_center_y = zone['center_y']
    rim_width = zone['width']
    rim_height = zone['height']

    rim_region_hits = 0
    rim_region_x_tol = rim_width * 1.5
    rim_region_y_tol = rim_height * 2.5
    for t in ordered:
        x, y = _track_xy(t)
        if abs(x - rim_center_x) <= rim_region_x_tol and abs(y - rim_center_y) <= rim_region_y_tol:
            rim_region_hits += 1

    total_movement = math.sqrt((late_x - early_x) ** 2 + (late_y - early_y) ** 2)
    early_dist_to_rim = math.sqrt((early_x - rim_center_x) ** 2 + (early_y - rim_center_y) ** 2)
    late_dist_to_rim = math.sqrt((late_x - rim_center_x) ** 2 + (late_y - rim_center_y) ** 2)

    if camera_position and camera_position.startswith('sideline'):
        starts_near_rim = early_dist_to_rim < rim_width * 1.5
    else:
        starts_near_rim = early_dist_to_rim < rim_width * 2.5
    tight_rim_x_tol = (rim_width / 2.0) * 1.2
    tight_rim_y_tol = (rim_height / 2.0) * 1.2
    starts_in_tight_rim_box = (
        abs(early_x - rim_center_x) <= tight_rim_x_tol and
        abs(early_y - rim_center_y) <= tight_rim_y_tol
    )
    moves_away = late_dist_to_rim > early_dist_to_rim * 1.15
    rim_distance_ratio = late_dist_to_rim / max(early_dist_to_rim, 1.0)
    strong_moves_away = late_dist_to_rim > early_dist_to_rim and rim_distance_ratio >= 2.0
    gets_larger = size_ratio > 1.1
    if camera_position and camera_position.startswith('sideline'):
        minimal_movement_threshold = 50.0
    elif camera_position == 'behind_shooter':
        minimal_movement_threshold = 15.0
    elif camera_position == 'in_front_of_shooter':
        minimal_movement_threshold = 35.0
    else:
        minimal_movement_threshold = 30.0
    minimal_movement = total_movement < minimal_movement_threshold
    size_unchanged = 0.9 < size_ratio < 1.1
    no_shrink = size_ratio > 0.85

    if camera_position == 'behind_shooter':
        size_changed_significantly = not (0.85 < size_ratio < 1.15)
        if size_changed_significantly:
            minimal_movement = False

    signals = {
        'starts_near_rim': starts_near_rim,
        'starts_in_tight_rim_box': starts_in_tight_rim_box,
        'moves_away': moves_away,
        'strong_moves_away': strong_moves_away,
        'distance_ratio': round(float(rim_distance_ratio), 3),
        'gets_larger': gets_larger,
        'minimal_movement': minimal_movement,
        'size_unchanged': size_unchanged,
        'no_shrink': no_shrink
    }

    is_unknown_camera = camera_position in (None, '', 'unknown')
    is_behind_camera = camera_position == 'behind_shooter'

    if is_behind_camera and minimal_movement and size_unchanged:
        return {
            'is_rebound': False,
            'confidence': 0.0,
            'reason': 'behind_stationary_lenient',
            'signals': signals
        }

    # Behind-the-shooter / unknown camera views are prone to false "rebound" triggers
    # because the ball often doesn't visit the rim region within the pose-based window.
    # Be strict: only flag classic rebound patterns that clearly originate at the rim.
    if (is_behind_camera or is_unknown_camera) and not starts_near_rim:
        return {
            'is_rebound': False,
            'confidence': 0.0,
            'reason': 'no_rim_origin',
            'signals': signals
        }

    # Behind-the-shooter: only flag rebound if ball clearly visits rim region.
    if is_behind_camera and rim_region_hits < 2:
        return {
            'is_rebound': False,
            'confidence': 0.0,
            'reason': 'no_rim_contact',
            'signals': {**signals, 'rim_region_hits': rim_region_hits}
        }
    # Sideline: reduce false positives by requiring clear rim-region contact.
    if camera_position and camera_position.startswith('sideline') and rim_region_hits < 4:
        return {
            'is_rebound': False,
            'confidence': 0.0,
            'reason': 'no_rim_contact',
            'signals': {**signals, 'rim_region_hits': rim_region_hits}
        }
    if camera_position and camera_position.startswith('sideline'):
        if total_movement < rim_width * 1.5:
            return {
                'is_rebound': False,
                'confidence': 0.0,
                'reason': 'sideline_movement_small',
                'signals': {**signals, 'rim_region_hits': rim_region_hits}
            }

    if starts_in_tight_rim_box and strong_moves_away and rim_region_hits >= 2:
        return {
            'is_rebound': True,
            'confidence': 0.90,
            'reason': 'strong_rim_origin_rebound_pattern',
            'signals': signals
        }

    if camera_position == 'in_front_of_shooter':
        # Behind-basket view: rebounds often begin near rim and then move away
        # toward the lower corners with little shrink-away pattern.
        if starts_near_rim and moves_away and rim_region_hits >= 2 and total_movement >= rim_width * 1.2 and no_shrink:
            return {
                'is_rebound': True,
                'confidence': 0.78,
                'reason': 'front_rim_origin_rebound_pattern',
                'signals': {
                    **signals,
                    'rim_region_hits': rim_region_hits,
                    'total_movement': round(float(total_movement), 2),
                }
            }

    return {
        'is_rebound': False,
        'confidence': 0.0,
        'reason': 'rebound_evidence_not_strong',
        'signals': signals
    }


def _extract_shoulder_y(pose_frames, frame_height):
    if not pose_frames or not frame_height:
        return None

    frames_iter = pose_frames.values() if isinstance(pose_frames, dict) else pose_frames
    for entry in frames_iter:
        landmarks = None
        if entry is None:
            continue
        if hasattr(entry, 'pose_landmarks'):
            landmarks = getattr(entry.pose_landmarks, 'landmark', None)
        elif isinstance(entry, dict):
            if 'pose_landmarks' in entry and hasattr(entry['pose_landmarks'], 'landmark'):
                landmarks = entry['pose_landmarks'].landmark
            elif 'landmarks' in entry:
                landmarks = entry.get('landmarks')
            elif 'landmark' in entry:
                landmarks = entry.get('landmark')
        elif hasattr(entry, 'landmark'):
            landmarks = entry.landmark

        if not landmarks or len(landmarks) <= 12:
            continue

        left = landmarks[11]
        right = landmarks[12]
        left_vis = getattr(left, 'visibility', 0.0)
        right_vis = getattr(right, 'visibility', 0.0)
        if left_vis > 0.5 or right_vis > 0.5:
            left_y = getattr(left, 'y', 0.0)
            right_y = getattr(right, 'y', 0.0)
            return float(min(left_y, right_y)) * float(frame_height)

    return None


def is_likely_dribble(ball_tracks, pose_frames, frame_height, camera_position=None, rim_zone=None):
    """
    Conservative dribble filter:
    flag dribble only when ALL strict conditions are true.
    """
    if not ball_tracks or len(ball_tracks) < 5:
        return {'is_dribble': False, 'confidence': 0.0, 'reason': 'insufficient_tracks'}

    positions = [_track_xy(t) for t in ball_tracks]
    y_values = [float(p[1]) for p in positions]
    x_values = [float(p[0]) for p in positions]
    frame_height = float(frame_height or 0.0)
    if frame_height <= 0.0:
        frame_height = max(720.0, max(y_values) if y_values else 720.0)

    ball_stays_low_all = bool(y_values) and all(y > frame_height * 0.60 for y in y_values)

    direction_reversals = 0
    for i in range(2, len(y_values)):
        prev_dir = y_values[i - 1] - y_values[i - 2]
        curr_dir = y_values[i] - y_values[i - 1]
        if prev_dir * curr_dir < 0 and abs(prev_dir) > 3 and abs(curr_dir) > 3:
            direction_reversals += 1
    has_three_reversals = direction_reversals >= 3

    normalized_rim = _normalize_zone_box(rim_zone)
    ball_never_near_rim_2x = False
    near_rim_hits = 0
    if normalized_rim:
        rim_x1 = float(normalized_rim['x1'])
        rim_y1 = float(normalized_rim['y1'])
        rim_x2 = float(normalized_rim['x2'])
        rim_y2 = float(normalized_rim['y2'])
        rim_cx = (rim_x1 + rim_x2) / 2.0
        rim_cy = (rim_y1 + rim_y2) / 2.0
        rim_w = max(1.0, rim_x2 - rim_x1)
        rim_h = max(1.0, rim_y2 - rim_y1)
        x_tol = rim_w
        y_tol = rim_h
        near_rim_hits = sum(
            1 for x, y in zip(x_values, y_values)
            if abs(x - rim_cx) <= x_tol and abs(y - rim_cy) <= y_tol
        )
        ball_never_near_rim_2x = near_rim_hits == 0

    sizes = [
        float(t.get('width', 0) or 0) * float(t.get('height', 0) or 0)
        for t in ball_tracks
        if float(t.get('width', 0) or 0) > 0 and float(t.get('height', 0) or 0) > 0
    ]
    size_ratio = 1.0
    no_significant_size_change = False
    if len(sizes) >= 2:
        min_size = max(1.0, min(sizes))
        max_size = max(sizes)
        size_ratio = max_size / min_size
        no_significant_size_change = size_ratio < 1.2

    signals = {
        'ball_stays_low_all': ball_stays_low_all,
        'direction_reversals': int(direction_reversals),
        'has_three_reversals': has_three_reversals,
        'ball_never_near_rim_2x': ball_never_near_rim_2x,
        'near_rim_hits': int(near_rim_hits),
        'size_ratio': round(float(size_ratio), 3),
        'no_significant_size_change': no_significant_size_change,
        'rim_zone_available': normalized_rim is not None
    }

    if normalized_rim is None:
        return {
            'is_dribble': False,
            'confidence': 0.0,
            'reason': 'no_rim_zone_for_strict_dribble_check',
            'signals': signals
        }

    all_conditions_met = (
        ball_stays_low_all
        and has_three_reversals
        and ball_never_near_rim_2x
        and no_significant_size_change
    )
    if all_conditions_met:
        return {
            'is_dribble': True,
            'confidence': 0.90,
            'reason': 'strict_dribble_pattern',
            'signals': signals
        }

    return {
        'is_dribble': False,
        'confidence': 0.0,
        'reason': 'strict_dribble_conditions_not_met',
        'signals': signals
    }


def filter_false_positives(
    shots,
    all_ball_tracks,
    rim_zone,
    net_zone,
    pose_frames,
    frame_height,
    camera_position=None,
    yolo_detections=None,
    yolo_rim_position=None,
    debug=None
):
    """Filter out rebounds and dribbles from detected shots."""
    if debug is None:
        debug = {}

    debug['false_positive_filter'] = {
        'input_count': len(shots),
        'rebounds_removed': 0,
        'rebounds_kept_shoot_model_confirmed': 0,
        'dribbles_removed': 0,
        'shot_validity_removed': 0,
        'shoot_model_skipped': 0,
        'details': []
    }

    if not shots:
        debug['false_positive_filter']['output_count'] = 0
        return []

    ball_ordered, ball_frames = build_ball_track_index(all_ball_tracks or [])
    filtered_shots = []

    for shot in shots:
        start_frame = int(shot.get('start_frame', shot.get('release_frame', 0)))
        end_frame = int(shot.get('end_frame', shot.get('release_frame', start_frame)))
        release_frame = int(shot.get('release_frame', start_frame))
        shot_frames = list(range(start_frame, end_frame + 1))

        shot_tracks = get_tracks_in_window(ball_ordered, ball_frames, start_frame, end_frame)
        shoot_model_confirmed = bool(shot.get('shoot_model_confirmed', False))
        front_mode = str(camera_position or '') == 'in_front_of_shooter'
        if shoot_model_confirmed and not front_mode:
            shot_for_keep = dict(shot)
            shot_for_keep['_validity_tracks'] = shot_tracks
            shot_for_keep['_skip_validity_filter'] = True
            filtered_shots.append(shot_for_keep)
            debug['false_positive_filter']['shoot_model_skipped'] += 1
            debug['false_positive_filter']['details'].append({
                'release_frame': release_frame,
                'filtered_as': 'kept',
                'reason': 'shoot_model_confirmed_skip'
            })
            continue

        rebound_result = {'is_rebound': False, 'confidence': 0.0, 'reason': 'disabled'}
        dribble_result = {'is_dribble': False, 'confidence': 0.0, 'reason': 'disabled'}

        yolo_rebound_flag = False
        yolo_dribble_flag = False
        if yolo_detections and yolo_rim_position is not None:
            yolo_rebound_flag = is_likely_rebound_yolo(shot_frames, yolo_detections, yolo_rim_position)
        if yolo_detections:
            yolo_dribble_flag = is_likely_dribble_yolo(shot_frames, yolo_detections)

        if SHOTLAB_FILTER_REBOUNDS:
            rebound_result = is_likely_rebound(shot_tracks, rim_zone, net_zone, camera_position=camera_position)
            if yolo_rebound_flag:
                rebound_result = {
                    'is_rebound': True,
                    'confidence': max(0.9, rebound_result.get('confidence', 0.0)),
                    'reason': 'yolo_rebound_pattern',
                    'signals': rebound_result.get('signals', {})
                }
        if SHOTLAB_FILTER_DRIBBLES:
            dribble_result = is_likely_dribble(
                shot_tracks,
                pose_frames,
                frame_height,
                camera_position=camera_position,
                rim_zone=rim_zone
            )
            if yolo_dribble_flag and dribble_result.get('is_dribble'):
                dribble_result = {
                    'is_dribble': True,
                    'confidence': max(0.9, dribble_result.get('confidence', 0.0)),
                    'reason': 'strict_and_yolo_dribble_pattern',
                    'signals': dribble_result.get('signals', {})
                }
            elif yolo_dribble_flag:
                signals = dict(dribble_result.get('signals', {}))
                signals['yolo_dribble_pattern'] = True
                dribble_result = {
                    'is_dribble': False,
                    'confidence': 0.0,
                    'reason': 'yolo_dribble_ignored_by_strict_filter',
                    'signals': signals
                }
        if shoot_model_confirmed and front_mode:
            dribble_result = {
                'is_dribble': False,
                'confidence': 0.0,
                'reason': 'front_shoot_model_skip_dribble',
                'signals': dribble_result.get('signals', {}) if isinstance(dribble_result, dict) else {}
            }

        print(f"=== FALSE POSITIVE CHECK: Shot {release_frame} ===")
        print(f"  Camera position: {camera_position}")
        print(f"  Tracks in window: {len(shot_tracks)}")
        print(
            "  Rebound check: "
            f"is_rebound={rebound_result.get('is_rebound')}, "
            f"confidence={rebound_result.get('confidence', 0):.2f}, "
            f"reason={rebound_result.get('reason')}"
        )
        if rebound_result.get('signals'):
            signals = rebound_result['signals']
            print(
                "    Signals: "
                f"near_rim={signals.get('starts_near_rim')}, "
                f"tight_origin={signals.get('starts_in_tight_rim_box')}, "
                f"moves_away={signals.get('moves_away')}, "
                f"strong_away={signals.get('strong_moves_away')}, "
                f"dist_ratio={signals.get('distance_ratio')}, "
                f"gets_larger={signals.get('gets_larger')}, "
                f"minimal_move={signals.get('minimal_movement')}, "
                f"size_unchanged={signals.get('size_unchanged')}"
            )
        print(
            "  Dribble check: "
            f"is_dribble={dribble_result.get('is_dribble')}, "
            f"confidence={dribble_result.get('confidence', 0):.2f}, "
            f"reason={dribble_result.get('reason')}"
        )
        if dribble_result.get('signals'):
            signals = dribble_result['signals']
            print(
                "    Signals: "
                f"low_all={signals.get('ball_stays_low_all')}, "
                f"reversals={signals.get('direction_reversals')}, "
                f"near_rim_hits={signals.get('near_rim_hits')}, "
                f"size_ratio={signals.get('size_ratio')}, "
                f"strict_ok={signals.get('ball_never_near_rim_2x') and signals.get('no_significant_size_change')}"
            )
        if len(shot_tracks) >= 5:
            early_tracks = shot_tracks[:5]
            late_tracks = shot_tracks[-5:]
            early_x = sum(_track_xy(t)[0] for t in early_tracks) / len(early_tracks)
            late_x = sum(_track_xy(t)[0] for t in late_tracks) / len(late_tracks)
            early_y = sum(_track_xy(t)[1] for t in early_tracks) / len(early_tracks)
            late_y = sum(_track_xy(t)[1] for t in late_tracks) / len(late_tracks)

            early_sizes = [
                float(t.get('width', 0)) * float(t.get('height', 0))
                for t in early_tracks
                if float(t.get('width', 0) or 0) > 0 and float(t.get('height', 0) or 0) > 0
            ]
            late_sizes = [
                float(t.get('width', 0)) * float(t.get('height', 0))
                for t in late_tracks
                if float(t.get('width', 0) or 0) > 0 and float(t.get('height', 0) or 0) > 0
            ]
            avg_early_size = sum(early_sizes) / len(early_sizes) if early_sizes else 0
            avg_late_size = sum(late_sizes) / len(late_sizes) if late_sizes else 0

            print(
                f"  Trajectory: early_pos=({early_x:.0f},{early_y:.0f}), "
                f"late_pos=({late_x:.0f},{late_y:.0f})"
            )
            if avg_early_size > 0:
                print(
                    "  Size change: "
                    f"early={avg_early_size:.0f}, late={avg_late_size:.0f}, "
                    f"ratio={avg_late_size / avg_early_size:.2f}"
                )
            else:
                print("  Size change: N/A")

        keep_high_conf_shoot_model_rebound = (
            front_mode
            and shoot_model_confirmed
            and float(shot.get('shoot_model_confidence', 0.0) or 0.0) >= float(BB_SHOOT_CONFIRMED_REBOUND_KEEP_CONFIDENCE)
        )
        if rebound_result['is_rebound'] and rebound_result['confidence'] >= SHOTLAB_REBOUND_CONFIDENCE_THRESHOLD:
            if keep_high_conf_shoot_model_rebound:
                debug['false_positive_filter']['rebounds_kept_shoot_model_confirmed'] += 1
                debug['false_positive_filter']['details'].append({
                    'release_frame': release_frame,
                    'filtered_as': 'kept',
                    'reason': 'shoot_model_confirmed_override_rebound',
                    'shoot_model_confidence': float(shot.get('shoot_model_confidence', 0.0) or 0.0),
                    'rebound_reason': rebound_result.get('reason'),
                })
            else:
                debug['false_positive_filter']['rebounds_removed'] += 1
                debug['false_positive_filter']['details'].append({
                    'release_frame': release_frame,
                    'filtered_as': 'rebound',
                    'confidence': rebound_result['confidence'],
                    'reason': rebound_result['reason']
                })
                if SHOTLAB_DEBUG_REJECTED_SHOTS:
                    print(
                        f"Filtered shot {release_frame} as rebound "
                        f"(confidence={rebound_result['confidence']:.2f}, reason={rebound_result['reason']})"
                    )
                continue

        if dribble_result['is_dribble'] and dribble_result['confidence'] >= SHOTLAB_DRIBBLE_CONFIDENCE_THRESHOLD:
            debug['false_positive_filter']['dribbles_removed'] += 1
            debug['false_positive_filter']['details'].append({
                'release_frame': release_frame,
                'filtered_as': 'dribble',
                'confidence': dribble_result['confidence'],
                'reason': dribble_result['reason']
            })
            if SHOTLAB_DEBUG_REJECTED_SHOTS:
                print(
                    f"Filtered shot {release_frame} as dribble "
                    f"(confidence={dribble_result['confidence']:.2f}, reason={dribble_result['reason']})"
                )
            continue

        shot_for_keep = dict(shot)
        shot_for_keep['_validity_tracks'] = shot_tracks
        filtered_shots.append(shot_for_keep)

    if SHOTLAB_V5_SHOT_VALIDITY_ENABLE and filtered_shots:
        locked_shots = [s for s in filtered_shots if bool(s.get('_skip_validity_filter', False))]
        validity_pool = [s for s in filtered_shots if not bool(s.get('_skip_validity_filter', False))]
        validity_model = get_shot_validity_model()
        kept_shots, validity_debug = apply_shot_validity_filter(
            validity_pool,
            model=validity_model,
            min_probability=SHOTLAB_V5_SHOT_VALIDITY_THRESHOLD,
            max_drop_share=SHOTLAB_V5_SHOT_VALIDITY_MAX_DROP_SHARE,
        ) if validity_pool else ([], {'input_count': 0, 'kept_count': 0, 'dropped_count': 0})
        dropped_by_validity = max(0, len(validity_pool) - len(kept_shots))
        debug['false_positive_filter']['shot_validity_removed'] = int(dropped_by_validity)
        debug['false_positive_filter']['shot_validity'] = validity_debug
        filtered_shots = sorted(
            kept_shots + locked_shots,
            key=lambda s: int(s.get('release_frame', s.get('start_frame', 0)))
        )

    debug['false_positive_filter']['output_count'] = len(filtered_shots)
    if SHOTLAB_DEBUG_REJECTED_SHOTS:
        print(
            "False positive filter: "
            f"{len(shots)} → {len(filtered_shots)} "
            f"(removed {debug['false_positive_filter']['rebounds_removed']} rebounds, "
            f"kept {debug['false_positive_filter']['rebounds_kept_shoot_model_confirmed']} rebound-overrides, "
            f"{debug['false_positive_filter']['dribbles_removed']} dribbles, "
            f"{debug['false_positive_filter']['shot_validity_removed']} low-validity, "
            f"skipped {debug['false_positive_filter']['shoot_model_skipped']} shoot-model confirmed)"
        )
    for shot in filtered_shots:
        if '_validity_tracks' in shot:
            del shot['_validity_tracks']
        if '_skip_validity_filter' in shot:
            del shot['_skip_validity_filter']
    return filtered_shots

def update_shotlab_status(stage, detail=None, progress=None):
    shotlab_status['stage'] = stage
    if detail is not None:
        shotlab_status['detail'] = detail
    if progress is not None:
        shotlab_status['progress'] = float(progress)
    shotlab_status['updated_at'] = time.time()

def log_shotlab_summary(
    processing_seconds,
    shots_analysis,
    pose_shot_count,
    dedupe_debug,
    filter_debug,
    ball_debug,
    rim_position,
    rim_zone,
    net_zone,
    transform_matrix,
    shot_detection_debug=None,
    warnings=None,
    unknown_zones=None,
    anchor_source_counts=None,
    unknown_zone_reasons=None
):
    total_attempts = len(shots_analysis)
    total_makes = sum(1 for s in shots_analysis if s.get('outcome') == 'make')
    total_misses = total_attempts - total_makes
    make_reasons = defaultdict(int)
    miss_reasons = defaultdict(int)
    for shot in shots_analysis:
        reason = shot.get('debug', {}).get('outcome_reason') or 'unknown'
        if shot.get('outcome') == 'make':
            make_reasons[reason] += 1
        elif shot.get('outcome') == 'miss':
            miss_reasons[reason] += 1

    make_reason = max(make_reasons, key=make_reasons.get) if make_reasons else None
    miss_reason = max(miss_reasons, key=miss_reasons.get) if miss_reasons else None

    dedupe_kept = dedupe_debug.get('kept') if dedupe_debug else None
    dedupe_dropped = dedupe_debug.get('dropped') if dedupe_debug else None
    filter_dropped = filter_debug.get('dropped') if filter_debug else None
    drop_reasons = filter_debug.get('dropped_reasons') if filter_debug else {}

    print("\n=== SHOTLAB ANALYSIS SUMMARY ===")
    print(f"Processing time: {processing_seconds / 60.0:.2f} minutes")
    print(f"Shots detected: {total_attempts}")
    make_suffix = f" ({make_reason})" if make_reason else ""
    miss_suffix = f" ({miss_reason})" if miss_reason else ""
    print(f"  - Makes: {total_makes}{make_suffix}")
    print(f"  - Misses: {total_misses}{miss_suffix}")

    print("\nShot Detection Pipeline:")
    print(f"  - Pose detections: {pose_shot_count}")
    yolo_launch_debug = (shot_detection_debug or {}).get('yolo_launch', {}) if shot_detection_debug else {}
    if yolo_launch_debug:
        print(f"  - YOLO launch detections: {yolo_launch_debug.get('launch_candidates', 0)}")
    shoot_model_debug = (shot_detection_debug or {}).get('shoot_model', {}) if shot_detection_debug else {}
    if shoot_model_debug:
        print(f"  - Shoot model detections: {shoot_model_debug.get('candidate_hits', 0)}")
        print(f"  - Shoot model gap fills: {shoot_model_debug.get('gap_fill_added', 0)}")
    if dedupe_kept is not None and dedupe_dropped is not None:
        print(f"  - After deduplication: {dedupe_kept} (dropped {dedupe_dropped} duplicates)")
    if filter_dropped is not None:
        trajectory = filter_debug.get('trajectory_filter') if filter_debug else None
        if trajectory:
            traj_input = trajectory.get('input_count', 0)
            traj_valid = trajectory.get('valid_shots', 0)
            traj_invalid = trajectory.get('invalid_shots', 0)
            print(f"  - After trajectory validation: {traj_valid} (dropped {traj_invalid})")
        ball_flight = filter_debug.get('ball_flight_confirmation') if filter_debug else None
        if ball_flight:
            bf_input = ball_flight.get('input_count', 0)
            bf_confirmed = ball_flight.get('confirmed', 0)
            bf_rejected = ball_flight.get('rejected', 0)
            print(f"  - After ball-flight confirmation: {bf_confirmed} (dropped {bf_rejected} / {bf_input})")
        if drop_reasons:
            for reason, count in drop_reasons.items():
                print(f"      - {reason}: {count}")
        dropped_shots = filter_debug.get('dropped_shots') if filter_debug else []
        if dropped_shots:
            for idx, shot in enumerate(dropped_shots, start=1):
                release_frame = shot.get('release_frame')
                reason = shot.get('reason', 'unknown')
                print(f"      - Shot {idx} (release {release_frame}): {reason}")
        false_positive = filter_debug.get('false_positive_filter') if filter_debug else None
        if false_positive:
            output_count = false_positive.get('output_count', None)
            rebounds_removed = false_positive.get('rebounds_removed', 0)
            dribbles_removed = false_positive.get('dribbles_removed', 0)
            shoot_model_skipped = false_positive.get('shoot_model_skipped', 0)
            if output_count is not None:
                print(f"  - After false positive filter: {output_count}")
            print("  - False positive filter:")
            print(f"      - Rebounds removed: {rebounds_removed}")
            print(f"      - Dribbles removed: {dribbles_removed}")
            print(f"      - Shoot-model skips: {shoot_model_skipped}")
        yolo_share_cap = filter_debug.get('yolo_share_cap') if filter_debug else None
        if yolo_share_cap:
            max_allowed = yolo_share_cap.get('max_yolo_allowed')
            dropped = yolo_share_cap.get('dropped', 0)
            print(
                "  - YOLO share cap:"
                f" pose={yolo_share_cap.get('pose_count', 0)}"
                f", yolo={yolo_share_cap.get('yolo_count', 0)}"
                f", max_allowed={max_allowed}"
                f", dropped={dropped}"
            )
        duplicate_echo = filter_debug.get('duplicate_echo_suppression') if filter_debug else None
        if duplicate_echo and duplicate_echo.get('applied'):
            print(
                "  - Behind-basket duplicate suppression:"
                f" input={duplicate_echo.get('input_count', 0)}"
                f", output={duplicate_echo.get('output_count', 0)}"
                f", dropped={duplicate_echo.get('dropped', 0)}"
            )

    if shots_analysis:
        print("\nShot-by-shot detail:")
        for shot in shots_analysis:
            shot_number = int(shot.get('shot_number', 0) or 0)
            release_frame = int(shot.get('release_frame', 0) or 0)
            shot_debug = shot.get('debug') or {}
            source = str(shot_debug.get('candidate_source') or 'pose')
            shoot_model_confirmed = bool(shot_debug.get('shoot_model_confirmed', False))
            if source == 'shoot_model_only':
                source_text = 'shoot_model_only (gap fill)'
            elif shoot_model_confirmed:
                source_text = f"{source} + shoot_model ✓"
            else:
                source_text = f"{source} only"
            print(f"  - Shot {shot_number} (frame {release_frame}): {source_text}")

    print("\nBall Tracking:")
    if ball_debug:
        coverage = ball_debug.get('coverage')
        coverage_pct = f"{coverage * 100:.1f}%" if coverage is not None else "n/a"
        valid_pct = ball_debug.get('valid_non_rim_detection_pct')
        valid_pct_str = f"{valid_pct * 100:.1f}%" if valid_pct is not None else "n/a"
        print(f"  - Total tracks: {ball_debug.get('tracks', 0)}")
        print(f"  - Detection coverage: {coverage_pct}")
        print(f"  - Valid non-rim coverage: {valid_pct_str}")
        print(f"  - Rim-excluded candidates: {ball_debug.get('rim_excluded_candidates', 0)}")
        print(f"  - Tracks with rim exclusions: {ball_debug.get('tracks_with_rim_exclusions', 0)}")
        print(f"  - Timeouts: {ball_debug.get('timeouts', 0)}")

    print("\nCalibration:")
    rim_available = rim_position is not None or rim_zone is not None
    rim_status = "OK" if rim_available else "MISSING"
    court_status = "OK" if transform_matrix is not None else "Not calibrated"
    net_status = "OK" if net_zone is not None else "Not calibrated"
    print(f"  - Rim: {rim_status}{' ✅' if rim_available else ' ❌'}")
    print(f"  - Net zone: {net_status}")
    print(f"  - Court: {court_status}")

    if unknown_zones is not None:
        print("\nCourt Mapping:")
        print(f"  - Unknown zones: {int(unknown_zones)} / {total_attempts}")
    if anchor_source_counts:
        print("  - Anchor sources:")
        for source, count in sorted(anchor_source_counts.items()):
            print(f"      - {source}: {count}")
    if unknown_zone_reasons:
        print("  - Unknown zone reasons:")
        for reason, count in sorted(unknown_zone_reasons.items()):
            print(f"      - {reason}: {count}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  ⚠️ {warning}")
    if not rim_available and net_zone is None:
        print("\n⚠️ WARNING: Rim calibration missing - results may be inaccurate!")

def get_player_foot_position_from_landmarks(landmarks, width, height):
    left_ankle = get_3d_point(landmarks, 27, width, height)
    right_ankle = get_3d_point(landmarks, 28, width, height)
    if left_ankle is None or right_ankle is None:
        return None
    return [
        float((left_ankle[0] + right_ankle[0]) / 2.0),
        float((left_ankle[1] + right_ankle[1]) / 2.0)
    ]

def get_player_hip_position_from_landmarks(landmarks, width, height):
    left_hip = get_3d_point(landmarks, 23, width, height)
    right_hip = get_3d_point(landmarks, 24, width, height)
    if left_hip is None or right_hip is None:
        return None
    return [
        float((left_hip[0] + right_hip[0]) / 2.0),
        float((left_hip[1] + right_hip[1]) / 2.0)
    ]

def get_right_wrist_point(landmarks, width, height, camera_position=None):
    wrist_threshold = get_visibility_threshold(camera_position, 'wrist')
    wrist = get_3d_point(landmarks, 16, width, height, min_visibility=wrist_threshold)
    if wrist is None:
        return None
    return [float(wrist[0]), float(wrist[1])]


def get_left_wrist_point(landmarks, width, height, camera_position=None):
    wrist_threshold = get_visibility_threshold(camera_position, 'wrist')
    wrist = get_3d_point(landmarks, 15, width, height, min_visibility=wrist_threshold)
    if wrist is None:
        return None
    return [float(wrist[0]), float(wrist[1])]


def get_right_elbow_point(landmarks, width, height, camera_position=None):
    elbow_threshold = get_visibility_threshold(camera_position, 'elbow')
    elbow = get_3d_point(landmarks, 14, width, height, min_visibility=elbow_threshold)
    if elbow is None:
        return None
    return [float(elbow[0]), float(elbow[1])]


def get_left_elbow_point(landmarks, width, height, camera_position=None):
    elbow_threshold = get_visibility_threshold(camera_position, 'elbow')
    elbow = get_3d_point(landmarks, 13, width, height, min_visibility=elbow_threshold)
    if elbow is None:
        return None
    return [float(elbow[0]), float(elbow[1])]


def get_right_shoulder_point(landmarks, width, height, camera_position=None):
    shoulder_threshold = get_visibility_threshold(camera_position, 'shoulder')
    shoulder = get_3d_point(landmarks, 12, width, height, min_visibility=shoulder_threshold)
    if shoulder is None:
        return None
    return [float(shoulder[0]), float(shoulder[1])]


def get_left_shoulder_point(landmarks, width, height, camera_position=None):
    shoulder_threshold = get_visibility_threshold(camera_position, 'shoulder')
    shoulder = get_3d_point(landmarks, 11, width, height, min_visibility=shoulder_threshold)
    if shoulder is None:
        return None
    return [float(shoulder[0]), float(shoulder[1])]

def compute_release_angles(landmarks, width, height, camera_position=None, visibility_overrides=None):
    overrides = visibility_overrides if isinstance(visibility_overrides, dict) else {}
    shoulder_threshold = float(overrides.get('shoulder', get_visibility_threshold(camera_position, 'shoulder')))
    elbow_threshold = float(overrides.get('elbow', get_visibility_threshold(camera_position, 'elbow')))
    wrist_threshold = float(overrides.get('wrist', get_visibility_threshold(camera_position, 'wrist')))
    lower_body_threshold = float(overrides.get('hip', min(shoulder_threshold, SHOTLAB_POSE_MIN_VISIBILITY)))
    right_shoulder = get_3d_point(landmarks, 12, width, height, min_visibility=shoulder_threshold)
    right_elbow = get_3d_point(landmarks, 14, width, height, min_visibility=elbow_threshold)
    right_wrist = get_3d_point(landmarks, 16, width, height, min_visibility=wrist_threshold)
    right_index = get_3d_point(landmarks, 20, width, height, min_visibility=wrist_threshold)
    left_shoulder = get_3d_point(landmarks, 11, width, height, min_visibility=shoulder_threshold)
    left_elbow = get_3d_point(landmarks, 13, width, height, min_visibility=elbow_threshold)
    left_wrist = get_3d_point(landmarks, 15, width, height, min_visibility=wrist_threshold)
    left_index = get_3d_point(landmarks, 19, width, height, min_visibility=wrist_threshold)
    right_hip = get_3d_point(landmarks, 24, width, height, min_visibility=lower_body_threshold)
    left_hip = get_3d_point(landmarks, 23, width, height, min_visibility=lower_body_threshold)
    right_knee = get_3d_point(landmarks, 26, width, height, min_visibility=lower_body_threshold)
    left_knee = get_3d_point(landmarks, 25, width, height, min_visibility=lower_body_threshold)
    right_ankle = get_3d_point(landmarks, 28, width, height, min_visibility=lower_body_threshold)
    left_ankle = get_3d_point(landmarks, 27, width, height, min_visibility=lower_body_threshold)

    right_complete = (right_shoulder is not None and right_elbow is not None and right_wrist is not None)
    left_complete = (left_shoulder is not None and left_elbow is not None and left_wrist is not None)

    shooting_arm = None
    shoulder_pt = None
    elbow_pt = None
    wrist_pt = None
    index_pt = None
    opposite_shoulder = None
    if right_complete:
        shooting_arm = 'right'
        shoulder_pt = right_shoulder
        elbow_pt = right_elbow
        wrist_pt = right_wrist
        index_pt = right_index
        opposite_shoulder = left_shoulder
    elif left_complete:
        shooting_arm = 'left'
        shoulder_pt = left_shoulder
        elbow_pt = left_elbow
        wrist_pt = left_wrist
        index_pt = left_index
        opposite_shoulder = right_shoulder

    if shooting_arm is None:
        if SHOTLAB_DEBUG_POSE:
            count = getattr(compute_release_angles, '_debug_count', 0) + 1
            compute_release_angles._debug_count = count
            if count <= 3:
                visibility = {
                    idx: float(landmarks[idx].visibility)
                    for idx in [11, 12, 13, 14, 15, 16, 19, 20]
                    if idx < len(landmarks)
                }
                print(f"\n⚠️  compute_release_angles missing both arm chains (sample {count})")
                print(f"   visibility: {visibility}")
        return None

    elbow_angle = calculate_3d_angle(shoulder_pt, elbow_pt, wrist_pt)
    if index_pt is not None:
        wrist_angle = calculate_3d_angle(elbow_pt, wrist_pt, index_pt)
    else:
        wrist_angle = SHOTLAB_WRIST_FALLBACK_ANGLE
    arm_angle = calculate_3d_angle(opposite_shoulder, shoulder_pt, elbow_pt) if opposite_shoulder is not None else None
    raw_form_score = compute_overall_form(elbow_angle, wrist_angle, arm_angle)

    right_elbow_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
    left_elbow_angle = calculate_3d_angle(left_shoulder, left_elbow, left_wrist)
    right_shoulder_angle = calculate_3d_angle(right_hip, right_shoulder, right_elbow)
    left_shoulder_angle = calculate_3d_angle(left_hip, left_shoulder, left_elbow)
    right_hip_angle = calculate_3d_angle(right_shoulder, right_hip, right_knee)
    left_hip_angle = calculate_3d_angle(left_shoulder, left_hip, left_knee)
    right_knee_angle = calculate_3d_angle(right_hip, right_knee, right_ankle)
    left_knee_angle = calculate_3d_angle(left_hip, left_knee, left_ankle)
    right_wrist_angle = calculate_3d_angle(right_elbow, right_wrist, right_index) if right_index is not None else None
    left_wrist_angle = calculate_3d_angle(left_elbow, left_wrist, left_index) if left_index is not None else None

    follow_through_extension = None
    if shooting_arm == 'right' and right_shoulder is not None and right_wrist is not None:
        norm = max(1.0, float(height) * 0.35)
        follow_through_extension = max(0.0, min(1.0, float((right_shoulder[1] - right_wrist[1]) / norm)))
    elif shooting_arm == 'left' and left_shoulder is not None and left_wrist is not None:
        norm = max(1.0, float(height) * 0.35)
        follow_through_extension = max(0.0, min(1.0, float((left_shoulder[1] - left_wrist[1]) / norm)))

    return {
        'elbow_angle': float(elbow_angle) if elbow_angle is not None else None,
        'wrist_angle': float(wrist_angle) if wrist_angle is not None else None,
        'arm_angle': float(arm_angle) if arm_angle is not None else None,
        'form_score': float(raw_form_score) if raw_form_score is not None else None,
        'raw_form_score_before_scaling': float(raw_form_score) if raw_form_score is not None else None,
        'right_elbow_angle': float(right_elbow_angle) if right_elbow_angle is not None else None,
        'left_elbow_angle': float(left_elbow_angle) if left_elbow_angle is not None else None,
        'right_shoulder_angle': float(right_shoulder_angle) if right_shoulder_angle is not None else None,
        'left_shoulder_angle': float(left_shoulder_angle) if left_shoulder_angle is not None else None,
        'right_hip_angle': float(right_hip_angle) if right_hip_angle is not None else None,
        'left_hip_angle': float(left_hip_angle) if left_hip_angle is not None else None,
        'right_knee_angle': float(right_knee_angle) if right_knee_angle is not None else None,
        'left_knee_angle': float(left_knee_angle) if left_knee_angle is not None else None,
        'right_wrist_angle': float(right_wrist_angle) if right_wrist_angle is not None else None,
        'left_wrist_angle': float(left_wrist_angle) if left_wrist_angle is not None else None,
        'follow_through_extension': float(follow_through_extension) if follow_through_extension is not None else None,
        'shooting_arm': shooting_arm,
    }


def _get_yolo_detection_for_frame(yolo_detections, frame_idx):
    if not yolo_detections:
        return {}
    det = yolo_detections.get(int(frame_idx))
    if det is None:
        det = yolo_detections.get(str(int(frame_idx)))
    return det if isinstance(det, dict) else {}


def _get_yolo_players_with_source_frame(yolo_detections, frame_idx, max_gap_frames=0):
    max_gap = max(0, int(max_gap_frames or 0))
    best_players = []
    best_frame = None
    best_gap = None
    for gap in range(0, max_gap + 1):
        candidates = [int(frame_idx)] if gap == 0 else [int(frame_idx) - gap, int(frame_idx) + gap]
        for cand_frame in candidates:
            if cand_frame < 0:
                continue
            det = _get_yolo_detection_for_frame(yolo_detections, cand_frame)
            players = det.get('players', []) if isinstance(det, dict) else []
            if players:
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_players = players
                    best_frame = int(cand_frame)
        if best_gap is not None:
            break
    return best_players, best_frame


def _get_yolo_players_near_frame(yolo_detections, frame_idx, max_gap_frames=0):
    players, _ = _get_yolo_players_with_source_frame(
        yolo_detections,
        frame_idx,
        max_gap_frames=max_gap_frames,
    )
    return players


def _bbox_iou_xyxy(a, b):
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != 4 or len(b) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
    except Exception:
        return 0.0
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def _bbox_center_xy(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    return (
        float((float(bbox[0]) + float(bbox[2])) * 0.5),
        float((float(bbox[1]) + float(bbox[3])) * 0.5),
    )


def _sanitize_bbox_xyxy(bbox, frame_shape):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    h, w = frame_shape[:2]
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(x1, float(max(1, w - 1))))
        x2 = max(x1 + 1.0, min(x2, float(max(1, w - 1))))
        y1 = max(0.0, min(y1, float(max(1, h - 1))))
        y2 = max(y1 + 1.0, min(y2, float(max(1, h - 1))))
        return [x1, y1, x2, y2]
    except Exception:
        return None


def _is_valid_player_bbox_for_rtmpose(bbox, frame_shape):
    if bbox is None:
        return False
    h, w = frame_shape[:2]
    bw = max(0.0, float(bbox[2]) - float(bbox[0]))
    bh = max(0.0, float(bbox[3]) - float(bbox[1]))
    if bw < float(max(1.0, float(w) * float(BB_RTMPOSE_MIN_PLAYER_WIDTH_RATIO))):
        return False
    aspect = bh / max(1e-6, bw)
    if aspect < float(BB_RTMPOSE_MIN_PLAYER_BBOX_ASPECT_RATIO):
        return False
    if aspect > float(BB_RTMPOSE_MAX_PLAYER_BBOX_ASPECT_RATIO):
        return False
    return True


def _is_static_candidate_by_motion(yolo_detections, frame_idx, candidate_bbox, frame_shape):
    lookback_frame = int(frame_idx) - int(BB_RTMPOSE_STATIC_BBOX_LOOKBACK_FRAMES)
    if lookback_frame < 0:
        return False
    prior_players, _ = _get_yolo_players_with_source_frame(
        yolo_detections,
        lookback_frame,
        max_gap_frames=0,
    )
    if not prior_players:
        return False
    sorted_prior_players = sorted(
        prior_players,
        key=lambda p: float((p or {}).get('confidence', 0.0) or 0.0),
        reverse=True,
    )
    best_iou = 0.0
    matched_prior_bbox = None
    for prior in sorted_prior_players:
        prior_bbox = _sanitize_bbox_xyxy((prior or {}).get('bbox'), frame_shape)
        if prior_bbox is None:
            continue
        iou = _bbox_iou_xyxy(candidate_bbox, prior_bbox)
        if iou > best_iou:
            best_iou = float(iou)
            matched_prior_bbox = prior_bbox
    if matched_prior_bbox is None:
        return False
    center_now = _bbox_center_xy(candidate_bbox)
    center_prev = _bbox_center_xy(matched_prior_bbox)
    if center_now is None or center_prev is None:
        return False
    movement = math.sqrt(
        (float(center_now[0]) - float(center_prev[0])) ** 2
        + (float(center_now[1]) - float(center_prev[1])) ** 2
    )
    frame_w = float(max(1.0, frame_shape[1]))
    min_motion = float(BB_RTMPOSE_STATIC_CENTER_MIN_MOTION_RATIO) * frame_w
    return movement < min_motion


def _select_person_candidate_from_yolo(
    yolo_detections,
    frame_idx,
    frame_shape,
    behind_basket_mode=False,
    max_gap_frames=None,
):
    if max_gap_frames is None:
        search_gap = BB_RTMPOSE_PLAYER_FRAME_SEARCH_GAP if behind_basket_mode else 0
    else:
        search_gap = max(0, int(max_gap_frames))
    players, source_frame = _get_yolo_players_with_source_frame(
        yolo_detections,
        frame_idx,
        max_gap_frames=search_gap,
    )
    if not players:
        return None
    sorted_players = sorted(
        players,
        key=lambda p: float((p or {}).get('confidence', 0.0) or 0.0),
        reverse=True,
    )
    for player in sorted_players:
        confidence = float((player or {}).get('confidence', 0.0) or 0.0)
        candidate_bbox = _sanitize_bbox_xyxy((player or {}).get('bbox'), frame_shape)
        if candidate_bbox is None:
            continue
        if behind_basket_mode:
            if not _is_valid_player_bbox_for_rtmpose(candidate_bbox, frame_shape):
                continue
            if _is_static_candidate_by_motion(yolo_detections, frame_idx, candidate_bbox, frame_shape):
                continue
        return {
            'bbox': candidate_bbox,
            'confidence': confidence,
            'source_frame': int(source_frame) if source_frame is not None else int(frame_idx),
        }
    return None


def _select_person_bbox_from_yolo(yolo_detections, frame_idx, frame_shape, behind_basket_mode=False):
    candidate = _select_person_candidate_from_yolo(
        yolo_detections,
        frame_idx,
        frame_shape,
        behind_basket_mode=behind_basket_mode,
    )
    return (candidate or {}).get('bbox')


def process_video_for_pose(
    video_path,
    frame_stride=1,
    progress_callback=None,
    camera_position=None,
    active_regions=None,
    visibility_overrides=None,
    pose_backend='mediapipe',
    yolo_detections=None,
    rtmpose_estimator=None,
):
    """Process video for pose landmarks and shot state per frame."""
    backend = str(pose_backend or 'mediapipe').strip().lower()
    estimator = rtmpose_estimator
    if backend == 'rtmpose':
        if estimator is None:
            estimator = get_rtmpose_estimator()
        if estimator is None or not bool(getattr(estimator, 'available', False)):
            logger.warning("RTMPose backend unavailable; falling back to MediaPipe for pose pass.")
            backend = 'mediapipe'
    if backend == 'mediapipe' and not MEDIAPIPE_AVAILABLE:
        return [], 0.0

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    pose_frames = []
    last_progress_update = -1
    stats = {
        'total_frames': 0,
        'frames_with_pose': 0,
        'frames_with_angles': 0,
        'frames_with_form_score': 0,
        'frames_skipped_by_stride': 0,
        'frames_skipped_by_active_regions': 0,
        'missing_landmarks': [],
        'angle_computation_failures': [],
        'active_regions': []
    }
    last_rtmpose_person_bbox = None
    last_rtmpose_person_bbox_frame = -1
    rtmpose_bbox_fallback_counts = {
        'guarded': 0,
        'reuse_last': 0,
        'none': 0,
        'full_frame': 0,
    }
    regions = []
    if active_regions:
        for start, end in active_regions:
            s = max(0, int(start))
            e = max(s, int(end))
            regions.append((s, e))
        regions = merge_frame_windows(regions)
    stats['active_regions'] = regions
    active_idx = 0
    if regions:
        logger.info("Pose active-region mode enabled with %d region(s): %s", len(regions), regions)
    logger.info("Pose pass backend: %s", backend)

    if SHOTLAB_DEBUG_POSE:
        print("\n=== SHOT DETECTION DEBUG ===")
        print(f"  Camera position: {camera_position}")
        print(
            f"  Wrist visibility threshold: "
            f"{(visibility_overrides or {}).get('wrist', get_visibility_threshold(camera_position, 'wrist'))}"
        )
        print(
            f"  Elbow visibility threshold: "
            f"{(visibility_overrides or {}).get('elbow', get_visibility_threshold(camera_position, 'elbow'))}"
        )
        print(
            f"  Shoulder visibility threshold: "
            f"{(visibility_overrides or {}).get('shoulder', get_visibility_threshold(camera_position, 'shoulder'))}"
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if regions:
            while active_idx < len(regions) and frame_idx > regions[active_idx][1]:
                active_idx += 1
            in_active_region = (
                active_idx < len(regions)
                and regions[active_idx][0] <= frame_idx <= regions[active_idx][1]
            )
            if not in_active_region:
                stats['frames_skipped_by_active_regions'] += 1
                frame_idx += 1
                continue
        if frame_stride > 1 and frame_idx % frame_stride != 0:
            stats['frames_skipped_by_stride'] += 1
            frame_idx += 1
            continue

        stats['total_frames'] += 1
        h, w, _ = frame.shape
        timestamp = frame_idx / fps if fps else 0.0
        if backend == 'rtmpose':
            behind_basket_mode = (str(camera_position or '') == 'in_front_of_shooter')
            person_bbox = _select_person_bbox_from_yolo(
                yolo_detections,
                frame_idx,
                frame.shape,
                behind_basket_mode=behind_basket_mode,
            )
            if person_bbox is not None:
                rtmpose_bbox_fallback_counts['guarded'] += 1
                last_rtmpose_person_bbox = list(person_bbox)
                last_rtmpose_person_bbox_frame = int(frame_idx)
            elif behind_basket_mode:
                # Safety fallback: keep form scoring alive even if strict guards temporarily reject all boxes.
                if (
                    last_rtmpose_person_bbox is not None
                    and int(frame_idx) - int(last_rtmpose_person_bbox_frame) <= 12
                ):
                    person_bbox = list(last_rtmpose_person_bbox)
                    rtmpose_bbox_fallback_counts['reuse_last'] += 1
                else:
                    rtmpose_bbox_fallback_counts['none'] += 1
            if estimator is None:
                results = None
            else:
                if behind_basket_mode and person_bbox is None:
                    # Keep scoring alive when filtered YOLO boxes are unavailable for a frame.
                    rtmpose_bbox_fallback_counts['full_frame'] += 1
                results = estimator.infer(frame, person_bbox=person_bbox)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

        if results is not None and getattr(results, 'pose_landmarks', None):
            stats['frames_with_pose'] += 1
            landmarks = results.pose_landmarks.landmark
            state = get_arm_state(landmarks, w, h, camera_position=camera_position)
            foot_pos = get_player_foot_position_from_landmarks(landmarks, w, h)
            hip_pos = get_player_hip_position_from_landmarks(landmarks, w, h)
            landmark_visibility = {
                'right_shoulder': float(landmarks[12].visibility) if len(landmarks) > 12 else None,
                'right_elbow': float(landmarks[14].visibility) if len(landmarks) > 14 else None,
                'right_wrist': float(landmarks[16].visibility) if len(landmarks) > 16 else None,
                'left_shoulder': float(landmarks[11].visibility) if len(landmarks) > 11 else None,
                'left_elbow': float(landmarks[13].visibility) if len(landmarks) > 13 else None,
                'left_wrist': float(landmarks[15].visibility) if len(landmarks) > 15 else None,
            }
            angles = compute_release_angles(
                landmarks,
                w,
                h,
                camera_position=camera_position,
                visibility_overrides=visibility_overrides,
            )
            if angles is not None:
                stats['frames_with_angles'] += 1
                if angles.get('form_score') is not None:
                    stats['frames_with_form_score'] += 1
                else:
                    if SHOTLAB_DEBUG_POSE and len(stats['angle_computation_failures']) < 3:
                        stats['angle_computation_failures'].append({
                            'frame': frame_idx,
                            'angles': angles,
                            'reason': 'form_score is None'
                        })
            else:
                if SHOTLAB_DEBUG_POSE and len(stats['missing_landmarks']) < 3:
                    stats['missing_landmarks'].append({
                        'frame': frame_idx,
                        'reason': 'angles returned None'
                    })
            right_wrist = get_right_wrist_point(landmarks, w, h, camera_position=camera_position)
            left_wrist = get_left_wrist_point(landmarks, w, h, camera_position=camera_position)
            right_elbow = get_right_elbow_point(landmarks, w, h, camera_position=camera_position)
            left_elbow = get_left_elbow_point(landmarks, w, h, camera_position=camera_position)
            right_shoulder = get_right_shoulder_point(landmarks, w, h, camera_position=camera_position)
            left_shoulder = get_left_shoulder_point(landmarks, w, h, camera_position=camera_position)
            if SHOTLAB_DEBUG_POSE and state in ('pre_shot', 'follow_through'):
                def _lm(idx):
                    return landmarks[idx] if idx < len(landmarks) else None
                right_shoulder_lm = _lm(12)
                right_elbow_lm = _lm(14)
                right_wrist_lm = _lm(16)
                def _vis(lm):
                    return f"{lm.visibility:.2f}" if lm is not None else "n/a"
                def _y(lm):
                    return f"{lm.y:.3f}" if lm is not None else "n/a"
                print(f"  Frame {frame_idx}: arm_state={state}")
                print(
                    f"    Shoulder vis={_vis(right_shoulder_lm)}, "
                    f"Elbow vis={_vis(right_elbow_lm)}, "
                    f"Wrist vis={_vis(right_wrist_lm)}"
                )
                print(
                    f"    Shoulder Y={_y(right_shoulder_lm)}, "
                    f"Elbow Y={_y(right_elbow_lm)}, "
                    f"Wrist Y={_y(right_wrist_lm)}"
                )
        else:
            state = "neutral"
            foot_pos = None
            hip_pos = None
            angles = None
            landmark_visibility = None
            right_wrist = None
            left_wrist = None
            right_elbow = None
            left_elbow = None
            right_shoulder = None
            left_shoulder = None

        raw_keypoints = None
        if results is not None and getattr(results, 'pose_landmarks', None):
            lms = results.pose_landmarks.landmark
            kps = []
            for kp_name, kp_idx in BB_SKELETON_KEYPOINT_INDICES.items():
                if kp_idx < len(lms):
                    lm = lms[kp_idx]
                    kps.append({
                        'name': kp_name,
                        'x': round(float(lm.x), 4),
                        'y': round(float(lm.y), 4),
                        'conf': round(float(lm.visibility), 3),
                    })
            if kps:
                raw_keypoints = kps

        pose_frames.append({
            'frame_idx': frame_idx,
            'timestamp': float(timestamp),
            'state': state,
            'foot_pos': foot_pos,
            'hip_pos': hip_pos,
            'angles': angles,
            'right_wrist': right_wrist,
            'left_wrist': left_wrist,
            'right_elbow': right_elbow,
            'left_elbow': left_elbow,
            'right_shoulder': right_shoulder,
            'left_shoulder': left_shoulder,
            'landmark_visibility': landmark_visibility,
            'frame_height': h,
            'raw_keypoints': raw_keypoints,
        })
        frame_idx += 1

        if progress_callback and total_frames > 0:
            progress = min(1.0, frame_idx / total_frames)
            if frame_idx % max(1, 10 * frame_stride) == 0 and progress != last_progress_update:
                last_progress_update = progress
                progress_callback(frame_idx, total_frames, progress)

    cap.release()
    logger.info(
        "Pose processing stats: processed=%d with_pose=%d skipped_stride=%d skipped_active=%d stride=%d",
        stats['total_frames'],
        stats['frames_with_pose'],
        stats['frames_skipped_by_stride'],
        stats['frames_skipped_by_active_regions'],
        int(frame_stride or 1)
    )
    if backend == 'rtmpose':
        logger.info(
            "RTMPose person-bbox selection: guarded=%d reuse_last=%d none=%d full_frame=%d",
            int(rtmpose_bbox_fallback_counts['guarded']),
            int(rtmpose_bbox_fallback_counts['reuse_last']),
            int(rtmpose_bbox_fallback_counts['none']),
            int(rtmpose_bbox_fallback_counts['full_frame']),
        )
    if SHOTLAB_DEBUG_POSE and stats['total_frames'] > 0:
        print("\n=== POSE PROCESSING STATISTICS ===")
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Frames with pose detected: {stats['frames_with_pose']} ({stats['frames_with_pose']/stats['total_frames']*100:.1f}%)")
        print(f"Frames with angles computed: {stats['frames_with_angles']} ({stats['frames_with_angles']/stats['total_frames']*100:.1f}%)")
        print(f"Frames with form_score: {stats['frames_with_form_score']} ({stats['frames_with_form_score']/stats['total_frames']*100:.1f}%)")
        if stats['missing_landmarks']:
            print(f"\n⚠️  Angles missing in {len(stats['missing_landmarks'])} frames")
            print(f"   Sample failures: {stats['missing_landmarks']}")
        if stats['angle_computation_failures']:
            print(f"\n⚠️  Form score missing in {len(stats['angle_computation_failures'])} frames with angles")
            print(f"   Sample failures: {stats['angle_computation_failures']}")
    return pose_frames, fps

def interpolate_missing_angles(pose_frames):
    """Fill missing angles by interpolating from nearby frames."""
    if not pose_frames:
        return pose_frames
    frames_with_scores = [
        e for e in pose_frames
        if e.get('angles') is not None and e['angles'].get('form_score') is not None
    ]
    if len(frames_with_scores) < 2:
        return pose_frames

    for entry in pose_frames:
        if entry.get('angles') is not None and entry['angles'].get('form_score') is not None:
            continue
        frame_idx = entry.get('frame_idx', 0)
        before = [e for e in frames_with_scores if e.get('frame_idx', 0) < frame_idx]
        after = [e for e in frames_with_scores if e.get('frame_idx', 0) > frame_idx]
        if before and after:
            prev = before[-1]
            nxt = after[0]
            gap = max(1, int(nxt['frame_idx'] - prev['frame_idx']))
            weight = (frame_idx - prev['frame_idx']) / gap
            interpolated = {}
            for key in ['elbow_angle', 'wrist_angle', 'arm_angle', 'form_score']:
                val_prev = prev['angles'].get(key)
                val_next = nxt['angles'].get(key)
                if val_prev is not None and val_next is not None:
                    interpolated[key] = float(val_prev) * (1 - weight) + float(val_next) * weight
            if interpolated:
                entry['angles'] = interpolated
        elif before:
            entry['angles'] = dict(before[-1]['angles'])
        elif after:
            entry['angles'] = dict(after[0]['angles'])
    return pose_frames

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

def detect_shot_attempts_from_pose_state_machine(pose_frames, fps, camera_position=None):
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
        'skipped_no_neutral_reset': 0,
        'allow_follow_only': False,
        'warmup_follow_only_used': 0
    }
    if not pose_frames or not fps:
        return [], debug

    is_sideline = bool(camera_position and str(camera_position).startswith('sideline'))
    min_gap_frames = max(1, int(SHOTLAB_SHOT_MIN_GAP_SECONDS * fps))
    max_duration_frames = max(1, int(SHOTLAB_SHOT_MAX_DURATION_SECONDS * fps))
    pre_follow_gap_frames = max(1, int(SHOTLAB_PRE_FOLLOW_MAX_GAP_SECONDS * fps))
    if is_sideline:
        min_gap_frames = max(1, int(min_gap_frames * max(0.5, SHOTLAB_SIDELINE_MIN_GAP_MULT)))
        pre_follow_gap_frames = max(1, int(pre_follow_gap_frames * max(1.0, SHOTLAB_SIDELINE_PRE_FOLLOW_GAP_MULT)))
    min_release_to_end_frames = max(1, int(0.3 * fps))
    follow_only_lookback_frames = max(1, int(SHOTLAB_FOLLOW_ONLY_LOOKBACK_SECONDS * fps))
    warmup_follow_only_frames = max(1, int(SHOTLAB_POSE_WARMUP_FOLLOW_ONLY_SECONDS * fps))

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
    neutral_frames = sorted(int(e.get('frame_idx', 0)) for e in pose_frames if e.get('state') == 'neutral')

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
    allow_follow_only = (
        SHOTLAB_ALLOW_FOLLOW_ONLY
        or len(pre_shot_entries) < max(1, int(SHOTLAB_MIN_PRESHOT_ENTRIES_FOR_FULL_GATING))
    )
    if is_sideline and SHOTLAB_SIDELINE_ALLOW_FOLLOW_ONLY:
        allow_follow_only = True
    follow_candidates = []
    for segment in follow_segments:
        if is_sideline and len(segment) < 2:
            continue
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
        if is_sideline and wrist_velocity_map:
            sideline_v = float(wrist_velocity_map.get(best['frame_idx'], 0.0) or 0.0)
            if sideline_v < 30.0 and len(segment) < 3:
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
        elif allow_follow_only or release_frame <= warmup_follow_only_frames:
            start_frame = max(0, release_frame - follow_only_lookback_frames)
            start_time = release_time
            debug['follow_only_used'] += 1
            if release_frame <= warmup_follow_only_frames:
                debug['warmup_follow_only_used'] += 1
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

    def has_neutral_reset(prev_release, curr_release):
        if prev_release is None:
            return True
        lo = int(min(prev_release, curr_release))
        hi = int(max(prev_release, curr_release))
        for frame in neutral_frames:
            if lo < frame < hi:
                return True
        return False

    for cand in release_candidates:
        release_frame = cand['frame_idx']
        release_time = cand.get('timestamp', 0.0)
        score = float(cand.get('score', 1.0))
        gap = release_frame - last_release_frame

        if is_sideline and shots:
            if not has_neutral_reset(last_release_frame, release_frame) and gap < int(1.6 * float(fps or 30.0)):
                debug['skipped_no_neutral_reset'] += 1
                continue

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

def detect_shot_attempts_from_pose_skeleton(pose_frames, fps, camera_position=None):
    """Detect shots using skeleton-viewer recording logic (pre_shot -> follow_through before neutral)."""
    debug = {
        'pre_shot_entries': 0,
        'follow_entries': 0,
        'cancelled_no_follow': 0,
        'shots_detected': 0,
        'mode': 'skeleton'
    }
    if not pose_frames or not fps:
        return [], debug

    pre_shot_entries = [e for e in pose_frames if e.get('state') == 'pre_shot']
    follow_entries = [e for e in pose_frames if e.get('state') == 'follow_through']
    debug['pre_shot_entries'] = len(pre_shot_entries)
    debug['follow_entries'] = len(follow_entries)

    min_gap_frames = max(1, int(SHOTLAB_SHOT_MIN_GAP_SECONDS * fps))
    require_wrist_drop = camera_position not in ('sideline_left', 'sideline_right')
    follow_hold_frames_required = max(1, int(SHOTLAB_NEUTRAL_GRACE_SECONDS * fps))

    shots = []
    in_shot = False
    seen_follow = False
    start_frame = None
    start_time = None
    release_frame = None
    release_time = None
    last_release_frame = -min_gap_frames
    follow_hold_frames = 0

    for entry in pose_frames:
        state = entry.get('state', 'neutral')
        frame_idx = int(entry.get('frame_idx', 0))
        timestamp = entry.get('timestamp', 0.0)

        if not in_shot:
            if state == 'pre_shot':
                if frame_idx - last_release_frame < min_gap_frames:
                    continue
                in_shot = True
                seen_follow = False
                start_frame = frame_idx
                start_time = timestamp
                release_frame = None
                release_time = None
            continue

        # in_shot
        if not seen_follow:
            if state == 'follow_through':
                seen_follow = True
                release_frame = frame_idx
                release_time = timestamp
                follow_hold_frames = 0
            elif state == 'neutral':
                in_shot = False
                seen_follow = False
                start_frame = None
                start_time = None
                release_frame = None
                release_time = None
                debug['cancelled_no_follow'] += 1
        else:
            if state == 'follow_through':
                follow_hold_frames += 1
            if state == 'neutral' or (not require_wrist_drop and state == 'pre_shot') or (
                not require_wrist_drop and state == 'follow_through' and follow_hold_frames >= follow_hold_frames_required
            ):
                end_frame = frame_idx
                if release_frame is not None:
                    shots.append({
                        'start_frame': start_frame,
                        'release_frame': release_frame,
                        'end_frame': end_frame,
                        'timestamp': float(release_time if release_time is not None else start_time or 0.0)
                    })
                    last_release_frame = release_frame
                in_shot = False
                seen_follow = False
                start_frame = None
                start_time = None
                release_frame = None
                release_time = None

    if in_shot and seen_follow and release_frame is not None:
        end_frame = int(pose_frames[-1].get('frame_idx', release_frame))
        shots.append({
            'start_frame': start_frame,
            'release_frame': release_frame,
            'end_frame': end_frame,
            'timestamp': float(release_time if release_time is not None else start_time or 0.0)
        })

    debug['shots_detected'] = len(shots)
    return shots, debug

def detect_shot_attempts_from_pose_hybrid(pose_frames, fps, camera_position=None):
    """Multi-signal scoring shot detection (pose + jump + wrist velocity)."""
    debug = {
        'pre_shot_entries': 0,
        'follow_entries': 0,
        'velocity_candidates': 0,
        'score_candidates': 0,
        'kept_after_dedupe': 0,
        'jump_rejected': 0,
        'jump_missing': 0,
        'shots_detected': 0,
        'mode': 'score'
    }
    if not pose_frames or not fps:
        return [], debug

    pre_shot_entries = [e for e in pose_frames if e.get('state') == 'pre_shot']
    follow_entries = [e for e in pose_frames if e.get('state') == 'follow_through']
    debug['pre_shot_entries'] = len(pre_shot_entries)
    debug['follow_entries'] = len(follow_entries)

    min_gap_frames = max(1, int(SHOTLAB_SHOT_MIN_GAP_SECONDS * fps))
    window_frames = max(3, int(0.6 * fps))
    min_end_gap_frames = max(1, int(0.4 * fps))
    is_sideline = bool(camera_position and str(camera_position).startswith('sideline'))

    # Build wrist velocity map.
    wrist_entries = [e for e in pose_frames if e.get('right_wrist') is not None]
    wrist_velocity = {}
    velocities = []
    for i in range(1, len(wrist_entries)):
        prev = wrist_entries[i - 1]
        curr = wrist_entries[i]
        dt = max(1e-4, curr['timestamp'] - prev['timestamp'])
        vy_up = -(curr['right_wrist'][1] - prev['right_wrist'][1]) / dt
        wrist_velocity[curr['frame_idx']] = float(vy_up)
        if vy_up > 0:
            velocities.append(float(vy_up))

    vel_threshold = None
    if velocities:
        base = float(np.percentile(np.array(velocities, dtype=np.float32), 85))
        vel_threshold = max(20.0, base * 0.7)
    spike_frames = set()
    if vel_threshold is not None:
        for frame_idx, vel in wrist_velocity.items():
            if vel >= vel_threshold:
                spike_frames.add(frame_idx)
    debug['velocity_candidates'] = len(spike_frames)

    # Build pose index map.
    pose_by_frame = {int(e['frame_idx']): e for e in pose_frames}
    sorted_frames = sorted(pose_by_frame.keys())

    def get_window_indices(center):
        start = max(sorted_frames[0], center - window_frames)
        end = min(sorted_frames[-1], center + window_frames)
        return [f for f in sorted_frames if start <= f <= end]

    def find_first_state_after(frames, state, after_frame):
        for f in frames:
            if f > after_frame and pose_by_frame[f].get('state') == state:
                return f
        return None

    def find_last_state_before(frames, state, before_frame):
        for f in reversed(frames):
            if f < before_frame and pose_by_frame[f].get('state') == state:
                return f
        return None

    def jump_confirmed_in_window(frames):
        positions = []
        for f in frames:
            entry = pose_by_frame.get(f)
            if not entry:
                continue
            pos = entry.get('foot_pos') or entry.get('hip_pos')
            if pos is None:
                continue
            positions.append((f, pos[1], entry.get('frame_height')))
        if len(positions) < SHOTLAB_JUMP_MIN_FRAMES:
            return None
        positions.sort(key=lambda x: x[0])
        sample_count = min(3, len(positions))
        baseline = float(np.median(np.array([y for _, y, _ in positions[:sample_count]], dtype=np.float32)))
        min_y = min(y for _, y, _ in positions)
        frame_height = next((h for _, _, h in positions if h), None) or 720
        threshold = max(4.0, SHOTLAB_JUMP_MIN_FRAC * frame_height)
        return (baseline - min_y) >= threshold

    # Seed candidates from follow_through and wrist spikes.
    seeds = set(spike_frames)
    for entry in follow_entries:
        seeds.add(int(entry['frame_idx']))
    if not seeds:
        return [], debug

    candidates = []
    for seed in sorted(seeds):
        window = get_window_indices(seed)
        if not window:
            continue
        states = [pose_by_frame[f].get('state', 'neutral') for f in window]
        has_pre = 'pre_shot' in states
        has_follow = 'follow_through' in states
        has_spike = any(f in spike_frames for f in window)
        jump_ok = jump_confirmed_in_window(window)

        if is_sideline and not has_follow:
            continue

        score = 0
        if has_pre:
            score += 2
        if has_follow:
            score += 2
        if has_spike:
            score += 2
        if jump_ok:
            score += 3

        min_score = 6 if is_sideline else 5
        if score < min_score:
            continue

        # Determine release frame: first follow_through in window, else max velocity.
        release_frame = None
        if has_follow:
            release_frame = next((f for f in window if pose_by_frame[f].get('state') == 'follow_through'), None)
        if release_frame is None and has_spike:
            release_frame = max(window, key=lambda f: wrist_velocity.get(f, 0.0))
        if release_frame is None:
            release_frame = seed

        start_frame = find_last_state_before(window, 'pre_shot', release_frame) or max(window[0], release_frame - int(0.4 * fps))
        end_frame = find_first_state_after(window, 'pre_shot', release_frame)
        if end_frame is None:
            end_frame = find_first_state_after(window, 'neutral', release_frame)
        if end_frame is None:
            end_frame = min(window[-1], release_frame + int(1.2 * fps))

        if SHOTLAB_REQUIRE_JUMP:
            if jump_ok is False:
                debug['jump_rejected'] += 1
                continue
            if jump_ok is None:
                debug['jump_missing'] += 1

        candidates.append({
            'start_frame': int(start_frame),
            'release_frame': int(release_frame),
            'end_frame': int(max(end_frame, release_frame + min_end_gap_frames)),
            'timestamp': float(pose_by_frame[release_frame].get('timestamp', 0.0)),
            'score': score
        })

    debug['score_candidates'] = len(candidates)
    if not candidates:
        return [], debug

    # Deduplicate by release frame proximity, keep highest score.
    candidates.sort(key=lambda c: (c['release_frame'], -c['score']))
    shots = []
    for cand in candidates:
        if not shots:
            shots.append(cand)
            continue
        if cand['release_frame'] - shots[-1]['release_frame'] < min_gap_frames:
            if cand['score'] > shots[-1].get('score', 0):
                shots[-1] = cand
            continue
        shots.append(cand)

    debug['kept_after_dedupe'] = len(shots)
    debug['shots_detected'] = len(shots)
    return shots, debug

def detect_shot_attempts_from_pose(pose_frames, fps, camera_position=None):
    """Select shot detection strategy for ShotLab sessions."""
    mode = (SHOTLAB_SHOT_DETECTION_MODE or 'state_machine').lower()
    is_sideline = bool(camera_position and str(camera_position).startswith('sideline'))

    def _merge_pose_candidates(candidate_lists):
        merged = []
        for shots in candidate_lists:
            for shot in shots or []:
                entry = dict(shot)
                entry['source'] = str(entry.get('source') or 'pose')
                merged.append(entry)
        if not merged:
            return []
        merged.sort(key=lambda s: int(s.get('release_frame', s.get('start_frame', 0))))
        merge_gap = max(1, int((0.95 if is_sideline else SHOTLAB_SHOT_MIN_GAP_SECONDS) * float(fps or 30.0)))
        deduped = []
        for shot in merged:
            release = int(shot.get('release_frame', shot.get('start_frame', 0)))
            if not deduped:
                deduped.append(shot)
                continue
            prev = deduped[-1]
            prev_release = int(prev.get('release_frame', prev.get('start_frame', 0)))
            if release - prev_release < merge_gap:
                curr_score = float(shot.get('score', 1.0) or 1.0)
                prev_score = float(prev.get('score', 1.0) or 1.0)
                if curr_score > prev_score:
                    deduped[-1] = shot
                continue
            deduped.append(shot)
        return deduped

    if mode == 'auto' and is_sideline:
        state_shots, state_debug = detect_shot_attempts_from_pose_state_machine(
            pose_frames,
            fps,
            camera_position=camera_position
        )
        hybrid_shots, hybrid_debug = detect_shot_attempts_from_pose_hybrid(
            pose_frames,
            fps,
            camera_position=camera_position
        )
        merged = _merge_pose_candidates([state_shots, hybrid_shots])
        debug = {
            'mode': 'sideline_fusion',
            'state_machine': state_debug,
            'hybrid': hybrid_debug,
            'shots_detected': len(merged),
        }
        return merged, debug
    if mode == 'hybrid':
        return detect_shot_attempts_from_pose_hybrid(pose_frames, fps, camera_position=camera_position)
    if mode == 'skeleton':
        return detect_shot_attempts_from_pose_skeleton(pose_frames, fps, camera_position=camera_position)
    if is_sideline:
        state_shots, state_debug = detect_shot_attempts_from_pose_state_machine(
            pose_frames,
            fps,
            camera_position=camera_position
        )
        hybrid_shots, hybrid_debug = detect_shot_attempts_from_pose_hybrid(
            pose_frames,
            fps,
            camera_position=camera_position
        )
        merged = _merge_pose_candidates([state_shots, hybrid_shots])
        debug = {
            'mode': 'sideline_fusion',
            'state_machine': state_debug,
            'hybrid': hybrid_debug,
            'shots_detected': len(merged),
        }
        return merged, debug
    return detect_shot_attempts_from_pose_state_machine(pose_frames, fps, camera_position=camera_position)

def detect_shot_attempts_from_pose_webcam(pose_frames, fps):
    """Detect shots using the webcam logic: pre_shot → follow_through before returning to neutral."""
    debug = {
        'segments': 0,
        'valid_segments': 0,
        'shots_detected': 0
    }
    if not pose_frames or not fps:
        return [], debug

    shots = []
    segment = []
    for entry in pose_frames:
        state = entry.get('state', 'neutral')
        if state == 'neutral':
            if segment:
                debug['segments'] += 1
                shot = _build_shot_from_segment(segment, fps)
                if shot is not None:
                    shots.append(shot)
                    debug['valid_segments'] += 1
                segment = []
            continue
        segment.append(entry)

    if segment:
        debug['segments'] += 1
        shot = _build_shot_from_segment(segment, fps)
        if shot is not None:
            shots.append(shot)
            debug['valid_segments'] += 1

    debug['shots_detected'] = len(shots)
    return shots, debug

def _build_shot_from_segment(segment, fps):
    """Return a shot dict if the segment contains pre_shot followed by follow_through."""
    pre_idx = None
    follow_idx = None
    for idx, entry in enumerate(segment):
        if pre_idx is None and entry.get('state') == 'pre_shot':
            pre_idx = idx
        if pre_idx is not None and entry.get('state') == 'follow_through':
            follow_idx = idx
            break

    if pre_idx is None or follow_idx is None:
        return None

    start_entry = segment[pre_idx]
    release_entry = segment[follow_idx]
    start_frame = int(start_entry['frame_idx'])
    release_frame = int(release_entry['frame_idx'])

    # End frame: first neutral after follow_through isn't in segment; use last entry.
    end_frame = int(segment[-1]['frame_idx'])
    timestamp = release_entry.get('timestamp', start_entry.get('timestamp', 0.0))

    if SHOTLAB_MIN_SHOT_DURATION_FRAMES > 0:
        if (end_frame - start_frame) < SHOTLAB_MIN_SHOT_DURATION_FRAMES:
            return None

    return {
        'start_frame': start_frame,
        'release_frame': release_frame,
        'end_frame': end_frame,
        'timestamp': float(timestamp) if timestamp is not None else 0.0
    }

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


def _is_valid_anchor_point(point):
    if not point or len(point) < 2:
        return False
    x, y = point[0], point[1]
    if x is None or y is None:
        return False
    try:
        x_val = float(x)
        y_val = float(y)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(x_val) or not math.isfinite(y_val):
        return False
    return True


def _find_nearest_valid_anchor(pose_frames, frame_idx, max_frame_gap):
    nearest = None
    for entry in pose_frames or []:
        frame = int(entry.get('frame_idx', 0))
        gap = abs(frame - int(frame_idx))
        if gap > int(max_frame_gap):
            continue
        foot_pos = entry.get('foot_pos')
        hip_pos = entry.get('hip_pos')
        if _is_valid_anchor_point(foot_pos):
            candidate = (gap, 0, frame, (float(foot_pos[0]), float(foot_pos[1])), 'nearest_foot')
            if nearest is None or candidate < nearest:
                nearest = candidate
        if _is_valid_anchor_point(hip_pos):
            candidate = (gap, 1, frame, (float(hip_pos[0]), float(hip_pos[1])), 'nearest_hip')
            if nearest is None or candidate < nearest:
                nearest = candidate
    if nearest is None:
        return None, 'none'
    return nearest[3], nearest[4]


def _extract_anchor_from_player_detection(player_det):
    if not isinstance(player_det, dict):
        return None
    bbox = player_det.get('bbox')
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            if math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2) and x2 > x1 and y2 > y1:
                # Use bottom-center of player box as a foot proxy.
                return (float((x1 + x2) * 0.5), float(y2)), 'yolo_player_bbox_bottom'
        except (TypeError, ValueError):
            pass
    center = player_det.get('center')
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            cx = float(center[0])
            cy = float(center[1])
            if math.isfinite(cx) and math.isfinite(cy):
                return (cx, cy), 'yolo_player_center'
        except (TypeError, ValueError):
            pass
    return None


def _find_yolo_player_anchor(yolo_detections, frame_idx, max_frame_gap):
    if not yolo_detections:
        return None, 'none', 'no_yolo_detections'

    nearest = None
    for key in yolo_detections.keys():
        try:
            frame = int(key)
        except (TypeError, ValueError):
            continue
        gap = abs(frame - int(frame_idx))
        if gap > int(max_frame_gap):
            continue
        det = yolo_detections.get(key, {}) or {}
        players = det.get('players', []) if isinstance(det, dict) else []
        if not players:
            continue
        best_player = max(
            players,
            key=lambda p: float(p.get('area', 0.0) or 0.0) + float(p.get('confidence', 0.0) or 0.0)
        )
        extracted = _extract_anchor_from_player_detection(best_player)
        if extracted is None:
            continue
        anchor, source = extracted
        area = float(best_player.get('area', 0.0) or 0.0)
        candidate = (gap, -area, frame, anchor, source)
        if nearest is None or candidate < nearest:
            nearest = candidate

    if nearest is None:
        return None, 'none', 'no_yolo_player_anchor'
    return nearest[3], nearest[4], None


def resolve_shot_court_anchor(pose_frames, release_frame, release_search_gap_frames, yolo_detections=None):
    """
    Resolve the most reliable player anchor for court mapping.
    Order:
      1) release pose foot
      2) release pose hip
      3) nearest valid foot/hip within release search gap
    """
    pose_entry = get_pose_entry_with_angles(pose_frames, release_frame, release_search_gap_frames)
    if pose_entry is None:
        pose_entry = get_nearest_pose_entry(pose_frames, release_frame)

    if pose_entry is not None:
        foot_pos = pose_entry.get('foot_pos')
        if _is_valid_anchor_point(foot_pos):
            return (float(foot_pos[0]), float(foot_pos[1])), 'foot', None
        hip_pos = pose_entry.get('hip_pos')
        if _is_valid_anchor_point(hip_pos):
            return (float(hip_pos[0]), float(hip_pos[1])), 'hip', None

    nearest_anchor, nearest_source = _find_nearest_valid_anchor(
        pose_frames,
        release_frame,
        release_search_gap_frames
    )
    if nearest_anchor is not None:
        return nearest_anchor, nearest_source, None

    yolo_anchor, yolo_source, yolo_reason = _find_yolo_player_anchor(
        yolo_detections,
        release_frame,
        release_search_gap_frames
    )
    if yolo_anchor is not None:
        return yolo_anchor, yolo_source, None

    if pose_entry is None:
        return None, 'none', f"missing_pose_entry_{yolo_reason or 'no_yolo_fallback'}"
    return None, 'none', f"no_valid_anchor_{yolo_reason or 'no_yolo_fallback'}"

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


def identify_active_regions(
    yolo_detections,
    total_frames,
    min_gap_frames=30,
    min_region_length=15,
    pad_before=30,
    pad_after=60,
    min_ball_confidence=0.30,
    min_motion_px=4.0
):
    """
    Build compact frame regions where ball activity is present.
    Used to avoid running MediaPipe across long inactive stretches.
    """
    if not yolo_detections or not total_frames:
        return []

    candidate = []
    prev_frame = None
    prev_xy = None
    for frame_idx in sorted(int(k) for k in yolo_detections.keys()):
        det = yolo_detections.get(frame_idx, {})
        balls = det.get('balls', [])
        if not balls:
            continue
        best = max(balls, key=lambda b: float(b.get('confidence', 0.0) or 0.0))
        conf = float(best.get('confidence', 0.0) or 0.0)
        if conf < float(min_ball_confidence):
            continue
        if bool(best.get('interpolated')):
            continue
        bx, by = _track_xy(best)
        include = prev_frame is None
        if prev_frame is not None and prev_xy is not None:
            move_dist = _distance_2d((bx, by), prev_xy)
            if move_dist >= float(min_motion_px) or (frame_idx - prev_frame) <= 2:
                include = True
        if include:
            candidate.append(frame_idx)
            prev_frame = frame_idx
            prev_xy = (bx, by)

    if not candidate:
        return []

    regions = []
    region_start = candidate[0]
    prev = candidate[0]
    for frame_idx in candidate[1:]:
        if frame_idx - prev > int(min_gap_frames):
            if prev - region_start >= int(min_region_length):
                regions.append((region_start, prev))
            region_start = frame_idx
        prev = frame_idx
    if prev - region_start >= int(min_region_length):
        regions.append((region_start, prev))

    if not regions:
        return []

    last_frame = max(0, int(total_frames) - 1)
    expanded = []
    for start, end in regions:
        expanded.append((
            max(0, int(start) - int(pad_before)),
            min(last_frame, int(end) + int(pad_after))
        ))

    merged = merge_frame_windows(expanded)
    logger.info(
        "Active regions from YOLO: %d regions (%d candidate frames, total=%d)",
        len(merged),
        len(candidate),
        int(total_frames)
    )
    logger.info("Active regions detail: %s", merged)
    return merged


def build_pose_refine_windows(shot_attempts, total_frames, before_frames, after_frames):
    if not shot_attempts or not total_frames:
        return []
    last_frame = max(0, int(total_frames) - 1)
    windows = []
    for shot in shot_attempts:
        release = int(shot.get('release_frame', shot.get('start_frame', 0)))
        start = max(0, release - int(before_frames))
        end = min(last_frame, release + int(after_frames))
        if end >= start:
            windows.append((start, end))
    return merge_frame_windows(windows)


def build_form_pose_windows(shot_attempts, total_frames):
    return build_pose_refine_windows(
        shot_attempts,
        total_frames=total_frames,
        before_frames=BB_FORM_POSE_WINDOW_BEFORE_FRAMES,
        after_frames=BB_FORM_POSE_WINDOW_AFTER_FRAMES,
    )


def merge_pose_frames(coarse_pose_frames, dense_pose_frames):
    if not coarse_pose_frames:
        return list(dense_pose_frames or [])
    if not dense_pose_frames:
        return list(coarse_pose_frames or [])
    merged = {int(e.get('frame_idx', -1)): e for e in coarse_pose_frames if int(e.get('frame_idx', -1)) >= 0}
    for entry in dense_pose_frames:
        frame_idx = int(entry.get('frame_idx', -1))
        if frame_idx >= 0:
            merged[frame_idx] = entry
    return [merged[k] for k in sorted(merged.keys())]


def run_form_only_pose_pass(video_path, shot_attempts, total_frames, camera_position, pose_frames, yolo_detections=None):
    debug = {
        'applied': False,
        'windows': [],
        'frames_added': 0,
        'elapsed_seconds': 0.0,
        'backend': 'mediapipe',
        'reason': 'disabled_or_no_shots',
    }
    if not shot_attempts or not total_frames:
        return pose_frames, debug
    form_windows = build_form_pose_windows(shot_attempts, total_frames=total_frames)
    if not form_windows:
        debug['reason'] = 'no_form_windows'
        return pose_frames, debug
    started = time.time()
    visibility_overrides = None
    pose_backend = 'mediapipe'
    estimator = None
    if str(camera_position or '') == 'in_front_of_shooter':
        visibility_overrides = {
            'shoulder': float(BB_FORM_SCORING_MIN_VISIBILITY),
            'elbow': float(BB_FORM_SCORING_MIN_VISIBILITY),
            'wrist': float(BB_FORM_SCORING_MIN_VISIBILITY),
        }
        estimator = get_rtmpose_estimator()
        if estimator is not None and bool(getattr(estimator, 'available', False)):
            pose_backend = 'rtmpose'
    debug['backend'] = str(pose_backend)

    refined_pose_frames, _ = process_video_for_pose(
        video_path,
        frame_stride=1,
        progress_callback=None,
        camera_position=camera_position,
        active_regions=form_windows,
        visibility_overrides=visibility_overrides,
        pose_backend=pose_backend,
        yolo_detections=yolo_detections if pose_backend == 'rtmpose' else None,
        rtmpose_estimator=estimator if pose_backend == 'rtmpose' else None,
    )
    if pose_backend == 'rtmpose' and (not refined_pose_frames) and MEDIAPIPE_AVAILABLE:
        logger.warning("RTMPose form-only pass produced no frames; retrying with MediaPipe fallback.")
        debug['backend'] = 'mediapipe_fallback'
        refined_pose_frames, _ = process_video_for_pose(
            video_path,
            frame_stride=1,
            progress_callback=None,
            camera_position=camera_position,
            active_regions=form_windows,
            visibility_overrides=visibility_overrides,
            pose_backend='mediapipe',
        )
    merged_pose_frames = merge_pose_frames(pose_frames, refined_pose_frames)
    if SHOTLAB_INTERPOLATE_ANGLES:
        merged_pose_frames = interpolate_missing_angles(merged_pose_frames)
    debug.update({
        'applied': True,
        'windows': form_windows,
        'frames_added': int(len(refined_pose_frames or [])),
        'elapsed_seconds': round(float(max(0.0, time.time() - started)), 3),
        'reason': 'ok',
    })
    return merged_pose_frames, debug


def _interpolate_skeleton_keypoints(frames):
    """Fill low-confidence keypoints by interpolating from neighboring frames."""
    if len(frames) < 2:
        return frames
    kp_names = set()
    for f in frames:
        for kp in f.get('keypoints', []):
            kp_names.add(kp['name'])
    min_conf = 0.15
    for name in kp_names:
        values = []
        for i, f in enumerate(frames):
            kp = next((k for k in f.get('keypoints', []) if k['name'] == name), None)
            if kp and kp.get('conf', 0) >= min_conf:
                values.append((i, kp['x'], kp['y'], kp['conf']))
        if len(values) < 2:
            continue
        for i, f in enumerate(frames):
            kp = next((k for k in f.get('keypoints', []) if k['name'] == name), None)
            if kp and kp.get('conf', 0) >= min_conf:
                continue
            before = None
            after = None
            for vi, vx, vy, vc in values:
                if vi < i:
                    before = (vi, vx, vy, vc)
                elif vi > i and after is None:
                    after = (vi, vx, vy, vc)
                    break
            if before is None or after is None:
                continue
            span = after[0] - before[0]
            if span <= 0 or span > 10:
                continue
            t = (i - before[0]) / span
            interp_x = round(before[1] + t * (after[1] - before[1]), 4)
            interp_y = round(before[2] + t * (after[2] - before[2]), 4)
            interp_conf = round(min(before[3], after[3]) * 0.8, 3)
            if kp:
                kp['x'] = interp_x
                kp['y'] = interp_y
                kp['conf'] = interp_conf
            else:
                f['keypoints'].append({'name': name, 'x': interp_x, 'y': interp_y, 'conf': interp_conf})
    return frames


def _extract_skeleton_frames(release_frame, pose_frame_lookup, window=None):
    """Extract skeleton keypoint frames around release for frontend animation."""
    if window is None:
        window = int(BB_SKELETON_WINDOW_FRAMES)
    frame_to_entry, frame_keys = pose_frame_lookup
    if not frame_to_entry or not frame_keys:
        return []
    start = max(0, int(release_frame) - window)
    end = int(release_frame) + window
    skeleton_frames = []
    for fidx in range(start, end + 1):
        entry = frame_to_entry.get(fidx)
        if entry is None:
            continue
        kps = entry.get('raw_keypoints')
        if not kps:
            continue
        skeleton_frames.append({
            'frame_offset': int(fidx - release_frame),
            'keypoints': [dict(kp) for kp in kps],
        })
    skeleton_frames = _interpolate_skeleton_keypoints(skeleton_frames)
    return skeleton_frames


# ---------------------------------------------------------------------------
# Per-shot coaching checks (derived from angle snapshot)
# ---------------------------------------------------------------------------

def _coaching_status(points, max_points):
    """Return 'good', 'needs_work', or 'poor' based on points vs max."""
    if points is None or max_points is None or max_points <= 0:
        return 'unavailable'
    ratio = float(points) / float(max_points)
    if ratio >= 0.70:
        return 'good'
    if ratio >= 0.40:
        return 'needs_work'
    return 'poor'


def _coaching_check(points, max_points, feedback=None):
    status = _coaching_status(points, max_points)
    result = {'status': status, 'points': round(float(points), 1) if points is not None else 0}
    if feedback:
        result['feedback'] = str(feedback)
    return result


def _empty_coaching_check():
    return {'status': 'unavailable', 'points': 0}


def _empty_coaching():
    return {
        'elbow_alignment': _empty_coaching_check(),
        'release_height': _empty_coaching_check(),
        'follow_through': _empty_coaching_check(),
        'base_and_balance': _empty_coaching_check(),
        'shoulder_alignment': _empty_coaching_check(),
        'guide_hand': _empty_coaching_check(),
    }


def _score_elbow_alignment(elbow_angle, max_pts=20):
    """Score elbow alignment. Ideal release elbow angle: ~155-175 degrees."""
    if elbow_angle is None:
        return _empty_coaching_check()
    angle = float(elbow_angle)
    if 155 <= angle <= 175:
        pts = max_pts
        fb = 'Great elbow extension at release.'
    elif 140 <= angle < 155:
        pts = max_pts * 0.75
        fb = 'Elbow nearly extended — try to straighten a bit more at release.'
    elif 175 < angle <= 185:
        pts = max_pts * 0.70
        fb = 'Slight hyperextension — aim for a natural full extension.'
    elif 120 <= angle < 140:
        pts = max_pts * 0.50
        fb = 'Elbow is under-extended. Focus on full arm extension at release.'
    elif 185 < angle <= 195:
        pts = max_pts * 0.50
        fb = 'Noticeable hyperextension. Relax the follow-through.'
    else:
        pts = max_pts * 0.25
        fb = 'Elbow alignment is significantly off. Work on extending through the shot.'
    return _coaching_check(pts, max_pts, fb)


def _score_release_height(shoulder_angle, max_pts=20):
    """Score release height via shooting-arm shoulder angle. Higher is better."""
    if shoulder_angle is None:
        return _empty_coaching_check()
    angle = float(shoulder_angle)
    if 140 <= angle <= 175:
        pts = max_pts
        fb = 'Excellent release height.'
    elif 120 <= angle < 140:
        pts = max_pts * 0.70
        fb = 'Release is a bit low. Try to extend your arm higher.'
    elif 175 < angle <= 190:
        pts = max_pts * 0.70
        fb = 'Release point is very high — good arc but watch for strain.'
    elif 100 <= angle < 120:
        pts = max_pts * 0.45
        fb = 'Release height needs work. Raise your shooting pocket.'
    else:
        pts = max_pts * 0.20
        fb = 'Release is too low. Focus on getting the ball up and out.'
    return _coaching_check(pts, max_pts, fb)


def _score_follow_through(extension, max_pts=20):
    """Score follow-through based on wrist extension (0-1)."""
    if extension is None:
        return _empty_coaching_check()
    ext = float(extension)
    if ext >= 0.70:
        pts = max_pts
        fb = 'Strong follow-through — great wrist snap.'
    elif ext >= 0.50:
        pts = max_pts * 0.70
        fb = 'Decent follow-through. Hold the finish a bit longer.'
    elif ext >= 0.30:
        pts = max_pts * 0.45
        fb = 'Follow-through is short. Extend and snap your wrist fully.'
    else:
        pts = max_pts * 0.20
        fb = 'Very little follow-through detected. Focus on reaching toward the rim.'
    return _coaching_check(pts, max_pts, fb)


def _score_base_and_balance(knee_angles, hip_angles, max_pts=15):
    """Score base and balance from knee/hip angles. Slight bend is ideal."""
    right_knee, left_knee = knee_angles
    right_hip, left_hip = hip_angles
    available = [a for a in (right_knee, left_knee, right_hip, left_hip) if a is not None]
    if not available:
        return _empty_coaching_check()
    knees = [a for a in (right_knee, left_knee) if a is not None]
    if knees:
        avg_knee = float(sum(knees)) / len(knees)
        if 145 <= avg_knee <= 175:
            pts = max_pts
            fb = 'Good base — athletic stance with slight knee bend.'
        elif 130 <= avg_knee < 145:
            pts = max_pts * 0.70
            fb = 'A bit deep in the legs. Try a slightly taller stance.'
        elif 175 < avg_knee <= 185:
            pts = max_pts * 0.65
            fb = 'Legs are too straight. Bend your knees slightly for balance.'
        else:
            pts = max_pts * 0.35
            fb = 'Stance needs improvement. Focus on an athletic base.'
    else:
        hips = [a for a in (right_hip, left_hip) if a is not None]
        avg_hip = float(sum(hips)) / len(hips)
        if 160 <= avg_hip <= 180:
            pts = max_pts * 0.80
            fb = 'Hip posture looks balanced.'
        else:
            pts = max_pts * 0.45
            fb = 'Hip alignment suggests uneven balance.'
    return _coaching_check(pts, max_pts, fb)


def _score_shoulder_alignment(shoulder_angles, max_pts=15):
    """Score shoulder alignment. Shoulders should be fairly level/square."""
    right_shoulder, left_shoulder = shoulder_angles
    if right_shoulder is None or left_shoulder is None:
        if right_shoulder is not None or left_shoulder is not None:
            return _coaching_check(max_pts * 0.55, max_pts, 'Only one shoulder visible — cannot fully assess alignment.')
        return _empty_coaching_check()
    diff = abs(float(right_shoulder) - float(left_shoulder))
    if diff <= 15:
        pts = max_pts
        fb = 'Shoulders are square to the basket — great alignment.'
    elif diff <= 30:
        pts = max_pts * 0.70
        fb = 'Slight shoulder tilt. Try to keep shoulders more level.'
    elif diff <= 50:
        pts = max_pts * 0.45
        fb = 'Shoulders are noticeably tilted. Work on squaring up to the basket.'
    else:
        pts = max_pts * 0.25
        fb = 'Significant shoulder imbalance. Face the basket more directly.'
    return _coaching_check(pts, max_pts, fb)


def _score_guide_hand(guide_elbow_angle, guide_wrist_angle, max_pts=10):
    """Score guide hand. It should be relaxed and supportive, not pushing."""
    if guide_elbow_angle is None:
        return _empty_coaching_check()
    angle = float(guide_elbow_angle)
    if 70 <= angle <= 130:
        pts = max_pts
        fb = 'Guide hand is well positioned — steady and supportive.'
    elif 50 <= angle < 70 or 130 < angle <= 150:
        pts = max_pts * 0.65
        fb = 'Guide hand is slightly off. Keep it relaxed and on the side of the ball.'
    else:
        pts = max_pts * 0.30
        fb = 'Guide hand may be interfering with the shot. Keep it as a support only.'
    return _coaching_check(pts, max_pts, fb)


def _compute_coaching_for_shot(form_visibility_debug, form_score):
    """Derive per-check coaching data from angle snapshot for a single shot."""
    angles = None
    if isinstance(form_visibility_debug, dict):
        angles = form_visibility_debug.get('selected_angles')
    if not angles or not isinstance(angles, dict):
        return _empty_coaching()

    shooting_arm = str(angles.get('shooting_arm') or 'right')
    guide_arm = 'left' if shooting_arm == 'right' else 'right'

    elbow_check = _score_elbow_alignment(angles.get(f'{shooting_arm}_elbow_angle'))
    release_check = _score_release_height(angles.get(f'{shooting_arm}_shoulder_angle'))
    ft_check = _score_follow_through(angles.get('follow_through_extension'))
    base_check = _score_base_and_balance(
        (angles.get('right_knee_angle'), angles.get('left_knee_angle')),
        (angles.get('right_hip_angle'), angles.get('left_hip_angle')),
    )
    shoulder_check = _score_shoulder_alignment(
        (angles.get('right_shoulder_angle'), angles.get('left_shoulder_angle')),
    )
    guide_check = _score_guide_hand(
        angles.get(f'{guide_arm}_elbow_angle'),
        angles.get(f'{guide_arm}_wrist_angle'),
    )

    return {
        'elbow_alignment': elbow_check,
        'release_height': release_check,
        'follow_through': ft_check,
        'base_and_balance': base_check,
        'shoulder_alignment': shoulder_check,
        'guide_hand': guide_check,
    }


def _rotate_point(cx, cy, px, py, angle_delta_rad):
    """Rotate point (px,py) around (cx,cy) by angle_delta_rad."""
    cos_a = math.cos(angle_delta_rad)
    sin_a = math.sin(angle_delta_rad)
    dx = px - cx
    dy = py - cy
    return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)


def _compute_limb_angle(ax, ay, bx, by, cx, cy):
    """Compute angle at joint B formed by segments A-B and B-C, in degrees."""
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    m2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if m1 < 1e-9 or m2 < 1e-9:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (m1 * m2)))
    return math.degrees(math.acos(cos_angle))


_IDEAL_ANGLE_RANGES = {
    'elbow_alignment': (155.0, 175.0),
    'release_height': (140.0, 175.0),
    'base_and_balance': (145.0, 175.0),
    'guide_hand': (70.0, 130.0),
}

_STILL_VECTOR_DEFS = {
    'release': [
        {
            'check': 'elbow_alignment',
            'joints': ('shoulder', 'elbow', 'wrist'),
            'arm': 'shooting',
        },
        {
            'check': 'release_height',
            'joints': ('hip', 'shoulder', 'elbow'),
            'arm': 'shooting',
        },
        {
            'check': 'guide_hand',
            'joints': ('shoulder', 'elbow', 'wrist'),
            'arm': 'guide',
        },
    ],
    'setup': [
        {
            'check': 'base_and_balance',
            'joints': ('hip', 'knee', 'ankle'),
            'arm': 'right',
        },
    ],
    'follow_through': [
        {
            'check': 'follow_through',
            'joints': ('shoulder', 'elbow', 'wrist'),
            'arm': 'shooting',
        },
    ],
}


def _detect_shot_moments(skeleton_frames, shooting_arm):
    """Find setup, release, follow-through frames by analyzing wrist trajectory.

    Only considers frames where the upper body pose is well-detected
    (shoulder + elbow + wrist all present with decent confidence) to
    avoid picking frames where the pose model tracked the hoop or background.
    """
    wrist_key = f"{shooting_arm}_wrist"
    elbow_key = f"{shooting_arm}_elbow"
    shoulder_key = f"{shooting_arm}_shoulder"
    required_keys = [wrist_key, elbow_key, shoulder_key]
    MIN_CONF = 0.70

    trajectory = []
    for sf in sorted(skeleton_frames, key=lambda f: f['frame_offset']):
        kps = {kp['name']: kp for kp in sf.get('keypoints', [])}
        if not all(kps.get(k) and kps[k].get('conf', 0) >= MIN_CONF for k in required_keys):
            continue
        wrist = kps[wrist_key]
        shoulder = kps[shoulder_key]
        wrist_y = wrist['y']
        shoulder_y = shoulder['y']
        if wrist_y < shoulder_y * 0.3:
            continue
        trajectory.append((sf['frame_offset'], wrist_y))

    if len(trajectory) < 5:
        return None, None, None

    y_vals = [t[1] for t in trajectory]
    y_range = max(y_vals) - min(y_vals)
    if y_range < 0.03:
        return None, None, None

    near_release = [t for t in trajectory if -8 <= t[0] <= 8]
    if near_release:
        release = min(near_release, key=lambda t: t[1])
    else:
        release = min(trajectory, key=lambda t: t[1])

    before = [t for t in trajectory if t[0] < release[0]]
    setup = max(before, key=lambda t: t[1]) if before else None
    if setup and release[0] - setup[0] < 4:
        earlier = [t for t in before if t[0] <= release[0] - 4]
        setup = max(earlier, key=lambda t: t[1]) if earlier else setup

    after = [t for t in trajectory if release[0] + 3 <= t[0] <= release[0] + 12]
    follow = after[len(after) // 2] if after else None

    return (
        setup[0] if setup else None,
        release[0],
        follow[0] if follow else None,
    )


def _select_best_release_still_offset(release_frame, frames_by_offset, yolo_detections, frame_shape):
    if not frames_by_offset or not yolo_detections or frame_shape is None:
        return None
    best = None
    for offset in range(-5, 6):
        if int(offset) not in frames_by_offset:
            continue
        abs_frame = int(release_frame) + int(offset)
        candidate = _select_person_candidate_from_yolo(
            yolo_detections,
            abs_frame,
            frame_shape,
            behind_basket_mode=True,
            max_gap_frames=0,
        )
        if not candidate:
            continue
        confidence = float(candidate.get('confidence', 0.0) or 0.0)
        rank = (confidence, -abs(int(offset)))
        if best is None or rank > best[0]:
            best = (rank, int(offset))
    if best is None:
        return None
    return int(best[1])


def _build_annotated_stills(
    release_frame,
    skeleton_frames,
    coaching,
    session_id,
    shot_index,
    fps=30.0,
    yolo_detections=None,
    frame_shape=None,
    behind_basket_mode=False,
):
    """Build annotated still descriptors with form vectors for frontend rendering."""
    if not skeleton_frames or not coaching:
        return [], None

    frames_by_offset = {}
    for sf in skeleton_frames:
        frames_by_offset[int(sf['frame_offset'])] = sf

    shooting_arm = 'right'
    guide_arm = 'left'
    for sf in skeleton_frames:
        kps = {kp['name']: kp for kp in sf.get('keypoints', [])}
        if 'left_wrist' in kps and 'right_wrist' in kps:
            if kps['left_wrist'].get('y', 1) < kps['right_wrist'].get('y', 1):
                shooting_arm = 'left'
                guide_arm = 'right'
            break

    setup_offset, release_offset, ft_offset = _detect_shot_moments(skeleton_frames, shooting_arm)
    if setup_offset is None:
        setup_offset = next((o for o in [-10, -9, -8, -11, -7, -12, -6] if o in frames_by_offset), None)
    if release_offset is None:
        release_offset = next((o for o in [0, -1, 1] if o in frames_by_offset), None)
    if behind_basket_mode:
        best_release_offset = _select_best_release_still_offset(
            release_frame,
            frames_by_offset,
            yolo_detections,
            frame_shape,
        )
        if best_release_offset is None:
            return [], "Shooter not clearly visible"
        release_offset = int(best_release_offset)
    if ft_offset is None:
        ft_offset = next((o for o in [6, 7, 5, 8, 4, 9] if o in frames_by_offset), None)

    setup_frame = frames_by_offset.get(setup_offset)
    release_frame_data = frames_by_offset.get(release_offset)
    ft_frame = frames_by_offset.get(ft_offset)

    stills = []
    moment_map = [
        ('Setup', setup_offset, setup_frame, 'setup'),
        ('Release', release_offset, release_frame_data, 'release'),
        ('Follow-Through', ft_offset, ft_frame, 'follow_through'),
    ]

    for label, offset, frame_data, moment_key in moment_map:
        if frame_data is None:
            continue

        abs_frame = int(release_frame) + int(offset)
        kps = {kp['name']: kp for kp in frame_data.get('keypoints', [])}

        vectors = []
        vec_defs = _STILL_VECTOR_DEFS.get(moment_key, [])
        for vdef in vec_defs:
            check_key = vdef['check']
            check_data = coaching.get(check_key)
            if not check_data or not isinstance(check_data, dict):
                continue
            status = str(check_data.get('status', 'unavailable')).lower()
            if status == 'unavailable':
                continue

            arm = vdef['arm']
            if arm == 'shooting':
                arm = shooting_arm
            elif arm == 'guide':
                arm = guide_arm

            joint_a_name = f"{arm}_{vdef['joints'][0]}"
            joint_b_name = f"{arm}_{vdef['joints'][1]}"
            joint_c_name = f"{arm}_{vdef['joints'][2]}"

            kp_a = kps.get(joint_a_name)
            kp_b = kps.get(joint_b_name)
            kp_c = kps.get(joint_c_name)

            conf_min = 0.10
            if not kp_a or not kp_b or not kp_c:
                continue
            if kp_a.get('conf', 0) < conf_min or kp_b.get('conf', 0) < conf_min or kp_c.get('conf', 0) < conf_min:
                continue

            actual_angle = _compute_limb_angle(
                kp_a['x'], kp_a['y'], kp_b['x'], kp_b['y'], kp_c['x'], kp_c['y']
            )

            ideal_range_val = _IDEAL_ANGLE_RANGES.get(check_key)
            check_label = check_key.replace('_', ' ').title()

            vectors.append({
                'check': check_key,
                'check_label': check_label,
                'status': status,
                'joint_dot': {
                    'x': round(float(kp_b['x']), 4),
                    'y': round(float(kp_b['y']), 4),
                },
                'actual_angle': round(float(actual_angle), 1) if actual_angle is not None else None,
                'ideal_range': list(ideal_range_val) if ideal_range_val else None,
            })

        # --- Compute person bounding box from keypoints for cropping ---
        xs, ys = [], []
        for kp in frame_data.get('keypoints', []):
            if kp.get('conf', 0) >= 0.15:
                xs.append(float(kp['x']))
                ys.append(float(kp['y']))

        crop = None
        if xs and ys:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            pad_x = (max_x - min_x) * 0.35
            pad_y = (max_y - min_y) * 0.35
            cx = max(0.0, min_x - pad_x)
            cy = max(0.0, min_y - pad_y)
            cw = min(1.0, max_x + pad_x) - cx
            ch = min(1.0, max_y + pad_y) - cy
            cw = max(cw, 0.05)
            ch = max(ch, 0.05)
            crop = {'x': round(cx, 4), 'y': round(cy, 4),
                    'w': round(cw, 4), 'h': round(ch, 4)}

        # --- Renormalize joint dot coordinates to crop region ---
        if crop:
            for vec in vectors:
                dot = vec.get('joint_dot')
                if dot:
                    dot['x'] = round((dot['x'] - crop['x']) / crop['w'], 4)
                    dot['y'] = round((dot['y'] - crop['y']) / crop['h'], 4)

        # --- Build frame URL with crop params ---
        frame_url = f'/api/shotlab_frame?session_id={session_id}&frame_idx={abs_frame}'
        if crop:
            frame_url += f'&cx={crop["x"]}&cy={crop["y"]}&cw={crop["w"]}&ch={crop["h"]}'

        stills.append({
            'label': label,
            'frame_idx': abs_frame,
            'frame_url': frame_url,
            'crop': crop,
            'vectors': vectors,
            'has_issues': any(v['status'] in ('poor', 'needs_work') for v in vectors),
            'all_good': all(v['status'] == 'good' for v in vectors) and len(vectors) > 0,
        })

    return stills, None


def _distance_2d(a, b):
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _build_pose_frame_lookup(pose_frames):
    frame_to_entry = {}
    frame_keys = []
    for entry in pose_frames or []:
        frame_idx = int(entry.get('frame_idx', -1))
        if frame_idx < 0:
            continue
        frame_to_entry[frame_idx] = entry
        frame_keys.append(frame_idx)
    frame_keys = sorted(set(frame_keys))
    return frame_to_entry, frame_keys


def _get_pose_entry_for_frame(frame_idx, frame_to_entry, frame_keys, max_gap_frames):
    frame_idx = int(frame_idx)
    direct = frame_to_entry.get(frame_idx)
    if direct is not None:
        return direct
    if not frame_keys:
        return None
    pos = bisect.bisect_left(frame_keys, frame_idx)
    candidates = []
    if pos < len(frame_keys):
        candidates.append(frame_keys[pos])
    if pos > 0:
        candidates.append(frame_keys[pos - 1])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda f: abs(f - frame_idx))
    if abs(nearest - frame_idx) <= int(max_gap_frames):
        return frame_to_entry.get(nearest)
    return None


def _get_wrist_position_for_frame(frame_idx, frame_to_entry, frame_keys, max_gap_frames):
    entry = _get_pose_entry_for_frame(frame_idx, frame_to_entry, frame_keys, max_gap_frames)
    if not entry:
        return None
    wrists = []
    right = entry.get('right_wrist')
    left = entry.get('left_wrist')
    if right is not None:
        wrists.append((float(right[0]), float(right[1])))
    if left is not None:
        wrists.append((float(left[0]), float(left[1])))
    if not wrists:
        return None
    if len(wrists) == 1:
        return wrists[0]
    # Use midpoint between wrists when both are available.
    return ((wrists[0][0] + wrists[1][0]) / 2.0, (wrists[0][1] + wrists[1][1]) / 2.0)


def _get_shooter_reference_points_for_frame(frame_idx, frame_to_entry, frame_keys, max_gap_frames):
    entry = _get_pose_entry_for_frame(frame_idx, frame_to_entry, frame_keys, max_gap_frames)
    if not entry:
        return []

    points = []
    for key in (
        'right_wrist', 'left_wrist',
        'right_elbow', 'left_elbow',
        'right_shoulder', 'left_shoulder'
    ):
        pt = entry.get(key)
        if pt is not None:
            points.append((float(pt[0]), float(pt[1])))

    for right_key, left_key in (
        ('right_wrist', 'left_wrist'),
        ('right_elbow', 'left_elbow'),
        ('right_shoulder', 'left_shoulder')
    ):
        right = entry.get(right_key)
        left = entry.get(left_key)
        if right is not None and left is not None:
            points.append((
                (float(right[0]) + float(left[0])) / 2.0,
                (float(right[1]) + float(left[1])) / 2.0
            ))

    return points


def detect_ball_launches(yolo_detections, frame_height, min_launch_frames=None, fps=None):
    """
    Detect likely release frames from YOLO ball tracks when pose misses shots.
    Returns: sorted list of release frame indices.
    """
    if not yolo_detections:
        return []

    min_frames = int(min_launch_frames or SHOTLAB_YOLO_LAUNCH_MIN_TRACK_FRAMES)
    min_frames = max(3, min_frames)
    conf_threshold = float(SHOTLAB_YOLO_LAUNCH_MIN_CONFIDENCE)
    upward_delta = float(SHOTLAB_YOLO_LAUNCH_UPWARD_DELTA_PX)
    shrink_ratio = float(SHOTLAB_YOLO_LAUNCH_SHRINK_RATIO)
    min_upward = max(1, int(SHOTLAB_YOLO_LAUNCH_MIN_UPWARD_COUNT))
    min_shrink = max(1, int(SHOTLAB_YOLO_LAUNCH_MIN_SHRINK_COUNT))
    window = max(min_frames, int(SHOTLAB_YOLO_LAUNCH_WINDOW_FRAMES))
    cooldown_frames = max(1, int((fps or 30.0) * SHOTLAB_YOLO_LAUNCH_COOLDOWN_SECONDS))

    ball_frames = []
    for fn in sorted(int(k) for k in yolo_detections.keys()):
        det = yolo_detections.get(fn, {})
        balls = det.get('balls', [])
        if not balls:
            continue
        best = max(balls, key=lambda b: float(b.get('confidence', 0.0) or 0.0))
        if float(best.get('confidence', 0.0) or 0.0) < conf_threshold:
            continue
        center = best.get('center') or (None, None)
        if center[0] is None or center[1] is None:
            continue
        area = float(best.get('area', 0.0) or 0.0)
        ball_frames.append((fn, float(center[1]), area))

    if len(ball_frames) < min_frames:
        return []

    launches = []
    i = 0
    while i < len(ball_frames) - min_frames:
        upper = min(i + window, len(ball_frames) - 1)
        upward_count = 0
        shrink_count = 0
        for j in range(i, upper):
            curr_y = ball_frames[j][1]
            next_y = ball_frames[j + 1][1]
            curr_area = max(1.0, ball_frames[j][2])
            next_area = max(1.0, ball_frames[j + 1][2])
            if next_y < curr_y - upward_delta:
                upward_count += 1
            if next_area < curr_area * shrink_ratio:
                shrink_count += 1

        if upward_count >= min_upward or shrink_count >= min_shrink:
            release_frame = int(ball_frames[i][0])
            launches.append(release_frame)
            cooldown_until = release_frame + cooldown_frames
            while i < len(ball_frames) and int(ball_frames[i][0]) <= cooldown_until:
                i += 1
            continue
        i += 1

    return sorted(set(launches))


def merge_shot_candidates_with_yolo_launches(shot_attempts, yolo_launches, fps, total_frames):
    """
    Add YOLO launch-based candidates when no pose candidate exists nearby.
    """
    pose_shots = []
    for shot in shot_attempts or []:
        tagged = dict(shot)
        tagged['source'] = str(tagged.get('source') or 'pose')
        pose_shots.append(tagged)
    launch_frames = sorted(int(f) for f in (yolo_launches or []))
    debug = {
        'pose_candidates': len(pose_shots),
        'launch_candidates': len(launch_frames),
        'added': 0,
        'skipped_near_pose': 0,
        'skipped_startup_guard': 0,
        'skipped_cap': 0
    }
    if not launch_frames:
        return pose_shots, debug

    fps_safe = float(fps or 30.0)
    merge_gap_frames = max(1, int(SHOTLAB_YOLO_LAUNCH_MERGE_GAP_FRAMES))
    startup_guard_frame = max(0, int(SHOTLAB_YOLO_LAUNCH_MIN_START_FRAME))
    max_added = max(2, int(math.ceil(0.25 * float(len(pose_shots)))))
    lookback_frames = max(1, int(0.45 * fps_safe))
    clip_len_frames = max(1, int(1.3 * fps_safe))
    max_frame = max(0, int(total_frames) - 1) if total_frames else None

    releases = [
        int(s.get('release_frame', s.get('start_frame', 0)))
        for s in pose_shots
    ]
    merged = list(pose_shots)
    for launch_frame in launch_frames:
        if launch_frame < startup_guard_frame:
            debug['skipped_startup_guard'] += 1
            continue
        if debug['added'] >= max_added:
            debug['skipped_cap'] += 1
            continue
        if any(abs(launch_frame - rf) < merge_gap_frames for rf in releases):
            debug['skipped_near_pose'] += 1
            continue

        start_frame = max(0, launch_frame - lookback_frames)
        end_frame = launch_frame + clip_len_frames
        if max_frame is not None:
            end_frame = min(end_frame, max_frame)
        merged.append({
            'start_frame': int(start_frame),
            'release_frame': int(launch_frame),
            'end_frame': int(max(end_frame, launch_frame + 1)),
            'timestamp': float(launch_frame / fps_safe),
            'source': 'yolo_launch'
        })
        releases.append(launch_frame)
        debug['added'] += 1
        logger.info("YOLO launch candidate added at frame %s", launch_frame)

    debug['startup_guard_frame'] = int(startup_guard_frame)
    debug['max_added'] = int(max_added)
    debug['output_candidates'] = len(merged)
    return merged, debug


def apply_yolo_launch_fallback_candidates(
    shot_attempts,
    yolo_detections,
    frame_height,
    fps,
    total_frames,
    shot_detection_debug,
    pose_debug
):
    yolo_launches = []
    yolo_launch_debug = {
        'pose_candidates': len(shot_attempts or []),
        'launch_candidates': 0,
        'added': 0,
        'skipped_near_pose': 0,
        'output_candidates': len(shot_attempts or [])
    }
    if SHOTLAB_YOLO_LAUNCH_ENABLE and yolo_detections:
        yolo_launches = detect_ball_launches(
            yolo_detections,
            frame_height=frame_height,
            min_launch_frames=SHOTLAB_YOLO_LAUNCH_MIN_TRACK_FRAMES,
            fps=fps
        )
        shot_attempts, yolo_launch_debug = merge_shot_candidates_with_yolo_launches(
            shot_attempts,
            yolo_launches,
            fps=fps,
            total_frames=total_frames
        )
        shot_detection_debug['yolo_launch'] = {
            **yolo_launch_debug,
            'detected_launches': yolo_launches
        }
        pose_debug['shot_attempts_after_yolo_launch_merge'] = len(shot_attempts)
        logger.info(
            "YOLO launch fallback: launches=%d added=%d skipped_near_pose=%d output_candidates=%d",
            len(yolo_launches),
            yolo_launch_debug.get('added', 0),
            yolo_launch_debug.get('skipped_near_pose', 0),
            yolo_launch_debug.get('output_candidates', len(shot_attempts))
        )
    return shot_attempts, yolo_launches, yolo_launch_debug


def _release_frame_for_candidate(shot):
    return int(shot.get('release_frame', shot.get('start_frame', 0)))


def _read_video_frames_sparse(video_path, frame_indices):
    unique_indices = sorted(set(int(f) for f in (frame_indices or []) if int(f) >= 0))
    if not unique_indices:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    frames_by_idx = {}
    try:
        min_target = int(unique_indices[0])
        max_target = int(unique_indices[-1])
        if min_target > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min_target)
        frame_idx = min_target
        target_pos = 0
        next_target = int(unique_indices[target_pos])
        while frame_idx <= max_target:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx == next_target:
                frames_by_idx[frame_idx] = frame.copy()
                target_pos += 1
                if target_pos >= len(unique_indices):
                    break
                next_target = int(unique_indices[target_pos])
            frame_idx += 1
    finally:
        cap.release()

    return frames_by_idx


def _best_class_confidence_from_result(result, class_id):
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None
    try:
        conf_arr = result.boxes.conf.cpu().numpy()
    except Exception:
        return None
    if conf_arr is None or len(conf_arr) == 0:
        return None
    cls_arr = None
    if getattr(result.boxes, 'cls', None) is not None:
        try:
            cls_arr = result.boxes.cls.cpu().numpy().astype(np.int32)
        except Exception:
            cls_arr = None
    best = 0.0
    for idx, conf in enumerate(conf_arr):
        cls_id = int(cls_arr[idx]) if cls_arr is not None and idx < len(cls_arr) else int(class_id)
        if cls_id != int(class_id):
            continue
        conf_f = float(conf)
        if conf_f > best:
            best = conf_f
    return best if best > 0.0 else None


def _best_shoot_confidence_from_result(result):
    return _best_class_confidence_from_result(result, SHOTLAB_EVENT_SHOOT_CLASS_ID)


def detect_event_class_on_frames(
    video_path,
    frame_indices,
    event_class_id,
    min_confidence,
):
    debug = {
        'enabled': bool(SHOTLAB_EVENT_MODEL_ENABLE),
        'event_class_id': int(event_class_id),
        'requested_frames': 0,
        'frames_loaded': 0,
        'frames_inferred': 0,
        'detections': 0,
        'errors': 0,
        'elapsed_seconds': 0.0,
    }
    started = time.time()
    unique_indices = sorted(set(int(f) for f in (frame_indices or []) if int(f) >= 0))
    debug['requested_frames'] = len(unique_indices)
    if not unique_indices:
        debug['elapsed_seconds'] = float(max(0.0, time.time() - started))
        return {}, debug

    detector = get_detection_model()
    event_model = getattr(detector, 'event_model', None)
    if not SHOTLAB_EVENT_MODEL_ENABLE:
        debug['reason'] = 'disabled'
        debug['elapsed_seconds'] = float(max(0.0, time.time() - started))
        return {}, debug
    if event_model is None:
        debug['reason'] = 'model_unavailable'
        debug['elapsed_seconds'] = float(max(0.0, time.time() - started))
        return {}, debug

    frames_by_idx = _read_video_frames_sparse(video_path, unique_indices)
    if not frames_by_idx:
        debug['reason'] = 'no_frames_loaded'
        debug['elapsed_seconds'] = float(max(0.0, time.time() - started))
        return {}, debug

    ordered = sorted(frames_by_idx.keys())
    debug['frames_loaded'] = len(ordered)
    results_by_frame = {}
    batch_size = max(1, int(SHOTLAB_EVENT_BATCH_SIZE))
    conf_threshold = max(0.01, float(min_confidence))
    for start in range(0, len(ordered), batch_size):
        batch_frames = ordered[start:start + batch_size]
        batch_images = [frames_by_idx[f] for f in batch_frames]
        try:
            predictions = event_model.predict(
                batch_images,
                imgsz=Config.BALL_IMG_SIZE,
                conf=conf_threshold,
                iou=Config.BALL_IOU,
                classes=[int(event_class_id)],
                device=detector.device,
                verbose=False
            )
        except Exception as exc:
            debug['errors'] += 1
            logger.warning("Shot event model inference failed on batch starting frame %s: %s", batch_frames[0], exc)
            continue

        debug['frames_inferred'] += len(batch_frames)
        for frame_idx, result in zip(batch_frames, predictions or []):
            best_conf = _best_class_confidence_from_result(result, event_class_id)
            if best_conf is None or best_conf < conf_threshold:
                continue
            results_by_frame[frame_idx] = max(float(best_conf), float(results_by_frame.get(frame_idx, 0.0)))

    debug['detections'] = len(results_by_frame)
    debug['elapsed_seconds'] = float(max(0.0, time.time() - started))
    return results_by_frame, debug


def detect_shoot_events_on_frames(video_path, frame_indices, min_confidence=SHOTLAB_EVENT_CANDIDATE_CONFIDENCE):
    return detect_event_class_on_frames(
        video_path=video_path,
        frame_indices=frame_indices,
        event_class_id=SHOTLAB_EVENT_SHOOT_CLASS_ID,
        min_confidence=min_confidence,
    )


def _is_frame_in_active_regions(frame_idx, active_regions):
    for region in active_regions or []:
        if isinstance(region, (list, tuple)) and len(region) >= 2:
            start = int(region[0])
            end = int(region[1])
        elif isinstance(region, dict):
            start = int(region.get('start_frame', region.get('start', 0)))
            end = int(region.get('end_frame', region.get('end', 0)))
        else:
            continue
        if start <= int(frame_idx) <= end:
            return True
    return False


def _collect_gap_sample_frames(candidate_release_frames, total_frames, active_regions=None, sample_stride=None):
    ordered = sorted(set(int(f) for f in (candidate_release_frames or []) if int(f) >= 0))
    if len(ordered) < 2:
        return []

    gap_min = max(1, int(SHOTLAB_EVENT_GAP_MIN_FRAMES))
    stride = max(1, int(sample_stride if sample_stride is not None else SHOTLAB_EVENT_GAP_SAMPLE_STRIDE))
    max_frame = max(0, int(total_frames) - 1) if total_frames else None
    sampled = []
    for prev_frame, next_frame in zip(ordered, ordered[1:]):
        gap_len = int(next_frame - prev_frame)
        if gap_len <= gap_min:
            continue
        start = prev_frame + stride
        end = next_frame - 1
        if max_frame is not None:
            start = min(max_frame, max(0, start))
            end = min(max_frame, max(0, end))
        if end <= start:
            continue
        mid = int((prev_frame + next_frame) * 0.5)
        for frame_idx in range(start, end + 1, stride):
            if prev_frame < frame_idx < next_frame:
                in_active = _is_frame_in_active_regions(frame_idx, active_regions)
                sampled.append({
                    'frame_idx': int(frame_idx),
                    'gap_len': int(gap_len),
                    'in_active_region': bool(in_active),
                    'center_dist': abs(int(frame_idx) - mid),
                })

    sampled.sort(
        key=lambda item: (
            -int(item.get('in_active_region', False)),
            -int(item.get('gap_len', 0)),
            int(item.get('center_dist', 0)),
            int(item.get('frame_idx', 0)),
        )
    )
    ordered_unique = []
    seen = set()
    for item in sampled:
        frame_idx = int(item.get('frame_idx', -1))
        if frame_idx < 0 or frame_idx in seen:
            continue
        seen.add(frame_idx)
        ordered_unique.append(frame_idx)
    return ordered_unique


def _event_thresholds_for_camera(camera_position, event_angle_override=None):
    override_angle = _normalize_event_camera_angle_override(event_angle_override)
    angle = override_angle or _camera_position_to_event_angle(camera_position)
    candidate_conf = float(SHOTLAB_EVENT_CANDIDATE_CONF_BY_ANGLE.get(angle, SHOTLAB_EVENT_CANDIDATE_CONFIDENCE))
    gap_conf = float(SHOTLAB_EVENT_GAP_CONF_BY_ANGLE.get(angle, SHOTLAB_EVENT_GAP_CONFIDENCE))
    gap_enabled = bool(SHOTLAB_EVENT_ENABLE_GAP_FILL_BY_ANGLE.get(angle, True))
    return {
        'angle': angle,
        'angle_override': override_angle,
        'candidate_confidence': max(0.01, min(0.99, candidate_conf)),
        'gap_confidence': max(0.01, min(0.99, gap_conf)),
        'gap_fill_enabled': gap_enabled,
    }


def _candidate_priority_from_source(source):
    source_s = str(source or 'pose')
    if source_s == 'yolo_launch':
        return 3.0
    if source_s == 'pose':
        return 2.0
    return 1.0


def _select_temporally_spread_frames(frames, limit):
    ordered = sorted(set(int(f) for f in (frames or []) if int(f) >= 0))
    if limit <= 0:
        return []
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]

    selected = []
    used = set()
    max_idx = len(ordered) - 1
    for i in range(limit):
        idx = int(round((i * max_idx) / float(limit - 1)))
        idx = max(0, min(max_idx, idx))
        frame_idx = ordered[idx]
        if frame_idx in used:
            continue
        selected.append(frame_idx)
        used.add(frame_idx)

    if len(selected) < limit:
        for frame_idx in ordered:
            if frame_idx in used:
                continue
            selected.append(frame_idx)
            used.add(frame_idx)
            if len(selected) >= limit:
                break

    return sorted(selected[:limit])


def apply_shoot_model_candidates(
    video_path,
    shot_attempts,
    yolo_launches,
    fps,
    total_frames,
    camera_position='unknown',
    camera_mode='auto',
    event_angle_override=None,
    active_regions=None,
    shot_detection_debug=None,
    pose_debug=None
):
    shots = [dict(s) for s in (shot_attempts or [])]
    stage_started = time.time()
    thresholds = _event_thresholds_for_camera(camera_position, event_angle_override=event_angle_override)
    behind_basket_mode = _is_behind_basket_mode(camera_mode)
    if behind_basket_mode:
        thresholds['candidate_confidence'] = min(float(thresholds['candidate_confidence']), 0.30)
        thresholds['gap_confidence'] = min(float(thresholds['gap_confidence']), 0.40)
    fps_safe = max(1.0, float(fps or 30.0))
    video_seconds = float(total_frames) / fps_safe if total_frames else None
    duration_budget_frames = None
    if video_seconds is not None:
        duration_budget_frames = int(max(24, min(64, round(video_seconds * 0.8))))
    max_total_frames = int(SHOTLAB_EVENT_MAX_FRAMES_PER_VIDEO)
    if duration_budget_frames is not None:
        max_total_frames = (
            int(duration_budget_frames)
            if max_total_frames <= 0
            else int(min(max_total_frames, duration_budget_frames))
        )
    dynamic_candidate_frames = int(round(max_total_frames * 0.75)) if max_total_frames > 0 else 0
    dynamic_gap_frames = int(max(0, max_total_frames - dynamic_candidate_frames)) if max_total_frames > 0 else 0
    max_candidate_frames = int(SHOTLAB_EVENT_MAX_CANDIDATE_FRAMES_PER_VIDEO)
    if dynamic_candidate_frames > 0:
        max_candidate_frames = (
            int(dynamic_candidate_frames)
            if max_candidate_frames <= 0
            else int(min(max_candidate_frames, dynamic_candidate_frames))
        )
    max_gap_frames = int(SHOTLAB_EVENT_MAX_GAP_FRAMES_PER_VIDEO)
    if dynamic_gap_frames > 0:
        max_gap_frames = (
            int(dynamic_gap_frames)
            if max_gap_frames <= 0
            else int(min(max_gap_frames, dynamic_gap_frames))
        )
    max_event_seconds = float(max(0.0, SHOTLAB_EVENT_MAX_SECONDS_PER_VIDEO))
    neighborhood_frames = max(0, int(SHOTLAB_EVENT_CANDIDATE_NEIGHBORHOOD_FRAMES))
    neighborhood_frames = min(neighborhood_frames, 2)
    neighborhood_radius = max(1, int(SHOTLAB_EVENT_CANDIDATE_NEIGHBOR_RADIUS_FRAMES))
    neighborhood_radius = min(neighborhood_radius, 6)
    budget_unlimited = max_total_frames <= 0
    candidate_budget_unlimited = max_candidate_frames <= 0
    stage_budget_unlimited = max_event_seconds <= 0.0
    if not budget_unlimited and not candidate_budget_unlimited:
        max_candidate_frames = min(max_candidate_frames, max_total_frames)

    if behind_basket_mode:
        # Recall-first behind-basket mode needs broader timeline coverage so late shots
        # are not starved by candidate budgets.
        if video_seconds is not None:
            bb_budget_frames = int(max(96, min(160, round(video_seconds * 2.0))))
        else:
            bb_budget_frames = 120
        if max_total_frames <= 0:
            max_total_frames = bb_budget_frames
        else:
            max_total_frames = max(int(max_total_frames), bb_budget_frames)

        bb_candidate_target = int(max(64, round(max_total_frames * 0.65)))
        bb_gap_target = int(max(24, max_total_frames - bb_candidate_target))
        max_candidate_frames = max(int(max_candidate_frames), bb_candidate_target)
        max_gap_frames = max(int(max_gap_frames), bb_gap_target)
        max_event_seconds = max(float(max_event_seconds), 24.0)

    debug = {
        'enabled': bool(SHOTLAB_EVENT_MODEL_ENABLE),
        'camera_angle': thresholds['angle'],
        'camera_mode': _normalize_camera_mode(camera_mode),
        'event_angle_override': thresholds.get('angle_override'),
        'candidate_confidence_threshold': float(thresholds['candidate_confidence']),
        'gap_confidence_threshold': float(thresholds['gap_confidence']),
        'gap_fill_enabled': bool(thresholds['gap_fill_enabled']),
        'video_seconds': float(video_seconds) if video_seconds is not None else None,
        'duration_budget_frames': int(duration_budget_frames) if duration_budget_frames is not None else None,
        'max_frames_per_video': int(max_total_frames),
        'max_candidate_frames_per_video': int(max_candidate_frames),
        'max_gap_frames_per_video': int(max_gap_frames),
        'frame_budget_unlimited': bool(budget_unlimited),
        'candidate_frame_budget_unlimited': bool(candidate_budget_unlimited),
        'max_event_stage_seconds': float(max_event_seconds),
        'time_budget_unlimited': bool(stage_budget_unlimited),
        'input_candidates': len(shots),
        'candidate_core_requested': 0,
        'candidate_core_selected': 0,
        'candidate_neighbor_selected': 0,
        'candidate_core_dropped_by_budget': 0,
        'candidate_neighbor_dropped_by_budget': 0,
        'candidate_frames_checked': 0,
        'candidate_hits': 0,
        'candidate_confirmed': 0,
        'candidate_forwarded_to_existing': 0,
        'candidate_added_from_shoot_model': 0,
        'candidate_window_per_side': int(neighborhood_frames),
        'candidate_window_max_delta_frames': int(neighborhood_radius),
        'candidate_neighbor_radius_frames': int(neighborhood_radius),
        'candidate_match_frames': {},
        'candidate_skipped_by_budget': 0,
        'gap_frames_requested': 0,
        'gap_frames_checked': 0,
        'gap_frames_dropped_by_budget': 0,
        'gap_hits': 0,
        'gap_fill_added': 0,
        'gap_hits_near_existing': 0,
        'output_candidates': 0,
        'candidate_hit_frames': [],
        'candidate_added_frames': [],
        'gap_fill_frames': [],
        'inference_candidate': {},
        'inference_gap': {},
        'timing_seconds': {
            'candidate': 0.0,
            'gap': 0.0,
            'total': 0.0,
        },
        'runtime_guard_activated': False,
        'runtime_guard_reason': None,
        'frames_inferred_total': 0,
    }

    frame_limit = max(0, int(total_frames) - 1) if total_frames else None
    candidate_priority = {}
    for shot in shots:
        release_frame = _release_frame_for_candidate(shot)
        if release_frame < 0:
            continue
        score = _candidate_priority_from_source(shot.get('source'))
        existing = float(candidate_priority.get(release_frame, 0.0) or 0.0)
        if score > existing:
            candidate_priority[release_frame] = score
    for frame_idx in (yolo_launches or []):
        release_frame = int(frame_idx)
        if release_frame < 0:
            continue
        existing = float(candidate_priority.get(release_frame, 0.0) or 0.0)
        if 3.0 > existing:
            candidate_priority[release_frame] = 3.0
    prioritized_core = sorted(candidate_priority.keys(), key=lambda f: (-float(candidate_priority.get(f, 0.0)), int(f)))
    debug['candidate_core_requested'] = len(prioritized_core)

    if budget_unlimited and candidate_budget_unlimited:
        selected_core = list(prioritized_core)
    else:
        core_limit = len(prioritized_core)
        if not budget_unlimited:
            core_limit = min(core_limit, max_total_frames)
        if not candidate_budget_unlimited:
            core_limit = min(core_limit, max_candidate_frames)
        core_limit = max(0, int(core_limit))
        if behind_basket_mode and core_limit < len(prioritized_core):
            selected_core = _select_temporally_spread_frames(prioritized_core, core_limit)
        else:
            selected_core = list(prioritized_core[:core_limit])
    selected_core_set = set(selected_core)
    debug['candidate_core_selected'] = len(selected_core)
    debug['candidate_core_dropped_by_budget'] = max(0, len(prioritized_core) - len(selected_core))

    neighbor_candidates = []
    if neighborhood_frames > 0 and selected_core:
        for release_frame in selected_core:
            per_side_added = {-1: 0, 1: 0}
            for step in range(1, neighborhood_radius + 1):
                for sign in (-1, 1):
                    if per_side_added[sign] >= neighborhood_frames:
                        continue
                    delta = int(sign * step)
                    candidate_frame = int(release_frame + delta)
                    if candidate_frame < 0:
                        continue
                    if frame_limit is not None and candidate_frame > frame_limit:
                        continue
                    if candidate_frame in selected_core_set:
                        continue
                    neighbor_candidates.append({
                        'frame_idx': candidate_frame,
                        'priority': float(candidate_priority.get(release_frame, 1.0)),
                        'delta': abs(delta),
                    })
                    per_side_added[sign] += 1
        neighbor_candidates.sort(key=lambda item: (-float(item['priority']), int(item['delta']), int(item['frame_idx'])))
    unique_neighbors = []
    seen_neighbors = set()
    for item in neighbor_candidates:
        frame_idx = int(item['frame_idx'])
        if frame_idx in seen_neighbors or frame_idx in selected_core_set:
            continue
        seen_neighbors.add(frame_idx)
        unique_neighbors.append(frame_idx)

    if budget_unlimited and candidate_budget_unlimited:
        selected_neighbors = list(unique_neighbors)
    else:
        remaining_after_core = len(unique_neighbors)
        if not budget_unlimited:
            remaining_after_core = min(remaining_after_core, max(0, max_total_frames - len(selected_core)))
        if not candidate_budget_unlimited:
            remaining_after_core = min(remaining_after_core, max(0, max_candidate_frames - len(selected_core)))
        selected_neighbors = list(unique_neighbors[:remaining_after_core])
    debug['candidate_neighbor_selected'] = len(selected_neighbors)
    debug['candidate_neighbor_dropped_by_budget'] = max(0, len(unique_neighbors) - len(selected_neighbors))

    candidate_inference_frames = sorted(set(int(f) for f in (selected_core + selected_neighbors)))
    debug['candidate_frames_checked'] = len(candidate_inference_frames)
    if not budget_unlimited:
        debug['candidate_skipped_by_budget'] = max(0, len(candidate_priority) - len(selected_core))

    candidate_detected, inference_candidate_debug = detect_shoot_events_on_frames(
        video_path,
        candidate_inference_frames,
        min_confidence=thresholds['candidate_confidence']
    )
    debug['inference_candidate'] = inference_candidate_debug
    debug['timing_seconds']['candidate'] = float(inference_candidate_debug.get('elapsed_seconds', 0.0) or 0.0)
    debug['frames_inferred_total'] += int(inference_candidate_debug.get('frames_inferred', 0) or 0)

    candidate_hits = {}
    selected_neighbors_set = set(int(f) for f in selected_neighbors)
    for release_frame in selected_core:
        best_conf = None
        best_frame = None
        for delta in range(-neighborhood_radius, neighborhood_radius + 1):
            frame_idx = int(release_frame + delta)
            if delta != 0 and frame_idx not in selected_neighbors_set:
                continue
            conf = candidate_detected.get(frame_idx)
            if conf is None:
                continue
            conf_v = float(conf)
            if best_conf is None or conf_v > best_conf:
                best_conf = conf_v
                best_frame = frame_idx
        if best_conf is None:
            continue
        candidate_hits[int(release_frame)] = (float(best_conf), int(best_frame if best_frame is not None else release_frame))

    debug['candidate_hits'] = len(candidate_hits)
    debug['candidate_hit_frames'] = sorted(int(f) for f in candidate_hits.keys())
    lookback_frames = max(1, int(0.45 * fps_safe))
    clip_len_frames = max(1, int(1.3 * fps_safe))
    max_frame = max(0, int(total_frames) - 1) if total_frames else None
    proximity_gap = max(1, int(SHOTLAB_EVENT_MERGE_GAP_FRAMES))
    matched_release_frames = set()
    yolo_launch_set = set(int(f) for f in (yolo_launches or []))
    for shot in shots:
        release_frame = _release_frame_for_candidate(shot)
        candidate_hit = candidate_hits.get(release_frame)
        if candidate_hit is None:
            shot.setdefault('shoot_model_confirmed', False)
            if release_frame not in selected_core_set and release_frame in candidate_priority:
                shot['shoot_model_budget_skipped'] = True
            continue
        shoot_conf, match_frame = candidate_hit
        shot['shoot_model_confirmed'] = True
        shot['shoot_model_confidence'] = round(float(shoot_conf), 4)
        shot['shoot_model_frame'] = int(match_frame)
        shot['shoot_model_match_type'] = 'candidate' if int(match_frame) == int(release_frame) else 'candidate_neighbor'
        shot['candidate_confidence'] = 'high'
        debug['candidate_match_frames'][str(int(release_frame))] = int(match_frame)
        debug['candidate_confirmed'] += 1
        matched_release_frames.add(int(release_frame))

    # Bridge unmatched shoot-model hits into the shot list so they cannot be lost before ball-flight.
    for release_frame, (shoot_conf, match_frame) in sorted(candidate_hits.items(), key=lambda item: int(item[0])):
        release_frame = int(release_frame)
        if release_frame in matched_release_frames:
            continue

        nearest_idx = None
        nearest_dist = None
        for idx, shot in enumerate(shots):
            shot_release = _release_frame_for_candidate(shot)
            dist = abs(int(shot_release) - release_frame)
            if dist > proximity_gap:
                continue
            if nearest_dist is None or dist < nearest_dist:
                nearest_idx = idx
                nearest_dist = dist

        if nearest_idx is not None:
            shot = shots[nearest_idx]
            was_confirmed = bool(shot.get('shoot_model_confirmed', False))
            prior_conf = float(shot.get('shoot_model_confidence', 0.0) or 0.0)
            if (not was_confirmed) or float(shoot_conf) > prior_conf:
                shot['shoot_model_confirmed'] = True
                shot['shoot_model_confidence'] = round(float(max(prior_conf, shoot_conf)), 4)
                shot['shoot_model_frame'] = int(match_frame)
                shot['shoot_model_match_type'] = 'candidate_bridge'
                shot['candidate_confidence'] = 'high'
            if not was_confirmed:
                debug['candidate_confirmed'] += 1
            debug['candidate_forwarded_to_existing'] += 1
            debug['candidate_match_frames'][str(release_frame)] = int(match_frame)
            matched_release_frames.add(release_frame)
            continue

        start_frame = max(0, release_frame - lookback_frames)
        end_frame = max(release_frame + 1, release_frame + clip_len_frames)
        if max_frame is not None:
            end_frame = min(end_frame, max_frame)
        inferred_source = 'yolo_launch' if release_frame in yolo_launch_set else 'shoot_model'
        added_shot = {
            'start_frame': int(start_frame),
            'release_frame': int(release_frame),
            'end_frame': int(max(int(start_frame) + 1, int(end_frame))),
            'timestamp': float(release_frame / fps_safe),
            'source': inferred_source,
            'shoot_model_confirmed': True,
            'shoot_model_confidence': round(float(shoot_conf), 4),
            'shoot_model_frame': int(match_frame),
            'shoot_model_match_type': 'candidate_bridge',
            'candidate_confidence': 'high'
        }
        shots.append(added_shot)
        debug['candidate_added_from_shoot_model'] += 1
        debug['candidate_added_frames'].append(int(release_frame))
        debug['candidate_match_frames'][str(release_frame)] = int(match_frame)
        debug['candidate_confirmed'] += 1
        matched_release_frames.add(release_frame)
        logger.info(
            "Shoot model bridged candidate added at frame %s (source=%s conf=%.3f)",
            release_frame,
            inferred_source,
            shoot_conf,
        )

    elapsed_after_candidate = float(max(0.0, time.time() - stage_started))
    runtime_budget_exceeded = (not stage_budget_unlimited) and elapsed_after_candidate >= max_event_seconds
    if runtime_budget_exceeded:
        debug['runtime_guard_activated'] = True
        debug['runtime_guard_reason'] = 'event_time_budget_exceeded_after_candidate'
        logger.info(
            "Event model runtime guard activated (candidate stage %.2fs >= budget %.2fs); skipping gap fill",
            elapsed_after_candidate,
            max_event_seconds,
        )

    gap_frames_all = _collect_gap_sample_frames(
        sorted(int(f) for f in candidate_priority.keys()),
        total_frames=total_frames,
        active_regions=active_regions,
        sample_stride=2 if behind_basket_mode else None,
    )
    debug['gap_frames_requested'] = len(gap_frames_all)
    gap_frames_selected = []
    if thresholds['gap_fill_enabled'] and not runtime_budget_exceeded:
        if budget_unlimited:
            remaining_budget = None
        else:
            remaining_budget = max(0, max_total_frames - len(candidate_inference_frames))
        gap_budget = len(gap_frames_all)
        if max_gap_frames > 0:
            gap_budget = min(gap_budget, max_gap_frames)
        if remaining_budget is not None:
            gap_budget = min(gap_budget, remaining_budget)
        gap_budget = max(0, int(gap_budget))
        if behind_basket_mode and gap_budget < len(gap_frames_all):
            gap_frames_selected = _select_temporally_spread_frames(gap_frames_all, gap_budget)
        else:
            gap_frames_selected = list(gap_frames_all[:gap_budget])
    debug['gap_frames_checked'] = len(gap_frames_selected)
    debug['gap_frames_dropped_by_budget'] = max(0, len(gap_frames_all) - len(gap_frames_selected))

    gap_detected = {}
    inference_gap_debug = {
        'enabled': bool(thresholds['gap_fill_enabled']),
        'requested_frames': len(gap_frames_selected),
        'frames_inferred': 0,
        'detections': 0,
        'errors': 0,
        'elapsed_seconds': 0.0,
    }
    if gap_frames_selected:
        gap_detected, inference_gap_debug = detect_shoot_events_on_frames(
            video_path,
            gap_frames_selected,
            min_confidence=thresholds['gap_confidence']
        )
    debug['inference_gap'] = inference_gap_debug
    debug['timing_seconds']['gap'] = float(inference_gap_debug.get('elapsed_seconds', 0.0) or 0.0)
    debug['frames_inferred_total'] += int(inference_gap_debug.get('frames_inferred', 0) or 0)

    gap_hits = sorted(
        (
            int(frame_idx),
            float(conf)
        )
        for frame_idx, conf in (gap_detected or {}).items()
        if frame_idx in gap_frames_selected and float(conf) >= float(thresholds['gap_confidence'])
    )
    debug['gap_hits'] = len(gap_hits)

    existing_release_frames = sorted(_release_frame_for_candidate(shot) for shot in shots)
    for gap_frame, shoot_conf in gap_hits:
        if any(abs(gap_frame - rf) < proximity_gap for rf in existing_release_frames):
            debug['gap_hits_near_existing'] += 1
            continue
        start_frame = max(0, gap_frame - lookback_frames)
        end_frame = max(gap_frame + 1, gap_frame + clip_len_frames)
        if max_frame is not None:
            end_frame = min(end_frame, max_frame)
        added_shot = {
            'start_frame': int(start_frame),
            'release_frame': int(gap_frame),
            'end_frame': int(max(int(start_frame) + 1, int(end_frame))),
            'timestamp': float(gap_frame / fps_safe),
            'source': 'shoot_model_only',
            'shoot_model_confirmed': True,
            'shoot_model_confidence': round(float(shoot_conf), 4),
            'shoot_model_frame': int(gap_frame),
            'shoot_model_match_type': 'gap_fill',
            'candidate_confidence': 'high'
        }
        shots.append(added_shot)
        existing_release_frames.append(int(gap_frame))
        debug['gap_fill_added'] += 1
        debug['gap_fill_frames'].append(int(gap_frame))
        logger.info("Shoot model gap-fill candidate added at frame %s (conf=%.3f)", gap_frame, shoot_conf)

    shots.sort(key=lambda s: _release_frame_for_candidate(s))
    debug['output_candidates'] = len(shots)
    debug['timing_seconds']['total'] = float(max(0.0, time.time() - stage_started))
    if shot_detection_debug is not None:
        shot_detection_debug['shoot_model'] = dict(debug)
    if pose_debug is not None:
        pose_debug['shot_attempts_after_shoot_model'] = len(shots)
    logger.info(
        "Shoot model integration: checked=%d candidate_hits=%d confirmed=%d bridged=%d added=%d gap_checked=%d gap_hits=%d gap_added=%d output=%d (candidate=%.2fs gap=%.2fs guard=%s)",
        debug['candidate_frames_checked'],
        debug['candidate_hits'],
        debug['candidate_confirmed'],
        debug.get('candidate_forwarded_to_existing', 0),
        debug.get('candidate_added_from_shoot_model', 0),
        debug['gap_frames_checked'],
        debug['gap_hits'],
        debug['gap_fill_added'],
        debug['output_candidates'],
        debug['timing_seconds']['candidate'],
        debug['timing_seconds']['gap'],
        bool(debug.get('runtime_guard_activated', False)),
    )
    return shots, debug


def has_upward_arc(tracks_in_window, frame_height, min_apex_delta=55.0):
    """
    A valid shot should show a clear upward arc before descent.
    Returns (passed, arc_delta, reason).
    """
    if len(tracks_in_window) < 3:
        return True, 0.0, 'insufficient_data_pass'

    ys = []
    for track in tracks_in_window:
        _, ty = _track_xy(track)
        ys.append(float(ty))

    if not ys:
        return True, 0.0, 'no_y_data_pass'

    first_y = float(ys[0])
    min_y = float(min(ys))
    max_y = float(max(ys))
    apex_delta = first_y - min_y

    if apex_delta >= float(min_apex_delta):
        return True, apex_delta, 'upward_arc_confirmed'

    fall_delta = max_y - first_y
    starts_high = first_y < (float(frame_height or 0.0) * SHOTLAB_UPWARD_ARC_HIGH_START_RATIO)
    if starts_high and fall_delta >= SHOTLAB_UPWARD_ARC_MID_FLIGHT_FALL_DELTA_PX:
        return True, fall_delta, 'mid_flight_detected'

    return False, apex_delta, f'no_upward_arc(delta={apex_delta:.1f}px, need>={float(min_apex_delta):.1f}px)'


def _get_frame_detection(yolo_detections, frame_idx):
    if not yolo_detections:
        return {}
    det = yolo_detections.get(int(frame_idx))
    if det is None:
        det = yolo_detections.get(str(int(frame_idx)))
    return det or {}


def _is_ball_near_player_boxes(ball_xy, players, expand_scale=1.12):
    if not players:
        return False
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    for player in players:
        bbox = player.get('bbox') if isinstance(player, dict) else None
        if not bbox or len(bbox) != 4:
            continue
        try:
            expanded = _expand_bbox(bbox, expand_scale)
        except Exception:
            continue
        if _point_in_bbox(bx, by, expanded):
            return True
    return False


def _yolo_candidate_has_shooter_proximity(
    release_frame,
    tracks_in_window,
    pose_frames,
    yolo_detections,
    fps
):
    max_dist_px = max(80.0, float(SHOTLAB_YOLO_CONFIRM_NEAR_SHOOTER_MAX_DIST_PX))
    min_near_frames = max(1, int(SHOTLAB_YOLO_CONFIRM_NEAR_SHOOTER_MIN_FRAMES))
    pose_gap = max(1, int(SHOTLAB_YOLO_CONFIRM_POSE_GAP_FRAMES))
    probe_before = max(1, int(round(0.15 * float(fps or 30.0))))
    probe_after = max(1, int(round(0.70 * float(fps or 30.0))))
    probe_start = int(release_frame) - probe_before
    probe_end = int(release_frame) + probe_after

    frame_to_entry, frame_keys = _build_pose_frame_lookup(pose_frames)

    near_frames = 0
    pose_hits = 0
    player_hits = 0
    checked_frames = 0
    min_pose_distance = None

    for track in tracks_in_window:
        frame_idx = int(track.get('frame', 0))
        if frame_idx < probe_start or frame_idx > probe_end:
            continue
        checked_frames += 1
        tx = float(track.get('x', _track_xy(track)[0]))
        ty = float(track.get('y', _track_xy(track)[1]))

        pose_near = False
        shooter_points = _get_shooter_reference_points_for_frame(
            frame_idx,
            frame_to_entry,
            frame_keys,
            pose_gap
        )
        if shooter_points:
            local_min = min(_distance_2d((tx, ty), pt) for pt in shooter_points)
            min_pose_distance = (
                local_min
                if min_pose_distance is None
                else min(min_pose_distance, local_min)
            )
            if local_min <= max_dist_px:
                pose_near = True
                pose_hits += 1

        player_near = False
        frame_det = _get_frame_detection(yolo_detections, frame_idx)
        players = frame_det.get('players', []) if isinstance(frame_det, dict) else []
        if _is_ball_near_player_boxes((tx, ty), players):
            player_near = True
            player_hits += 1

        if pose_near or player_near:
            near_frames += 1

    signals = {
        'shooter_proximity_checked_frames': int(checked_frames),
        'shooter_proximity_near_frames': int(near_frames),
        'shooter_proximity_pose_hits': int(pose_hits),
        'shooter_proximity_player_hits': int(player_hits),
        'shooter_proximity_min_frames_required': int(min_near_frames),
        'shooter_proximity_max_dist_px': float(max_dist_px),
        'shooter_proximity_probe_window': [int(probe_start), int(probe_end)],
    }
    if min_pose_distance is not None:
        signals['shooter_proximity_min_pose_distance_px'] = round(float(min_pose_distance), 2)

    return near_frames >= min_near_frames, signals


def confirm_shot_with_ball_flight(
    release_frame,
    pose_frames,
    yolo_detections,
    rim_position,
    frame_width,
    frame_height,
    fps,
    camera_position=None,
    candidate_source='pose',
    candidate_metadata=None
):
    """
    Confirm a shot candidate using spatial trajectory evidence.
    Returns: {'is_confirmed': bool, 'reason': str, 'signals': {...}}
    """
    release_frame = int(release_frame or 0)
    source = str(candidate_source or 'pose')
    candidate_metadata = dict(candidate_metadata or {})
    shoot_model_confirmed = bool(candidate_metadata.get('shoot_model_confirmed', False))
    shoot_model_confidence = float(candidate_metadata.get('shoot_model_confidence', 0.0) or 0.0)
    shoot_model_only = bool(source == 'shoot_model_only' or candidate_metadata.get('shoot_model_match_type') == 'gap_fill')
    frame_w = float(frame_width or 0)
    frame_h = float(frame_height or 0)
    if frame_w <= 1.0:
        frame_w = 1920.0
    if frame_h <= 1.0:
        frame_h = 1080.0

    if not yolo_detections:
        if source == 'yolo_launch':
            return {
                'is_confirmed': False,
                'reason': 'yolo_launch_no_yolo_tracks',
                'signals': {
                    'candidate_source': source,
                    'fallback': False
                }
            }
        fallback_reason = 'no_yolo_fallback'
        return {
            'is_confirmed': True,
            'reason': fallback_reason,
            'signals': {
                'candidate_source': source,
                'fallback': True
            }
        }
    search_before = 20
    search_after = 60
    min_track_points = 3
    min_coverage = 0.15
    min_displacement_px = 30.0
    big_move_override_px = 150.0
    disappearance_gap_min = 5
    require_strong_spatial_signal = False
    if source != 'pose' and not shoot_model_confirmed:
        min_track_points = 6
        min_coverage = 0.25
        min_displacement_px = 55.0
        require_strong_spatial_signal = True
    if shoot_model_only:
        min_track_points = max(3, min_track_points - 1)
        min_coverage = max(0.12, min_coverage - 0.05)
        min_displacement_px = max(25.0, min_displacement_px - 10.0)
    if shoot_model_confirmed:
        min_track_points = max(3, min_track_points - 2)
        min_coverage = min(0.15, max(0.10, min_coverage - 0.08))
        min_displacement_px = max(20.0, min_displacement_px - 20.0)
        disappearance_gap_min = max(3, disappearance_gap_min - 1)
        require_strong_spatial_signal = False

    window_start = release_frame - search_before
    window_end = release_frame + search_after
    window_frames = max(1, window_end - window_start + 1)

    tracks_in_window = []
    for fn in range(window_start, window_end + 1):
        ball = get_ball_position_at_frame(yolo_detections, fn)
        if not ball:
            continue
        center = ball.get('center')
        bx = ball.get('x', None)
        by = ball.get('y', None)
        if center and len(center) >= 2 and center[0] is not None and center[1] is not None:
            bx = float(center[0])
            by = float(center[1])
        elif bx is not None and by is not None:
            bx = float(bx)
            by = float(by)
        else:
            continue
        bbox = ball.get('bbox') or [None, None, None, None]
        width = 0.0
        height = 0.0
        if bbox[0] is not None and bbox[2] is not None:
            width = max(0.0, float(bbox[2]) - float(bbox[0]))
        if bbox[1] is not None and bbox[3] is not None:
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
        tracks_in_window.append({
            'frame': int(fn),
            'x': float(bx),
            'y': float(by),
            'center': (float(bx), float(by)),
            'width': float(width),
            'height': float(height),
            'area': float(ball.get('area', 0.0) or 0.0),
            'confidence': float(ball.get('confidence', 0.0) or 0.0)
        })

    tracks_in_window = sorted(tracks_in_window, key=lambda t: int(t.get('frame', 0)))
    coverage_ratio = len(tracks_in_window) / float(window_frames)
    signals = {
        'candidate_source': source,
        'window_start': int(window_start),
        'window_end': int(window_end),
        'window_frames': int(window_frames),
        'tracks_in_window': int(len(tracks_in_window)),
        'coverage_ratio': round(float(coverage_ratio), 4),
        'min_track_points': int(min_track_points),
        'coverage_threshold': float(min_coverage),
        'displacement_threshold_px': float(min_displacement_px),
        'big_move_override_px': float(big_move_override_px),
        'upward_arc_min_apex_delta_px': float(SHOTLAB_UPWARD_ARC_MIN_APEX_DELTA_PX),
        'shoot_model_confirmed': bool(shoot_model_confirmed),
        'shoot_model_confidence': round(float(shoot_model_confidence), 4) if shoot_model_confidence > 0 else 0.0,
        'shoot_model_only': bool(shoot_model_only)
    }

    if len(tracks_in_window) < min_track_points:
        if source == 'yolo_launch':
            return {
                'is_confirmed': False,
                'reason': 'insufficient_ball_detections',
                'signals': signals
            }
        return {
            'is_confirmed': False,
            'reason': 'insufficient_ball_detections',
            'signals': signals
        }

    if coverage_ratio < min_coverage:
        return {
            'is_confirmed': False,
            'reason': 'low_ball_coverage',
            'signals': signals
        }

    if camera_position == 'in_front_of_shooter':
        early_end = int(release_frame + max(6, int(0.50 * float(fps or 30.0))))
        early_tracks = [
            t for t in tracks_in_window
            if int(release_frame) <= int(t.get('frame', 0)) <= early_end
        ]
        if len(early_tracks) >= 3:
            early_tracks = sorted(early_tracks, key=lambda t: int(t.get('frame', 0)))
            start_y = float(early_tracks[0].get('y', _track_xy(early_tracks[0])[1]))
            min_y = min(float(t.get('y', _track_xy(t)[1])) for t in early_tracks)
            upward_delta = float(start_y - min_y)
            size_values = []
            for t in early_tracks:
                width = float(t.get('width', 0.0) or 0.0)
                height = float(t.get('height', 0.0) or 0.0)
                area = float(t.get('area', 0.0) or 0.0)
                size_values.append(max(area, width * height))
            start_size = float(size_values[0]) if size_values else 0.0
            min_size = float(min(size_values)) if size_values else 0.0
            shrink_ratio = (min_size / start_size) if start_size > 1e-6 else 1.0
            away_motion_ok = bool(upward_delta >= 6.0 and shrink_ratio <= 0.94)
            signals['behind_basket_away_window'] = [int(release_frame), int(early_end)]
            signals['behind_basket_upward_delta_px'] = round(float(upward_delta), 2)
            signals['behind_basket_shrink_ratio'] = round(float(shrink_ratio), 4)
            signals['behind_basket_away_motion_ok'] = away_motion_ok
            # Behind-basket away-motion is used as a soft signal only.

    if shoot_model_only:
        release_support_window = max(4, int(0.25 * float(fps or 30.0)))
        support_start = release_frame - release_support_window
        support_end = release_frame + release_support_window
        local_tracks = [
            t for t in tracks_in_window
            if support_start <= int(t.get('frame', 0)) <= support_end
        ]
        local_motion = 0.0
        if len(local_tracks) >= 2:
            for i in range(1, len(local_tracks)):
                px, py = _track_xy(local_tracks[i - 1])
                cx, cy = _track_xy(local_tracks[i])
                local_motion = max(local_motion, _distance_2d((px, py), (cx, cy)))
        signals['shoot_model_only_support_window'] = [int(support_start), int(support_end)]
        signals['shoot_model_only_support_points'] = int(len(local_tracks))
        signals['shoot_model_only_support_motion_px'] = round(float(local_motion), 2)
        if len(local_tracks) < 2 or local_motion < 16.0:
            return {
                'is_confirmed': False,
                'reason': 'shoot_model_only_insufficient_ball_support',
                'signals': signals
            }

    proximity_failed = False
    if source == 'yolo_launch' and not shoot_model_confirmed:
        proximity_ok, proximity_signals = _yolo_candidate_has_shooter_proximity(
            release_frame,
            tracks_in_window,
            pose_frames,
            yolo_detections,
            fps,
        )
        signals.update(proximity_signals)
        if camera_position == 'in_front_of_shooter' and proximity_ok:
            # In behind-basket/front mode, require repeated proximity for yolo-only candidates.
            near_frames = int(proximity_signals.get('shooter_proximity_near_frames', 0) or 0)
            min_near_frames_front = 2
            signals['shooter_proximity_front_min_frames_required'] = int(min_near_frames_front)
            if near_frames < min_near_frames_front:
                proximity_ok = False
        if not proximity_ok:
            proximity_failed = True
        signals['shooter_proximity_passed'] = bool(proximity_ok)

    def finalize_spatial_pass(spatial_reason):
        arc_passed, arc_delta, arc_reason = has_upward_arc(
            tracks_in_window,
            frame_h,
            min_apex_delta=SHOTLAB_UPWARD_ARC_MIN_APEX_DELTA_PX
        )
        signals['upward_arc_passed'] = bool(arc_passed)
        signals['upward_arc_delta_px'] = round(float(arc_delta), 2)
        signals['upward_arc_reason'] = str(arc_reason)
        logger.info(
            "Ball-flight upward arc check frame=%s source=%s passed=%s delta=%.1f reason=%s spatial_reason=%s",
            release_frame,
            source,
            arc_passed,
            arc_delta,
            arc_reason,
            spatial_reason
        )
        if not arc_passed:
            return {
                'is_confirmed': False,
                'reason': 'no_upward_arc',
                'signals': signals
            }
        if source == 'yolo_launch' and not shoot_model_confirmed and proximity_failed:
            signals['spatial_reason_without_proximity'] = str(spatial_reason)
            return {
                'is_confirmed': False,
                'reason': 'no_shooter_proximity',
                'signals': signals
            }
        return {
            'is_confirmed': True,
            'reason': spatial_reason,
            'signals': signals
        }

    rim_bbox = None
    if rim_position is not None and len(rim_position) >= 4:
        rx1, ry1, rx2, ry2 = [float(v) for v in rim_position[:4]]
        rim_bbox = (rx1, ry1, rx2, ry2)
    if rim_bbox is not None:
        rim_cx = (rim_bbox[0] + rim_bbox[2]) / 2.0
        rim_cy = (rim_bbox[1] + rim_bbox[3]) / 2.0
        rim_w = max(1.0, rim_bbox[2] - rim_bbox[0])
        rim_h = max(1.0, rim_bbox[3] - rim_bbox[1])
    else:
        rim_cx = frame_w * 0.5
        rim_cy = frame_h * 0.3
        rim_w = 100.0
        rim_h = 100.0

    shooter_zone_y_min = frame_h * 0.35
    rim_zone_radius_x = max(200.0, rim_w)
    rim_zone_radius_y = max(200.0, rim_h)
    shooter_frame = None
    rim_frame = None
    for track in tracks_in_window:
        tx = float(track.get('x', _track_xy(track)[0]))
        ty = float(track.get('y', _track_xy(track)[1]))
        if shooter_frame is None and ty > shooter_zone_y_min:
            shooter_frame = int(track['frame'])
        if abs(tx - rim_cx) < rim_zone_radius_x and abs(ty - rim_cy) < rim_zone_radius_y:
            rim_frame = int(track['frame'])

    signals.update({
        'shooter_zone_y_min': round(float(shooter_zone_y_min), 2),
        'rim_center': [round(float(rim_cx), 2), round(float(rim_cy), 2)],
        'rim_zone_radius_x': round(float(rim_zone_radius_x), 2),
        'rim_zone_radius_y': round(float(rim_zone_radius_y), 2),
        'first_shooter_frame': shooter_frame,
        'last_rim_frame': rim_frame
    })
    shooter_to_rim_trajectory = bool(shooter_frame is not None and rim_frame is not None and rim_frame > shooter_frame)

    first = tracks_in_window[0]
    last = tracks_in_window[-1]
    fx, fy = _track_xy(first)
    lx, ly = _track_xy(last)
    displacement = _distance_2d((fx, fy), (lx, ly))
    signals['net_displacement_px'] = round(float(displacement), 2)
    if shooter_to_rim_trajectory:
        if (
            source == 'yolo_launch'
            and not shoot_model_confirmed
            and camera_position == 'in_front_of_shooter'
            and displacement < max(min_displacement_px, 85.0)
        ):
            signals['front_yolo_displacement_min_px'] = float(max(min_displacement_px, 85.0))
        else:
            return finalize_spatial_pass('shooter_to_rim_trajectory')
    if (
        source == 'yolo_launch'
        and not shoot_model_confirmed
        and camera_position == 'in_front_of_shooter'
        and displacement <= max(min_displacement_px, 85.0)
    ):
        signals['front_yolo_displacement_min_px'] = float(max(min_displacement_px, 85.0))
    elif displacement > min_displacement_px:
        if (
            source == 'yolo_launch'
            and not shoot_model_confirmed
            and camera_position == 'in_front_of_shooter'
            and shooter_frame is None
        ):
            return {
                'is_confirmed': False,
                'reason': 'no_shooter_proximity' if proximity_failed else 'front_no_shooter_anchor',
                'signals': signals
            }
        return finalize_spatial_pass('directed_displacement')

    xs = [float(_track_xy(t)[0]) for t in tracks_in_window]
    ys = [float(_track_xy(t)[1]) for t in tracks_in_window]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    signals['x_range_px'] = round(float(x_range), 2)
    signals['y_range_px'] = round(float(y_range), 2)
    if not require_strong_spatial_signal and (x_range > big_move_override_px or y_range > big_move_override_px):
        return finalize_spatial_pass('big_movement_override')

    frames_present = sorted(int(t['frame']) for t in tracks_in_window)
    max_gap = 0
    for i in range(1, len(frames_present)):
        max_gap = max(max_gap, frames_present[i] - frames_present[i - 1])
    signals['max_detection_gap_frames'] = int(max_gap)
    if not require_strong_spatial_signal and max_gap >= disappearance_gap_min:
        return finalize_spatial_pass('ball_disappearance')

    if source == 'yolo_launch':
        if camera_position == 'in_front_of_shooter' and not shoot_model_confirmed:
            # For front/behind-basket yolo-only shots, avoid accepting candidates with no shooter anchor.
            if signals.get('first_shooter_frame') is None:
                return {
                    'is_confirmed': False,
                    'reason': 'no_shooter_proximity' if proximity_failed else 'front_no_shooter_anchor',
                    'signals': signals
                }
        return {
            'is_confirmed': False,
            'reason': 'no_shooter_proximity' if proximity_failed else 'no_strong_trajectory_confirmed',
            'signals': signals
        }

    return {
        'is_confirmed': False,
        'reason': 'no_trajectory_confirmed',
        'signals': signals
    }


def _strong_spatial_signal_from_confirmation(signals):
    if not isinstance(signals, dict):
        return False
    shooter_frame = signals.get('first_shooter_frame')
    rim_frame = signals.get('last_rim_frame')
    if shooter_frame is not None and rim_frame is not None:
        try:
            if int(rim_frame) > int(shooter_frame):
                return True
        except (TypeError, ValueError):
            pass
    displacement = float(signals.get('net_displacement_px', 0.0) or 0.0)
    if displacement >= 90.0:
        return True
    x_range = float(signals.get('x_range_px', 0.0) or 0.0)
    y_range = float(signals.get('y_range_px', 0.0) or 0.0)
    return x_range >= 150.0 or y_range >= 150.0


def _should_accept_pose_second_chance(reason, signals, camera_position=None):
    if not SHOTLAB_POSE_SECOND_CHANCE_ENABLE:
        return False, 'disabled'
    if not isinstance(signals, dict):
        return False, 'no_signals'
    allowed_reasons = {
        'no_upward_arc',
        'no_trajectory_confirmed',
        'no_strong_trajectory_confirmed',
    }
    if str(reason or '') not in allowed_reasons:
        return False, 'reason_not_eligible'

    tracks = int(signals.get('tracks_in_window', 0) or 0)
    coverage = float(signals.get('coverage_ratio', 0.0) or 0.0)
    arc_passed = bool(signals.get('upward_arc_passed', False))
    strong_spatial = _strong_spatial_signal_from_confirmation(signals)
    is_sideline = bool(camera_position and str(camera_position).startswith('sideline'))

    if reason == 'no_upward_arc':
        min_tracks = 9 if is_sideline else 10
        min_cov = 0.24 if is_sideline else 0.26
        if tracks >= min_tracks and coverage >= min_cov and strong_spatial:
            return True, 'arc_relaxed_strong_spatial'
        return False, 'no_upward_arc_guard_failed'

    if tracks >= 7 and coverage >= 0.24 and arc_passed and strong_spatial:
        return True, 'trajectory_relaxed_strong_spatial'
    return False, 'trajectory_guard_failed'


def _should_accept_yolo_no_shooter_recall_rescue(reason, source, signals, camera_position=None):
    if str(reason or '') != 'no_shooter_proximity':
        return False, 'reason_not_eligible'
    if str(source or '') != 'yolo_launch':
        return False, 'source_not_eligible'
    if bool((signals or {}).get('shoot_model_confirmed', False)):
        return False, 'already_shoot_confirmed'
    if str(camera_position or '') != 'in_front_of_shooter':
        return False, 'not_behind_basket'

    tracks = int((signals or {}).get('tracks_in_window', 0) or 0)
    coverage = float((signals or {}).get('coverage_ratio', 0.0) or 0.0)
    upward_arc_passed = bool((signals or {}).get('upward_arc_passed', False))
    upward_arc_delta = float((signals or {}).get('upward_arc_delta_px', 0.0) or 0.0)
    displacement = float((signals or {}).get('net_displacement_px', 0.0) or 0.0)
    first_shooter = (signals or {}).get('first_shooter_frame')
    last_rim = (signals or {}).get('last_rim_frame')
    shooter_to_rim = False
    try:
        shooter_to_rim = first_shooter is not None and last_rim is not None and int(last_rim) > int(first_shooter)
    except (TypeError, ValueError):
        shooter_to_rim = False
    if coverage < 0.55:
        return False, 'coverage_below_rescue_floor'
    if tracks < 25:
        return False, 'tracks_below_rescue_floor'
    if (not upward_arc_passed) or upward_arc_delta < 100.0:
        return False, 'upward_arc_below_rescue_floor'
    if not shooter_to_rim and displacement < 120.0:
        return False, 'direction_or_displacement_below_floor'
    return True, 'no_shooter_proximity_recall_rescue'


def confirm_shot_candidates_with_ball_flight(
    shot_candidates,
    pose_frames,
    yolo_detections,
    rim_position,
    frame_width,
    frame_height,
    fps,
    camera_position=None
):
    """Confirm pose-based shot candidates using ball flight checks."""
    debug = {
        'input_count': len(shot_candidates or []),
        'confirmed': 0,
        'rejected': 0,
        'dropped_reasons': defaultdict(int),
        'details': [],
        'dropped_shots': []
    }
    if not shot_candidates:
        debug['dropped_reasons'] = dict(debug['dropped_reasons'])
        return [], debug

    confirmed = []
    for shot in shot_candidates:
        release_frame = int(shot.get('release_frame', shot.get('start_frame', 0)))
        candidate_source = str(shot.get('source') or 'pose')
        confirmation = confirm_shot_with_ball_flight(
            release_frame=release_frame,
            pose_frames=pose_frames,
            yolo_detections=yolo_detections,
            rim_position=rim_position,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            camera_position=camera_position,
            candidate_source=candidate_source,
            candidate_metadata=shot
        )
        is_confirmed = bool(confirmation.get('is_confirmed'))
        reason = confirmation.get('reason', 'unknown')
        signals = confirmation.get('signals', {})

        debug['details'].append({
            'release_frame': release_frame,
            'source': candidate_source,
            'is_confirmed': is_confirmed,
            'reason': reason,
            'signals': signals
        })

        if is_confirmed:
            debug['confirmed'] += 1
            confirmed_shot = dict(shot)
            confirmed_shot['confirmation_reason'] = reason
            confirmed_shot['confirmation_signals'] = signals
            if bool(confirmed_shot.get('shoot_model_confirmed', False)):
                confirmed_shot['candidate_confidence'] = 'high'
            else:
                confirmed_shot['candidate_confidence'] = str(confirmed_shot.get('candidate_confidence') or 'standard')
            confirmed.append(confirmed_shot)
            logger.info(
                "Shot candidate %s (%s) confirmed by ball flight: %s | signals=%s",
                release_frame,
                candidate_source,
                reason,
                signals
            )
        else:
            rescue_accept, rescue_reason = _should_accept_yolo_no_shooter_recall_rescue(
                reason,
                candidate_source,
                signals,
                camera_position=camera_position,
            )
            if rescue_accept:
                restored = dict(shot)
                restored['confirmation_reason'] = 'no_shooter_proximity_rescue'
                restored['confirmation_signals'] = dict(signals or {})
                restored['confirmation_signals']['recall_rescue_reason'] = rescue_reason
                restored['candidate_confidence'] = str(restored.get('candidate_confidence') or 'standard')
                confirmed.append(restored)
                debug['confirmed'] += 1
                logger.info(
                    "Shot candidate %s (%s) recovered by recall rescue: %s | signals=%s",
                    release_frame,
                    candidate_source,
                    rescue_reason,
                    signals,
                )
                continue
            debug['rejected'] += 1
            debug['dropped_reasons'][reason] += 1
            debug['dropped_shots'].append({
                'release_frame': release_frame,
                'source': candidate_source,
                'reason': reason,
                'signals': signals
            })
            logger.info(
                "Shot candidate %s (%s) rejected by ball flight: %s | signals=%s",
                release_frame,
                candidate_source,
                reason,
                signals
            )

    second_chance_added = 0
    if SHOTLAB_POSE_SECOND_CHANCE_ENABLE and debug['dropped_shots']:
        for dropped in debug['dropped_shots']:
            if second_chance_added >= max(0, int(SHOTLAB_POSE_SECOND_CHANCE_MAX)):
                break
            if str(dropped.get('source') or 'pose') != 'pose':
                continue
            accept, second_reason = _should_accept_pose_second_chance(
                dropped.get('reason'),
                dropped.get('signals') or {},
                camera_position=camera_position
            )
            if not accept:
                continue
            release_frame = int(dropped.get('release_frame', 0))
            shot = next(
                (s for s in (shot_candidates or []) if int(s.get('release_frame', s.get('start_frame', 0))) == release_frame),
                None
            )
            if shot is None:
                continue
            restored = dict(shot)
            restored['confirmation_reason'] = 'second_chance_pose'
            restored['confirmation_signals'] = dict(dropped.get('signals') or {})
            restored['confirmation_signals']['second_chance_reason'] = second_reason
            confirmed.append(restored)
            second_chance_added += 1
            debug['confirmed'] += 1
            debug['rejected'] = max(0, debug['rejected'] - 1)
            debug['dropped_reasons'][str(dropped.get('reason', 'unknown'))] = max(
                0,
                int(debug['dropped_reasons'].get(str(dropped.get('reason', 'unknown')), 0)) - 1
            )

    if second_chance_added:
        debug['second_chance_added'] = int(second_chance_added)
        confirmed.sort(key=lambda s: int(s.get('release_frame', s.get('start_frame', 0))))

    debug['dropped_reasons'] = dict(debug['dropped_reasons'])
    return confirmed, debug


def enforce_yolo_source_share_cap(shot_attempts, max_share=0.35):
    """Cap YOLO/ball-first sourced shots so pose remains dominant when available."""
    candidates = [dict(s) for s in (shot_attempts or [])]
    debug = {
        'input_count': len(candidates),
        'max_share': float(max_share),
        'pose_count': 0,
        'yolo_count': 0,
        'max_yolo_allowed': None,
        'dropped': 0,
        'dropped_release_frames': [],
        'applied': False,
    }
    if not candidates:
        return candidates, debug

    max_share = min(0.95, max(0.0, float(max_share)))

    pose_candidates = []
    yolo_candidates = []
    for shot in candidates:
        source = str(shot.get('source') or 'pose').lower()
        if 'yolo' in source or 'ball_first' in source:
            yolo_candidates.append(shot)
        else:
            pose_candidates.append(shot)

    pose_count = len(pose_candidates)
    yolo_count = len(yolo_candidates)
    debug['pose_count'] = int(pose_count)
    debug['yolo_count'] = int(yolo_count)

    # Recovery mode: if pose failed entirely, keep YOLO candidates to preserve recall.
    if pose_count <= 0:
        debug['max_yolo_allowed'] = int(yolo_count)
        return sorted(candidates, key=lambda s: int(s.get('release_frame', s.get('start_frame', 0)))), debug

    if max_share <= 0.0:
        max_yolo_allowed = 0
    elif max_share >= 1.0:
        max_yolo_allowed = yolo_count
    else:
        max_yolo_allowed = int(math.floor((max_share / (1.0 - max_share)) * float(pose_count)))
    max_yolo_allowed = max(0, min(yolo_count, max_yolo_allowed))
    debug['max_yolo_allowed'] = int(max_yolo_allowed)

    if yolo_count <= max_yolo_allowed:
        return sorted(candidates, key=lambda s: int(s.get('release_frame', s.get('start_frame', 0)))), debug

    def _rank_yolo_shot(shot):
        signals = shot.get('confirmation_signals') or {}
        return (
            float(shot.get('ball_first_support', 0.0) or 0.0),
            float(shot.get('ball_first_score', 0.0) or 0.0),
            float(signals.get('tracks_in_window', 0.0) or 0.0),
            float(signals.get('coverage_ratio', 0.0) or 0.0),
            -int(shot.get('release_frame', shot.get('start_frame', 0)) or 0),
        )

    kept_yolo = sorted(yolo_candidates, key=_rank_yolo_shot, reverse=True)[:max_yolo_allowed]
    kept_ids = {id(s) for s in kept_yolo}
    dropped = [s for s in yolo_candidates if id(s) not in kept_ids]
    filtered = pose_candidates + kept_yolo
    filtered.sort(key=lambda s: int(s.get('release_frame', s.get('start_frame', 0))))

    debug['applied'] = True
    debug['dropped'] = len(dropped)
    debug['dropped_release_frames'] = [
        int(s.get('release_frame', s.get('start_frame', 0)) or 0)
        for s in dropped
    ]
    return filtered, debug


def suppress_behind_basket_duplicate_echoes(shot_attempts, fps, camera_position=None):
    """
    Remove likely duplicate echoes in behind-basket mode.
    Typical pattern: a strong shot detection followed by a pose-only rebound/continuation
    within ~3 seconds.
    """
    shots = sorted(
        [dict(s) for s in (shot_attempts or [])],
        key=lambda s: int(s.get('release_frame', s.get('start_frame', 0)) or 0)
    )
    debug = {
        'applied': False,
        'input_count': len(shots),
        'output_count': len(shots),
        'dropped': 0,
        'dropped_release_frames': [],
        'reason': None,
        'max_gap_frames': 0,
    }
    if len(shots) < 2:
        return shots, debug
    if str(camera_position or '') != 'in_front_of_shooter':
        return shots, debug

    fps_safe = max(1.0, float(fps or 30.0))
    max_gap_frames = int(max(1, round(float(SHOTLAB_BEHIND_BASKET_MIN_GAP_SECONDS) * fps_safe)))
    debug['max_gap_frames'] = int(max_gap_frames)
    kept = []
    for shot in shots:
        if not kept:
            kept.append(shot)
            continue
        prev = kept[-1]
        prev_release = int(prev.get('release_frame', prev.get('start_frame', 0)) or 0)
        curr_release = int(shot.get('release_frame', shot.get('start_frame', 0)) or 0)
        frame_gap = int(curr_release - prev_release)
        if frame_gap > max_gap_frames:
            kept.append(shot)
            continue

        prev_confirmed = bool(prev.get('shoot_model_confirmed', False))
        curr_confirmed = bool(shot.get('shoot_model_confirmed', False))
        prev_source = str(prev.get('source') or 'pose')
        curr_source = str(shot.get('source') or 'pose')

        # Keep strongest candidate when two close detections likely represent the same attempt.
        if prev_confirmed and not curr_confirmed:
            debug['dropped'] += 1
            debug['dropped_release_frames'].append(int(curr_release))
            continue
        if curr_confirmed and not prev_confirmed:
            kept[-1] = shot
            debug['dropped'] += 1
            debug['dropped_release_frames'].append(int(prev_release))
            continue
        if prev_confirmed and curr_confirmed:
            prev_conf = float(prev.get('shoot_model_confidence', 0.0) or 0.0)
            curr_conf = float(shot.get('shoot_model_confidence', 0.0) or 0.0)
            if curr_conf > prev_conf:
                kept[-1] = shot
                debug['dropped'] += 1
                debug['dropped_release_frames'].append(int(prev_release))
            else:
                debug['dropped'] += 1
                debug['dropped_release_frames'].append(int(curr_release))
            continue

        # Both unconfirmed: keep non-pose source, otherwise keep earlier one.
        if prev_source != 'pose' and curr_source == 'pose':
            debug['dropped'] += 1
            debug['dropped_release_frames'].append(int(curr_release))
            continue
        if prev_source == 'pose' and curr_source != 'pose':
            kept[-1] = shot
            debug['dropped'] += 1
            debug['dropped_release_frames'].append(int(prev_release))
            continue

        debug['dropped'] += 1
        debug['dropped_release_frames'].append(int(curr_release))

    debug['applied'] = True
    debug['output_count'] = len(kept)
    debug['reason'] = 'front_mode_echo_suppression'
    return kept, debug


def _bbox_area(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def _expand_bbox(bbox, scale):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * float(scale)
    h = (y2 - y1) * float(scale)
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _point_in_bbox(x, y, bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x1 <= float(x) <= x2 and y1 <= float(y) <= y2


def _get_best_rim_bbox_for_frame(frame_idx, yolo_detections, stable_rim_bbox):
    if yolo_detections:
        frame_det = yolo_detections.get(int(frame_idx), {})
        rims = frame_det.get('rims', []) if frame_det else []
        if rims:
            best_rim = max(rims, key=lambda r: float(r.get('confidence', 0.0)))
            bbox = best_rim.get('bbox')
            if bbox and len(bbox) == 4:
                return [float(v) for v in bbox]
    if stable_rim_bbox and len(stable_rim_bbox) == 4:
        return [float(v) for v in stable_rim_bbox]
    return None


def _detections_from_yolo_balls(yolo_detections, frame_idx):
    frame_det = yolo_detections.get(int(frame_idx), {}) if yolo_detections else {}
    balls = frame_det.get('balls', []) if frame_det else []
    boxes = []
    confidences = []
    for ball in balls:
        class_id = ball.get('class_id')
        if class_id is not None:
            try:
                class_id = int(class_id)
            except (TypeError, ValueError):
                class_id = None
            if class_id is not None and class_id not in (0, 1):
                continue
        bbox = ball.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        confidences.append(float(ball.get('confidence', 0.0)))
    return _detections_from_boxes(boxes, confidences)


def _is_not_clearly_moving(center_x, center_y, recent_centers, frame_shape):
    if not recent_centers:
        return False
    h = float(frame_shape[0]) if frame_shape is not None else 0.0
    w = float(frame_shape[1]) if frame_shape is not None else 0.0
    move_threshold = max(6.0, 0.005 * max(1.0, w, h))
    px, py = recent_centers[-1]
    dist = math.sqrt((float(center_x) - float(px)) ** 2 + (float(center_y) - float(py)) ** 2)
    return dist < move_threshold


def _exclude_rim_like_ball_detections(detections, rim_bbox, recent_centers=None, frame_shape=None):
    if detections is None or len(detections) == 0 or rim_bbox is None:
        return detections, {
            'excluded_count': 0,
            'excluded_overlap': 0,
            'excluded_size_near_rim': 0,
            'excluded_stationary_in_rim': 0
        }

    kept = []
    excluded_overlap = 0
    excluded_size_near_rim = 0
    excluded_stationary_in_rim = 0
    rim_area = _bbox_area(rim_bbox)
    expanded_rim_bbox = _expand_bbox(rim_bbox, 1.5)

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        cand_bbox = [float(x1), float(y1), float(x2), float(y2)]
        det_area = _bbox_area(cand_bbox)
        if det_area <= 0:
            continue

        cx = (cand_bbox[0] + cand_bbox[2]) / 2.0
        cy = (cand_bbox[1] + cand_bbox[3]) / 2.0
        overlap_frac = _bbox_intersection_area(cand_bbox, rim_bbox) / det_area
        size_similar_to_rim = False
        if rim_area > 0:
            area_ratio = det_area / rim_area
            size_similar_to_rim = 0.70 <= area_ratio <= 1.30

        in_rim = _point_in_bbox(cx, cy, rim_bbox)
        in_expanded_rim = _point_in_bbox(cx, cy, expanded_rim_bbox)
        not_moving = _is_not_clearly_moving(cx, cy, recent_centers or [], frame_shape)

        exclude = False
        if overlap_frac > 0.50:
            exclude = True
            excluded_overlap += 1
        elif in_expanded_rim and size_similar_to_rim:
            exclude = True
            excluded_size_near_rim += 1
        elif in_rim and not_moving:
            exclude = True
            excluded_stationary_in_rim += 1

        if not exclude:
            kept.append(i)

    if not kept:
        filtered = sv.Detections.empty()
    else:
        filtered = detections[kept]

    excluded_count = max(0, len(detections) - len(kept))
    return filtered, {
        'excluded_count': excluded_count,
        'excluded_overlap': excluded_overlap,
        'excluded_size_near_rim': excluded_size_near_rim,
        'excluded_stationary_in_rim': excluded_stationary_in_rim
    }

def process_video_for_ball_tracking(
    video_path,
    frame_stride=1,
    shot_attempts=None,
    fps=None,
    custom_windows=None,
    progress_callback=None,
    collection_context=None,
    yolo_detections=None,
    yolo_rim_position=None
):
    """Track basketball positions across the video."""
    detector = get_detection_model()
    use_yolo_ball_only = bool(yolo_detections)
    if detector.model is None and not use_yolo_ball_only:
        return [], {'processed_frames': 0, 'timeouts': 0, 'errors': 0, 'total_frames': 0, 'tracks': 0}

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tracks = []
    last_progress_update = -1
    timeout_count = 0
    error_count = 0
    processed_frames = 0
    window_frames_processed = 0
    detected_frames = 0
    debug_done = False
    rim_excluded_candidates = 0
    frames_with_rim_exclusion = 0
    valid_non_rim_detection_frames = 0
    tracks_with_rim_exclusion = 0
    rim_exclusion_reasons = defaultdict(int)
    window_diagnostics = []

    release_frames = {
        int(shot.get('release_frame'))
        for shot in (shot_attempts or [])
        if shot.get('release_frame') is not None
    }
    collector = None
    if collection_context and collection_context.get('session_id'):
        collector = TrainingDataCollector(
            collection_context['session_id'],
            video_metadata=collection_context.get('video_metadata')
        )
        if Config.ENABLE_VIDEO_DEDUPLICATION and collector.storage:
            is_dup, video_hash = collector.storage.is_video_processed(
                video_path,
                session_id=collector.session_id
            )
            collector.is_duplicate = is_dup
            collector.video_hash = video_hash

    # Only run ball detection near candidate-centered windows to reduce cost.
    shot_windows = custom_windows or build_ball_windows_from_shots(shot_attempts or [], total_frames, fps)
    if not shot_windows and total_frames > 0 and shot_attempts is None:
        shot_windows = [(0, max(0, total_frames - 1))]
    if not shot_windows:
        cap.release()
        return [], {
            'processed_frames': 0,
            'timeouts': 0,
            'errors': 0,
            'total_frames': total_frames,
            'tracks': 0,
            'coverage': 0.0,
            'valid_non_rim_detection_frames': 0,
            'valid_non_rim_detection_pct': 0.0,
            'rim_exclusion_frames': 0,
            'rim_excluded_candidates': 0,
            'tracks_with_rim_exclusions': 0,
            'rim_exclusion_reasons': {},
            'windows': [],
            'window_diagnostics': [],
            'window_frames': 0,
            'window_count': 0,
            'stride': frame_stride,
            'detection_source': 'yolo_ball_classes' if use_yolo_ball_only else 'local_model',
            'skipped': 'no_shot_windows'
        }
    total_window_frames = sum((end - start + 1) for start, end in shot_windows) if shot_windows else 0

    local_tracker = sv.ByteTrack() if SUPERVISION_AVAILABLE else None
    recent_track_centers = []

    for window_idx, (win_start, win_end) in enumerate(shot_windows):
        window_start_time = time.time()
        print(f"Processing shot window {window_idx + 1}/{len(shot_windows)} (frames {win_start}-{win_end})")
        window_processed_frames = 0
        window_valid_non_rim_frames = 0
        window_rim_exclusion_frames = 0
        window_rim_excluded_candidates = 0
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, win_start)
        frame_idx = win_start
        while frame_idx <= win_end:
            ret, frame = cap.read()
            if not ret:
                break
            if SHOTLAB_DEBUG_BALL and not debug_done:
                debug_done = True
                debug_ball_detection_frame(frame, frame_idx, detector)
            window_frames_processed += 1
            if frame_stride > 1 and frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            processed_frames += 1
            window_processed_frames += 1
            try:
                if use_yolo_ball_only:
                    detections = _detections_from_yolo_balls(yolo_detections, frame_idx)
                else:
                    detections = detect_basketball_local(frame)
            except Exception as e:
                error_count += 1
                print(f"ShotLab ball detection error at frame {frame_idx}: {e}")
                frame_idx += 1
                continue

            rim_bbox = _get_best_rim_bbox_for_frame(frame_idx, yolo_detections, yolo_rim_position)
            detections, exclusion_info = _exclude_rim_like_ball_detections(
                detections,
                rim_bbox,
                recent_centers=recent_track_centers,
                frame_shape=frame.shape if frame is not None else None
            )
            if exclusion_info['excluded_count'] > 0:
                frames_with_rim_exclusion += 1
                rim_excluded_candidates += exclusion_info['excluded_count']
                window_rim_exclusion_frames += 1
                window_rim_excluded_candidates += exclusion_info['excluded_count']
                rim_exclusion_reasons['overlap'] += exclusion_info['excluded_overlap']
                rim_exclusion_reasons['size_near_rim'] += exclusion_info['excluded_size_near_rim']
                rim_exclusion_reasons['stationary_in_rim'] += exclusion_info['excluded_stationary_in_rim']

            if detections is not None and len(detections) > 0:
                detected_frames += 1
                valid_non_rim_detection_frames += 1
                window_valid_non_rim_frames += 1

            tracked = False
            if SUPERVISION_AVAILABLE and local_tracker is not None and detections is not None and len(detections) > 0:
                try:
                    tracked_detections = local_tracker.update_with_detections(detections)
                    if len(tracked_detections) > 0:
                        confs = tracked_detections.confidence if tracked_detections.confidence is not None else None
                        idx = int(np.argmax(confs)) if confs is not None else 0
                        x1, y1, x2, y2 = tracked_detections.xyxy[idx]
                        tracks.append({
                            'frame': frame_idx,
                            'x': float((x1 + x2) / 2.0),
                            'y': float((y1 + y2) / 2.0),
                            'width': float(x2 - x1),
                            'height': float(y2 - y1),
                            'confidence': float(confs[idx]) if confs is not None else None,
                            'tracker_id': int(tracked_detections.tracker_id[idx]) if tracked_detections.tracker_id is not None else None,
                            'rim_excluded_candidates': int(exclusion_info['excluded_count']),
                            'detection_source': 'yolo_balls' if use_yolo_ball_only else 'local_model'
                        })
                        recent_track_centers.append((float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)))
                        if len(recent_track_centers) > 8:
                            recent_track_centers = recent_track_centers[-8:]
                        if exclusion_info['excluded_count'] > 0:
                            tracks_with_rim_exclusion += 1
                        tracked = True
                except Exception:
                    tracked = False

            if not tracked:
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
                        'tracker_id': None,
                        'rim_excluded_candidates': int(exclusion_info['excluded_count']),
                        'detection_source': 'yolo_balls' if use_yolo_ball_only else 'local_model'
                    })
                    recent_track_centers.append((float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)))
                    if len(recent_track_centers) > 8:
                        recent_track_centers = recent_track_centers[-8:]
                    if exclusion_info['excluded_count'] > 0:
                        tracks_with_rim_exclusion += 1
            current_idx = frame_idx
            if collector:
                frame_metadata = {
                    'shot_detected_by_pose': True,
                    'window_index': int(window_idx),
                    'window_start': int(win_start),
                    'window_end': int(win_end),
                    'is_release_point': current_idx in release_frames,
                    'model_type': detector.model_label
                }
                if collector.should_save_frame(current_idx, detections, frame_metadata):
                    collector.save_frame_async(frame, detections, frame_metadata, current_idx)

            frame_idx += 1

            if progress_callback and total_window_frames > 0:
                progress = min(1.0, window_frames_processed / total_window_frames)
                if processed_frames % max(1, 10 * frame_stride) == 0 and progress != last_progress_update:
                    last_progress_update = progress
                    progress_callback(window_frames_processed, total_window_frames, progress)
        elapsed = time.time() - window_start_time
        valid_pct = (float(window_valid_non_rim_frames) / float(window_processed_frames)) if window_processed_frames else 0.0
        window_diag = {
            'window_index': int(window_idx),
            'start_frame': int(win_start),
            'end_frame': int(win_end),
            'processed_frames': int(window_processed_frames),
            'valid_non_rim_detection_frames': int(window_valid_non_rim_frames),
            'valid_non_rim_detection_pct': float(valid_pct),
            'rim_exclusion_frames': int(window_rim_exclusion_frames),
            'rim_excluded_candidates': int(window_rim_excluded_candidates)
        }
        window_diagnostics.append(window_diag)
        print(f"Finished window {window_idx + 1}/{len(shot_windows)} in {elapsed:.1f}s")
        logger.info(
            (
                "Ball tracking window %d (%d-%d): valid_non_rim_pct=%.3f "
                "rim_exclusion_frames=%d rim_excluded_candidates=%d"
            ),
            window_idx + 1,
            win_start,
            win_end,
            valid_pct,
            window_rim_exclusion_frames,
            window_rim_excluded_candidates
        )

    cap.release()
    if collector and collector.storage and not collector.is_duplicate:
        try:
            collector.storage.mark_video_processed(
                video_path,
                video_hash=collector.video_hash,
                session_id=collector.session_id
            )
        except Exception as exc:
            print(f"Warning: unable to mark video processed: {exc}")
    return tracks, {
        'processed_frames': processed_frames,
        'timeouts': timeout_count,
        'errors': error_count,
        'total_frames': total_frames,
        'tracks': len(tracks),
        'coverage': float(detected_frames / processed_frames) if processed_frames else 0.0,
        'valid_non_rim_detection_frames': int(valid_non_rim_detection_frames),
        'valid_non_rim_detection_pct': float(valid_non_rim_detection_frames / processed_frames) if processed_frames else 0.0,
        'rim_exclusion_frames': int(frames_with_rim_exclusion),
        'rim_excluded_candidates': int(rim_excluded_candidates),
        'tracks_with_rim_exclusions': int(tracks_with_rim_exclusion),
        'rim_exclusion_reasons': dict(rim_exclusion_reasons),
        'windows': shot_windows,
        'window_diagnostics': window_diagnostics,
        'window_frames': total_window_frames,
        'window_count': len(shot_windows),
        'stride': frame_stride,
        'detection_source': 'yolo_ball_classes' if use_yolo_ball_only else 'local_model'
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

def mark_shots_with_ball_validation(shots, filtered_shots, fps):
    """Mark shots with a ball_valid flag based on filtered shots proximity."""
    if not shots:
        return shots
    if not filtered_shots or not fps:
        for shot in shots:
            shot['ball_valid'] = False
        return shots

    tolerance = max(1, int(0.2 * fps))
    filtered_releases = [
        int(s.get('release_frame', s.get('start_frame', 0)))
        for s in filtered_shots
    ]
    for shot in shots:
        release = int(shot.get('release_frame', shot.get('start_frame', 0)))
        shot['ball_valid'] = any(abs(release - fr) <= tolerance for fr in filtered_releases)
    return shots

def build_ball_track_index(ball_tracks):
    if not ball_tracks:
        return [], []
    ordered = sorted(ball_tracks, key=lambda t: t.get('frame', 0))
    frames = [int(t.get('frame', 0)) for t in ordered]
    return ordered, frames

def get_tracks_in_window(ordered, frames, start_frame, end_frame):
    if not ordered:
        return []
    start_idx = bisect.bisect_left(frames, int(start_frame))
    end_idx = bisect.bisect_right(frames, int(end_frame))
    return ordered[start_idx:end_idx]

def compute_ball_quality(ball_tracks, start_frame, end_frame, fps, stride=1):
    """Compute a 0-1 quality score for ball tracking in a shot window."""
    if not ball_tracks or fps is None:
        return 0.0
    expected_frames = max(1, int(end_frame - start_frame + 1))
    expected_detections = max(1, int(math.ceil(expected_frames / max(1, stride))))
    coverage = min(1.0, len(ball_tracks) / expected_detections)

    continuity = 0.0
    if len(ball_tracks) > 1:
        frame_gaps = [
            max(1, int(ball_tracks[i]['frame'] - ball_tracks[i - 1]['frame']))
            for i in range(1, len(ball_tracks))
        ]
        avg_gap = float(np.mean(np.array(frame_gaps, dtype=np.float32))) if frame_gaps else 0.0
        avg_gap_norm = avg_gap / max(1.0, float(stride))
        continuity = 1.0 / (1.0 + avg_gap_norm / 3.0)

    smoothness = 0.0
    if len(ball_tracks) >= 3:
        positions = np.array([[t['x'], t['y']] for t in ball_tracks], dtype=np.float32)
        velocities = np.diff(positions, axis=0)
        if len(velocities) >= 2:
            accelerations = np.diff(velocities, axis=0)
            accel_mag = np.linalg.norm(accelerations, axis=1)
            avg_accel = float(np.mean(accel_mag)) if len(accel_mag) else 0.0
            smoothness = 1.0 / (1.0 + avg_accel / 25.0)

    quality = 0.4 * coverage + 0.3 * continuity + 0.3 * smoothness
    return float(max(0.0, min(1.0, quality)))

def classify_shot_outcome_tiered(
    ball_tracks,
    rim_position,
    net_zone,
    shot_start,
    shot_end,
    ball_quality,
    yolo_detections=None,
    yolo_rim_position=None
):
    """Tiered outcome classification with quality-aware fallbacks."""
    shot_frames = list(range(int(shot_start), int(shot_end) + 1))

    yolo_outcome = classify_shot_yolo(shot_frames, yolo_detections, yolo_rim_position)
    logger.info(
        "Tiered classifier YOLO signal (non-final): outcome=%s tracks=%s",
        yolo_outcome,
        len(ball_tracks) if ball_tracks else 0
    )

    if not ball_tracks:
        return 'unknown', {'reason': 'no_ball_tracks', 'trajectory_points': 0, 'tier': 'none', 'confidence': 0.0}

    track_count = len(ball_tracks)

    if net_zone:
        if ball_quality >= 0.6 and track_count >= 8:
            outcome, debug = analyze_shot_outcome_net_zone_with_debug(ball_tracks, net_zone, shot_start, shot_end)
            debug['tier'] = 'net_zone_full'
            debug['confidence'] = debug.get('confidence', 0.75 if outcome != 'unknown' else 0.4)
            return outcome, debug

        if ball_quality >= 0.3 and track_count >= 4:
            outcome, debug = analyze_shot_outcome_net_zone_with_debug(ball_tracks, net_zone, shot_start, shot_end)
            debug['tier'] = 'net_zone_partial'
            if outcome == 'unknown':
                return 'miss', {
                    'reason': 'net_zone_partial_no_signal',
                    'trajectory_points': track_count,
                    'tier': 'net_zone_partial',
                    'confidence': 0.45
                }
            debug['confidence'] = min(0.7, debug.get('confidence', 0.6))
            return outcome, debug

        return 'miss', {
            'reason': 'net_zone_low_quality_default_miss',
            'trajectory_points': track_count,
            'tier': 'net_zone_low',
            'confidence': 0.35
        }

    if not rim_position:
        return 'unknown', {'reason': 'no_rim_position', 'trajectory_points': 0, 'tier': 'none', 'confidence': 0.0}

    if ball_quality >= 0.6 and track_count >= 10:
        outcome, debug = analyze_shot_outcome_with_debug(ball_tracks, rim_position, shot_start, shot_end)
        debug['tier'] = 'full'
        debug['confidence'] = 0.8 if outcome != 'unknown' else 0.4
        return outcome, debug

    if ball_quality >= 0.3 and track_count >= 5:
        rim_center_x = rim_position['x']
        rim_center_y = rim_position['y']
        rim_width = rim_position.get('width', 0)
        rim_height = rim_position.get('height', 0)
        rim_radius_x = rim_width / 2.0 if rim_width else 20.0
        rim_radius_y = rim_height / 2.0 if rim_height else 18.0
        x_tolerance = rim_radius_x * 1.1
        y_tolerance = rim_radius_y * 0.9

        ordered = sorted(ball_tracks, key=lambda t: t['frame'])
        for i in range(len(ordered) - 1):
            curr = ordered[i]
            next_pt = ordered[i + 1]
            if next_pt['y'] <= curr['y']:
                continue
            near_rim = abs(curr['x'] - rim_center_x) <= x_tolerance and abs(curr['y'] - rim_center_y) <= y_tolerance
            if near_rim:
                return 'make', {
                    'reason': 'partial_descending_near_rim',
                    'trajectory_points': track_count,
                    'tier': 'partial',
                    'confidence': 0.6
                }
        return 'miss', {
            'reason': 'partial_no_clear_make',
            'trajectory_points': track_count,
            'tier': 'partial',
            'confidence': 0.5
        }

    # Low-quality tracking: default to miss (to minimize unknowns).
    return 'miss', {
        'reason': 'low_quality_default_miss',
        'trajectory_points': track_count,
        'tier': 'low',
        'confidence': 0.35
    }


def cluster_behind_basket_candidates(shot_attempts, fps):
    shots = sorted(
        [dict(s) for s in (shot_attempts or [])],
        key=lambda s: int(s.get('release_frame', s.get('start_frame', 0)) or 0)
    )
    debug = {
        'applied': False,
        'input_count': len(shots),
        'output_count': len(shots),
        'cluster_gap_frames': 0,
        'clusters': 0,
        'dropped': 0,
    }
    if not shots:
        return shots, debug

    fps_safe = max(1.0, float(fps or 30.0))
    cluster_gap_frames = int(max(1, round(float(SHOTLAB_BEHIND_BASKET_MIN_GAP_SECONDS) * fps_safe)))
    debug['cluster_gap_frames'] = int(cluster_gap_frames)

    clusters = []
    current = [shots[0]]
    prev_release = int(shots[0].get('release_frame', shots[0].get('start_frame', 0)) or 0)
    for shot in shots[1:]:
        release = int(shot.get('release_frame', shot.get('start_frame', 0)) or 0)
        if (release - prev_release) <= cluster_gap_frames:
            current.append(shot)
        else:
            clusters.append(current)
            current = [shot]
        prev_release = int(release)
    if current:
        clusters.append(current)

    debug['clusters'] = int(len(clusters))

    def _rank_key(shot):
        source = str(shot.get('source') or 'pose')
        source_priority = 2.0 if source == 'yolo_launch' else (1.0 if source == 'pose' else 0.0)
        shoot_confirmed = 1 if bool(shot.get('shoot_model_confirmed', False)) else 0
        shoot_conf = float(shot.get('shoot_model_confidence', 0.0) or 0.0)
        launch_support = float(shot.get('ball_first_support', 0.0) or 0.0)
        launch_score = float(shot.get('ball_first_score', 0.0) or 0.0)
        shooter_prox_evidence = float(shot.get('shooter_proximity_evidence', 0.0) or 0.0)
        release = int(shot.get('release_frame', shot.get('start_frame', 0)) or 0)
        return (
            int(shoot_confirmed),
            float(launch_support),
            float(shooter_prox_evidence),
            float(launch_score),
            float(shoot_conf),
            float(source_priority),
            -int(release),
        )

    clustered = []
    for cluster in clusters:
        best = max(cluster, key=_rank_key)
        clustered.append(best)
    clustered.sort(key=lambda s: int(s.get('release_frame', s.get('start_frame', 0)) or 0))

    debug['applied'] = True
    debug['output_count'] = int(len(clustered))
    debug['dropped'] = int(max(0, len(shots) - len(clustered)))
    return clustered, debug

def dedupe_shots_by_release(shots, fps, min_gap_seconds=SHOTLAB_SHOT_DEDUPE_SECONDS, camera_position=None):
    """Remove near-duplicate shots based on release frame proximity."""
    debug = {
        'kept': 0,
        'dropped': 0,
        'replaced': 0
    }
    if not shots or not fps:
        debug['kept'] = len(shots) if shots else 0
        return shots or [], debug

    gap_seconds = float(min_gap_seconds)
    if camera_position and str(camera_position).startswith('sideline'):
        gap_seconds = max(0.55, gap_seconds * max(0.25, SHOTLAB_SIDELINE_MIN_GAP_MULT))
    if str(camera_position or '') == 'in_front_of_shooter':
        gap_seconds = max(gap_seconds, float(SHOTLAB_BEHIND_BASKET_MIN_GAP_SECONDS))
    min_gap_frames = max(1, int(gap_seconds * fps))
    ordered = sorted(shots, key=lambda s: int(s.get('release_frame', s.get('start_frame', 0))))
    deduped = []
    for shot in ordered:
        release = int(shot.get('release_frame', shot.get('start_frame', 0)))
        duration = int(shot.get('end_frame', release)) - int(shot.get('start_frame', release))
        if not deduped:
            deduped.append(shot)
            continue
        prev = deduped[-1]
        prev_release = int(prev.get('release_frame', prev.get('start_frame', 0)))
        if release - prev_release < min_gap_frames:
            shot_confirmed = bool(shot.get('shoot_model_confirmed', False))
            prev_confirmed = bool(prev.get('shoot_model_confirmed', False))
            if shot_confirmed and prev_confirmed:
                if str(camera_position or '') == 'in_front_of_shooter':
                    prev_conf = float(prev.get('shoot_model_confidence', 0.0) or 0.0)
                    curr_conf = float(shot.get('shoot_model_confidence', 0.0) or 0.0)
                    if curr_conf > prev_conf:
                        deduped[-1] = shot
                        debug['replaced'] += 1
                    else:
                        debug['dropped'] += 1
                else:
                    deduped.append(shot)
                continue
            if shot_confirmed and not prev_confirmed:
                deduped[-1] = shot
                debug['replaced'] += 1
                continue
            if prev_confirmed and not shot_confirmed:
                debug['dropped'] += 1
                continue
            if str(camera_position or '') == 'in_front_of_shooter':
                prev_support = float(prev.get('ball_first_support', 0.0) or 0.0)
                curr_support = float(shot.get('ball_first_support', 0.0) or 0.0)
                prev_score = float(prev.get('ball_first_score', 0.0) or 0.0)
                curr_score = float(shot.get('ball_first_score', 0.0) or 0.0)
                if (curr_support > prev_support) or (
                    curr_support == prev_support and curr_score > prev_score
                ):
                    deduped[-1] = shot
                    debug['replaced'] += 1
                    continue
                if (curr_support == prev_support) and (curr_score == prev_score):
                    # In behind-basket mode, later release is usually closer to true release.
                    if int(release) > int(prev_release):
                        deduped[-1] = shot
                        debug['replaced'] += 1
                    else:
                        debug['dropped'] += 1
                    continue
            prev_duration = int(prev.get('end_frame', prev_release)) - int(prev.get('start_frame', prev_release))
            # Keep the longer clip as the representative shot.
            if duration > prev_duration:
                deduped[-1] = shot
                debug['replaced'] += 1
            else:
                debug['dropped'] += 1
            continue
        deduped.append(shot)

    debug['kept'] = len(deduped)
    return deduped, debug

def detect_rim_brightness_fallback_legacy(frame):
    """Brightness-based fallback to approximate rim position."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        roi_top = int(height * 0.1)
        roi_bottom = int(height * 0.6)
        roi = gray[roi_top:roi_bottom, :]
        _, bright_mask = cv2.threshold(roi, 178, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < 200 or area > 20000:
            return None
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        y += roi_top
        if radius < 10 or radius > 150:
            return None
        return {
            'x': float(x),
            'y': float(y),
            'width': float(radius * 2.0),
            'height': float(radius * 2.0),
            'confidence': 0.2,
            'source': 'brightness'
        }
    except Exception:
        return None


def detect_rim(frame):
    """Detect rim position in a given frame."""
    try:
        detector = get_detection_model()
        rim = detector.detect_rim(frame)
        if rim and rim.get('confidence', 0.0) >= SHOTLAB_RIM_MIN_CONFIDENCE:
            return rim
        fallback = detect_rim_brightness_fallback(frame)
        if isinstance(fallback, tuple) and len(fallback) >= 4:
            found, x, y, radius = fallback[:4]
            if found:
                fallback = {
                    'x': float(x),
                    'y': float(y),
                    'width': float(radius * 2.0),
                    'height': float(radius * 2.0),
                    'confidence': 0.2,
                    'source': 'brightness'
                }
            else:
                fallback = None
        return fallback or rim
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
        rim_source = rim_position.get('source', 'auto')
    elif best:
        rim_position = None
        rim_confidence = best['rim'].get('confidence', 0.0)
        rim_source = best['rim'].get('source', 'auto')

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

def manual_net_zone_from_request(form, frame_shape):
    """Convert manual net zone selection (normalized coords) into pixel bounds."""
    if frame_shape is None or not form:
        return None
    try:
        x1_norm = form.get('net_x1_norm') or form.get('net_zone_x1_norm')
        y1_norm = form.get('net_y1_norm') or form.get('net_zone_y1_norm')
        x2_norm = form.get('net_x2_norm') or form.get('net_zone_x2_norm')
        y2_norm = form.get('net_y2_norm') or form.get('net_zone_y2_norm')
        if x1_norm is None or y1_norm is None or x2_norm is None or y2_norm is None:
            return None
        x1_norm = float(x1_norm)
        y1_norm = float(y1_norm)
        x2_norm = float(x2_norm)
        y2_norm = float(y2_norm)
        h, w = frame_shape[:2]
        x1_px = float(max(0.0, min(1.0, x1_norm)) * w)
        y1_px = float(max(0.0, min(1.0, y1_norm)) * h)
        x2_px = float(max(0.0, min(1.0, x2_norm)) * w)
        y2_px = float(max(0.0, min(1.0, y2_norm)) * h)
        left = min(x1_px, x2_px)
        right = max(x1_px, x2_px)
        top = min(y1_px, y2_px)
        bottom = max(y1_px, y2_px)
        if right - left < 4 or bottom - top < 4:
            return None
        return {
            'x1': left,
            'y1': top,
            'x2': right,
            'y2': bottom,
            'confidence': 1.0,
            'source': 'manual'
        }
    except Exception:
        return None

def normalize_net_zone(net_zone, frame_shape):
    """Normalize pixel net zone bounds to 0-1 for session storage."""
    if not net_zone or frame_shape is None:
        return None
    h, w = frame_shape[:2]
    if w <= 0 or h <= 0:
        return None
    x1 = float(net_zone.get('x1', 0))
    y1 = float(net_zone.get('y1', 0))
    x2 = float(net_zone.get('x2', 0))
    y2 = float(net_zone.get('y2', 0))
    return {
        'x1_norm': max(0.0, min(1.0, x1 / w)),
        'y1_norm': max(0.0, min(1.0, y1 / h)),
        'x2_norm': max(0.0, min(1.0, x2 / w)),
        'y2_norm': max(0.0, min(1.0, y2 / h))
    }

def normalize_rim_zone(rim_zone, frame_shape):
    """Normalize pixel rim zone (x,y,width,height) to 0-1."""
    if not rim_zone or frame_shape is None:
        return None
    h, w = frame_shape[:2]
    if w <= 0 or h <= 0:
        return None
    x = float(rim_zone.get('x', 0))
    y = float(rim_zone.get('y', 0))
    width = float(rim_zone.get('width', 0))
    height = float(rim_zone.get('height', 0))
    return {
        'x_norm': max(0.0, min(1.0, x / w)),
        'y_norm': max(0.0, min(1.0, y / h)),
        'width_norm': max(0.0, min(1.0, width / w)),
        'height_norm': max(0.0, min(1.0, height / h)),
        'confidence': rim_zone.get('confidence'),
        'method': rim_zone.get('method')
    }

def denormalize_rim_zone(rim_zone_norm, frame_shape):
    """Convert normalized rim zone to pixel bounds (top-left, width/height)."""
    if not rim_zone_norm or frame_shape is None:
        return None
    h, w = frame_shape[:2]
    if w <= 0 or h <= 0:
        return None
    x = float(rim_zone_norm.get('x_norm', rim_zone_norm.get('x', 0))) * (w if rim_zone_norm.get('x_norm') is not None else 1)
    y = float(rim_zone_norm.get('y_norm', rim_zone_norm.get('y', 0))) * (h if rim_zone_norm.get('y_norm') is not None else 1)
    width = float(rim_zone_norm.get('width_norm', rim_zone_norm.get('width', 0))) * (w if rim_zone_norm.get('width_norm') is not None else 1)
    height = float(rim_zone_norm.get('height_norm', rim_zone_norm.get('height', 0))) * (h if rim_zone_norm.get('height_norm') is not None else 1)
    return {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
        'confidence': rim_zone_norm.get('confidence'),
        'method': rim_zone_norm.get('method')
    }

def denormalize_net_zone(net_zone_norm, frame_shape):
    """Convert normalized net zone bounds into pixel bounds."""
    if not net_zone_norm or frame_shape is None:
        return None
    h, w = frame_shape[:2]
    if w <= 0 or h <= 0:
        return None
    x1 = float(net_zone_norm.get('x1_norm', net_zone_norm.get('x1', 0)))
    y1 = float(net_zone_norm.get('y1_norm', net_zone_norm.get('y1', 0)))
    x2 = float(net_zone_norm.get('x2_norm', net_zone_norm.get('x2', 0)))
    y2 = float(net_zone_norm.get('y2_norm', net_zone_norm.get('y2', 0)))
    if max(x1, y1, x2, y2) > 1.0:
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        return {'x1': left, 'y1': top, 'x2': right, 'y2': bottom}
    left = min(x1, x2) * w
    right = max(x1, x2) * w
    top = min(y1, y2) * h
    bottom = max(y1, y2) * h
    return {'x1': left, 'y1': top, 'x2': right, 'y2': bottom}

def build_net_zone_from_rim(rim_position, frame_shape):
    """Derive a net zone rectangle from a detected rim."""
    if rim_position is None or frame_shape is None:
        return None
    h, w = frame_shape[:2]
    rim_x = float(rim_position.get('x', 0))
    rim_y = float(rim_position.get('y', 0))
    rim_w = float(rim_position.get('width', 0) or 0)
    rim_h = float(rim_position.get('height', 0) or 0)
    rim_radius_x = rim_w / 2.0 if rim_w else 20.0
    rim_radius_y = rim_h / 2.0 if rim_h else 18.0

    left = rim_x - rim_radius_x * 1.4
    right = rim_x + rim_radius_x * 1.4
    top = rim_y - rim_radius_y * 0.15
    bottom = rim_y + rim_radius_y * 4.2

    left = max(0.0, min(left, w - 1))
    right = max(0.0, min(right, w - 1))
    top = max(0.0, min(top, h - 1))
    bottom = max(0.0, min(bottom, h - 1))
    if right - left < 4 or bottom - top < 4:
        return None
    return {'x1': left, 'y1': top, 'x2': right, 'y2': bottom, 'source': 'rim_auto'}

def rim_position_to_zone(rim_position):
    """Convert rim center/size into a top-left rim box."""
    if rim_position is None:
        return None
    x = float(rim_position.get('x', 0))
    y = float(rim_position.get('y', 0))
    width = float(rim_position.get('width', 0) or 0)
    height = float(rim_position.get('height', 0) or 0)
    if width <= 0 or height <= 0:
        return None
    return {
        'x': x - width / 2.0,
        'y': y - height / 2.0,
        'width': width,
        'height': height,
        'confidence': rim_position.get('confidence'),
        'method': rim_position.get('source')
    }

def rim_zone_to_position(rim_zone):
    """Convert rim box into center-based rim position."""
    if rim_zone is None:
        return None
    x = float(rim_zone.get('x', 0))
    y = float(rim_zone.get('y', 0))
    width = float(rim_zone.get('width', 0) or 0)
    height = float(rim_zone.get('height', 0) or 0)
    if width <= 0 or height <= 0:
        return None
    return {
        'x': x + width / 2.0,
        'y': y + height / 2.0,
        'width': width,
        'height': height,
        'confidence': rim_zone.get('confidence'),
        'source': rim_zone.get('method', 'manual')
    }

def rim_zone_from_net_zone(net_zone):
    """Approximate rim box from the top of the net zone."""
    if not net_zone:
        return None
    left = float(net_zone.get('x1', 0))
    right = float(net_zone.get('x2', 0))
    top = float(net_zone.get('y1', 0))
    bottom = float(net_zone.get('y2', 0))
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    zone_width = max(1.0, right - left)
    zone_height = max(1.0, bottom - top)
    rim_width = max(4.0, zone_width * SHOTLAB_RIM_FALLBACK_WIDTH_RATIO)
    rim_height = max(3.0, zone_width * SHOTLAB_RIM_FALLBACK_HEIGHT_RATIO)
    center_x = (left + right) / 2.0
    rim_x = center_x - rim_width / 2.0
    rim_y = top - rim_height * SHOTLAB_RIM_FALLBACK_Y_OFFSET_RATIO
    return {
        'x': rim_x,
        'y': rim_y,
        'width': rim_width,
        'height': rim_height,
        'confidence': 0.2,
        'method': 'net_zone_fallback'
    }

def yolo_bbox_to_rim_zone(rim_bbox):
    """Convert YOLO rim bbox [x1,y1,x2,y2] into rim-zone format."""
    if not rim_bbox or len(rim_bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in rim_bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        'x': x1,
        'y': y1,
        'width': x2 - x1,
        'height': y2 - y1,
        'confidence': 1.0,
        'method': 'yolo_stable'
    }

def yolo_bbox_to_rim_position(rim_bbox):
    """Convert YOLO rim bbox [x1,y1,x2,y2] into center-based rim position."""
    zone = yolo_bbox_to_rim_zone(rim_bbox)
    if zone is None:
        return None
    return rim_zone_to_position(zone)


def apply_camera_position_fallback(camera_info, frame_width, rim_bbox=None):
    """
    If camera estimation is unknown, fall back to basket X-position heuristic.
    """
    info = dict(camera_info or {})
    position = str(info.get('position', 'unknown') or 'unknown')
    if position != 'unknown':
        return info

    if not frame_width or not rim_bbox or len(rim_bbox) < 4:
        return info

    rim_x1, _, rim_x2, _ = [float(v) for v in rim_bbox[:4]]
    basket_x = (rim_x1 + rim_x2) / 2.0
    basket_x_pct = basket_x / float(frame_width)

    if 0.35 < basket_x_pct < 0.65:
        fallback_position = 'behind_shooter'
        rim_axis = 'depth'
    else:
        fallback_position = 'sideline_left' if basket_x_pct < 0.5 else 'sideline_right'
        rim_axis = 'horizontal'

    info['position'] = fallback_position
    info['confidence'] = max(float(info.get('confidence', 0.0) or 0.0), 0.55)
    info['rim_axis'] = rim_axis
    info['fallback'] = 'basket_x_heuristic'
    info['basket_x_pct'] = float(basket_x_pct)
    logger.info(
        "Camera fallback applied: %s (basket_x=%.1f%%, confidence=%.2f)",
        fallback_position,
        basket_x_pct * 100.0,
        info['confidence']
    )
    return info

def classify_shot_yolo(shot_frames, yolo_detections, rim_position):
    """
    Classify a detected shot as make/miss using YOLO ball-to-rim analysis.

    Returns: 'make', 'miss', or 'uncertain'
    """
    if not shot_frames or not yolo_detections or rim_position is None:
        return 'uncertain'

    start_frame = int(min(shot_frames))
    end_frame = int(max(shot_frames))
    ball_positions = interpolate_ball_positions(yolo_detections, start_frame, end_frame)

    if len(ball_positions) < 3:
        return 'uncertain'

    rim_x1, rim_y1, rim_x2, rim_y2 = [float(v) for v in rim_position]
    rim_width = max(1.0, rim_x2 - rim_x1)
    margin = rim_width * 0.3

    ball_above_rim = 0
    ball_inside_rim = 0
    ball_below_rim_after_inside = 0
    entered_rim_zone = False

    for fn in sorted(ball_positions.keys()):
        bcx, bcy = ball_positions[fn]
        in_rim_x = (rim_x1 - margin) < bcx < (rim_x2 + margin)

        if in_rim_x:
            if bcy < rim_y1:
                ball_above_rim += 1
            elif rim_y1 <= bcy <= rim_y2:
                ball_inside_rim += 1
                entered_rim_zone = True
            elif bcy > rim_y2 and entered_rim_zone:
                ball_below_rim_after_inside += 1

    if entered_rim_zone and ball_below_rim_after_inside >= 2:
        return 'make'
    if entered_rim_zone and ball_inside_rim >= 3:
        return 'make'
    if ball_above_rim > 0 and not entered_rim_zone:
        return 'miss'
    return 'uncertain'

def is_likely_rebound_yolo(shot_frames, yolo_detections, rim_position):
    """Rebound heuristic from YOLO: ball starts near rim and moves away."""
    if not shot_frames or not yolo_detections or rim_position is None:
        return False

    rim_cx = (rim_position[0] + rim_position[2]) / 2.0
    rim_cy = (rim_position[1] + rim_position[3]) / 2.0
    rim_w = max(1.0, float(rim_position[2] - rim_position[0]))
    rim_h = max(1.0, float(rim_position[3] - rim_position[1]))
    tight_x_tol = (rim_w / 2.0) * 1.2
    tight_y_tol = (rim_h / 2.0) * 1.2

    ordered_frames = sorted(int(f) for f in shot_frames)
    early_frames = ordered_frames[:5]
    late_frames = ordered_frames[-5:]

    early_positions = []
    early_distances = []
    for fn in early_frames:
        ball = get_ball_position_at_frame(yolo_detections, fn)
        if ball:
            bx, by = float(ball['center'][0]), float(ball['center'][1])
            early_positions.append((bx, by))
            dx = ball['center'][0] - rim_cx
            dy = ball['center'][1] - rim_cy
            early_distances.append((dx ** 2 + dy ** 2) ** 0.5)

    late_distances = []
    for fn in late_frames:
        ball = get_ball_position_at_frame(yolo_detections, fn)
        if ball:
            dx = ball['center'][0] - rim_cx
            dy = ball['center'][1] - rim_cy
            late_distances.append((dx ** 2 + dy ** 2) ** 0.5)

    if early_positions and early_distances and late_distances:
        early_avg_x = sum(p[0] for p in early_positions) / len(early_positions)
        early_avg_y = sum(p[1] for p in early_positions) / len(early_positions)
        starts_in_tight_rim_box = (
            abs(early_avg_x - rim_cx) <= tight_x_tol and
            abs(early_avg_y - rim_cy) <= tight_y_tol
        )
        if not starts_in_tight_rim_box:
            return False

        early_avg = sum(early_distances) / len(early_distances)
        late_avg = sum(late_distances) / len(late_distances)
        return late_avg > early_avg and late_avg >= early_avg * 2.0
    return False

def is_likely_dribble_yolo(shot_frames, yolo_detections):
    """Dribble heuristic from YOLO: ball stays low relative to player bottoms."""
    if not shot_frames or not yolo_detections:
        return False

    ball_ys = []
    player_bottoms = []

    for fn in sorted(int(f) for f in shot_frames):
        ball = get_ball_position_at_frame(yolo_detections, fn)
        if ball:
            ball_ys.append(ball['center'][1])

        det = yolo_detections.get(fn, {})
        players = det.get('players', [])
        if players:
            biggest = max(players, key=lambda p: p['area'])
            player_bottoms.append(biggest['bbox'][3])

    if ball_ys and player_bottoms:
        avg_ball_y = sum(ball_ys) / len(ball_ys)
        avg_player_bottom = sum(player_bottoms) / len(player_bottoms)
        return avg_ball_y > avg_player_bottom * 0.85
    return False

def interpolate_ball_tracks(points, max_gap=6):
    """Fill short gaps in ball tracks to smooth make/miss inference."""
    if not points:
        return []
    ordered = sorted(points, key=lambda t: t['frame'])
    filled = [ordered[0]]
    for curr, nxt in zip(ordered, ordered[1:]):
        gap = int(nxt['frame'] - curr['frame'])
        if gap > 1 and gap <= max_gap:
            for step in range(1, gap):
                t = step / gap
                filled.append({
                    'frame': int(curr['frame'] + step),
                    'x': curr['x'] + t * (nxt['x'] - curr['x']),
                    'y': curr['y'] + t * (nxt['y'] - curr['y']),
                    'width': (curr.get('width') or 0) + t * ((nxt.get('width') or 0) - (curr.get('width') or 0)),
                    'height': (curr.get('height') or 0) + t * ((nxt.get('height') or 0) - (curr.get('height') or 0))
                })
        filled.append(nxt)
    return filled

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

    ordered = interpolate_ball_tracks(trajectory)

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

    saw_above = False
    saw_below = False
    entered = False
    entry_frame = None
    exited_below = False

    for pt in ordered:
        if pt['y'] < rim_center_y - y_margin:
            saw_above = True
        if pt['y'] > rim_center_y + y_margin:
            saw_below = True
        inside = (
            abs(pt['x'] - rim_center_x) <= inner_x and
            abs(pt['y'] - rim_center_y) <= inner_y
        )
        if inside and saw_above:
            entered = True
            entry_frame = pt['frame']
        if entered and pt['y'] > rim_center_y + y_margin and abs(pt['x'] - rim_center_x) <= x_tolerance:
            if entry_frame is None or pt['frame'] > entry_frame:
                exited_below = True
                break

    if entered and exited_below:
        return 'make', {
            'reason': 'ball_entered_and_exited_below',
            'trajectory_points': len(ordered)
        }

    # Fallback make: clear center-line crossing while descending.
    if saw_above and saw_below:
        for i in range(len(ordered) - 1):
            curr = ordered[i]
            next_pt = ordered[i + 1]
            dy = next_pt['y'] - curr['y']
            if dy <= 0:
                continue
            if curr['y'] <= rim_center_y - y_margin and next_pt['y'] >= rim_center_y + y_margin:
                t = (rim_center_y - curr['y']) / dy if dy != 0 else 0.0
                x_at = curr['x'] + t * (next_pt['x'] - curr['x'])
                if abs(x_at - rim_center_x) <= x_tolerance * 1.15:
                    return 'make', {
                        'reason': 'ball_crossed_center_descending',
                        'trajectory_points': len(ordered)
                    }

    # Anything that doesn't show a clear enter+exit below is treated as miss.
    return 'miss', {'reason': 'no_clear_make_signal', 'trajectory_points': len(ordered)}

def analyze_shot_outcome_net_zone_with_debug(ball_tracks, net_zone, shot_frame_start, shot_frame_end):
    """Determine make/miss using a net zone rectangle with debug info."""
    if not ball_tracks:
        return 'unknown', {'reason': 'no_ball_tracks', 'trajectory_points': 0}
    if not net_zone:
        return 'unknown', {'reason': 'no_net_zone', 'trajectory_points': 0}

    trajectory = [
        track for track in ball_tracks
        if shot_frame_start <= track['frame'] <= shot_frame_end
    ]
    if len(trajectory) < 2:
        return 'unknown', {'reason': 'insufficient_ball_points', 'trajectory_points': len(trajectory)}

    ordered = interpolate_ball_tracks(trajectory)

    left = float(net_zone.get('x1', 0))
    right = float(net_zone.get('x2', 0))
    top = float(net_zone.get('y1', 0))
    bottom = float(net_zone.get('y2', 0))
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top

    zone_width = max(1.0, right - left)
    zone_height = max(1.0, bottom - top)
    pad_x = max(SHOTLAB_NET_ZONE_PAD_MIN_PX, zone_width * SHOTLAB_NET_ZONE_PAD_X)
    pad_y = max(SHOTLAB_NET_ZONE_PAD_MIN_PX, zone_height * SHOTLAB_NET_ZONE_PAD_Y)
    left -= pad_x
    right += pad_x
    top -= pad_y * 0.5
    bottom += pad_y
    zone_width = max(1.0, right - left)
    zone_height = max(1.0, bottom - top)
    zone_center_x = (left + right) / 2.0
    rim_y = top
    net_bottom_y = bottom

    entered_from_top = False
    exited_from_bottom = False
    entered_sideways = False
    max_depth_in_zone = 0.0
    frames_in_zone = 0
    frames_below_rim = 0
    in_zone_x_positions = []

    for i, ball in enumerate(ordered):
        x = float(ball.get('x', 0))
        y = float(ball.get('y', 0))
        in_zone_horizontally = left <= x <= right
        in_zone_vertically = top <= y <= bottom
        in_zone = in_zone_horizontally and in_zone_vertically

        if in_zone:
            frames_in_zone += 1
            in_zone_x_positions.append(x)
            depth = y - rim_y
            if depth > max_depth_in_zone:
                max_depth_in_zone = depth

            if i > 0:
                prev_ball = ordered[i - 1]
                if prev_ball['y'] < rim_y and y >= rim_y and in_zone_horizontally:
                    entered_from_top = True
                if prev_ball['y'] < net_bottom_y and y >= net_bottom_y and in_zone_horizontally:
                    exited_from_bottom = True

        if y > rim_y:
            frames_below_rim += 1

        if i > 0 and in_zone_vertically and not in_zone_horizontally:
            prev_ball = ordered[i - 1]
            if abs(prev_ball['x'] - zone_center_x) > zone_width * 0.6:
                entered_sideways = True

    make_score = 0.0
    miss_score = 0.0

    if entered_from_top and exited_from_bottom:
        make_score += 0.7
        if in_zone_x_positions:
            avg_x_in_zone = float(np.mean(np.array(in_zone_x_positions, dtype=np.float32)))
            if abs(avg_x_in_zone - zone_center_x) < zone_width * 0.3:
                make_score += 0.2
    elif entered_from_top and max_depth_in_zone > zone_height * 0.5:
        make_score += 0.6

    if frames_in_zone >= 5:
        make_score += 0.3
    elif frames_in_zone >= 3:
        make_score += 0.2

    if frames_below_rim > 5 and frames_in_zone == 0:
        miss_score += 0.6

    if entered_sideways:
        miss_score += 0.4

    closest_approach = min((abs(float(b.get('x', 0)) - zone_center_x) for b in ordered), default=999.0)
    if closest_approach < zone_width * 1.5 and frames_in_zone == 0:
        miss_score += 0.5

    near_rim_points = [b for b in ordered if abs(float(b.get('y', 0)) - rim_y) < zone_width]
    if len(near_rim_points) >= 2:
        y_positions = [float(b.get('y', 0)) for b in near_rim_points]
        descending = all(y_positions[i] < y_positions[i + 1] for i in range(len(y_positions) - 1))
        if descending and frames_in_zone > 0:
            make_score += 0.2
        elif descending and frames_in_zone == 0:
            miss_score += 0.3

    outcome = 'unknown'
    confidence = max(make_score, miss_score)
    if make_score > 0.7:
        outcome = 'make'
        confidence = min(0.95, make_score)
    elif make_score > 0.5 and make_score > miss_score * 1.3:
        outcome = 'make'
        confidence = make_score * 0.8
    elif miss_score > 0.6:
        outcome = 'miss'
        confidence = min(0.9, miss_score)
    elif miss_score > 0.4 and miss_score > make_score * 1.3:
        outcome = 'miss'
        confidence = miss_score * 0.8
    else:
        if frames_in_zone > 0 and entered_from_top:
            outcome = 'make'
            confidence = max(confidence, 0.4)
        elif frames_below_rim > 5:
            outcome = 'miss'
            confidence = max(confidence, 0.4)

    if outcome == 'make':
        if entered_from_top and exited_from_bottom:
            reason = 'net_zone_enter_exit'
        elif entered_from_top and max_depth_in_zone > zone_height * 0.5:
            reason = 'net_zone_deep_entry'
        elif frames_in_zone >= 3:
            reason = 'net_zone_in_zone'
        else:
            reason = 'net_zone_weak_make'
    elif outcome == 'miss':
        if frames_below_rim > 5 and frames_in_zone == 0:
            reason = 'net_zone_missed_below'
        elif entered_sideways:
            reason = 'net_zone_side_entry'
        elif closest_approach < zone_width * 1.5 and frames_in_zone == 0:
            reason = 'net_zone_close_miss'
        else:
            reason = 'net_zone_no_clear_make'
    else:
        reason = 'net_zone_weak_signal'

    return outcome, {
        'reason': reason,
        'trajectory_points': len(ordered),
        'confidence': float(max(0.0, min(0.95, confidence))),
        'frames_in_zone': int(frames_in_zone),
        'entered_from_top': bool(entered_from_top),
        'exited_from_bottom': bool(exited_from_bottom),
        'closest_approach': float(closest_approach),
        'zone_width': float(zone_width),
        'zone_height': float(zone_height),
        'pad_x': float(pad_x),
        'pad_y': float(pad_y)
    }

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

def save_rejected_shot_debug(video_path, dropped_shots):
    if not video_path or not dropped_shots:
        return
    try:
        debug_dir = project_root / 'debug' / 'rejected_shots'
        debug_dir.mkdir(parents=True, exist_ok=True)
        indices = [
            int(s.get('release_frame', 0))
            for s in dropped_shots
            if s.get('release_frame') is not None
        ]
        frames = fetch_frames_by_index(video_path, indices)
        for idx, shot in enumerate(dropped_shots[:20], start=1):
            frame_idx = int(shot.get('release_frame', 0))
            frame = frames.get(frame_idx)
            if frame is None:
                continue
            reason = shot.get('reason', 'unknown')
            label = f"REJECTED: {reason}"
            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
            out_path = debug_dir / f"shot_{idx}_frame_{frame_idx}_{reason}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"Saved rejected shot debug: {out_path}")
    except Exception as exc:
        print(f"Rejected shot debug save failed: {exc}")

def register_shotlab_session(video_path, fps, shot_attempts, rim_position=None, net_zone=None, ball_tracks=None):
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
        'video_name': os.path.basename(video_path) if video_path else None,
        'fps': float(fps or 0.0),
        'shots': shot_attempts,
        'shots_analysis': None,
        'total_frames': total_frames,
        'rim_position': rim_position,
        'net_zone': net_zone,
        'ball_tracks': ball_tracks,
        'ball_tracks_by_frame': None,
        'created_at': time.time()
    }
    _save_shotlab_session(session_id, shotlab_sessions[session_id])
    return session_id

def extract_shot_clip(session_id, shot_index):
    """Extract (or reuse) a short clip around a detected shot."""
    session = shotlab_sessions.get(session_id)
    if session is None:
        session = _load_shotlab_session(session_id)
        if session is not None:
            shotlab_sessions[session_id] = session
    if session is None:
        raise ValueError('Invalid session_id')
    shots = session.get('shots') or []
    if shot_index < 0 or shot_index >= len(shots):
        raise ValueError('Invalid shot_index')

    video_path = session['video_path']
    if video_path and not os.path.exists(video_path):
        alt_path = os.path.join(str(project_root), video_path)
        if os.path.exists(alt_path):
            video_path = alt_path
        else:
            alt_path = os.path.join(os.path.dirname(__file__), video_path)
            if os.path.exists(alt_path):
                video_path = alt_path
    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"Video file not found for session: {session.get('video_path')}")
    fps = session.get('fps') or 30.0
    shot = shots[shot_index]
    before_frames = max(1, int(SHOTLAB_CLIP_BEFORE_SECONDS * fps))
    after_frames = max(1, int(SHOTLAB_CLIP_AFTER_SECONDS * fps))
    release_frame = int(shot.get('release_frame', shot.get('start_frame', 0)))
    start_frame = int(shot.get('clip_start_frame', max(0, release_frame - before_frames)))
    end_frame = int(shot.get('clip_end_frame', release_frame + after_frames))

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
    net_zone = session.get('net_zone') if SHOTLAB_CLIP_OVERLAY else None
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
            if net_zone is not None:
                x1 = int(net_zone.get('x1', 0) * scale)
                y1 = int(net_zone.get('y1', 0) * scale)
                x2 = int(net_zone.get('x2', 0) * scale)
                y2 = int(net_zone.get('y2', 0) * scale)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)
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


def collect_made_shadow_signals(video_path, shot_attempts, fps):
    debug = {
        'enabled': bool(SHOTLAB_EVENT_MADE_SHADOW_ENABLE),
        'class_id': int(SHOTLAB_EVENT_MADE_CLASS_ID),
        'window_frames': int(SHOTLAB_EVENT_MADE_SHADOW_WINDOW_FRAMES),
        'stride': int(max(1, SHOTLAB_EVENT_MADE_SHADOW_STRIDE)),
        'threshold': float(SHOTLAB_EVENT_MADE_SHADOW_MIN_CONFIDENCE),
        'requested_frames': 0,
        'frames_checked': 0,
        'frames_dropped_by_budget': 0,
        'shots_with_signal': 0,
        'max_confidence': 0.0,
        'inference': {},
        'by_shot_index': {},
    }
    if not SHOTLAB_EVENT_MADE_SHADOW_ENABLE:
        debug['reason'] = 'disabled'
        return debug
    if not shot_attempts or not fps:
        debug['reason'] = 'no_shots_or_fps'
        return debug

    window = max(1, int(SHOTLAB_EVENT_MADE_SHADOW_WINDOW_FRAMES))
    stride = max(1, int(SHOTLAB_EVENT_MADE_SHADOW_STRIDE))
    max_frames = int(SHOTLAB_EVENT_MADE_SHADOW_MAX_FRAMES)
    frame_usage = []
    frame_to_shots = defaultdict(set)
    for shot_idx, shot in enumerate(shot_attempts):
        release_frame = int(shot.get('release_frame', shot.get('start_frame', 0)))
        for frame_idx in range(release_frame - window, release_frame + window + 1, stride):
            if frame_idx < 0:
                continue
            frame_usage.append((abs(frame_idx - release_frame), frame_idx, shot_idx))
            frame_to_shots[int(frame_idx)].add(int(shot_idx))

    unique_frames = sorted(set(int(item[1]) for item in frame_usage if int(item[1]) >= 0))
    debug['requested_frames'] = len(unique_frames)
    if max_frames > 0 and len(unique_frames) > max_frames:
        frame_usage_sorted = sorted(frame_usage, key=lambda item: (int(item[0]), int(item[1]), int(item[2])))
        selected = []
        selected_set = set()
        for _, frame_idx, _ in frame_usage_sorted:
            fi = int(frame_idx)
            if fi in selected_set:
                continue
            selected_set.add(fi)
            selected.append(fi)
            if len(selected) >= max_frames:
                break
        unique_frames = sorted(selected)
    debug['frames_checked'] = len(unique_frames)
    debug['frames_dropped_by_budget'] = max(0, debug['requested_frames'] - debug['frames_checked'])

    detections, inference_debug = detect_event_class_on_frames(
        video_path=video_path,
        frame_indices=unique_frames,
        event_class_id=SHOTLAB_EVENT_MADE_CLASS_ID,
        min_confidence=SHOTLAB_EVENT_MADE_SHADOW_MIN_CONFIDENCE,
    )
    debug['inference'] = inference_debug

    per_shot = {}
    for shot_idx, shot in enumerate(shot_attempts):
        release_frame = int(shot.get('release_frame', shot.get('start_frame', 0)))
        best_conf = None
        best_frame = None
        for frame_idx in range(release_frame - window, release_frame + window + 1, stride):
            conf = detections.get(int(frame_idx))
            if conf is None:
                continue
            conf_f = float(conf)
            if best_conf is None or conf_f > best_conf:
                best_conf = conf_f
                best_frame = int(frame_idx)
        if best_conf is None:
            per_shot[int(shot_idx)] = {
                'detected': False,
                'confidence': 0.0,
                'frame': None,
            }
            continue
        per_shot[int(shot_idx)] = {
            'detected': True,
            'confidence': float(best_conf),
            'frame': int(best_frame) if best_frame is not None else None,
        }
        debug['shots_with_signal'] += 1
        if float(best_conf) > float(debug['max_confidence']):
            debug['max_confidence'] = float(best_conf)
    debug['max_confidence'] = round(float(debug['max_confidence']), 4)
    debug['by_shot_index'] = per_shot
    return debug


def build_shotlab_session_summary(shots_analysis):
    shots = list(shots_analysis or [])
    total_shots = len(shots)
    makes = sum(1 for s in shots if str(s.get('outcome', '')).lower() == 'make')
    misses = max(0, total_shots - makes)
    shooting_pct = (100.0 * makes / float(total_shots)) if total_shots > 0 else 0.0

    zones = defaultdict(lambda: {
        'attempts': 0,
        'makes': 0,
        'form_total': 0.0,
        'form_count': 0,
    })
    make_form_scores = []
    miss_form_scores = []
    for shot in shots:
        zone = str(shot.get('zone') or shot.get('shooting_zone') or 'unknown')
        zones[zone]['attempts'] += 1
        if str(shot.get('outcome', '')).lower() == 'make':
            zones[zone]['makes'] += 1
        form_score = shot.get('form_score')
        if form_score is not None:
            form_f = float(form_score)
            zones[zone]['form_total'] += form_f
            zones[zone]['form_count'] += 1
            if str(shot.get('outcome', '')).lower() == 'make':
                make_form_scores.append(form_f)
            else:
                miss_form_scores.append(form_f)

    zone_summary = {}
    for zone, stats in zones.items():
        avg_form = None
        if stats['form_count'] > 0:
            avg_form = round(float(stats['form_total'] / float(stats['form_count'])), 1)
        zone_summary[zone] = {
            'attempts': int(stats['attempts']),
            'makes': int(stats['makes']),
            'avg_form_score': avg_form
        }

    makes_avg = round(float(np.mean(np.array(make_form_scores, dtype=np.float32))), 1) if make_form_scores else None
    misses_avg = round(float(np.mean(np.array(miss_form_scores, dtype=np.float32))), 1) if miss_form_scores else None
    if makes_avg is not None and misses_avg is not None:
        diff = round(float(makes_avg - misses_avg), 1)
        if diff >= 0:
            insight = (
                f"Your makes have {diff:.1f} points higher form score on average. "
                "Keep your release angle and follow-through consistent."
            )
        else:
            insight = (
                f"Your misses currently score {-diff:.1f} points higher than makes. "
                "Review release timing and wrist follow-through consistency."
            )
    else:
        insight = "Collect a few more shots to unlock make-vs-miss form comparison insight."

    return {
        'total_shots': int(total_shots),
        'makes': int(makes),
        'misses': int(misses),
        'shooting_percentage': round(float(shooting_pct), 1),
        'zones': zone_summary,
        'form_comparison': {
            'makes_avg_form_score': makes_avg,
            'misses_avg_form_score': misses_avg,
            'insight': insight,
        }
    }


def _generate_session_coaching_text(shots_analysis, zone_percentages):
    """Generate meaningful overall_consistency and zone_breakdown text from shot data."""
    shots = list(shots_analysis or [])
    check_keys = [
        'elbow_alignment', 'release_height', 'follow_through',
        'base_and_balance', 'shoulder_alignment', 'guide_hand',
    ]
    check_labels = {
        'elbow_alignment': 'Elbow Alignment',
        'release_height': 'Release Height',
        'follow_through': 'Follow-Through',
        'base_and_balance': 'Base and Balance',
        'shoulder_alignment': 'Shoulder Alignment',
        'guide_hand': 'Guide Hand',
    }

    scores = [float(s['shotsync_score']) for s in shots if s.get('shotsync_score') is not None]
    overall_consistency = ''
    if len(scores) >= 2:
        lo, hi = int(round(min(scores))), int(round(max(scores)))
        spread = hi - lo
        if spread <= 12:
            overall_consistency = f'Your form is consistent across shots (ShotSync scores range {lo}\u2013{hi}).'
        elif spread <= 25:
            overall_consistency = f'Form varies moderately shot-to-shot (ShotSync scores range {lo}\u2013{hi}). Building a repeatable motion will help.'
        else:
            overall_consistency = f'Form varies significantly shot-to-shot (ShotSync scores range {lo}\u2013{hi}). Building a repeatable motion should be priority #1.'
        poor_counts = defaultdict(int)
        for s in shots:
            coaching = s.get('coaching') or {}
            for ck in check_keys:
                entry = coaching.get(ck)
                if isinstance(entry, dict) and entry.get('status') in ('poor', 'needs_work'):
                    poor_counts[ck] += 1
        if poor_counts:
            worst = max(poor_counts, key=poor_counts.get)
            overall_consistency += f' Focus on improving your {check_labels.get(worst, worst)}.'
    elif len(scores) == 1:
        overall_consistency = f'Only 1 shot scored (ShotSync {int(round(scores[0]))}). Upload more shots to track consistency.'
    else:
        overall_consistency = 'No ShotSync scores available for this session yet.'

    zone_breakdown = {}
    zone_order = ['left', 'center', 'right']
    for zone in zone_order:
        zp = (zone_percentages or {}).get(zone)
        if not zp or int(zp.get('attempts', 0)) == 0:
            continue
        attempts = int(zp['attempts'])
        makes = int(zp.get('makes', 0))
        zone_shots = [s for s in shots if str(s.get('zone', '')).lower() == zone]
        zone_poor = defaultdict(int)
        for s in zone_shots:
            coaching = s.get('coaching') or {}
            for ck in check_keys:
                entry = coaching.get(ck)
                if isinstance(entry, dict) and entry.get('status') in ('poor', 'needs_work'):
                    zone_poor[ck] += 1
        tip = f'{makes}/{attempts} makes.'
        if zone_poor:
            worst = max(zone_poor, key=zone_poor.get)
            tip += f' {check_labels.get(worst, worst)} is your weakest check from this angle.'
        zone_breakdown[zone] = tip
    return overall_consistency, zone_breakdown


def _get_vote_method_outcome(vote_breakdown, method_name):
    for item in (vote_breakdown or []):
        if not isinstance(item, dict):
            continue
        if str(item.get('method') or '') != str(method_name):
            continue
        return str(item.get('outcome') or 'unknown').lower()
    return 'unknown'


def _apply_low_evidence_default_miss_rule(outcome, outcome_debug, shot_metadata):
    """Default low-evidence outcomes to miss unless shoot-model evidence is strong."""
    vote_breakdown = (
        outcome_debug.get('method_details')
        or outcome_debug.get('vote_breakdown')
        or []
    )
    rim_outcome = _get_vote_method_outcome(vote_breakdown, 'rim_passage')
    net_outcome = _get_vote_method_outcome(vote_breakdown, 'net_zone')
    shoot_conf = float((shot_metadata or {}).get('shoot_model_confidence', 0.0) or 0.0)
    shoot_confirmed = bool((shot_metadata or {}).get('shoot_model_confirmed', False))
    shoot_exempt = bool(shoot_confirmed and shoot_conf >= 0.60)
    low_evidence = (rim_outcome == 'unknown' and net_outcome == 'unknown')

    outcome_debug['low_evidence_default_miss_context'] = {
        'rim_passage': rim_outcome,
        'net_zone': net_outcome,
        'shoot_model_confirmed': shoot_confirmed,
        'shoot_model_confidence': round(float(shoot_conf), 4),
        'shoot_model_exempt': shoot_exempt,
    }
    if low_evidence and not shoot_exempt:
        outcome_debug['low_evidence_default_miss_applied'] = True
        outcome_debug['low_evidence_default_miss_reason'] = 'rim_and_net_unknown_without_strong_shoot_model'
        if str(outcome).lower() != 'miss':
            outcome_debug['reason_pre_override'] = outcome_debug.get('reason')
            outcome_debug['reason'] = 'low_evidence_default_miss'
        return 'miss', outcome_debug

    outcome_debug['low_evidence_default_miss_applied'] = False
    return outcome, outcome_debug


def analyze_shot_attempts_common(
    video_path,
    shot_attempts,
    pose_frames,
    fps,
    ball_tracks,
    rim_position,
    rim_zone,
    net_zone,
    yolo_detections,
    yolo_rim_position,
    transform_matrix,
    transform_meta,
    court_points,
    first_frame,
    camera_mode='auto',
    camera_position='unknown'
):
    update_shotlab_status('analyze', 'Analyzing shots...', 0.86)
    session_id = register_shotlab_session(video_path, fps, shot_attempts, rim_position, net_zone, ball_tracks) if shot_attempts else None
    clip_schedule_start = time.time()
    if session_id and SHOTLAB_CLIP_PREGENERATE and not SHOTLAB_CLIP_DEFER:
        pre_generate_shot_clips_async(session_id)
        clip_mode = 'async_pregenerate'
    elif session_id and SHOTLAB_CLIP_DEFER:
        clip_mode = 'deferred'
    else:
        clip_mode = 'disabled'
    clip_schedule_seconds = float(max(0.0, time.time() - clip_schedule_start))

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

    ball_ordered, ball_frames = build_ball_track_index(ball_tracks)
    ball_quality_counts = {'high': 0, 'mid': 0, 'low': 0}
    classification_methods = defaultdict(int)
    candidate_source_counts = defaultdict(int)
    for candidate in shot_attempts:
        candidate_source_counts[str(candidate.get('source') or 'pose')] += 1
    anchor_source_counts = defaultdict(int)
    unknown_zone_reasons = defaultdict(int)
    rim_zone_box = rim_zone or rim_position_to_zone(rim_position)
    frame_height = int(first_frame.shape[0]) if first_frame is not None else None
    frame_width = int(first_frame.shape[1]) if first_frame is not None else None
    normalized_camera_mode = _normalize_camera_mode(camera_mode)
    behind_basket_mode = _is_behind_basket_mode(normalized_camera_mode)
    if behind_basket_mode:
        form_visibility_thresholds = {
            'shoulder': float(BB_FORM_SCORING_MIN_VISIBILITY),
            'elbow': float(BB_FORM_SCORING_MIN_VISIBILITY),
            'wrist': float(BB_FORM_SCORING_MIN_VISIBILITY),
        }
    else:
        form_visibility_thresholds = {
            'shoulder': float(get_visibility_threshold(camera_position, 'shoulder')),
            'elbow': float(get_visibility_threshold(camera_position, 'elbow')),
            'wrist': float(get_visibility_threshold(camera_position, 'wrist')),
        }

    classifier = (
        IntelligentOutcomeClassifier(
            rim_zone=rim_zone_box,
            net_zone=net_zone,
            court_points=court_points,
            frame_width=frame_width,
            frame_height=frame_height,
            camera_position_override=camera_position,
            camera_mode=normalized_camera_mode,
        )
        if SHOTLAB_USE_INTELLIGENT_OUTCOME else None
    )
    made_shadow_debug = collect_made_shadow_signals(video_path, shot_attempts, fps)
    made_shadow_by_shot = dict(made_shadow_debug.get('by_shot_index') or {})
    skeleton_pose_lookup = _build_pose_frame_lookup(pose_frames) if behind_basket_mode else (None, None)

    for idx, shot in enumerate(shot_attempts):
        shot_started = time.time()
        raw_release_frame = int(shot.get('release_frame', shot['start_frame']))
        refine_tracks = get_tracks_in_window(
            ball_ordered,
            ball_frames,
            raw_release_frame - 18,
            raw_release_frame + 18
        )
        release_refine = refine_release_frame(
            raw_release_frame,
            pose_frames=pose_frames,
            ball_tracks=refine_tracks,
            fps=fps,
            search_radius_frames=18
        )
        release_frame = int(release_refine.get('release_frame_refined', raw_release_frame))
        shoot_model_frame = shot.get('shoot_model_frame')
        shoot_model_conf = float(shot.get('shoot_model_confidence', 0.0) or 0.0)
        if (
            behind_basket_mode
            and bool(shot.get('shoot_model_confirmed', False))
            and shoot_model_frame is not None
            and shoot_model_conf >= float(BB_SHOOT_FRAME_ANCHOR_MIN_CONF)
        ):
            try:
                shoot_frame_i = int(shoot_model_frame)
            except (TypeError, ValueError):
                shoot_frame_i = None
            if shoot_frame_i is not None and abs(int(release_frame) - shoot_frame_i) >= 6:
                release_frame = int(shoot_frame_i)
                refine_signals = dict(release_refine.get('release_refine_signals') or {})
                used_signals = list(refine_signals.get('used_signals') or [])
                if 'shoot_model_frame_anchor' not in used_signals:
                    used_signals.append('shoot_model_frame_anchor')
                refine_signals['used_signals'] = used_signals
                signal_map = dict(refine_signals.get('signals') or {})
                signal_map['shoot_model_frame_anchor'] = {
                    'frame': int(shoot_frame_i),
                    'weight': 0.6,
                    'confidence': round(float(shoot_model_conf), 4),
                }
                refine_signals['signals'] = signal_map
                release_refine['release_refine_signals'] = refine_signals
                release_refine['release_frame_refined'] = int(shoot_frame_i)
                release_refine['release_refine_confidence'] = max(
                    float(release_refine.get('release_refine_confidence', 0.0) or 0.0),
                    min(1.0, 0.55 + 0.35 * float(shoot_model_conf)),
                )
        shot['release_frame_raw'] = int(raw_release_frame)
        shot['release_frame'] = int(release_frame)
        if fps:
            shot['timestamp'] = float(release_frame / fps)

        anchor_pos, anchor_source, anchor_reason = resolve_shot_court_anchor(
            pose_frames,
            release_frame,
            release_search_gap_frames,
            yolo_detections=yolo_detections
        )
        anchor_source_counts[anchor_source] += 1
        court_pos = player_to_court_position(
            anchor_pos,
            transform_matrix,
            transform_meta=transform_meta
        ) if anchor_pos and transform_matrix is not None else None

        zone = 'unknown'
        shooter_x_norm = None
        shooter_y_norm = None
        if behind_basket_mode:
            zone_result = classify_shot_zone_behind_basket(
                anchor_pos,
                frame_width,
                frame_height,
                rim_position=rim_position
            )
            zone = str(zone_result.get('zone') or 'center')
            shooter_x_norm = zone_result.get('shooter_x')
            shooter_y_norm = zone_result.get('shooter_y')
            if anchor_pos is None:
                unknown_zones += 1
                unknown_zone_reasons['behind_basket_no_anchor'] += 1
        elif court_pos is not None:
            zone = classify_shot_zone(float(court_pos[0]), float(court_pos[1]))
            if anchor_pos is not None and frame_width and frame_height:
                shooter_x_norm = max(0.0, min(1.0, float(anchor_pos[0]) / float(frame_width)))
                shooter_y_norm = max(0.0, min(1.0, float(anchor_pos[1]) / float(frame_height)))
        else:
            unknown_zones += 1
            reason = anchor_reason or ('court_transform_unavailable' if transform_matrix is None else 'court_projection_failed')
            unknown_zone_reasons[reason] += 1

        shot_type = 'jump_shot'
        if behind_basket_mode:
            shot_type = classify_shot_type_behind_basket(anchor_pos, rim_zone_box)

        shot_rim = rim_position
        if shot_rim is None and release_frame in release_frames:
            shot_rim = detect_rim(release_frames[release_frame])

        outcome_window_extra = int(SHOTLAB_OUTCOME_WINDOW_AFTER_SECONDS * fps) if fps else 0
        default_outcome_end = release_frame + max(0, outcome_window_extra)
        if behind_basket_mode:
            default_outcome_end = release_frame + int(BB_CLASSIFICATION_AFTER_FRAMES)
        outcome_end_frame = int(max(int(shot['start_frame']) + 1, default_outcome_end))
        classification_window_strategy = (
            f"release_plus_{int(BB_CLASSIFICATION_AFTER_FRAMES)}"
            if behind_basket_mode
            else 'default'
        )

        def _classify_with_end(local_end_frame):
            local_end = int(max(int(shot['start_frame']) + 1, local_end_frame))
            local_shot_frames = list(range(int(shot['start_frame']), int(local_end) + 1))
            local_shot_tracks = get_tracks_in_window(ball_ordered, ball_frames, shot['start_frame'], local_end)
            local_ball_quality = compute_ball_quality(
                local_shot_tracks,
                shot['start_frame'],
                local_end,
                fps,
                stride=SHOTLAB_BALL_STRIDE
            )
            local_tracks_for_classifier = None
            local_yolo_outcome = classify_shot_yolo(local_shot_frames, yolo_detections, yolo_rim_position)
            local_classifier_method = None
            if classifier is not None:
                local_tracks_for_classifier = [
                    {
                        'frame': t.get('frame'),
                        'center': (t.get('x'), t.get('y')),
                        'confidence': t.get('confidence'),
                        'width': t.get('width'),
                        'height': t.get('height')
                    }
                    for t in local_shot_tracks
                ]
                result = classifier.classify_shot(
                    local_tracks_for_classifier,
                    yolo_outcome=local_yolo_outcome
                )
                local_outcome = result.outcome
                local_outcome_debug = {
                    'reason': result.method,
                    'confidence': result.confidence,
                    'camera_position': result.camera_position,
                    'yolo_signal': local_yolo_outcome,
                    'method_details': result.method_details,
                    'details': result.details,
                    'low_evidence': bool((result.details or {}).get('low_evidence', False)),
                    'vote_breakdown': (result.details or {}).get('vote_breakdown', result.method_details),
                }
                local_classifier_method = str(result.method)
            else:
                local_outcome, local_outcome_debug = classify_shot_outcome_tiered(
                    local_shot_tracks,
                    shot_rim,
                    net_zone,
                    shot['start_frame'],
                    local_end,
                    local_ball_quality,
                    yolo_detections=yolo_detections,
                    yolo_rim_position=yolo_rim_position
                )
            local_outcome_debug['ball_quality'] = float(local_ball_quality)
            local_outcome_debug['trajectory_points'] = local_outcome_debug.get('trajectory_points', len(local_shot_tracks))
            return {
                'outcome': local_outcome,
                'outcome_debug': local_outcome_debug,
                'shot_frames': local_shot_frames,
                'shot_tracks': local_shot_tracks,
                'shot_tracks_for_classifier': local_tracks_for_classifier,
                'ball_quality': float(local_ball_quality),
                'yolo_outcome': str(local_yolo_outcome),
                'classifier_method': local_classifier_method,
                'window_end': int(local_end),
            }

        classification_result = _classify_with_end(outcome_end_frame)
        rim_contact_frame = _extract_rim_contact_frame_from_vote_breakdown(
            classification_result['outcome_debug'].get('vote_breakdown')
        )
        if behind_basket_mode and rim_contact_frame is not None:
            bounded_end = int(min(int(outcome_end_frame), int(rim_contact_frame) + 45))
            if bounded_end < int(classification_result['window_end']):
                outcome_end_frame = int(max(int(shot['start_frame']) + 1, bounded_end))
                classification_window_strategy = 'rim_contact_capped'
                classification_result = _classify_with_end(outcome_end_frame)
            else:
                outcome_end_frame = int(classification_result['window_end'])
        else:
            outcome_end_frame = int(classification_result['window_end'])

        outcome = classification_result['outcome']
        outcome_debug = classification_result['outcome_debug']
        shot_frames = classification_result['shot_frames']
        shot_tracks = classification_result['shot_tracks']
        shot_tracks_for_classifier = classification_result['shot_tracks_for_classifier']
        ball_quality = float(classification_result['ball_quality'])
        yolo_outcome = classification_result['yolo_outcome']
        if behind_basket_mode:
            outcome, outcome_debug = _apply_low_evidence_default_miss_rule(
                outcome,
                outcome_debug,
                shot,
            )

        if ball_quality >= 0.6:
            ball_quality_counts['high'] += 1
        elif ball_quality >= 0.3:
            ball_quality_counts['mid'] += 1
        else:
            ball_quality_counts['low'] += 1

        classification_methods[f"yolo_signal_{yolo_outcome}"] += 1
        if classification_result.get('classifier_method'):
            classification_methods[str(classification_result['classifier_method'])] += 1
        outcome_debug['classification_window'] = [int(shot['start_frame']), int(outcome_end_frame)]
        outcome_debug['classification_window_strategy'] = str(classification_window_strategy)
        outcome_reasons[outcome_debug.get('reason', 'unknown')] += 1

        form_score, form_score_window_strategy, form_visibility_debug = get_form_score_for_shot(
            pose_frames,
            release_frame,
            shot['start_frame'],
            outcome_end_frame,
            release_search_gap_frames,
            return_debug=True,
            form_visibility_thresholds=form_visibility_thresholds,
        )
        raw_form_score = float(form_score) if form_score is not None else None
        calibration_mean = float(SHOTLAB_FORM_SCORE_CALIBRATION_MEAN)
        calibration_std = float(SHOTLAB_FORM_SCORE_CALIBRATION_STD)
        calibration_scale = float(SHOTLAB_FORM_SCORE_CALIBRATION_SCALE)
        if behind_basket_mode and raw_form_score is not None:
            form_score = compute_calibrated_form_score(
                raw_form_score,
                calibration_mean,
                calibration_std,
                score_scale=calibration_scale,
            )
        elif raw_form_score is not None:
            form_score = float(raw_form_score)
        if isinstance(form_visibility_debug, dict):
            form_visibility_debug['raw_form_score_before_scaling'] = raw_form_score
            form_visibility_debug['final_form_score'] = float(form_score) if form_score is not None else None
            form_visibility_debug['score_calibration'] = {
                'benchmark_mean': calibration_mean,
                'benchmark_std': calibration_std,
                'score_scale': calibration_scale,
            }
        if form_score is not None:
            tracks_for_logging = shot_tracks_for_classifier if shot_tracks_for_classifier is not None else shot_tracks
            _log_ball_trajectory(release_frame, tracks_for_logging)

        benchmark_mean, benchmark_std = get_zone_benchmark(zone)
        if behind_basket_mode:
            benchmark_mean = calibration_mean
            benchmark_std = calibration_std
        zone_stats[zone]['benchmark_mean'] = benchmark_mean
        zone_stats[zone]['benchmark_std'] = benchmark_std
        if behind_basket_mode:
            shotsync_score = float(form_score) if form_score is not None else None
        else:
            shotsync_score = compute_shotsync_score(form_score, benchmark_mean, benchmark_std) if form_score is not None else None

        zone_stats[zone]['attempts'] += 1
        if outcome == 'make':
            zone_stats[zone]['makes'] += 1
        if shotsync_score is not None:
            zone_stats[zone]['score_total'] += shotsync_score
            zone_stats[zone]['score_count'] += 1
            shots_with_scores += 1

        clip_window = compute_dynamic_clip_window(
            release_frame,
            shot_tracks=shot_tracks,
            fps=fps,
            total_frames=total_frames
        )
        clip_start_frame = int(clip_window.get('clip_start_frame', max(0, release_frame - clip_before_frames)))
        clip_end_frame = int(clip_window.get('clip_end_frame', release_frame + clip_after_frames))
        # In behind-basket MVP mode, keep at least a full tail window so users can
        # visually confirm make/miss before the clip ends.
        if behind_basket_mode:
            min_after_frames = max(
                int(clip_after_frames),
                int(round(float(BB_MIN_CLIP_AFTER_SECONDS) * max(1.0, float(fps or 30.0))))
            )
            min_clip_end = int(release_frame + min_after_frames)
            if int(total_frames or 0) > 0:
                min_clip_end = min(min_clip_end, int(total_frames) - 1)
            clip_end_frame = max(int(clip_end_frame), int(min_clip_end))
        clip_start = float(clip_start_frame / max(1.0, float(fps or 30.0)))
        clip_end = float(clip_end_frame / max(1.0, float(fps or 30.0)))
        shot['clip_start_frame'] = int(clip_start_frame)
        shot['clip_end_frame'] = int(clip_end_frame)

        outcome_confidence = float(outcome_debug.get('confidence', 0.0) or 0.0)
        vote_breakdown = outcome_debug.get('method_details') or outcome_debug.get('vote_breakdown') or []
        weak_votes = False
        if isinstance(vote_breakdown, list) and vote_breakdown:
            vote_scores = [
                float(v.get('confidence', 0.0) or 0.0)
                for v in vote_breakdown
                if isinstance(v, dict) and str(v.get('outcome', 'unknown')) in ('make', 'miss')
            ]
            weak_votes = bool(vote_scores) and max(vote_scores) < 0.55
        low_evidence = bool(outcome_debug.get('low_evidence', outcome_confidence < 0.55))
        if outcome_confidence < 0.52 and weak_votes:
            low_evidence = True
            outcome_debug['low_evidence'] = True
            outcome_debug['low_evidence_reason'] = 'confidence_below_0_52_and_weak_votes'
        made_shadow_for_shot = dict(made_shadow_by_shot.get(int(idx)) or {})
        made_shadow_conf = float(made_shadow_for_shot.get('confidence', 0.0) or 0.0)
        shot_runtime_ms = float(max(0.0, (time.time() - shot_started) * 1000.0))
        source_label = str(shot.get('source') or 'pose')
        form_text = 'null' if form_score is None else str(int(round(float(form_score))))
        shot_line = (
            f"Shot {idx + 1}: frame={release_frame} source={source_label} outcome={outcome} "
            f"zone={zone} form={form_text} runtime={shot_runtime_ms:.1f}ms"
        )
        logger.info(shot_line)
        shots_analysis.append({
            'shot_number': idx + 1,
            'shot_num': idx + 1,
            'timestamp': shot['timestamp'],
            'release_frame': release_frame,
            'release_time': float(release_frame) / max(1.0, float(fps or 30.0)),
            'zone': zone,
            'shooting_zone': zone,
            'shooter_x': shooter_x_norm,
            'shooter_y': shooter_y_norm,
            'shot_type': shot_type,
            'court_position': {'x': float(court_pos[0]), 'y': float(court_pos[1])} if court_pos is not None else None,
            'outcome': outcome,
            'skeleton_frames': (
                _extract_skeleton_frames(release_frame, skeleton_pose_lookup)
                if behind_basket_mode and skeleton_pose_lookup[0]
                else []
            ),
            'candidate_confidence': str(shot.get('candidate_confidence') or 'standard'),
            'coaching': _compute_coaching_for_shot(form_visibility_debug, form_score),
            'annotated_stills': [],
            'form_score': form_score,
            'raw_form_score_before_scaling': raw_form_score,
            'form_score_reason': (
                None if form_score is not None else str(form_visibility_debug.get('form_score_reason') or 'none_found')
            ),
            'shotsync_score': shotsync_score,
            'outcome_confidence': outcome_confidence,
            'ball_quality': ball_quality,
            'clip_url': f'/api/shotlab_clip?session_id={session_id}&shot_index={idx}' if session_id else None,
            'clip_start': clip_start,
            'clip_end': clip_end,
            'clip_start_frame': int(clip_start_frame),
            'clip_end_frame': int(clip_end_frame),
            'clip_start_time': float(clip_start if clip_start is not None else (clip_start_frame / max(1.0, float(fps or 30.0)))),
            'clip_end_time': float(clip_end if clip_end is not None else (clip_end_frame / max(1.0, float(fps or 30.0)))),
            'video_url': video_url,
            'debug': {
                'outcome_reason': outcome_debug.get('reason'),
                'trajectory_points': outcome_debug.get('trajectory_points', 0),
                'release_frame_raw': int(raw_release_frame),
                'release_frame_refined': int(release_frame),
                'release_refine_confidence': release_refine.get('release_refine_confidence'),
                'release_refine_signals': release_refine.get('release_refine_signals'),
                'start_frame': shot['start_frame'],
                'end_frame': shot['end_frame'],
                'clip_start_frame': int(clip_start_frame),
                'clip_end_frame': int(clip_end_frame),
                'candidate_source': str(shot.get('source') or 'pose'),
                'anchor_source': anchor_source,
                'anchor_reason': anchor_reason,
                'shoot_model_confirmed': bool(shot.get('shoot_model_confirmed', False)),
                'shoot_model_confidence': float(shot.get('shoot_model_confidence', 0.0) or 0.0),
                'shoot_model_match_type': shot.get('shoot_model_match_type'),
                'candidate_confidence': str(shot.get('candidate_confidence') or 'standard'),
                'ball_quality': ball_quality,
                'outcome_tier': outcome_debug.get('tier'),
                'outcome_confidence': outcome_confidence,
                'low_evidence': low_evidence,
                'classification_window': outcome_debug.get('classification_window'),
                'classification_window_strategy': outcome_debug.get('classification_window_strategy'),
                'vote_breakdown': vote_breakdown,
                'made_shadow_detected': bool(made_shadow_for_shot.get('detected', False)),
                'made_shadow_confidence': made_shadow_conf,
                'made_shadow_frame': made_shadow_for_shot.get('frame'),
                'form_score_occluded': bool(form_score is None),
                'form_score_window_strategy': form_score_window_strategy,
                'form_visibility': form_visibility_debug,
                'raw_form_score_before_scaling': raw_form_score,
                'final_form_score': float(form_score) if form_score is not None else None,
                'score_calibration': {
                    'benchmark_mean': calibration_mean,
                    'benchmark_std': calibration_std,
                    'score_scale': calibration_scale,
                },
                'shot_line': shot_line,
                'shot_runtime_ms': round(float(shot_runtime_ms), 1),
            }
        })

    for sa_idx, sa in enumerate(shots_analysis):
        stills, stills_message = _build_annotated_stills(
            sa['release_frame'],
            sa.get('skeleton_frames', []),
            sa.get('coaching', {}),
            session_id,
            sa_idx,
            fps=fps or 30.0,
            yolo_detections=yolo_detections,
            frame_shape=(first_frame.shape if first_frame is not None else None),
            behind_basket_mode=bool(behind_basket_mode),
        )
        sa['annotated_stills'] = stills
        if stills_message:
            sa['annotated_stills'] = None
            sa['stills_message'] = str(stills_message)

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
    session_summary = build_shotlab_session_summary(shots_analysis)
    coaching_consistency, coaching_zone_breakdown = _generate_session_coaching_text(shots_analysis, zone_percentages)
    session_summary['session_coaching'] = {
        'overall_consistency': coaching_consistency,
        'zone_breakdown': coaching_zone_breakdown,
    }
    if session_id and session_id in shotlab_sessions:
        session_clip_shots = []
        for s in shots_analysis:
            debug = s.get('debug') or {}
            session_clip_shots.append({
                'start_frame': int(debug.get('start_frame', 0)),
                'release_frame': int(debug.get('release_frame_refined', debug.get('release_frame_raw', 0))),
                'end_frame': int(debug.get('end_frame', 0)),
                'clip_start_frame': int(debug.get('clip_start_frame', 0)),
                'clip_end_frame': int(debug.get('clip_end_frame', 0)),
                'timestamp': float(s.get('timestamp', 0.0) or 0.0),
                'source': debug.get('candidate_source', 'pose'),
                'shoot_model_confirmed': bool(debug.get('shoot_model_confirmed', False)),
                'shoot_model_confidence': float(debug.get('shoot_model_confidence', 0.0) or 0.0)
            })
        shotlab_sessions[session_id]['shots'] = session_clip_shots
        shotlab_sessions[session_id]['shots_analysis'] = list(shots_analysis)
        shotlab_sessions[session_id]['video_name'] = os.path.basename(video_path) if video_path else None
        shotlab_sessions[session_id]['debug'] = {
            'shots': {
                'candidate_source_counts': dict(candidate_source_counts),
                'anchor_source_counts': dict(anchor_source_counts),
                'unknown_zone_reasons': dict(unknown_zone_reasons),
            }
        }
        _save_shotlab_session(session_id, shotlab_sessions[session_id])
    return {
        'session_id': session_id,
        'video_url': video_url,
        'shots_analysis': shots_analysis,
        'zone_percentages': zone_percentages,
        'session_summary': session_summary,
        'total_makes': total_makes,
        'shots_with_scores': shots_with_scores,
        'unknown_zones': unknown_zones,
        'outcome_reasons': dict(outcome_reasons),
        'ball_quality_counts': ball_quality_counts,
        'classification_methods': dict(classification_methods),
        'candidate_source_counts': dict(candidate_source_counts),
        'anchor_source_counts': dict(anchor_source_counts),
        'unknown_zone_reasons': dict(unknown_zone_reasons),
        'made_shadow': made_shadow_debug,
        'clip_mode': clip_mode,
        'clip_seconds': clip_schedule_seconds,
    }


def _init_timing_debug():
    return {
        'pose_seconds': 0.0,
        'yolo_seconds': 0.0,
        'event_candidate_seconds': 0.0,
        'event_gap_seconds': 0.0,
        'clip_seconds': 0.0,
        'total_seconds': 0.0,
    }


def _apply_event_timing(timing_debug, shoot_model_debug):
    timing = ((shoot_model_debug or {}).get('timing_seconds') or {})
    timing_debug['event_candidate_seconds'] = float(timing.get('candidate', 0.0) or 0.0)
    timing_debug['event_gap_seconds'] = float(timing.get('gap', 0.0) or 0.0)


def _finalize_timing_debug(timing_debug, overall_start):
    timing_debug['total_seconds'] = float(max(0.0, time.time() - float(overall_start or time.time())))
    return timing_debug

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

@app.after_request
def apply_no_cache_headers(response):
    """Avoid stale ShotLab UI assets during active development."""
    no_cache_paths = {
        '/shotlab',
        '/shotlab/',
        '/shotlab/app.js',
        '/shotlab/style.css',
        '/tool',
        '/tool/',
        '/tool/app.js',
        '/tool/style.css',
    }
    if request.path in no_cache_paths:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

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

@app.route('/api/shotlab_build')
def shotlab_build_endpoint():
    """Return build identifier to confirm the running backend."""
    return jsonify({'build_id': shotlab_build_id})

@app.route('/api/test_ball_detection', methods=['POST'])
def test_ball_detection():
    """Debug endpoint to test ball detection on a single frame."""
    video_path = session.get('shotlab_video_path')
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'No video uploaded'}), 400

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    middle_frame_idx = total_frames // 2 if total_frames > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return jsonify({'error': 'Could not read frame'}), 400

    detector = get_detection_model()
    if detector.model is None:
        return jsonify({'error': 'No detection model loaded'}), 500

    try:
        if detector.model_type == 'roboflow':
            return jsonify({'error': 'Roboflow model active; switch to local model for this test'}), 400

        results = detector.model.predict(
            frame,
            imgsz=Config.BALL_IMG_SIZE,
            conf=0.01,
            iou=Config.BALL_IOU,
            classes=None,
            device=detector.device,
            verbose=True
        )
        detections = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                cls_name = detector.model.names.get(cls_id, 'unknown') if hasattr(detector.model, 'names') else 'unknown'
                detections.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })

        basketball_count = len([d for d in detections if d['class_name'] == 'basketball' or d['class_id'] == 0])

        return jsonify({
            'success': True,
            'model_path': detector.model_path,
            'model_label': detector.model_label,
            'model_classes': detector.model.names if hasattr(detector.model, 'names') else {},
            'frame_shape': list(frame.shape),
            'frame_index': middle_frame_idx,
            'device': detector.device,
            'total_detections': len(detections),
            'basketball_detections': basketball_count,
            'detections': detections[:10],
            'message': f"Found {basketball_count} basketball(s) out of {len(detections)} detections"
        })
    except Exception as exc:
        import traceback
        return jsonify({
            'success': False,
            'error': str(exc),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/save_court_calibration', methods=['POST'])
def save_court_calibration():
    """Persist court calibration points in the session."""
    try:
        data = request.json or {}
        source = data.get('source', 'manual')
        camera_position_override = _normalize_camera_position_override(data.get('camera_position_override'))
        landmarks = data.get('landmarks')
        if isinstance(landmarks, dict):
            normalized = {}
            for name, payload in landmarks.items():
                if not isinstance(payload, dict):
                    continue
                normalized[name] = {
                    'x_norm': float(payload.get('x_norm', 0)),
                    'y_norm': float(payload.get('y_norm', 0))
                }
            required = ['basket', 'free_throw', 'left_wing', 'right_wing']
            missing = [name for name in required if name not in normalized]
            has_baseline = 'baseline_left' in normalized or 'baseline_right' in normalized
            if len(normalized) < 5 or missing or not has_baseline:
                error_parts = []
                if missing:
                    error_parts.append(f"Missing required landmarks: {', '.join(missing)}")
                if not has_baseline:
                    error_parts.append("Add at least one baseline corner")
                return jsonify({'success': False, 'error': '; '.join(error_parts) or 'Invalid court landmarks'}), 400
            session['court_landmarks'] = normalized
            session['court_calibration_video_id'] = session.get('shotlab_video_id')
            session['court_calibration_signature'] = session.get('shotlab_video_signature')
            session.pop('court_points', None)
        else:
            points = data.get('points')
            if not isinstance(points, list) or len(points) < 4:
                return jsonify({'success': False, 'error': 'Invalid court points'}), 400
            normalized = []
            for pt in points[:4]:
                normalized.append({
                    'x_norm': float(pt.get('x_norm', 0)),
                    'y_norm': float(pt.get('y_norm', 0))
                })
            session['court_points'] = normalized
            session['court_calibration_video_id'] = session.get('shotlab_video_id')
            session['court_calibration_signature'] = session.get('shotlab_video_signature')
            session.pop('court_landmarks', None)
        if camera_position_override:
            session['court_camera_position_override'] = camera_position_override
        else:
            session.pop('court_camera_position_override', None)
        session.modified = True
        if SHOTLAB_DEBUG_CALIBRATION:
            print("\n=== COURT CALIBRATION SAVED ===")
            print(f"Source: {source}")
            print(f"Camera position override: {camera_position_override or 'none'}")
            if isinstance(landmarks, dict):
                print(f"Landmarks: {list(normalized.keys())}")
            else:
                print(f"Points: {len(normalized)}")
            print(f"Session video id: {session.get('shotlab_video_id')}")
            print(f"Session signature: {session.get('shotlab_video_signature')}")
            print(f"Session keys: {list(session.keys())}")
        storage = get_training_storage() if session.get('shotlab_collect_training', True) else None
        if storage:
            annotation = {
                'type': 'court_landmarks' if isinstance(landmarks, dict) else 'court_calibration',
                'source': source,
                'video_id': session.get('shotlab_video_id'),
                'court_points': normalized if isinstance(normalized, list) else None,
                'court_landmarks': normalized if isinstance(normalized, dict) else None,
                'camera_position_override': camera_position_override,
                'auto_landmarks': data.get('auto_landmarks'),
                'note': 'User confirmed court calibration.'
            }
            storage.save_manual_annotation(annotation)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reset_calibration', methods=['POST'])
def reset_calibration():
    """Clear saved calibration."""
    session.pop('net_zone', None)
    session.pop('net_zone_video_id', None)
    session.pop('net_zone_signature', None)
    session.pop('rim_zone', None)
    session.pop('rim_zone_video_id', None)
    session.pop('rim_zone_signature', None)
    session.pop('court_points', None)
    session.pop('court_landmarks', None)
    session.pop('court_calibration_video_id', None)
    session.pop('court_calibration_signature', None)
    session.pop('court_camera_position_override', None)
    session.modified = True
    return jsonify({'success': True})

@app.route('/api/shotlab_clip')
def shotlab_clip_endpoint():
    """Serve a short clip for a detected shot."""
    try:
        session_id = request.args.get('session_id', '').strip()
        shot_index = int(request.args.get('shot_index', '-1'))
        clip_path = extract_shot_clip(session_id, shot_index)
        return send_file(clip_path, mimetype='video/mp4')
    except Exception as e:
        import traceback
        print(f"ShotLab clip error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/shotlab_frame')
def shotlab_frame_endpoint():
    """Serve a single JPEG frame from the session video at a given frame index."""
    try:
        session_id = request.args.get('session_id', '').strip()
        frame_idx = int(request.args.get('frame_idx', '0'))
        session = shotlab_sessions.get(session_id)
        if session is None:
            abort(404)
        video_path = session.get('video_path')
        if not video_path or not os.path.exists(video_path):
            abort(404)
        frames = _read_video_frames_sparse(video_path, [frame_idx])
        frame = frames.get(frame_idx)
        if frame is None:
            abort(404)

        # Optional crop params (normalized 0-1)
        cx = request.args.get('cx', type=float)
        cy = request.args.get('cy', type=float)
        cw = request.args.get('cw', type=float)
        ch = request.args.get('ch', type=float)
        if cx is not None and cy is not None and cw is not None and ch is not None:
            fh, fw = frame.shape[:2]
            x0 = max(0, int(cx * fw))
            y0 = max(0, int(cy * fh))
            x1 = min(fw, int((cx + cw) * fw))
            y1 = min(fh, int((cy + ch) * fh))
            if x1 > x0 and y1 > y0:
                frame = frame[y0:y1, x0:x1]

        max_dim = 720
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            abort(500)
        return Response(buf.tobytes(), mimetype='image/jpeg',
                        headers={'Cache-Control': 'public, max-age=3600'})
    except (ValueError, TypeError):
        abort(400)
    except Exception as e:
        logger.error("shotlab_frame error: %s", e)
        abort(500)


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

@app.route('/api/process_shotlab_v4', methods=['POST'])
def process_shotlab_v4():
    """Process session video with one-time net zone calibration."""
    try:
        overall_start = time.time()
        timing_debug = _init_timing_debug()
        if SHOTLAB_DEBUG_CALIBRATION:
            print("\n=== STARTING ANALYSIS ===")
            print(f"Session keys: {list(session.keys())}")
        update_shotlab_status('starting', 'Initializing ShotLab...', 0.0)
        session_id = None
        camera_mode = _normalize_camera_mode(request.form.get('camera_mode'))
        behind_basket_mode = _is_behind_basket_mode(camera_mode)
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video provided'}), 400
        rtmpose_available = False
        if behind_basket_mode:
            estimator = get_rtmpose_estimator()
            rtmpose_available = bool(estimator is not None and getattr(estimator, 'available', False))
        if not MEDIAPIPE_AVAILABLE and not (behind_basket_mode and rtmpose_available):
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

        collect_training_data = request.form.get('collect_training_data', 'true').lower() == 'true'
        force_court_recalibration = _form_flag_true(request.form, 'force_court_recalibration')
        confirm_court_reuse = _form_flag_true(request.form, 'confirm_court_reuse')
        request_camera_position_override = _normalize_camera_position_override(
            request.form.get('camera_position_override')
        )
        request_event_angle_override = _normalize_event_camera_angle_override(
            request.form.get('event_camera_angle_override')
        )
        collection_session_id = None
        collection_metadata = {
            'video_id': save_name,
            'original_filename': filename,
            'analysis_version': 'shotlab_v4',
            'collect_training_data': collect_training_data
        }
        if collect_training_data and Config.COLLECT_TRAINING_DATA:
            collection_session_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
            session['shotlab_collection_id'] = collection_session_id
        session['shotlab_video_id'] = save_name
        session['shotlab_video_path'] = video_path
        session['shotlab_video_signature'] = build_video_signature(video_path, filename)
        session['shotlab_collect_training'] = collect_training_data
        session.modified = True

        warnings = []
        camera_position = None
        camera_info = {'confidence': 0.0}

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        ret, first_frame = cap.read()
        cap.release()
        if not ret or first_frame is None:
            update_shotlab_status('error', 'Unable to read video', 0.0)
            return jsonify({'success': False, 'error': 'Unable to read video'}), 500

        update_shotlab_status('court', 'Preparing court calibration...', 0.06)
        transform_matrix = None
        transform_meta = None
        court_keypoints = []
        court_preview = None
        court_shape = first_frame.shape
        court_auto_debug = {
            'samples': 0,
            'total_frames': total_frames,
            'best_keypoints': 0,
            'transform_available': False,
            'manual_only': True
        }
        manual_court_shape = first_frame.shape
        court_calibration = None
        if manual_court_shape is not None:
            h_c, w_c = manual_court_shape[:2]
            court_calibration = {
                'court_frame': court_preview,
                'width': int(w_c),
                'height': int(h_c),
                'frame_idx': court_auto_debug.get('best_frame'),
                'landmark_mode': True,
                'landmarks': list(COURT_LANDMARKS.keys())
            }
            if court_calibration['court_frame'] is None and first_frame is not None:
                court_calibration['court_frame'] = encode_frame_base64(first_frame)

        saved_landmarks = session.get('court_landmarks') if session else None
        saved_points = session.get('court_points') if session else None
        camera_position_override = session.get('court_camera_position_override') if session else None
        if request_camera_position_override is not None:
            camera_position_override = request_camera_position_override
        if behind_basket_mode:
            camera_position_override = 'in_front_of_shooter'
            saved_landmarks = None
            saved_points = None
        calibration_video_id = session.get('court_calibration_video_id') if session else None
        calibration_signature = session.get('court_calibration_signature') if session else None
        current_signature = session.get('shotlab_video_signature') if session else None
        reuse_state = resolve_court_reuse_gate(
            calibration_signature,
            current_signature,
            force_court_recalibration=force_court_recalibration,
            confirm_court_reuse=confirm_court_reuse,
        )
        court_calibration = _augment_court_calibration_payload(
            court_calibration,
            reuse_state,
            current_signature,
            camera_position_override=camera_position_override,
        )

        court_source = 'manual'
        if SHOTLAB_DEBUG_CALIBRATION:
            print(f"Loaded court landmarks: {list(saved_landmarks.keys()) if isinstance(saved_landmarks, dict) else None}")
            print(f"Loaded court points: {len(saved_points) if isinstance(saved_points, list) else 0}")
            print(f"Court calibration video id: {calibration_video_id} | current video id: {save_name}")
            print(f"Court calibration signature: {calibration_signature} | current signature: {current_signature}")
            print(f"Court reuse state: {reuse_state}")

        if (not behind_basket_mode) and reuse_state.get('court_confirm_required') and (saved_landmarks or saved_points):
            update_shotlab_status('court', 'Confirm court calibration reuse', 0.08)
            _finalize_timing_debug(timing_debug, overall_start)
            return jsonify({
                'success': True,
                'shots': [],
                'zone_stats': {},
                'total_attempts': 0,
                'total_makes': 0,
                'warnings': warnings,
                'debug': {
                    'court': {
                        'court_keypoints': len(court_keypoints),
                        'court_transform_available': False,
                        'auto': court_auto_debug,
                        'preview_available': court_preview is not None,
                        'source': court_source,
                        'transform_type': None,
                        'reuse_state': reuse_state,
                    },
                    'timing': timing_debug
                },
                'calibration': {
                    'net_zone_required': False,
                    'court': court_calibration,
                    'court_required': False
                },
                'session_id': session_id
            })

        if not reuse_state.get('court_can_use_saved', False):
            saved_landmarks = None
            saved_points = None

        if transform_matrix is None and not behind_basket_mode:
            if saved_landmarks:
                manual_transform, transform_meta = manual_court_transform_from_landmarks(saved_landmarks, manual_court_shape)
                if manual_transform is not None:
                    transform_matrix = manual_transform
                    court_source = 'landmarks'
            if transform_matrix is None:
                manual_transform = manual_court_transform_from_points(saved_points, manual_court_shape)
                if manual_transform is not None:
                    transform_matrix = manual_transform
                    transform_meta = {'type': 'legacy'}
                    court_source = 'session'
            if transform_matrix is None:
                warnings.append('court_transform_unavailable')
                logger.warning("Court transform unavailable; continuing without court mapping.")

        court_debug = {
            'court_keypoints': len(court_keypoints),
            'court_transform_available': transform_matrix is not None,
            'auto': court_auto_debug,
            'preview_available': court_preview is not None,
            'source': court_source,
            'transform_type': transform_meta.get('type') if transform_meta else None,
            'reuse_state': reuse_state,
            'court_signature': current_signature,
            'camera_mode': camera_mode,
            'camera_position_override': _normalize_camera_position_override(camera_position_override),
        }

        if transform_matrix is None and not behind_basket_mode:
            update_shotlab_status('court', 'Manual court calibration required', 0.08)
            _finalize_timing_debug(timing_debug, overall_start)
            return jsonify({
                'success': True,
                'shots': [],
                'zone_stats': {},
                'total_attempts': 0,
                'total_makes': 0,
                'warnings': warnings,
                'debug': {
                    'court': court_debug,
                    'timing': timing_debug
                },
                'calibration': {
                    'net_zone_required': False,
                    'court': court_calibration,
                    'court_required': True
                },
                'session_id': session_id
            })

        court_points = session.get('court_landmarks') if session else None
        if court_points is None and session:
            court_points = session.get('court_points')
        frame_height = int(first_frame.shape[0]) if first_frame is not None else None
        frame_width = int(first_frame.shape[1]) if first_frame is not None else None
        camera_info = {'position': camera_position or 'unknown', 'confidence': 0.0}
        if behind_basket_mode:
            camera_info = {
                'position': 'in_front_of_shooter',
                'confidence': 1.0,
                'camera_mode': camera_mode,
                'override_applied': True,
                'camera_position_override': 'in_front_of_shooter',
            }
        elif camera_position is None:
            camera_info = estimate_camera_position(court_points, frame_width, frame_height)
        yolo_detections = {}
        yolo_rim_position = None
        yolo_active_regions = []
        yolo_process_stats = {}
        yolo_elapsed = 0.0
        yolo_frame_skip = (
            int(SHOTLAB_BEHIND_BASKET_YOLO_FRAME_SKIP)
            if behind_basket_mode
            else int(SHOTLAB_YOLO_FRAME_SKIP)
        )
        if yolo_detector.available:
            update_shotlab_status('yolo', 'Running YOLO object detection...', 0.09)
            yolo_start = time.time()
            yolo_detections = yolo_detector.process_video(video_path, frame_skip=yolo_frame_skip)
            yolo_rim_position = get_stable_rim_position(yolo_detections)
            yolo_elapsed = time.time() - yolo_start
            yolo_process_stats = dict(getattr(yolo_detector, 'last_process_stats', {}) or {})
            logger.info(
                "YOLO detection completed in %.1fs (processed=%s returned=%d skip=%d)",
                yolo_elapsed,
                yolo_process_stats.get('processed_frames', len(yolo_detections)),
                len(yolo_detections),
                int(yolo_frame_skip)
            )
            if yolo_rim_position is not None:
                logger.info("YOLO stable rim position: %s", yolo_rim_position)
            if SHOTLAB_ACTIVE_REGION_ENABLE:
                yolo_active_regions = identify_active_regions(
                    yolo_detections,
                    total_frames=total_frames,
                    min_gap_frames=SHOTLAB_ACTIVE_REGION_MIN_GAP_FRAMES,
                    min_region_length=SHOTLAB_ACTIVE_REGION_MIN_LENGTH_FRAMES,
                    pad_before=SHOTLAB_ACTIVE_REGION_PAD_BEFORE,
                    pad_after=SHOTLAB_ACTIVE_REGION_PAD_AFTER,
                    min_ball_confidence=SHOTLAB_YOLO_LAUNCH_MIN_CONFIDENCE,
                    min_motion_px=SHOTLAB_ACTIVE_REGION_MIN_BALL_MOTION_PX
                )
        else:
            logger.info("YOLO detector unavailable; using existing ball/rim pipeline only")
        timing_debug['yolo_seconds'] = float(yolo_elapsed)

        if not behind_basket_mode:
            camera_info = apply_camera_position_fallback(camera_info, frame_width, yolo_rim_position)
            camera_info = _apply_camera_position_override(camera_info, camera_position_override)
        camera_position = camera_info.get('position', 'unknown')
        if SHOTLAB_DEBUG_POSE:
            print(
                "=== CAMERA POSITION (pre-pose): "
                f"{camera_position} (confidence: {camera_info.get('confidence', 0):.2f}) ==="
            )

        pose_frames = []
        shot_attempts = []
        pose_shot_count = 0
        shot_detection_debug = {}
        pose_elapsed = 0.0
        pose_debug = {
            'fps': float(fps or 0.0),
            'frame_stride': 0,
            'active_regions': yolo_active_regions,
            'active_region_mode': bool(yolo_active_regions),
            'shot_attempts_detected': 0,
            'shot_detection': {},
            'behind_basket_mode': bool(behind_basket_mode),
            'form_backend': 'mediapipe',
        }
        if behind_basket_mode:
            update_shotlab_status('pose', 'Skipping full pose scan in behind-basket mode...', 0.1)
            pose_debug.update({
                'primary_scan': 'skipped',
                'fallback_stride': int(max(1, BB_POSE_FALLBACK_STRIDE)),
                'fallback_min_candidates': int(BB_POSE_FALLBACK_MIN_CANDIDATES),
            })
        else:
            update_shotlab_status('pose', 'Running pose detection...', 0.1)
            pose_start = time.time()
            pose_frames, fps = process_video_for_pose(
                video_path,
                frame_stride=max(1, int(SHOTLAB_POSE_DETECTION_STRIDE)),
                progress_callback=lambda i, t, p: update_shotlab_status('pose', f'Pose frames {i}/{t}', 0.1 + p * 0.35),
                camera_position=camera_position,
                active_regions=yolo_active_regions if yolo_active_regions else None
            )
            if SHOTLAB_INTERPOLATE_ANGLES:
                pose_frames = interpolate_missing_angles(pose_frames)
            pose_elapsed = time.time() - pose_start
            print(f"Pose detection completed in {pose_elapsed:.1f}s")
            shot_attempts, shot_detection_debug = detect_shot_attempts_from_pose(
                pose_frames,
                fps,
                camera_position=camera_position
            )
            pose_shot_count = len(shot_attempts)
            pose_debug = summarize_pose_states(pose_frames)
            pose_debug.update({
                'fps': float(fps or 0.0),
                'frame_stride': int(max(1, SHOTLAB_POSE_DETECTION_STRIDE)),
                'active_regions': yolo_active_regions,
                'active_region_mode': bool(yolo_active_regions),
                'shot_attempts_detected': len(shot_attempts),
                'shot_detection': shot_detection_debug,
                'behind_basket_mode': False,
                'primary_scan': 'full',
            })

        recovery_max_yolo_candidates = (
            int(BB_MAX_BALL_FIRST_RECOVERY_CANDIDATES)
            if behind_basket_mode
            else 4
        )
        candidate_pipeline = run_v5_candidate_pipeline(
            shot_attempts,
            yolo_detections=yolo_detections,
            fps=fps,
            total_frames=total_frames,
            frame_height=frame_height,
            detect_ball_launches_fn=detect_ball_launches,
            recovery_max_yolo_candidates=recovery_max_yolo_candidates,
            prefer_recall=behind_basket_mode,
        )
        shot_attempts = list(candidate_pipeline.get('shot_attempts') or [])
        ball_first_candidates = list(candidate_pipeline.get('ball_first_candidates') or [])
        yolo_launches = list(candidate_pipeline.get('yolo_launches') or [])
        shot_detection_debug = dict(shot_detection_debug or {})
        shot_detection_debug['ball_first'] = dict(candidate_pipeline.get('shot_detection_ball_first') or {})
        pose_debug['ball_first'] = dict(candidate_pipeline.get('shot_detection_ball_first') or {})
        pose_debug['pose_fallback_applied'] = False
        if behind_basket_mode and len(shot_attempts) < int(BB_POSE_FALLBACK_MIN_CANDIDATES):
            update_shotlab_status('pose', 'Running lightweight pose fallback...', 0.14)
            fallback_regions = yolo_active_regions if yolo_active_regions else None
            fallback_start = time.time()
            fallback_pose_frames, fallback_fps = process_video_for_pose(
                video_path,
                frame_stride=max(1, int(BB_POSE_FALLBACK_STRIDE)),
                progress_callback=None,
                camera_position=camera_position,
                active_regions=fallback_regions
            )
            if SHOTLAB_INTERPOLATE_ANGLES:
                fallback_pose_frames = interpolate_missing_angles(fallback_pose_frames)
            fallback_elapsed = time.time() - fallback_start
            pose_elapsed += fallback_elapsed
            if fallback_fps:
                fps = float(fallback_fps)
            fallback_candidates, fallback_detection_debug = detect_shot_attempts_from_pose(
                fallback_pose_frames,
                fps,
                camera_position=camera_position
            )
            pose_frames = merge_pose_frames(pose_frames, fallback_pose_frames) if pose_frames else fallback_pose_frames
            pose_shot_count = max(int(pose_shot_count), len(fallback_candidates))
            pose_debug = summarize_pose_states(pose_frames)
            pose_debug.update({
                'fps': float(fps or 0.0),
                'frame_stride': int(max(1, BB_POSE_FALLBACK_STRIDE)),
                'active_regions': fallback_regions,
                'active_region_mode': bool(fallback_regions),
                'shot_attempts_detected': len(fallback_candidates),
                'shot_detection': fallback_detection_debug,
                'behind_basket_mode': True,
                'primary_scan': 'skipped',
                'pose_fallback_applied': True,
                'pose_fallback_elapsed_seconds': round(float(fallback_elapsed), 3),
                'pose_fallback_candidate_count': len(fallback_candidates),
            })
            shot_detection_debug['pose_fallback'] = {
                'applied': True,
                'reason': f"candidate_count_below_{int(BB_POSE_FALLBACK_MIN_CANDIDATES)}",
                'input_candidates': len(shot_attempts),
                'fallback_pose_candidates': len(fallback_candidates),
                'fallback_pose_elapsed_seconds': round(float(fallback_elapsed), 3),
            }
            candidate_pipeline = run_v5_candidate_pipeline(
                fallback_candidates,
                yolo_detections=yolo_detections,
                fps=fps,
                total_frames=total_frames,
                frame_height=frame_height,
                detect_ball_launches_fn=detect_ball_launches,
                recovery_max_yolo_candidates=recovery_max_yolo_candidates,
                prefer_recall=behind_basket_mode,
            )
            shot_attempts = list(candidate_pipeline.get('shot_attempts') or [])
            ball_first_candidates = list(candidate_pipeline.get('ball_first_candidates') or [])
            yolo_launches = list(candidate_pipeline.get('yolo_launches') or [])
            shot_detection_debug['ball_first'] = dict(candidate_pipeline.get('shot_detection_ball_first') or {})
            pose_debug['ball_first'] = dict(candidate_pipeline.get('shot_detection_ball_first') or {})
        elif behind_basket_mode:
            shot_detection_debug['pose_fallback'] = {
                'applied': False,
                'reason': 'candidate_floor_met',
                'input_candidates': len(shot_attempts),
            }
        timing_debug['pose_seconds'] = float(pose_elapsed)
        collection_metadata['fps'] = float(fps or 0.0)

        if not shot_attempts and not yolo_detector.available:
            update_shotlab_status('complete', 'No shots detected', 1.0)
            _finalize_timing_debug(timing_debug, overall_start)
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
                'source': 'manual',
                'auto': {'samples': 0, 'rim_available': False, 'manual_only': True},
                'available': False
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
                    'rim': rim_debug,
                    'net_zone': {'available': False},
                    'timing': timing_debug,
                },
                'calibration': {
                    'net_zone_required': False,
                    'court': court_calibration,
                    'court_required': (transform_matrix is None and not behind_basket_mode)
                },
                'session_id': session_id
            })

        if not ball_first_candidates:
            shot_attempts, yolo_launches, yolo_launch_debug = apply_yolo_launch_fallback_candidates(
                shot_attempts=shot_attempts,
                yolo_detections=yolo_detections,
                frame_height=frame_height,
                fps=fps,
                total_frames=total_frames,
                shot_detection_debug=shot_detection_debug,
                pose_debug=pose_debug
            )
        else:
            yolo_launch_debug = {
                'pose_candidates': len(shot_attempts),
                'launch_candidates': len(yolo_launches),
                'added': 0,
                'skipped_near_pose': 0,
                'output_candidates': len(shot_attempts),
                'mode': 'ball_first_primary'
            }
            shot_detection_debug['yolo_launch'] = {
                **yolo_launch_debug,
                'detected_launches': yolo_launches
            }

        shot_attempts, shoot_model_debug = apply_shoot_model_candidates(
            video_path=video_path,
            shot_attempts=shot_attempts,
            yolo_launches=yolo_launches,
            fps=fps,
            total_frames=total_frames,
            camera_position=camera_position,
            camera_mode=camera_mode,
            event_angle_override=request_event_angle_override,
            active_regions=yolo_active_regions if yolo_active_regions else None,
            shot_detection_debug=shot_detection_debug,
            pose_debug=pose_debug
        )
        _apply_event_timing(timing_debug, shoot_model_debug)
        if behind_basket_mode:
            shot_attempts, cluster_debug = cluster_behind_basket_candidates(shot_attempts, fps=fps)
            shot_detection_debug['behind_basket_candidate_clusters'] = cluster_debug

        if not shot_attempts:
            update_shotlab_status('complete', 'No shots detected', 1.0)
            _finalize_timing_debug(timing_debug, overall_start)
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
                'source': 'auto',
                'auto': {'samples': 0, 'rim_available': bool(yolo_rim_position), 'manual_only': False},
                'available': bool(yolo_rim_position)
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
                    'rim': rim_debug,
                    'net_zone': {'available': False},
                    'yolo': {
                        'available': yolo_detector.available,
                        'frames_analyzed': int(yolo_process_stats.get('processed_frames', len(yolo_detections))),
                        'frames_returned': len(yolo_detections),
                        'frame_skip': int(yolo_process_stats.get('frame_skip', yolo_frame_skip)),
                        'interpolated_frames': int(yolo_process_stats.get('interpolated_frames', 0)),
                        'rim_available': yolo_rim_position is not None,
                        'launch_fallback': {
                            **yolo_launch_debug,
                            'detected_launches': yolo_launches
                        },
                        'shoot_model': dict(shoot_model_debug or {})
                    },
                    'timing': timing_debug,
                },
                'calibration': {
                    'net_zone_required': False,
                    'court': court_calibration,
                    'court_required': (transform_matrix is None and not behind_basket_mode)
                },
                'session_id': session_id
            })

        net_zone = None
        rim_zone = None
        net_zone_source = None

        update_shotlab_status('rim', 'Preparing net zone calibration...', 0.48)
        rim_position = yolo_bbox_to_rim_position(yolo_rim_position) if yolo_rim_position is not None else None
        rim_source = 'auto'
        rim_shape = first_frame.shape
        rim_auto_debug = {
            'samples': 0,
            'rim_available': False,
            'manual_only': False
        }
        rim_calibration = {
            'rim_frame': encode_frame_base64(first_frame),
            'width': int(first_frame.shape[1]),
            'height': int(first_frame.shape[0]),
            'frame_idx': 0
        }
        if rim_position is not None:
            rim_source = 'yolo'
            rim_auto_debug['source'] = 'yolo_stable'
            rim_auto_debug['rim_available'] = True
            net_zone = build_net_zone_from_rim(rim_position, rim_shape or first_frame.shape)
            if net_zone is not None:
                net_zone_source = 'rim_auto'
            rim_zone = rim_position_to_zone(rim_position)
        else:
            warnings.append('auto_rim_detection_failed')
            logger.warning("Automatic YOLO rim detection failed for this video.")

        deduped_shots, dedupe_debug = dedupe_shots_by_release(
            shot_attempts,
            fps,
            camera_position=camera_position
        )
        if dedupe_debug:
            shot_detection_debug['dedupe'] = {
                'kept': dedupe_debug.get('kept', 0),
                'dropped': dedupe_debug.get('dropped', 0)
            }
        if behind_basket_mode:
            camera_info = {
                'position': 'in_front_of_shooter',
                'confidence': 1.0,
                'camera_mode': camera_mode,
                'override_applied': True,
                'camera_position_override': 'in_front_of_shooter',
            }
        else:
            camera_info = {'position': camera_position or 'unknown', 'confidence': 0.0}
            if camera_position is None:
                camera_info = estimate_camera_position(court_points, frame_width, frame_height)
            camera_info = apply_camera_position_fallback(camera_info, frame_width, yolo_rim_position)
            camera_info = _apply_camera_position_override(camera_info, camera_position_override)
        camera_position = camera_info.get('position', 'unknown')
        print(
            "=== CAMERA POSITION: "
            f"{camera_position} (confidence: {camera_info.get('confidence', 0):.2f}) ==="
        )
        confirmed_shots, flight_debug = confirm_shot_candidates_with_ball_flight(
            deduped_shots,
            pose_frames,
            yolo_detections,
            yolo_rim_position,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            camera_position=camera_position
        )
        filter_debug = {
            'input': len(deduped_shots),
            'kept': len(confirmed_shots),
            'dropped': max(0, len(deduped_shots) - len(confirmed_shots)),
            'dropped_reasons': dict(flight_debug.get('dropped_reasons', {})),
            'dropped_shots': list(flight_debug.get('dropped_shots', [])),
            'ball_flight_confirmation': flight_debug
        }
        shot_detection_debug['ball_flight_confirmation'] = flight_debug
        if SHOTLAB_DEBUG_REJECTED_SHOTS and filter_debug.get('dropped_shots'):
            save_rejected_shot_debug(video_path, filter_debug.get('dropped_shots'))

        update_shotlab_status('ball', f'Tracking ball near {len(confirmed_shots)} shots...', 0.5)
        collection_metadata['model'] = get_detection_model().model_label
        collection_context = None
        if collection_session_id:
            collection_context = {
                'session_id': collection_session_id,
                'video_metadata': collection_metadata
            }
        ball_windows = candidate_centered_ball_windows(
            confirmed_shots,
            total_frames=total_frames,
            fps=fps
        )
        ball_start = time.time()
        ball_tracks, ball_debug = process_video_for_ball_tracking(
            video_path,
            frame_stride=SHOTLAB_BALL_STRIDE,
            shot_attempts=confirmed_shots,
            fps=fps,
            custom_windows=ball_windows,
            progress_callback=lambda i, t, p: update_shotlab_status('ball', f'Ball window frames {i}/{t}', 0.5 + p * 0.35),
            collection_context=collection_context,
            yolo_detections=yolo_detections,
            yolo_rim_position=yolo_rim_position
        )
        ball_elapsed = time.time() - ball_start
        print(f"Ball tracking completed in {ball_elapsed:.1f}s")

        print("=== COURT CALIBRATION DATA ===")
        print(f"  court_points: {court_points}")
        print(f"  court_landmarks: {session.get('court_landmarks') if session else None}")
        filtered_shots = filter_false_positives(
            confirmed_shots,
            ball_tracks,
            rim_zone,
            net_zone,
            pose_frames,
            frame_height,
            camera_position=camera_position,
            yolo_detections=yolo_detections,
            yolo_rim_position=yolo_rim_position,
            debug=filter_debug
        )
        shot_detection_debug['filter'] = filter_debug
        shot_detection_debug['false_positive_filter'] = filter_debug.get('false_positive_filter', {})
        shot_attempts = filtered_shots
        shot_attempts, duplicate_echo_debug = suppress_behind_basket_duplicate_echoes(
            shot_attempts,
            fps=fps,
            camera_position=camera_position,
        )
        shot_detection_debug['duplicate_echo_suppression'] = duplicate_echo_debug
        filter_debug['duplicate_echo_suppression'] = duplicate_echo_debug
        if behind_basket_mode:
            yolo_share_cap_debug = {
                'input_count': len(shot_attempts),
                'max_share': float(SHOTLAB_YOLO_MAX_ACCEPTED_SHARE),
                'pose_count': sum(1 for s in shot_attempts if str(s.get('source') or 'pose') == 'pose'),
                'yolo_count': sum(1 for s in shot_attempts if str(s.get('source') or 'pose') != 'pose'),
                'max_yolo_allowed': None,
                'dropped': 0,
                'dropped_release_frames': [],
                'applied': False,
                'reason': 'behind_basket_disabled'
            }
        else:
            shot_attempts, yolo_share_cap_debug = enforce_yolo_source_share_cap(
                shot_attempts,
                max_share=SHOTLAB_YOLO_MAX_ACCEPTED_SHARE
            )
        shot_detection_debug['yolo_share_cap'] = yolo_share_cap_debug
        filter_debug['yolo_share_cap'] = yolo_share_cap_debug
        if SHOTLAB_DEBUG_REJECTED_SHOTS and filter_debug.get('dropped_shots'):
            save_rejected_shot_debug(video_path, filter_debug.get('dropped_shots'))

        pose_refine_windows = []
        pose_refine_gate = {'enabled': False, 'reason': 'pose_refine_disabled'}
        if SHOTLAB_POSE_REFINE_ENABLE:
            should_refine, pose_refine_gate = should_run_pose_refinement(
                shot_attempts,
                total_frames=total_frames,
                frames_with_pose=int(pose_debug.get('frames_with_pose', 0)),
                pose_frame_stride=int(max(1, SHOTLAB_POSE_DETECTION_STRIDE))
            )
        if behind_basket_mode and shot_attempts and int(max(1, SHOTLAB_POSE_DETECTION_STRIDE)) > 1:
            pose_refine_gate = {'enabled': True, 'reason': 'behind_basket_skeleton_dense'}
        if SHOTLAB_POSE_REFINE_ENABLE and pose_refine_gate.get('enabled'):
            pose_refine_windows = build_pose_refine_windows(
                shot_attempts,
                total_frames=total_frames,
                before_frames=max(int(SHOTLAB_POSE_REFINE_WINDOW_BEFORE_FRAMES), int(BB_SKELETON_WINDOW_FRAMES) + 5),
                after_frames=max(int(SHOTLAB_POSE_REFINE_WINDOW_AFTER_FRAMES), int(BB_SKELETON_WINDOW_FRAMES) + 5)
            )
            if pose_refine_windows:
                update_shotlab_status('pose_refine', 'Refining pose around detected shots...', 0.83)
                pose_refine_start = time.time()
                refined_pose_frames, _ = process_video_for_pose(
                    video_path,
                    frame_stride=1,
                    progress_callback=None,
                    camera_position=camera_position,
                    active_regions=pose_refine_windows
                )
                pose_frames = merge_pose_frames(pose_frames, refined_pose_frames)
                if SHOTLAB_INTERPOLATE_ANGLES:
                    pose_frames = interpolate_missing_angles(pose_frames)
                logger.info(
                    "Pose refinement completed in %.1fs (windows=%s, dense_frames=%d, merged_frames=%d)",
                    time.time() - pose_refine_start,
                    pose_refine_windows,
                    len(refined_pose_frames),
                    len(pose_frames)
                )
        pose_debug['refine_windows'] = pose_refine_windows
        pose_debug['refine_gate'] = pose_refine_gate
        form_only_pose_debug = {
            'applied': False,
            'windows': [],
            'frames_added': 0,
            'elapsed_seconds': 0.0,
            'reason': 'not_behind_basket',
        }
        if behind_basket_mode:
            update_shotlab_status('pose_form', 'Scoring ShotSync form...', 0.855)
            pose_frames, form_only_pose_debug = run_form_only_pose_pass(
                video_path,
                shot_attempts=shot_attempts,
                total_frames=total_frames,
                camera_position=camera_position,
                pose_frames=pose_frames,
                yolo_detections=yolo_detections,
            )
        pose_debug['form_only_pass'] = form_only_pose_debug
        pose_debug['form_backend'] = str(form_only_pose_debug.get('backend', 'mediapipe'))

        analysis_result = analyze_shot_attempts_common(
            video_path=video_path,
            shot_attempts=shot_attempts,
            pose_frames=pose_frames,
            fps=fps,
            ball_tracks=ball_tracks,
            rim_position=rim_position,
            rim_zone=rim_zone,
            net_zone=net_zone,
            yolo_detections=yolo_detections,
            yolo_rim_position=yolo_rim_position,
            transform_matrix=transform_matrix,
            transform_meta=transform_meta,
            court_points=court_points,
            first_frame=first_frame,
            camera_mode=camera_mode,
            camera_position=camera_position,
        )
        timing_debug['clip_seconds'] = float(analysis_result.get('clip_seconds', 0.0) or 0.0)
        session_id = analysis_result['session_id']
        video_url = analysis_result['video_url']
        shots_analysis = analysis_result['shots_analysis']
        zone_percentages = analysis_result['zone_percentages']
        total_makes = analysis_result['total_makes']
        shots_with_scores = analysis_result['shots_with_scores']
        unknown_zones = analysis_result['unknown_zones']
        outcome_reasons = analysis_result['outcome_reasons']
        ball_quality_counts = analysis_result['ball_quality_counts']
        classification_methods = analysis_result['classification_methods']
        candidate_source_counts = analysis_result['candidate_source_counts']
        anchor_source_counts = analysis_result['anchor_source_counts']
        unknown_zone_reasons = analysis_result['unknown_zone_reasons']
        session_summary = analysis_result.get('session_summary', {})

        processing_seconds = time.time() - overall_start
        _finalize_timing_debug(timing_debug, overall_start)
        log_shotlab_summary(
            processing_seconds,
            shots_analysis,
            pose_shot_count,
            dedupe_debug,
            filter_debug,
            ball_debug,
            rim_position,
            rim_zone,
            net_zone,
            transform_matrix,
            shot_detection_debug=shot_detection_debug,
            warnings=warnings,
            unknown_zones=unknown_zones,
            anchor_source_counts=anchor_source_counts,
            unknown_zone_reasons=unknown_zone_reasons,
        )
        debug_summary = {
            'court': court_debug,
            'pose': pose_debug,
            'ball': ball_debug,
            'rim': {
                'source': rim_source,
                'auto': rim_auto_debug,
                'available': rim_position is not None
            },
            'net_zone': {
                'source': net_zone_source,
                'available': net_zone is not None
            },
            'yolo': {
                'available': yolo_detector.available,
                'frames_analyzed': int(yolo_process_stats.get('processed_frames', len(yolo_detections))),
                'frames_returned': len(yolo_detections),
                'frame_skip': int(yolo_process_stats.get('frame_skip', yolo_frame_skip)),
                'interpolated_frames': int(yolo_process_stats.get('interpolated_frames', 0)),
                'rim_available': yolo_rim_position is not None,
                'launch_fallback': shot_detection_debug.get('yolo_launch', {}),
                'shoot_model': {
                    **dict(shot_detection_debug.get('shoot_model', {}) or {}),
                    'made_shadow': analysis_result.get('made_shadow', {})
                }
            },
            'timing': timing_debug,
            'shots': {
                'shot_attempts': len(shot_attempts),
                'shots_with_scores': shots_with_scores,
                'unknown_zones': unknown_zones,
                'outcome_reasons': dict(outcome_reasons),
                'ball_quality': ball_quality_counts,
                'classification_methods': dict(classification_methods),
                'candidate_source_counts': dict(candidate_source_counts),
                'anchor_source_counts': dict(anchor_source_counts),
                'unknown_zone_reasons': dict(unknown_zone_reasons),
                'ball_flight_confirmation': filter_debug.get('ball_flight_confirmation', {}) if filter_debug else {},
                'yolo_share_cap': filter_debug.get('yolo_share_cap', {}) if filter_debug else {}
            }
        }
        if session_id and session_id in shotlab_sessions:
            shotlab_sessions[session_id]['debug'] = debug_summary
            shotlab_sessions[session_id]['video_name'] = os.path.basename(video_path) if video_path else None
            _save_shotlab_session(session_id, shotlab_sessions[session_id])

        update_shotlab_status('complete', 'Analysis complete', 1.0)
        return jsonify({
            'success': True,
            'shots': shots_analysis,
            'fps': float(fps or 30.0),
            'zone_stats': zone_percentages,
            'total_attempts': len(shots_analysis),
            'total_makes': total_makes,
            'session_summary': session_summary,
            'warnings': warnings,
            'debug': debug_summary,
            'calibration': {
                'net_zone_required': False,
                'court': court_calibration,
                'court_required': (transform_matrix is None and not behind_basket_mode)
            },
            'video_url': video_url,
            'session_id': session_id
        })

    except Exception as e:
        update_shotlab_status('error', str(e), 0.0)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process_shotlab_session', methods=['POST'])
def process_shotlab_session():
    """Process full shooting session video for ShotLab analysis."""
    try:
        overall_start = time.time()
        timing_debug = _init_timing_debug()
        update_shotlab_status('starting', 'Initializing ShotLab...', 0.0)
        session_id = None
        camera_mode = _normalize_camera_mode(request.form.get('camera_mode'))
        behind_basket_mode = _is_behind_basket_mode(camera_mode)
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video provided'}), 400
        rtmpose_available = False
        if behind_basket_mode:
            estimator = get_rtmpose_estimator()
            rtmpose_available = bool(estimator is not None and getattr(estimator, 'available', False))
        if not MEDIAPIPE_AVAILABLE and not (behind_basket_mode and rtmpose_available):
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

        collect_training_data = request.form.get('collect_training_data', 'true').lower() == 'true'
        force_court_recalibration = _form_flag_true(request.form, 'force_court_recalibration')
        confirm_court_reuse = _form_flag_true(request.form, 'confirm_court_reuse')
        request_camera_position_override = _normalize_camera_position_override(
            request.form.get('camera_position_override')
        )
        request_event_angle_override = _normalize_event_camera_angle_override(
            request.form.get('event_camera_angle_override')
        )
        collection_session_id = None
        collection_metadata = {
            'video_id': save_name,
            'original_filename': filename,
            'analysis_version': 'shotlab_session',
            'collect_training_data': collect_training_data
        }
        if collect_training_data and Config.COLLECT_TRAINING_DATA:
            collection_session_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
            session['shotlab_collection_id'] = collection_session_id
        session['shotlab_video_id'] = save_name
        session['shotlab_video_path'] = video_path
        session['shotlab_video_signature'] = build_video_signature(video_path, filename)
        session['shotlab_collect_training'] = collect_training_data
        session.modified = True

        warnings = []

        # Read a reference frame for sizing/manual calibration.
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        ret, first_frame = cap.read()
        cap.release()
        if not ret or first_frame is None:
            update_shotlab_status('error', 'Unable to read video', 0.0)
            return jsonify({'success': False, 'error': 'Unable to read video'}), 500

        manual_rim = None
        net_zone = None

        # Manual court calibration only.
        update_shotlab_status('court', 'Preparing court calibration...', 0.06)
        transform_matrix = None
        transform_meta = None
        court_keypoints = []
        court_preview = None
        court_shape = first_frame.shape
        court_auto_debug = {
            'samples': 0,
            'total_frames': total_frames,
            'best_keypoints': 0,
            'transform_available': False,
            'manual_only': True
        }
        manual_court_shape = first_frame.shape
        court_source = 'manual'
        saved_landmarks = session.get('court_landmarks') if session else None
        saved_points = session.get('court_points') if session else None
        camera_position_override = session.get('court_camera_position_override') if session else None
        if request_camera_position_override is not None:
            camera_position_override = request_camera_position_override
        if behind_basket_mode:
            camera_position_override = 'in_front_of_shooter'
            saved_landmarks = None
            saved_points = None
        calibration_video_id = session.get('court_calibration_video_id') if session else None
        calibration_signature = session.get('court_calibration_signature') if session else None
        current_signature = session.get('shotlab_video_signature') if session else None
        reuse_state = resolve_court_reuse_gate(
            calibration_signature,
            current_signature,
            force_court_recalibration=force_court_recalibration,
            confirm_court_reuse=confirm_court_reuse,
        )
        if not reuse_state.get('court_can_use_saved', False):
            saved_landmarks = None
            saved_points = None
        if (not behind_basket_mode) and saved_landmarks:
            manual_court_transform, transform_meta = manual_court_transform_from_landmarks(saved_landmarks, manual_court_shape)
            if manual_court_transform is not None:
                transform_matrix = manual_court_transform
                court_source = 'landmarks'
        if (not behind_basket_mode) and transform_matrix is None and saved_points:
            manual_court_transform = manual_court_transform_from_points(saved_points, manual_court_shape)
            if manual_court_transform is not None:
                transform_matrix = manual_court_transform
                transform_meta = {'type': 'legacy'}
                court_source = 'session'
        if (not behind_basket_mode) and transform_matrix is None:
            manual_court_transform = manual_court_transform_from_request(request.form or {}, manual_court_shape)
            if manual_court_transform is not None:
                transform_matrix = manual_court_transform
                transform_meta = {'type': 'legacy'}
                court_source = 'manual'
                court_auto_debug['manual_override'] = True
        court_calibration = None
        if manual_court_shape is not None:
            h_c, w_c = manual_court_shape[:2]
            court_calibration = {
                'court_frame': court_preview,
                'width': int(w_c),
                'height': int(h_c),
                'frame_idx': court_auto_debug.get('best_frame'),
                'landmark_mode': True,
                'landmarks': list(COURT_LANDMARKS.keys())
            }
            if court_calibration['court_frame'] is None and first_frame is not None:
                court_calibration['court_frame'] = encode_frame_base64(first_frame)
        court_calibration = _augment_court_calibration_payload(
            court_calibration,
            reuse_state,
            current_signature,
            camera_position_override=camera_position_override,
        )

        if (not behind_basket_mode) and reuse_state.get('court_confirm_required') and (session.get('court_landmarks') or session.get('court_points')):
            update_shotlab_status('court', 'Confirm court calibration reuse', 0.08)
            _finalize_timing_debug(timing_debug, overall_start)
            return jsonify({
                'success': True,
                'shots': [],
                'zone_stats': {},
                'total_attempts': 0,
                'total_makes': 0,
                'warnings': warnings,
                'debug': {
                    'court': {
                        'court_keypoints': len(court_keypoints),
                        'court_transform_available': False,
                        'auto': court_auto_debug,
                        'preview_available': court_preview is not None,
                        'source': court_source,
                        'transform_type': None,
                        'reuse_state': reuse_state,
                    },
                    'timing': timing_debug
                },
                'calibration': {
                    'rim': None,
                    'rim_required': False,
                    'net_zone': None,
                    'net_zone_required': False,
                    'court': court_calibration,
                    'court_required': False
                },
                'session_id': session_id
            })

        court_debug = {
            'court_keypoints': len(court_keypoints),
            'court_transform_available': transform_matrix is not None,
            'auto': court_auto_debug,
            'preview_available': court_preview is not None,
            'source': court_source,
            'transform_type': transform_meta.get('type') if transform_meta else None,
            'reuse_state': reuse_state,
            'court_signature': current_signature,
            'camera_mode': camera_mode,
            'camera_position_override': _normalize_camera_position_override(camera_position_override),
        }

        if transform_matrix is None and not behind_basket_mode:
            warnings.append('court_transform_unavailable')
            logger.warning("Court transform unavailable; continuing without court mapping.")
            update_shotlab_status('court', 'Manual court calibration required', 0.08)
            _finalize_timing_debug(timing_debug, overall_start)
            return jsonify({
                'success': True,
                'shots': [],
                'zone_stats': {},
                'total_attempts': 0,
                'total_makes': 0,
                'warnings': warnings,
                'debug': {
                    'court': court_debug,
                    'timing': timing_debug
                },
                'calibration': {
                    'rim': None,
                    'rim_required': False,
                    'net_zone': None,
                    'net_zone_required': False,
                    'court': court_calibration,
                    'court_required': True
                },
                'session_id': session_id
            })

        frame_height = int(first_frame.shape[0]) if first_frame is not None else None
        frame_width = int(first_frame.shape[1]) if first_frame is not None else None
        court_points = session.get('court_landmarks') if session else None
        if court_points is None and session:
            court_points = session.get('court_points')
        if behind_basket_mode:
            camera_info = {
                'position': 'in_front_of_shooter',
                'confidence': 1.0,
                'camera_mode': camera_mode,
                'override_applied': True,
                'camera_position_override': 'in_front_of_shooter',
            }
        else:
            camera_info = estimate_camera_position(court_points, frame_width, frame_height)
        yolo_detections = {}
        yolo_rim_position = None
        yolo_active_regions = []
        yolo_process_stats = {}
        yolo_elapsed = 0.0
        yolo_frame_skip = (
            int(SHOTLAB_BEHIND_BASKET_YOLO_FRAME_SKIP)
            if behind_basket_mode
            else int(SHOTLAB_YOLO_FRAME_SKIP)
        )
        if yolo_detector.available:
            update_shotlab_status('yolo', 'Running YOLO object detection...', 0.09)
            yolo_start = time.time()
            yolo_detections = yolo_detector.process_video(video_path, frame_skip=yolo_frame_skip)
            yolo_rim_position = get_stable_rim_position(yolo_detections)
            yolo_elapsed = time.time() - yolo_start
            yolo_process_stats = dict(getattr(yolo_detector, 'last_process_stats', {}) or {})
            logger.info(
                "YOLO detection completed in %.1fs (processed=%s returned=%d skip=%d)",
                yolo_elapsed,
                yolo_process_stats.get('processed_frames', len(yolo_detections)),
                len(yolo_detections),
                int(yolo_frame_skip)
            )
            if yolo_rim_position is not None:
                logger.info("YOLO stable rim position: %s", yolo_rim_position)
            if SHOTLAB_ACTIVE_REGION_ENABLE:
                yolo_active_regions = identify_active_regions(
                    yolo_detections,
                    total_frames=total_frames,
                    min_gap_frames=SHOTLAB_ACTIVE_REGION_MIN_GAP_FRAMES,
                    min_region_length=SHOTLAB_ACTIVE_REGION_MIN_LENGTH_FRAMES,
                    pad_before=SHOTLAB_ACTIVE_REGION_PAD_BEFORE,
                    pad_after=SHOTLAB_ACTIVE_REGION_PAD_AFTER,
                    min_ball_confidence=SHOTLAB_YOLO_LAUNCH_MIN_CONFIDENCE,
                    min_motion_px=SHOTLAB_ACTIVE_REGION_MIN_BALL_MOTION_PX
                )
        else:
            logger.info("YOLO detector unavailable; using existing ball/rim pipeline only")
        timing_debug['yolo_seconds'] = float(yolo_elapsed)

        if not behind_basket_mode:
            camera_info = apply_camera_position_fallback(camera_info, frame_width, yolo_rim_position)
            camera_info = _apply_camera_position_override(camera_info, camera_position_override)
        camera_position = camera_info.get('position', 'unknown')
        if SHOTLAB_DEBUG_POSE:
            print(
                "=== CAMERA POSITION (pre-pose): "
                f"{camera_position} (confidence: {camera_info.get('confidence', 0):.2f}) ==="
            )

        pose_frames = []
        shot_attempts = []
        pose_shot_count = 0
        shot_detection_debug = {}
        pose_elapsed = 0.0
        pose_debug = {
            'fps': float(fps or 0.0),
            'frame_stride': 0,
            'active_regions': yolo_active_regions,
            'active_region_mode': bool(yolo_active_regions),
            'shot_attempts_detected': 0,
            'shot_detection': {},
            'behind_basket_mode': bool(behind_basket_mode),
            'form_backend': 'mediapipe',
        }
        if behind_basket_mode:
            update_shotlab_status('pose', 'Skipping full pose scan in behind-basket mode...', 0.1)
            pose_debug.update({
                'primary_scan': 'skipped',
                'fallback_stride': int(max(1, BB_POSE_FALLBACK_STRIDE)),
                'fallback_min_candidates': int(BB_POSE_FALLBACK_MIN_CANDIDATES),
            })
        else:
            update_shotlab_status('pose', 'Running pose detection...', 0.1)
            pose_start = time.time()
            pose_frames, fps = process_video_for_pose(
                video_path,
                frame_stride=max(1, int(SHOTLAB_POSE_DETECTION_STRIDE)),
                progress_callback=lambda i, t, p: update_shotlab_status('pose', f'Pose frames {i}/{t}', 0.1 + p * 0.35),
                camera_position=camera_position,
                active_regions=yolo_active_regions if yolo_active_regions else None
            )
            if SHOTLAB_INTERPOLATE_ANGLES:
                pose_frames = interpolate_missing_angles(pose_frames)
            pose_elapsed = time.time() - pose_start
            print(f"Pose detection completed in {pose_elapsed:.1f}s")
            shot_attempts, shot_detection_debug = detect_shot_attempts_from_pose(
                pose_frames,
                fps,
                camera_position=camera_position
            )
            pose_shot_count = len(shot_attempts)
            pose_debug = summarize_pose_states(pose_frames)
            pose_debug.update({
                'fps': float(fps or 0.0),
                'frame_stride': int(max(1, SHOTLAB_POSE_DETECTION_STRIDE)),
                'active_regions': yolo_active_regions,
                'active_region_mode': bool(yolo_active_regions),
                'shot_attempts_detected': len(shot_attempts),
                'shot_detection': shot_detection_debug,
                'behind_basket_mode': False,
                'primary_scan': 'full',
            })

        recovery_max_yolo_candidates = (
            int(BB_MAX_BALL_FIRST_RECOVERY_CANDIDATES)
            if behind_basket_mode
            else 4
        )
        candidate_pipeline = run_v5_candidate_pipeline(
            shot_attempts,
            yolo_detections=yolo_detections,
            fps=fps,
            total_frames=total_frames,
            frame_height=frame_height,
            detect_ball_launches_fn=detect_ball_launches,
            recovery_max_yolo_candidates=recovery_max_yolo_candidates,
            prefer_recall=behind_basket_mode,
        )
        shot_attempts = list(candidate_pipeline.get('shot_attempts') or [])
        ball_first_candidates = list(candidate_pipeline.get('ball_first_candidates') or [])
        yolo_launches = list(candidate_pipeline.get('yolo_launches') or [])
        shot_detection_debug = dict(shot_detection_debug or {})
        shot_detection_debug['ball_first'] = dict(candidate_pipeline.get('shot_detection_ball_first') or {})
        pose_debug['ball_first'] = dict(candidate_pipeline.get('shot_detection_ball_first') or {})
        pose_debug['pose_fallback_applied'] = False
        if behind_basket_mode and len(shot_attempts) < int(BB_POSE_FALLBACK_MIN_CANDIDATES):
            update_shotlab_status('pose', 'Running lightweight pose fallback...', 0.14)
            fallback_regions = yolo_active_regions if yolo_active_regions else None
            fallback_start = time.time()
            fallback_pose_frames, fallback_fps = process_video_for_pose(
                video_path,
                frame_stride=max(1, int(BB_POSE_FALLBACK_STRIDE)),
                progress_callback=None,
                camera_position=camera_position,
                active_regions=fallback_regions
            )
            if SHOTLAB_INTERPOLATE_ANGLES:
                fallback_pose_frames = interpolate_missing_angles(fallback_pose_frames)
            fallback_elapsed = time.time() - fallback_start
            pose_elapsed += fallback_elapsed
            if fallback_fps:
                fps = float(fallback_fps)
            fallback_candidates, fallback_detection_debug = detect_shot_attempts_from_pose(
                fallback_pose_frames,
                fps,
                camera_position=camera_position
            )
            pose_frames = merge_pose_frames(pose_frames, fallback_pose_frames) if pose_frames else fallback_pose_frames
            pose_shot_count = max(int(pose_shot_count), len(fallback_candidates))
            pose_debug = summarize_pose_states(pose_frames)
            pose_debug.update({
                'fps': float(fps or 0.0),
                'frame_stride': int(max(1, BB_POSE_FALLBACK_STRIDE)),
                'active_regions': fallback_regions,
                'active_region_mode': bool(fallback_regions),
                'shot_attempts_detected': len(fallback_candidates),
                'shot_detection': fallback_detection_debug,
                'behind_basket_mode': True,
                'primary_scan': 'skipped',
                'pose_fallback_applied': True,
                'pose_fallback_elapsed_seconds': round(float(fallback_elapsed), 3),
                'pose_fallback_candidate_count': len(fallback_candidates),
            })
            shot_detection_debug['pose_fallback'] = {
                'applied': True,
                'reason': f"candidate_count_below_{int(BB_POSE_FALLBACK_MIN_CANDIDATES)}",
                'input_candidates': len(shot_attempts),
                'fallback_pose_candidates': len(fallback_candidates),
                'fallback_pose_elapsed_seconds': round(float(fallback_elapsed), 3),
            }
            candidate_pipeline = run_v5_candidate_pipeline(
                fallback_candidates,
                yolo_detections=yolo_detections,
                fps=fps,
                total_frames=total_frames,
                frame_height=frame_height,
                detect_ball_launches_fn=detect_ball_launches,
                recovery_max_yolo_candidates=recovery_max_yolo_candidates,
                prefer_recall=behind_basket_mode,
            )
            shot_attempts = list(candidate_pipeline.get('shot_attempts') or [])
            ball_first_candidates = list(candidate_pipeline.get('ball_first_candidates') or [])
            yolo_launches = list(candidate_pipeline.get('yolo_launches') or [])
            shot_detection_debug['ball_first'] = dict(candidate_pipeline.get('shot_detection_ball_first') or {})
            pose_debug['ball_first'] = dict(candidate_pipeline.get('shot_detection_ball_first') or {})
        elif behind_basket_mode:
            shot_detection_debug['pose_fallback'] = {
                'applied': False,
                'reason': 'candidate_floor_met',
                'input_candidates': len(shot_attempts),
            }
        timing_debug['pose_seconds'] = float(pose_elapsed)
        collection_metadata['fps'] = float(fps or 0.0)

        if not shot_attempts and not yolo_detector.available:
            update_shotlab_status('complete', 'No shots detected', 1.0)
            _finalize_timing_debug(timing_debug, overall_start)
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
                'source': 'auto',
                'auto': {'samples': 0, 'rim_available': False},
                'available': False
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
                    'rim': rim_debug,
                    'timing': timing_debug,
                },
                'calibration': {
                    'rim': None,
                    'rim_required': False,
                    'net_zone': None,
                    'net_zone_required': False,
                    'court': court_calibration,
                    'court_required': (transform_matrix is None and not behind_basket_mode)
                },
                'session_id': session_id
            })

        if not ball_first_candidates:
            shot_attempts, yolo_launches, yolo_launch_debug = apply_yolo_launch_fallback_candidates(
                shot_attempts=shot_attempts,
                yolo_detections=yolo_detections,
                frame_height=frame_height,
                fps=fps,
                total_frames=total_frames,
                shot_detection_debug=shot_detection_debug,
                pose_debug=pose_debug
            )
        else:
            yolo_launch_debug = {
                'pose_candidates': len(shot_attempts),
                'launch_candidates': len(yolo_launches),
                'added': 0,
                'skipped_near_pose': 0,
                'output_candidates': len(shot_attempts),
                'mode': 'ball_first_primary'
            }
            shot_detection_debug['yolo_launch'] = {
                **yolo_launch_debug,
                'detected_launches': yolo_launches
            }

        shot_attempts, shoot_model_debug = apply_shoot_model_candidates(
            video_path=video_path,
            shot_attempts=shot_attempts,
            yolo_launches=yolo_launches,
            fps=fps,
            total_frames=total_frames,
            camera_position=camera_position,
            camera_mode=camera_mode,
            event_angle_override=request_event_angle_override,
            active_regions=yolo_active_regions if yolo_active_regions else None,
            shot_detection_debug=shot_detection_debug,
            pose_debug=pose_debug
        )
        _apply_event_timing(timing_debug, shoot_model_debug)
        if behind_basket_mode:
            shot_attempts, cluster_debug = cluster_behind_basket_candidates(shot_attempts, fps=fps)
            shot_detection_debug['behind_basket_candidate_clusters'] = cluster_debug

        if not shot_attempts:
            update_shotlab_status('complete', 'No shots detected', 1.0)
            _finalize_timing_debug(timing_debug, overall_start)
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
                'source': 'auto',
                'auto': {'samples': 0, 'rim_available': bool(yolo_rim_position)},
                'available': bool(yolo_rim_position)
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
                    'rim': rim_debug,
                    'yolo': {
                        'available': yolo_detector.available,
                        'frames_analyzed': int(yolo_process_stats.get('processed_frames', len(yolo_detections))),
                        'frames_returned': len(yolo_detections),
                        'frame_skip': int(yolo_process_stats.get('frame_skip', yolo_frame_skip)),
                        'interpolated_frames': int(yolo_process_stats.get('interpolated_frames', 0)),
                        'rim_available': yolo_rim_position is not None,
                        'launch_fallback': {
                            **yolo_launch_debug,
                            'detected_launches': yolo_launches
                        },
                        'shoot_model': dict(shoot_model_debug or {})
                    },
                    'timing': timing_debug,
                },
                'calibration': {
                    'rim': None,
                    'rim_required': False,
                    'net_zone': None,
                    'net_zone_required': False,
                    'court': court_calibration,
                    'court_required': (transform_matrix is None and not behind_basket_mode)
                },
                'session_id': session_id
            })

        update_shotlab_status('rim', 'Preparing rim/net zone calibration...', 0.48)
        rim_position = yolo_bbox_to_rim_position(yolo_rim_position) if yolo_rim_position is not None else None
        rim_source = 'auto'
        rim_shape = first_frame.shape
        rim_auto_debug = {
            'samples': 0,
            'rim_available': False,
            'manual_only': False
        }
        rim_calibration = {
            'rim_frame': encode_frame_base64(first_frame),
            'width': int(first_frame.shape[1]),
            'height': int(first_frame.shape[0]),
            'frame_idx': 0
        }
        net_zone = None
        rim_zone = None
        net_zone_source = None
        net_zone_calibration = dict(rim_calibration)
        net_zone_calibration['net_zone_frame'] = rim_calibration.get('rim_frame')

        if rim_position is not None:
            rim_source = 'yolo'
            rim_auto_debug['source'] = 'yolo_stable'
            rim_auto_debug['rim_available'] = True
            net_zone = build_net_zone_from_rim(rim_position, rim_shape or first_frame.shape)
            if net_zone is not None:
                net_zone_source = 'rim_auto'
            rim_zone = rim_position_to_zone(rim_position)
        else:
            warnings.append('auto_rim_detection_failed')
            logger.warning("Automatic YOLO rim detection failed for this video.")
            rim_auto_debug['rim_available'] = False

        deduped_shots, dedupe_debug = dedupe_shots_by_release(
            shot_attempts,
            fps,
            camera_position=camera_position
        )
        if dedupe_debug:
            shot_detection_debug['dedupe'] = {
                'kept': dedupe_debug.get('kept', 0),
                'dropped': dedupe_debug.get('dropped', 0)
            }
        court_points = session.get('court_landmarks') if session else None
        if court_points is None and session:
            court_points = session.get('court_points')
        if behind_basket_mode:
            camera_info = {
                'position': 'in_front_of_shooter',
                'confidence': 1.0,
                'camera_mode': camera_mode,
                'override_applied': True,
                'camera_position_override': 'in_front_of_shooter',
            }
        else:
            camera_info = estimate_camera_position(court_points, frame_width, frame_height)
            camera_info = apply_camera_position_fallback(camera_info, frame_width, yolo_rim_position)
            camera_info = _apply_camera_position_override(camera_info, camera_position_override)
        camera_position = camera_info.get('position', 'unknown')
        confirmed_shots, flight_debug = confirm_shot_candidates_with_ball_flight(
            deduped_shots,
            pose_frames,
            yolo_detections,
            yolo_rim_position,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            camera_position=camera_position
        )
        filter_debug = {
            'input': len(deduped_shots),
            'kept': len(confirmed_shots),
            'dropped': max(0, len(deduped_shots) - len(confirmed_shots)),
            'dropped_reasons': dict(flight_debug.get('dropped_reasons', {})),
            'dropped_shots': list(flight_debug.get('dropped_shots', [])),
            'ball_flight_confirmation': flight_debug
        }
        shot_detection_debug['ball_flight_confirmation'] = flight_debug
        if SHOTLAB_DEBUG_REJECTED_SHOTS and filter_debug.get('dropped_shots'):
            save_rejected_shot_debug(video_path, filter_debug.get('dropped_shots'))

        update_shotlab_status('ball', f'Tracking ball near {len(confirmed_shots)} shots...', 0.5)
        collection_metadata['model'] = get_detection_model().model_label
        collection_context = None
        if collection_session_id:
            collection_context = {
                'session_id': collection_session_id,
                'video_metadata': collection_metadata
            }
        ball_windows = candidate_centered_ball_windows(
            confirmed_shots,
            total_frames=total_frames,
            fps=fps
        )
        ball_start = time.time()
        ball_tracks, ball_debug = process_video_for_ball_tracking(
            video_path,
            frame_stride=SHOTLAB_BALL_STRIDE,
            shot_attempts=confirmed_shots,
            fps=fps,
            custom_windows=ball_windows,
            progress_callback=lambda i, t, p: update_shotlab_status('ball', f'Ball window frames {i}/{t}', 0.5 + p * 0.35),
            collection_context=collection_context,
            yolo_detections=yolo_detections,
            yolo_rim_position=yolo_rim_position
        )
        ball_elapsed = time.time() - ball_start
        print(f"Ball tracking completed in {ball_elapsed:.1f}s")

        print("=== COURT CALIBRATION DATA ===")
        print(f"  court_points: {court_points}")
        print(f"  court_landmarks: {session.get('court_landmarks') if session else None}")
        filtered_shots = filter_false_positives(
            confirmed_shots,
            ball_tracks,
            rim_zone,
            net_zone,
            pose_frames,
            frame_height,
            camera_position=camera_position,
            yolo_detections=yolo_detections,
            yolo_rim_position=yolo_rim_position,
            debug=filter_debug
        )
        shot_detection_debug['filter'] = filter_debug
        shot_detection_debug['false_positive_filter'] = filter_debug.get('false_positive_filter', {})
        shot_attempts = filtered_shots
        shot_attempts, duplicate_echo_debug = suppress_behind_basket_duplicate_echoes(
            shot_attempts,
            fps=fps,
            camera_position=camera_position,
        )
        shot_detection_debug['duplicate_echo_suppression'] = duplicate_echo_debug
        filter_debug['duplicate_echo_suppression'] = duplicate_echo_debug
        if behind_basket_mode:
            yolo_share_cap_debug = {
                'input_count': len(shot_attempts),
                'max_share': float(SHOTLAB_YOLO_MAX_ACCEPTED_SHARE),
                'pose_count': sum(1 for s in shot_attempts if str(s.get('source') or 'pose') == 'pose'),
                'yolo_count': sum(1 for s in shot_attempts if str(s.get('source') or 'pose') != 'pose'),
                'max_yolo_allowed': None,
                'dropped': 0,
                'dropped_release_frames': [],
                'applied': False,
                'reason': 'behind_basket_disabled'
            }
        else:
            shot_attempts, yolo_share_cap_debug = enforce_yolo_source_share_cap(
                shot_attempts,
                max_share=SHOTLAB_YOLO_MAX_ACCEPTED_SHARE
            )
        shot_detection_debug['yolo_share_cap'] = yolo_share_cap_debug
        filter_debug['yolo_share_cap'] = yolo_share_cap_debug

        pose_refine_windows = []
        pose_refine_gate = {'enabled': False, 'reason': 'pose_refine_disabled'}
        if SHOTLAB_POSE_REFINE_ENABLE:
            should_refine, pose_refine_gate = should_run_pose_refinement(
                shot_attempts,
                total_frames=total_frames,
                frames_with_pose=int(pose_debug.get('frames_with_pose', 0)),
                pose_frame_stride=int(max(1, SHOTLAB_POSE_DETECTION_STRIDE))
            )
        if behind_basket_mode and shot_attempts and int(max(1, SHOTLAB_POSE_DETECTION_STRIDE)) > 1:
            pose_refine_gate = {'enabled': True, 'reason': 'behind_basket_skeleton_dense'}
        if SHOTLAB_POSE_REFINE_ENABLE and pose_refine_gate.get('enabled'):
            pose_refine_windows = build_pose_refine_windows(
                shot_attempts,
                total_frames=total_frames,
                before_frames=max(int(SHOTLAB_POSE_REFINE_WINDOW_BEFORE_FRAMES), int(BB_SKELETON_WINDOW_FRAMES) + 5),
                after_frames=max(int(SHOTLAB_POSE_REFINE_WINDOW_AFTER_FRAMES), int(BB_SKELETON_WINDOW_FRAMES) + 5)
            )
            if pose_refine_windows:
                update_shotlab_status('pose_refine', 'Refining pose around detected shots...', 0.83)
                pose_refine_start = time.time()
                refined_pose_frames, _ = process_video_for_pose(
                    video_path,
                    frame_stride=1,
                    progress_callback=None,
                    camera_position=camera_position,
                    active_regions=pose_refine_windows
                )
                pose_frames = merge_pose_frames(pose_frames, refined_pose_frames)
                if SHOTLAB_INTERPOLATE_ANGLES:
                    pose_frames = interpolate_missing_angles(pose_frames)
                logger.info(
                    "Pose refinement completed in %.1fs (windows=%s, dense_frames=%d, merged_frames=%d)",
                    time.time() - pose_refine_start,
                    pose_refine_windows,
                    len(refined_pose_frames),
                    len(pose_frames)
                )
        pose_debug['refine_windows'] = pose_refine_windows
        pose_debug['refine_gate'] = pose_refine_gate
        form_only_pose_debug = {
            'applied': False,
            'windows': [],
            'frames_added': 0,
            'elapsed_seconds': 0.0,
            'reason': 'not_behind_basket',
        }
        if behind_basket_mode:
            update_shotlab_status('pose_form', 'Scoring ShotSync form...', 0.855)
            pose_frames, form_only_pose_debug = run_form_only_pose_pass(
                video_path,
                shot_attempts=shot_attempts,
                total_frames=total_frames,
                camera_position=camera_position,
                pose_frames=pose_frames,
                yolo_detections=yolo_detections,
            )
        pose_debug['form_only_pass'] = form_only_pose_debug
        pose_debug['form_backend'] = str(form_only_pose_debug.get('backend', 'mediapipe'))

        analysis_result = analyze_shot_attempts_common(
            video_path=video_path,
            shot_attempts=shot_attempts,
            pose_frames=pose_frames,
            fps=fps,
            ball_tracks=ball_tracks,
            rim_position=rim_position,
            rim_zone=rim_zone,
            net_zone=net_zone,
            yolo_detections=yolo_detections,
            yolo_rim_position=yolo_rim_position,
            transform_matrix=transform_matrix,
            transform_meta=transform_meta,
            court_points=court_points,
            first_frame=first_frame,
            camera_mode=camera_mode,
            camera_position=camera_position,
        )
        timing_debug['clip_seconds'] = float(analysis_result.get('clip_seconds', 0.0) or 0.0)
        session_id = analysis_result['session_id']
        video_url = analysis_result['video_url']
        shots_analysis = analysis_result['shots_analysis']
        zone_percentages = analysis_result['zone_percentages']
        total_makes = analysis_result['total_makes']
        shots_with_scores = analysis_result['shots_with_scores']
        unknown_zones = analysis_result['unknown_zones']
        outcome_reasons = analysis_result['outcome_reasons']
        ball_quality_counts = analysis_result['ball_quality_counts']
        classification_methods = analysis_result['classification_methods']
        candidate_source_counts = analysis_result['candidate_source_counts']
        anchor_source_counts = analysis_result['anchor_source_counts']
        unknown_zone_reasons = analysis_result['unknown_zone_reasons']
        session_summary = analysis_result.get('session_summary', {})

        processing_seconds = time.time() - overall_start
        _finalize_timing_debug(timing_debug, overall_start)
        log_shotlab_summary(
            processing_seconds,
            shots_analysis,
            pose_shot_count,
            dedupe_debug,
            filter_debug,
            ball_debug,
            rim_position,
            rim_zone,
            net_zone,
            transform_matrix,
            shot_detection_debug=shot_detection_debug,
            warnings=warnings,
            unknown_zones=unknown_zones,
            anchor_source_counts=anchor_source_counts,
            unknown_zone_reasons=unknown_zone_reasons,
        )
        debug_summary = {
            'court': court_debug,
            'pose': pose_debug,
            'ball': ball_debug,
            'rim': {
                'source': rim_source,
                'auto': rim_auto_debug,
                'available': rim_position is not None
            },
            'net_zone': {
                'source': net_zone_source,
                'available': net_zone is not None
            },
            'yolo': {
                'available': yolo_detector.available,
                'frames_analyzed': int(yolo_process_stats.get('processed_frames', len(yolo_detections))),
                'frames_returned': len(yolo_detections),
                'frame_skip': int(yolo_process_stats.get('frame_skip', yolo_frame_skip)),
                'interpolated_frames': int(yolo_process_stats.get('interpolated_frames', 0)),
                'rim_available': yolo_rim_position is not None,
                'launch_fallback': shot_detection_debug.get('yolo_launch', {}),
                'shoot_model': {
                    **dict(shot_detection_debug.get('shoot_model', {}) or {}),
                    'made_shadow': analysis_result.get('made_shadow', {})
                }
            },
            'timing': timing_debug,
            'shots': {
                'shot_attempts': len(shot_attempts),
                'shots_with_scores': shots_with_scores,
                'unknown_zones': unknown_zones,
                'outcome_reasons': dict(outcome_reasons),
                'ball_quality': ball_quality_counts,
                'classification_methods': dict(classification_methods),
                'candidate_source_counts': dict(candidate_source_counts),
                'anchor_source_counts': dict(anchor_source_counts),
                'unknown_zone_reasons': dict(unknown_zone_reasons),
                'ball_flight_confirmation': filter_debug.get('ball_flight_confirmation', {}) if filter_debug else {},
                'yolo_share_cap': filter_debug.get('yolo_share_cap', {}) if filter_debug else {}
            }
        }
        if session_id and session_id in shotlab_sessions:
            shotlab_sessions[session_id]['debug'] = debug_summary
            shotlab_sessions[session_id]['video_name'] = os.path.basename(video_path) if video_path else None
            _save_shotlab_session(session_id, shotlab_sessions[session_id])

        update_shotlab_status('complete', 'Analysis complete', 1.0)
        return jsonify({
            'success': True,
            'shots': shots_analysis,
            'fps': float(fps or 30.0),
            'zone_stats': zone_percentages,
            'total_attempts': len(shots_analysis),
            'total_makes': total_makes,
            'session_summary': session_summary,
            'warnings': warnings,
            'debug': debug_summary,
            'calibration': {
                'rim': rim_calibration,
                'rim_required': False,
                'net_zone': net_zone_calibration,
                'net_zone_required': False,
                'court': court_calibration,
                'court_required': (transform_matrix is None and not behind_basket_mode)
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

@app.route('/api/save_corrections', methods=['POST'])
def save_corrections():
    """Persist manual make/miss corrections from dashboard."""
    try:
        payload = request.get_json(silent=True) or {}
        corrections = payload.get('corrections')
        if not isinstance(corrections, list):
            return jsonify({'success': False, 'error': 'corrections must be an array'}), 400

        raw_video_name = str(payload.get('video_name') or 'unknown_video')
        safe_video_name = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(raw_video_name).name).strip('._')
        if not safe_video_name:
            safe_video_name = 'unknown_video'

        correction_dir = Path(__file__).parent / 'correction_data'
        correction_dir.mkdir(parents=True, exist_ok=True)
        output_path = correction_dir / f'{safe_video_name}_corrections.json'

        output_payload = {
            'video_name': raw_video_name,
            'saved_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'corrections': corrections,
        }
        output_path.write_text(json.dumps(output_payload, indent=2), encoding='utf-8')

        return jsonify({
            'success': True,
            'saved_count': len(corrections),
            'path': str(output_path),
        })
    except Exception as e:
        logger.exception('Failed to save corrections')
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
    debug_mode = os.environ.get('SHOTLAB_DEBUG', '0') == '1'
    use_reloader = os.environ.get('SHOTLAB_USE_RELOADER', '0') == '1'
    app.run(debug=debug_mode, use_reloader=use_reloader, host='0.0.0.0', port=port)
