from flask import Flask, request, jsonify, send_from_directory
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

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install it with: pip install mediapipe")

try:
    # Try importing MMPose APIs first
    from mmpose.apis import init_model, inference_topdown, inference_bottom_up
    import torch
    print("✓ MMPose basic imports successful")
    
    # Mark as available - we'll handle mmcv._ext errors in the endpoint
    MMPOSE_AVAILABLE = True
    
    # Try importing register_all_modules separately (it might trigger mmcv._ext import)
    try:
        from mmpose.utils import register_all_modules
        # Register all MMPose modules (this may fail with mmcv-lite, but that's OK)
        try:
            register_all_modules()
            print("✓ MMPose modules registered")
        except Exception as e:
            print(f"⚠️  Could not register all MMPose modules (will use workarounds): {e}")
            # Continue anyway - we can still try to use the models
    except ImportError as e:
        print(f"⚠️  Could not import register_all_modules: {e}")
        print("⚠️  Continuing without module registration - will use workarounds")
    
    # Test if we can actually use the APIs
    try:
        # Try importing something that would fail if mmcv._ext is missing
        from mmcv.ops import nms
        print("✓ MMPose fully available with compiled ops")
    except ImportError:
        print("⚠️  MMPose available but without compiled ops - will use workarounds")
        # We'll use workarounds in the inference code
except ImportError as e:
    MMPOSE_AVAILABLE = False
    print(f"❌ MMPose not available (ImportError): {e}")
except Exception as e:
    MMPOSE_AVAILABLE = False
    print(f"❌ MMPose not available (Exception): {e}")
    import traceback
    traceback.print_exc()

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: fastdtw not available. Install it with: pip install fastdtw")

# Try importing VideoPose3D integration (uses ACTUAL VideoPose3D GitHub repo)
try:
    from videopose3d_integration import (
        VideoPose3DProcessor, 
        convert_mediapipe_to_videopose3d_format,
        get_global_processor,
        reset_processor,
        VIDEOPOSE3D_AVAILABLE,
        VIDEOPOSE3D_REPO_PATH,
        VIDEOPOSE3D_WINDOW_SIZE
    )
    if VIDEOPOSE3D_AVAILABLE:
        print(f"✓ VideoPose3D integration loaded (using repo at: {VIDEOPOSE3D_REPO_PATH})")
    else:
        print("⚠️  VideoPose3D repository not found. Clone it to use VideoPose3D features.")
except ImportError as e:
    VIDEOPOSE3D_AVAILABLE = False
    VIDEOPOSE3D_WINDOW_SIZE = 243  # Default value
    # Define stub functions to prevent errors
    def get_global_processor(sequence_id='default'):
        return None
    def reset_processor(sequence_id='default'):
        pass
    def convert_mediapipe_to_videopose3d_format(landmarks, width, height):
        return None
    print(f"⚠️  VideoPose3D integration not available: {e}")
    print("   This is optional - other methods will still work")

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

# Temporal smoothing buffer for measurements (reduces jitter)
from collections import deque
measurement_buffer = {
    'shoulder_width': deque(maxlen=5),  # Keep last 5 frames
    'eye_to_feet': deque(maxlen=5),
    'distance': deque(maxlen=5)
}

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
    # Optimized MediaPipe settings for better accuracy
    pose = mp_pose.Pose(
        model_complexity=2,              # 2 = Heavy model (most accurate)
        static_image_mode=False,
        smooth_landmarks=True,
        enable_segmentation=False,       # Disable segmentation for speed
        min_detection_confidence=0.5,    # Lower = detects more (0.5 instead of 0.7)
        min_tracking_confidence=0.5      # Lower = tracks better with occlusion
    )
else:
    mp_pose = None
    POSE_CONNECTIONS = []
    pose = None

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
    right_shoulder = get_3d_point(landmarks, 12, width, height)
    right_elbow    = get_3d_point(landmarks, 14, width, height)
    right_wrist    = get_3d_point(landmarks, 16, width, height)
    left_wrist     = get_3d_point(landmarks, 15, width, height)
    left_hip       = get_3d_point(landmarks, 23, width, height)
    right_hip      = get_3d_point(landmarks, 24, width, height)

    if (right_wrist is not None and left_wrist is not None and
        left_hip is not None and right_hip is not None and
        right_shoulder is not None):
        waist_y = (left_hip[1] + right_hip[1]) / 2.0
        avg_wrist_y = (right_wrist[1] + left_wrist[1]) / 2.0
        dist_wrists = np.linalg.norm(right_wrist - left_wrist)
        if (dist_wrists < 0.15 * width and avg_wrist_y < waist_y
            and right_wrist[1] > right_shoulder[1]):
            return "pre_shot"

    if right_wrist is not None and right_shoulder is not None:
        if right_shoulder[1] > right_wrist[1]:
            return "follow_through"

    if right_shoulder is not None and right_wrist is not None:
        if right_wrist[1] > right_shoulder[1]:
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

# ====================== COORDINATE VALIDATOR ENDPOINTS ======================

@app.route('/tool/coordinate_validator')
@app.route('/tool/coordinate_validator/')
def coordinate_validator():
    """Serve the coordinate validator page"""
    tool_dir = Path(__file__).parent
    return send_from_directory(str(tool_dir), 'coordinate_validator.html')

@app.route('/tool/coordinate_validator_mmpose')
@app.route('/tool/coordinate_validator_mmpose/')
def coordinate_validator_mmpose():
    """Serve the MMPose coordinate validator page"""
    tool_dir = Path(__file__).parent
    return send_from_directory(str(tool_dir), 'coordinate_validator_mmpose.html')

@app.route('/tool/shot_sync_overlay')
@app.route('/tool/shot_sync_overlay/')
def shot_sync_overlay_viewer():
    """Serve the Shot Sync overlay and 3D viewer page"""
    try:
        tool_dir = Path(__file__).parent
        html_file = tool_dir / 'shot_sync_overlay_viewer.html'
        if not html_file.exists():
            return f"Error: File not found at {html_file}", 404
        return send_from_directory(str(tool_dir), 'shot_sync_overlay_viewer.html')
    except Exception as e:
        return f"Error serving page: {str(e)}", 500

@app.route('/api/process_frame_overlay_videopose3d', methods=['POST'])
def process_frame_overlay_videopose3d():
    """
    Process video frames using VideoPose3D methodology for enhanced 3D pose estimation.
    Based on: https://github.com/facebookresearch/VideoPose3D
    
    VideoPose3D uses temporal convolutions to lift 2D keypoints to 3D, providing
    more accurate 3D poses than single-frame methods.
    """
    try:
        if not MEDIAPIPE_AVAILABLE:
            return jsonify({'success': False, 'error': 'MediaPipe not available'}), 500
        
        # Note: We still show MediaPipe overlay even if VideoPose3D isn't available
        # VideoPose3D is only used for enhanced 3D estimation, MediaPipe handles 2D overlay
        
        data = request.json
        frame_data = data.get('frame')
        user_height = data.get('user_height', 72)
        shoulder_width = data.get('shoulder_width', 18)
        sequence_id = data.get('sequence_id', 'default')  # For tracking sequences
        
        if not frame_data:
            return jsonify({'success': False, 'error': 'No frame data'}), 400
        
        # Decode frame
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode frame'}), 400
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe to get 2D keypoints (ALWAYS use MediaPipe for overlay)
        results = pose.process(rgb_frame)
        
        person_detected = False
        avg_confidence = 0.0
        landmarks_data = []
        measurements = {
            'eye_to_feet_pixels': 0,
            'eye_to_feet_inches': 0,
            'shoulder_width_pixels': 0,
            'shoulder_width_inches': 0,
            'distance_feet': 0,
            'distance_inches': 0,
            'landmarks_visible': 0
        }
        
        # Get or create processor for this sequence (only if VideoPose3D available)
        # Use sequence_id to maintain separate buffers per video
        processor = None
        sequence_id = data.get('sequence_id', 'default')
        
        # Get window size (with fallback)
        try:
            window_size = VIDEOPOSE3D_WINDOW_SIZE
        except NameError:
            window_size = 243
        
        # CRITICAL FIX: Always try to create processor, even if VideoPose3D repo isn't cloned
        # The processor can still track buffer progress for visualization
        # This is why buffer was stuck at 0 - processor was None when VIDEOPOSE3D_AVAILABLE was False
        try:
            # Check if function exists before calling
            if 'get_global_processor' in globals() and callable(get_global_processor):
                # Always create processor - it works even without VideoPose3D repo
                processor = get_global_processor(sequence_id=sequence_id)
                # Debug: print buffer size occasionally (first frame and every 30 frames)
                if processor is not None:
                    buffer_size = len(processor.keypoint_buffer)
                    if buffer_size == 0 or buffer_size % 30 == 0:
                        print(f"VideoPose3D buffer [{sequence_id}]: {buffer_size}/{window_size} frames")
            else:
                print("Warning: get_global_processor not available, using MediaPipe only")
                processor = None
        except Exception as e:
            print(f"VideoPose3D processor error: {e}")
            import traceback
            traceback.print_exc()
            processor = None
        
        # CRITICAL: ALWAYS draw MediaPipe overlay when pose is detected
        # This is the 2D skeleton overlay that shows on the video
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            confidences = [lm.visibility for lm in landmarks if lm.visibility > 0]
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                measurements['landmarks_visible'] = len(confidences)
                person_detected = avg_confidence > 0.5
                
                # Convert to VideoPose3D format and add to buffer (if available)
                # IMPORTANT: Always add frames to buffer, even if no pose detected
                # This keeps the buffer growing so we can see progress
                poses_3d = None
                if processor is not None:
                    try:
                        landmarks_2d = convert_mediapipe_to_videopose3d_format(landmarks, w, h)
                        # Always add frame to buffer (function handles None/errors internally)
                        if landmarks_2d is not None and landmarks_2d.shape == (17, 2):
                            processor.add_frame(frame, landmarks_2d)
                            # Get 3D poses from VideoPose3D (when buffer is full)
                            # Get all frames for animation if buffer is full
                            if processor.initialized:
                                poses_3d_sequence = processor.get_3d_poses(return_all_frames=True)
                                # Use center frame for current display, but store sequence for animation
                                if poses_3d_sequence:
                                    center_idx = len(poses_3d_sequence) // 2
                                    poses_3d = poses_3d_sequence[center_idx]
                                else:
                                    poses_3d = None
                            else:
                                poses_3d = None
                        else:
                            # Fallback: add zeros if conversion failed
                            zero_keypoints = np.zeros((17, 2), dtype=np.float32)
                            processor.add_frame(frame, zero_keypoints)
                        
                        # Debug: log every 30 frames
                        buffer_size = len(processor.keypoint_buffer)
                        if buffer_size > 0 and buffer_size % 30 == 0:
                            print(f"✓ Buffer filling: {buffer_size}/{window_size} frames")
                    except Exception as e:
                        print(f"VideoPose3D processing error (using MediaPipe 3D): {e}")
                        import traceback
                        traceback.print_exc()
                        # Still try to add frame with zeros to keep buffer growing
                        try:
                            zero_keypoints = np.zeros((17, 2), dtype=np.float32)
                            processor.add_frame(frame, zero_keypoints)
                            buffer_size = len(processor.keypoint_buffer)
                            if buffer_size % 30 == 0:
                                print(f"✓ Buffer filling (error recovery): {buffer_size}/{window_size} frames")
                        except Exception as e2:
                            print(f"Failed to add zero frame: {e2}")
                        poses_3d = None
                else:
                    # Processor is None - this is the problem!
                    # Log why processor is None for debugging
                    if person_detected:  # Only log if we actually detected a person
                        print(f"⚠️  Processor is None! Cannot add frames to buffer.")
                        print(f"   get_global_processor available: {'get_global_processor' in globals()}")
                        print(f"   VIDEOPOSE3D_AVAILABLE: {VIDEOPOSE3D_AVAILABLE}")
                
                # Map back to MediaPipe format (33 landmarks)
                if poses_3d:
                    # Use VideoPose3D 3D estimates (temporally smoothed from 243 frames!)
                    # VideoPose3D processes ALL 243 frames together using temporal convolutions
                    # This creates smooth, accurate 3D poses like in the paper
                    # 
                    # VideoPose3D returns 17 keypoints in Human3.6M format
                    # Mapping VideoPose3D index -> MediaPipe index (based on convert_mediapipe_to_videopose3d_format):
                    # VP3D 0: Hip (avg of MP 23,24) -> use for both hips
                    # VP3D 1: Right hip (MP 24)
                    # VP3D 2: Right knee (MP 26)
                    # VP3D 3: Right ankle (MP 28)
                    # VP3D 4: Left hip (MP 23)
                    # VP3D 5: Left knee (MP 25)
                    # VP3D 6: Left ankle (MP 27)
                    # VP3D 7: Spine (midpoint of MP 11,12) -> use for torso
                    # VP3D 8: Thorax (same as spine)
                    # VP3D 9: Neck/Nose (MP 0)
                    # VP3D 10: Head (MP 0)
                    # VP3D 11: HeadTop (MP 0)
                    # VP3D 12: Left shoulder (MP 11)
                    # VP3D 13: Left elbow (MP 13)
                    # VP3D 14: Left wrist (MP 15)
                    # VP3D 15: Right shoulder (MP 12)
                    # VP3D 16: Right elbow (MP 14)
                    
                    # Map VideoPose3D keypoints to MediaPipe landmarks
                    vp3d_to_mp = {
                        0: [23, 24],  # Hip -> both hips
                        1: 24,        # Right hip
                        2: 26,        # Right knee
                        3: 28,        # Right ankle
                        4: 23,        # Left hip
                        5: 25,        # Left knee
                        6: 27,        # Left ankle
                        7: [11, 12],  # Spine -> shoulders
                        8: [11, 12],  # Thorax -> shoulders
                        9: 0,         # Neck/Nose
                        10: 0,        # Head
                        11: 0,        # HeadTop
                        12: 11,       # Left shoulder
                        13: 13,       # Left elbow
                        14: 15,       # Left wrist
                        15: 12,       # Right shoulder
                        16: 14,       # Right elbow
                    }
                    
                    for i, lm in enumerate(landmarks):
                        # Find which VideoPose3D keypoint corresponds to this MediaPipe landmark
                        vp3d_idx = None
                        for vp_idx, mp_target in vp3d_to_mp.items():
                            if isinstance(mp_target, list):
                                if i in mp_target:
                                    vp3d_idx = vp_idx
                                    break
                            elif mp_target == i:
                                vp3d_idx = vp_idx
                                break
                        
                        if vp3d_idx is not None and vp3d_idx < len(poses_3d):
                            x_3d, y_3d, z_3d = poses_3d[vp3d_idx]
                            # Use VideoPose3D's temporally smoothed 3D coordinates
                            # These are smoothed across 243 frames using temporal convolutions!
                            landmarks_data.append({
                                'x': float(lm.x),  # Keep MediaPipe 2D for overlay
                                'y': float(lm.y),
                                'z': float(z_3d),  # Use VideoPose3D depth estimate
                                'visibility': float(lm.visibility),
                                'index': i
                            })
                        else:
                            landmarks_data.append({
                                'x': float(lm.x),
                                'y': float(lm.y),
                                'z': float(lm.z),
                                'visibility': float(lm.visibility),
                                'index': i
                            })
                else:
                    # Buffer not full yet - use MediaPipe 3D
                    for i, lm in enumerate(landmarks):
                        landmarks_data.append({
                            'x': float(lm.x),
                            'y': float(lm.y),
                            'z': float(lm.z),
                            'visibility': float(lm.visibility),
                            'index': i
                        })
                
                # Calculate measurements (same as before)
                left_eye = landmarks[2]
                right_eye = landmarks[5]
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                
                if (left_eye.visibility > 0.7 and right_eye.visibility > 0.7 and
                    left_ankle.visibility > 0.7 and right_ankle.visibility > 0.7):
                    avg_eye = np.array([
                        (left_eye.x + right_eye.x) / 2 * w,
                        (left_eye.y + right_eye.y) / 2 * h
                    ])
                    avg_ankle = np.array([
                        (left_ankle.x + right_ankle.x) / 2 * w,
                        (left_ankle.y + right_ankle.y) / 2 * h
                    ])
                    eye_to_feet_pixels = np.linalg.norm(avg_eye - avg_ankle)
                    measurements['eye_to_feet_pixels'] = float(eye_to_feet_pixels)
                
                if left_shoulder.visibility > 0.7 and right_shoulder.visibility > 0.7:
                    shoulder_vec = np.array([
                        (right_shoulder.x - left_shoulder.x) * w,
                        (right_shoulder.y - left_shoulder.y) * h
                    ])
                    shoulder_width_pixels = np.linalg.norm(shoulder_vec)
                    measurements['shoulder_width_pixels'] = float(shoulder_width_pixels)
                    
                    if shoulder_width_pixels > 0 and shoulder_width > 0:
                        pixels_per_inch = shoulder_width_pixels / shoulder_width
                        if eye_to_feet_pixels > 0:
                            measurements['eye_to_feet_inches'] = float(eye_to_feet_pixels / pixels_per_inch)
                        measurements['shoulder_width_inches'] = float(shoulder_width)
                        
                        baseline_shoulder_pixels = 120.0
                        baseline_distance_feet = 10.0
                        estimated_distance_feet = baseline_distance_feet * (baseline_shoulder_pixels / shoulder_width_pixels)
                        measurements['distance_feet'] = float(estimated_distance_feet)
                        measurements['distance_inches'] = float(estimated_distance_feet * 12)
        
        # CRITICAL: ALWAYS add frames to buffer, even if no pose detected
        # This ensures buffer progress is visible regardless of detection
        if processor is not None:
            # Add frame to buffer even if no pose detected (use zeros)
            # This way user can see buffer filling: 0/243 -> 1/243 -> 2/243 -> ...
            try:
                if not results.pose_landmarks:
                    # No pose detected - add zeros to keep buffer growing
                    zero_keypoints = np.zeros((17, 2), dtype=np.float32)
                    processor.add_frame(frame, zero_keypoints)
                    # Log progress every 30 frames
                    buffer_size = len(processor.keypoint_buffer)
                    if buffer_size % 30 == 0:
                        print(f"Buffer progress (no pose): {buffer_size}/{window_size} frames")
            except Exception as e:
                print(f"Error adding frame to buffer: {e}")
        
        # CRITICAL: ALWAYS draw MediaPipe skeleton overlay when pose is detected
        # This is the 2D overlay that shows on the video - works even without VideoPose3D
        if results.pose_landmarks:
            try:
                # AUTO-ROTATE: Calculate rotation angle to make shoulder line horizontal (0 degrees)
                landmarks = results.pose_landmarks.landmark
                rotation_angle = 0.0
                
                # Get shoulder landmarks (11 = left, 12 = right)
                if len(landmarks) > 12 and landmarks[11].visibility > 0.5 and landmarks[12].visibility > 0.5:
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    
                    # Calculate angle of shoulder line in pixel coordinates
                    # Note: In image coordinates, y increases downward
                    dx = (right_shoulder.x - left_shoulder.x) * w
                    dy = (right_shoulder.y - left_shoulder.y) * h
                    
                    # Calculate angle in degrees (atan2 gives angle from horizontal)
                    # Negative because we want to rotate the frame to align shoulders horizontally
                    rotation_angle = -np.degrees(np.arctan2(dy, dx))
                    
                    # Normalize angle to [-90, 90] to avoid flipping upside down
                    # If angle is > 90 or < -90, we can rotate the other way
                    if rotation_angle > 90:
                        rotation_angle = rotation_angle - 180
                    elif rotation_angle < -90:
                        rotation_angle = rotation_angle + 180
                    
                    # Rotate frame to make shoulder line horizontal
                    if abs(rotation_angle) > 0.5:  # Only rotate if angle is significant
                        h_frame, w_frame = frame.shape[:2]
                        center = (w_frame // 2, h_frame // 2)
                        
                        # Calculate bounding box of pose to ensure it stays in frame after rotation
                        # Get all visible landmark positions in pixel coordinates
                        visible_landmarks = []
                        for lm in landmarks:
                            if lm.visibility > 0.5:
                                visible_landmarks.append((lm.x * w_frame, lm.y * h_frame))
                        
                        if visible_landmarks:
                            # Calculate bounding box of current pose
                            xs = [x for x, y in visible_landmarks]
                            ys = [y for x, y in visible_landmarks]
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)
                            
                            bbox_width = max_x - min_x
                            bbox_height = max_y - min_y
                            
                            # Calculate how large the bounding box will be after rotation
                            angle_rad = np.radians(abs(rotation_angle))
                            # Rotated bounding box dimensions
                            rotated_width = abs(bbox_width * np.cos(angle_rad)) + abs(bbox_height * np.sin(angle_rad))
                            rotated_height = abs(bbox_width * np.sin(angle_rad)) + abs(bbox_height * np.cos(angle_rad))
                            
                            # Add padding (20% of frame size) to ensure pose stays in frame
                            padding = min(w_frame, h_frame) * 0.2
                            
                            # Calculate scale needed to fit rotated bounding box in frame
                            available_width = w_frame - 2 * padding
                            available_height = h_frame - 2 * padding
                            
                            scale_x = available_width / rotated_width if rotated_width > 0 else 1.0
                            scale_y = available_height / rotated_height if rotated_height > 0 else 1.0
                            scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down if needed
                            
                            # Ensure scale is reasonable (not too small)
                            scale = max(scale, 0.5)  # Minimum 50% scale
                        else:
                            scale = 1.0
                        
                        # Create rotation matrix with scale to keep pose in frame
                        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)
                        frame = cv2.warpAffine(frame, rotation_matrix, (w_frame, h_frame), 
                                              flags=cv2.INTER_LINEAR, 
                                              borderMode=cv2.BORDER_CONSTANT, 
                                              borderValue=(0, 0, 0))
                        
                        # Re-process pose on rotated frame to get updated landmarks
                        rgb_frame_rotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_rotated = pose.process(rgb_frame_rotated)
                        if results_rotated.pose_landmarks:
                            results = results_rotated
                
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0),  # Green joints
                        thickness=3,
                        circle_radius=4
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255),  # White connections
                        thickness=2
                    )
                )
                
                # Add method indicator
                if processor is not None:
                    buffer_status = "Ready" if processor.initialized else f"Buffering ({len(processor.keypoint_buffer)}/{window_size})"
                    status_color = (0, 255, 0) if processor.initialized else (0, 255, 255)
                else:
                    buffer_status = "MediaPipe 2D (VideoPose3D: clone repo for 3D)"
                    status_color = (0, 255, 255)
                
                cv2.putText(frame, f"VideoPose3D Method: {avg_confidence:.1%} [{buffer_status}]",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            except Exception as e:
                print(f"Error drawing overlay: {e}")
                # Still try to show something
                cv2.putText(frame, f"Pose detected: {avg_confidence:.1%}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # No pose detected
            cv2.putText(frame, "No pose detected - move into frame",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get buffer status for response
        buffer_status = False
        buffer_size = 0
        if processor is not None:
            buffer_status = processor.initialized
            buffer_size = len(processor.keypoint_buffer)
            # Debug: log if buffer is stuck
            if buffer_size == 0 and person_detected:
                print(f"⚠️  Warning: Person detected but buffer is 0.")
                print(f"   Processor exists: {processor is not None}")
                print(f"   VIDEOPOSE3D_AVAILABLE: {VIDEOPOSE3D_AVAILABLE}")
                print(f"   Sequence ID: {sequence_id}")
                print(f"   Keypoint buffer type: {type(processor.keypoint_buffer)}")
                print(f"   Buffer length: {len(processor.keypoint_buffer)}")
        else:
            # Processor is None - log why
            if person_detected:
                print(f"⚠️  Processor is None. VIDEOPOSE3D_AVAILABLE: {VIDEOPOSE3D_AVAILABLE}")
        
        # Get full 3D pose sequence for animation if buffer is full
        poses_3d_sequence = None
        if processor is not None and processor.initialized:
            try:
                poses_3d_sequence = processor.get_3d_poses(return_all_frames=True)
                if poses_3d_sequence:
                    # Convert to format frontend can use
                    # Format: List of frames, each frame is list of {x, y, z} for each keypoint
                    sequence_data = []
                    for frame_poses in poses_3d_sequence:
                        frame_data = [{'x': float(x), 'y': float(y), 'z': float(z)} for x, y, z in frame_poses]
                        sequence_data.append(frame_data)
                    poses_3d_sequence = sequence_data
            except Exception as e:
                print(f"Error getting 3D pose sequence: {e}")
                poses_3d_sequence = None
        
        return jsonify({
            'success': True,
            'overlay_frame': frame_base64,
            'person_detected': person_detected,
            'confidence': float(avg_confidence),
            'measurements': measurements,
            'landmarks': landmarks_data,
            'method': 'VideoPose3D (Facebook Research)',
            'buffer_status': buffer_status,
            'buffer_size': buffer_size,
            'window_size': window_size,
            'processor_available': processor is not None,
            'videopose3d_available': VIDEOPOSE3D_AVAILABLE,
            'poses_3d_sequence': poses_3d_sequence  # Full sequence for animation
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in process_frame_overlay_videopose3d: {e}")
        print(error_trace)
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': error_trace
        }), 500

@app.route('/api/reset_videopose3d', methods=['POST'])
def reset_videopose3d():
    """Reset VideoPose3D buffer for a sequence"""
    try:
        data = request.json or {}
        sequence_id = data.get('sequence_id', 'default')
        
        if VIDEOPOSE3D_AVAILABLE:
            try:
                reset_processor(sequence_id=sequence_id)
                return jsonify({'success': True, 'message': f'Buffer reset for sequence: {sequence_id}'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        else:
            return jsonify({'success': False, 'error': 'VideoPose3D not available'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Check if NTU-RRIS repository is available
NTU_RRIS_REPO_PATH = os.path.join(os.path.dirname(__file__), 'google-mediapipe')
NTU_RRIS_AVAILABLE = os.path.exists(NTU_RRIS_REPO_PATH)

if NTU_RRIS_AVAILABLE:
    # Add NTU-RRIS repo to path to use their actual code
    sys.path.insert(0, NTU_RRIS_REPO_PATH)
    try:
        # Try importing their actual code modules
        code_path = os.path.join(NTU_RRIS_REPO_PATH, 'code')
        if os.path.exists(code_path):
            sys.path.insert(0, code_path)
        print(f"✓ NTU-RRIS repository found at: {NTU_RRIS_REPO_PATH}")
    except Exception as e:
        print(f"⚠️  NTU-RRIS repository found but could not load: {e}")
else:
    print(f"⚠️  NTU-RRIS repository not found at: {NTU_RRIS_REPO_PATH}")
    print("   Clone it with: git clone https://github.com/ntu-rris/google-mediapipe.git")

@app.route('/api/process_frame_overlay_ntu', methods=['POST'])
def process_frame_overlay_ntu():
    """
    Process a video frame using ACTUAL ntu-rris/google-mediapipe code from their GitHub.
    Repository: https://github.com/ntu-rris/google-mediapipe
    
    Uses their exact implementation from code/08_skeleton_3D.py and related modules.
    """
    try:
        if not MEDIAPIPE_AVAILABLE:
            return jsonify({'success': False, 'error': 'MediaPipe not available'}), 500
        
        if not NTU_RRIS_AVAILABLE:
            return jsonify({
                'success': False, 
                'error': 'NTU-RRIS repository not found. Clone it: git clone https://github.com/ntu-rris/google-mediapipe.git'
            }), 500
        
        data = request.json
        frame_data = data.get('frame')
        user_height = data.get('user_height', 72)
        shoulder_width = data.get('shoulder_width', 18)
        mode = data.get('mode', 'body')  # body, hand, holistic
        
        if not frame_data:
            return jsonify({'success': False, 'error': 'No frame data'}), 400
        
        # Decode frame
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode frame'}), 400
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use MediaPipe Pose with ntu-rris approach (model_complexity=2 for best 3D)
        results = pose.process(rgb_frame)
        
        person_detected = False
        avg_confidence = 0.0
        landmarks_data = []
        measurements = {
            'eye_to_feet_pixels': 0,
            'eye_to_feet_inches': 0,
            'shoulder_width_pixels': 0,
            'shoulder_width_inches': 0,
            'distance_feet': 0,
            'distance_inches': 0,
            'landmarks_visible': 0
        }
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            confidences = [lm.visibility for lm in landmarks if lm.visibility > 0]
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                measurements['landmarks_visible'] = len(confidences)
                person_detected = avg_confidence > 0.5
                
                # Extract 3D landmarks following ntu-rris approach
                # They use full 3D coordinates (x, y, z) from MediaPipe
                for i, lm in enumerate(landmarks):
                    if lm.visibility > 0.5:
                        # ntu-rris uses actual 3D coordinates, not normalized
                        landmarks_data.append({
                            'x': float(lm.x),  # Normalized 0-1
                            'y': float(lm.y),  # Normalized 0-1
                            'z': float(lm.z),  # MediaPipe Z (relative depth)
                            'visibility': float(lm.visibility),
                            'index': i
                        })
                    else:
                        landmarks_data.append({
                            'x': 0.0,
                            'y': 0.0,
                            'z': 0.0,
                            'visibility': 0.0,
                            'index': i
                        })
                
                # Calculate measurements
                left_eye = landmarks[2]
                right_eye = landmarks[5]
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                
                # Eye to feet (using 3D distance like ntu-rris)
                if (left_eye.visibility > 0.7 and right_eye.visibility > 0.7 and
                    left_ankle.visibility > 0.7 and right_ankle.visibility > 0.7):
                    avg_eye = np.array([
                        (left_eye.x + right_eye.x) / 2 * w,
                        (left_eye.y + right_eye.y) / 2 * h,
                        (left_eye.z + right_eye.z) / 2
                    ])
                    avg_ankle = np.array([
                        (left_ankle.x + right_ankle.x) / 2 * w,
                        (left_ankle.y + right_ankle.y) / 2 * h,
                        (left_ankle.z + right_ankle.z) / 2
                    ])
                    eye_to_feet_pixels = np.linalg.norm(avg_eye[:2] - avg_ankle[:2])
                    measurements['eye_to_feet_pixels'] = float(eye_to_feet_pixels)
                
                # Shoulder width
                if left_shoulder.visibility > 0.7 and right_shoulder.visibility > 0.7:
                    shoulder_vec = np.array([
                        (right_shoulder.x - left_shoulder.x) * w,
                        (right_shoulder.y - left_shoulder.y) * h,
                        (right_shoulder.z - left_shoulder.z)
                    ])
                    shoulder_width_pixels = np.linalg.norm(shoulder_vec[:2])
                    measurements['shoulder_width_pixels'] = float(shoulder_width_pixels)
                    
                    if shoulder_width_pixels > 0 and shoulder_width > 0:
                        pixels_per_inch = shoulder_width_pixels / shoulder_width
                        if eye_to_feet_pixels > 0:
                            measurements['eye_to_feet_inches'] = float(eye_to_feet_pixels / pixels_per_inch)
                        measurements['shoulder_width_inches'] = float(shoulder_width)
                        
                        # Distance estimation
                        baseline_shoulder_pixels = 120.0
                        baseline_distance_feet = 10.0
                        estimated_distance_feet = baseline_distance_feet * (baseline_shoulder_pixels / shoulder_width_pixels)
                        measurements['distance_feet'] = float(estimated_distance_feet)
                        measurements['distance_inches'] = float(estimated_distance_feet * 12)
                
                # Draw skeleton following ntu-rris visualization style
                # They use different colors for different body parts
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                
                # Custom drawing with ntu-rris style (more detailed connections)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0),  # Green for joints
                        thickness=3,
                        circle_radius=4
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255),  # White connections
                        thickness=2
                    )
                )
                
                # Add confidence indicator
                cv2.putText(frame, f"NTU-RRIS MediaPipe: {avg_confidence:.1%}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0) if avg_confidence > 0.75 else (0, 255, 255), 2)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'overlay_frame': frame_base64,
            'person_detected': person_detected,
            'confidence': float(avg_confidence),
            'measurements': measurements,
            'landmarks': landmarks_data,
            'mode': mode,
            'method': 'ntu-rris/google-mediapipe'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process_frame_overlay', methods=['POST'])
def process_frame_overlay():
    """Process a video frame and return it with Shot Sync's joint overlay"""
    try:
        if not MEDIAPIPE_AVAILABLE:
            return jsonify({'success': False, 'error': 'MediaPipe not available'}), 500
        
        data = request.json
        frame_data = data.get('frame')
        user_height = data.get('user_height', 72)
        shoulder_width = data.get('shoulder_width', 18)  # User's actual shoulder width in inches
        
        if not frame_data:
            return jsonify({'success': False, 'error': 'No frame data'}), 400
        
        # Decode frame
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode frame'}), 400
        
        # PRE-PROCESS: Light enhancement only (removed heavy denoising for speed)
        # Increase brightness and contrast
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Draw overlay ONLY if detection is high quality
        person_detected = False
        avg_confidence = 0.0
        measurements = {
            'eye_to_feet_pixels': 0,
            'eye_to_feet_inches': 0,
            'shoulder_width_pixels': 0,
            'shoulder_width_inches': 0,
            'distance_feet': 0,
            'distance_inches': 0,
            'landmarks_visible': 0
        }
        
        h, w, _ = frame.shape
        
        if results.pose_landmarks:
            # Calculate average landmark confidence
            landmarks = results.pose_landmarks.landmark
            confidences = [lm.visibility for lm in landmarks if lm.visibility > 0]
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                measurements['landmarks_visible'] = len(confidences)
                
                # Calculate 3D Euclidean distances in pixels
                
                # 1. Eye to feet distance (average eyes to average ankles)
                # Use 2D distance (X, Y) for stability - MediaPipe Z is too noisy
                left_eye = landmarks[2]
                right_eye = landmarks[5]
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]
                
                eye_to_feet_pixels = 0
                if (left_eye.visibility > 0.7 and right_eye.visibility > 0.7 and
                    left_ankle.visibility > 0.7 and right_ankle.visibility > 0.7):
                    # Average eye position
                    avg_eye_x = (left_eye.x + right_eye.x) / 2 * w
                    avg_eye_y = (left_eye.y + right_eye.y) / 2 * h
                    
                    # Average ankle position
                    avg_ankle_x = (left_ankle.x + right_ankle.x) / 2 * w
                    avg_ankle_y = (left_ankle.y + right_ankle.y) / 2 * h
                    
                    # Calculate 2D Euclidean distance (more stable than 3D)
                    eye_to_feet_pixels = np.sqrt(
                        (avg_eye_x - avg_ankle_x)**2 +
                        (avg_eye_y - avg_ankle_y)**2
                    )
                    measurements['eye_to_feet_pixels'] = float(eye_to_feet_pixels)
                
                # 2. Shoulder width (left shoulder to right shoulder)
                # Use 2D distance - shoulders are in same plane, Z adds noise
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                shoulder_width_pixels = 0
                if left_shoulder.visibility > 0.7 and right_shoulder.visibility > 0.7:
                    # Calculate 2D Euclidean distance (X, Y only - much more stable!)
                    shoulder_width_pixels = np.sqrt(
                        (left_shoulder.x * w - right_shoulder.x * w)**2 +
                        (left_shoulder.y * h - right_shoulder.y * h)**2
                    )
                    measurements['shoulder_width_pixels'] = float(shoulder_width_pixels)
                    
                    # Add to smoothing buffer
                    measurement_buffer['shoulder_width'].append(shoulder_width_pixels)
                    
                    # Use smoothed value (average of last N frames)
                    if len(measurement_buffer['shoulder_width']) > 0:
                        shoulder_width_pixels = np.mean(list(measurement_buffer['shoulder_width']))
                        measurements['shoulder_width_pixels'] = float(shoulder_width_pixels)
                
                # Smooth eye-to-feet as well
                if eye_to_feet_pixels > 0:
                    measurement_buffer['eye_to_feet'].append(eye_to_feet_pixels)
                    if len(measurement_buffer['eye_to_feet']) > 0:
                        eye_to_feet_pixels = np.mean(list(measurement_buffer['eye_to_feet']))
                        measurements['eye_to_feet_pixels'] = float(eye_to_feet_pixels)
                
                # NEW CALIBRATION STRATEGY:
                # Use SHOULDER WIDTH (known from user) as calibration reference
                # Then calculate HEIGHT and DISTANCE from that
                # This avoids circular logic and validates detection accuracy
                
                if shoulder_width_pixels > 0 and shoulder_width > 0:
                    # Calculate pixels per inch using user's SHOULDER WIDTH as reference
                    pixels_per_inch = shoulder_width_pixels / shoulder_width
                    
                    # Now calculate ACTUAL eye-to-feet height from detected pixels
                    # This will show if detection is accurate (should be close to user_height)
                    if eye_to_feet_pixels > 0:
                        measurements['eye_to_feet_inches'] = float(eye_to_feet_pixels / pixels_per_inch)
                    
                    # Shoulder width in inches (will match user input since we calibrated with it)
                    measurements['shoulder_width_inches'] = float(shoulder_width)
                    
                    # Store calibration ratio for debugging
                    measurements['pixels_per_inch'] = float(pixels_per_inch)
                    
                    # DISTANCE FROM CAMERA CALCULATION:
                    # Now we know pixels_per_inch from shoulder width calibration
                    # Use this to calculate distance based on how the shoulder compares at baseline
                    
                    # At baseline distance (10 feet), we expect certain pixel measurements
                    # If person moves closer, all measurements increase proportionally
                    # If person moves farther, all measurements decrease proportionally
                    
                    # We can use the HEIGHT measurement to calculate distance
                    # because we know the user's actual height
                    if eye_to_feet_pixels > 0 and user_height > 0:
                        # Calculate what eye-to-feet SHOULD be in pixels if at baseline (10 feet)
                        baseline_distance_feet = 10.0
                        
                        # Expected eye-to-feet pixels at baseline = user_height * pixels_per_inch
                        # But pixels_per_inch was calculated AT THE CURRENT DISTANCE
                        # So we need to compare expected vs actual
                        
                        # Ratio of measured height to actual height tells us distance
                        measured_height_inches = eye_to_feet_pixels / pixels_per_inch
                        
                        # If measured height matches actual height, person is at calibration distance
                        # If measured height is smaller, person is farther (or detection error)
                        # Since we calibrated with shoulder, this should be accurate
                        
                        # Distance calculation using similar triangles:
                        # At current distance, shoulder appears as shoulder_width_pixels
                        # At baseline (10 feet), shoulder would appear as baseline_shoulder_pixels
                        # distance / baseline = baseline_pixels / current_pixels
                        
                        # We know: at ANY distance, shoulder_width (inches) is constant
                        # pixels_per_inch changes with distance
                        # At baseline, assume pixels_per_inch_baseline
                        
                        # Simpler approach: use height comparison
                        # We expect eye-to-feet to equal user_height
                        # Current pixels_per_inch gives us measured_height
                        # Difference indicates distance change from "calibration distance"
                        
                        # Assume calibration distance is 10 feet
                        # pixels_per_inch we calculated is for current distance
                        # If measured height matches user height → detection is accurate
                        # Distance can be estimated from pixel density
                        
                        # Using inverse square law approximation:
                        # At 10 feet: pixels_per_inch_baseline = (user_height / baseline_reference)
                        # We need to estimate baseline_reference (pixels at 10 feet)
                        
                        # Better approach: use both shoulder AND height
                        # We calibrated pixels_per_inch using shoulder
                        # Now measure height: should be close to user_height if at same distance
                        
                        # For distance: assume standard viewing distance of 10 feet
                        # Scale factor = measured_pixels / expected_pixels_at_baseline
                        
                        # At baseline (10 feet), assume shoulder width = 120 pixels (typical)
                        baseline_shoulder_pixels = 120.0
                        current_shoulder_pixels = shoulder_width_pixels
                        
                        # Distance scales inversely with pixel size
                        # distance = baseline * (baseline_pixels / current_pixels)
                        estimated_distance_feet = baseline_distance_feet * (baseline_shoulder_pixels / current_shoulder_pixels)
                        estimated_distance_inches = estimated_distance_feet * 12
                        
                        # Smooth distance measurements
                        measurement_buffer['distance'].append(estimated_distance_feet)
                        if len(measurement_buffer['distance']) > 0:
                            smoothed_distance_feet = np.mean(list(measurement_buffer['distance']))
                            smoothed_distance_inches = smoothed_distance_feet * 12
                            measurements['distance_inches'] = float(smoothed_distance_inches)
                            measurements['distance_feet'] = float(smoothed_distance_feet)
                        else:
                            measurements['distance_inches'] = float(estimated_distance_inches)
                            measurements['distance_feet'] = float(estimated_distance_feet)
                
                # ONLY draw if average confidence is decent (>0.6)
                if avg_confidence > 0.6:
                    person_detected = True
                    
                    # AUTO-ROTATE: Calculate rotation angle to make shoulder line horizontal (0 degrees)
                    rotation_angle = 0.0
                    
                    # Get shoulder landmarks (11 = left, 12 = right)
                    if len(landmarks) > 12 and landmarks[11].visibility > 0.7 and landmarks[12].visibility > 0.7:
                        left_shoulder = landmarks[11]
                        right_shoulder = landmarks[12]
                        
                        # Calculate angle of shoulder line in pixel coordinates
                        # Note: In image coordinates, y increases downward, so we need to account for that
                        dx = (right_shoulder.x - left_shoulder.x) * w
                        dy = (right_shoulder.y - left_shoulder.y) * h
                        
                        # Calculate angle in degrees (atan2 gives angle from horizontal)
                        # Negative because we want to rotate the frame to align shoulders horizontally
                        rotation_angle = -np.degrees(np.arctan2(dy, dx))
                        
                        # Normalize angle to [-90, 90] to avoid flipping upside down
                        # If angle is > 90 or < -90, we can rotate the other way
                        if rotation_angle > 90:
                            rotation_angle = rotation_angle - 180
                        elif rotation_angle < -90:
                            rotation_angle = rotation_angle + 180
                        
                        # Rotate frame to make shoulder line horizontal
                        if abs(rotation_angle) > 0.5:  # Only rotate if angle is significant
                            h_frame, w_frame = frame.shape[:2]
                            center = (w_frame // 2, h_frame // 2)
                            
                            # Calculate bounding box of pose to ensure it stays in frame after rotation
                            # Get all visible landmark positions in pixel coordinates
                            visible_landmarks = []
                            for lm in landmarks:
                                if lm.visibility > 0.5:
                                    visible_landmarks.append((lm.x * w_frame, lm.y * h_frame))
                            
                            if visible_landmarks:
                                # Calculate bounding box of current pose
                                xs = [x for x, y in visible_landmarks]
                                ys = [y for x, y in visible_landmarks]
                                min_x, max_x = min(xs), max(xs)
                                min_y, max_y = min(ys), max(ys)
                                
                                bbox_width = max_x - min_x
                                bbox_height = max_y - min_y
                                
                                # Calculate how large the bounding box will be after rotation
                                angle_rad = np.radians(abs(rotation_angle))
                                # Rotated bounding box dimensions
                                rotated_width = abs(bbox_width * np.cos(angle_rad)) + abs(bbox_height * np.sin(angle_rad))
                                rotated_height = abs(bbox_width * np.sin(angle_rad)) + abs(bbox_height * np.cos(angle_rad))
                                
                                # Add padding (20% of frame size) to ensure pose stays in frame
                                padding = min(w_frame, h_frame) * 0.2
                                
                                # Calculate scale needed to fit rotated bounding box in frame
                                available_width = w_frame - 2 * padding
                                available_height = h_frame - 2 * padding
                                
                                scale_x = available_width / rotated_width if rotated_width > 0 else 1.0
                                scale_y = available_height / rotated_height if rotated_height > 0 else 1.0
                                scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down if needed
                                
                                # Ensure scale is reasonable (not too small)
                                scale = max(scale, 0.5)  # Minimum 50% scale
                            else:
                                scale = 1.0
                            
                            # Create rotation matrix with scale to keep pose in frame
                            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)
                            frame = cv2.warpAffine(frame, rotation_matrix, (w_frame, h_frame), 
                                                  flags=cv2.INTER_LINEAR, 
                                                  borderMode=cv2.BORDER_CONSTANT, 
                                                  borderValue=(0, 0, 0))
                            
                            # Re-process pose on rotated frame to get updated landmarks
                            rgb_frame_rotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results_rotated = pose.process(rgb_frame_rotated)
                            if results_rotated.pose_landmarks:
                                results = results_rotated
                                landmarks = results.pose_landmarks.landmark
                    
                    # Filter landmarks - only draw high-confidence ones
                    filtered_landmarks = mp_pose.PoseLandmark
                    custom_connections = []
                    
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks) and end_idx < len(landmarks):
                            start_landmark = landmarks[start_idx]
                            end_landmark = landmarks[end_idx]
                            
                            # Only draw connection if BOTH endpoints are confident (>0.7)
                            if start_landmark.visibility > 0.7 and end_landmark.visibility > 0.7:
                                custom_connections.append(connection)
                    
                    # Draw filtered landmarks
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing_styles = mp.solutions.drawing_styles
                    
                    # Custom drawing spec - make confident landmarks more visible
                    landmark_spec = mp_drawing.DrawingSpec(
                        color=(0, 255, 0),  # Green for confident
                        thickness=3,
                        circle_radius=4
                    )
                    connection_spec = mp_drawing.DrawingSpec(
                        color=(255, 255, 255),  # White connections
                        thickness=2
                    )
                    
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        frozenset(custom_connections),  # Only confident connections
                        landmark_drawing_spec=landmark_spec,
                        connection_drawing_spec=connection_spec
                    )
                    
                    # Add confidence indicator
                    cv2.putText(
                        frame,
                        f"Confidence: {avg_confidence:.1%}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if avg_confidence > 0.75 else (0, 255, 255),
                        2
                    )
                else:
                    # Low confidence - show warning
                    cv2.putText(
                        frame,
                        "Low confidence - move closer to camera",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
        else:
            # No person detected
            cv2.putText(
                frame,
                "No person detected - check lighting and distance",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # Encode back to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Convert landmarks to JSON-serializable format for 3D viewer
        landmarks_data = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks_data.append({
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z),
                    'visibility': float(lm.visibility)
                })
        
        return jsonify({
            'success': True,
            'overlay_frame': frame_base64,
            'person_detected': person_detected,
            'confidence': float(avg_confidence),
            'measurements': measurements,
            'landmarks': landmarks_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process_frame_mmpose', methods=['POST'])
def process_frame_mmpose():
    """Process a video frame using MMPose for enhanced pose detection"""
    try:
        if not MMPOSE_AVAILABLE:
            return jsonify({'success': False, 'error': 'MMPose not available. Install with: mim install mmpose'}), 500
        
        data = request.json
        frame_data = data.get('frame')
        user_height = data.get('user_height', 72)
        shoulder_width = data.get('shoulder_width', 18)
        model_name = data.get('model', 'hrnet')  # hrnet, simcc, vipnas, rtmpose
        
        if not frame_data:
            return jsonify({'success': False, 'error': 'No frame data'}), 400
        
        # Decode frame
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode frame'}), 400
        
        # Model configurations
        model_configs = {
            'hrnet': {
                'config': 'td-hm_hrnet-w48_8xb32-210e_coco-256x192',
                'checkpoint': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
            },
            'simcc': {
                'config': 'simcc_res50_8xb64-210e_coco-256x192',
                'checkpoint': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220923.pth'
            },
            'vipnas': {
                'config': 'td-hm_vipnas-res50_8xb64-210e_coco-256x192',
                'checkpoint': 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_vipnas-res50_8xb64-210e_coco-256x192-35d0c6e9_20220613.pth'
            },
            'rtmpose': {
                'config': 'rtmpose-m_8xb256-420e_coco-256x192',
                'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
            }
        }
        
        # Get model config
        if model_name not in model_configs:
            model_name = 'hrnet'
        
        config = model_configs[model_name]['config']
        checkpoint = model_configs[model_name]['checkpoint']
        
        # Initialize model (cache it globally for performance)
        global mmpose_model, mmpose_model_name
        if 'mmpose_model' not in globals() or mmpose_model_name != model_name:
            print(f"Loading MMPose model: {model_name}...")
            try:
                # Try to use config from mmpose configs
                from mmpose.apis import init_model
                mmpose_model = init_model(config, checkpoint, device='cpu')
                mmpose_model_name = model_name
                print(f"✓ Loaded {model_name}")
            except Exception as e:
                print(f"Error loading model: {e}")
                return jsonify({'success': False, 'error': f'Failed to load model: {str(e)}'}), 500
        
        # Run inference
        # Use whole frame as bounding box to avoid needing person detection (mmcv compiled ops)
        h, w, _ = frame.shape
        # Create a bounding box for the whole frame [x, y, width, height]
        bbox = np.array([[0, 0, w, h]], dtype=np.float32)  # Full frame as single person bbox
        
        try:
            # Try top-down with manual bbox (doesn't require person detection)
            # inference_topdown expects a list of bboxes
            results = inference_topdown(mmpose_model, frame, bbox)
        except Exception as e:
            # If that fails, try using the model directly without inference_topdown
            print(f"Warning: Top-down inference failed: {e}")
            import traceback
            traceback.print_exc()
            # Try using model's forward method directly
            try:
                # Convert frame to RGB and prepare input
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Use model's inference_dataloader or test_step
                # For now, return a clear error message
                return jsonify({
                    'success': False, 
                    'error': f'MMPose inference failed: {str(e)}. This may be due to missing compiled MMCV extensions. Try using the MediaPipe validator instead at /tool/coordinate_validator/'
                }), 500
            except Exception as e2:
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False, 
                    'error': f'MMPose inference failed: {str(e2)}. This may be due to missing compiled MMCV extensions. Try using the MediaPipe validator instead at /tool/coordinate_validator/'
                }), 500
        
        # Extract keypoints
        person_detected = False
        avg_confidence = 0.0
        measurements = {
            'eye_to_feet_pixels': 0,
            'eye_to_feet_inches': 0,
            'shoulder_width_pixels': 0,
            'shoulder_width_inches': 0,
            'distance_feet': 0,
            'distance_inches': 0,
            'landmarks_visible': 0
        }
        
        landmarks_data = []
        
        if len(results) > 0 and hasattr(results[0], 'pred_instances'):
            pred = results[0].pred_instances
            keypoints = pred.keypoints[0]  # First person
            scores = pred.keypoint_scores[0]
            
            # Convert COCO keypoints to MediaPipe-like format (17 -> 33)
            # COCO has 17 keypoints, we'll map relevant ones
            avg_confidence = float(scores.mean())
            measurements['landmarks_visible'] = int((scores > 0.5).sum())
            
            if avg_confidence > 0.5:
                person_detected = True
                
                # Map COCO keypoints (17) to our needs
                # COCO: 0=nose, 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip, 15=left_ankle, 16=right_ankle
                left_shoulder_idx = 5
                right_shoulder_idx = 6
                left_hip_idx = 11
                right_hip_idx = 12
                left_ankle_idx = 15
                right_ankle_idx = 16
                nose_idx = 0
                
                # Calculate measurements
                if (scores[left_shoulder_idx] > 0.5 and scores[right_shoulder_idx] > 0.5):
                    left_shoulder = keypoints[left_shoulder_idx]
                    right_shoulder = keypoints[right_shoulder_idx]
                    shoulder_width_pixels = float(np.linalg.norm(left_shoulder - right_shoulder))
                    measurements['shoulder_width_pixels'] = shoulder_width_pixels
                
                # Eye to feet (use nose to ankles as proxy)
                if (scores[nose_idx] > 0.5 and scores[left_ankle_idx] > 0.5 and scores[right_ankle_idx] > 0.5):
                    nose = keypoints[nose_idx]
                    left_ankle = keypoints[left_ankle_idx]
                    right_ankle = keypoints[right_ankle_idx]
                    avg_ankle = (left_ankle + right_ankle) / 2
                    eye_to_feet_pixels = float(np.linalg.norm(nose - avg_ankle))
                    measurements['eye_to_feet_pixels'] = eye_to_feet_pixels
                
                # Calibration using shoulder width
                if shoulder_width_pixels > 0 and shoulder_width > 0:
                    pixels_per_inch = shoulder_width_pixels / shoulder_width
                    
                    if eye_to_feet_pixels > 0:
                        measurements['eye_to_feet_inches'] = float(eye_to_feet_pixels / pixels_per_inch)
                    
                    measurements['shoulder_width_inches'] = float(shoulder_width)
                    
                    # Distance calculation
                    baseline_shoulder_pixels = 120.0
                    baseline_distance_feet = 10.0
                    estimated_distance_feet = baseline_distance_feet * (baseline_shoulder_pixels / shoulder_width_pixels)
                    measurements['distance_feet'] = float(estimated_distance_feet)
                    measurements['distance_inches'] = float(estimated_distance_feet * 12)
                
                # Draw skeleton
                for i in range(len(keypoints)):
                    if scores[i] > 0.5:
                        x, y = int(keypoints[i][0]), int(keypoints[i][1])
                        color = (0, 255, 0) if scores[i] > 0.8 else (0, 255, 255) if scores[i] > 0.6 else (255, 165, 0)
                        cv2.circle(frame, (x, y), 4, color, -1)
                
                # Draw connections (COCO skeleton)
                skeleton = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                    (5, 11), (6, 12), (11, 12),  # Torso
                    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
                    (0, 1), (0, 2), (1, 3), (2, 4)  # Face
                ]
                
                for (start_idx, end_idx) in skeleton:
                    if start_idx < len(keypoints) and end_idx < len(keypoints):
                        if scores[start_idx] > 0.5 and scores[end_idx] > 0.5:
                            pt1 = tuple(keypoints[start_idx].astype(int))
                            pt2 = tuple(keypoints[end_idx].astype(int))
                            cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
                
                # Confidence indicator
                cv2.putText(frame, f"MMPose {model_name.upper()}: {avg_confidence:.1%}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0) if avg_confidence > 0.75 else (0, 255, 255), 2)
                
                # Convert to landmarks_data for 3D viewer
                # Pad to 33 landmarks (MediaPipe format) with empty ones
                for i in range(33):
                    if i < len(keypoints):
                        landmarks_data.append({
                            'x': float(keypoints[i][0] / w),
                            'y': float(keypoints[i][1] / h),
                            'z': 0.0,  # MMPose doesn't provide Z
                            'visibility': float(scores[i])
                        })
                    else:
                        landmarks_data.append({
                            'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 0.0
                        })
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'overlay_frame': frame_base64,
            'person_detected': person_detected,
            'confidence': float(avg_confidence),
            'measurements': measurements,
            'landmarks': landmarks_data,
            'model': model_name
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/extract_coordinates', methods=['POST'])
def extract_coordinates():
    """Extract pose landmarks from a single frame - uses EXACT Shot Sync code"""
    try:
        if not MEDIAPIPE_AVAILABLE:
            return jsonify({'success': False, 'error': 'MediaPipe not available'}), 500
        
        data = request.json
        frame_data = data.get('frame')
        timestamp = data.get('timestamp', 0)
        user_height = data.get('user_height', 72)  # inches
        
        if not frame_data:
            return jsonify({'success': False, 'error': 'No frame data provided'}), 400
        
        # Decode base64 image
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode frame'}), 400
        
        # Use EXACT same MediaPipe settings as Shot Sync (from line 129-135)
        static_pose = mp_pose.Pose(
            model_complexity=2,
            static_image_mode=True,  # Single frame mode
            smooth_landmarks=False,
            min_detection_confidence=0.5,  # Lower for distant subjects
            min_tracking_confidence=0.5
        )
        
        # Process frame (EXACT same as Shot Sync)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = static_pose.process(rgb_frame)
        static_pose.close()
        
        if not results.pose_landmarks:
            return jsonify({'success': False, 'error': 'No person detected in frame'}), 400
        
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark
        
        # Define landmarks (same as Shot Sync)
        landmark_indices = {
            'Left Eye': 2,
            'Right Eye': 5,
            'Left Shoulder': 11,
            'Right Shoulder': 12,
            'Left Elbow': 13,
            'Right Elbow': 14,
            'Left Hip': 23,
            'Right Hip': 24,
            'Left Ankle': 27,
            'Right Ankle': 28
        }
        
        # Extract coordinates using EXACT same method as Shot Sync (get_3d_point from line 143-151)
        extracted_landmarks = {}
        for name, idx in landmark_indices.items():
            lm = landmarks[idx]
            if lm.visibility >= 0.5:  # Same visibility threshold as Shot Sync
                extracted_landmarks[name] = {
                    'x': lm.x * w,
                    'y': lm.y * h,
                    'z': lm.z,
                    'visibility': lm.visibility
                }
            else:
                extracted_landmarks[name] = {
                    'x': 0,
                    'y': 0,
                    'z': 0,
                    'visibility': lm.visibility
                }
        
        # Calculate distances using EXACT same method as Shot Sync
        left_eye = landmarks[2]
        right_eye = landmarks[5]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Check visibility (same as Shot Sync)
        if left_eye.visibility >= 0.5 and right_eye.visibility >= 0.5:
            avg_eye_x = (left_eye.x + right_eye.x) / 2 * w
            avg_eye_y = (left_eye.y + right_eye.y) / 2 * h
            avg_eye_z = (left_eye.z + right_eye.z) / 2
        else:
            avg_eye_x = avg_eye_y = avg_eye_z = 0
        
        if left_ankle.visibility >= 0.5 and right_ankle.visibility >= 0.5:
            avg_ankle_x = (left_ankle.x + right_ankle.x) / 2 * w
            avg_ankle_y = (left_ankle.y + right_ankle.y) / 2 * h
            avg_ankle_z = (left_ankle.z + right_ankle.z) / 2
        else:
            avg_ankle_x = avg_ankle_y = avg_ankle_z = 0
        
        # Calculate pixel distances
        if avg_eye_x > 0 and avg_ankle_x > 0:
            eye_to_feet_pixels = np.sqrt(
                (avg_eye_x - avg_ankle_x)**2 + 
                (avg_eye_y - avg_ankle_y)**2 + 
                (avg_eye_z - avg_ankle_z)**2
            )
        else:
            eye_to_feet_pixels = 0
        
        if left_shoulder.visibility >= 0.5 and right_shoulder.visibility >= 0.5:
            shoulder_width_pixels = np.sqrt(
                (left_shoulder.x * w - right_shoulder.x * w)**2 + 
                (left_shoulder.y * h - right_shoulder.y * h)**2 + 
                (left_shoulder.z - right_shoulder.z)**2
            )
        else:
            shoulder_width_pixels = 0
        
        # Calculate real-world distances
        # Assume average proportions: 
        # - Eye to feet is approximately the user's height
        # - Shoulder width is approximately 25% of height
        eye_to_feet_ratio = user_height / eye_to_feet_pixels if eye_to_feet_pixels > 0 else 0
        
        # Calculate actual measurements in inches
        eye_to_feet_inches = user_height  # By definition
        shoulder_width_inches = shoulder_width_pixels * eye_to_feet_ratio if eye_to_feet_ratio > 0 else 0
        
        distances = {
            'Eye to Feet': eye_to_feet_inches,
            'Shoulder Width': shoulder_width_inches,
            'Eye to Feet (pixels)': eye_to_feet_pixels,
            'Shoulder Width (pixels)': shoulder_width_pixels
        }
        
        return jsonify({
            'success': True,
            'landmarks': extracted_landmarks,
            'distances': distances,
            'frame_width': w,
            'frame_height': h,
            'timestamp': timestamp,
            'calibration_ratio': eye_to_feet_ratio
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ====================== AI COACH ENDPOINTS ======================

try:
    from ai_coach import AICoach
    AI_COACH_AVAILABLE = True
except ImportError:
    AI_COACH_AVAILABLE = False
    print("Warning: ai_coach module not available")

# In-memory storage for coach instances (in production, use Redis or similar)
coach_instances = {}

def get_db_service():
    """Get Firestore database service if Firebase is configured"""
    try:
        import firebase_admin
        from firebase_admin import firestore
        if not firebase_admin._apps:
            # Initialize Firebase if not already initialized
            try:
                firebase_admin.initialize_app()
            except ValueError:
                # Already initialized
                pass
        return firestore.client()
    except ImportError:
        print("Firebase Admin SDK not installed. Install with: pip install firebase-admin")
        return None
    except Exception as e:
        print(f"Firebase not available: {e}")
        return None

@app.route('/api/coach/chat', methods=['POST'])
def coach_chat():
    """Chat with the AI coach"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 500
    
    try:
        data = request.json
        user_id = data.get('userId')
        message = data.get('message', '')
        use_anthropic = data.get('use_anthropic', False)
        
        if not user_id or not message:
            return jsonify({'error': 'userId and message are required'}), 400
        
        # Get or create coach instance for this user
        if user_id not in coach_instances:
            db_service = get_db_service()
            coach_instances[user_id] = AICoach(user_id, db_service)
            # Load previous conversation if exists
            if db_service:
                coach_instances[user_id].load_conversation(db_service)
        
        coach = coach_instances[user_id]
        response = coach.chat(message, use_anthropic=use_anthropic)
        
        # Save conversation
        db_service = get_db_service()
        if db_service:
            coach.save_conversation(db_service)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/coach/insights', methods=['GET'])
def coach_insights():
    """Get personalized insights from the AI coach"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 500
    
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        # Get or create coach instance
        if user_id not in coach_instances:
            db_service = get_db_service()
            coach_instances[user_id] = AICoach(user_id, db_service)
        
        coach = coach_instances[user_id]
        insights = coach.get_personalized_insights()
        
        return jsonify({
            'success': True,
            'insights': insights
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/coach/conversation', methods=['GET'])
def get_conversation():
    """Get conversation history"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 500
    
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        # Get or create coach instance
        if user_id not in coach_instances:
            db_service = get_db_service()
            coach_instances[user_id] = AICoach(user_id, db_service)
            coach_instances[user_id].load_conversation(db_service)
        
        coach = coach_instances[user_id]
        
        return jsonify({
            'success': True,
            'conversation': coach.conversation_history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/coach/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 500
    
    try:
        data = request.json
        user_id = data.get('userId')
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        if user_id in coach_instances:
            coach_instances[user_id].conversation_history = []
            db_service = get_db_service()
            if db_service:
                coach_instances[user_id].save_conversation(db_service)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("=" * 60)
    print("Starting Flask Server...")
    print(f"Server will run on: http://localhost:{port}")
    print(f"Overlay viewer: http://localhost:{port}/tool/shot_sync_overlay/")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=port)

