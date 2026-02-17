from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Some features may be limited.")
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Some features may be limited.")
import time
import base64
import io
import json
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    print("Warning: email modules not available.")
import os
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("Warning: boto3 not available.")
import requests

from metrics import calculate_angle, calculate_all_single_frame_metrics

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
            
            # Store full body landmarks
            frame_landmarks_3d = np.full((33,3), np.nan, dtype=np.float32)
            for i in range(33):
                p = get_3d_point(landmarks, i, w, h)
                if p is not None:
                    frame_landmarks_3d[i] = p

            # Compute all metrics from the landmark array
            frame_metrics = calculate_all_single_frame_metrics(frame_landmarks_3d)
            elbow_angle = frame_metrics['elbow_angle']
            wrist_angle = frame_metrics['wrist_angle']
            arm_angle   = frame_metrics['arm_angle']
            
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
                            'arm_angle': float(arm_angle) if arm_angle else None,
                            'elbow_flare': frame_metrics['elbow_flare'],
                            'trunk_lean': frame_metrics['trunk_lean'],
                            'knee_bend': frame_metrics['knee_bend'],
                            'elbow_extension': frame_metrics['elbow_extension'],
                            'release_height': frame_metrics['release_height'],
                            'foot_alignment': frame_metrics['foot_alignment'],
                            'foot_stance': frame_metrics['foot_stance'],
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

@app.route('/shotsync')
@app.route('/shotsync/')
def shotsync_index():
    """Serve the ShotSync application from shotsync directory"""
    return send_from_directory(str(project_root / 'shotsync'), 'index.html')

@app.route('/shotsync/<path:filename>')
def shotsync_static(filename):
    """Serve static files from shotsync directory"""
    shotsync_dir = project_root / 'shotsync'
    file_path = shotsync_dir / filename
    if file_path.exists():
        return send_from_directory(str(shotsync_dir), filename)
    from flask import abort
    abort(404)

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

@app.route('/api/list_player_data', methods=['GET'])
def list_player_data():
    try:
        player_data_dir = os.path.join(app.root_path, 'player_data')
        if not os.path.exists(player_data_dir):
            return jsonify({'files': []}), 200
        
        # Get all .js and .json files, but exclude test/user extraction files
        files = []
        exclude_patterns = ['trial_data', 'user_extraction', 'test_', '_test']
        
        for filename in os.listdir(player_data_dir):
            if filename.endswith('.js') or filename.endswith('.json'):
                # Remove extension
                player_id = filename.rsplit('.', 1)[0]
                
                # Skip test/user extraction files
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern in player_id.lower():
                        should_exclude = True
                        break
                
                if not should_exclude:
                    files.append(player_id)
        
        # Sort files for consistent ordering
        files.sort()
        
        return jsonify({'files': files}), 200
    except Exception as e:
        app.logger.error(f"Error listing player data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_player_data', methods=['POST'])
def save_player_data():
    """Save player data to player_data folder"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        player_id = data.get('player_id')
        player_data = data.get('player_data')
        first_name = data.get('first_name', '')
        last_name = data.get('last_name', '')
        
        if not player_id or not player_data:
            return jsonify({'error': 'Missing player_id or player_data'}), 400
        
        # Get the player_data directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        player_data_dir = os.path.join(script_dir, 'player_data')
        
        # Create player_data directory if it doesn't exist
        os.makedirs(player_data_dir, exist_ok=True)
        
        # Create filename
        filename = f"{player_id}.js"
        filepath = os.path.join(player_data_dir, filename)
        
        # Generate file content
        file_content = f"// {first_name} {last_name} player data (extracted from video)\n"
        file_content += f"const {player_id}_data = {json.dumps(player_data, indent=2)};\n"
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(file_content)
        
        return jsonify({
            'success': True,
            'message': f'Player data saved to {filename}',
            'filepath': filepath,
            'filename': filename
        })
        
    except Exception as e:
        print(f"Error saving player data: {e}")
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)

