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
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: fastdtw not available. Install it with: pip install fastdtw")

app = Flask(__name__, static_folder='static')
CORS(app)

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
    return send_from_directory('static', 'index.html')

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

