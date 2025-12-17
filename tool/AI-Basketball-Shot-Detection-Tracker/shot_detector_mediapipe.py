# Enhanced Basketball Shot Detector with MediaPipe Pose Tracking
# Combines pose detection with hoop change detection

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import sys
import os
import threading
import time
from flask import Flask, Response, render_template_string, jsonify, request
from utils import get_device
from cvzone.PoseModule import PoseDetector

app = Flask(__name__)

class ShotDetectorMediaPipe:
    def __init__(self, video_path=None, use_webcam=False):
        # Load YOLO model for hoop detection
        self.model = YOLO("best.pt")
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        
        # Initialize MediaPipe pose detector
        self.pose_detector = PoseDetector(
            staticMode=False,
            modelComplexity=1,
            smoothLandmarks=True,
            detectionCon=0.5,
            trackCon=0.5
        )
        
        # Initialize video capture
        if use_webcam:
            self.cap = cv2.VideoCapture(0)
            print("Using webcam...")
        elif video_path:
            if not os.path.exists(video_path):
                print(f"Error: Video file '{video_path}' not found!")
                sys.exit(1)
            self.cap = cv2.VideoCapture(video_path)
            print(f"Using video: {video_path}")
        else:
            default_video = "video_test_5.mp4"
            if os.path.exists(default_video):
                self.cap = cv2.VideoCapture(default_video)
                print(f"Using default video: {default_video}")
            else:
                print("Error: No video file provided and default video not found!")
                sys.exit(1)

        # Hoop tracking
        self.locked_hoop = None
        self.locked_hoop_confidence = 0.0
        self.hoop_roi = None  # Region of Interest for hoop
        
        # Shot detection state
        self.shooting_state = "idle"  # idle, preparing, shooting, follow_through, checking
        self.shot_attempts = 0
        self.shot_makes = 0
        
        # Reference images for hoop change detection
        self.hoop_reference_image = None
        self.reference_captured_frame = None
        
        # Pose tracking
        self.pose_landmarks = None
        self.shooting_arm_raised = False
        self.shooting_form_started = False
        self.shooting_form_completed_frame = None
        
        # Timing
        self.frame_count = 0
        self.current_frame = None
        self.lock = threading.Lock()
        
        # Detection parameters
        self.hoop_lock_threshold = 0.5
        self.hoop_check_delay_frames = 30  # Check hoop 1 second after form completion (30fps)
        self.hoop_change_threshold = 0.15  # Threshold for detecting hoop change
        
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()

    def detect_shooting_form(self, landmarks):
        """Detect if player is in shooting form based on pose landmarks"""
        if landmarks is None or len(landmarks) < 11:
            return False
        
        # Key landmarks for shooting detection
        # MediaPipe pose landmarks: https://google.github.io/mediapipe/solutions/pose.html
        left_shoulder = landmarks[11]  # Left shoulder
        right_shoulder = landmarks[12]  # Right shoulder
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Check if either arm is raised (wrist above shoulder)
        left_arm_raised = left_wrist[2] > left_shoulder[2]  # z is depth, y is vertical
        right_arm_raised = right_wrist[2] > right_shoulder[2]
        
        # Check if arm is extended (elbow-wrist distance)
        left_arm_extended = math.sqrt((left_wrist[1] - left_elbow[1])**2 + 
                                     (left_wrist[2] - left_elbow[2])**2) > 0.15
        right_arm_extended = math.sqrt((right_wrist[1] - right_elbow[1])**2 + 
                                      (right_wrist[2] - right_elbow[2])**2) > 0.15
        
        return (left_arm_raised and left_arm_extended) or (right_arm_raised and right_arm_extended)

    def detect_form_completion(self, landmarks):
        """Detect when shooting form is completed (arm comes down)"""
        if landmarks is None or len(landmarks) < 11:
            return False
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Form is complete when wrist is below shoulder level
        left_arm_down = left_wrist[2] < left_shoulder[2]
        right_arm_down = right_wrist[2] < right_shoulder[2]
        
        return left_arm_down or right_arm_down

    def capture_hoop_reference(self, frame, hoop_bbox):
        """Capture reference image of hoop region"""
        if hoop_bbox is None:
            return None
        
        x1, y1, x2, y2 = hoop_bbox
        # Expand region slightly
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        roi = frame[y1:y2, x1:x2].copy()
        return roi

    def detect_hoop_change(self, current_frame, hoop_bbox):
        """Detect if hoop region has changed (ball went through)"""
        if self.hoop_reference_image is None or hoop_bbox is None:
            return False
        
        # Extract current hoop region
        x1, y1, x2, y2 = hoop_bbox
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(current_frame.shape[1], x2 + margin)
        y2 = min(current_frame.shape[0], y2 + margin)
        
        current_roi = current_frame[y1:y2, x1:x2]
        
        if current_roi.shape != self.hoop_reference_image.shape:
            # Resize to match
            current_roi = cv2.resize(current_roi, 
                                    (self.hoop_reference_image.shape[1], 
                                     self.hoop_reference_image.shape[0]))
        
        # Convert to grayscale for comparison
        ref_gray = cv2.cvtColor(self.hoop_reference_image, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate structural similarity or difference
        diff = cv2.absdiff(ref_gray, curr_gray)
        mean_diff = np.mean(diff)
        
        # Normalize difference (0-1 scale)
        normalized_diff = mean_diff / 255.0
        
        # Also check for motion in the region (net movement)
        # Use optical flow or frame difference
        if normalized_diff > self.hoop_change_threshold:
            return True
        
        return False

    def process_video(self):
        """Main video processing loop"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0
                self.shooting_state = "idle"
                continue

            # Detect hoop with YOLO
            results = self.model(frame, stream=True, device=self.device, conf=0.25)
            hoop_detected = False
            hoop_bbox = None
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]
                    
                    if current_class == "Basketball Hoop" and conf > self.hoop_lock_threshold:
                        hoop_detected = True
                        hoop_bbox = (x1, y1, x2, y2)
                        
                        # Lock hoop if good detection
                        if self.locked_hoop is None or conf > self.locked_hoop_confidence:
                            center = (int(x1 + (x2-x1)/2), int(y1 + (y2-y1)/2))
                            self.locked_hoop = (center, self.frame_count, x2-x1, y2-y1, conf)
                            self.locked_hoop_confidence = conf
                            self.hoop_roi = hoop_bbox
                        
                        cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), colorR=(0, 255, 0))
                        cv2.putText(frame, f"Hoop: {conf:.2f}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Use locked hoop if no current detection
            if not hoop_detected and self.locked_hoop is not None:
                center, _, w, h, conf = self.locked_hoop
                x1 = int(center[0] - w/2)
                y1 = int(center[1] - h/2)
                hoop_bbox = (x1, y1, x1+w, y1+h)
                cvzone.cornerRect(frame, (x1, y1, w, h), colorR=(255, 255, 0))
                cv2.putText(frame, "LOCKED", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Detect pose with full visualization - draw all landmarks and connections
            frame = self.pose_detector.findPose(frame, draw=True)
            landmarks, bbox = self.pose_detector.findPosition(frame, draw=True, bboxWithHands=True)
            
            # Draw additional visual feedback for shooting arm detection
            if landmarks and len(landmarks) >= 16:
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                left_wrist = landmarks[15]
                right_wrist = landmarks[16]
                
                # Check which arm is raised (shooting arm)
                # MediaPipe landmarks format: [id, x, y, visibility]
                # y coordinate: lower value = higher on screen
                if len(left_wrist) > 2 and len(left_shoulder) > 2:
                    left_raised = left_wrist[2] < left_shoulder[2]  # wrist y < shoulder y means raised
                    if left_raised:
                        # Highlight left shooting arm with yellow circle
                        cv2.circle(frame, (int(left_wrist[1]), int(left_wrist[2])), 20, (0, 255, 255), 4)
                        cv2.putText(frame, "SHOOTING ARM", (int(left_wrist[1])-60, int(left_wrist[2])-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
                
                if len(right_wrist) > 2 and len(right_shoulder) > 2:
                    right_raised = right_wrist[2] < right_shoulder[2]
                    if right_raised:
                        # Highlight right shooting arm with yellow circle
                        cv2.circle(frame, (int(right_wrist[1]), int(right_wrist[2])), 20, (0, 255, 255), 4)
                        cv2.putText(frame, "SHOOTING ARM", (int(right_wrist[1])-60, int(right_wrist[2])-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
            
            # State machine for shot detection
            if self.shooting_state == "idle":
                if self.detect_shooting_form(landmarks):
                    self.shooting_state = "preparing"
                    self.shooting_form_started = True
                    # Capture reference when form starts
                    if hoop_bbox is not None:
                        self.hoop_reference_image = self.capture_hoop_reference(frame, hoop_bbox)
                        self.reference_captured_frame = self.frame_count
                        cv2.putText(frame, "SHOT STARTED - Reference captured", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            elif self.shooting_state == "preparing":
                if self.detect_form_completion(landmarks):
                    self.shooting_state = "follow_through"
                    self.shooting_form_completed_frame = self.frame_count
                    cv2.putText(frame, "FORM COMPLETE", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            elif self.shooting_state == "follow_through":
                # Wait for ball to reach hoop
                frames_since_completion = self.frame_count - self.shooting_form_completed_frame
                if frames_since_completion >= self.hoop_check_delay_frames:
                    self.shooting_state = "checking"
            
            elif self.shooting_state == "checking":
                # Check if hoop changed
                if hoop_bbox is not None and self.hoop_reference_image is not None:
                    hoop_changed = self.detect_hoop_change(frame, hoop_bbox)
                    
                    if hoop_changed:
                        self.shot_makes += 1
                        self.shot_attempts += 1
                        cv2.putText(frame, "MAKE!", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    else:
                        self.shot_attempts += 1
                        cv2.putText(frame, "MISS", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    
                    # Reset for next shot
                    self.shooting_state = "idle"
                    self.hoop_reference_image = None
                    self.shooting_form_started = False
                    time.sleep(1)  # Brief pause to show result

            # Display score
            score_text = f"Makes: {self.shot_makes} / Attempts: {self.shot_attempts}"
            cv2.putText(frame, score_text, (50, frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            
            # Display state
            cv2.putText(frame, f"State: {self.shooting_state.upper()}", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with self.lock:
                self.current_frame = frame.copy()
            
            self.frame_count += 1
            time.sleep(0.03)

    def get_frame(self):
        """Get current frame as JPEG bytes"""
        with self.lock:
            if self.current_frame is None:
                return None
            ret, buffer = cv2.imencode('.jpg', self.current_frame)
            if not ret:
                return None
            return buffer.tobytes()

    def get_stats(self):
        """Get current statistics"""
        return {
            'makes': self.shot_makes,
            'attempts': self.shot_attempts,
            'state': self.shooting_state,
            'hoop_locked': self.locked_hoop is not None
        }

# Global detector instance
detector = None

@app.route('/')
def index():
    """Main page"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Basketball Shot Detector - MediaPipe</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px;
            }
            h1 { text-align: center; }
            #videoStream {
                max-width: 100%;
                border-radius: 15px;
            }
            .stats {
                display: flex;
                justify-content: space-around;
                margin-top: 20px;
            }
            .stat-box {
                background: rgba(255, 255, 255, 0.2);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏀 Basketball Shot Detector (MediaPipe)</h1>
            <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Video">
            <div class="stats">
                <div class="stat-box">
                    <div>Makes</div>
                    <div id="makes" style="font-size: 48px;">0</div>
                </div>
                <div class="stat-box">
                    <div>Attempts</div>
                    <div id="attempts" style="font-size: 48px;">0</div>
                </div>
                <div class="stat-box">
                    <div>State</div>
                    <div id="state" style="font-size: 24px;">IDLE</div>
                </div>
            </div>
        </div>
        <script>
            setInterval(() => {
                fetch('/stats')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('makes').textContent = data.makes;
                        document.getElementById('attempts').textContent = data.attempts;
                        document.getElementById('state').textContent = data.state.toUpperCase();
                    });
            }, 500);
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = detector.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(detector.get_stats())

if __name__ == '__main__':
    video_path = None
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    detector = ShotDetectorMediaPipe(video_path=video_path)
    print("\n" + "="*50)
    print("Basketball Shot Detector - MediaPipe Version")
    print("="*50)
    print("Open your browser: http://localhost:8888")
    print("="*50 + "\n")
    
    app.run(host='127.0.0.1', port=8888, debug=False, threaded=True)

