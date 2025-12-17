# Avi Shah - Basketball Shot Detector/Tracker - July 2023
# Web-based version for localhost viewing

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import sys
import os
import threading
import time
from flask import Flask, Response, render_template_string, jsonify, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
from cvzone.PoseModule import PoseDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class ShotDetectorWeb:
    def __init__(self, video_path=None, use_webcam=False, model_size='medium'):
        # Load the YOLO model
        # Model options: 'nano', 'small', 'medium', 'large', 'xlarge', or 'custom'
        # - nano (yolov8n): Fastest, least accurate
        # - small (yolov8s): Fast, decent accuracy
        # - medium (yolov8m): Good balance ⭐ Recommended
        # - large (yolov8l): Better accuracy, slower
        # - xlarge (yolov8x): Best accuracy, slowest
        # - custom: Uses best.pt (your trained model)
        
        self.overlay_text = "Waiting..."
        
        model_map = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt',
            'medium': 'yolov8m.pt',
            'large': 'yolov8l.pt',
            'xlarge': 'yolov8x.pt',
            'custom': 'best.pt'
        }
        
        model_name = model_map.get(model_size, 'yolov8m.pt')
        
        # Try to load model, fallback to smaller if not available
        # Load model in a way that doesn't block video initialization
        self.model = None
        self.model_size = model_size
        self.model_name = model_map.get(model_size, 'yolov8m.pt')
        
        # Load model in background to not block video start
        def load_model():
            try:
                print(f"Loading YOLO model: {self.model_name}")
                self.model = YOLO(self.model_name)
                print(f"✓ Successfully loaded {self.model_name}")
            except Exception as e:
                print(f"Warning: Could not load {self.model_name}: {e}")
                print("Falling back to yolov8n.pt")
                try:
                    self.model = YOLO('yolov8n.pt')
                except:
                    print("Error: Could not load any model!")
        
        # Start model loading in background thread
        model_thread = threading.Thread(target=load_model, daemon=True)
        model_thread.start()
        
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        self.model_size = model_size
        
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
                print("Usage: python shot_detector_web.py [video_path] or python shot_detector_web.py --webcam")
                sys.exit(1)

        self.ball_pos = []
        self.hoop_pos = []
        self.locked_hoop = None  # Locked hoop position (best detection)
        self.locked_hoop_confidence = 0.0
        self.all_ball_detections = []
        self.frame_count = 0
        self.frame = None
        self.current_frame = None
        self.makes = 0
        self.attempts = 0
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        self.lock = threading.Lock()
        self.video_path = video_path
        self.use_webcam = use_webcam
        
        # Initialize MediaPipe pose detector
        self.pose_detector = PoseDetector(
            staticMode=False,
            modelComplexity=1,
            smoothLandmarks=True,
            detectionCon=0.5,
            trackCon=0.5
        )
        self.use_mediapipe = True  # Enable MediaPipe pose detection
        
        # Detection thresholds (can be adjusted)
        self.ball_confidence_threshold = 0.01  # Accept almost any detection
        self.ball_confidence_threshold_near_hoop = 0.01  # Accept almost any when near hoop
        self.hoop_confidence_threshold = 0.25  # For initial detection
        self.hoop_lock_threshold = 0.5  # Lock hoop once we get this confidence
        self.debug_mode = True  # Show all detections
        self.all_ball_detections = []  # Track all detections for debugging
        
        # Shot detection requirements (to prevent false positives)
        self.min_ball_positions_for_shot = 5  # Need at least 5 ball detections
        self.min_frames_between_up_down = 10  # Need at least 10 frames between up and down
        self.min_ball_travel_distance = 50  # Minimum pixels ball must travel
        
        # MediaPipe-based shot detection state
        self.shooting_state = "idle"  # idle, preparing, shooting, follow_through, checking
        self.hoop_reference_image = None
        self.reference_captured_frame = None
        self.shooting_form_completed_frame = None
        self.shooting_form_started = False
        self.hoop_check_delay_frames = 30  # Check hoop 1 second after form completion (30fps)
        self.hoop_change_threshold = 0.15  # Threshold for detecting hoop change
        self.state_timeout_frames = 120  # 4 seconds timeout for any state
        self.state_start_frame = 0  # Track when state started
        
        # Initialize with a placeholder frame so video stream works immediately
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Initializing...", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        with self.lock:
            self.current_frame = placeholder
        
        # Start processing in background thread
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()
    
    def switch_video(self, new_video_path):
        """Switch to a new video file"""
        # Stop current processing
        self.running = False
        
        # Wait for thread to finish
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Release old video
        if self.cap is not None:
            self.cap.release()
        
        # Reset state
        with self.lock:
            self.ball_pos = []
            self.hoop_pos = []
            self.all_ball_detections = []
            self.frame_count = 0
            self.makes = 0
            self.attempts = 0
            self.up = False
            self.down = False
            self.overlay_text = "Waiting..."
            self.overlay_color = (0, 0, 0)
            self.fade_counter = 0
            self.current_frame = None
        
        # Load new video
        if os.path.exists(new_video_path):
            self.cap = cv2.VideoCapture(new_video_path)
            self.video_path = new_video_path
            self.use_webcam = False
            self.running = True
            
            # Restart processing thread
            self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
            self.processing_thread.start()
            print(f"Switched to new video: {new_video_path}")
            return True
        else:
            print(f"Error: Video file '{new_video_path}' not found!")
            return False

    def process_video(self):
        """Process video frames in background thread"""
        print("Starting video processing thread...")
        
        # Check if video can be opened
        if not self.cap.isOpened():
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "ERROR: Cannot open video", (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(error_frame, f"File: {self.video_path}", (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            with self.lock:
                self.current_frame = error_frame
            print(f"ERROR: Cannot open video file: {self.video_path}")
            return
        
        print(f"Video opened successfully: {self.video_path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
        
        # Start processing video immediately, even if model isn't loaded yet
        while self.running:
            ret, self.frame = self.cap.read()
            
            if not ret:
                # Loop video if it ends
                print("Video ended, looping...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0
                self.ball_pos = []
                self.hoop_pos = []
                self.all_ball_detections = []
                # Keep locked hoop when looping
                continue
            
            # Ensure frame is valid
            if self.frame is None or self.frame.size == 0:
                print("Warning: Invalid frame read")
                continue
            
            # Show video immediately, with status message if model not ready
            status_text = ""
            if self.model is None:
                # Calculate wait time
                model_wait_time = time.time() - getattr(self, 'start_time', time.time())
                if not hasattr(self, 'start_time'):
                    self.start_time = time.time()
                
                if model_wait_time < 60:  # Show loading for up to 60 seconds
                    status_text = f"Loading model... ({int(model_wait_time)}s)"
                else:
                    status_text = "Model loading timeout - using video only"
            
            # Add status text to frame if needed
            if status_text:
                cv2.putText(self.frame, status_text, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Only do detection if model is loaded
            if self.model is None:
                # Just show the video frame without detection
                with self.lock:
                    self.current_frame = self.frame.copy()
                self.frame_count += 1
                time.sleep(0.03)
                continue
            
            # Model is loaded - do detection
            try:
                # Enhance image for better detection
                enhanced_frame = self.enhance_image(self.frame)
                
                # Process frame with YOLO - use improved settings for better detection
                # Larger image size helps detect small objects like basketballs
                # Adjust based on model size: larger models can handle larger images
                img_size = 1280 if self.model_size in ['large', 'xlarge'] else 640
                
                results = self.model(enhanced_frame, stream=True, device=self.device, 
                                   conf=0.01, iou=0.45, imgsz=img_size)
            except Exception as e:
                print(f"Error in YOLO detection: {e}")
                # Continue with frame even if detection fails
                results = []

            for r in results:
                boxes = r.boxes
                # Get model's class names (might be COCO classes if using pretrained model)
                model_class_names = self.model.names if hasattr(self.model, 'names') else {}
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get class name from model (handles both custom and COCO models)
                    if cls in model_class_names:
                        detected_class_name = model_class_names[cls]
                    else:
                        continue  # Skip if class index is invalid
                    
                    # Only process if it's a class we care about
                    if detected_class_name not in self.class_names:
                        continue  # Skip COCO classes we don't need
                    
                    current_class = detected_class_name
                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Basketball detection with very low thresholds
                    if current_class == "Basketball":
                        # Track all detections for debugging
                        self.all_ball_detections.append((center, self.frame_count, w, h, conf))
                        # Keep only last 100 detections
                        if len(self.all_ball_detections) > 100:
                            self.all_ball_detections.pop(0)
                        
                        # Show all detections in debug mode
                        if self.debug_mode:
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                            cv2.putText(self.frame, f"Ball: {conf:.3f}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        # Add to tracking - accept almost any detection
                        # Only filter out if confidence is extremely low (< 0.01)
                        if conf >= self.ball_confidence_threshold:
                            self.ball_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(0, 0, 255))
                            cv2.putText(self.frame, f"{conf:.3f}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Basketball Hoop detection with locking mechanism
                    if current_class == "Basketball Hoop":
                        # Show all detections in debug mode
                        if self.debug_mode:
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(self.frame, f"Hoop: {conf:.2f}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Lock hoop if we get a good detection
                        if conf > self.hoop_lock_threshold:
                            # Update locked hoop if this is better than current
                            if self.locked_hoop is None or conf > self.locked_hoop_confidence:
                                self.locked_hoop = (center, self.frame_count, w, h, conf)
                                self.locked_hoop_confidence = conf
                                print(f"Hoop locked with confidence: {conf:.3f}")
                        
                        # Add to tracking if confidence is high enough OR use locked hoop
                        if conf > self.hoop_confidence_threshold:
                            self.hoop_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h), colorR=(0, 255, 0))
                            cv2.putText(self.frame, f"{conf:.2f}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Use locked hoop if we have one and current detections are poor
            if self.locked_hoop is not None:
                # Check if we have recent hoop detections
                recent_hoop = False
                if len(self.hoop_pos) > 0:
                    last_hoop_frame = self.hoop_pos[-1][1]
                    if self.frame_count - last_hoop_frame < 30:  # Had detection in last 30 frames
                        recent_hoop = True
                
                # If no recent detections, use locked hoop
                if not recent_hoop:
                    # Update locked hoop frame count and add to tracking
                    locked_center, _, locked_w, locked_h, locked_conf = self.locked_hoop
                    self.hoop_pos.append((locked_center, self.frame_count, locked_w, locked_h, locked_conf))
                    
                    # Draw locked hoop with special color (cyan)
                    x = int(locked_center[0] - locked_w / 2)
                    y = int(locked_center[1] - locked_h / 2)
                    cvzone.cornerRect(self.frame, (x, y, locked_w, locked_h), colorR=(255, 255, 0))
                    cv2.putText(self.frame, f"LOCKED: {locked_conf:.2f}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # MediaPipe pose detection
            if self.use_mediapipe:
                self.frame = self.pose_detector.findPose(self.frame, draw=True)
                landmarks, bbox = self.pose_detector.findPosition(self.frame, draw=True, bboxWithHands=True)
                
                # Draw shooting arm highlight
                if landmarks and len(landmarks) >= 16:
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    left_wrist = landmarks[15]
                    right_wrist = landmarks[16]
                    
                    # Check which arm is raised (shooting arm)
                    if len(left_wrist) > 2 and len(left_shoulder) > 2:
                        left_raised = left_wrist[2] < left_shoulder[2]  # y coordinate (lower = higher on screen)
                        if left_raised:
                            cv2.circle(self.frame, (int(left_wrist[1]), int(left_wrist[2])), 20, (0, 255, 255), 4)
                            cv2.putText(self.frame, "SHOOTING ARM", (int(left_wrist[1])-60, int(left_wrist[2])-30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
                    
                    if len(right_wrist) > 2 and len(right_shoulder) > 2:
                        right_raised = right_wrist[2] < right_shoulder[2]
                        if right_raised:
                            cv2.circle(self.frame, (int(right_wrist[1]), int(right_wrist[2])), 20, (0, 255, 255), 4)
                            cv2.putText(self.frame, "SHOOTING ARM", (int(right_wrist[1])-60, int(right_wrist[2])-30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
                
                # MediaPipe-based shot detection
                self.mediapipe_shot_detection(landmarks)
            
            # Use less aggressive cleaning - only remove very old points
            self.ball_pos = self.clean_ball_pos_gentle(self.ball_pos, self.frame_count)
            if len(self.hoop_pos) > 1:
                self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            
            self.display_tracking()
            # IMPORTANT: Only use MediaPipe shot detection when enabled
            # Traditional ball-based detection is completely disabled when MediaPipe is active
            if not self.use_mediapipe:
                self.shot_detection()  # Only runs if MediaPipe is disabled
            # MediaPipe detection runs in mediapipe_shot_detection() called above
            self.display_score()
            
            # Store current frame with lock
            with self.lock:
                self.current_frame = self.frame.copy()
            
            self.frame_count += 1
            time.sleep(0.03)  # ~30 FPS

    def clean_ball_pos_gentle(self, ball_pos, frame_count):
        """Less aggressive cleaning - only remove very old or obviously wrong detections"""
        # Only remove points older than 60 frames (was 30)
        if len(ball_pos) > 0:
            while len(ball_pos) > 0 and frame_count - ball_pos[0][1] > 60:
                ball_pos.pop(0)
        
        # Only remove if ball jumps an unrealistic distance (more lenient)
        if len(ball_pos) > 1:
            w1 = ball_pos[-2][2]
            h1 = ball_pos[-2][3]
            x1 = ball_pos[-2][0][0]
            y1 = ball_pos[-2][0][1]
            x2 = ball_pos[-1][0][0]
            y2 = ball_pos[-1][0][1]
            f1 = ball_pos[-2][1]
            f2 = ball_pos[-1][1]
            f_dif = f2 - f1
            
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            max_dist = 8 * math.sqrt((w1) ** 2 + (h1) ** 2)  # Was 4x, now 8x
            
            # Only remove if it's a huge jump in very few frames
            if (dist > max_dist) and (f_dif < 3):  # Was 5 frames, now 3
                ball_pos.pop()
        
        return ball_pos
    
    def enhance_image(self, frame):
        """Enhance image for better detection"""
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def calculate_ball_travel(self):
        """Calculate total distance ball has traveled"""
        if len(self.ball_pos) < 2:
            return 0
        total = 0
        for i in range(1, len(self.ball_pos)):
            x1, y1 = self.ball_pos[i-1][0]
            x2, y2 = self.ball_pos[i][0]
            total += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return int(total)
    
    def detect_shooting_form(self, landmarks):
        """Detect if player is in shooting form based on pose landmarks"""
        if landmarks is None or len(landmarks) < 16:
            return False
        
        # MediaPipe landmarks format: [id, x, y, visibility]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Check if either arm is raised (wrist above shoulder - lower y value)
        if len(left_wrist) > 2 and len(left_shoulder) > 2:
            left_arm_raised = left_wrist[2] < left_shoulder[2]
            # Check if arm is extended
            if len(left_elbow) > 2:
                left_arm_extended = math.sqrt((left_wrist[1] - left_elbow[1])**2 + 
                                             (left_wrist[2] - left_elbow[2])**2) > 0.1
                if left_arm_raised and left_arm_extended:
                    return True
        
        if len(right_wrist) > 2 and len(right_shoulder) > 2:
            right_arm_raised = right_wrist[2] < right_shoulder[2]
            # Check if arm is extended
            if len(right_elbow) > 2:
                right_arm_extended = math.sqrt((right_wrist[1] - right_elbow[1])**2 + 
                                              (right_wrist[2] - right_elbow[2])**2) > 0.1
                if right_arm_raised and right_arm_extended:
                    return True
        
        return False

    def detect_form_completion(self, landmarks):
        """Detect when shooting form is completed (arm comes down)"""
        if landmarks is None or len(landmarks) < 16:
            return False
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Form is complete when wrist is below shoulder level (higher y value)
        if len(left_wrist) > 2 and len(left_shoulder) > 2:
            if left_wrist[2] > left_shoulder[2]:  # wrist y > shoulder y means arm down
                return True
        if len(right_wrist) > 2 and len(right_shoulder) > 2:
            if right_wrist[2] > right_shoulder[2]:
                return True
        
        return False

    def capture_hoop_reference(self, hoop_bbox):
        """Capture reference image of hoop region"""
        if hoop_bbox is None or len(self.hoop_pos) == 0:
            return None
        
        # Get hoop bounding box
        hoop_center = self.hoop_pos[-1][0]
        hoop_w = self.hoop_pos[-1][2]
        hoop_h = self.hoop_pos[-1][3]
        
        x1 = int(hoop_center[0] - hoop_w / 2)
        y1 = int(hoop_center[1] - hoop_h / 2)
        x2 = int(hoop_center[0] + hoop_w / 2)
        y2 = int(hoop_center[1] + hoop_h / 2)
        
        # Expand region slightly
        margin = 30
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(self.frame.shape[1], x2 + margin)
        y2 = min(self.frame.shape[0], y2 + margin)
        
        roi = self.frame[y1:y2, x1:x2].copy()
        return (roi, (x1, y1, x2, y2))

    def detect_hoop_change(self, hoop_bbox):
        """Detect if hoop region has changed (ball went through)"""
        if self.hoop_reference_image is None or len(self.hoop_pos) == 0:
            return False
        
        # Get current hoop region
        hoop_center = self.hoop_pos[-1][0]
        hoop_w = self.hoop_pos[-1][2]
        hoop_h = self.hoop_pos[-1][3]
        
        x1 = int(hoop_center[0] - hoop_w / 2)
        y1 = int(hoop_center[1] - hoop_h / 2)
        x2 = int(hoop_center[0] + hoop_w / 2)
        y2 = int(hoop_center[1] + hoop_h / 2)
        
        margin = 30
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(self.frame.shape[1], x2 + margin)
        y2 = min(self.frame.shape[0], y2 + margin)
        
        current_roi = self.frame[y1:y2, x1:x2]
        ref_roi, _ = self.hoop_reference_image
        
        if current_roi.shape != ref_roi.shape:
            current_roi = cv2.resize(current_roi, (ref_roi.shape[1], ref_roi.shape[0]))
        
        # Convert to grayscale for comparison
        if len(ref_roi.shape) == 3:
            ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_roi
        if len(current_roi.shape) == 3:
            curr_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = current_roi
        
        # Calculate difference
        diff = cv2.absdiff(ref_gray, curr_gray)
        mean_diff = np.mean(diff)
        normalized_diff = mean_diff / 255.0
        
        return normalized_diff > self.hoop_change_threshold

    def mediapipe_shot_detection(self, landmarks):
        """MediaPipe-based shot detection using pose and hoop change"""
        # Get current hoop position
        hoop_bbox = None
        if len(self.hoop_pos) > 0:
            hoop_center = self.hoop_pos[-1][0]
            hoop_w = self.hoop_pos[-1][2]
            hoop_h = self.hoop_pos[-1][3]
            hoop_bbox = (int(hoop_center[0] - hoop_w/2), int(hoop_center[1] - hoop_h/2),
                        int(hoop_center[0] + hoop_w/2), int(hoop_center[1] + hoop_h/2))
        
        # State machine for shot detection - ONLY count shots that go through full sequence
        # This is the ONLY place where makes/misses are counted when MediaPipe is enabled
        
        # Check for state timeout (prevent getting stuck)
        if self.shooting_state != "idle":
            frames_in_state = self.frame_count - self.state_start_frame
            if frames_in_state > self.state_timeout_frames:
                # Timeout - reset to idle
                self.shooting_state = "idle"
                self.state_start_frame = self.frame_count
                self.hoop_reference_image = None
                self.shooting_form_started = False
                cv2.putText(self.frame, "SHOT TIMEOUT - Reset", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                return
        
        if self.shooting_state == "idle":
            # Only start if we detect shooting form AND have a locked hoop AND valid pose
            if (self.detect_shooting_form(landmarks) and 
                hoop_bbox is not None and 
                self.locked_hoop is not None and
                landmarks is not None and 
                len(landmarks) >= 16):
                self.shooting_state = "preparing"
                self.state_start_frame = self.frame_count
                self.shooting_form_started = True
                # Capture reference when form starts
                ref_image = self.capture_hoop_reference(hoop_bbox)
                if ref_image:
                    self.hoop_reference_image = ref_image
                    self.reference_captured_frame = self.frame_count
                    cv2.putText(self.frame, "SHOT STARTED - Reference captured", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        elif self.shooting_state == "preparing":
            # Validate we still have pose and hoop
            if landmarks is None or len(landmarks) < 16 or hoop_bbox is None:
                # Lost tracking - cancel shot
                self.shooting_state = "idle"
                self.hoop_reference_image = None
                self.shooting_form_started = False
                return
            
            # Check if form completed (arm came down)
            if self.detect_form_completion(landmarks):
                # Validate minimum time in preparing state (prevent false triggers)
                frames_in_preparing = self.frame_count - self.reference_captured_frame
                if frames_in_preparing >= 5:  # At least 5 frames in preparing state
                    self.shooting_state = "follow_through"
                    self.state_start_frame = self.frame_count
                    self.shooting_form_completed_frame = self.frame_count
                    cv2.putText(self.frame, "FORM COMPLETE", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    # Too quick - probably false positive, cancel
                    self.shooting_state = "idle"
                    self.state_start_frame = self.frame_count
                    self.hoop_reference_image = None
                    self.shooting_form_started = False
        
        elif self.shooting_state == "follow_through":
            # Wait for ball to reach hoop
            frames_since_completion = self.frame_count - self.shooting_form_completed_frame
            if frames_since_completion >= self.hoop_check_delay_frames:
                self.shooting_state = "checking"
                self.state_start_frame = self.frame_count
        
        elif self.shooting_state == "checking":
            # Check if hoop changed - ONLY count if we have valid reference
            if hoop_bbox is not None and self.hoop_reference_image is not None:
                hoop_changed = self.detect_hoop_change(hoop_bbox)
                
                # Count the shot attempt
                self.attempts += 1
                
                if hoop_changed:
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)
                    self.overlay_text = "Make"
                    self.fade_counter = self.fade_frames
                    cv2.putText(self.frame, "MAKE!", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                else:
                    self.overlay_color = (255, 0, 0)
                    self.overlay_text = "Miss"
                    self.fade_counter = self.fade_frames
                    cv2.putText(self.frame, "MISS", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                # Reset for next shot
                self.shooting_state = "idle"
                self.state_start_frame = self.frame_count
                self.hoop_reference_image = None
                self.shooting_form_started = False
            else:
                # Missing reference or hoop - cancel
                self.shooting_state = "idle"
                self.state_start_frame = self.frame_count
                self.hoop_reference_image = None
                self.shooting_form_started = False

    def display_tracking(self):
        """Display ball and hoop tracking"""
        # Draw ball tracking path
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 3, (0, 0, 255), 2)
            if i > 0:
                cv2.line(self.frame, self.ball_pos[i-1][0], self.ball_pos[i][0], (0, 0, 255), 1)

        # Draw hoop center
        if len(self.hoop_pos) > 0:
            hoop_center = self.hoop_pos[-1][0]
            cv2.circle(self.frame, hoop_center, 5, (128, 128, 0), 3)
            # Draw a larger circle if using locked hoop
            if self.locked_hoop is not None and self.locked_hoop[0] == hoop_center:
                cv2.circle(self.frame, hoop_center, 10, (255, 255, 0), 2)
        
        # Display travel distance
        if len(self.ball_pos) > 1:
            travel = self.calculate_ball_travel()
            cv2.putText(self.frame, f"Travel: {travel}px", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) >= self.min_ball_positions_for_shot:
            # Calculate ball travel distance
            if len(self.ball_pos) >= 2:
                total_distance = 0
                for i in range(1, len(self.ball_pos)):
                    x1, y1 = self.ball_pos[i-1][0]
                    x2, y2 = self.ball_pos[i][0]
                    total_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            else:
                total_distance = 0
            
            # Only detect shots if ball has traveled enough distance
            if total_distance < self.min_ball_travel_distance:
                return  # Ball hasn't moved enough, probably still in hand
            
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # Check for valid shot attempt
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    # Additional validation: ensure enough time passed between up and down
                    frames_between = self.down_frame - self.up_frame
                    
                    if frames_between >= self.min_frames_between_up_down:
                        # Valid shot attempt
                        self.attempts += 1
                        self.up = False
                        self.down = False

                        if score(self.ball_pos, self.hoop_pos):
                            self.makes += 1
                            self.overlay_color = (0, 255, 0)
                            self.overlay_text = "Make"
                            self.fade_counter = self.fade_frames
                        else:
                            self.overlay_color = (255, 0, 0)
                            self.overlay_text = "Miss"
                            self.fade_counter = self.fade_frames
                    else:
                        # False positive - reset without counting
                        self.up = False
                        self.down = False

    def display_score(self):
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        if hasattr(self, 'overlay_text'):
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
            text_x = self.frame.shape[1] - text_width - 40
            text_y = 100

            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        self.overlay_color, 6)

        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1

    def get_frame(self):
        """Get current frame as JPEG bytes"""
        with self.lock:
            if self.current_frame is None:
                # Return a placeholder "loading" frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Loading video...", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    return buffer.tobytes()
                return None
            
            try:
                ret, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    return None
                return buffer.tobytes()
            except Exception as e:
                print(f"Error encoding frame: {e}")
                return None

    def get_stats(self):
        """Get current statistics"""
        # Convert color to list format
        if isinstance(self.overlay_color, np.ndarray):
            color_list = self.overlay_color.tolist()
        elif isinstance(self.overlay_color, tuple):
            color_list = list(self.overlay_color)
        else:
            color_list = [0, 0, 0]
        
        return {
            'makes': self.makes,
            'attempts': self.attempts,
            'overlay_text': self.overlay_text,
            'overlay_color': color_list,
            'ball_detections': len(self.ball_pos),
            'hoop_detections': len(self.hoop_pos),
            'frame_count': self.frame_count,
            'all_ball_detections': len(self.all_ball_detections),
            'ball_threshold': self.ball_confidence_threshold,
            'hoop_threshold': self.hoop_confidence_threshold,
            'ball_travel_distance': self.calculate_ball_travel() if len(self.ball_pos) > 1 else 0,
            'hoop_locked': self.locked_hoop is not None,
            'hoop_lock_confidence': self.locked_hoop_confidence if self.locked_hoop else 0.0,
            'shooting_state': self.shooting_state if self.use_mediapipe else 'traditional',
            'model_size': self.model_size
        }

# Global detector instance
detector = None

@app.route('/')
def index():
    """Main page with video stream"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Basketball Shot Detector</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .video-container {
                text-align: center;
                margin-bottom: 30px;
            }
            #videoStream {
                max-width: 100%;
                height: auto;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                background: #000;
            }
            .stats-container {
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 30px;
            }
            .stat-box {
                background: rgba(255, 255, 255, 0.2);
                padding: 20px 40px;
                border-radius: 15px;
                text-align: center;
                min-width: 200px;
                backdrop-filter: blur(5px);
            }
            .stat-value {
                font-size: 48px;
                font-weight: bold;
                margin: 10px 0;
            }
            .stat-label {
                font-size: 18px;
                opacity: 0.9;
            }
            .overlay-indicator {
                font-size: 36px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin-top: 20px;
                display: inline-block;
                transition: all 0.3s ease;
            }
            .make {
                background: rgba(0, 255, 0, 0.3);
                color: #00ff00;
                text-shadow: 0 0 10px #00ff00;
            }
            .miss {
                background: rgba(255, 0, 0, 0.3);
                color: #ff0000;
                text-shadow: 0 0 10px #ff0000;
            }
            .waiting {
                background: rgba(128, 128, 128, 0.3);
                color: #cccccc;
            }
            .upload-container {
                background: rgba(255, 255, 255, 0.15);
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 30px;
                backdrop-filter: blur(5px);
            }
            .upload-form {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 15px;
            }
            .file-input-wrapper {
                position: relative;
                display: inline-block;
                cursor: pointer;
            }
            .file-input-wrapper input[type=file] {
                position: absolute;
                left: -9999px;
            }
            .file-input-label {
                display: inline-block;
                padding: 12px 30px;
                background: rgba(255, 255, 255, 0.3);
                border: 2px dashed rgba(255, 255, 255, 0.5);
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 16px;
            }
            .file-input-label:hover {
                background: rgba(255, 255, 255, 0.4);
                border-color: rgba(255, 255, 255, 0.7);
            }
            .upload-btn {
                padding: 12px 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 10px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .upload-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .file-name {
                margin-top: 10px;
                font-size: 14px;
                opacity: 0.9;
            }
            .message {
                margin-top: 15px;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
            }
            .message.success {
                background: rgba(0, 255, 0, 0.2);
                color: #00ff00;
            }
            .message.error {
                background: rgba(255, 0, 0, 0.2);
                color: #ff0000;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏀 Basketball Shot Detector</h1>
            <div class="upload-container">
                <form class="upload-form" id="uploadForm" enctype="multipart/form-data">
                    <div class="file-input-wrapper">
                        <input type="file" id="videoFile" name="video" accept="video/*" required>
                        <label for="videoFile" class="file-input-label">📁 Choose Video File</label>
                    </div>
                    <div class="file-name" id="fileName"></div>
                    <button type="submit" class="upload-btn" id="uploadBtn">Upload & Analyze Video</button>
                    <div id="message"></div>
                </form>
            </div>
            <div class="video-container">
                <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>
            <div class="stats-container">
                <div class="stat-box">
                    <div class="stat-label">Makes</div>
                    <div class="stat-value" id="makes">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Attempts</div>
                    <div class="stat-value" id="attempts">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Score</div>
                    <div class="stat-value" id="score">0 / 0</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <div id="overlay" class="overlay-indicator waiting">Waiting...</div>
            </div>
            <div class="stats-container" style="margin-top: 20px;">
                <div class="stat-box">
                    <div class="stat-label">Ball Tracked</div>
                    <div class="stat-value" id="ballDetections" style="font-size: 32px;">0</div>
                    <div style="font-size: 12px; opacity: 0.7;">All: <span id="allBallDetections">0</span></div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Hoop Detections</div>
                    <div class="stat-value" id="hoopDetections" style="font-size: 32px;">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Frame</div>
                    <div class="stat-value" id="frameCount" style="font-size: 32px;">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Ball Travel</div>
                    <div class="stat-value" id="ballTravel" style="font-size: 28px;">0px</div>
                    <div style="font-size: 11px; opacity: 0.7;">Min: 50px</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Shooting State</div>
                    <div class="stat-value" id="shootingState" style="font-size: 24px;">IDLE</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 20px; padding: 15px; background: rgba(255, 255, 255, 0.1); border-radius: 10px;">
                <p style="margin: 5px 0; font-size: 14px;"><strong>Debug Mode:</strong> All detections are shown with confidence scores</p>
                <p style="margin: 5px 0; font-size: 12px; opacity: 0.8;">Yellow boxes = All detections | Red/Green boxes = Tracked objects</p>
                <p style="margin: 5px 0; font-size: 11px; opacity: 0.7;">Ball threshold: <span id="ballThreshold">0.10</span> | Hoop threshold: <span id="hoopThreshold">0.25</span></p>
                <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;" id="hoopLockStatus">🔓 Hoop: Not locked</p>
                <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;"><strong>Model:</strong> <span id="modelSize">medium</span></p>
            </div>
        </div>
        <script>
            // File input handler
            document.getElementById('videoFile').addEventListener('change', function(e) {
                const fileName = e.target.files[0] ? e.target.files[0].name : '';
                document.getElementById('fileName').textContent = fileName ? 'Selected: ' + fileName : '';
                document.getElementById('message').textContent = '';
            });

            // Upload form handler
            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData();
                const fileInput = document.getElementById('videoFile');
                
                if (!fileInput.files[0]) {
                    showMessage('Please select a video file', 'error');
                    return;
                }
                
                formData.append('video', fileInput.files[0]);
                const uploadBtn = document.getElementById('uploadBtn');
                uploadBtn.disabled = true;
                uploadBtn.textContent = 'Uploading...';
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage('Video uploaded successfully! Processing...', 'success');
                        // Reload video stream after a short delay
                        setTimeout(() => {
                            document.getElementById('videoStream').src = '{{ url_for("video_feed") }}?t=' + new Date().getTime();
                        }, 1000);
                    } else {
                        showMessage(data.message || 'Upload failed', 'error');
                    }
                })
                .catch(error => {
                    showMessage('Error uploading video: ' + error.message, 'error');
                })
                .finally(() => {
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = 'Upload & Analyze Video';
                });
            });

            function showMessage(text, type) {
                const messageDiv = document.getElementById('message');
                messageDiv.textContent = text;
                messageDiv.className = 'message ' + type;
                setTimeout(() => {
                    messageDiv.textContent = '';
                    messageDiv.className = 'message';
                }, 5000);
            }

            // Update stats every 500ms
            setInterval(function() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('makes').textContent = data.makes;
                        document.getElementById('attempts').textContent = data.attempts;
                        document.getElementById('score').textContent = data.makes + ' / ' + data.attempts;
                        document.getElementById('ballDetections').textContent = data.ball_detections || 0;
                        document.getElementById('allBallDetections').textContent = data.all_ball_detections || 0;
                        document.getElementById('hoopDetections').textContent = data.hoop_detections || 0;
                        document.getElementById('frameCount').textContent = data.frame_count || 0;
                        if (data.ball_threshold !== undefined) {
                            document.getElementById('ballThreshold').textContent = data.ball_threshold.toFixed(2);
                        }
                        if (data.hoop_threshold !== undefined) {
                            document.getElementById('hoopThreshold').textContent = data.hoop_threshold.toFixed(2);
                        }
                        if (data.shooting_state !== undefined) {
                            document.getElementById('shootingState').textContent = data.shooting_state.toUpperCase();
                        }
                        if (data.model_size !== undefined) {
                            document.getElementById('modelSize').textContent = data.model_size.toUpperCase();
                        }
                        if (data.ball_travel_distance !== undefined) {
                            document.getElementById('ballTravel').textContent = data.ball_travel_distance + 'px';
                        }
                        if (data.hoop_locked !== undefined) {
                            const lockStatus = document.getElementById('hoopLockStatus');
                            if (data.hoop_locked) {
                                lockStatus.innerHTML = '🔒 Hoop: <strong>LOCKED</strong> (Confidence: ' + data.hoop_lock_confidence.toFixed(2) + ')';
                                lockStatus.style.color = '#00ff00';
                            } else {
                                lockStatus.innerHTML = '🔓 Hoop: Not locked';
                                lockStatus.style.color = '#cccccc';
                            }
                        }
                        
                        const overlay = document.getElementById('overlay');
                        overlay.textContent = data.overlay_text;
                        overlay.className = 'overlay-indicator';
                        
                        if (data.overlay_text === 'Make') {
                            overlay.classList.add('make');
                        } else if (data.overlay_text === 'Miss') {
                            overlay.classList.add('miss');
                        } else {
                            overlay.classList.add('waiting');
                        }
                    })
                    .catch(error => console.error('Error:', error));
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
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Get current statistics as JSON"""
    return jsonify(detector.get_stats())

@app.route('/adjust_thresholds', methods=['POST'])
def adjust_thresholds():
    """Adjust detection thresholds"""
    data = request.json
    if detector:
        if 'ball_threshold' in data:
            detector.ball_confidence_threshold = float(data['ball_threshold'])
        if 'ball_near_hoop_threshold' in data:
            detector.ball_confidence_threshold_near_hoop = float(data['ball_near_hoop_threshold'])
        if 'hoop_threshold' in data:
            detector.hoop_confidence_threshold = float(data['hoop_threshold'])
        return jsonify({'success': True, 'message': 'Thresholds updated'})
    return jsonify({'success': False, 'message': 'Detector not initialized'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video file upload"""
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid conflicts
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            # Switch detector to new video
            if detector.switch_video(filepath):
                return jsonify({'success': True, 'message': 'Video uploaded and processing started'})
            else:
                return jsonify({'success': False, 'message': 'Failed to load video'}), 500
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error saving file: {str(e)}'}), 500
    else:
        return jsonify({'success': False, 'message': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, or WEBM'}), 400

if __name__ == '__main__':
    # Parse command-line arguments
    video_path = None
    use_webcam = False
    port = 8888  # Default port
    model_size = 'medium'  # Default to medium model for better detection
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--webcam" or arg == "-w":
            use_webcam = True
        elif arg == "--port" or arg == "-p":
            if i + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 1])
                    i += 1
                except ValueError:
                    print(f"Warning: Invalid port number '{sys.argv[i + 1]}', using default 8888")
            else:
                print("Warning: --port requires a port number, using default 8888")
        elif arg == "--model" or arg == "-m":
            if i + 1 < len(sys.argv):
                model_size = sys.argv[i + 1].lower()
                valid_models = ['nano', 'small', 'medium', 'large', 'xlarge', 'custom']
                if model_size not in valid_models:
                    print(f"Warning: Invalid model size '{model_size}', using 'medium'")
                    print(f"Valid options: {', '.join(valid_models)}")
                    model_size = 'medium'
                i += 1
            else:
                print("Warning: --model requires a model size, using default 'medium'")
        elif not arg.startswith("-"):
            video_path = arg
        i += 1
    
    # Initialize detector with specified model
    print(f"\nUsing YOLO model size: {model_size}")
    print("Model options: nano (fastest), small, medium (recommended), large, xlarge (best), custom (best.pt)")
    detector = ShotDetectorWeb(video_path=video_path, use_webcam=use_webcam, model_size=model_size)
    
    print("\n" + "="*50)
    print("Basketball Shot Detector - Web Interface")
    print("="*50)
    print(f"Open your browser and go to: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    # Run Flask app
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)

