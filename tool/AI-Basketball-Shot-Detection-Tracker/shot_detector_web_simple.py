# Avi Shah - Basketball Shot Detector/Tracker - July 2023
# Simple web-based version - based on original code with upload functionality

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
from werkzeug.utils import secure_filename
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
from cvzone.PoseModule import PoseDetector
import mediapipe as mp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Court image path
COURT_IMAGE_PATH = os.path.expanduser('~/Downloads/court_image.jpg')

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class ShotDetectorWeb:
    def __init__(self, video_path=None, use_webcam=False):
        # Load the YOLO model - original code
        self.overlay_text = "Waiting..."
        self.model = YOLO("best.pt")
        
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        
        # Initialize MediaPipe pose detector for 3D coordinates
        self.pose_detector = PoseDetector(
            staticMode=False,
            modelComplexity=1,
            smoothLandmarks=True,
            detectionCon=0.5,
            trackCon=0.5
        )
        # MediaPipe pose solution for world coordinates
        self.mp_pose = mp.solutions.pose
        self.pose_3d = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Rim reference: Standard basketball rim diameter = 18 inches = 45.72 cm = 0.4572 meters
        self.RIM_DIAMETER_METERS = 0.4572  # Exact value: 18 inches
        self.RIM_DIAMETER_CM = 45.72  # Exact value in centimeters
        self.RIM_HEIGHT_M = 3.048  # Rim height: 10 feet = 3.048 meters
        self.rim_scale_factor = None  # Will be calculated from rim width in MediaPipe space
        self.rim_pixel_scale = None  # Scale: pixels -> centimeters (from rim endpoints)
        self.rim_mp_scale = None  # Scale: MediaPipe units -> meters (from rim endpoints in MP space)
        self.rim_left_mp = None  # Rim left endpoint in MediaPipe 3D space
        self.rim_right_mp = None  # Rim right endpoint in MediaPipe 3D space
        
        # Camera position: 10 feet behind the 3-point line
        # 3-point line is 23.75 feet from hoop, so camera is 23.75 + 10 = 33.75 feet from hoop
        self.CAMERA_DISTANCE_FROM_HOOP_FEET = 33.75
        self.CAMERA_DISTANCE_FROM_HOOP_METERS = self.CAMERA_DISTANCE_FROM_HOOP_FEET * 0.3048  # Convert feet to meters
        
        # Court dimensions
        self.COURT_WIDTH_FEET = 50.0  # Total court width (25 feet each side from center)
        self.THREE_POINT_LINE_FEET = 23.75  # Distance from hoop to 3-point line
        
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

        self.ball_pos = []
        self.hoop_pos = []
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
        self.last_shot_frame = -100  # Track last shot to prevent duplicates
        self.video_fps = None  # Will be set from video
        self.frame_delay = 0.033  # Default ~30 FPS until we get actual FPS
        
        # Heatmap tracking: store (x, y, is_make) for each shot
        self.shot_heatmap = []  # List of tuples: (x, y, is_make)
        
        # MediaPipe 3D tracking
        self.current_player_world_pos = None
        self.rim_world_size = None  # Rim size in world coordinates (meters)
        self.shoulder_width_pixels = None  # Shoulder width in pixels (for depth estimation)
        self.shoulder_center_pixels = None  # Shoulder center position in pixels
        self.rim_width_pixels = None  # Rim width in pixels (for depth estimation)
        self.rim_height_pixels = None  # Rim height in pixels
        self.rim_endpoint_left_pix = None  # Rim left endpoint in pixels
        self.rim_endpoint_right_pix = None  # Rim right endpoint in pixels
        self.rim_bottom_left = None  # Rim bottom-left corner
        self.rim_bottom_right = None  # Rim bottom-right corner
        self.rim_bottom_center_pixels = None  # Rim bottom center position
        self.player_position_pixels = None  # Player position (hip center) in pixels
        self.player_distance_from_rim_feet = None  # Distance from rim to player in feet
        self.rim_scale_factor = None  # Scale factor: real_rim_width / mp_rim_width
        
        # Homography matrix for court projection
        self.homography_matrix = None  # Will be calculated from known court points
        self.court_points_video = []  # Points in video (pixels) - will be detected/calibrated
        self.court_points_real = []  # Corresponding real-world court coordinates (feet)
        
        # Start processing in background thread
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()

    def switch_video(self, new_video_path):
        """Switch to a new video file"""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        if self.cap is not None:
            self.cap.release()
        
        # Reset state
        with self.lock:
            self.ball_pos = []
            self.hoop_pos = []
            self.frame_count = 0
            self.makes = 0
            self.attempts = 0
            self.up = False
            self.down = False
            self.overlay_text = "Waiting..."
            self.overlay_color = (0, 0, 0)
            self.fade_counter = 0
            self.current_frame = None
            self.last_shot_frame = -100
            self.shot_heatmap = []  # Reset heatmap when switching videos
        
        if os.path.exists(new_video_path):
            self.cap = cv2.VideoCapture(new_video_path)
            self.running = True
            self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
            self.processing_thread.start()
            print(f"Switched to new video: {new_video_path}")
            return True
        else:
            print(f"Error: Video file '{new_video_path}' not found!")
            return False

    def process_video(self):
        """Process video frames - original algorithm"""
        while self.running:
            ret, self.frame = self.cap.read()
            
            if not ret:
                # Loop video if it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0
                self.ball_pos = []
                self.hoop_pos = []
                continue
            
            # Ensure we have a valid frame
            if self.frame is None or self.frame.size == 0:
                continue

            # Original detection code
            results = self.model(self.frame, stream=True, device=self.device)

            # Get MediaPipe 3D pose landmarks for court position calculation
            # Only process MediaPipe every few frames to avoid slowing down video
            pose_3d_landmarks = None
            player_world_pos = None
            if self.frame_count % 3 == 0:  # Process every 3rd frame to maintain video speed
                try:
                    # Convert frame to RGB for MediaPipe
                    frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose_3d.process(frame_rgb)
                    
                    if pose_results:
                        # Get 2D image coordinates for shoulder width calculation
                        if pose_results.pose_landmarks:
                            pose_2d_landmarks = pose_results.pose_landmarks.landmark
                            if len(pose_2d_landmarks) > 24:
                                # Get shoulder landmarks in image coordinates (normalized 0-1)
                                left_shoulder = pose_2d_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                                right_shoulder = pose_2d_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                                
                                # Get hip landmarks for player center position
                                left_hip = pose_2d_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                                right_hip = pose_2d_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                                
                                # Convert normalized coordinates to pixels
                                frame_height = self.frame.shape[0]
                                frame_width = self.frame.shape[1]
                                
                                left_shoulder_x = left_shoulder.x * frame_width
                                left_shoulder_y = left_shoulder.y * frame_height
                                right_shoulder_x = right_shoulder.x * frame_width
                                right_shoulder_y = right_shoulder.y * frame_height
                                
                                # Calculate shoulder width in pixels
                                shoulder_width = math.sqrt(
                                    (right_shoulder_x - left_shoulder_x)**2 + 
                                    (right_shoulder_y - left_shoulder_y)**2
                                )
                                
                                # Calculate player center position (midpoint of hips - this is where player stands)
                                hip_center_x = ((left_hip.x + right_hip.x) / 2) * frame_width
                                hip_center_y = ((left_hip.y + right_hip.y) / 2) * frame_height
                                
                                # Calculate shoulder center (midpoint between shoulders)
                                shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
                                shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2
                                
                                # Only store if visibility is good (MediaPipe provides visibility score)
                                if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                                    self.shoulder_width_pixels = shoulder_width
                                    self.shoulder_center_pixels = (shoulder_center_x, shoulder_center_y)
                                
                                # Store player position for shot mapping
                                if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                                    self.player_position_pixels = (hip_center_x, hip_center_y)
                        
                        # Get 3D world coordinates for player position
                        if pose_results.pose_world_landmarks:
                            pose_3d_landmarks = pose_results.pose_world_landmarks.landmark
                            # Get player's position (use midpoint of hips as reference)
                            # MediaPipe world coordinates: origin at center between hips, in meters
                            # Coordinate system:
                            #   X: left (negative) to right (positive)
                            #   Y: up (negative) to down (positive) 
                            #   Z: forward (positive, away from camera) to backward (negative, toward camera)
                            if len(pose_3d_landmarks) > 24:  # Ensure we have enough landmarks
                                left_hip = pose_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                                right_hip = pose_3d_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                                
                                # Player position relative to MediaPipe origin (hip center)
                                player_mp_x = (left_hip.x + right_hip.x) / 2  # Lateral (left/right)
                                player_mp_y = (left_hip.y + right_hip.y) / 2  # Vertical (up/down)
                                player_mp_z = (left_hip.z + right_hip.z) / 2  # Depth (forward/backward)
                                
                                # Transform to court coordinates:
                                # Camera is 10 feet behind 3pt line = 33.75 feet from hoop
                                # MediaPipe Z: positive = forward (away from camera)
                                # So player's distance from camera = CAMERA_DISTANCE - player_mp_z
                                # Player's distance from hoop = distance_from_camera - CAMERA_DISTANCE_FROM_HOOP
                                
                                # Convert MediaPipe coordinates (meters) to court coordinates (feet)
                                # X: lateral position (MediaPipe X is already left/right)
                                player_court_x_feet = player_mp_x * 3.28084  # Convert meters to feet
                                
                                # Z: depth position
                                # MediaPipe Z is relative to hip center, but we need absolute position
                                # For now, use Z as relative depth (positive = farther from camera)
                                # Distance from hoop = camera_distance - (camera_distance - player_depth)
                                # Simplified: if player is at origin, they're at camera_distance from hoop
                                # If player_mp_z is positive, they're farther from camera (closer to hoop)
                                # If player_mp_z is negative, they're closer to camera (farther from hoop)
                                
                                # Estimate: player_mp_z positive means closer to hoop
                                # Distance from hoop = CAMERA_DISTANCE - player_mp_z (in meters, converted to feet)
                                player_distance_from_hoop_feet = self.CAMERA_DISTANCE_FROM_HOOP_FEET - (player_mp_z * 3.28084)
                                
                                player_world_pos = {
                                    'x': player_court_x_feet,  # Lateral position in feet
                                    'y': player_mp_y,  # Vertical (not used for court mapping)
                                    'z': player_distance_from_hoop_feet,  # Distance from hoop in feet
                                    'mp_x': player_mp_x,  # Raw MediaPipe X
                                    'mp_z': player_mp_z   # Raw MediaPipe Z
                                }
                                
                                # Store for shot detection
                                self.current_player_world_pos = player_world_pos
                                
                                # Calculate rim scale factor using MediaPipe coordinates
                                if player_mp_z is not None:
                                    self.calculate_rim_scale_factor(player_mp_z)
                except Exception as e:
                    # If MediaPipe fails, continue without 3D coordinates
                    pass

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]
                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Original thresholds
                    if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        # DISABLED: Bounding box removed per user request
                        # cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    if conf > .5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        # Keep rim box visible
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))
                        
                        # Store rim width and corner positions in pixels for depth estimation
                        self.rim_width_pixels = w
                        self.rim_height_pixels = h
                        
                        # Rim endpoints (left and right edges) - these define the rim width
                        frame_height = self.frame.shape[0]
                        frame_width = self.frame.shape[1]
                        rim_center_y = int(y1 + h / 2)  # Vertical center of rim
                        
                        # Rim endpoints in pixels (left and right edges at vertical center)
                        self.rim_endpoint_left_pix = (x1, rim_center_y)
                        self.rim_endpoint_right_pix = (x2, rim_center_y)
                        
                        # Rim bounding box corners:
                        # Top-left: (x1, y1)
                        # Top-right: (x2, y1)
                        # Bottom-left: (x1, y2)
                        # Bottom-right: (x2, y2)
                        # Use bottom corners (closest to ground/player level) for better measurement
                        self.rim_bottom_left = (x1, y2)  # Bottom-left corner
                        self.rim_bottom_right = (x2, y2)  # Bottom-right corner
                        self.rim_bottom_center = (int(x1 + w/2), y2)  # Bottom center
                        
                        # Rim scale factor is calculated in MediaPipe processing section (every 30 frames)
                        # No need to calculate here to avoid duplicate calls
                        
                        # Draw lines from rim corners to shoulders if we have shoulder position
                        # DISABLED: Lines removed per user request
                        if hasattr(self, 'shoulder_center_pixels') and self.shoulder_center_pixels:
                            shoulder_x, shoulder_y = self.shoulder_center_pixels
                            
                            # Draw lines from rim bottom corners to shoulder center
                            # DISABLED: cv2.line(self.frame, self.rim_bottom_left, 
                            # DISABLED:         (int(shoulder_x), int(shoulder_y)), (0, 255, 255), 2)
                            # DISABLED: cv2.line(self.frame, self.rim_bottom_right, 
                            # DISABLED:         (int(shoulder_x), int(shoulder_y)), (0, 255, 255), 2)
                            
                            # Calculate distances from both corners to shoulder
                            dist_left = math.sqrt(
                                (shoulder_x - self.rim_bottom_left[0])**2 + 
                                (shoulder_y - self.rim_bottom_left[1])**2
                            )
                            dist_right = math.sqrt(
                                (shoulder_x - self.rim_bottom_right[0])**2 + 
                                (shoulder_y - self.rim_bottom_right[1])**2
                            )
                            
                            # Use average distance for more accuracy
                            distance_pixels = (dist_left + dist_right) / 2.0
                            
                            # Use rim width as reference to convert pixels to real distance
                            # Rim is 1.5 feet wide, so pixels_per_foot = rim_width / 1.5
                            if w > 0:
                                pixels_per_foot = w / 1.5
                                distance_feet = distance_pixels / pixels_per_foot
                                
                                # Store for shot detection
                                self.player_distance_from_rim_feet = distance_feet
                                
                                # Also store rim bottom center for lateral calculation
                                self.rim_bottom_center_pixels = self.rim_bottom_center
                        
                        # Calculate rim size in world coordinates using MediaPipe
                        if pose_3d_landmarks and player_world_pos:
                            # Use rim width in pixels and player's depth to estimate rim size
                            # Rim width in pixels = w
                            # We need to convert this to world coordinates
                            # Use player's Z depth as reference
                            self.rim_world_size = self.calculate_rim_world_size(w, h, player_world_pos)

            # Calculate homography less frequently to avoid performance issues
            # Only calculate every 10 frames (not every frame)
            if self.frame_count % 10 == 0:
                self.calculate_homography()
            
            self.clean_motion()
            self.shot_detection()
            # DISABLED: Text overlays removed per user request
            # self.display_score()
            
            with self.lock:
                self.current_frame = self.frame.copy()
            
            self.frame_count += 1
            # Don't add delay - process frames as fast as possible for smooth video
            # The video will play at processing speed, which is better than pausing

    def compute_scale_from_rim_mp(self, rim_left_mp, rim_right_mp):
        """
        Compute scale factor from MediaPipe rim endpoints.
        
        rim_left_mp/right_mp: numpy arrays [x,y,z] in MediaPipe 3D units
        Returns scale (meters per MediaPipe unit) or None if invalid.
        """
        rim_left_mp = np.asarray(rim_left_mp, dtype=float)
        rim_right_mp = np.asarray(rim_right_mp, dtype=float)
        d_mp = np.linalg.norm(rim_right_mp - rim_left_mp)
        if d_mp <= 1e-8:
            return None
        scale = self.RIM_DIAMETER_METERS / d_mp
        return scale
    
    def convert_mp_to_meters(self, joint_mp, scale_m_per_mp):
        """Convert MediaPipe coordinates to meters using scale factor."""
        return np.asarray(joint_mp, dtype=float) * float(scale_m_per_mp)
    
    def rim_center_mp_to_meters(self, rim_left_mp, rim_right_mp, scale_m_per_mp):
        """Convert rim center from MediaPipe space to meters."""
        center_mp = (np.asarray(rim_left_mp) + np.asarray(rim_right_mp)) / 2.0
        return self.convert_mp_to_meters(center_mp, scale_m_per_mp)
    
    def compute_distance_player_to_rim(self, player_mp, rim_left_mp, rim_right_mp):
        """
        Preferred path: uses MediaPipe rim endpoints to scale all MP units to meters 
        and compute Euclidean distance.
        
        Returns: (dist_m, player_m, rim_center_m) or (None, None, None) if can't compute
        """
        scale = self.compute_scale_from_rim_mp(rim_left_mp, rim_right_mp)
        if scale is None:
            return None, None, None  # can't compute
        
        player_m = self.convert_mp_to_meters(player_mp, scale)
        rim_center_m = self.rim_center_mp_to_meters(rim_left_mp, rim_right_mp, scale)
        dist_m = np.linalg.norm(player_m - rim_center_m)
        return dist_m, player_m, rim_center_m
    
    def compute_rim_3d_from_pixels(self, rim_left_px, rim_right_px, image_shape, f_px, cx=None, cy=None):
        """
        Compute rim 3D position from pixel coordinates using pinhole camera model.
        
        rim_left_px: (uL,vL), rim_right_px: (uR,vR)
        image_shape: (h, w)
        f_px: focal length in pixels (must be estimated or measured)
        cx, cy: principal point (defaults to image center)
        Returns rim_center_3d (X,Y,Z) in meters (camera frame) or None if error.
        """
        h, w = image_shape[:2]
        if cx is None:
            cx = w / 2.0
        if cy is None:
            cy = h / 2.0
        
        uL, vL = rim_left_px
        uR, vR = rim_right_px
        s_pix = np.hypot(uR - uL, vR - vL)
        if s_pix <= 1e-6:
            return None
        
        # depth of rim center in meters
        Z_rim = (f_px * self.RIM_DIAMETER_METERS) / float(s_pix)
        
        u_mid = 0.5 * (uL + uR)
        v_mid = 0.5 * (vL + vR)
        
        X_rim = (u_mid - cx) * Z_rim / f_px
        Y_rim = (v_mid - cy) * Z_rim / f_px
        
        return np.array([X_rim, Y_rim, Z_rim], dtype=float)
    
    def calculate_rim_scale_factor(self, player_mp_z):
        """
        STEP 1 & 2: Calculate scale factors using rim endpoints as reference.
        
        STEP 1: Compute pixel scale (pixels -> centimeters)
        STEP 2: Compute MediaPipe scale (MP units -> centimeters)
        
        OPTIMIZED: Only calculate every 30 frames to avoid freezing
        """
        # Only calculate every 30 frames to avoid performance issues
        if self.frame_count % 30 != 0:
            return  # Use cached value
        
        # If we already have valid scale factors, only recalculate occasionally
        if (self.rim_pixel_scale is not None and self.rim_mp_scale is not None and 
            self.frame_count % 60 != 0):
            return
        
        if (not hasattr(self, 'rim_endpoint_left_pix') or 
            not hasattr(self, 'rim_endpoint_right_pix') or
            self.rim_endpoint_left_pix is None or
            self.rim_endpoint_right_pix is None):
            return
        
        try:
            # STEP 1: Compute pixel scale (pixels -> centimeters)
            rim_left_x, rim_left_y = self.rim_endpoint_left_pix
            rim_right_x, rim_right_y = self.rim_endpoint_right_pix
            
            # Pixel width of rim
            w_px = np.linalg.norm(np.array([rim_right_x, rim_right_y]) - 
                                  np.array([rim_left_x, rim_left_y]))
            
            if w_px > 0:
                # Scale: pixels -> centimeters
                self.rim_pixel_scale = self.RIM_DIAMETER_CM / w_px  # cm per pixel
            
            # STEP 2: Compute MediaPipe scale (MP units -> centimeters)
            # Get frame dimensions
            frame_height = self.frame.shape[0]
            frame_width = self.frame.shape[1]
            
            # Convert pixel coordinates to normalized coordinates (0-1)
            rim_left_norm = (rim_left_x / frame_width, rim_left_y / frame_height)
            rim_right_norm = (rim_right_x / frame_width, rim_right_y / frame_height)
            
            # Estimate rim depth using player depth as reference
            if hasattr(self, 'shoulder_width_pixels') and self.shoulder_width_pixels and self.shoulder_width_pixels > 0:
                rim_mp_z = player_mp_z * (self.shoulder_width_pixels / self.rim_width_pixels)
            else:
                rim_mp_z = player_mp_z
            
            # Rim endpoints in MediaPipe 3D space
            rim_left_mp = np.array([
                rim_left_norm[0] - 0.5,  # Center X around 0
                rim_left_norm[1] - 0.5,  # Center Y around 0
                rim_mp_z
            ])
            
            rim_right_mp = np.array([
                rim_right_norm[0] - 0.5,
                rim_right_norm[1] - 0.5,
                rim_mp_z
            ])
            
            # Store rim endpoints in MediaPipe 3D space
            self.rim_left_mp = rim_left_mp
            self.rim_right_mp = rim_right_mp
            
            # Use the new function to compute scale (returns meters per MP unit)
            scale_m_per_mp = self.compute_scale_from_rim_mp(rim_left_mp, rim_right_mp)
            
            if scale_m_per_mp is not None and scale_m_per_mp > 0:
                # Store scale in meters per MP unit (preferred)
                self.rim_scale_factor = scale_m_per_mp
                # Also store in cm per MP unit for backward compatibility
                self.rim_mp_scale = scale_m_per_mp * 100.0  # cm per MP unit
                
                # Validate scale using rim height as sanity check
                rim_center_mp = (rim_left_mp + rim_right_mp) / 2.0
                rim_center_m = self.convert_mp_to_meters(rim_center_mp, scale_m_per_mp)
                # Rim should be at approximately RIM_HEIGHT_M (3.048m) above ground
                # Check if Z coordinate is reasonable (within 2-4 meters)
                if 2.0 <= abs(rim_center_m[2]) <= 4.0:
                    # Scale looks reasonable
                    pass
                else:
                    # Scale might be off, but continue anyway
                    pass
                
                # Debug: print scale factors occasionally
                if self.frame_count % 30 == 0:
                    print(f"Rim scales: pixel={self.rim_pixel_scale:.4f} cm/px, MP={scale_m_per_mp:.6f} m/MP-unit, rim_center_z={rim_center_m[2]:.2f}m")
        except Exception as e:
            # If calculation fails, keep existing scale factors or skip
            pass

    def calculate_homography(self):
        """
        Calculate homography matrix from detected court points.
        Maps video pixels -> real court coordinates (feet) using perspective transformation.
        Uses hoop position + estimated court points based on known dimensions.
        """
        if len(self.hoop_pos) == 0:
            return  # Need hoop as reference
        
        # Only calculate homography once when we first detect the hoop
        # Don't recalculate - it's expensive and causes freezing
        if self.homography_matrix is not None:
            return
        
        # Get hoop position
        hoop_x = self.hoop_pos[-1][0][0]
        hoop_y = self.hoop_pos[-1][0][1]
        rim_width = self.hoop_pos[-1][2]
        
        if rim_width <= 0:
            return
        
        # Estimate pixels per foot using rim width (rim is 1.5 feet wide)
        pixels_per_foot = rim_width / 1.5
        
        # Create 4 known points for homography:
        # 1. Hoop center (baseline, center) = (0, 0) in court coordinates
        # 2. Free throw line center (15 feet forward) = (0, 15)
        # 3. Left paint corner (8 feet left, 15 feet forward) = (-8, 15)
        # 4. Right paint corner (8 feet right, 15 feet forward) = (8, 15)
        
        # Video points (pixels) - estimated from hoop position
        # Note: This assumes camera is roughly perpendicular to court
        # For angled cameras, you'd need to detect actual court lines
        video_points = np.array([
            [hoop_x, hoop_y],  # Hoop center
            [hoop_x, hoop_y + 15 * pixels_per_foot],  # Free throw line (15 feet forward)
            [hoop_x - 8 * pixels_per_foot, hoop_y + 15 * pixels_per_foot],  # Left paint corner
            [hoop_x + 8 * pixels_per_foot, hoop_y + 15 * pixels_per_foot],  # Right paint corner
        ], dtype=np.float32)
        
        # Real court coordinates (feet, relative to hoop at origin)
        # X: left = negative, right = positive
        # Y: forward from hoop = positive
        court_points = np.array([
            [0, 0],      # Hoop center (baseline)
            [0, 15],     # Free throw line center
            [-8, 15],    # Left paint corner (paint is 16 feet wide, so ±8 feet)
            [8, 15],     # Right paint corner
        ], dtype=np.float32)
        
        # Calculate homography matrix using RANSAC for robustness
        try:
            # Use faster method - RANSAC with fewer iterations
            H, mask = cv2.findHomography(video_points, court_points, cv2.RANSAC, 5.0, maxIters=100)
            if H is not None:
                self.homography_matrix = H
                print(f"Homography calculated once: {len(video_points)} points")
        except Exception as e:
            print(f"Error calculating homography: {e}")
    
    def clean_motion(self):
        # Original cleaning code
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        # Improved shot detection - checks every frame and more lenient
        if len(self.hoop_pos) > 0 and len(self.ball_pos) >= 3:  # Need at least 3 ball positions
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # Check every frame instead of every 10 frames to catch all shots
            if self.up and self.down and self.up_frame < self.down_frame:
                # Ensure minimum time between up and down (at least 5 frames)
                if self.down_frame - self.up_frame >= 5:
                    # Prevent duplicate detections - only count if enough time has passed
                    if not hasattr(self, 'last_shot_frame') or self.frame_count - self.last_shot_frame > 15:
                        # CRITICAL FIX: Only use ball positions from THIS shot attempt
                        # Filter ball_pos to only include positions between up_frame and down_frame
                        shot_ball_pos = [ball for ball in self.ball_pos 
                                        if self.up_frame <= ball[1] <= self.down_frame]
                        
                        # Need at least 2 positions in the shot window to score
                        if len(shot_ball_pos) >= 2:
                            self.attempts += 1
                            self.last_shot_frame = self.frame_count
                            
                            # Score using ONLY the current shot's ball positions
                            is_make = self.improved_score(shot_ball_pos, self.hoop_pos)
                            if is_make:
                                self.makes += 1
                                self.overlay_color = (0, 255, 0)
                                self.overlay_text = "Make"
                                self.fade_counter = self.fade_frames
                            else:
                                self.overlay_color = (255, 0, 0)
                                self.overlay_text = "Miss"
                                self.fade_counter = self.fade_frames
                            
                            # Calculate shot position using HOMOGRAPHY (2D Court Projection)
                            # This maps video pixels to real court coordinates using perspective transformation
                            
                            # Standard court dimensions (in feet, relative to hoop at origin):
                            # - Hoop is at (0, 0) - baseline
                            # - Free throw line: 15 feet from baseline
                            # - 3-point line: 23.75 feet from basket
                            # - Court width: 50 feet total (25 feet each side from center)
                            
                            if len(shot_ball_pos) >= 2 and len(self.hoop_pos) > 0:
                                # Get frame dimensions
                                frame_height = self.frame.shape[0]
                                frame_width = self.frame.shape[1]
                                
                                # Hoop position in video (reference point - this is the baseline)
                                hoop_x = self.hoop_pos[-1][0][0]
                                hoop_y = self.hoop_pos[-1][0][1]
                                rim_width_pixels = self.hoop_pos[-1][2]  # Rim width in pixels
                                
                                # Get player/ball position in video pixels
                                # Use ball release position (simpler and faster than MediaPipe)
                                player_x = shot_ball_pos[0][0][0]
                                player_y = shot_ball_pos[0][0][1]
                                
                                # Debug: Check if ball positions are actually different
                                print(f"DEBUG: Shot #{self.attempts} - ball release pixel position: ({player_x:.1f}, {player_y:.1f}), frame: {shot_ball_pos[0][1]}")
                                
                                # PREFERRED METHOD: Use MediaPipe rim endpoints for scaling
                                if (hasattr(self, 'rim_left_mp') and hasattr(self, 'rim_right_mp') and
                                    self.rim_left_mp is not None and self.rim_right_mp is not None and
                                    hasattr(self, 'rim_scale_factor') and 
                                    self.rim_scale_factor is not None and
                                    self.rim_scale_factor > 0):
                                    
                                    # Use BALL RELEASE POSITION (varies per shot) instead of player hip position
                                    # Convert ball release pixel position to normalized coordinates
                                    ball_norm_x = player_x / frame_width
                                    ball_norm_y = player_y / frame_height
                                    
                                    # Estimate ball depth using player's MediaPipe depth as reference
                                    if (hasattr(self, 'current_player_world_pos') and 
                                        self.current_player_world_pos is not None):
                                        player_mp_z = self.current_player_world_pos['mp_z']
                                        # Ball is typically at similar depth to player (or slightly forward)
                                        ball_mp_z = player_mp_z
                                    else:
                                        # Fallback: estimate depth from rim
                                        if hasattr(self, 'rim_left_mp') and self.rim_left_mp is not None:
                                            # Use rim depth as reference
                                            ball_mp_z = self.rim_left_mp[2]
                                        else:
                                            ball_mp_z = 0.0
                                    
                                    # Ball release position in MediaPipe 3D space
                                    ball_mp = np.array([
                                        ball_norm_x - 0.5,  # Center X around 0
                                        ball_norm_y - 0.5,  # Center Y around 0
                                        ball_mp_z
                                    ])
                                    
                                    # Use preferred method: convert MP to meters using rim scale
                                    ball_m = self.convert_mp_to_meters(ball_mp, self.rim_scale_factor)
                                    
                                    # Get rim center in meters
                                    rim_center_m = self.rim_center_mp_to_meters(
                                        self.rim_left_mp, self.rim_right_mp, self.rim_scale_factor
                                    )
                                    
                                    # Translate to rim-centered coordinates (top-down court view)
                                    ball_rel_m = ball_m - rim_center_m
                                    
                                    # Project to court plane (drop Z height for top-down view)
                                    # Court coordinate system:
                                    # +X: Horizontal direction across rim (left/right)
                                    # +Y: Depth direction (toward free-throw line, away from camera)
                                    court_x_m = ball_rel_m[0]  # Lateral (left/right)
                                    court_y_m = ball_rel_m[1]  # Depth (forward/backward)
                                    
                                    # Convert meters to feet for court mapping
                                    court_x_feet = court_x_m * 3.28084  # 1 meter = 3.28084 feet
                                    court_y_feet = court_y_m * 3.28084
                                    
                                    # Convert to normalized heatmap coordinates (0-1)
                                    # X: -25 feet (left) to +25 feet (right) -> 0 to 1
                                    normalized_x = 0.5 + (court_x_feet / self.COURT_WIDTH_FEET)
                                    normalized_x = max(0.0, min(1.0, normalized_x))
                                    
                                    # Y: 0 feet (hoop) to 23.75 feet (3pt) -> 0 to 1
                                    # Note: court_y_feet is relative to rim, positive = forward
                                    court_y = min(max(0.0, court_y_feet) / self.THREE_POINT_LINE_FEET, 1.0)
                                    court_y = max(0.0, min(1.0, court_y))
                                    
                                    # Debug: print coordinates for each shot
                                    print(f"Shot (MP-scaled): ball_pix=({player_x:.1f}, {player_y:.1f}), ball_MP=({ball_mp[0]:.3f}, {ball_mp[1]:.3f}, {ball_mp[2]:.3f}), ball_m=({ball_m[0]:.2f}, {ball_m[1]:.2f}, {ball_m[2]:.2f})m, rim_center_m=({rim_center_m[0]:.2f}, {rim_center_m[1]:.2f}, {rim_center_m[2]:.2f})m, rel=({court_x_m:.2f}, {court_y_m:.2f})m, court=({court_x_feet:.2f}, {court_y_feet:.2f})ft, normalized=({normalized_x:.3f}, {court_y:.3f})")
                                
                                # FALLBACK: Use rim-based scaling method (old method for backward compatibility)
                                elif (hasattr(self, 'rim_mp_scale') and 
                                    self.rim_mp_scale is not None and
                                    self.rim_mp_scale > 0 and
                                    hasattr(self, 'rim_pixel_scale') and
                                    self.rim_pixel_scale is not None and
                                    self.rim_pixel_scale > 0):
                                    
                                    # Use BALL RELEASE POSITION (varies per shot) instead of player hip position
                                    # Convert ball release pixel position to normalized coordinates
                                    ball_norm_x = player_x / frame_width
                                    ball_norm_y = player_y / frame_height
                                    
                                    # Estimate ball depth using player's MediaPipe depth as reference
                                    if (hasattr(self, 'current_player_world_pos') and 
                                        self.current_player_world_pos is not None):
                                        player_mp_z = self.current_player_world_pos['mp_z']
                                        # Ball is typically at similar depth to player (or slightly forward)
                                        ball_mp_z = player_mp_z
                                    else:
                                        # Fallback: estimate depth from rim size
                                        if hasattr(self, 'shoulder_width_pixels') and self.shoulder_width_pixels and self.shoulder_width_pixels > 0:
                                            # Rough estimate: use rim depth estimation
                                            ball_mp_z = 0.0  # Will be estimated from rim
                                        else:
                                            ball_mp_z = 0.0
                                    
                                    # Ball release position in MediaPipe 3D space
                                    ball_mp = np.array([
                                        ball_norm_x - 0.5,  # Center X around 0
                                        ball_norm_y - 0.5,  # Center Y around 0
                                        ball_mp_z
                                    ])
                                    
                                    # Convert MediaPipe coordinates to real-world centimeters using rim scale
                                    ball_real_cm = ball_mp * self.rim_mp_scale
                                    
                                    # STEP 3: Build rim-centered coordinate system
                                    if (hasattr(self, 'rim_endpoint_left_pix') and 
                                        hasattr(self, 'rim_endpoint_right_pix') and
                                        self.rim_endpoint_left_pix and self.rim_endpoint_right_pix):
                                        
                                        frame_height = self.frame.shape[0]
                                        frame_width = self.frame.shape[1]
                                        
                                        # Get rim endpoints in normalized coordinates
                                        rim_left_x, rim_left_y = self.rim_endpoint_left_pix
                                        rim_right_x, rim_right_y = self.rim_endpoint_right_pix
                                        rim_left_norm = (rim_left_x / frame_width, rim_left_y / frame_height)
                                        rim_right_norm = (rim_right_x / frame_width, rim_right_y / frame_height)
                                        
                                        # Estimate rim depth
                                        if hasattr(self, 'shoulder_width_pixels') and self.shoulder_width_pixels and self.shoulder_width_pixels > 0:
                                            rim_mp_z = player_mp_z * (self.shoulder_width_pixels / self.rim_width_pixels)
                                        else:
                                            rim_mp_z = player_mp_z
                                        
                                        # Rim endpoints in MediaPipe 3D space
                                        rim_left_mp = np.array([
                                            rim_left_norm[0] - 0.5,
                                            rim_left_norm[1] - 0.5,
                                            rim_mp_z
                                        ])
                                        
                                        rim_right_mp = np.array([
                                            rim_right_norm[0] - 0.5,
                                            rim_right_norm[1] - 0.5,
                                            rim_mp_z
                                        ])
                                        
                                        # Rim center in MediaPipe space
                                        rim_center_mp = (rim_left_mp + rim_right_mp) / 2.0
                                        
                                        # Convert rim center to real-world centimeters
                                        rim_center_real_cm = rim_center_mp * self.rim_mp_scale
                                        
                                        # STEP 3: Translate ball release position to rim-centered coordinates
                                        ball_rel_cm = ball_real_cm - rim_center_real_cm
                                        
                                        # STEP 4: Project to court plane (top-down view, drop Z height)
                                        # Court coordinate system:
                                        # +X: Horizontal direction across rim (left/right)
                                        # +Y: Depth direction (toward free-throw line, away from camera)
                                        # Z: Upward (dropped for top-down view)
                                        court_x_cm = ball_rel_cm[0]  # Lateral (left/right)
                                        court_y_cm = ball_rel_cm[1]  # Depth (forward/backward)
                                        
                                        # Convert centimeters to feet for court mapping
                                        court_x_feet = court_x_cm / 30.48  # 1 foot = 30.48 cm
                                        court_y_feet = court_y_cm / 30.48
                                        
                                        # STEP 5: Convert to normalized heatmap coordinates (0-1)
                                        # X: -25 feet (left) to +25 feet (right) -> 0 to 1
                                        normalized_x = 0.5 + (court_x_feet / self.COURT_WIDTH_FEET)
                                        normalized_x = max(0.0, min(1.0, normalized_x))
                                        
                                        # Y: 0 feet (hoop) to 23.75 feet (3pt) -> 0 to 1
                                        # Note: court_y_feet is relative to rim, positive = forward
                                        court_y = min(max(0.0, court_y_feet) / self.THREE_POINT_LINE_FEET, 1.0)
                                        court_y = max(0.0, min(1.0, court_y))
                                        
                                        # Debug: print coordinates for each shot
                                        print(f"Shot (Rim-scaled fallback): ball_pix=({player_x:.1f}, {player_y:.1f}), ball_norm=({ball_norm_x:.3f}, {ball_norm_y:.3f}), ball_MP=({ball_mp[0]:.3f}, {ball_mp[1]:.3f}, {ball_mp[2]:.3f}), rim_center_MP=({rim_center_mp[0]:.3f}, {rim_center_mp[1]:.3f}, {rim_center_mp[2]:.3f}), rim_center_real=({rim_center_real_cm[0]:.1f}, {rim_center_real_cm[1]:.1f}, {rim_center_real_cm[2]:.1f})cm, ball_real=({ball_real_cm[0]:.1f}, {ball_real_cm[1]:.1f}, {ball_real_cm[2]:.1f})cm, rel=({court_x_cm:.1f}, {court_y_cm:.1f})cm, court=({court_x_feet:.2f}, {court_y_feet:.2f})ft, normalized=({normalized_x:.3f}, {court_y:.3f})")
                                
                                # Fallback: Use rim corners to calculate distance from rim to BALL RELEASE POINT
                                elif (hasattr(self, 'rim_bottom_left') and self.rim_bottom_left and
                                      hasattr(self, 'rim_bottom_right') and self.rim_bottom_right and
                                      rim_width_pixels > 0):
                                    
                                    # Calculate distance from rim bottom corners to ball release point
                                    rim_bottom_left_x, rim_bottom_left_y = self.rim_bottom_left
                                    rim_bottom_right_x, rim_bottom_right_y = self.rim_bottom_right
                                    
                                    # Distance from ball release to each rim corner
                                    dist_left = math.sqrt(
                                        (player_x - rim_bottom_left_x)**2 + 
                                        (player_y - rim_bottom_left_y)**2
                                    )
                                    dist_right = math.sqrt(
                                        (player_x - rim_bottom_right_x)**2 + 
                                        (player_y - rim_bottom_right_y)**2
                                    )
                                    
                                    # Use average distance for more accuracy
                                    distance_pixels = (dist_left + dist_right) / 2.0
                                    
                                    # Convert pixels to feet using rim width as scale (rim is 1.5 feet wide)
                                    pixels_per_foot = rim_width_pixels / 1.5
                                    distance_feet = distance_pixels / pixels_per_foot
                                    
                                    # Calculate lateral position using rim bottom center
                                    rim_bottom_center_x = (rim_bottom_left_x + rim_bottom_right_x) / 2
                                    rim_bottom_center_y = (rim_bottom_left_y + rim_bottom_right_y) / 2
                                    
                                    # Calculate lateral offset in pixels from rim bottom center
                                    dx_pixels = player_x - rim_bottom_center_x
                                    
                                    # Convert to feet using rim width as scale
                                    lateral_distance_feet = abs(dx_pixels) / pixels_per_foot
                                    
                                    # Normalize X: -25 feet (left) to +25 feet (right) -> 0 to 1
                                    if dx_pixels >= 0:
                                        normalized_x = 0.5 + min(lateral_distance_feet / 25.0, 0.5)
                                    else:
                                        normalized_x = 0.5 - min(lateral_distance_feet / 25.0, 0.5)
                                    
                                    normalized_x = max(0.0, min(1.0, normalized_x))
                                    
                                    # Y coordinate: distance from hoop in feet (calculated from ball release point)
                                    # Normalize: 0 feet (hoop) to 23.75 feet (3pt) -> 0 to 1
                                    court_y = min(distance_feet / 23.75, 1.0)
                                    court_y = max(0.0, min(1.0, court_y))
                                    
                                    # Debug: print coordinates for each shot
                                    print(f"Shot detected (rim corners): ball at ({player_x:.1f}, {player_y:.1f}), rim bottom at ({rim_bottom_center_x:.1f}, {rim_bottom_center_y:.1f}), distance={distance_feet:.2f}ft, normalized=({normalized_x:.3f}, {court_y:.3f})")
                                    
                                # Try to use homography if we have it calibrated
                                elif self.homography_matrix is not None:
                                    try:
                                        # Use homography to transform video pixel to court coordinates
                                        point_video = np.array([[[player_x, player_y]]], dtype=np.float32)
                                        point_court = cv2.perspectiveTransform(point_video, self.homography_matrix)
                                        
                                        # point_court is in real-world court coordinates (feet)
                                        court_x_feet = point_court[0][0][0]
                                        court_y_feet = point_court[0][0][1]
                                        
                                        # Normalize to 0-1 for heatmap
                                        # X: -25 feet (left) to +25 feet (right) -> 0 to 1
                                        normalized_x = 0.5 + (court_x_feet / 50.0)
                                        # Y: 0 feet (hoop) to 23.75 feet (3pt) -> 0 to 1
                                        court_y = min(court_y_feet / 23.75, 1.0)
                                        
                                        normalized_x = max(0.0, min(1.0, normalized_x))
                                        court_y = max(0.0, min(1.0, court_y))
                                    except Exception as e:
                                        # If homography transform fails, fall back to ratio method
                                        # Don't print - it causes blocking
                                        normalized_x = 0.5
                                        court_y = 0.5
                                    
                                # Fallback: Use shoulder-to-rim ratio method if homography not available
                                elif hasattr(self, 'shoulder_width_pixels') and hasattr(self, 'rim_width_pixels') and \
                                     self.shoulder_width_pixels is not None and self.rim_width_pixels is not None and \
                                     self.rim_width_pixels > 0 and self.shoulder_width_pixels > 0:
                                    # Real-world measurements:
                                    # - Rim diameter: 18 inches = 0.457 meters = 1.5 feet
                                    # - Average shoulder width: ~0.4-0.5 meters = ~1.3-1.6 feet
                                    
                                    shoulder_rim_ratio = self.shoulder_width_pixels / self.rim_width_pixels
                                    
                                    # Estimate distance from camera using ratio
                                    # At same distance: shoulder/rim ≈ 0.875 (since shoulder ~0.4m, rim ~0.457m)
                                    # Closer to camera: ratio increases
                                    # Further from camera: ratio decreases
                                    
                                    # Use rim size to estimate pixel-to-meter conversion
                                    # Rim is 0.457m in real world
                                    # If rim appears as rim_width_pixels, then: 1 pixel = 0.457 / rim_width_pixels meters
                                    
                                    # But we need to account for distance - use shoulder ratio
                                    # Estimate: if ratio is high, player is close (rim appears larger)
                                    # If ratio is low, player is far (rim appears smaller)
                                    
                                    # Simplified: use ratio to estimate distance scale
                                    # Expected ratio at ~15 feet (free throw): ~0.875
                                    # At 15 feet, rim should appear smaller than at 5 feet
                                    
                                    # Estimate distance from hoop in feet
                                    # This is approximate - would need camera calibration for accuracy
                                    if shoulder_rim_ratio > 1.2:
                                        # Very close: ~5-10 feet (paint area)
                                        estimated_distance_feet = 5 + (1.2 / shoulder_rim_ratio) * 5
                                    elif shoulder_rim_ratio > 0.8:
                                        # Mid-range: ~10-18 feet (free throw to mid-range)
                                        estimated_distance_feet = 10 + ((0.8 / shoulder_rim_ratio) - 1) * 8
                                    else:
                                        # Far: ~18-25 feet (3pt line)
                                        estimated_distance_feet = 18 + ((0.8 / shoulder_rim_ratio) - 1) * 7
                                    
                                    # Map distance to normalized Y coordinate
                                    # Court: 0 feet = hoop (baseline), 23.75 feet = 3pt line
                                    # Normalize: 0 = hoop, 1 = 3pt line (23.75 feet)
                                    court_y = min(estimated_distance_feet / 23.75, 1.0)
                                    court_y = max(0.0, min(1.0, court_y))
                                    
                                    # X coordinate: Calculate lateral position in feet
                                    # Use rim width to estimate scale at this distance
                                    # Rim is 1.5 feet wide, so pixels_per_foot ≈ rim_width_pixels / 1.5
                                    pixels_per_foot = rim_width_pixels / 1.5
                                    
                                    # Lateral distance in feet from hoop center
                                    lateral_distance_feet = abs(dx_pixels) / pixels_per_foot
                                    
                                    # Court width is 50 feet, so half court is 25 feet each side
                                    # Normalize: -25 feet (left) to +25 feet (right) -> 0 to 1
                                    if dx_pixels >= 0:
                                        # Right of hoop
                                        normalized_x = 0.5 + min(lateral_distance_feet / 25.0, 0.5)
                                    else:
                                        # Left of hoop
                                        normalized_x = 0.5 - min(lateral_distance_feet / 25.0, 0.5)
                                    
                                    normalized_x = max(0.0, min(1.0, normalized_x))
                                    
                                else:
                                    # Fallback: use simple pixel-based mapping
                                    # X: normalize lateral offset
                                    release_x_norm = release_x / frame_width
                                    hoop_x_norm = hoop_x / frame_width
                                    lateral_offset = (release_x_norm - hoop_x_norm) * 2.0
                                    normalized_x = 0.5 + lateral_offset * 0.4
                                    normalized_x = max(0.1, min(0.9, normalized_x))
                                    
                                    # Y: use Y position as fallback
                                    release_y_norm = release_y / frame_height
                                    hoop_y_norm = hoop_y / frame_height
                                    y_distance = release_y_norm - hoop_y_norm
                                    court_y = 0.1 + min(y_distance * 4.0, 0.8)
                                    court_y = max(0.1, min(0.95, court_y))
                                
                                # Always append to heatmap (even if using fallback)
                                # Removed debug prints to avoid blocking
                                self.shot_heatmap.append({
                                    'x': normalized_x,
                                    'y': court_y,
                                    'is_make': is_make
                                })
                                
                                # OLD CODE - REMOVED (was using player position which doesn't vary)
                                # Use MediaPipe 3D coordinates if available (quick check)
                                if False and hasattr(self, 'current_player_world_pos') and self.current_player_world_pos and hasattr(self, 'rim_world_size') and self.rim_world_size:
                                    # Calculate court position from 3D world coordinates
                                    player_pos = self.current_player_world_pos
                                    
                                    # MediaPipe world coordinates:
                                    # - Origin at center between hips
                                    # - X: left/right (positive = right)
                                    # - Y: up/down (positive = down)
                                    # - Z: forward/backward (positive = forward, away from camera)
                                    
                                    # Rim is at baseline, player is shooting from court
                                    # We need to map player's 3D position to court coordinates
                                    
                                    # Use rim as reference point (at baseline, Z = distance from camera)
                                    # Player's Z position relative to rim gives distance from hoop
                                    
                                    # Standard court dimensions:
                                    # - Free throw line: 15 feet (4.57 meters) from baseline
                                    # - 3pt line: ~23.75 feet (7.24 meters) from baseline at top
                                    
                                    # Map player's world Z to court Y (distance from hoop)
                                    # Assuming rim is at Z=0 (reference), player's Z is distance
                                    # Negative Z = closer to camera = further from hoop
                                    # Positive Z = further from camera = closer to hoop
                                    
                                    # Actually, MediaPipe Z: positive = forward (away from camera)
                                    # So if player is at Z < 0, they're closer to camera = further from hoop
                                    
                                    # Calculate court position
                                    # X: lateral position (left/right)
                                    # Y: distance from hoop (baseline to 3pt line)
                                    
                                    # Normalize X: -1 to 1 (left to right)
                                    # MediaPipe X: positive = right, negative = left
                                    # Court X: 0 = center, -1 = left, 1 = right
                                    normalized_x = 0.5 + (player_pos['x'] * 2.0)  # Scale and center
                                    normalized_x = max(0.05, min(0.95, normalized_x))
                                    
                                    # Normalize Y: distance from hoop
                                    # Use Z coordinate to estimate distance
                                    # Rim is at baseline, player shoots from court
                                    # Negative Z = closer to camera = further from hoop (3pt line)
                                    # Positive Z = further from camera = closer to hoop (paint)
                                    
                                    # Estimate distance using Z and rim size as scale
                                    # Use rim diameter (0.457m) as reference
                                    if self.rim_world_size:
                                        scale_factor = self.RIM_DIAMETER_METERS / self.rim_world_size
                                    else:
                                        scale_factor = 1.0
                                    
                                    # Player's distance from hoop in meters
                                    # Z coordinate gives relative position
                                    # We need to estimate absolute distance
                                    # Use a heuristic: if Z is very negative, player is far (3pt)
                                    # If Z is close to 0 or positive, player is close (paint)
                                    
                                    # Map Z to court Y coordinate
                                    # Assume Z range: -2 to 1 meters (3pt to paint)
                                    z_normalized = (player_pos['z'] + 2.0) / 3.0  # Normalize -2 to 1 -> 0 to 1
                                    z_normalized = max(0.0, min(1.0, z_normalized))
                                    
                                    # Invert: negative Z (far) should map to high court Y (3pt line)
                                    court_y = 0.2 + (1.0 - z_normalized) * 0.7  # 0.2 (paint) to 0.9 (3pt)
                                    court_y = max(0.15, min(0.9, court_y))
                                    
                                    # This code is disabled - we now always use ball release position
                                    pass
                            
                            # Clear old ball positions to prevent contamination for next shot
                            # Keep only positions from the last 10 frames to maintain continuity
                            self.ball_pos = [ball for ball in self.ball_pos 
                                            if ball[1] >= self.down_frame - 10]
                            
                            self.up = False
                            self.down = False

    def calculate_rim_world_size(self, rim_width_pixels, rim_height_pixels, player_world_pos):
        """Calculate rim size in world coordinates using player's depth as reference"""
        # Use player's Z depth to estimate pixel-to-meter conversion
        # This is a rough estimate - would need camera calibration for accuracy
        
        # Estimate: if player is at Z depth, use that to scale
        # Standard approach: use known object size (rim) and distance
        # But we're doing reverse: use rim pixels and player depth to estimate rim world size
        
        # Rough heuristic: assume player depth gives us scale
        # Player at Z=0 means they're at same depth as reference
        # Use a default scale based on typical camera setup
        # This is approximate - proper solution needs camera calibration
        
        # For now, use a simple scale factor
        # Typical: 1 meter at distance appears as ~100-200 pixels depending on camera
        pixels_per_meter = 150.0  # Rough estimate
        
        rim_size_meters = rim_width_pixels / pixels_per_meter
        return rim_size_meters

    def display_score(self):
        # DISABLED: All text overlays removed per user request
        # Updated display code with serif font (matching Roots branding)
        # text = str(self.makes) + " / " + str(self.attempts)
        # # Use serif font (FONT_HERSHEY_COMPLEX) for main score - matches "Roots" serif style
        # cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 6)
        # cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 3)

        # if hasattr(self, 'overlay_text'):
        #     (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_COMPLEX, 3, 6)
        #     text_x = self.frame.shape[1] - text_width - 40
        #     text_y = 100

        #     cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 3,
        #                 self.overlay_color, 6)

        # if self.fade_counter > 0:
        #     alpha = 0.2 * (self.fade_counter / self.fade_frames)
        #     self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
        #     self.fade_counter -= 1
        pass  # Function disabled - no visualizations on video

    def improved_score(self, ball_pos, hoop_pos):
        """Improved scoring function - more lenient and handles edge cases"""
        if len(ball_pos) < 2 or len(hoop_pos) < 1:
            return False
        
        rim_center_x = hoop_pos[-1][0][0]
        rim_center_y = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]  # Rim height
        rim_width = hoop_pos[-1][2]
        rim_height = hoop_pos[-1][3]
        
        # Method 1: Check if ball passed through rim area (more lenient)
        rim_x1 = rim_center_x - 0.5 * rim_width  # Wider tolerance
        rim_x2 = rim_center_x + 0.5 * rim_width
        rim_y1 = rim_center_y - 0.3 * rim_height  # Above rim
        rim_y2 = rim_center_y + 0.5 * rim_height  # Below rim
        
        ball_passed_through = False
        points_above = []
        points_below = []
        
        # Find points above and below rim
        for ball in ball_pos:
            bx, by = ball[0]
            if by < rim_center_y:  # Above rim
                points_above.append((bx, by))
            elif by > rim_center_y:  # Below rim
                points_below.append((bx, by))
            
            # Check if ball center was in rim area
            if rim_x1 < bx < rim_x2 and rim_y1 < by < rim_y2:
                ball_passed_through = True
        
        # Method 2: Trajectory prediction (original method, but more lenient)
        trajectory_make = False
        if len(points_above) > 0 and len(points_below) > 0:
            # Use multiple points for better trajectory
            x_coords = []
            y_coords = []
            
            # Get points near rim for trajectory
            for ball in ball_pos:
                bx, by = ball[0]
                # Include points within 2x hoop width horizontally
                if abs(bx - rim_center_x) < 2 * rim_width:
                    x_coords.append(bx)
                    y_coords.append(by)
            
            if len(x_coords) >= 2:
                try:
                    # Linear regression
                    m, b = np.polyfit(x_coords, y_coords, 1)
                    predicted_x = (rim_center_y - b) / m
                    
                    # More lenient rim check (1.2x rim width)
                    if rim_center_x - 0.6 * rim_width < predicted_x < rim_center_x + 0.6 * rim_width:
                        trajectory_make = True
                except:
                    pass
        
        # Method 3: Check if ball was near rim center (simple proximity check)
        proximity_make = False
        min_dist_to_rim = float('inf')
        for ball in ball_pos:
            bx, by = ball[0]
            dist = math.sqrt((bx - rim_center_x)**2 + (by - rim_center_y)**2)
            min_dist_to_rim = min(min_dist_to_rim, dist)
        
        # If ball got very close to rim center, likely a make
        if min_dist_to_rim < 0.4 * rim_width:
            proximity_make = True
        
        # Return True if any method suggests a make
        return ball_passed_through or trajectory_make or proximity_make

    def get_frame(self):
        """Get current frame as JPEG bytes"""
        with self.lock:
            if self.current_frame is None:
                # Return a placeholder frame if no frame available
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Loading video...", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    return buffer.tobytes()
                return None
            ret, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                return None
            return buffer.tobytes()

    def get_stats(self):
        """Get current statistics"""
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
            'heatmap': self.shot_heatmap  # Include heatmap data
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
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-container {
                background: rgba(255, 255, 255, 0.15);
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 30px;
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
            }
            .file-input-label:hover {
                background: rgba(255, 255, 255, 0.4);
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
            }
            .file-name {
                margin-top: 10px;
                font-size: 14px;
            }
            #message {
                margin-top: 15px;
                padding: 10px;
                border-radius: 8px;
            }
            .message.success {
                background: rgba(0, 255, 0, 0.2);
                color: #00ff00;
            }
            .message.error {
                background: rgba(255, 0, 0, 0.2);
                color: #ff0000;
            }
            .video-container {
                text-align: center;
                margin-bottom: 30px;
            }
            #videoStream {
                max-width: 100%;
                border-radius: 15px;
            }
            .heatmap-container {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 15px;
                margin-top: 30px;
                text-align: center;
            }
            .heatmap-title {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 15px;
            }
            #heatmapCanvas {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                max-width: 100%;
                height: auto;
            }
            .stats-container {
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 20px;
            }
            .stat-box {
                background: rgba(255, 255, 255, 0.2);
                padding: 20px 40px;
                border-radius: 15px;
                text-align: center;
                min-width: 200px;
            }
            .stat-value {
                font-size: 48px;
                font-weight: bold;
            }
            .stat-label {
                font-size: 18px;
            }
            .overlay-indicator {
                font-size: 36px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin-top: 20px;
                display: inline-block;
            }
            .make {
                background: rgba(0, 255, 0, 0.3);
                color: #00ff00;
            }
            .miss {
                background: rgba(255, 0, 0, 0.3);
                color: #ff0000;
            }
            .waiting {
                background: rgba(128, 128, 128, 0.3);
                color: #cccccc;
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
            <div class="heatmap-container">
                <div class="heatmap-title">Shot Heatmap</div>
                <canvas id="heatmapCanvas" width="800" height="600"></canvas>
                <div style="margin-top: 15px; font-size: 14px;">
                    <span style="color: #00ff00;">● Makes</span> | 
                    <span style="color: #ff0000;">✕ Misses</span>
                </div>
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
        </div>
        <script>
            document.getElementById('videoFile').addEventListener('change', function(e) {
                const fileName = e.target.files[0] ? e.target.files[0].name : '';
                document.getElementById('fileName').textContent = fileName ? 'Selected: ' + fileName : '';
            });

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

            // Heatmap canvas setup
            const canvas = document.getElementById('heatmapCanvas');
            const ctx = canvas.getContext('2d');
            let lastHeatmapLength = 0;
            
            // Load court image
            const courtImage = new Image();
            courtImage.src = '/court_image';
            let courtImageLoaded = false;
            
            courtImage.onload = function() {
                courtImageLoaded = true;
                // Redraw heatmap when image loads
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        if (data.heatmap && data.heatmap.length > 0) {
                            console.log('Court image loaded, drawing heatmap:', data.heatmap.length, 'shots');
                            drawHeatmap(data.heatmap);
                            lastHeatmapLength = data.heatmap.length;
                        }
                    });
            };
            
            courtImage.onerror = function() {
                console.log('Court image not found, using drawn court');
                courtImageLoaded = false;
            };
            
            function drawBasketballCourt(ctx, width, height) {
                // Draw a grid instead of court for debugging
                ctx.fillStyle = 'rgba(240, 240, 240, 0.8)';
                ctx.fillRect(0, 0, width, height);
                
                // Draw grid lines
                ctx.strokeStyle = 'rgba(200, 200, 200, 0.6)';
                ctx.lineWidth = 1;
                
                // Vertical lines (10 columns)
                const numCols = 10;
                for (let i = 0; i <= numCols; i++) {
                    const x = (width / numCols) * i;
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, height);
                    ctx.stroke();
                }
                
                // Horizontal lines (10 rows)
                const numRows = 10;
                for (let i = 0; i <= numRows; i++) {
                    const y = (height / numRows) * i;
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(width, y);
                    ctx.stroke();
                }
                
                // Draw coordinate labels
                ctx.fillStyle = 'rgba(100, 100, 100, 0.8)';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                
                // Label columns (X: 0.0 to 1.0)
                for (let i = 0; i <= numCols; i++) {
                    const x = (width / numCols) * i;
                    const label = (i / numCols).toFixed(1);
                    ctx.fillText(label, x, 15);
                }
                
                // Label rows (Y: 0.0 to 1.0)
                ctx.textAlign = 'left';
                for (let i = 0; i <= numRows; i++) {
                    const y = (height / numRows) * i;
                    const label = (i / numRows).toFixed(1);
                    ctx.fillText(label, 5, y);
                }
                
                // Mark center (0.5, 0.5) and hoop position
                ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
                ctx.lineWidth = 2;
                const centerX = width * 0.5;
                const centerY = height * 0.5;
                ctx.beginPath();
                ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
                ctx.stroke();
                ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
                ctx.font = 'bold 14px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('CENTER (0.5, 0.5)', centerX, centerY - 20);
                
                // Mark hoop position (top center, y ≈ 0.1)
                const hoopX = width * 0.5;
                const hoopY = height * 0.1;
                ctx.strokeStyle = 'rgba(0, 0, 255, 0.8)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(hoopX, hoopY, 8, 0, Math.PI * 2);
                ctx.stroke();
                ctx.fillStyle = 'rgba(0, 0, 255, 0.8)';
                ctx.fillText('HOOP (0.5, 0.1)', hoopX, hoopY - 25);
            }
            
            function drawHeatmap(heatmapData) {
                console.log('Drawing heatmap with', heatmapData.length, 'shots');
                // Clear canvas
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Draw basketball court (image or drawn)
                drawBasketballCourt(ctx, canvas.width, canvas.height);
                
                // Draw each shot
                // Court layout: hoop at top center (the T symbol with circle)
                // Hoop position on canvas matches the drawn hoop position
                const courtX = canvas.width * 0.05;  // Left edge of court
                const courtY = canvas.height * 0.1;  // Top edge of court
                const courtWidth = canvas.width * 0.9;  // Court width
                const courtHeight = canvas.height * 0.85;  // Court height
                
                // Hoop position (matches drawBasketballCourt function)
                const hoopX = courtX + courtWidth / 2;  // Hoop X (center of court)
                const hoopY = courtY + courtHeight * 0.05;  // Hoop Y (5% down from court top)
                
                heatmapData.forEach((shot, index) => {
                    // SIMPLIFIED MAPPING: Map normalized coordinates directly to court
                    // Backend sends shot.x and shot.y as normalized (0-1)
                    // We'll map them directly to the court area
                    
                    // X coordinate: 0 = left edge of court, 1 = right edge of court
                    // shot.x = 0.5 should be at hoop center (which is at court center)
                    const x = courtX + (shot.x * courtWidth);
                    
                    // Y coordinate: Map to court height
                    // The court drawing shows hoop at top (y = courtY + 5% of courtHeight)
                    // shot.y values from backend: smaller = closer to hoop, larger = farther
                    // Map directly: 0 = top of court (hoop area), 1 = bottom of court (3pt line)
                    const y = courtY + (shot.y * courtHeight);
                    
                    // Debug: log first shot to see what we're getting
                    if (index === 0 && heatmapData.length > 0) {
                        console.log(`First shot: normalized(${shot.x.toFixed(3)}, ${shot.y.toFixed(3)}) -> canvas(${x.toFixed(1)}, ${y.toFixed(1)}), hoop at (${hoopX.toFixed(1)}, ${hoopY.toFixed(1)})`);
                    }
                    
                    // Debug logging removed - uncomment if needed for troubleshooting
                    // if (index < 3) {
                    //     console.log(`Shot ${index}: normalized(${shot.x.toFixed(3)}, ${shot.y.toFixed(3)}), canvas(${x.toFixed(1)}, ${y.toFixed(1)})`);
                    // }
                    
                    if (shot.is_make) {
                        // Green circle for makes
                        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
                        ctx.strokeStyle = 'rgba(0, 200, 0, 1)';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(x, y, 12, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.stroke();
                    } else {
                        // Red X for misses
                        ctx.strokeStyle = 'rgba(255, 0, 0, 1)';
                        ctx.lineWidth = 3;
                        const size = 10;
                        ctx.beginPath();
                        ctx.moveTo(x - size, y - size);
                        ctx.lineTo(x + size, y + size);
                        ctx.moveTo(x + size, y - size);
                        ctx.lineTo(x - size, y + size);
                        ctx.stroke();
                    }
                });
            }
            
            setInterval(function() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('makes').textContent = data.makes;
                        document.getElementById('attempts').textContent = data.attempts;
                        document.getElementById('score').textContent = data.makes + ' / ' + data.attempts;
                        
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
                        
                        // Update heatmap if new shots detected
                        if (data.heatmap) {
                            // Always update if length changed OR if we haven't drawn yet
                            if (data.heatmap.length !== lastHeatmapLength || (lastHeatmapLength === 0 && data.heatmap.length > 0)) {
                                console.log('Updating heatmap:', data.heatmap.length, 'shots (was:', lastHeatmapLength, ')');
                                drawHeatmap(data.heatmap);
                                lastHeatmapLength = data.heatmap.length;
                            }
                        } else {
                            console.log('No heatmap data in response');
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
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Get current statistics as JSON"""
    return jsonify(detector.get_stats())

@app.route('/court_image')
def court_image():
    """Serve the court image"""
    if os.path.exists(COURT_IMAGE_PATH):
        return Response(open(COURT_IMAGE_PATH, 'rb').read(), mimetype='image/jpeg')
    else:
        # Return a placeholder if image doesn't exist
        return Response(status=404)

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
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            if detector.switch_video(filepath):
                return jsonify({'success': True, 'message': 'Video uploaded and processing started'})
            else:
                return jsonify({'success': False, 'message': 'Failed to load video'}), 500
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error saving file: {str(e)}'}), 500
    else:
        return jsonify({'success': False, 'message': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, or WEBM'}), 400

if __name__ == '__main__':
    video_path = None
    use_webcam = False
    port = 8888
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--webcam" or sys.argv[1] == "-w":
            use_webcam = True
        else:
            video_path = sys.argv[1]
    
    detector = ShotDetectorWeb(video_path=video_path, use_webcam=use_webcam)
    
    print("\n" + "="*50)
    print("Basketball Shot Detector - Web Interface")
    print("="*50)
    print(f"Open your browser and go to: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)

