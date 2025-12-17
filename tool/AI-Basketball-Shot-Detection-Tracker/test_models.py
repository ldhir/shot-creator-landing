"""
Quick script to test different YOLO model sizes and see which works best
"""

import sys
import os
import time
from ultralytics import YOLO
import cv2
from utils import get_device

def test_model(model_name, video_path, num_frames=100):
    """Test a model and count detections"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    try:
        model = YOLO(model_name)
        device = get_device()
        cap = cv2.VideoCapture(video_path)
        
        ball_detections = 0
        hoop_detections = 0
        frames_processed = 0
        
        start_time = time.time()
        
        while frames_processed < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Test detection
            results = model(frame, conf=0.1, device=device, verbose=False)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    conf = float(box.conf[0])
                    
                    if class_name == "Basketball" and conf > 0.1:
                        ball_detections += 1
                    elif class_name == "Basketball Hoop" and conf > 0.3:
                        hoop_detections += 1
            
            frames_processed += 1
            
            if frames_processed % 20 == 0:
                print(f"  Processed {frames_processed} frames...")
        
        elapsed = time.time() - start_time
        fps = frames_processed / elapsed if elapsed > 0 else 0
        
        cap.release()
        
        print(f"\nResults for {model_name}:")
        print(f"  Frames processed: {frames_processed}")
        print(f"  Ball detections: {ball_detections} ({ball_detections/frames_processed*100:.1f}% of frames)")
        print(f"  Hoop detections: {hoop_detections} ({hoop_detections/frames_processed*100:.1f}% of frames)")
        print(f"  Speed: {fps:.1f} FPS")
        print(f"  Time: {elapsed:.1f}s")
        
        return {
            'model': model_name,
            'ball_detections': ball_detections,
            'hoop_detections': hoop_detections,
            'fps': fps,
            'ball_detection_rate': ball_detections / frames_processed if frames_processed > 0 else 0
        }
        
    except Exception as e:
        print(f"  ❌ Error testing {model_name}: {e}")
        return None

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "video_test_5.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        sys.exit(1)
    
    print("="*60)
    print("YOLO Model Comparison Test")
    print("="*60)
    print(f"Testing on: {video_path}")
    print("\nThis will test different YOLO model sizes to find the best one.")
    print("Models will be downloaded automatically if not present.")
    print("="*60)
    
    models_to_test = [
        ('yolov8n.pt', 'Nano (current)'),
        ('yolov8s.pt', 'Small'),
        ('yolov8m.pt', 'Medium ⭐ Recommended'),
        ('yolov8l.pt', 'Large'),
    ]
    
    results = []
    
    for model_name, description in models_to_test:
        print(f"\n{description}")
        result = test_model(model_name, video_path, num_frames=100)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Best Model for Ball Detection")
    print("="*60)
    
    if results:
        # Sort by ball detection rate
        results.sort(key=lambda x: x['ball_detection_rate'], reverse=True)
        
        print("\nRanked by ball detection rate:")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['model']}: {r['ball_detection_rate']*100:.1f}% detection rate, {r['fps']:.1f} FPS")
        
        best = results[0]
        print(f"\n⭐ Best model: {best['model']}")
        print(f"   Ball detection: {best['ball_detection_rate']*100:.1f}%")
        print(f"   Speed: {best['fps']:.1f} FPS")
        print(f"\nTo use this model, run:")
        print(f"  python shot_detector_web.py {video_path} --model {best['model'].replace('.pt', '').replace('yolov8', '')}")
    else:
        print("No results to compare.")
    
    print("="*60)

