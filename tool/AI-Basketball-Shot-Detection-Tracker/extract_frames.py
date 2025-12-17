"""
Extract frames from videos for dataset creation
Use this to create frames that you can label for training
"""

import cv2
import os
import sys
import argparse

def extract_frames(video_path, output_dir, frame_interval=30, max_frames=None):
    """
    Extract frames from video for labeling
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame (default: 30 = 1 frame per second at 30fps)
        max_frames: Maximum number of frames to extract (None = all)
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return
    
    frame_count = 0
    saved_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Extracting every {frame_interval} frames...")
    print(f"Output directory: {output_dir}/images")
    print("-" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(output_dir, "images", frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Extracted {saved_count} frames...")
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print("-" * 50)
    print(f"✓ Extracted {saved_count} frames to {output_dir}/images")
    print(f"✓ Ready for labeling!")
    print(f"\nNext steps:")
    print(f"1. Use LabelImg or Roboflow to label basketballs in these frames")
    print(f"2. Save labels in YOLO format")
    print(f"3. Organize into train/val/test folders")
    print(f"4. Update config.yaml with paths")
    print(f"5. Run train_improved.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from video for dataset creation')
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('-o', '--output', default='frames_to_label', 
                       help='Output directory (default: frames_to_label)')
    parser.add_argument('-i', '--interval', type=int, default=30,
                       help='Extract every Nth frame (default: 30)')
    parser.add_argument('-m', '--max', type=int, default=None,
                       help='Maximum number of frames to extract (default: all)')
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.interval, args.max)

