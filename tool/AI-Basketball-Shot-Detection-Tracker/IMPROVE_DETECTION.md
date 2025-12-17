# Guide to Improving Basketball Detection

## Current Issues
- Ball detection is unreliable (rarely detects basketballs)
- Model may not be well-suited for your specific video conditions

## Solutions

### 1. **Improve the Dataset** (Most Important!)

#### Option A: Use Better Public Datasets
1. **Roboflow Basketball Dataset** (Recommended)
   - Download: https://universe.roboflow.com/robocon-qchql/hoops-chrie-ivyrc/dataset/3
   - 1300+ images with annotations
   - Better quality than the smaller dataset

2. **Create Your Own Dataset from Your Videos**
   - Extract frames from your videos where ball is clearly visible
   - Use labeling tools like:
     - **LabelImg** (https://github.com/HumanSignal/labelImg) - Free, easy to use
     - **Roboflow** (https://roboflow.com) - Online, collaborative
     - **CVAT** (https://cvat.org) - Advanced features
   - Label at least 500-1000 images for good results
   - Include diverse conditions:
     - Different lighting (sunny, cloudy, indoor)
     - Different angles
     - Ball in different positions (near/far, moving/stationary)
     - Different backgrounds

#### Option B: Fine-tune on Your Specific Videos
1. Extract frames from your videos
2. Label basketballs in those frames
3. Add to existing dataset
4. Retrain model

### 2. **Use a Larger YOLO Model**

Current: `yolov8n.pt` (nano - smallest, fastest, least accurate)

Better options:
- `yolov8s.pt` - Small (better accuracy, still fast)
- `yolov8m.pt` - Medium (good balance) ⭐ Recommended
- `yolov8l.pt` - Large (better accuracy, slower)
- `yolov8x.pt` - Extra Large (best accuracy, slowest)

**Trade-off**: Larger models = better detection but slower inference

### 3. **Improve Training Parameters**

Use the improved training script: `train_improved.py`

Key improvements:
- More epochs (200 instead of 100)
- Better data augmentation
- Optimized learning rate
- Early stopping to prevent overfitting

### 4. **Increase Image Resolution**

For small objects like basketballs:
- Current: 640x640
- Better: 1280x1280 (detects smaller objects better)
- Trade-off: Slower training and inference

### 5. **Transfer Learning from COCO**

COCO dataset includes "sports ball" class which can help:
1. Start with COCO-pretrained model
2. Fine-tune on basketball dataset
3. Better initial weights

### 6. **Data Augmentation**

The improved script includes:
- Color augmentation (hue, saturation, brightness)
- Geometric augmentation (rotation, scaling, translation)
- Mosaic augmentation (combines multiple images)
- Mixup augmentation

### 7. **Create Dataset from Your Videos**

Quick script to extract frames:

```python
import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=30):
    """Extract frames from video for labeling"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved_count:05d}.jpg", frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

# Use it:
extract_frames("your_video.mp4", "frames_to_label", frame_interval=30)
```

## Step-by-Step: Retrain with Better Dataset

### Step 1: Prepare Dataset
1. Download or create dataset with labeled basketballs
2. Organize in YOLO format:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

### Step 2: Update config.yaml
```yaml
train: /path/to/dataset/train/images
val: /path/to/dataset/val/images
test: /path/to/dataset/test/images

nc: 2
names: ['Basketball', 'Basketball Hoop']
```

### Step 3: Train Model
```bash
python train_improved.py
```

### Step 4: Use New Model
```bash
cp runs/detect/basketball_improved/weights/best.pt best.pt
```

## Quick Wins (No Retraining Required)

### 1. Try Different YOLO Model Sizes
Edit `shot_detector_web.py`:
```python
# Try larger model
self.model = YOLO("yolov8m.pt")  # or yolov8l.pt, yolov8x.pt
```

### 2. Increase Detection Confidence Threshold
Currently set very low (0.01). Try:
```python
results = self.model(enhanced_frame, stream=True, device=self.device, conf=0.25, iou=0.45)
```

### 3. Use Larger Image Size for Detection
```python
results = self.model(enhanced_frame, stream=True, device=self.device, conf=0.01, imgsz=1280)
```

## Recommended Approach

1. **Short term**: Try larger YOLO model (yolov8m.pt or yolov8l.pt)
2. **Medium term**: Create dataset from your videos and retrain
3. **Long term**: Continuously improve dataset with more diverse examples

## Tools for Dataset Creation

1. **LabelImg** - https://github.com/HumanSignal/labelImg
   - Free, open-source
   - Supports YOLO format
   - Easy to use

2. **Roboflow** - https://roboflow.com
   - Online platform
   - Automatic augmentation
   - Easy dataset management

3. **CVAT** - https://cvat.org
   - Advanced features
   - Team collaboration
   - Video annotation

## Expected Results

- **Current**: ~10-20% ball detection rate
- **With larger model**: ~40-60% detection rate
- **With custom dataset**: ~80-95% detection rate

## Next Steps

1. Try `train_improved.py` with existing dataset
2. Extract frames from your videos
3. Label basketballs in those frames
4. Retrain with combined dataset
5. Test and iterate!

