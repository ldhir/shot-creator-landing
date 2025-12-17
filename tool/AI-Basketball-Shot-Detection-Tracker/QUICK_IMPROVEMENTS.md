# Quick Improvements (No Retraining Required)

## 1. Try Larger YOLO Model (Fastest Improvement)

The current model uses `yolov8n.pt` (nano - smallest). Try a larger model:

### Option A: Download and Use Larger Model

```bash
# Download medium model (recommended)
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Or download large model (better accuracy, slower)
python -c "from ultralytics import YOLO; YOLO('yolov8l.pt')"
```

### Option B: Modify Code to Use Larger Model

Edit `shot_detector_web.py` line 32:

```python
# Change from:
self.model = YOLO("best.pt")

# To (try one of these):
self.model = YOLO("yolov8m.pt")  # Medium - good balance
# OR
self.model = YOLO("yolov8l.pt")  # Large - better accuracy
```

**Note**: Larger models will be slower but should detect basketballs much better.

## 2. Increase Detection Image Size

Edit `shot_detector_web.py` line 180:

```python
# Change from:
results = self.model(enhanced_frame, stream=True, device=self.device, conf=0.01, iou=0.45, imgsz=640)

# To:
results = self.model(enhanced_frame, stream=True, device=self.device, conf=0.01, iou=0.45, imgsz=1280)
```

This helps detect smaller objects (like basketballs) better.

## 3. Adjust Confidence Threshold

Currently set very low (0.01). Try:

```python
# For better quality detections (fewer false positives):
results = self.model(enhanced_frame, stream=True, device=self.device, conf=0.15, iou=0.45, imgsz=640)
```

## Recommended Quick Test

1. Try `yolov8m.pt` model first
2. If still poor, try `yolov8l.pt`
3. Increase image size to 1280 if needed

These changes take 2 minutes and can significantly improve detection!

