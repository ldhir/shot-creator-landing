"""
Improved YOLO Training Script for Basketball Detection
Includes better hyperparameters, data augmentation, and training options
"""

from ultralytics import YOLO
from utils import get_device
import os

if __name__ == "__main__":
    print("="*60)
    print("Improved Basketball Detection Model Training")
    print("="*60)
    
    # Select device for training
    device = get_device()
    print(f"Using device: {device}")
    
    # Model options - try larger models for better accuracy
    # Options: yolov8n.pt (nano - fastest), yolov8s.pt (small), 
    #          yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (extra large - best accuracy)
    MODEL_SIZE = 'yolov8m.pt'  # Medium model - good balance of speed and accuracy
    
    # Check if we should use existing best.pt as starting point
    USE_EXISTING = False
    if os.path.exists('best.pt') and USE_EXISTING:
        PRE_TRAINED_MODEL = 'best.pt'
        print("Using existing best.pt as starting point for fine-tuning")
    else:
        PRE_TRAINED_MODEL = MODEL_SIZE
        print(f"Using {MODEL_SIZE} as base model")
    
    # Load the model
    model = YOLO(PRE_TRAINED_MODEL)
    
    # Improved training parameters
    results = model.train(
        data='config.yaml',
        epochs=200,  # More epochs for better learning
        imgsz=640,   # Image size (can try 1280 for better small object detection)
        device=device,
        batch=16,    # Batch size (reduce if out of memory)
        
        # Data augmentation for better generalization
        hsv_h=0.015,      # Hue augmentation
        hsv_s=0.7,        # Saturation augmentation
        hsv_v=0.4,        # Value augmentation
        degrees=10,       # Rotation augmentation
        translate=0.1,    # Translation augmentation
        scale=0.5,        # Scale augmentation
        shear=2,          # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,       # Vertical flip (usually 0 for basketball)
        fliplr=0.5,       # Horizontal flip
        mosaic=1.0,       # Mosaic augmentation (1.0 = always on)
        mixup=0.1,        # Mixup augmentation probability
        
        # Learning rate and optimization
        lr0=0.01,         # Initial learning rate
        lrf=0.1,          # Final learning rate (lr0 * lrf)
        momentum=0.937,   # SGD momentum
        weight_decay=0.0005,  # Weight decay
        warmup_epochs=3,  # Warmup epochs
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Training settings
        patience=50,      # Early stopping patience
        save=True,        # Save checkpoints
        save_period=10,   # Save checkpoint every N epochs
        val=True,         # Validate during training
        plots=True,       # Generate training plots
        
        # Advanced options
        optimizer='AdamW',  # Optimizer: SGD, Adam, AdamW, NAdam, RAdam, RMSProp
        verbose=True,
        seed=42,          # Random seed for reproducibility
        deterministic=True,
        
        # Project settings
        project='runs/detect',
        name='basketball_improved',
        exist_ok=True,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: runs/detect/basketball_improved/weights/best.pt")
    print(f"Copy this file to 'best.pt' in the project root to use it.")
    print("="*60)

