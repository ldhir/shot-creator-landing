# VideoPose3D Setup Instructions

This integration uses the **actual VideoPose3D repository** from Facebook Research:
https://github.com/facebookresearch/VideoPose3D

## Installation

1. **Clone the VideoPose3D repository:**
   ```bash
   cd /Users/namrata/Desktop/shot-creator-landing-copy/tool
   git clone https://github.com/facebookresearch/VideoPose3D.git
   ```

2. **Install dependencies:**
   ```bash
   cd VideoPose3D
   pip install torch torchvision
   pip install matplotlib  # For visualization
   ```

3. **Download pretrained models:**
   ```bash
   mkdir -p checkpoint
   cd checkpoint
   wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin
   cd ..
   ```

4. **The integration will automatically use the cloned repository**

## Usage

The VideoPose3D endpoint (`/api/process_frame_overlay_videopose3d`) will:
- Use MediaPipe to extract 2D keypoints (as VideoPose3D expects 2D input)
- Feed 2D keypoint sequences to the actual VideoPose3D model
- Return 3D poses using VideoPose3D's temporal convolution approach

## Note

VideoPose3D requires sequences of frames (243 frames = ~8 seconds at 30fps).
The system will buffer frames until it has enough for VideoPose3D processing.
