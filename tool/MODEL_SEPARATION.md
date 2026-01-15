# Model Separation - Each Uses Its Own GitHub Repository

This implementation ensures each pose estimation model uses **ONLY** its own GitHub repository code, with no mixing or interchanging.

## Model Implementations

### 1. VideoPose3D (Facebook Research)
- **Repository**: https://github.com/facebookresearch/VideoPose3D
- **Location**: `tool/VideoPose3D/` (clone the repo here)
- **Endpoint**: `/api/process_frame_overlay_videopose3d`
- **Uses**: Their actual `TemporalModel` class and inference code
- **Setup**: See `VIDEOPOSE3D_SETUP.md`

### 2. NTU-RRIS Google-MediaPipe
- **Repository**: https://github.com/ntu-rris/google-mediapipe
- **Location**: `tool/google-mediapipe/` (clone the repo here)
- **Endpoint**: `/api/process_frame_overlay_ntu`
- **Uses**: Their actual code from `code/08_skeleton_3D.py` and related modules
- **Setup**: See `NTU_RRIS_SETUP.md`

### 3. Original Shot Sync
- **Repository**: This codebase
- **Endpoint**: `/api/process_frame_overlay`
- **Uses**: Original MediaPipe implementation from this project

## Key Principles

1. **No Code Mixing**: Each model's endpoint uses ONLY its own repository code
2. **Independent Paths**: Each repo is in its own directory
3. **Separate Imports**: Each model imports from its own repository
4. **No Shared Logic**: Models don't share processing logic - each is self-contained

## How It Works

- **VideoPose3D**: Imports `from common.model import TemporalModel` from VideoPose3D repo
- **NTU-RRIS**: Imports their actual code modules from `google-mediapipe/code/`
- **Original**: Uses MediaPipe directly without any external repos

Each model is completely independent and uses its own GitHub repository's code exclusively.
