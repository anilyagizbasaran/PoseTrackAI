# OpenCV YOLO Pose Detection

Real-time pose detection and tracking system with persistent person recognition.

## What Does This Do?

This system detects people, tracks their movements, and **remembers them forever**. When someone leaves the camera and comes back hours or days later, they get the same ID.

**Key Capabilities:**
- Detects 17 body keypoints per person
- Tracks multiple people simultaneously
- Recognizes people across sessions using appearance (ReID)
- Works with webcam or RTSP cameras
- Stores person database permanently

## Quick Start

### 1. Install Dependencies

```bash
pip install opencv-python ultralytics torch torchvision norfair
```

### 2. Run Detection

**Webcam:**
```bash
python pose_ultralytics.py
```

**RTSP Camera:**
```bash
# First, edit config_rtsp.yaml with your camera URL
python pose_rtsp.py
```

### 3. Keyboard Controls

- `Q` or `ESC` - Exit
- `P` - Pause/Resume
- `F` - Fullscreen
- `W` - Normal window

## Configuration

- **config_webcam.yaml** - Webcam settings (resolution, tracking, ReID)
- **config_rtsp.yaml** - RTSP camera settings

Edit these files to adjust performance, tracking sensitivity, and person recognition.

## Project Files

```
Main Scripts:
  pose_ultralytics.py    - Webcam detection
  pose_rtsp.py           - RTSP camera detection

Core Modules:
  tracking.py            - Tracking system (Norfair + ReID)
  person_database.py     - Person database (persistent storage)

Utilities:
  pose_utils.py          - Pose calculations
  ui.py                  - Screen display
  log.py                 - Logging
  camera_rtsp.py         - RTSP management
  config_manager.py      - Config loader

Configuration:
  config_webcam.yaml     - Webcam settings
  config_rtsp.yaml       - RTSP settings
```

## Output

- **Video**: `yolo11_object_pose_output.avi` or `rtsp_pose_output.avi`
- **Logs**: `logs/yolo_detection.log`
- **Database**: `person_database.json` (person embeddings)

## Documentation

For detailed information, features, and troubleshooting:
- [English Documentation](DOCUMENTATION_EN.md)
- [Turkish Documentation](DOCUMENTATION_TR.md)