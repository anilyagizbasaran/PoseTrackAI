# OpenCV YOLO Pose Detection - Documentation

Advanced pose detection and tracking system with persistent person recognition.

---

## Project Structure

```
PoseTrackAI/
├── pose_ultralytics.py      # Main webcam detection script 
├── pose_rtsp.py             # RTSP camera detection script 
├── pose_base.py             # Common code base 
├── tracking/                # Modular tracking system
│   ├── __init__.py          # Module exports
│   ├── skeletal_biometrics.py  # Bone structure matching
│   ├── reid_extractor.py    # ReID embeddings (ResNet50)
│   └── track_manager.py     # Main tracking logic
├── tracking.py              # Backward compatibility wrapper
├── person_database.py       # Persistent person database (JSON)
├── pose_utils.py            # Pose calculation and drawing utilities
├── ui.py                    # UI overlay and visualization
├── log.py                   # Logging system
├── camera_rtsp.py           # RTSP camera management
├── config_manager.py        # Configuration file manager
├── config_webcam.yaml       # Webcam configuration file
├── config_rtsp.yaml         # RTSP camera configuration
└── person_database.json     # Person database (auto-created)
---
## What Each File Does

### Main Scripts
- **pose_ultralytics.py** - Webcam pose detection with tracking 
- **pose_rtsp.py** - RTSP camera pose detection with tracking

### Base System
- **pose_base.py** - Common code base
  - `PoseDetectorBase` class
  - Common pose detection logic
  - Common tracking setup
  - Common UI and visualization
  - Common cleanup operations

### Core Modules
- **tracking/** - Modular tracking system
  - **skeletal_biometrics.py** - Bone length/ratio extraction (clothing-independent)
  - **reid_extractor.py** - ResNet50 appearance-based embeddings
  - **track_manager.py** - Norfair + ReID + Persistent database management
- **tracking.py** - Backward compatibility wrapper (old imports still work)
- **person_database.py** - Stores person embeddings permanently in JSON
- **pose_utils.py** - Pose keypoint calculations and skeleton drawing
- **ui.py** - On-screen information display

### Support Modules
- **camera_rtsp.py** - RTSP stream connection and management
- **config_manager.py** - Loads and manages YAML configuration
- **log.py** - Logging with timestamps and statistics

### Configuration
- **config_webcam.yaml** - Webcam configuration (camera, tracking, ReID, performance)
- **config_rtsp.yaml** - RTSP camera configuration (shares database with webcam)

---

## Quick Start

### Installation
```bash
# Install dependencies
pip install opencv-python ultralytics torch torchvision norfair

# Download YOLO model (auto-downloads on first run)
```

### Run Webcam Detection
```bash
python pose_ultralytics.py
```

### Run RTSP Detection
```bash
# Edit config_rtsp.yaml with your RTSP URL
python pose_rtsp.py
```

### Keyboard Controls
- **Q or ESC** - Exit
- **P** - Pause/Resume
- **F** - Fullscreen
- **W** - Normal window

---

## Configuration

All settings are in **config_webcam.yaml** and **config_rtsp.yaml** - edit these files to change behavior.

### Key Settings

**Camera**
```yaml
camera:
  source: 0                # 0=Webcam, "rtsp://..."=RTSP, "video.mp4"=File
  resolution: [640, 480]   # Lower=faster, Higher=better quality
```

**Tracking**
```yaml
tracking:
  use_norfair: true        # Advanced tracking (recommended)
  use_reid: true           # Person re-identification
  use_persistent_reid: true # Persistent database (recognize people forever)
```

**Performance**
```yaml
performance:
  track_every_n_frames: 1  # Process every N frames (higher=faster, less accurate)
```

See **config_webcam.yaml** and **config_rtsp.yaml** for all available parameters with detailed comments.

---

## Key Features

### 1. Persistent Person Recognition
People get the same ID even if they:
- Leave camera and return hours/days later
- Change clothes
- Change appearance

Database: `person_database.json` (auto-created and managed)

### 2. Advanced Tracking
- **Norfair**: Kalman filter + motion prediction
- **ReID**: Appearance-based re-identification
- **OKS Distance**: Pose-aware matching

### 3. RTSP Support
- Multiple cameras
- Shared person database across cameras
- Auto-reconnection on connection loss

---

## Output

### Screen Display
- Person count
- Active track IDs
- Pose quality percentage
- Head pose direction
- FPS counter

### Files
- **Video**: `yolo11_object_pose_output.avi` or `rtsp_pose_output.avi`
- **Logs**: `logs/yolo_detection.log`
- **Database**: `person_database.json` (person embeddings)

---

## Common Adjustments

### Improve Performance
```yaml
camera:
  resolution: [320, 240]   # Lower resolution
performance:
  track_every_n_frames: 3  # Skip frames
tracking:
  use_reid: false          # Disable ReID if not needed
```

### Better Person Recognition
```yaml
tracking:
  persistent_similarity_threshold: 0.85  # Stricter matching (0.65=flexible, 0.85=strict)
  reid_weight: 0.7                       # More weight on appearance
  keypoint_weight: 0.3                   # Less weight on pose
```

### RTSP Connection
```yaml
camera:
  source: "rtsp://username:password@ip:port/stream"
  buffer_size: 5           # Larger buffer for stability
```

---

## System Requirements

- **Python**: 3.8+
- **GPU**: Recommended (NVIDIA with CUDA)
- **RAM**: 4GB minimum, 8GB recommended
- **Webcam**: 720p or higher recommended

---

## Tips

1. **Start Simple**: Use default config first, adjust later
2. **Monitor FPS**: Check logs for performance
3. **Backup Database**: Export `person_database.json` regularly
4. **Test Configs**: Try different parameters for your use case
5. **GPU Usage**: Always use GPU if available (`device: "cuda"`)

---

## Advanced Features

### Persistent ID System
- Same person gets same ID even after closing and reopening the program
- Works across multiple sessions and days
- Threshold: `0.70` (optimal - recognizes same people, distinguishes different ones)

### Occlusion Support
- Tracks people even when they overlap or are partially hidden
- 10-second tracking resilience (`hit_counter_max: 300`)
- 15-second ReID active (`reid_hit_counter_max: 450`)
- Works with close proximity tracking

### Skeletal Biometrics
- Uses bone structure for identification (clothing-independent)
- More stable than appearance in occlusions
- Minimum 8 keypoints required for quality detection
- Minimum 10 measurements for high-quality skeletal matching

---

## Troubleshooting

**IDs changing when closing/opening:**
- Lower threshold: `persistent_similarity_threshold: 0.65`

**IDs flickering/unstable:**
- Already optimized with `initialization_delay: 5`

**False matches (different people same ID):**
- Increase threshold: `persistent_similarity_threshold: 0.75`
- Increase quality: `min_visible_keypoints: 10`

**People not detected far away:**
- Lower `conf_threshold: 0.15` in detection settings
- Lower `min_visible_keypoints: 5` for distant detection

---

For detailed parameter explanations, see comments in **config_webcam.yaml**