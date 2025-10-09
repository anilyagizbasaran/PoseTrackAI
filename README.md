# OpenCV YOLO Pose Detection

Real-time pose detection and tracking system with persistent person recognition.

## What Does This Do?

This system detects people, tracks their movements, and **remembers them forever**. When someone leaves the camera and comes back hours or days later, they get the same ID.

**Key Capabilities:**
- Detects 17 body keypoints per person
- Tracks multiple people simultaneously
- **Skeletal Biometrics**: Recognizes people by bone structure (independent of clothing)
- **ReID Embeddings**: Appearance-based recognition using ResNet50
- **Adaptive Matching**: Combines skeletal + appearance features intelligently
- Works with webcam or RTSP cameras
- Stores person database permanently

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/anilyagizbasaran/PoseTrackAI.git
cd PoseTrackAI
```

### 2. Install Dependencies

```bash
pip install opencv-python ultralytics torch torchvision norfair
```

### 3. Run Detection

**Webcam:**
```bash
python pose_ultralytics.py
```

**RTSP Camera:**
```bash
# First, edit config_rtsp.yaml with your camera URL
python pose_rtsp.py
```

### 4. Keyboard Controls

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
  tracking/              - Modular tracking system
    skeletal_biometrics  - Bone structure matching
    reid_extractor       - ReID embeddings (ResNet50)
    track_manager        - Main tracking logic
  person_database.py     - Person database (JSON storage)

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

## Known Issues

⚠️ **Person ID & Recognition Issues:**

The current implementation has some challenges with person identification and tracking:

- **Embedding Update Rate**: The exponential moving average (alpha=0.9) is too aggressive, causing person embeddings to drift over time and potentially lose original identity
- **Skeletal Weight Tuning**: Adaptive weighting between skeletal biometrics and ReID needs optimization for different scenarios
- **ID Persistence**: Person IDs may not remain stable across long sessions or when people change appearance (clothing, accessories)
- **Threshold Sensitivity**: Similarity thresholds (both skeletal and embedding) need fine-tuning for different environments and lighting conditions
- **Performance**: Auto-save on every update can cause slowdowns during high-frequency tracking
- **Database Management**: Missing export/import features and statistics tools make it difficult to manage the person database

**We welcome contributions to improve these issues!** See the Contributing section below.

## Contributing

We'd love your help improving PoseTrackAI! Here are areas where contributions would be especially valuable:

### Priority Issues:
- **Person Re-identification**: Improving the stability and accuracy of person recognition across sessions
- **Parameter Tuning**: Finding optimal values for embedding alpha, skeletal weights, and similarity thresholds
- **Database Tools**: Adding export/import functionality and management utilities
- **Performance**: Implementing batch saving or periodic database updates
- **Testing**: Creating test cases for different scenarios and edge cases

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes and test thoroughly
4. Commit with clear messages (`git commit -m 'Fix: Improve person ID stability'`)
5. Push to your branch (`git push origin feature/improvement`)
6. Open a Pull Request

### Guidelines:
- Document your changes clearly
- Include comments in Turkish or English
- Test with both webcam and RTSP sources
- Update configuration files if adding new parameters

**Questions or ideas?** Open an issue on [GitHub](https://github.com/anilyagizbasaran/PoseTrackAI/issues)