"""
YOLO Pose Detection - Advanced Tracking Module (Backward Compatibility)

⚠️ DEPRECATED: This file is kept for backward compatibility only.
✅ NEW: Use the modular tracking package instead:
  - tracking.skeletal_biometrics
  - tracking.reid_extractor
  - tracking.track_manager

Comprehensive track ID management, entry/exit monitoring, and historical tracking

Active Features:
- Norfair: Advanced tracking with Kalman filter + motion prediction
- ReID: Re-identification with ResNet50 (appearance-based matching)
- Skeletal Biometrics: Bone length/ratio matching (clothing-independent)
- Persistent Database: Long-term person storage (JSON/SQLite)

Combined Matching: Keypoint (30%) + ReID (40%) + Skeletal (30%)
"""

# Import everything from the new modular structure
from tracking.skeletal_biometrics import (
    extract_skeletal_features,
    print_skeletal_features,
    skeletal_distance
)

from tracking.reid_extractor import (
    EmbeddingExtractor
)

from tracking.track_manager import (
    TrackManager,
    keypoint_distance,
    create_reid_distance_function,
    draw_track_lines,
    COCO_KEYPOINT_SIGMAS
)

# Export all for backward compatibility
__all__ = [
    # Skeletal Biometrics
    'extract_skeletal_features',
    'print_skeletal_features',
    'skeletal_distance',
    
    # ReID
    'EmbeddingExtractor',
    
    # Tracking
    'TrackManager',
    'keypoint_distance',
    'create_reid_distance_function',
    'draw_track_lines',
    'COCO_KEYPOINT_SIGMAS',
]
