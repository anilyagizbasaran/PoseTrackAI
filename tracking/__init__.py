"""
Tracking Module - Modular Structure
Advanced pose tracking with Norfair + ReID + Skeletal Biometrics

Modules:
- skeletal_biometrics: Bone length/ratio extraction
- reid_extractor: Appearance-based embedding extraction
- track_manager: Main tracking logic with Norfair
"""

# Import all public functions and classes
from .skeletal_biometrics import (
    extract_skeletal_features,
    print_skeletal_features,
    skeletal_distance
)

from .reid_extractor import (
    EmbeddingExtractor
)

from .track_manager import (
    TrackManager,
    keypoint_distance,
    create_reid_distance_function,
    draw_track_lines,
    COCO_KEYPOINT_SIGMAS
)

# Export all
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

