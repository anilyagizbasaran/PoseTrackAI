# 📦 Tracking Module - Modular Structure

Advanced pose tracking with **Norfair + ReID + Skeletal Biometrics**

## 📂 Module Structure

```
tracking/
├── __init__.py                  # Module exports
├── skeletal_biometrics.py       # Bone length/ratio extraction (288 lines)
├── reid_extractor.py            # ReID embedding with ResNet50 (115 lines)
├── track_manager.py             # Main tracking logic (572 lines)
└── README.md                    # This file
```

**Total**: ~975 lines (previously 937 lines in single file)

## 🎯 Modules

### 1️⃣ `skeletal_biometrics.py`
**Skeletal feature extraction - clothing-independent identification**

```python
from tracking.skeletal_biometrics import (
    extract_skeletal_features,
    print_skeletal_features,
    skeletal_distance
)

# Extract bone measurements
features = extract_skeletal_features(keypoints)  # [16] array
# Calculate similarity
distance = skeletal_distance(features1, features2)  # 0.0-1.0
```

**Features**:
- ✅ 11 bone length measurements (normalized to shoulder width)
- ✅ 5 bone ratios (upper/lower arm, leg, torso)
- ✅ Invariant to camera distance and clothing
- ✅ Config-based filtering (min 8 visible keypoints)

---

### 2️⃣ `reid_extractor.py`
**Appearance-based re-identification with ResNet50**

```python
from tracking.reid_extractor import EmbeddingExtractor

# Initialize extractor
extractor = EmbeddingExtractor(device="cuda")

# Extract embedding from person crop
embedding = extractor.extract_embedding(image_crop)  # [2048] vector
```

**Features**:
- ✅ ResNet50 backbone (pretrained on ImageNet)
- ✅ 2048-dimensional embeddings
- ✅ L2 normalized for cosine similarity
- ✅ GPU/CPU support with automatic detection

---

### 3️⃣ `track_manager.py`
**Main tracking logic with Norfair integration**

```python
from tracking.track_manager import TrackManager

# Initialize tracker
tracker = TrackManager(
    use_norfair=True,
    use_reid=True,
    use_skeletal=True,
    use_persistent_reid=True,
    keypoint_weight=0.3,
    reid_weight=0.4,
    skeletal_weight=0.3
)

# Update tracks
track_ids, finished = tracker.update_tracks_with_norfair(
    pose_keypoints, boxes, frame
)
```

**Features**:
- ✅ Norfair Kalman filter tracking
- ✅ Combined distance: Keypoint + ReID + Skeletal
- ✅ Persistent database integration
- ✅ Track entry/exit logging
- ✅ Position history management

---

## 🔄 Backward Compatibility

The old `tracking.py` file is now a **wrapper** that imports from the new modules:

```python
# Old import (still works!)
from tracking import TrackManager, extract_skeletal_features

# New import (recommended)
from tracking.track_manager import TrackManager
from tracking.skeletal_biometrics import extract_skeletal_features
```

**No code changes needed** - all existing imports continue to work!

---

## 📊 Comparison

| Before | After |
|--------|-------|
| ❌ 1 file, 937 lines | ✅ 4 files, ~975 lines |
| ❌ Hard to navigate | ✅ Easy to find functions |
| ❌ Mixed responsibilities | ✅ Clear separation |
| ❌ Long scroll | ✅ Small, focused files |

---

## 🚀 Usage Example

```python
# Full tracking pipeline
from tracking import (
    TrackManager,
    extract_skeletal_features,
    EmbeddingExtractor
)

# Initialize
tracker = TrackManager(
    use_norfair=True,
    use_reid=True,
    use_skeletal=True,
    use_persistent_reid=True
)

# Process frame
track_ids, finished = tracker.update_tracks_with_norfair(
    pose_keypoints=keypoints,
    boxes=boxes,
    frame=frame
)

# Get skeletal features for display
for kp in keypoints:
    skel = extract_skeletal_features(kp)
    if skel is not None:
        print(f"Measurements: {np.sum(skel > 0.001)}/16")
```

---

## 🎯 Benefits

1. **Modularity**: Each file has a single responsibility
2. **Maintainability**: Easy to find and update code
3. **Testability**: Can test modules independently
4. **Readability**: Smaller files, clearer structure
5. **Backward Compatibility**: Existing code works without changes

---

## 📈 Module Sizes

| Module | Lines | Purpose |
|--------|-------|---------|
| `skeletal_biometrics.py` | 288 | Bone measurements |
| `reid_extractor.py` | 115 | ReID embeddings |
| `track_manager.py` | 572 | Tracking logic |
| `__init__.py` | 40 | Exports |

**Total**: 1015 lines (vs. 937 in old single file)

*Note: Slight increase due to module headers and improved documentation*

