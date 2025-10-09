"""
Skeletal Biometrics Module
Bone length and ratio extraction for person identification
"""

import numpy as np
from config_manager import get_config

# Get configuration
_config = None

def _get_config():
    """Get or initialize config"""
    global _config
    if _config is None:
        _config = get_config()
    return _config


def extract_skeletal_features(keypoints, min_visible_keypoints=None):
    """
    Extract bone lengths and ratios (Skeletal Biometrics)
    
    These features are person-specific and invariant:
    - Arm lengths (shoulder-elbow, elbow-wrist)
    - Leg lengths (hip-knee, knee-ankle)
    - Torso lengths (shoulder width, torso height)
    
    COCO Keypoint Indices:
    0: nose, 1-2: eyes, 3-4: ears
    5-6: shoulders, 7-8: elbows, 9-10: wrists
    11-12: hips, 13-14: knees, 15-16: ankles
    
    Args:
        keypoints: [17, 3] array (x, y, confidence)
        min_visible_keypoints: Minimum visible keypoint count (None = read from config)
    
    Returns:
        features: [N] array - normalized bone lengths and ratios
        None: If insufficient keypoints
    """
    # Get config
    config = _get_config()
    detection_config = config.get_detection_config()
    
    # Use config values if not provided - SECURITY: 8 minimum keypoints
    if min_visible_keypoints is None:
        min_visible_keypoints = detection_config.get('min_visible_keypoints', 8)
    
    keypoint_confidence = detection_config.get('keypoint_confidence', 0.3)
    
    # Keypoint indices
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16
    
    # Minimum visible keypoint check - STRONG FILTER
    visible_count = np.sum(keypoints[:, 2] > keypoint_confidence)
    if visible_count < min_visible_keypoints:
        return None  # Insufficient keypoints - prevents bad recognition!
    
    features = []
    
    def get_distance(kp1_idx, kp2_idx):
        """Euclidean distance between two keypoints"""
        kp1 = keypoints[kp1_idx]
        kp2 = keypoints[kp2_idx]
        
        # Are both keypoints visible?
        if kp1[2] < keypoint_confidence or kp2[2] < keypoint_confidence:
            return None
        
        dist = np.sqrt((kp1[0] - kp2[0])**2 + (kp1[1] - kp2[1])**2)
        return dist
    
    # === 1. Shoulder Width (invariant!) ===
    shoulder_width = get_distance(L_SHOULDER, R_SHOULDER)
    
    # === 2. Hip Width ===
    hip_width = get_distance(L_HIP, R_HIP)
    
    # === 3. Torso Height (shoulder-hip) ===
    # Left side
    torso_left = get_distance(L_SHOULDER, L_HIP)
    # Right side
    torso_right = get_distance(R_SHOULDER, R_HIP)
    
    # === 4. Upper Arm Lengths (shoulder-elbow) ===
    upper_arm_left = get_distance(L_SHOULDER, L_ELBOW)
    upper_arm_right = get_distance(R_SHOULDER, R_ELBOW)
    
    # === 5. Forearm Lengths (elbow-wrist) ===
    forearm_left = get_distance(L_ELBOW, L_WRIST)
    forearm_right = get_distance(R_ELBOW, R_WRIST)
    
    # === 6. Upper Leg Lengths (hip-knee) ===
    upper_leg_left = get_distance(L_HIP, L_KNEE)
    upper_leg_right = get_distance(R_HIP, R_KNEE)
    
    # === 7. Lower Leg Lengths (knee-ankle) ===
    lower_leg_left = get_distance(L_KNEE, L_ANKLE)
    lower_leg_right = get_distance(R_KNEE, R_ANKLE)
    
    # === Reference for normalization: Shoulder width (or alternatives) ===
    # Find reference distance (for camera-independent scaling)
    reference_distance = None
    
    # Priority order: Shoulder > Hip > Torso
    if shoulder_width is not None and shoulder_width > 1e-6:
        reference_distance = shoulder_width
    elif hip_width is not None and hip_width > 1e-6:
        reference_distance = hip_width
    elif torso_left is not None and torso_left > 1e-6:
        reference_distance = torso_left
    elif torso_right is not None and torso_right > 1e-6:
        reference_distance = torso_right
    else:
        # If no reference, use any visible bone length
        for measurement in [upper_arm_left, upper_arm_right, forearm_left, forearm_right,
                           upper_leg_left, upper_leg_right, lower_leg_left, lower_leg_right]:
            if measurement is not None and measurement > 1e-6:
                reference_distance = measurement
                break
    
    # Still no reference found, cannot extract skeletal features
    if reference_distance is None or reference_distance < 1e-6:
        return None
    
    # Normalize all distances to reference distance
    normalized_features = []
    
    measurements = [
        hip_width, torso_left, torso_right,
        upper_arm_left, upper_arm_right,
        forearm_left, forearm_right,
        upper_leg_left, upper_leg_right,
        lower_leg_left, lower_leg_right
    ]
    
    for m in measurements:
        if m is not None:
            normalized_features.append(m / reference_distance)
        else:
            normalized_features.append(0.0)  # Invisible keypoint
    
    # === Ratios (more robust!) ===
    # Arm ratio: upper arm / forearm (characteristic!)
    if upper_arm_left and forearm_left:
        normalized_features.append(upper_arm_left / (forearm_left + 1e-6))
    else:
        normalized_features.append(0.0)
    
    if upper_arm_right and forearm_right:
        normalized_features.append(upper_arm_right / (forearm_right + 1e-6))
    else:
        normalized_features.append(0.0)
    
    # Leg ratio: upper leg / lower leg
    if upper_leg_left and lower_leg_left:
        normalized_features.append(upper_leg_left / (lower_leg_left + 1e-6))
    else:
        normalized_features.append(0.0)
    
    if upper_leg_right and lower_leg_right:
        normalized_features.append(upper_leg_right / (lower_leg_right + 1e-6))
    else:
        normalized_features.append(0.0)
    
    # Torso/Reference ratio (height characteristic)
    if torso_left:
        normalized_features.append(torso_left / reference_distance)
    else:
        normalized_features.append(0.0)
    
    # Return array
    features_array = np.array(normalized_features)
    return features_array


def print_skeletal_features(skeletal_features, person_id="Unknown"):
    """
    Print skeletal features in readable format
    
    Args:
        skeletal_features: [16] array - skeletal biometrics
        person_id: Person ID
    """
    if skeletal_features is None:
        print(f"[SKELETAL] {person_id}: Insufficient keypoints")
        return
    
    print(f"\n{'='*60}")
    print(f"[SKELETAL BIOMETRICS] Person ID: {person_id}")
    print(f"{'='*60}")
    
    # Feature names
    feature_names = [
        "Hip Width / Shoulder",
        "Torso Left (Shoulder-Hip) / Shoulder", 
        "Torso Right (Shoulder-Hip) / Shoulder",
        "Upper Arm Left (Shoulder-Elbow) / Shoulder",
        "Upper Arm Right (Shoulder-Elbow) / Shoulder",
        "Forearm Left (Elbow-Wrist) / Shoulder",
        "Forearm Right (Elbow-Wrist) / Shoulder",
        "Upper Leg Left (Hip-Knee) / Shoulder",
        "Upper Leg Right (Hip-Knee) / Shoulder",
        "Lower Leg Left (Knee-Ankle) / Shoulder",
        "Lower Leg Right (Knee-Ankle) / Shoulder",
        "RATIO: Upper/Forearm Left",
        "RATIO: Upper/Forearm Right",
        "RATIO: Upper/Lower Leg Left",
        "RATIO: Upper/Lower Leg Right",
        "RATIO: Torso/Shoulder"
    ]
    
    print(f"\n📏 Bone Lengths (normalized to shoulder width):")
    print(f"{'-'*60}")
    
    for i, (name, value) in enumerate(zip(feature_names, skeletal_features)):
        if value > 0.001:  # Only show visible ones
            # Display ratios differently
            if "RATIO:" in name:
                print(f"  {i+1:2d}. {name:40s}: {value:.3f}")
            else:
                print(f"  {i+1:2d}. {name:40s}: {value:.3f}x")
        else:
            print(f"  {i+1:2d}. {name:40s}: --- (not visible)")
    
    # Summary statistics
    visible_count = np.sum(skeletal_features > 0.001)
    visible_values = skeletal_features[skeletal_features > 0]
    
    print(f"\n📊 Summary:")
    print(f"  - Visible Measurements: {visible_count}/16")
    if len(visible_values) > 0:
        print(f"  - Average Ratio: {np.mean(visible_values):.3f}")
        print(f"  - Standard Deviation: {np.std(visible_values):.3f}")
    else:
        print(f"  - Average Ratio: --- (none)")
        print(f"  - Standard Deviation: --- (none)")
    print(f"{'='*60}\n")


def skeletal_distance(skel1, skel2):
    """
    Distance between two skeletal features
    
    Args:
        skel1, skel2: Skeletal feature vectors
    
    Returns:
        distance: Normalized distance 0-1
    """
    if skel1 is None or skel2 is None:
        return 1.0  # Maximum distance
    
    # Euclidean distance
    diff = skel1 - skel2
    
    # Only compute for non-zero (visible) features
    valid_mask = (skel1 != 0) & (skel2 != 0)
    
    if not np.any(valid_mask):
        return 1.0
    
    # Distance for valid features only
    valid_diff = diff[valid_mask]
    distance = np.sqrt(np.mean(valid_diff ** 2))
    
    # Normalize (typical distance ~0.1-0.3)
    normalized_distance = min(distance / 0.5, 1.0)
    
    return normalized_distance

