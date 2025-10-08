"""
YOLO Pose Detection - Pose Utilities
Pose calculation, drawing and visualization functions
"""

import cv2
import numpy as np
import math


# Pose skeleton connections (COCO format - 17 points)
POSE_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # Head and face
    [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Shoulders and arms
    [5, 6], [5, 11], [6, 12], [11, 12],  # Body center
    [11, 13], [13, 15], [12, 14], [14, 16]  # Legs and feet
]


def calculate_head_pose(keypoints):
    """Calculate yaw, pitch, and roll angles from head keypoints"""
    if keypoints is None or len(keypoints.shape) != 2 or keypoints.shape[0] < 5:
        return None, None, None
    
    # Head keypoints (nose, left eye, right eye, left ear, right ear)
    nose = keypoints[0]  # [x, y, confidence]
    left_eye = keypoints[1]  # [x, y, confidence]
    right_eye = keypoints[2]  # [x, y, confidence]
    left_ear = keypoints[3]  # [x, y, confidence]
    right_ear = keypoints[4]  # [x, y, confidence]
    
    # Check if keypoints are visible enough
    confidence_threshold = 0.3
    if (nose[2] < confidence_threshold or 
        left_eye[2] < confidence_threshold or 
        right_eye[2] < confidence_threshold):
        return None, None, None
    
    # Calculate head center (average of nose and eyes)
    head_center_x = (nose[0] + left_eye[0] + right_eye[0]) / 3
    head_center_y = (nose[1] + left_eye[1] + right_eye[1]) / 3
    
    # Calculate face width and height
    face_width = abs(right_eye[0] - left_eye[0])
    
    # Only calculate if face width is reasonable
    if face_width < 20:  # Minimum face width threshold
        return None, None, None
    
    # Yaw calculation (left-right rotation)
    yaw = 0.0
    if left_ear[2] > confidence_threshold and right_ear[2] > confidence_threshold:
        ear_midpoint_x = (left_ear[0] + right_ear[0]) / 2
        yaw = ((ear_midpoint_x - nose[0]) / face_width) * 30  # Scale to degrees
    
    # Pitch calculation (up-down rotation)
    eye_midpoint_y = (left_eye[1] + right_eye[1]) / 2
    pitch = ((nose[1] - eye_midpoint_y) / face_width) * 45  # Scale to degrees
    
    # Roll calculation (head tilt)
    roll = 0.0
    if left_eye[2] > confidence_threshold and right_eye[2] > confidence_threshold:
        eye_angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        roll = math.degrees(eye_angle)
    
    # Clamp values to reasonable ranges
    yaw = max(-90, min(90, yaw))
    pitch = max(-90, min(90, pitch))
    roll = max(-90, min(90, roll))
    
    return yaw, pitch, roll, (head_center_x, head_center_y)


def draw_head_pose(frame, yaw, pitch, roll, head_center):
    """Draw head pose visualization"""
    if head_center is None:
        return frame
    
    center_x, center_y = int(head_center[0]), int(head_center[1])
    
    # Draw only direction indicators (no text)
    # Yaw indicator (horizontal) - below head center
    yaw_length = 30
    yaw_end_x = center_x + int(yaw_length * math.sin(math.radians(yaw)))
    cv2.arrowedLine(frame, (center_x, center_y + 40), (yaw_end_x, center_y + 40), 
                    (255, 0, 0), 2, tipLength=0.3)
    
    # Pitch indicator (vertical) - on the right
    pitch_length = 20
    pitch_end_y = center_y + int(pitch_length * math.sin(math.radians(pitch)))
    cv2.arrowedLine(frame, (center_x + 50, center_y), (center_x + 50, pitch_end_y), 
                    (0, 255, 0), 2, tipLength=0.3)
    
    return frame


def draw_pose_skeleton(frame, keypoints, confidence_threshold=0.3, show_numbers=True):
    """Draw pose keypoints and skeleton lines"""
    if keypoints is None or len(keypoints) == 0:
        return frame
    
    for person_keypoints in keypoints:
        # Draw keypoints
        for i, (x, y, conf) in enumerate(person_keypoints):
            if conf > confidence_threshold:
                # Point color (based on confidence score)
                color_intensity = int(255 * conf)
                cv2.circle(frame, (int(x), int(y)), 6, (0, color_intensity, 255-color_intensity), -1)
                cv2.circle(frame, (int(x), int(y)), 8, (255, 255, 255), 2)
                
                # Draw point number (small with white background)
                if show_numbers:
                    # Background (black box)
                    text_size = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(frame, 
                                (int(x) + 8, int(y) - text_size[1] - 12),
                                (int(x) + text_size[0] + 12, int(y) - 8),
                                (0, 0, 0), -1)
                    # Number
                    cv2.putText(frame, str(i), (int(x) + 10, int(y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw skeleton lines
        for connection in POSE_SKELETON:
            pt1_idx = connection[0]
            pt2_idx = connection[1]
            
            # Valid index check (YOLO11 uses 0-based indexing)
            if pt1_idx < len(person_keypoints) and pt2_idx < len(person_keypoints):
                x1, y1, conf1 = person_keypoints[pt1_idx]
                x2, y2, conf2 = person_keypoints[pt2_idx]
                
                # Draw line if both points are confident enough
                if conf1 > confidence_threshold and conf2 > confidence_threshold:
                    # Determine line color (based on confidence score)
                    avg_conf = (conf1 + conf2) / 2
                    color_intensity = int(255 * avg_conf)
                    
                    # Different colors for different connections
                    if connection in [[11, 13], [13, 15], [12, 14], [14, 16]]:  # Legs
                        color = (0, color_intensity, 255)  # Blue tones
                    elif connection in [[5, 7], [7, 9], [6, 8], [8, 10]]:  # Arms
                        color = (0, 255, color_intensity)  # Green tones
                    elif connection in [[0, 1], [0, 2], [1, 3], [2, 4]]:  # Head and face
                        color = (color_intensity, 0, 255)  # Red tones
                    elif connection in [[5, 11], [6, 12], [11, 12]]:  # Body center
                        color = (255, color_intensity, 0)  # Yellow tones
                    elif connection in [[0, 5], [0, 6], [5, 6]]:  # Shoulders
                        color = (255, 255, color_intensity)  # White/Yellow tones
                    else:  # Other connections
                        color = (color_intensity, color_intensity, 255)  # Purple tones
                    
                    # Adjust line thickness based on confidence score
                    thickness = max(2, int(avg_conf * 4))
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    return frame