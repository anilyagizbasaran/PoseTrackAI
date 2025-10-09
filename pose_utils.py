"""
YOLO Pose Detection - Pose Utilities
Pose calculation, drawing and visualization functions
"""

import cv2
import numpy as np
import math
from config_manager import get_config

# Get configuration
_config = None

def _get_config():
    """Get or initialize config"""
    global _config
    if _config is None:
        _config = get_config()
    return _config

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
    
    # Get config
    config = _get_config()
    pose_config = config.get_pose_measurement_config()
    detection_config = config.get_detection_config()
    
    # Head keypoints (nose, left eye, right eye, left ear, right ear)
    nose = keypoints[0]  # [x, y, confidence]
    left_eye = keypoints[1]  # [x, y, confidence]
    right_eye = keypoints[2]  # [x, y, confidence]
    left_ear = keypoints[3]  # [x, y, confidence]
    right_ear = keypoints[4]  # [x, y, confidence]
    
    # Check if keypoints are visible enough
    confidence_threshold = detection_config.get('keypoint_confidence', 0.3)
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
    min_face_width = pose_config.get('min_face_width_pixels', 20)
    if face_width < min_face_width:
        return None, None, None
    
    # Yaw calculation (left-right rotation)
    yaw = 0.0
    yaw_scale = pose_config.get('yaw_scale_degrees', 30)
    if left_ear[2] > confidence_threshold and right_ear[2] > confidence_threshold:
        ear_midpoint_x = (left_ear[0] + right_ear[0]) / 2
        yaw = ((ear_midpoint_x - nose[0]) / face_width) * yaw_scale
    
    # Pitch calculation (up-down rotation)
    pitch_scale = pose_config.get('pitch_scale_degrees', 45)
    eye_midpoint_y = (left_eye[1] + right_eye[1]) / 2
    pitch = ((nose[1] - eye_midpoint_y) / face_width) * pitch_scale
    
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


def draw_pose_skeleton(frame, keypoints, confidence_threshold=None, show_numbers=None):
    """Draw pose keypoints and skeleton lines"""
    if keypoints is None or len(keypoints) == 0:
        return frame
    
    # Get config
    config = _get_config()
    detection_config = config.get_detection_config()
    
    # Use config values if not provided
    if confidence_threshold is None:
        confidence_threshold = detection_config.get('keypoint_confidence', 0.3)
    if show_numbers is None:
        show_numbers = detection_config.get('show_keypoint_numbers', True)
    
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


def draw_shoulder_measurement(frame, keypoints, body_height_cm=None):
    """
    Omuzlar arası mesafeyi CM cinsinden göster ve GERÇEK ÖLÇÜM yap (QuickPose tarzı)
    
    Args:
        frame: Frame to draw on
        keypoints: [17, 3] array (x, y, confidence)
        body_height_cm: Kullanıcının boy uzunluğu (opsiyonel, yoksa config'den alınır)
    
    Returns:
        frame: Frame with shoulder measurement drawn
    """
    # Get config
    config = _get_config()
    pose_config = config.get_pose_measurement_config()
    detection_config = config.get_detection_config()
    
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    L_ANKLE, R_ANKLE = 15, 16
    
    # Omuz keypoint'leri
    left_shoulder = keypoints[L_SHOULDER]
    right_shoulder = keypoints[R_SHOULDER]
    
    # Her iki omuz da görünür mü?
    confidence_threshold = detection_config.get('keypoint_confidence', 0.3)
    if left_shoulder[2] < confidence_threshold or right_shoulder[2] < confidence_threshold:
        return frame  # Omuzlar görünmüyor
    
    # Omuzlar arası pixel mesafesi
    shoulder_dist_px = np.sqrt(
        (left_shoulder[0] - right_shoulder[0])**2 + 
        (left_shoulder[1] - right_shoulder[1])**2
    )
    
    # === GERÇEK ÖLÇÜM: Vücut yüksekliğinden scale factor hesapla ===
    
    # Vücut yüksekliğini pixel cinsinden hesapla (omuz-ayak arası)
    body_height_px = None
    
    # Her iki taraf da görünürse ortalamasını al (en doğrusu)
    if (left_shoulder[2] > confidence_threshold and keypoints[L_ANKLE][2] > confidence_threshold and 
        right_shoulder[2] > confidence_threshold and keypoints[R_ANKLE][2] > confidence_threshold):
        left_height = abs(left_shoulder[1] - keypoints[L_ANKLE][1])
        right_height = abs(right_shoulder[1] - keypoints[R_ANKLE][1])
        body_height_px = (left_height + right_height) / 2
    # Sadece sol taraf görünürse
    elif left_shoulder[2] > confidence_threshold and keypoints[L_ANKLE][2] > confidence_threshold:
        body_height_px = abs(left_shoulder[1] - keypoints[L_ANKLE][1])
    # Sadece sağ taraf görünürse
    elif right_shoulder[2] > confidence_threshold and keypoints[R_ANKLE][2] > confidence_threshold:
        body_height_px = abs(right_shoulder[1] - keypoints[R_ANKLE][1])
    
    # Scale factor hesapla
    measurement_accurate = False  # Ölçüm ne kadar doğru?
    
    if body_height_px is not None and body_height_px > 10:  # Tam vücut görünür
        # Kullanıcı boyu (yoksa config'den al)
        if body_height_cm is None:
            body_height_cm = pose_config.get('default_body_height_cm', 170)
        
        # Pixel to CM dönüşüm faktörü
        scale_factor = body_height_cm / body_height_px
        
        # GERÇEK omuz mesafesi
        shoulder_dist_cm = shoulder_dist_px * scale_factor
        measurement_accurate = True
        
        # DEBUG: İlk hesaplamada yazdır (her 30 frame'de bir)
        # print(f"[MEASURE] ✅ GERÇEK ÖLÇÜM: Body: {body_height_px:.0f}px, Scale: {scale_factor:.3f}, Shoulder: {shoulder_dist_cm:.1f}cm")
    else:
        # AYAKLAR GÖRÜNMÜYOR AMA TAHMİNİ ÖLÇÜM YAP
        # Varsayılan vücut yüksekliği kullanarak yaklaşık hesapla
        
        # TAHMİNİ scale factor (ortalama mesafe: 2m, ortalama boy: 170cm)
        # Tipik omuz pixel değeri: 80-120 px (üst vücut için)
        estimated_scale = 170 / 350  # Ortalama: 170cm / 350px = 0.486 cm/px
        
        shoulder_dist_cm = shoulder_dist_px * estimated_scale
        measurement_accurate = False
        
        # DEBUG: Neden tahmini kullanıldığını göster (sadece bir kez)
        # print(f"[MEASURE] ⚠️ TAHMİNİ ÖLÇÜM: Ayaklar görünmüyor, tahmini: {shoulder_dist_cm:.1f}cm (~±5cm hata payı)")
    
    # Çizgi çiz (omuzlar arası)
    p1 = (int(left_shoulder[0]), int(left_shoulder[1]))
    p2 = (int(right_shoulder[0]), int(right_shoulder[1]))
    
    # Omuz çizgisi (kalın, parlak)
    cv2.line(frame, p1, p2, (0, 255, 255), 4)  # Sarı çizgi
    
    # Çizginin orta noktası
    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)
    
    # Mesafe metni (accurate mi yoksa tahmini mi?)
    if measurement_accurate:
        text = f"{shoulder_dist_cm:.1f} cm"  # Gerçek ölçüm ✅
        debug_text = f"(H:{body_height_px:.0f}px)"
    else:
        text = f"~{shoulder_dist_cm:.1f} cm"  # Tahmini ölçüm (~)
        debug_text = "(estimate)"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Metin boyutunu al
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Metin arkaplanı (siyah kutu)
    bg_x1 = mid_x - text_width // 2 - 8
    bg_y1 = mid_y - text_height - 15
    bg_x2 = mid_x + text_width // 2 + 8
    bg_y2 = mid_y - 5
    
    # Arka plan çiz
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 255), 2)
    
    # Metin çiz (çizginin üstünde)
    text_x = mid_x - text_width // 2
    text_y = mid_y - 10
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
    
    # Debug text (altında, küçük)
    debug_size = cv2.getTextSize(debug_text, font, 0.4, 1)[0]
    debug_x = mid_x - debug_size[0] // 2
    debug_y = mid_y + 15
    cv2.putText(frame, debug_text, (debug_x, debug_y), font, 0.4, (150, 150, 150), 1)
    
    # Uç noktalarda daireler
    cv2.circle(frame, p1, 6, (0, 255, 255), -1)
    cv2.circle(frame, p2, 6, (0, 255, 255), -1)
    
    return frame