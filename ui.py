"""
YOLO Pose Detection - User Interface Module
Information overlay and visualization functions for screen display
"""

import cv2
import numpy as np
from collections import deque


def draw_info_overlay(frame, info_data):
    """
    Draw information overlay on frame
    
    Args:
        frame: Frame to process
        info_data: Information dictionary {
            'person_count': int,
            'track_ids': list,
            'track_history_count': int,
            'pose_quality': float,
            'visible_keypoints': int,
            'total_keypoints': int,
            'head_pose_data': list,
        }
    
    Returns:
        frame: Frame with information added
    """
    person_count = info_data.get('person_count', 0)
    track_ids = info_data.get('track_ids', [])
    track_history_count = info_data.get('track_history_count', 0)
    pose_quality = info_data.get('pose_quality', 0.0)
    visible_keypoints = info_data.get('visible_keypoints', 0)
    total_keypoints = info_data.get('total_keypoints', 0)
    head_pose_data = info_data.get('head_pose_data', [])
    
    # Starting position
    start_y = 25
    current_y = start_y
    
    # Person count
    cv2.putText(frame, f"Person Count: {person_count}", 
               (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    current_y += 22
    
    # Active tracks
    if track_ids and any(tid is not None for tid in track_ids):
        valid_track_ids = [tid for tid in track_ids if tid is not None]
        ids_text = ", ".join([f"ID:{tid}" for tid in valid_track_ids])
        cv2.putText(frame, f"Active Tracks: {ids_text}", 
                   (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
        current_y += 24
    
    # Track history
    if track_history_count > 0:
        cv2.putText(frame, f"Track History: {track_history_count} objects", 
                   (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        current_y += 20
    
    # Pose quality
    cv2.putText(frame, f"Pose Quality: {pose_quality:.1f}%", 
               (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    current_y += 22
    
    # Detected points
    cv2.putText(frame, f"Detected Points: {visible_keypoints}/{total_keypoints}", 
               (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    current_y += 25
    
    # Head pose summary
    if head_pose_data:
        cv2.putText(frame, f"Head Pose: {len(head_pose_data)} tracked", 
                   (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        current_y += 20
        
        # Individual head pose data
        for i, data in enumerate(head_pose_data[:2]):  # Max 2 people
            yaw_status = "Left" if data['yaw'] < -10 else "Right" if data['yaw'] > 10 else "Forward"
            pitch_status = "Down" if data['pitch'] < -5 else "Up" if data['pitch'] > 5 else "Straight"
            
            track_info = f"ID:{data['track_id']}" if data.get('track_id') is not None else f"#{data['person']}"
            cv2.putText(frame, f"  {track_info}: {yaw_status}, {pitch_status}", 
                       (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            current_y += 18
    
    return frame


def draw_no_person_message(frame):
    """
    Display message when no person is detected
    
    Args:
        frame: Frame to process
    
    Returns:
        frame: Frame with message added
    """
    cv2.putText(frame, "No Person Detected", 
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


def draw_track_id_on_head(frame, head_center, track_id):
    """
    Draw track ID above person's head - BÜYÜK ve GÖRÜNÜLEBİLİR
    
    Args:
        frame: Frame to process
        head_center: Head center coordinates (x, y)
        track_id: Track ID
    
    Returns:
        frame: Frame with ID added
    """
    if head_center is None or track_id is None:
        return frame
    
    # ID metnini hazırla
    track_text = f"ID: {track_id}"
    
    # BÜYÜK FONT - Daha görünür (1.5 → 1.2'den büyük)
    font_scale = 1.5
    font_thickness = 3
    font = cv2.FONT_HERSHEY_DUPLEX  # Daha kalın font
    
    # Metin boyutunu hesapla
    track_size = cv2.getTextSize(track_text, font, font_scale, font_thickness)[0]
    track_x = int(head_center[0]) - track_size[0] // 2
    track_y = int(head_center[1]) - 80  # Daha yukarıda
    
    # Arka plan kutucuğu - Siyah + Beyaz border
    padding = 10
    
    # Beyaz border (dış çerçeve)
    cv2.rectangle(frame, 
                (track_x - padding - 2, track_y - track_size[1] - padding - 2),
                (track_x + track_size[0] + padding + 2, track_y + padding + 2),
                (255, 255, 255), -1)
    
    # Siyah arka plan (iç kutu)
    cv2.rectangle(frame, 
                (track_x - padding, track_y - track_size[1] - padding),
                (track_x + track_size[0] + padding, track_y + padding),
                (0, 0, 0), -1)
    
    # ID yazısı - PARLAK YEŞİL (person_XXXX) veya PARLAK MAVİ (temp_XXXX)
    if isinstance(track_id, str) and track_id.startswith("temp_"):
        id_color = (255, 150, 0)  # Turuncu - Geçici ID
    elif isinstance(track_id, str) and track_id.startswith("person_"):
        id_color = (0, 255, 0)  # Parlak yeşil - Persistent ID
    else:
        id_color = (0, 200, 255)  # Sarı - YOLO ID
    
    cv2.putText(frame, track_text, 
              (track_x, track_y), 
              font, font_scale, id_color, font_thickness)
    
    return frame


# Global: Skeletal features'ları smooth etmek için (titremeleri azalt)
_skeletal_smoothing = {}  # person_id: deque of skeletal_features

def draw_skeletal_info(frame, skeletal_data_list):
    """
    Ekranda skeletal biometrics bilgilerini göster (smoothed)
    
    Args:
        frame: Frame to process
        skeletal_data_list: Liste of skeletal data [{
            'person_id': str,
            'skeletal_features': np.ndarray,
            'visible_count': int
        }]
    
    Returns:
        frame: Frame with skeletal info
    """
    if not skeletal_data_list:
        return frame
    
    global _skeletal_smoothing
    
    # Sağ üst köşe
    frame_h, frame_w = frame.shape[:2]
    start_x = frame_w - 280
    start_y = 25
    
    # Arka plan (siyah, yarı saydam)
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x - 10, start_y - 5), 
                  (frame_w - 5, start_y + 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Başlık
    cv2.putText(frame, "SKELETAL BIOMETRICS", 
               (start_x, start_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (0, 255, 255), 1)
    
    current_y = start_y + 35
    
    # Her kişi için
    for data in skeletal_data_list[:2]:  # Max 2 kişi göster
        person_id = data.get('person_id', 'Unknown')
        skeletal = data.get('skeletal_features')
        visible = data.get('visible_count', 0)
        
        if skeletal is None:
            continue
        
        # SMOOTHING: Son 10 frame'in ortalamasını al (titremeleri azalt)
        if person_id not in _skeletal_smoothing:
            _skeletal_smoothing[person_id] = deque(maxlen=10)  # Son 10 frame
        
        _skeletal_smoothing[person_id].append(skeletal)
        
        # Smoothed değerleri hesapla (son 10 frame'in ortalaması)
        if len(_skeletal_smoothing[person_id]) > 0:
            smoothed = np.mean(list(_skeletal_smoothing[person_id]), axis=0)
        else:
            smoothed = skeletal
        
        # Kişi ID
        cv2.putText(frame, f"ID: {person_id}", 
                   (start_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.45, (255, 200, 0), 1)
        current_y += 18
        
        # Görünür ölçüm sayısı
        cv2.putText(frame, f"Measurements: {visible}/16", 
                   (start_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (200, 200, 200), 1)
        current_y += 16
        
        # En önemli 2 özellik göster (SMOOTHED değerler!)
        if smoothed[0] > 0.001:  # Kalça/Omuz
            cv2.putText(frame, f"Hip/Shoulder: {smoothed[0]:.3f}x", 
                       (start_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.35, (100, 255, 100), 1)
            current_y += 14
        
        if smoothed[11] > 0.001:  # Kol oranı
            cv2.putText(frame, f"Arm Ratio: {smoothed[11]:.3f}", 
                       (start_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.35, (100, 255, 100), 1)
            current_y += 14
        
        # Stabilite göstergesi
        if len(_skeletal_smoothing[person_id]) >= 5:
            cv2.putText(frame, "[STABLE]", 
                       (start_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.3, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "[CALIBRATING...]", 
                       (start_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.3, (0, 165, 255), 1)
        current_y += 18
    
    return frame


def create_info_data(person_count=0, track_ids=None, track_history_count=0,
                     pose_quality=0.0, visible_keypoints=0, total_keypoints=0,
                     head_pose_data=None, skeletal_data=None):
    """
    Create information dictionary for UI (helper function)
    
    Returns:
        dict: UI information
    """
    return {
        'person_count': person_count,
        'track_ids': track_ids or [],
        'track_history_count': track_history_count,
        'pose_quality': pose_quality,
        'visible_keypoints': visible_keypoints,
        'total_keypoints': total_keypoints,
        'head_pose_data': head_pose_data or [],
        'skeletal_data': skeletal_data or [],
    }