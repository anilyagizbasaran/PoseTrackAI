"""
YOLO Pose Detection - User Interface Module
Information overlay and visualization functions for screen display
"""

import cv2


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
    Draw track ID above person's head
    
    Args:
        frame: Frame to process
        head_center: Head center coordinates (x, y)
        track_id: Track ID
    
    Returns:
        frame: Frame with ID added
    """
    if head_center is None or track_id is None:
        return frame
    
    track_text = f"ID:{track_id}"
    track_size = cv2.getTextSize(track_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    track_x = int(head_center[0]) - track_size[0] // 2
    track_y = int(head_center[1]) - 60
    
    # Black background
    cv2.rectangle(frame, 
                (track_x - 5, track_y - track_size[1] - 5),
                (track_x + track_size[0] + 5, track_y + 5),
                (0, 0, 0), -1)
    
    # Blue YOLO track ID
    cv2.putText(frame, track_text, 
              (track_x, track_y), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    
    return frame


def create_info_data(person_count=0, track_ids=None, track_history_count=0,
                     pose_quality=0.0, visible_keypoints=0, total_keypoints=0,
                     head_pose_data=None):
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
    }