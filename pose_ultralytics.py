import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
from log import (log_with_timestamp, log_frame_info, log_detection_results, 
                 log_system_start, log_system_stats)
from pose_utils import calculate_head_pose, draw_head_pose, draw_pose_skeleton
from tracking import TrackManager
from ui import draw_info_overlay, draw_no_person_message, draw_track_id_on_head, create_info_data
from config_manager import get_config

# Global counters for tracking
total_person_count = 0  # Total count that never decreases

def handle_keyboard_input(window_name):
    """Handle keyboard input - prevent repeated code"""
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27:  # 'q' or ESC key
        log_with_timestamp("User issued exit command...", "EXIT")
        return 'quit'
    elif key == ord('p'):
        log_with_timestamp("Video paused - press any key to continue...", "PAUSE")
        cv2.waitKey(0)
        log_with_timestamp("Video resuming...", "RESUME")
    elif key == ord('f'):  # 'f' key for fullscreen toggle
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key == ord('w'):  # 'w' key for normal window mode
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    return None


def main():
    """Ultralytics Pose Detection with Tracking"""
    print("YOLO11 Pose Detection with Tracking starting...")
    print("Pose Detection with Track History")
    print("Controls:")
    print("  'q' or ESC = Exit")
    print("  'p' = Pause")
    print("  'f' = Fullscreen")
    print("  'w' = Normal Window")
    
    # Load configuration
    config = get_config()
    
    # Get device from config
    device = config.get_device()
    print(f"Device: {device}")
    
    # Track Manager - Norfair + ReID + Persistent Database
    # Persistent ReID: People recognized FOREVER! (Same ID even days later)
    USE_NORFAIR = config.get('tracking.use_norfair', True)
    
    if USE_NORFAIR:
        try:
            # Get tracking parameters from config
            tracking_config = config.get_tracking_config()
            
            track_manager = TrackManager(
                max_history_length=tracking_config.get('max_history_length', 30),
                use_norfair=tracking_config.get('use_norfair', True),
                distance_function=tracking_config.get('distance_function', 'keypoint'),
                distance_threshold=tracking_config.get('distance_threshold', 0.8),
                hit_counter_max=tracking_config.get('hit_counter_max', 60),
                initialization_delay=tracking_config.get('initialization_delay', 2),
                pointwise_hit_counter_max=tracking_config.get('pointwise_hit_counter_max', 4),
                # ReID Parameters
                use_reid=tracking_config.get('use_reid', True),
                reid_distance_threshold=tracking_config.get('reid_distance_threshold', 0.3),
                reid_hit_counter_max=tracking_config.get('reid_hit_counter_max', 150),
                keypoint_weight=tracking_config.get('keypoint_weight', 0.6),
                reid_weight=tracking_config.get('reid_weight', 0.4),
                # Persistent Database
                use_persistent_reid=tracking_config.get('use_persistent_reid', True),
                persistent_db_path=tracking_config.get('persistent_db_path', 'person_database.json'),
                persistent_db_type=tracking_config.get('persistent_db_type', 'json'),
                persistent_similarity_threshold=tracking_config.get('persistent_similarity_threshold', 0.85)
            )
            print("NORFAIR + ReID + DATABASE ACTIVE!")
        except Exception as e:
            print(f"WARNING: Failed to initialize Norfair: {e}")
            print("WARNING: Switching to YOLO ByteTrack...")
            USE_NORFAIR = False
    
    if not USE_NORFAIR:
        track_manager = TrackManager(
            max_history_length=30,
            use_norfair=False,
            use_reid=False,
            use_persistent_reid=False
        )
        print("YOLO ByteTrack active (Simple mode)")
        print("INFO: For Norfair: Python 3.11 + pip install norfair")
    
    # Frame skip - Get from config
    performance_config = config.get_performance_config()
    TRACK_EVERY_N_FRAMES = performance_config.get('track_every_n_frames', 1)
    DISPLAY_EVERY_N_FRAMES = 2  # Visualization every 2 frames (for performance)
    last_annotated_frame = None  # Store last processed frame
    
    # Video source - Get from config
    camera_config = config.get_camera_config()
    video_source = camera_config.get('source', 0)
    
    # Video capture with optimizations
    cap = cv2.VideoCapture(video_source)
    
    # Webcam optimizations - Get resolution from config
    resolution = camera_config.get('resolution', [640, 480])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, camera_config.get('buffer_size', 1))
    
    if not cap.isOpened():
        print(f"Error: Unable to connect to webcam")
        print(f"Video source: {video_source}")
        print("Please check:")
        print("1. Webcam is connected and not used by another application")
        print("2. Camera permissions are enabled")
        return
    
    print(f"Successfully connected to webcam!")
    
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 60
    
    print(f"Resolution: {w}x{h}")
    print(f"Target FPS: 60")
    print(f"Actual FPS: {fps}")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("yolo11_object_pose_output.avi", fourcc, fps, (w, h))
    
    # Load pose detection model
    print("Loading YOLO11 Pose model...")
    
    # Pose Detection model - Get model path from config
    yolo_config = config.get_yolo_config()
    pose_model = YOLO(yolo_config.get('model_path', 'yolo11n-pose.pt'))
    print("Pose Detection model ready!")
    
    # System startup logs
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    log_system_start(device, gpu_name)
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps_counter = 0
    fps_start_time = time.time()
    
    # Create normal window - Get window name from config
    ui_config = config.get_ui_config()
    window_name = ui_config.get('window_name', 'YOLO11 Pose Detection')
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set window size - suitable for low resolution
    cv2.resizeWindow(window_name, 800, 600)
    
    while cap.isOpened():
        frame_start_time = time.time()  # Start time for frame processing
        frame_count += 1
        fps_counter += 1
        
        success, frame = cap.read()
        if not success:
            log_with_timestamp("Frame could not be read! Exiting...", "WARNING")
            break
        
        # POSE DETECTION (Pure detection if Norfair available, otherwise YOLO tracking)
        pose_start = time.time()
        if USE_NORFAIR:
            # Norfair will handle its own tracking, only detection
            pose_results = pose_model(
                frame, 
                save=False, 
                show=False, 
                conf=0.25,
                verbose=False, 
                device=device, 
                half=False
            )
        else:
            # Let YOLO handle its own tracking
            pose_results = pose_model.track(
                frame, 
                persist=True,
                save=False, 
                show=False, 
                conf=0.25,
                iou=0.4,
                verbose=False, 
                device=device, 
                half=False,
                tracker="bytetrack.yaml"
            )
        pose_time = time.time() - pose_start
        
        # Use original frame as base
        annotated_frame = frame.copy()
        
        # Process pose detection information
        pose_keypoints = None
        person_count = 0
        track_ids = []
        boxes = []
        
        if pose_results[0].keypoints is not None:
            pose_keypoints = pose_results[0].keypoints.data.cpu().numpy()
            person_count = len(pose_keypoints)
            
            # Extract bounding boxes
            if pose_results[0].boxes is not None:
                boxes_xyxy = pose_results[0].boxes.xyxy.cpu().numpy()
                for box in boxes_xyxy:
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    boxes.append([x_center, y_center, width, height])
            
            # Tracking (Norfair or YOLO) - USE ONLY ONE
            if USE_NORFAIR and len(pose_keypoints) > 0 and len(boxes) > 0:
                # Norfair + ReID tracking
                track_ids, finished_tracks = track_manager.update_tracks_with_norfair(
                    pose_keypoints, boxes, frame=frame
                )
            elif not USE_NORFAIR:
                # YOLO tracking
                from tracking import extract_boxes_from_results
                track_ids, boxes = extract_boxes_from_results(pose_results)
            else:
                # Norfair active but no detection
                track_ids = []
        
        # FULL VISUALIZATION EVERY N FRAMES (for performance)
        should_display = (frame_count % DISPLAY_EVERY_N_FRAMES == 0)
        
        # Process pose information and add to frame
        if pose_keypoints is not None and len(pose_keypoints) > 0:
            # Draw pose skeleton lines with numbers (only on display frames)
            if should_display:
                annotated_frame = draw_pose_skeleton(annotated_frame, pose_keypoints, confidence_threshold=0.3, show_numbers=True)
            
            # Keypoint analysis
            person_count = len(pose_keypoints)
            
            # Keypoint counts and confidence scores
            total_keypoints = 0
            visible_keypoints = 0
            head_pose_data = []
            
            # Log track IDs (Norfair or YOLO)
            if track_ids and frame_count % 30 == 0:  # Every 30 frames
                log_with_timestamp(f"Track IDs: {track_ids}", "TRACK")
            
            for person_idx, keypoints in enumerate(pose_keypoints):
                # Use track ID (Norfair or YOLO)
                track_id = track_ids[person_idx] if person_idx < len(track_ids) else None
                
                # Calculate head pose for each person
                result = calculate_head_pose(keypoints)
                
                if result[0] is not None:  # yaw is not None
                    yaw, pitch, roll, head_center = result
                    
                    # Draw head pose visualization (only on display frames)
                    if should_display:
                        annotated_frame = draw_head_pose(annotated_frame, yaw, pitch, roll, head_center)
                        
                        # Show YOLO Track ID above head
                        annotated_frame = draw_track_id_on_head(annotated_frame, head_center, track_id)
                    
                    # Store head pose data with track ID
                    head_pose_data.append({
                        'person': person_idx,
                        'track_id': track_id,
                        'yaw': yaw,
                        'pitch': pitch,
                        'roll': roll,
                        'head_center': head_center
                    })
                
                for kp_idx, (x, y, conf) in enumerate(keypoints):
                    total_keypoints += 1
                    if conf > 0.3:  # Visible point threshold
                        visible_keypoints += 1


            # Update total person count (never decreases)
            global total_person_count
            if person_count > total_person_count:
                total_person_count = person_count
            
            # Pose quality analysis
            pose_quality = (visible_keypoints / total_keypoints) * 100 if total_keypoints > 0 else 0
            
            # Log detailed detection results (every 60 frames)
            if frame_count % 60 == 0:
                log_detection_results(person_count, pose_quality, head_pose_data)
            
            # Draw UI overlay (only on display frames)
            if should_display:
                info_data = create_info_data(
                    person_count=person_count,
                    track_ids=track_ids,
                    track_history_count=len(track_manager.get_all_track_history()),
                    pose_quality=pose_quality,
                    visible_keypoints=visible_keypoints,
                    total_keypoints=total_keypoints,
                    head_pose_data=head_pose_data
                )
                annotated_frame = draw_info_overlay(annotated_frame, info_data)
            
            # ===== TRACK HISTORY and LINES =====
            # Update tracks (for YOLO tracking - Norfair already updated)
            if not USE_NORFAIR and track_ids and boxes:
                finished_tracks = track_manager.update_tracks(track_ids, boxes)
            
            # Optional: Draw tracking lines
            # if track_ids:
            #     from tracking import draw_track_lines
            #     annotated_frame = draw_track_lines(
            #         annotated_frame, 
            #         track_manager.get_all_track_history(), 
            #         track_ids
            #     )
        else:
            if should_display:
                annotated_frame = draw_no_person_message(annotated_frame)
        
        
        
        
        
        # Calculate and log performance metrics
        frame_processing_time = time.time() - frame_start_time
        
        # Calculate FPS every 30 frames
        if fps_counter >= 30:
            current_time = time.time()
            fps = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time
            
            # Log frame information
            log_frame_info(frame_count, fps, person_count, frame_processing_time)
        
        # Save and display frame (only on display frames)
        if should_display:
            video_writer.write(annotated_frame)
            last_annotated_frame = annotated_frame
            cv2.imshow(window_name, annotated_frame)
        else:
            # If not display frame, show last frame
            if last_annotated_frame is not None:
                cv2.imshow(window_name, last_annotated_frame)
        
        # Keyboard control
        if handle_keyboard_input(window_name) == 'quit':
            break
    
    # Cleanup
    total_runtime = time.time() - start_time
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    # Finalize still active tracks
    track_durations = track_manager.finalize_active_tracks()
    
    # Final statistics
    log_system_stats(frame_count, total_runtime, 
                    track_manager.get_all_track_history(), w, h, device, track_durations)

if __name__ == "__main__":
    main()