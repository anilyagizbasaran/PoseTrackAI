"""
YOLO Pose Detection - RTSP Camera Version
Pose detection with RTSP camera stream
"""

import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
from log import (log_with_timestamp, log_frame_info, log_detection_results, 
                 log_system_start, log_system_stats)
from pose_utils import calculate_head_pose, draw_head_pose, draw_pose_skeleton, draw_shoulder_measurement
from tracking import TrackManager, extract_skeletal_features
from ui import draw_info_overlay, draw_no_person_message, draw_track_id_on_head, create_info_data, draw_skeletal_info
from camera_rtsp import RTSPCamera
from config_manager import get_config

# (Removed unused global counter)


def handle_keyboard_input(window_name):
    """Handle keyboard input"""
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27:
        log_with_timestamp("User issued exit command...", "EXIT")
        return 'quit'
    elif key == ord('p'):
        log_with_timestamp("Video paused - press any key to continue...", "PAUSE")
        cv2.waitKey(0)
        log_with_timestamp("Video resuming...", "RESUME")
    elif key == ord('f'):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key == ord('w'):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    elif key == ord('r'):  # 'r' key to reconnect
        return 'reconnect'
    
    return None


def main():
    """Pose Detection with RTSP Camera"""
    print("YOLO11 Pose Detection - RTSP Camera")
    print("Controls:")
    print("  'q' or ESC = Exit")
    print("  'p' = Pause")
    print("  'f' = Fullscreen")
    print("  'w' = Normal Window")
    print("  'r' = Reconnect to Camera")
    
    # Load RTSP Configuration
    try:
        config = get_config('config_rtsp.yaml')
        print("RTSP Config loaded: config_rtsp.yaml")
    except:
        config = get_config()  # Fallback to main config
        print("WARNING: RTSP Config not found, using main config")
    
    # Get device from config
    device = config.get_device()
    print(f"Device: {device}")
    
    # Track Manager - Norfair + ReID + Persistent Database
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
                keypoint_weight=tracking_config.get('keypoint_weight', 0.1),
                reid_weight=tracking_config.get('reid_weight', 0.2),
                skeletal_weight=tracking_config.get('skeletal_weight', 0.7),
                use_skeletal=tracking_config.get('use_skeletal', True),
                # Persistent Database - SAME DATABASE!
                use_persistent_reid=tracking_config.get('use_persistent_reid', True),
                persistent_db_path=tracking_config.get('persistent_db_path', 'person_database.json'),
                persistent_db_type=tracking_config.get('persistent_db_type', 'json'),
                persistent_similarity_threshold=tracking_config.get('persistent_similarity_threshold', 0.65)
            )
            print("NORFAIR + ReID + SKELETAL + DATABASE ACTIVE!")
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
    detection_config = config.get_detection_config()
    rtsp_config = config.get_rtsp_config()
    
    PROCESS_EVERY_N_FRAMES = performance_config.get('track_every_n_frames', 1)
    last_annotated_frame = None
    
    # Get detection parameters from config
    conf_threshold = detection_config.get('conf_threshold', 0.25)
    iou_threshold = detection_config.get('iou_threshold', 0.4)
    keypoint_confidence = detection_config.get('keypoint_confidence', 0.3)
    show_numbers = detection_config.get('show_keypoint_numbers', True)
    
    # RTSP settings
    MAX_CONSECUTIVE_FAILURES = rtsp_config.get('max_consecutive_failures', 15)
    
    # RTSP Camera settings - Get from config
    camera_config = config.get_camera_config()
    
    # Get RTSP URL from config
    rtsp_url = camera_config.get('source', 'rtsp://admin:admin123@192.168.1.64:554/stream1')
    resolution = camera_config.get('resolution', [1280, 720])
    
    print(f"\nRTSP Camera Settings:")
    print(f"  Source: {rtsp_url}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    
    # Create camera directly from config
    camera = RTSPCamera(
        rtsp_url=rtsp_url,
        width=resolution[0],
        height=resolution[1]
    )
    
    # Connect to camera
    if not camera.connect():
        print("ERROR: Could not connect to camera!")
        return
    
    # Get camera properties
    cam_props = camera.get_properties()
    
    if cam_props is None:
        print("ERROR: Could not get camera properties!")
        camera.release()
        return
    
    w = cam_props['width']
    h = cam_props['height']
    fps = cam_props['fps'] if cam_props['fps'] > 0 else 25
    
    # Resolution check
    if w <= 0 or h <= 0:
        print(f"ERROR: Invalid resolution: {w}x{h}")
        print("   RTSP stream may not be working properly.")
        camera.release()
        return
    
    print(f"\nCamera active:")
    print(f"   Resolution: {w}x{h}")
    print(f"   FPS: {fps}")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("rtsp_pose_output.avi", fourcc, fps, (w, h))
    
    # Load YOLO model - Get model path from config
    print("\nLoading YOLO11 Pose model...")
    yolo_config = config.get_yolo_config()
    pose_model = YOLO(yolo_config.get('model_path', 'yolo11n-pose.pt'))
    print("Pose Detection model ready!")
    
    # System startup logs
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    log_system_start(device, gpu_name)
    log_with_timestamp("RTSP Camera mode active", "SYSTEM")
    log_with_timestamp("", "SYSTEM")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps_counter = 0
    fps_start_time = time.time()
    
    # Create window - Get window name from config
    ui_config = config.get_ui_config()
    window_name = ui_config.get('window_name', 'YOLO11 Pose Detection - RTSP')
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    consecutive_failures = 0
    
    while True:
        frame_start_time = time.time()
        frame_count += 1
        fps_counter += 1
        
        # Read frame
        success, frame = camera.read_frame()
        
        if not success:
            consecutive_failures += 1
            
            if consecutive_failures <= 3:  # Only warning for first 3 errors
                log_with_timestamp(f"WARNING: Frame could not be read! (Failed: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})", "WARNING")
            
            # Too many failed reads, reconnect
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log_with_timestamp(f"ERROR: {MAX_CONSECUTIVE_FAILURES} frames failed - reconnecting...", "ERROR")
                if not camera.reconnect(max_attempts=5, wait_time=3):
                    log_with_timestamp("ERROR: Could not reconnect to camera, exiting...", "ERROR")
                    break
                consecutive_failures = 0
            
            time.sleep(0.2)  # Short wait
            continue
        
        # Successful read
        consecutive_failures = 0
        
        # FRAME SKIP
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            if last_annotated_frame is not None:
                cv2.imshow(window_name, last_annotated_frame)
            
            action = handle_keyboard_input(window_name)
            if action == 'quit':
                break
            elif action == 'reconnect':
                camera.reconnect()
            continue
        
        # Pose Detection (Pure detection if Norfair available, otherwise YOLO tracking)
        pose_start = time.time()
        if USE_NORFAIR:
            # Norfair will handle its own tracking, only detection
            pose_results = pose_model(
                frame, 
                save=False, 
                show=False, 
                conf=conf_threshold,
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
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False, 
                device=device, 
                half=False,
                tracker="bytetrack.yaml"
            )
        pose_time = time.time() - pose_start
        
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
            
            # Tracking (Norfair)
            if USE_NORFAIR and len(pose_keypoints) > 0 and len(boxes) > 0:
                # Norfair + ReID tracking
                track_ids, finished_tracks = track_manager.update_tracks_with_norfair(
                    pose_keypoints, boxes, frame=frame
                )
        
        # Pose processing
        if pose_keypoints is not None and len(pose_keypoints) > 0:
            annotated_frame = draw_pose_skeleton(annotated_frame, pose_keypoints, 
                                                confidence_threshold=keypoint_confidence, 
                                                show_numbers=show_numbers)
            
            total_keypoints = 0
            visible_keypoints = 0
            head_pose_data = []
            skeletal_data = []  # Skeletal biometrics data
            
            # Track ID logging
            if track_ids and frame_count % 30 == 0:
                log_with_timestamp(f"YOLO Track IDs: {track_ids}", "TRACK")
            
            for person_idx, keypoints in enumerate(pose_keypoints):
                track_id = track_ids[person_idx] if person_idx < len(track_ids) else None
                
                # Skeletal features'ı hesapla
                skeletal_features = extract_skeletal_features(keypoints)
                if skeletal_features is not None:
                    visible_skeletal = int(np.sum(skeletal_features > 0.001))
                    skeletal_data.append({
                        'person_id': track_id if track_id else f"#{person_idx}",
                        'skeletal_features': skeletal_features,
                        'visible_count': visible_skeletal
                    })
                
                # Calculate head pose
                result = calculate_head_pose(keypoints)
                
                if result[0] is not None:
                    yaw, pitch, roll, head_center = result
                    annotated_frame = draw_head_pose(annotated_frame, yaw, pitch, roll, head_center)
                    annotated_frame = draw_track_id_on_head(annotated_frame, head_center, track_id)
                    
                    # Omuz mesafesini CM cinsinden göster (QuickPose tarzı)
                    annotated_frame = draw_shoulder_measurement(annotated_frame, keypoints)
                    
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
                    if conf > 0.3:
                        visible_keypoints += 1
            
            # Pose quality
            pose_quality = (visible_keypoints / total_keypoints) * 100 if total_keypoints > 0 else 0
            
            # Logging
            if frame_count % 60 == 0:
                log_detection_results(person_count, pose_quality, head_pose_data)
            
            # UI overlay
            info_data = create_info_data(
                person_count=person_count,
                track_ids=track_ids,
                track_history_count=len(track_manager.get_all_track_history()),
                pose_quality=pose_quality,
                visible_keypoints=visible_keypoints,
                total_keypoints=total_keypoints,
                head_pose_data=head_pose_data,
                skeletal_data=skeletal_data
            )
            annotated_frame = draw_info_overlay(annotated_frame, info_data)
            
            # Skeletal biometrics bilgilerini sağ üstte göster
            if skeletal_data:
                annotated_frame = draw_skeletal_info(annotated_frame, skeletal_data)
        else:
            annotated_frame = draw_no_person_message(annotated_frame)
        
        # Calculate FPS
        frame_processing_time = time.time() - frame_start_time
        
        if fps_counter >= 30:
            current_time = time.time()
            fps_calc = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time
            
            log_frame_info(frame_count, fps_calc, person_count, frame_processing_time)
        
        # Save and display
        video_writer.write(annotated_frame)
        last_annotated_frame = annotated_frame
        cv2.imshow(window_name, annotated_frame)
        
        # Keyboard control
        action = handle_keyboard_input(window_name)
        if action == 'quit':
            break
        elif action == 'reconnect':
            camera.reconnect()
    
    # Cleanup
    total_runtime = time.time() - start_time
    
    camera.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    track_durations = track_manager.finalize_active_tracks()
    log_system_stats(frame_count, total_runtime, 
                    track_manager.get_all_track_history(), w, h, device, track_durations)
    
    log_with_timestamp(f"Output file: 'rtsp_pose_output.avi'", "COMPLETE")


if __name__ == "__main__":
    main()