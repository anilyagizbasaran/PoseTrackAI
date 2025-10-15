"""
YOLO Pose Detection - RTSP Camera Version
Pose detection with RTSP camera stream using PoseDetectorBase
"""

import cv2
import time
from log import log_with_timestamp
from camera_rtsp import RTSPCamera
from pose_base import PoseDetectorBase


def main():
    """Pose Detection with RTSP Camera"""
    print("YOLO11 Pose Detection - RTSP Camera")
    print("Controls:")
    print("  'q' or ESC = Exit")
    print("  'p' = Pause")
    print("  'f' = Fullscreen")
    print("  'w' = Normal Window")
    print("  'r' = Reconnect to Camera")
    
    # Initialize PoseDetectorBase with RTSP config
    detector = PoseDetectorBase('config_rtsp.yaml')
    
    # Setup all components
    if not detector.setup_tracking():
        return
    
    detector.setup_detection_parameters()
    detector.setup_yolo_model()
    detector.setup_ui()
    
    # Get RTSP specific settings
    rtsp_config = detector.get_rtsp_config()
    MAX_CONSECUTIVE_FAILURES = rtsp_config.get('max_consecutive_failures', 15)
    
    # RTSP Camera settings
    camera_config = detector.get_camera_config()
    rtsp_url = camera_config.get('source', 'rtsp://admin:admin123@192.168.1.64:554/stream1')
    resolution = camera_config.get('resolution', [1280, 720])
    
    print(f"\nRTSP Camera Settings:")
    print(f"  Source: {rtsp_url}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    
    # Create camera
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
    
    # Setup video writer
    output_filename = detector.setup_video_writer(w, h, fps)
    
    # Initialize performance tracking
    detector.initialize_performance_tracking()
    detector.log_system_startup("RTSP Camera")
    
    # Get performance config
    performance_config = detector.get_performance_config()
    PROCESS_EVERY_N_FRAMES = performance_config.get('track_every_n_frames', 1)
    
    consecutive_failures = 0
    
    while True:
        detector.set_frame_start_time()
        detector.increment_frame_count()
        
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
        if not detector.should_process_frame(PROCESS_EVERY_N_FRAMES):
            if detector.last_annotated_frame is not None:
                cv2.imshow(detector.window_name, detector.last_annotated_frame)
            
            action = detector.handle_keyboard_input(detector.window_name)
            if action == 'quit':
                break
            elif action == 'reconnect':
                camera.reconnect()
            continue
        
        # POSE DETECTION
        pose_results, pose_time = detector.detect_poses(frame)
        
        # Process pose results
        annotated_frame, pose_keypoints, person_count, track_ids, boxes = detector.process_pose_results(pose_results, frame)
        
        # Draw pose visualization
        annotated_frame = detector.draw_pose_visualization(annotated_frame, pose_keypoints, track_ids, person_count)
        
        # Update performance stats
        detector.update_performance_stats(person_count)
        
        # Save and display
        detector.video_writer.write(annotated_frame)
        detector.last_annotated_frame = annotated_frame
        cv2.imshow(detector.window_name, annotated_frame)
        
        # Keyboard control
        action = detector.handle_keyboard_input(detector.window_name)
        if action == 'quit':
            break
        elif action == 'reconnect':
            camera.reconnect()
    
    # Cleanup
    camera.release()
    detector.cleanup(w, h)


if __name__ == "__main__":
    main()