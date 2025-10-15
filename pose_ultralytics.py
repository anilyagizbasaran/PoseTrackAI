"""
YOLO Pose Detection - Webcam Version
Pose detection with webcam using PoseDetectorBase
"""

import cv2
import time
from log import log_with_timestamp
from pose_base import PoseDetectorBase


def main():
    """Ultralytics Pose Detection with Tracking"""
    print("YOLO11 Pose Detection with Tracking starting...")
    print("Pose Detection with Track History")
    print("Controls:")
    print("  'q' or ESC = Exit")
    print("  'p' = Pause")
    print("  'f' = Fullscreen")
    print("  'w' = Normal Window")
    
    # Initialize PoseDetectorBase with webcam config
    detector = PoseDetectorBase('config_webcam.yaml')
    
    # Setup all components
    if not detector.setup_tracking():
        return
    
    detector.setup_detection_parameters()
    detector.setup_yolo_model()
    detector.setup_ui()
    
    # Video source - Get from config
    camera_config = detector.get_camera_config()
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
    
    # Setup video writer
    output_filename = detector.setup_video_writer(w, h, fps)
    
    # Initialize performance tracking
    detector.initialize_performance_tracking()
    detector.log_system_startup("Webcam")
    
    # Get performance config
    performance_config = detector.get_performance_config()
    TRACK_EVERY_N_FRAMES = performance_config.get('track_every_n_frames', 1)
    DISPLAY_EVERY_N_FRAMES = 2  # Visualization every 2 frames (for performance)
    
    while cap.isOpened():
        detector.set_frame_start_time()
        detector.increment_frame_count()
        
        success, frame = cap.read()
        if not success:
            log_with_timestamp("Frame could not be read! Exiting...", "WARNING")
            break
        
        # POSE DETECTION
        pose_results, pose_time = detector.detect_poses(frame)
        
        # Process pose results
        annotated_frame, pose_keypoints, person_count, track_ids, boxes = detector.process_pose_results(pose_results, frame)
        
        # FULL VISUALIZATION EVERY N FRAMES (for performance)
        should_display = (detector.frame_count % DISPLAY_EVERY_N_FRAMES == 0)
        
        # Draw pose visualization (only on display frames)
        if should_display:
            annotated_frame = detector.draw_pose_visualization(annotated_frame, pose_keypoints, track_ids, person_count)
        
        # Update performance stats
        detector.update_performance_stats(person_count)
        
        # Save and display frame (only on display frames)
        if should_display:
            detector.video_writer.write(annotated_frame)
            detector.last_annotated_frame = annotated_frame
            cv2.imshow(detector.window_name, annotated_frame)
        else:
            # If not display frame, show last frame
            if detector.last_annotated_frame is not None:
                cv2.imshow(detector.window_name, detector.last_annotated_frame)
        
        # Keyboard control
        if detector.handle_keyboard_input(detector.window_name) == 'quit':
            break
    
    # Cleanup
    cap.release()
    detector.cleanup(w, h)

if __name__ == "__main__":
    main()