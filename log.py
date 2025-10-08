"""
YOLO Pose Detection - Logging Module
Logging functions and helper utilities
"""

from datetime import datetime
import sys
import time


def log_with_timestamp(message, log_type="INFO"):
    """Print log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3] 
    print(f"[{timestamp}] [{log_type}] {message}")
    sys.stdout.flush()  


def log_frame_info(frame_count, fps, person_count, processing_time):
    """Log frame processing information"""
    log_with_timestamp(f"Frame #{frame_count:06d} | FPS: {fps:.1f} | "
                      f"Persons: {person_count} | "
                      f"Process Time: {processing_time:.3f}s", "FRAME")


def log_detection_results(person_count, pose_quality, head_poses):
    """Log detailed detection results"""
    log_with_timestamp(f"Detection Results - Persons: {person_count}, "
                      f"Pose Quality: {pose_quality:.1f}%", "DETECT")
    
    if head_poses:
        for i, pose in enumerate(head_poses[:3]):
            track_info = f"ID:{pose['track_id']}" if pose.get('track_id') is not None else f"#{pose['person']}"
            log_with_timestamp(f"Person {track_info}: Yaw={pose['yaw']:.1f}°, "
                              f"Pitch={pose['pitch']:.1f}°, Roll={pose['roll']:.1f}°", "POSE")


def log_system_start(device, gpu_name=None):
    """System startup logs"""
    log_with_timestamp("Starting Pose detection with TRACKING...", "START")
    log_with_timestamp("Webcam detection active - 60 FPS target!", "SYSTEM")
    log_with_timestamp("", "SYSTEM")
    log_with_timestamp(f"DEVICE: {device.upper()}", "SYSTEM")
    if gpu_name:
        log_with_timestamp(f"   GPU: {gpu_name}", "SYSTEM")
    log_with_timestamp("", "SYSTEM")
    log_with_timestamp("  YOLO TRACKING SYSTEM ACTIVE!", "SYSTEM")
    log_with_timestamp("- model.track() is being used (persist=True)", "SYSTEM")
    log_with_timestamp("- stream=False - frame by frame tracking", "SYSTEM")
    log_with_timestamp("- Track history: Last 30 frames (1 second)", "SYSTEM")
    log_with_timestamp("- Finished tracks are automatically detected", "SYSTEM")
    log_with_timestamp("- ONLY YOLO Track ID is used", "SYSTEM")
    log_with_timestamp("", "SYSTEM")


def log_track_entry(track_id):
    """New track ID entry"""
    log_with_timestamp(f"ID:{track_id} - ENTERED (came in front of camera)", "ENTRY")


def log_track_exit(track_id, duration_seconds):
    """Track ID exit and duration information"""
    if duration_seconds < 60:
        duration_str = f"{duration_seconds:.1f} seconds"
    else:
        minutes = int(duration_seconds // 60)
        seconds = duration_seconds % 60
        duration_str = f"{minutes} minutes {seconds:.1f} seconds"
    
    log_with_timestamp(f"ID:{track_id} - EXITED (left the screen) | Total Duration: {duration_str}", "EXIT")


def log_system_stats(frame_count, total_runtime, track_history, w, h, device, track_durations):
    """System shutdown statistics"""
    log_with_timestamp("System shutting down...", "CLEANUP")
    
    log_with_timestamp("=== SYSTEM STATISTICS ===", "STATS")
    log_with_timestamp(f"Total frames processed: {frame_count}", "STATS")
    log_with_timestamp(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.1f} minutes)", "STATS")
    log_with_timestamp(f"Average FPS: {frame_count/total_runtime:.2f}", "STATS")
    log_with_timestamp("", "STATS")
    log_with_timestamp("=== TRACKING SYSTEM STATISTICS ===", "STATS")
    log_with_timestamp(f"Total track history records: {len(track_history)}", "STATS")
    log_with_timestamp(f"Last active tracks: {list(track_history.keys()) if track_history else 'None'}", "STATS")
    
    # Track duration statistics
    if track_durations:
        log_with_timestamp("", "STATS")
        log_with_timestamp("=== TOTAL STAY DURATIONS ===", "STATS")
        for track_id, duration in sorted(track_durations.items()):
            if duration < 60:
                duration_str = f"{duration:.1f} seconds"
            else:
                minutes = int(duration // 60)
                seconds = duration % 60
                duration_str = f"{minutes} minutes {seconds:.1f} seconds"
            log_with_timestamp(f"  ID:{track_id} → {duration_str}", "STATS")
    
    log_with_timestamp("", "STATS")
    log_with_timestamp(f"Output file: 'yolo11_object_pose_output.avi'", "STATS")
    log_with_timestamp(f"Resolution: {w}x{h}", "STATS")
    log_with_timestamp("System: YOLO11 Pose + YOLO Track", "STATS")
    log_with_timestamp(f"Device: {device.upper()}", "STATS")
    log_with_timestamp("Tracking: model.track(persist=True, stream=False)", "STATS")
    log_with_timestamp("Camera Source: Laptop Webcam (60 FPS)", "STATS")
    log_with_timestamp("Pose detection + Tracking completed!", "COMPLETE")