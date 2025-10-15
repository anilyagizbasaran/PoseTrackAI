"""
YOLO Pose Detection - Base Class
Base class containing common code for RTSP and Webcam
"""

import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from log import (log_with_timestamp, log_frame_info, log_detection_results, 
                 log_system_start, log_system_stats)
from pose_utils import calculate_head_pose, draw_head_pose, draw_pose_skeleton, draw_shoulder_measurement
from tracking import TrackManager, extract_skeletal_features
from ui import draw_info_overlay, draw_no_person_message, draw_track_id_on_head, create_info_data, draw_skeletal_info
from config_manager import get_config


class PoseDetectorBase:
    """
    YOLO Pose Detection Base Class
    Contains common code for RTSP and Webcam
    """
    
    def __init__(self, config_file="config_webcam.yaml"):
        """
        Initialize base class
        
        Args:
            config_file
        """
        self.config_file = config_file
        self.config = get_config(config_file)
        
        # Basic variables
        self.device = self.config.get_device()
        self.track_manager = None
        self.pose_model = None
        self.video_writer = None
        self.window_name = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.fps_counter = 0
        self.fps_start_time = None
        
        # Frame skip
        self.last_annotated_frame = None
        
        # Detection parameters
        self.conf_threshold = None
        self.iou_threshold = None
        self.keypoint_confidence = None
        self.show_numbers = None
        
        print(f"PoseDetectorBase initialized with {config_file}")
    
    def setup_tracking(self):
        """Setup Track Manager"""
        try:
            tracking_config = self.config.get_tracking_config()
            
            self.track_manager = TrackManager(
                max_history_length=tracking_config.get('max_history_length', 30),
                use_norfair=True,
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
                # Persistent Database
                use_persistent_reid=tracking_config.get('use_persistent_reid', True),
                persistent_db_path=tracking_config.get('persistent_db_path', 'person_database.json'),
                persistent_db_type=tracking_config.get('persistent_db_type', 'json'),
                persistent_similarity_threshold=tracking_config.get('persistent_similarity_threshold', 0.65)
            )
            print("NORFAIR + ReID + SKELETAL + DATABASE ACTIVE!")
            return True
        except Exception as e:
            print(f"ERROR: Failed to initialize Norfair: {e}")
            print("ERROR: Norfair is required for this system!")
            return False
    
    def setup_detection_parameters(self):
        """Setup detection parameters"""
        detection_config = self.config.get_detection_config()
        self.conf_threshold = detection_config.get('conf_threshold', 0.25)
        self.iou_threshold = detection_config.get('iou_threshold', 0.4)
        self.keypoint_confidence = detection_config.get('keypoint_confidence', 0.3)
        self.show_numbers = detection_config.get('show_keypoint_numbers', True)
    
    def setup_yolo_model(self):
        """Load YOLO model"""
        print("\nLoading YOLO11 Pose model...")
        yolo_config = self.config.get_yolo_config()
        self.pose_model = YOLO(yolo_config.get('model_path', 'yolo11n-pose.pt'))
        print("Pose Detection model ready!")
    
    def setup_ui(self):
        """Setup UI settings"""
        ui_config = self.config.get_ui_config()
        self.window_name = ui_config.get('window_name', 'YOLO11 Pose Detection')
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
    
    def setup_video_writer(self, width, height, fps):
        """Setup video writer"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_filename = f"{self.config_file.replace('.yaml', '').replace('config_', '')}_pose_output.avi"
        self.video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        return output_filename
    
    def handle_keyboard_input(self, window_name):
        """Common keyboard input handler"""
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
        elif key == ord('r'):  # 'r' key to reconnect (for RTSP)
            return 'reconnect'
        
        return None
    
    def detect_poses(self, frame):
        """Pose detection"""
        pose_start = time.time()
        pose_results = self.pose_model(
            frame, 
            save=False, 
            show=False, 
            conf=self.conf_threshold,
            verbose=False, 
            device=self.device, 
            half=False
        )
        pose_time = time.time() - pose_start
        
        return pose_results, pose_time
    
    def process_pose_results(self, pose_results, frame):
        """Process pose results"""
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
            
            # Tracking
            if len(pose_keypoints) > 0 and len(boxes) > 0:
                track_ids, finished_tracks = self.track_manager.update_tracks_with_norfair(
                    pose_keypoints, boxes, frame=frame
                )
            else:
                track_ids = []
        
        return annotated_frame, pose_keypoints, person_count, track_ids, boxes
    
    def draw_pose_visualization(self, annotated_frame, pose_keypoints, track_ids, person_count):
        """Draw pose visualization"""
        if pose_keypoints is not None and len(pose_keypoints) > 0:
            annotated_frame = draw_pose_skeleton(annotated_frame, pose_keypoints, 
                                                confidence_threshold=self.keypoint_confidence, 
                                                show_numbers=self.show_numbers)
            
            total_keypoints = 0
            visible_keypoints = 0
            head_pose_data = []
            skeletal_data = []
            
            # Track ID logging
            if track_ids and self.frame_count % 30 == 0:
                log_with_timestamp(f"Track IDs: {track_ids}", "TRACK")
            
            for person_idx, keypoints in enumerate(pose_keypoints):
                track_id = track_ids[person_idx] if person_idx < len(track_ids) else None
                
                # Calculate skeletal features
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
                    
                    # Show shoulder distance
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
            if self.frame_count % 60 == 0:
                log_detection_results(person_count, pose_quality, head_pose_data)
            
            # UI overlay
            info_data = create_info_data(
                person_count=person_count,
                track_ids=track_ids,
                track_history_count=len(self.track_manager.get_all_track_history()),
                pose_quality=pose_quality,
                visible_keypoints=visible_keypoints,
                total_keypoints=total_keypoints,
                head_pose_data=head_pose_data,
                skeletal_data=skeletal_data
            )
            annotated_frame = draw_info_overlay(annotated_frame, info_data)
            
            # Skeletal biometrics bilgilerini göster
            if skeletal_data:
                annotated_frame = draw_skeletal_info(annotated_frame, skeletal_data)
        else:
            annotated_frame = draw_no_person_message(annotated_frame)
        
        return annotated_frame
    
    def update_performance_stats(self, person_count):
        """Update performance statistics"""
        frame_processing_time = time.time() - self.frame_start_time
        
        if self.fps_counter >= 30:
            current_time = time.time()
            fps_calc = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            
            log_frame_info(self.frame_count, fps_calc, person_count, frame_processing_time)
    
    def initialize_performance_tracking(self):
        """Initialize performance tracking"""
        self.start_time = time.time()
        self.fps_counter = 0
        self.fps_start_time = time.time()
    
    def log_system_startup(self, camera_type="Webcam"):
        """System startup logs"""
        gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else None
        log_system_start(self.device, gpu_name, True)
        log_with_timestamp(f"{camera_type} mode active", "SYSTEM")
        log_with_timestamp("", "SYSTEM")
    
    def cleanup(self, width, height):
        """System cleanup"""
        total_runtime = time.time() - self.start_time
        
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        track_durations = self.track_manager.finalize_active_tracks()
        log_system_stats(self.frame_count, total_runtime, 
                        self.track_manager.get_all_track_history(), width, height, self.device, track_durations, True)
        
        output_filename = f"{self.config_file.replace('.yaml', '').replace('config_', '')}_pose_output.avi"
        log_with_timestamp(f"Output file: '{output_filename}'", "COMPLETE")
    
    def should_process_frame(self, process_every_n_frames=1):
        """Frame processing control"""
        return self.frame_count % process_every_n_frames == 0
    
    def increment_frame_count(self):
        """Increment frame counter"""
        self.frame_count += 1
        self.fps_counter += 1
    
    def set_frame_start_time(self):
        """Set frame start time"""
        self.frame_start_time = time.time()
    
    def get_performance_config(self):
        """Get performance configuration"""
        return self.config.get_performance_config()
    
    def get_camera_config(self):
        """Get camera configuration"""
        return self.config.get_camera_config()
    
    def get_rtsp_config(self):
        """Get RTSP configuration"""
        return self.config.get_rtsp_config()
