"""
YOLO Pose Detection - Advanced Tracking Module with Norfair + ReID + Persistent Database

Comprehensive track ID management, entry/exit monitoring, and historical tracking
Norfair: Advanced tracking with Kalman filter + motion prediction + ReID integration
ReID: Re-identification system for recognizing people who leave and re-enter camera view
Persistent Database: Long-term person storage ensuring consistent IDs across sessions
"""

import time
import cv2
import numpy as np
from log import log_track_entry, log_track_exit

# Optional imports (loaded only when needed)
try:
    import torch
    import torchvision
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from norfair import Detection, Tracker
    from norfair.distances import mean_euclidean, frobenius, iou
    NORFAIR_AVAILABLE = True
except ImportError:
    NORFAIR_AVAILABLE = False
    Detection = None
    Tracker = None

try:
    from person_database import PersonDatabase
    PERSON_DB_AVAILABLE = True
except ImportError:
    PERSON_DB_AVAILABLE = False


# COCO pose keypoint sigmas for OKS calculation - GLOBAL
# Smaller sigma values = more sensitive matching
COCO_KEYPOINT_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89
]) / 10.0

class EmbeddingExtractor:
    """
    ReID Appearance Embedding Extractor
    
    Converts person appearances to vectors using ResNet50 backbone
    Provides robust person re-identification capabilities
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize embedding extractor
        
        Args:
            device: 'cuda' or 'cpu' for computation
        """
        self.device = device
        
        # ResNet50 model (pretrained on ImageNet)
        # Remove final FC layer to extract embedding vectors
        self.model = torchvision.models.resnet50(pretrained=True)
        
        # Remove final FC layer (2048-dimensional embedding)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.model.to(device)
        self.model.eval()  # Evaluation mode
        
        # Image preprocessing pipeline (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),  # Standard ReID dimensions
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"ReID Embedding Extractor ready! (Device: {device})")
    
    def extract_embedding(self, image_crop):
        """
        Extract embedding from cropped image
        
        Args:
            image_crop: NumPy array [H, W, 3] in BGR format
        
        Returns:
            embedding: NumPy array [2048] - appearance vector
        """
        if image_crop is None or image_crop.size == 0:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        
        # Convert NumPy to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transformations
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference (no gradient computation)
        with torch.no_grad():
            embedding = self.model(input_tensor)
        
        # Reshape from [1, 2048, 1, 1] to [2048]
        embedding = embedding.squeeze().cpu().numpy()
        
        # L2 normalization
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        return embedding
    
    def extract_embeddings_batch(self, image_crops):
        """
        Extract embeddings in batch (faster processing)
        
        Args:
            image_crops: List of NumPy arrays
        
        Returns:
            embeddings: List of NumPy arrays [2048]
        """
        if not image_crops:
            return []
        
        embeddings = []
        for crop in image_crops:
            emb = self.extract_embedding(crop)
            embeddings.append(emb)
        
        return embeddings


def keypoint_distance(detected_pose, tracked_pose):
    """
    Custom distance function for Norfair - Pose keypoint distance (without ReID)
    
    Combines OKS (Object Keypoint Similarity) with Euclidean distance
    
    Args:
        detected_pose: Norfair Detection object (new detection)
        tracked_pose: Norfair TrackedObject object (existing track)
    
    Returns:
        distance: Lower value = better match, higher value = worse match
    """
    # Extract keypoints from detection and track
    det_points = detected_pose.points  # [N, 2] - (x, y) coordinates
    track_estimate = tracked_pose.estimate  # [N, 2] - Kalman filter estimate
    
    # Return maximum distance if no points available
    if det_points is None or track_estimate is None:
        return 1e6
    
    # Calculate Euclidean distance
    distances = np.linalg.norm(det_points - track_estimate, axis=1)
    
    # OKS-style scoring: Some keypoints are more important
    # Example: nose, eyes are more stable -> lower sigma
    # Elbows, wrists are more mobile -> higher sigma
    
    # Estimate bounding box area (for normalization)
    det_min = np.min(det_points, axis=0)
    det_max = np.max(det_points, axis=0)
    area = (det_max[0] - det_min[0]) * (det_max[1] - det_min[1])
    s = np.sqrt(area) + 1e-6
    
    # OKS formula: e = d / (2 * s * sigma)
    # Only use keypoints with confidence > 0 (all are >0 by default in Norfair)
    e = distances / (2 * s * COCO_KEYPOINT_SIGMAS[:len(distances)])
    
    # Return mean squared error (Norfair minimizes this)
    # Lower value = better match
    mean_distance = np.mean(e ** 2)
    
    return mean_distance


def create_reid_distance_function(keypoint_weight=0.6, reid_weight=0.4):
    """
    ReID distance function factory - Keypoint + Appearance combination
    
    Args:
        keypoint_weight: Weight for keypoint similarity (0-1)
        reid_weight: Weight for ReID similarity (0-1)
    
    Returns:
        distance_function: Distance function for Norfair
    """
    def reid_distance(detected_pose, tracked_pose):
        """
        ReID + Keypoint combined distance calculation
        
        Args:
            detected_pose: Norfair Detection object (contains embedding)
            tracked_pose: Norfair TrackedObject object (contains embedding)
        
        Returns:
            distance: Lower value = better match, higher value = worse match
        """
        # 1. Keypoint distance
        kp_dist = keypoint_distance(detected_pose, tracked_pose)
        
        # 2. ReID (Appearance) distance
        det_embedding = getattr(detected_pose, 'embedding', None)
        track_embedding = getattr(tracked_pose, 'last_detection', None)
        
        if track_embedding is not None:
            track_embedding = getattr(track_embedding, 'embedding', None)
        
        # If no embedding available, use only keypoint distance
        if det_embedding is None or track_embedding is None:
            return kp_dist
        
        # Cosine distance (1 - cosine_similarity)
        # Cosine similarity: range [-1, 1], 1 = identical
        cosine_sim = np.dot(det_embedding, track_embedding)
        cosine_dist = 1.0 - cosine_sim  # range [0, 2]
        
        # Normalize to [0, 1] range
        reid_dist = cosine_dist / 2.0
        
        # Combined distance
        # Keypoint: 0.6, ReID: 0.4 (balanced weights)
        combined_dist = keypoint_weight * kp_dist + reid_weight * reid_dist
        
        return combined_dist
    
    return reid_distance


class TrackManager:
    """
    Track ID and history management - Integrated with Norfair
    
    Norfair features:
    - Motion prediction with Kalman filter
    - Automatic recovery of lost tracks
    - Reliable track filtering with hit counter
    """
    
    def __init__(self, max_history_length=30, use_norfair=True, 
                 distance_function="keypoint",
                 distance_threshold=0.5,
                 hit_counter_max=15,
                 initialization_delay=3,
                 pointwise_hit_counter_max=4,
                 use_reid=False,
                 reid_distance_threshold=0.3,
                 reid_hit_counter_max=100,
                 keypoint_weight=0.6,
                 reid_weight=0.4,
                 use_persistent_reid=False,
                 persistent_db_path="person_database.json",
                 persistent_db_type="json",
                 persistent_similarity_threshold=0.65):
        """
        Initialize TrackManager with comprehensive configuration
        
        Args:
            max_history_length: Maximum position count to store per track
            use_norfair: Use Norfair tracker (True recommended)
            distance_function: "keypoint", "euclidean", or "iou"
            distance_threshold: Maximum distance for matching (lower = stricter)
            hit_counter_max: Maximum age before track is lost (frames)
            initialization_delay: Frames to wait before track becomes visible (false positive filter)
            pointwise_hit_counter_max: Hit counter for keypoints
            use_reid: Use ReID (Re-Identification) - Recognizes people who leave and re-enter
            reid_distance_threshold: ReID matching threshold (lower = stricter)
            reid_hit_counter_max: ReID track max age (longer)
            keypoint_weight: Keypoint weight in combined distance
            reid_weight: ReID weight in combined distance
            use_persistent_reid: Use persistent ReID database - People recognized days later!
            persistent_db_path: Database file path
            persistent_db_type: "json" or "sqlite"
            persistent_similarity_threshold: Minimum similarity for DB matching
        """
        self.track_history = {}  # track_id: [(x, y), ...]
        self.track_entry_times = {}  # track_id: entry time
        self.track_durations = {}  # track_id: total duration
        self.max_history_length = max_history_length
        
        # Norfair tracker configuration
        self.use_norfair = use_norfair
        self.use_reid = use_reid
        self.use_persistent_reid = use_persistent_reid
        
        # ReID Embedding Extractor setup
        if use_reid:
            self.embedding_extractor = EmbeddingExtractor()
        else:
            self.embedding_extractor = None
        
        # Persistent ReID Database setup
        if use_persistent_reid:
            if not use_reid:
                raise ValueError("Persistent ReID requires use_reid=True!")
            
            self.person_db = PersonDatabase(
                db_path=persistent_db_path,
                db_type=persistent_db_type,
                similarity_threshold=persistent_similarity_threshold,
                auto_save=True,
                max_persons=1000
            )
            
            # Track ID to Database Person ID mapping
            self.track_to_person = {}  # norfair_track_id: database_person_id
            
            print(f"Persistent ReID Database active! People recognized forever!")
        else:
            self.person_db = None
            self.track_to_person = {}
        
        if use_norfair:
            # Distance function selection
            if use_reid:
                # Using ReID - combined distance
                dist_func = create_reid_distance_function(
                    keypoint_weight=keypoint_weight,
                    reid_weight=reid_weight
                )
                print(f"ReID active! (Keypoint: {keypoint_weight}, ReID: {reid_weight})")
            elif distance_function == "keypoint":
                dist_func = keypoint_distance
            elif distance_function == "euclidean":
                dist_func = mean_euclidean
            elif distance_function == "iou":
                dist_func = iou
            else:
                dist_func = keypoint_distance
            
            # Create Norfair Tracker
            self.norfair_tracker = Tracker(
                distance_function=dist_func,
                distance_threshold=distance_threshold,
                hit_counter_max=hit_counter_max,  # Longer track lifetime
                initialization_delay=initialization_delay,  # False positive filter
                pointwise_hit_counter_max=pointwise_hit_counter_max,  # Keypoint hit counter
                past_detections_length=20,  # Store past detections
                reid_distance_threshold=reid_distance_threshold if use_reid else None,  # ReID threshold
                reid_hit_counter_max=reid_hit_counter_max if use_reid else None,  # ReID max age
            )
            
            print(f"Norfair Tracker active!")
            print(f"   - Distance: {'ReID+Keypoint' if use_reid else distance_function}")
            print(f"   - Threshold: {distance_threshold}")
            print(f"   - Max Age: {hit_counter_max} frames")
            print(f"   - Init Delay: {initialization_delay} frames")
            if use_reid:
                print(f"   - ReID Threshold: {reid_distance_threshold}")
                print(f"   - ReID Max Age: {reid_hit_counter_max} frames")
    
    def update_tracks_with_norfair(self, pose_keypoints, boxes, frame=None):
        """
        Update tracks with Norfair (ReID supported)
        
        Args:
            pose_keypoints: Pose keypoints array [N, 17, 3] (N = person count)
            boxes: Bounding box list [N, 4] [(x_center, y_center, w, h), ...]
            frame: Original frame [H, W, 3] (required for ReID)
        
        Returns:
            track_ids: Assigned track IDs for each detection
            finished_tracks: Track IDs that are no longer visible
        """
        if not self.use_norfair:
            raise ValueError("Norfair mode not active! Initialize with use_norfair=True.")
        
        current_time = time.time()
        finished_tracks = set()
        
        # If no detections, update Norfair (age tracks)
        if pose_keypoints is None or len(pose_keypoints) == 0:
            tracked_objects = self.norfair_tracker.update([])
            return [], finished_tracks
        
        # Create Norfair Detection objects
        detections = []
        for i, (keypoints, box) in enumerate(zip(pose_keypoints, boxes)):
            # Convert keypoints to [N, 2] format (only x, y)
            points = keypoints[:, :2]  # [17, 2]
            
            # Create Norfair Detection object
            # scores: confidence values (optional)
            scores = keypoints[:, 2]  # [17]
            
            detection = Detection(
                points=points,
                scores=scores,
                label="person"  # Optional: class label
            )
            
            # ReID: Add embedding
            if self.use_reid and frame is not None:
                # Extract crop from bounding box
                x_c, y_c, w, h = box
                x1 = int(x_c - w / 2)
                y1 = int(y_c - h / 2)
                x2 = int(x_c + w / 2)
                y2 = int(y_c + h / 2)
                
                # Ensure crop stays within frame boundaries
                h_frame, w_frame = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_frame, x2)
                y2 = min(h_frame, y2)
                
                # Extract crop
                crop = frame[y1:y2, x1:x2]
                
                # Extract embedding
                if crop.size > 0:
                    embedding = self.embedding_extractor.extract_embedding(crop)
                    detection.embedding = embedding  # Add embedding to detection
            
            detections.append(detection)
        
        # Update tracks with Norfair (Kalman filter runs automatically!)
        tracked_objects = self.norfair_tracker.update(detections=detections)
        
        # Extract track IDs and positions
        track_ids = []
        current_track_ids_set = set()
        used_person_ids = set()  # Prevent multiple IDs
        
        for i, tracked_obj in enumerate(tracked_objects):
            norfair_track_id = tracked_obj.id
            
            # PERSISTENT ReID: Search in database
            if self.use_persistent_reid:
                # Does this track already have a person_id?
                if norfair_track_id in self.track_to_person:
                    # Use existing person_id
                    person_id = self.track_to_person[norfair_track_id]
                    
                    # Update database (embedding refresh)
                    if hasattr(tracked_obj.last_detection, 'embedding'):
                        embedding = tracked_obj.last_detection.embedding
                        if embedding is not None:
                            self.person_db.update_person(
                                person_id, 
                                embedding=embedding,
                                increment_count=True
                            )
                else:
                    # New track! Search in database
                    if hasattr(tracked_obj.last_detection, 'embedding'):
                        embedding = tracked_obj.last_detection.embedding
                        
                        if embedding is not None:
                            # Is there a similar person in database?
                            person_id, similarity = self.person_db.find_person(
                                embedding, 
                                return_similarity=True
                            )
                            
                            if person_id is not None:
                                # Found! Existing person
                                print(f"   Recognized person: {person_id} (similarity: {similarity:.3f})")
                                self.person_db.update_person(person_id, embedding=embedding)
                                self.track_to_person[norfair_track_id] = person_id
                            else:
                                # Not found! Add new person
                                print(f"   Person not recognized! Highest similarity: {similarity:.3f} (threshold: {self.person_db.similarity_threshold})")
                                person_id = self.person_db.add_person(embedding)
                                self.track_to_person[norfair_track_id] = person_id
                            
                            # Multiple ID check - is same person_id already in use?
                            if person_id in used_person_ids:
                                # Another track exists for same person, skip this track
                                print(f"   WARNING: Multiple ID prevented: {person_id} already in use")
                                continue
                            
                            used_person_ids.add(person_id)
                        else:
                            # No embedding, use temporary ID
                            person_id = f"temp_{norfair_track_id}"
                    else:
                        # No embedding, use temporary ID
                        person_id = f"temp_{norfair_track_id}"
                
                # Use Person ID (database person ID)
                track_id = person_id
            else:
                # Normal Norfair track ID
                track_id = norfair_track_id
            
            track_ids.append(track_id)
            current_track_ids_set.add(track_id)
            
            # New track entry
            if track_id not in self.track_entry_times:
                self.track_entry_times[track_id] = current_time
                log_track_entry(track_id)
            
            # Get track position (first keypoint or centroid)
            if tracked_obj.estimate is not None and len(tracked_obj.estimate) > 0:
                # Average of all keypoints (centroid)
                centroid = np.mean(tracked_obj.estimate, axis=0)
                x, y = centroid
            else:
                # Fallback: use box center
                if len(boxes) > 0:
                    # Find closest box
                    box_idx = len(track_ids) - 1
                    if box_idx < len(boxes):
                        x, y = boxes[box_idx][0], boxes[box_idx][1]
                    else:
                        x, y = 0, 0
                else:
                    x, y = 0, 0
            
            # Update position history
            track = self.track_history.get(track_id, [])
            track.append((float(x), float(y)))
            
            # Don't exceed maximum length
            if len(track) > self.max_history_length:
                track.pop(0)
            
            self.track_history[track_id] = track
        
        # Detect finished tracks
        finished_tracks = set(self.track_history.keys()) - current_track_ids_set
        
        # Process finished tracks
        if finished_tracks:
            for ft_id in finished_tracks:
                # Calculate duration
                if ft_id in self.track_entry_times:
                    duration = current_time - self.track_entry_times[ft_id]
                    self.track_durations[ft_id] = duration
                    log_track_exit(ft_id, duration)
                    del self.track_entry_times[ft_id]
                
                # Clear history
                self.track_history.pop(ft_id, None)
        
        return track_ids, finished_tracks
    
    def update_tracks(self, track_ids, boxes):
        """
        Update tracks (for YOLO tracking - without Norfair)
        
        Args:
            track_ids: List of track IDs in current frame
            boxes: Bounding box list [(x_center, y_center, w, h), ...]
        
        Returns:
            finished_tracks: Track IDs that are no longer visible
        """
        current_time = time.time()
        finished_tracks = set()
        
        if not track_ids or not boxes:
            return finished_tracks
        
        # Add new tracks and update existing tracks
        for box, track_id in zip(boxes, track_ids):
            # NEW TRACK ENTRY
            if track_id not in self.track_entry_times:
                self.track_entry_times[track_id] = current_time
                log_track_entry(track_id)
            
            # Update position history
            x, y, w, h = box
            track = self.track_history.get(track_id, [])
            track.append((float(x), float(y)))
            
            # Don't exceed maximum length
            if len(track) > self.max_history_length:
                track.pop(0)
            
            self.track_history[track_id] = track
        
        # Detect finished tracks
        current_track_ids = set(track_ids)
        finished_tracks = set(self.track_history.keys()) - current_track_ids
        
        # Process finished tracks
        if finished_tracks:
            for ft_id in finished_tracks:
                # Calculate duration
                if ft_id in self.track_entry_times:
                    duration = current_time - self.track_entry_times[ft_id]
                    self.track_durations[ft_id] = duration
                    log_track_exit(ft_id, duration)
                    del self.track_entry_times[ft_id]
                
                # Clear history
                self.track_history.pop(ft_id, None)
        
        return finished_tracks
    
    def get_track_history(self, track_id):
        """Get history for a specific track"""
        return self.track_history.get(track_id, [])
    
    def get_all_track_history(self):
        """Get all track histories"""
        return self.track_history
    
    def finalize_active_tracks(self):
        """
        Finalize still active tracks (when system is shutting down)
        
        Returns:
            track_durations: Durations of all tracks
        """
        if self.track_entry_times:
            current_time = time.time()
            for track_id, entry_time in self.track_entry_times.items():
                duration = current_time - entry_time
                self.track_durations[track_id] = duration
                log_track_exit(track_id, duration)
        
        return self.track_durations
    
    def get_all_durations(self):
        """Get all track durations"""
        return self.track_durations.copy()


def extract_boxes_from_results(pose_results):
    """
    Extract bounding box information from YOLO pose results
    
    Args:
        pose_results: YOLO pose detection results
    
    Returns:
        track_ids: List of track IDs
        boxes: Bounding box list [(x_center, y_center, w, h), ...]
    """
    track_ids = []
    boxes = []
    
    if pose_results[0].boxes is not None and pose_results[0].boxes.id is not None:
        track_ids = pose_results[0].boxes.id.int().cpu().tolist()
        
        # Get bounding box centers
        boxes_xyxy = pose_results[0].boxes.xyxy.cpu().numpy()
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            # Box center and dimensions
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            boxes.append([x_center, y_center, width, height])
    
    return track_ids, boxes


def draw_track_lines(frame, track_history, track_ids, color=(230, 230, 230), thickness=3):
    """
    Display track history with lines
    
    Args:
        frame: Frame to process
        track_history: Track histories dictionary
        track_ids: Current track IDs
        color: Line color (BGR)
        thickness: Line thickness
    
    Returns:
        frame: Frame with lines added
    """
    for track_id in track_ids:
        track = track_history.get(track_id, [])
        
        if len(track) > 1:
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, 
                         color=color, thickness=thickness)
    
    return frame