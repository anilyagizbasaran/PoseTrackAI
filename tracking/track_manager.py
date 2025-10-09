"""
Track Manager Module
Advanced tracking with Norfair + ReID + Skeletal Biometrics
"""

import time
import cv2
import numpy as np
from log import log_track_entry, log_track_exit, log_with_timestamp
from config_manager import get_config
from .skeletal_biometrics import extract_skeletal_features, print_skeletal_features, skeletal_distance
from .reid_extractor import EmbeddingExtractor

# Get configuration
_config = None

def _get_config():
    """Get or initialize config"""
    global _config
    if _config is None:
        _config = get_config()
    return _config

# Optional Norfair import
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


def create_reid_distance_function(keypoint_weight=0.4, reid_weight=0.3, skeletal_weight=0.3):
    """
    ReID distance function factory - Keypoint + Appearance + Skeletal Biometrics
    
    Args:
        keypoint_weight: Keypoint pozisyon ağırlığı (0-1)
        reid_weight: Görsel özellik ağırlığı (0-1)
        skeletal_weight: Kemik yapısı ağırlığı (0-1)
    
    Returns:
        distance_function: Norfair için mesafe fonksiyonu
    """
    def reid_distance(detected_pose, tracked_pose):
        """
        ReID + Keypoint + Skeletal combined distance
        
        Args:
            detected_pose: Norfair Detection object (embedding + skeletal)
            tracked_pose: Norfair TrackedObject object (embedding + skeletal)
        
        Returns:
            distance: Düşük değer = iyi eşleşme, yüksek değer = kötü eşleşme
        """
        # 1. Keypoint distance (pozisyon)
        kp_dist = keypoint_distance(detected_pose, tracked_pose)
        
        # 2. ReID (Görsel) distance
        det_embedding = getattr(detected_pose, 'embedding', None)
        track_embedding = getattr(tracked_pose, 'last_detection', None)
        
        if track_embedding is not None:
            track_embedding = getattr(track_embedding, 'embedding', None)
        
        # ReID mesafesi
        if det_embedding is not None and track_embedding is not None:
            cosine_sim = np.dot(det_embedding, track_embedding)
            cosine_dist = 1.0 - cosine_sim  # range [0, 2]
            reid_dist = cosine_dist / 2.0  # normalize [0, 1]
        else:
            reid_dist = 1.0  # Embedding yoksa maksimum mesafe
        
        # 3. Skeletal (Kemik yapısı) distance
        det_skeletal = getattr(detected_pose, 'skeletal_features', None)
        track_skeletal = getattr(tracked_pose, 'last_detection', None)
        
        if track_skeletal is not None:
            track_skeletal = getattr(track_skeletal, 'skeletal_features', None)
        
        # Skeletal mesafesi
        if det_skeletal is not None and track_skeletal is not None:
            skel_dist = skeletal_distance(det_skeletal, track_skeletal)
        else:
            skel_dist = 1.0  # Skeletal feature yoksa maksimum mesafe
        
        # === COMBINED DISTANCE ===
        # Keypoint: 0.4 (hareketli, kısa vadeli eşleşme)
        # ReID: 0.3 (görsel, kıyafete bağlı)
        # Skeletal: 0.3 (değişmez, uzun vadeli!)
        combined_dist = (
            keypoint_weight * kp_dist + 
            reid_weight * reid_dist + 
            skeletal_weight * skel_dist
        )
        
        return combined_dist
    
    return reid_distance


class TrackManager:
    """
    Advanced Track ID and History Management - Norfair Integration
    
    Features:
    - Norfair: Kalman filter + motion prediction + track recovery
    - ReID: Appearance-based re-identification (ResNet50)
    - Skeletal Biometrics: Bone structure matching (clothing-independent)
    - Persistent Database: Long-term person recognition across sessions
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
                 keypoint_weight=0.4,
                 reid_weight=0.3,
                 skeletal_weight=0.3,
                 use_skeletal=True,
                 use_persistent_reid=False,
                 persistent_db_path="person_database.json",
                 persistent_db_type="json",
                 persistent_similarity_threshold=0.65):
        """
        Initialize TrackManager with Norfair-based tracking
        
        Args:
            max_history_length: Maximum position count to store per track
            use_norfair: Use Norfair tracker (must be True - only supported method)
            distance_function: "keypoint" (OKS-based) - other options deprecated
            distance_threshold: Maximum distance for matching (lower = stricter)
            hit_counter_max: Maximum age before track is lost (frames)
            initialization_delay: Frames to wait before track becomes visible (false positive filter)
            pointwise_hit_counter_max: Hit counter for keypoints
            use_reid: Use ReID (Re-Identification) - Recognizes people who leave and re-enter
            reid_distance_threshold: ReID matching threshold (lower = stricter)
            reid_hit_counter_max: ReID track max age (longer)
            keypoint_weight: Keypoint weight in combined distance (0.3 recommended)
            reid_weight: ReID appearance weight (0.4 recommended)
            skeletal_weight: Skeletal biometrics weight (0.3 recommended)
            use_skeletal: Use skeletal features (bone lengths/ratios) - Clothing independent!
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
        self.use_skeletal = use_skeletal
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
            
            log_with_timestamp("Persistent ReID Database active! People recognized forever!", "DATABASE")
        else:
            self.person_db = None
            self.track_to_person = {}
        
        if use_norfair:
            # Distance function selection
            if use_reid:
                # Using ReID - combined distance (+ Skeletal if enabled)
                dist_func = create_reid_distance_function(
                    keypoint_weight=keypoint_weight,
                    reid_weight=reid_weight,
                    skeletal_weight=skeletal_weight
                )
                if use_skeletal:
                    log_with_timestamp(f"SKELETAL-FIRST Matching! (Skeletal: {skeletal_weight}, ReID: {reid_weight}, Keypoint: {keypoint_weight})", "TRACKING")
                    log_with_timestamp("   → Skeletal biometrics is PRIMARY identifier!", "TRACKING")
                else:
                    log_with_timestamp(f"ReID active! (Keypoint: {keypoint_weight}, ReID: {reid_weight})", "TRACKING")
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
            
            log_with_timestamp("Norfair Tracker active!", "TRACKING")
            log_with_timestamp(f"   - Distance: {'ReID+Keypoint' if use_reid else distance_function}", "TRACKING")
            log_with_timestamp(f"   - Threshold: {distance_threshold}", "TRACKING")
            log_with_timestamp(f"   - Max Age: {hit_counter_max} frames", "TRACKING")
            log_with_timestamp(f"   - Init Delay: {initialization_delay} frames", "TRACKING")
            if use_reid:
                log_with_timestamp(f"   - ReID Threshold: {reid_distance_threshold}", "TRACKING")
                log_with_timestamp(f"   - ReID Max Age: {reid_hit_counter_max} frames", "TRACKING")
    
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
            # ✅ MİNİMUM KEYPOINT KONTROLÜ - Config'den al (Güvenlik Katmanı)
            config = _get_config()
            detection_config = config.get_detection_config()
            keypoint_confidence = detection_config.get('keypoint_confidence', 0.3)
            min_visible = detection_config.get('min_visible_keypoints', 8)  # ✅ Default: 8
            
            visible_keypoints = np.sum(keypoints[:, 2] > keypoint_confidence)
            if visible_keypoints < min_visible:
                continue  # ✅ Kalitesiz tespit, atla!
            
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
            
            # Skeletal Biometrics: Add bone measurements
            if self.use_skeletal:
                skeletal_features = extract_skeletal_features(keypoints)
                detection.skeletal_features = skeletal_features  # Add skeletal features to detection
            
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
                    
                    # Update database (embedding + skeletal refresh)
                    if hasattr(tracked_obj.last_detection, 'embedding'):
                        embedding = tracked_obj.last_detection.embedding
                        skeletal = getattr(tracked_obj.last_detection, 'skeletal_features', None)
                        
                        if embedding is not None:
                            self.person_db.update_person(
                                person_id, 
                                embedding=embedding,
                                skeletal_features=skeletal,
                                increment_count=True
                            )
                else:
                    # New track! Search in database
                    if hasattr(tracked_obj.last_detection, 'embedding'):
                        embedding = tracked_obj.last_detection.embedding
                        skeletal = getattr(tracked_obj.last_detection, 'skeletal_features', None)
                        
                        if embedding is not None:
                            # Is there a similar person in database?
                            # SKELETAL-FIRST: Pass skeletal features as primary identifier
                            person_id, similarity = self.person_db.find_person(
                                embedding, 
                                skeletal_features=skeletal,
                                return_similarity=True
                            )
                            
                            # ✅ EMBEDDING VARSA ID VER (Skeletal opsiyonel)
                            # Skeletal sadece matching kalitesini arttırır, zorunlu değil
                            if person_id is not None:
                                # Found! Existing person
                                log_with_timestamp(f"Recognized person: {person_id} (similarity: {similarity:.3f})", "MATCH")
                                self.person_db.update_person(person_id, embedding=embedding, skeletal_features=skeletal)
                                self.track_to_person[norfair_track_id] = person_id
                            else:
                                # Not found! Add new person
                                log_with_timestamp(f"NEW PERSON! Highest similarity: {similarity:.3f} (threshold: {self.person_db.similarity_threshold})", "NEW")
                                person_id = self.person_db.add_person(embedding, skeletal_features=skeletal)
                                self.track_to_person[norfair_track_id] = person_id
                                
                                # YENİ KİŞİ: Skeletal features'ı göster
                                if skeletal is not None:
                                    print_skeletal_features(skeletal, person_id)
                            
                            # ✅ ÇOKLU ID ENGELLEME - Aynı kişi ID'si birden fazla track'te kullanılmasın!
                            if person_id in used_person_ids and not person_id.startswith("temp_"):
                                # Başka bir track aynı person_id kullanıyor!
                                log_with_timestamp(f"DUPLICATE PREVENTED: {person_id} already tracked!", "WARNING")
                                continue
                            
                            used_person_ids.add(person_id)
                        else:
                            # No embedding, use temporary ID
                            person_id = f"temp_{norfair_track_id}"
                    else:
                        # No embedding attribute, use temporary ID
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

