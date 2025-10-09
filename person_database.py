"""
Persistent ReID Person Database
Stores person embeddings permanently
People get the same ID whenever they return
"""

import json
import numpy as np
import os
from datetime import datetime
from typing import Optional
from config_manager import get_config
from log import log_with_timestamp

# Get configuration
_config = None

def _get_config():
    """Get or initialize config"""
    global _config
    if _config is None:
        _config = get_config()
    return _config


class PersonDatabase:
    """
    Persistent ReID Identity System
    
    Features:
    - Store embeddings + skeletal features in JSON
    - Skeletal-first matching with adaptive weighting
    - Exponential moving average for feature updates
    - Auto-save on every change
    """
    
    def __init__(self, db_path="person_database.json", 
                 db_type="json",
                 similarity_threshold=0.65,
                 auto_save=True,
                 max_persons=1000):
        """
        Args:
            db_path: Database file path
            db_type: "json" (only supported type)
            similarity_threshold: Minimum similarity for matching (0-1)
            auto_save: Auto-save on every change
            max_persons: Maximum person count (memory limit)
        """
        self.db_path = db_path
        self.db_type = db_type
        self.similarity_threshold = similarity_threshold
        self.auto_save = auto_save
        self.max_persons = max_persons
        
        # In-memory database
        self.persons = {}  # person_id: PersonData
        
        # Statistics
        self.total_matches = 0
        self.total_new_persons = 0
        
        # Load existing database
        self._load_database()
        
        log_with_timestamp("PersonDatabase ready!", "DATABASE")
        log_with_timestamp(f"   - Database: {db_path}", "DATABASE")
        log_with_timestamp(f"   - Type: {db_type}", "DATABASE")
        log_with_timestamp(f"   - Threshold: {similarity_threshold}", "DATABASE")
        log_with_timestamp(f"   - Person count: {len(self.persons)}", "DATABASE")
    
    def _load_database(self):
        """Load database from JSON"""
        if not os.path.exists(self.db_path):
            log_with_timestamp(f"Creating new database: {self.db_path}", "INFO")
            return
        
        try:
            self._load_from_json()
        except Exception as e:
            log_with_timestamp(f"Database could not be loaded: {e}", "WARNING")
            log_with_timestamp("Starting new database...", "INFO")
    
    def _load_from_json(self):
        """Load from JSON"""
        with open(self.db_path, 'r') as f:
            data = json.load(f)
        
        for person_id, person_data in data['persons'].items():
            self.persons[person_id] = {
                'embedding': np.array(person_data['embedding']),
                'skeletal_features': np.array(person_data['skeletal_features']) if 'skeletal_features' in person_data else None,
                'first_seen': person_data['first_seen'],
                'last_seen': person_data['last_seen'],
                'seen_count': person_data['seen_count']
            }
        
        self.total_matches = data.get('total_matches', 0)
        self.total_new_persons = data.get('total_new_persons', 0)
    
    def _save_database(self):
        """Save database to JSON"""
        if not self.auto_save:
            return
        
        try:
            self._save_to_json()
        except Exception as e:
            log_with_timestamp(f"Database could not be saved: {e}", "WARNING")
    
    def _save_to_json(self):
        """Save to JSON"""
        data = {
            'persons': {},
            'total_matches': self.total_matches,
            'total_new_persons': self.total_new_persons,
            'last_updated': datetime.now().isoformat()
        }
        
        for person_id, person_data in self.persons.items():
            data['persons'][person_id] = {
                'embedding': person_data['embedding'].tolist(),
                'skeletal_features': person_data['skeletal_features'].tolist() if person_data.get('skeletal_features') is not None else None,
                'first_seen': person_data['first_seen'],
                'last_seen': person_data['last_seen'],
                'seen_count': person_data['seen_count']
            }
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def find_person(self, embedding: np.ndarray, skeletal_features: Optional[np.ndarray] = None, 
                    return_similarity=False) -> Optional[str]:
        """
        Find person by skeletal features (primary) + embedding (secondary)
        
        SKELETAL-FIRST APPROACH: Prioritizes bone structure over appearance
        
        Args:
            embedding: Query embedding [2048]
            skeletal_features: Query skeletal features [16] - PRIMARY!
            return_similarity: Also return similarity score
        
        Returns:
            person_id: Matching person ID or None
            similarity: (optional) Combined similarity score
        """
        if not self.persons:
            return (None, 0.0) if return_similarity else None
        
        best_person_id = None
        best_similarity = -1.0
        
        # Get config
        config = _get_config()
        skeletal_config = config.get_skeletal_config()
        min_quality = skeletal_config.get('min_quality_measurements', 10)
        
        # ADAPTIVE WEIGHTING: Skeletal varsa primary, yoksa embedding primary
        has_skeletal = skeletal_features is not None
        
        # Compare with all persons
        for person_id, person_data in self.persons.items():
            combined_similarity = 0.0
            db_has_skeletal = person_data.get('skeletal_features') is not None
            
            # AĞIRLIK STRATEJİSİ - Kalite kontrolü
            skeletal_quality_ok = False
            
            if has_skeletal and db_has_skeletal:
                # Skeletal features kalitesi yeterli mi?
                new_skel_arr = np.array(skeletal_features) if isinstance(skeletal_features, list) else skeletal_features
                db_skel_arr = np.array(person_data['skeletal_features']) if isinstance(person_data['skeletal_features'], list) else person_data['skeletal_features']
                
                new_visible = np.sum(new_skel_arr > 0.001)
                db_visible = np.sum(db_skel_arr > 0.001)
                
                # Her ikisinde de yeterli ölçüm varsa skeletal güvenilir
                if new_visible >= min_quality and db_visible >= min_quality:
                    skeletal_quality_ok = True
            
            if skeletal_quality_ok:
                # Yüksek kalite skeletal → SKELETAL-FIRST
                skeletal_weight = 0.7
                embedding_weight = 0.3
            elif has_skeletal and db_has_skeletal:
                # Düşük kalite skeletal → EMBEDDING-FIRST
                skeletal_weight = 0.2
                embedding_weight = 0.8
            elif has_skeletal or db_has_skeletal:
                # Birinde var birinde yok → EMBEDDING-FIRST
                skeletal_weight = 0.1
                embedding_weight = 0.9
            else:
                # İkisinde de yok → EMBEDDING ONLY
                skeletal_weight = 0.0
                embedding_weight = 1.0
            
            # 1. SKELETAL SIMILARITY (PRIMARY when available)
            if has_skeletal and db_has_skeletal:
                db_skeletal = person_data['skeletal_features']
                
                # Skeletal distance → similarity (0 = perfect match, higher = worse)
                # Only compare non-zero features
                valid_mask = (skeletal_features > 0.001) & (db_skeletal > 0.001)
                if np.any(valid_mask):
                    diff = skeletal_features[valid_mask] - db_skeletal[valid_mask]
                    skeletal_distance = np.sqrt(np.mean(diff ** 2))
                    # Convert distance to similarity (inverse, normalized)
                    skeletal_similarity = max(0.0, 1.0 - (skeletal_distance / 0.5))  # 0.5 = threshold
                    combined_similarity += skeletal_weight * skeletal_similarity
            
            # 2. EMBEDDING SIMILARITY (FALLBACK when skeletal not available)
            db_embedding = person_data['embedding']
            embedding_similarity = np.dot(embedding, db_embedding)
            combined_similarity += embedding_weight * embedding_similarity
            
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_person_id = person_id
        
        # ADAPTIVE THRESHOLD: Skeletal yoksa threshold düşür
        effective_threshold = self.similarity_threshold
        if not has_skeletal:
            # Skeletal features yoksa threshold düşür (daha kolay tanıma)
            effective_threshold = self.similarity_threshold * 0.8  # %20 daha düşük
        
        # Threshold check
        if best_similarity >= effective_threshold:
            if return_similarity:
                return best_person_id, best_similarity
            return best_person_id
        
        if return_similarity:
            return None, best_similarity
        return None
    
    def add_person(self, embedding: np.ndarray, person_id: Optional[str] = None,
                   skeletal_features: Optional[np.ndarray] = None) -> str:
        """
        Add new person
        
        Args:
            embedding: Person's embedding [2048]
            person_id: Optional custom ID (auto-generated if not provided)
            skeletal_features: Optional skeletal biometrics (bone lengths/ratios)
        
        Returns:
            person_id: Assigned or created ID
        """
        # Max person check
        if len(self.persons) >= self.max_persons:
            log_with_timestamp(f"Maximum person count reached ({self.max_persons})", "WARNING")
            # Delete oldest person
            oldest_id = min(self.persons.keys(), 
                          key=lambda k: self.persons[k]['last_seen'])
            self.delete_person(oldest_id)
        
        # Create Person ID
        if person_id is None:
            person_id = f"person_{len(self.persons) + 1:04d}"
            # ID collision check
            while person_id in self.persons:
                person_id = f"person_{np.random.randint(10000):04d}"
        
        # Add person
        now = datetime.now().isoformat()
        self.persons[person_id] = {
            'embedding': embedding.copy(),
            'skeletal_features': skeletal_features.copy() if skeletal_features is not None else None,
            'first_seen': now,
            'last_seen': now,
            'seen_count': 1
        }
        
        self.total_new_persons += 1
        self._save_database()
        
        log_with_timestamp(f"New person added: {person_id}", "DATABASE")
        
        return person_id
    
    def update_person(self, person_id: str, embedding: Optional[np.ndarray] = None,
                     skeletal_features: Optional[np.ndarray] = None,
                     increment_count=True):
        """
        Update existing person
        
        Args:
            person_id: Person ID to update
            embedding: New embedding (averaged with old if provided)
            skeletal_features: New skeletal features (averaged with old if provided)
            increment_count: Increment seen count
        """
        if person_id not in self.persons:
            log_with_timestamp(f"Person not found: {person_id}", "WARNING")
            return
        
        person_data = self.persons[person_id]
        
        # Get config
        config = _get_config()
        skeletal_config = config.get_skeletal_config()
        
        # Update embedding (exponential moving average)
        if embedding is not None:
            alpha = skeletal_config.get('alpha_embedding', 0.9)  # New embedding weight
            person_data['embedding'] = alpha * embedding + (1 - alpha) * person_data['embedding']
            # Re-normalize
            person_data['embedding'] /= np.linalg.norm(person_data['embedding'])
        
        # Update skeletal features (exponential moving average)
        if skeletal_features is not None:
            # person_data içinde skeletal_features var mı ve None değil mi kontrol et
            existing_skeletal = person_data.get('skeletal_features')
            
            # Mevcut skeletal_features geçerli bir numpy array mi?
            if existing_skeletal is not None and isinstance(existing_skeletal, np.ndarray) and existing_skeletal.size > 0:
                # Güvenli ortalamalama
                try:
                    alpha = skeletal_config.get('alpha_skeletal', 0.7)  # Skeletal features are more stable
                    person_data['skeletal_features'] = alpha * skeletal_features + (1 - alpha) * existing_skeletal
                except (TypeError, ValueError):
                    # Hata olursa yeni değeri kullan
                    person_data['skeletal_features'] = skeletal_features.copy()
            else:
                # İlk kez skeletal features alınıyor veya eski değer geçersiz
                person_data['skeletal_features'] = skeletal_features.copy()
        
        # Update statistics
        person_data['last_seen'] = datetime.now().isoformat()
        if increment_count:
            person_data['seen_count'] += 1
        
        self.total_matches += 1
        self._save_database()
    
    def delete_person(self, person_id: str) -> bool:
        """
        Delete person
        
        Args:
            person_id: Person ID to delete
        
        Returns:
            success: Was deletion successful
        """
        if person_id not in self.persons:
            return False
        
        del self.persons[person_id]
        self._save_database()
        
        log_with_timestamp(f"Person deleted: {person_id}", "DATABASE")
        return True
    
    def __len__(self):
        """Number of persons in database"""
        return len(self.persons)
    
    def __contains__(self, person_id):
        """Is person ID in database"""
        return person_id in self.persons
