"""
Persistent ReID Person Database
Stores person embeddings permanently
People get the same ID whenever they return
"""

import json
import sqlite3
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class PersonDatabase:
    """
    Persistent ReID Identity System
    
    Features:
    - Store embeddings in JSON or SQLite
    - Matching with cosine similarity
    - Person deletion from database
    - Statistics (first seen, last seen, seen count)
    - Export/Import support
    """
    
    def __init__(self, db_path="person_database.json", 
                 db_type="json",
                 similarity_threshold=0.65,
                 auto_save=True,
                 max_persons=1000):
        """
        Args:
            db_path: Database file path
            db_type: "json" or "sqlite"
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
        
        print(f"PersonDatabase ready!")
        print(f"   - Database: {db_path}")
        print(f"   - Type: {db_type}")
        print(f"   - Threshold: {similarity_threshold}")
        print(f"   - Person count: {len(self.persons)}")
    
    def _load_database(self):
        """Load database"""
        if not os.path.exists(self.db_path):
            print(f"   INFO: Creating new database: {self.db_path}")
            return
        
        try:
            if self.db_type == "json":
                self._load_from_json()
            elif self.db_type == "sqlite":
                self._load_from_sqlite()
        except Exception as e:
            print(f"   WARNING: Database could not be loaded: {e}")
            print(f"   INFO: Starting new database...")
    
    def _load_from_json(self):
        """Load from JSON"""
        with open(self.db_path, 'r') as f:
            data = json.load(f)
        
        for person_id, person_data in data['persons'].items():
            self.persons[person_id] = {
                'embedding': np.array(person_data['embedding']),
                'first_seen': person_data['first_seen'],
                'last_seen': person_data['last_seen'],
                'seen_count': person_data['seen_count'],
                'metadata': person_data.get('metadata', {})
            }
        
        self.total_matches = data.get('total_matches', 0)
        self.total_new_persons = data.get('total_new_persons', 0)
    
    def _load_from_sqlite(self):
        """Load from SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                seen_count INTEGER DEFAULT 1,
                metadata TEXT
            )
        ''')
        
        # Load data
        cursor.execute('SELECT * FROM persons')
        rows = cursor.fetchall()
        
        for row in rows:
            person_id, embedding_blob, first_seen, last_seen, seen_count, metadata = row
            self.persons[person_id] = {
                'embedding': np.frombuffer(embedding_blob, dtype=np.float32),
                'first_seen': first_seen,
                'last_seen': last_seen,
                'seen_count': seen_count,
                'metadata': json.loads(metadata) if metadata else {}
            }
        
        conn.close()
    
    def _save_database(self):
        """Save database"""
        if not self.auto_save:
            return
        
        try:
            if self.db_type == "json":
                self._save_to_json()
            elif self.db_type == "sqlite":
                self._save_to_sqlite()
        except Exception as e:
            print(f"   WARNING: Database could not be saved: {e}")
    
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
                'first_seen': person_data['first_seen'],
                'last_seen': person_data['last_seen'],
                'seen_count': person_data['seen_count'],
                'metadata': person_data['metadata']
            }
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_to_sqlite(self):
        """Save to SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear all data and rewrite (simple approach)
        cursor.execute('DELETE FROM persons')
        
        for person_id, person_data in self.persons.items():
            cursor.execute('''
                INSERT INTO persons (person_id, embedding, first_seen, last_seen, seen_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                person_id,
                person_data['embedding'].tobytes(),
                person_data['first_seen'],
                person_data['last_seen'],
                person_data['seen_count'],
                json.dumps(person_data['metadata'])
            ))
        
        conn.commit()
        conn.close()
    
    def find_person(self, embedding: np.ndarray, return_similarity=False) -> Optional[str]:
        """
        Find person by embedding
        
        Args:
            embedding: Query embedding [2048]
            return_similarity: Also return similarity score
        
        Returns:
            person_id: Matching person ID or None
            similarity: (optional) Similarity score
        """
        if not self.persons:
            return (None, 0.0) if return_similarity else None
        
        best_person_id = None
        best_similarity = -1.0
        
        # Compare with all persons
        for person_id, person_data in self.persons.items():
            db_embedding = person_data['embedding']
            
            # Cosine similarity
            similarity = np.dot(embedding, db_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_person_id = person_id
        
        # Threshold check
        if best_similarity >= self.similarity_threshold:
            if return_similarity:
                return best_person_id, best_similarity
            return best_person_id
        
        if return_similarity:
            return None, best_similarity
        return None
    
    def add_person(self, embedding: np.ndarray, person_id: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> str:
        """
        Add new person
        
        Args:
            embedding: Person's embedding [2048]
            person_id: Optional custom ID (auto-generated if not provided)
            metadata: Optional metadata (name, notes, etc.)
        
        Returns:
            person_id: Assigned or created ID
        """
        # Max person check
        if len(self.persons) >= self.max_persons:
            print(f"   WARNING: Maximum person count reached ({self.max_persons})")
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
            'first_seen': now,
            'last_seen': now,
            'seen_count': 1,
            'metadata': metadata or {}
        }
        
        self.total_new_persons += 1
        self._save_database()
        
        print(f"   New person added: {person_id}")
        
        return person_id
    
    def update_person(self, person_id: str, embedding: Optional[np.ndarray] = None,
                     metadata: Optional[Dict] = None, increment_count=True):
        """
        Update existing person
        
        Args:
            person_id: Person ID to update
            embedding: New embedding (averaged with old if provided)
            metadata: Metadata to update
            increment_count: Increment seen count
        """
        if person_id not in self.persons:
            print(f"   WARNING: Person not found: {person_id}")
            return
        
        person_data = self.persons[person_id]
        
        # Update embedding (exponential moving average)
        if embedding is not None:
            alpha = 0.9  # New embedding weight
            person_data['embedding'] = alpha * embedding + (1 - alpha) * person_data['embedding']
            # Re-normalize
            person_data['embedding'] /= np.linalg.norm(person_data['embedding'])
        
        # Update metadata
        if metadata is not None:
            person_data['metadata'].update(metadata)
        
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
        
        print(f"   Person deleted: {person_id}")
        return True
    
    def get_person_info(self, person_id: str) -> Optional[Dict]:
        """Get person information"""
        if person_id not in self.persons:
            return None
        
        person_data = self.persons[person_id]
        return {
            'person_id': person_id,
            'first_seen': person_data['first_seen'],
            'last_seen': person_data['last_seen'],
            'seen_count': person_data['seen_count'],
            'metadata': person_data['metadata']
        }
    
    def get_all_persons(self) -> List[Dict]:
        """List all persons"""
        return [self.get_person_info(pid) for pid in self.persons.keys()]
    
    def clear_database(self):
        """Clear database"""
        self.persons = {}
        self.total_matches = 0
        self.total_new_persons = 0
        self._save_database()
        print(f"   Database cleared")
    
    def export_database(self, export_path: str):
        """Export database (in JSON format)"""
        data = {
            'persons': {},
            'total_matches': self.total_matches,
            'total_new_persons': self.total_new_persons,
            'exported_at': datetime.now().isoformat()
        }
        
        for person_id, person_data in self.persons.items():
            data['persons'][person_id] = {
                'embedding': person_data['embedding'].tolist(),
                'first_seen': person_data['first_seen'],
                'last_seen': person_data['last_seen'],
                'seen_count': person_data['seen_count'],
                'metadata': person_data['metadata']
            }
        
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   Database exported: {export_path}")
    
    def import_database(self, import_path: str, merge=True):
        """
        Import database
        
        Args:
            import_path: JSON file to import
            merge: True to merge with existing data, False to replace
        """
        with open(import_path, 'r') as f:
            data = json.load(f)
        
        if not merge:
            self.persons = {}
        
        for person_id, person_data in data['persons'].items():
            if person_id not in self.persons:
                self.persons[person_id] = {
                    'embedding': np.array(person_data['embedding']),
                    'first_seen': person_data['first_seen'],
                    'last_seen': person_data['last_seen'],
                    'seen_count': person_data['seen_count'],
                    'metadata': person_data.get('metadata', {})
                }
        
        self._save_database()
        print(f"   Database imported: {len(data['persons'])} persons")
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        if not self.persons:
            return {
                'total_persons': 0,
                'total_matches': self.total_matches,
                'total_new_persons': self.total_new_persons
            }
        
        seen_counts = [p['seen_count'] for p in self.persons.values()]
        
        return {
            'total_persons': len(self.persons),
            'total_matches': self.total_matches,
            'total_new_persons': self.total_new_persons,
            'avg_seen_count': np.mean(seen_counts),
            'max_seen_count': np.max(seen_counts),
            'min_seen_count': np.min(seen_counts)
        }
    
    def __len__(self):
        """Number of persons in database"""
        return len(self.persons)
    
    def __contains__(self, person_id):
        """Is person ID in database"""
        return person_id in self.persons