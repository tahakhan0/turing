import os
import json
import pickle
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

class PersistentStorage:
    """Manages persistent storage for face recognition and segmentation data"""
    
    def __init__(self, base_path: str = None):
        # Use environment variable or default path
        if base_path is None:
            base_path = os.getenv("PERSISTENT_STORAGE_PATH", "/Users/tahakhan/Desktop/home-security-db")
            
        self.base_path = base_path
        self.face_recognition_path = os.path.join(base_path, "face_recognition")
        self.segmentation_path = os.path.join(base_path, "segmentation")
        self.uploads_path = os.path.join(base_path, "uploads")
        
        # Ensure directories exist
        os.makedirs(self.face_recognition_path, exist_ok=True)
        os.makedirs(self.segmentation_path, exist_ok=True)
        os.makedirs(self.uploads_path, exist_ok=True)
        
        logger.info(f"Persistent storage initialized at {base_path}")
    
    # Face Recognition Storage Methods
    def save_face_labels(self, user_id: str, labels_data: Dict[str, Any]) -> str:
        """Save face recognition labels for a user"""
        user_face_dir = os.path.join(self.face_recognition_path, user_id)
        os.makedirs(user_face_dir, exist_ok=True)
        
        labels_file = os.path.join(user_face_dir, "labels.json")
        
        # Add timestamp
        labels_data["last_updated"] = datetime.now().isoformat()
        
        with open(labels_file, 'w') as f:
            json.dump(labels_data, f, indent=2)
        
        logger.info(f"Saved face labels for user {user_id}")
        return labels_file
    
    def load_face_labels(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load face recognition labels for a user"""
        labels_file = os.path.join(self.face_recognition_path, user_id, "labels.json")
        
        if not os.path.exists(labels_file):
            return None
        
        try:
            with open(labels_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading face labels for user {user_id}: {str(e)}")
            return None
    
    def save_face_encodings(self, user_id: str, encodings_data: Dict[str, Any]) -> str:
        """Save face encodings for a user"""
        user_face_dir = os.path.join(self.face_recognition_path, user_id)
        os.makedirs(user_face_dir, exist_ok=True)
        
        encodings_file = os.path.join(user_face_dir, "encodings.pkl")
        
        # Add timestamp
        encodings_data["last_updated"] = datetime.now().isoformat()
        
        with open(encodings_file, 'wb') as f:
            pickle.dump(encodings_data, f)
        
        logger.info(f"Saved face encodings for user {user_id}")
        return encodings_file
    
    def load_face_encodings(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load face encodings for a user"""
        encodings_file = os.path.join(self.face_recognition_path, user_id, "encodings.pkl")
        
        if not os.path.exists(encodings_file):
            return None
        
        try:
            with open(encodings_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading face encodings for user {user_id}: {str(e)}")
            return None
    
    def save_extracted_frame(self, user_id: str, frame_data: bytes, frame_filename: str) -> str:
        """Save an extracted video frame"""
        user_frames_dir = os.path.join(self.face_recognition_path, user_id, "extracted_frames")
        os.makedirs(user_frames_dir, exist_ok=True)
        
        frame_path = os.path.join(user_frames_dir, frame_filename)
        
        with open(frame_path, 'wb') as f:
            f.write(frame_data)
        
        logger.info(f"Saved extracted frame {frame_filename} for user {user_id}")
        return frame_path
    
    # Segmentation Storage Methods
    def save_segmentation_data(self, user_id: str, segmentation_data: Dict[str, Any]) -> str:
        """Save segmentation results for a user"""
        user_seg_dir = os.path.join(self.segmentation_path, user_id)
        os.makedirs(user_seg_dir, exist_ok=True)
        
        segments_file = os.path.join(user_seg_dir, "segments.json")
        
        # Load existing data or create new
        if os.path.exists(segments_file):
            try:
                with open(segments_file, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = {"segments": [], "permissions": []}
        else:
            existing_data = {"segments": [], "permissions": []}
        
        # Merge new segmentation data
        if "segments" in segmentation_data:
            existing_data["segments"].extend(segmentation_data["segments"])
        
        existing_data["last_updated"] = datetime.now().isoformat()
        
        with open(segments_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Saved segmentation data for user {user_id}")
        return segments_file
    
    def load_segmentation_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load segmentation data for a user"""
        segments_file = os.path.join(self.segmentation_path, user_id, "segments.json")
        
        if not os.path.exists(segments_file):
            return {"segments": [], "permissions": []}
        
        try:
            with open(segments_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading segmentation data for user {user_id}: {str(e)}")
            return {"segments": [], "permissions": []}
    
    def save_permissions_data(self, user_id: str, permissions: List[Dict[str, Any]]) -> str:
        """Save permissions data for a user"""
        user_seg_dir = os.path.join(self.segmentation_path, user_id)
        os.makedirs(user_seg_dir, exist_ok=True)
        
        segments_file = os.path.join(user_seg_dir, "segments.json")
        
        # Load existing data
        existing_data = self.load_segmentation_data(user_id)
        existing_data["permissions"] = permissions
        existing_data["last_updated"] = datetime.now().isoformat()
        
        with open(segments_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Saved permissions data for user {user_id}")
        return segments_file
    
    def save_uploaded_file(self, user_id: str, file_data: bytes, filename: str) -> str:
        """Save an uploaded file"""
        user_uploads_dir = os.path.join(self.uploads_path, user_id)
        os.makedirs(user_uploads_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        file_path = os.path.join(user_uploads_dir, unique_filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        logger.info(f"Saved uploaded file {unique_filename} for user {user_id}")
        return file_path
    
    def get_user_files(self, user_id: str) -> Dict[str, List[str]]:
        """Get all files for a user"""
        files = {
            "face_recognition": [],
            "segmentation": [],
            "uploads": []
        }
        
        # Face recognition files
        user_face_dir = os.path.join(self.face_recognition_path, user_id)
        if os.path.exists(user_face_dir):
            for item in os.listdir(user_face_dir):
                files["face_recognition"].append(item)
        
        # Segmentation files
        user_seg_dir = os.path.join(self.segmentation_path, user_id)
        if os.path.exists(user_seg_dir):
            for item in os.listdir(user_seg_dir):
                files["segmentation"].append(item)
        
        # Upload files
        user_uploads_dir = os.path.join(self.uploads_path, user_id)
        if os.path.exists(user_uploads_dir):
            for item in os.listdir(user_uploads_dir):
                files["uploads"].append(item)
        
        return files
    
    def cleanup_old_files(self, days_old: int = 30):
        """Clean up files older than specified days"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error cleaning up file {file_path}: {str(e)}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "total_users": 0,
            "face_recognition_users": 0,
            "segmentation_users": 0,
            "total_size_mb": 0,
            "face_recognition_size_mb": 0,
            "segmentation_size_mb": 0,
            "uploads_size_mb": 0
        }
        
        try:
            # Count users and calculate sizes
            if os.path.exists(self.face_recognition_path):
                face_users = [d for d in os.listdir(self.face_recognition_path) if os.path.isdir(os.path.join(self.face_recognition_path, d))]
                stats["face_recognition_users"] = len(face_users)
                stats["face_recognition_size_mb"] = self._get_directory_size(self.face_recognition_path) / (1024 * 1024)
            
            if os.path.exists(self.segmentation_path):
                seg_users = [d for d in os.listdir(self.segmentation_path) if os.path.isdir(os.path.join(self.segmentation_path, d))]
                stats["segmentation_users"] = len(seg_users)
                stats["segmentation_size_mb"] = self._get_directory_size(self.segmentation_path) / (1024 * 1024)
            
            if os.path.exists(self.uploads_path):
                stats["uploads_size_mb"] = self._get_directory_size(self.uploads_path) / (1024 * 1024)
            
            stats["total_users"] = max(stats["face_recognition_users"], stats["segmentation_users"])
            stats["total_size_mb"] = stats["face_recognition_size_mb"] + stats["segmentation_size_mb"] + stats["uploads_size_mb"]
            
        except Exception as e:
            logger.error(f"Error calculating storage stats: {str(e)}")
        
        return stats
    
    def _get_directory_size(self, directory: str) -> int:
        """Calculate total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    pass
        return total_size