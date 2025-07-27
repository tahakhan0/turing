from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import requests

from .schemas import SegmentedArea, ResidentPermission, AreaType, PermissionCondition
from ..storage.persistent_storage import PersistentStorage

logger = logging.getLogger(__name__)

class SegmentManager:
    """Manages segmented areas and resident permissions"""
    
    def __init__(self, face_recognition_base_url: str = "http://localhost:8000"):
        self.segments: Dict[str, SegmentedArea] = {}
        self.permissions: List[ResidentPermission] = []
        self.face_recognition_base_url = face_recognition_base_url
        self.storage = PersistentStorage()
        self._users_loaded = set()  # Track which users' data has been loaded
    
    def add_segment(self, segment_data: Dict[str, Any], user_id: str) -> str:
        """Add a new segment from segmentation results"""
        try:
            # Load user data if not already loaded
            self._load_user_data(user_id)
            
            # Convert area_type string back to enum
            area_type = AreaType(segment_data['area_type'])
            
            segment = SegmentedArea(
                area_id=segment_data['area_id'],
                area_type=area_type,
                polygon=segment_data['polygon'],
                confidence=segment_data['confidence'],
                dimensions=segment_data['dimensions'],
                user_id=user_id,
                verified=segment_data.get('verified', False)
            )
            
            self.segments[segment.area_id] = segment
            self._save_user_data(user_id)
            
            return segment.area_id
            
        except Exception as e:
            logger.error(f"Error adding segment: {str(e)}")
            raise
    
    def get_segment(self, area_id: str) -> Optional[Dict[str, Any]]:
        """Get a segment by ID"""
        if area_id not in self.segments:
            return None
        
        segment = self.segments[area_id]
        return {
            "area_id": segment.area_id,
            "area_type": segment.area_type.value,
            "polygon": segment.polygon,
            "confidence": segment.confidence,
            "dimensions": segment.dimensions,
            "user_id": segment.user_id,
            "verified": segment.verified,
            "created_at": segment.created_at.isoformat(),
            "updated_at": segment.updated_at.isoformat()
        }
    
    def verify_segment(self, area_id: str, user_id: str, approved: bool) -> Dict[str, Any]:
        """User verification of segmented area"""
        # Load user data if not already loaded
        self._load_user_data(user_id)
        
        if area_id not in self.segments:
            return {"error": "Segment not found"}
        
        segment = self.segments[area_id]
        if segment.user_id != user_id:
            return {"error": "Unauthorized: Segment belongs to different user"}
        
        if approved:
            segment.verified = True
            segment.updated_at = datetime.now()
            self._save_user_data(user_id)
            return {"status": "success", "message": "Segment verified"}
        else:
            # Remove unverified segment
            del self.segments[area_id]
            # Also remove any permissions for this area
            self.permissions = [p for p in self.permissions if p.area_id != area_id]
            self._save_user_data(user_id)
            return {"status": "success", "message": "Segment rejected and removed"}
    
    def get_user_segments(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all segments for a specific user"""
        # Load user data if not already loaded
        self._load_user_data(user_id)
        
        user_segments = []
        for segment in self.segments.values():
            if segment.user_id == user_id:
                user_segments.append({
                    "area_id": segment.area_id,
                    "area_type": segment.area_type.value,
                    "polygon": segment.polygon,
                    "confidence": segment.confidence,
                    "dimensions": segment.dimensions,
                    "verified": segment.verified,
                    "created_at": segment.created_at.isoformat(),
                    "updated_at": segment.updated_at.isoformat()
                })
        return user_segments
    
    def _load_user_data(self, user_id: str):
        """Load segments and permissions for a user from persistent storage"""
        if user_id in self._users_loaded:
            return
        
        # Load segmentation data
        seg_data = self.storage.load_segmentation_data(user_id)
        
        # Load segments
        for segment_data in seg_data.get("segments", []):
            try:
                area_type = AreaType(segment_data['area_type'])
                segment = SegmentedArea(
                    area_id=segment_data['area_id'],
                    area_type=area_type,
                    polygon=segment_data['polygon'],
                    confidence=segment_data['confidence'],
                    dimensions=segment_data['dimensions'],
                    user_id=user_id,
                    verified=segment_data.get('verified', False),
                    created_at=datetime.fromisoformat(segment_data.get('created_at', datetime.now().isoformat())),
                    updated_at=datetime.fromisoformat(segment_data.get('updated_at', datetime.now().isoformat()))
                )
                self.segments[segment.area_id] = segment
            except Exception as e:
                logger.error(f"Error loading segment {segment_data.get('area_id', 'unknown')}: {str(e)}")
        
        # Load permissions
        for perm_data in seg_data.get("permissions", []):
            try:
                conditions = [PermissionCondition(c) for c in perm_data.get('conditions', [])]
                permission = ResidentPermission(
                    person_name=perm_data['person_name'],
                    user_id=perm_data['user_id'],
                    area_id=perm_data['area_id'],
                    allowed=perm_data['allowed'],
                    conditions=conditions,
                    created_at=datetime.fromisoformat(perm_data.get('created_at', datetime.now().isoformat()))
                )
                self.permissions.append(permission)
            except Exception as e:
                logger.error(f"Error loading permission: {str(e)}")
        
        self._users_loaded.add(user_id)
        logger.info(f"Loaded data for user {user_id}")
    
    def _save_user_data(self, user_id: str):
        """Save segments and permissions for a user to persistent storage"""
        # Prepare segments data
        user_segments = []
        for segment in self.segments.values():
            if segment.user_id == user_id:
                user_segments.append({
                    "area_id": segment.area_id,
                    "area_type": segment.area_type.value,
                    "polygon": segment.polygon,
                    "confidence": segment.confidence,
                    "dimensions": segment.dimensions,
                    "user_id": segment.user_id,
                    "verified": segment.verified,
                    "created_at": segment.created_at.isoformat(),
                    "updated_at": segment.updated_at.isoformat()
                })
        
        # Prepare permissions data
        user_permissions = []
        for permission in self.permissions:
            if permission.user_id == user_id:
                user_permissions.append({
                    "person_name": permission.person_name,
                    "user_id": permission.user_id,
                    "area_id": permission.area_id,
                    "allowed": permission.allowed,
                    "conditions": [c.value for c in permission.conditions],
                    "created_at": permission.created_at.isoformat()
                })
        
        # Save to storage
        segmentation_data = {
            "segments": user_segments,
            "permissions": user_permissions
        }
        
        self.storage.save_segmentation_data(user_id, segmentation_data)
        logger.info(f"Saved data for user {user_id}")
    
    def get_labeled_people(self, user_id: str) -> List[str]:
        """Fetch labeled faces from face recognition service to get list of known people"""
        try:
            response = requests.get(f"{self.face_recognition_base_url}/face-recognition/labels/{user_id}")
            if response.status_code == 200:
                labels_data = response.json()
                # Extract unique person names from labeled faces
                people = set()
                for face_data in labels_data.get('labeled_faces', []):
                    person_name = face_data.get('person_name')
                    if person_name and person_name.strip():
                        people.add(person_name.strip())
                return list(people)
            else:
                logger.warning(f"Failed to fetch labels for user {user_id}: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching labeled people: {str(e)}")
            return []
    
    def is_labeled_person(self, person_name: str, user_id: str) -> bool:
        """Check if person_name is a labeled person for the user"""
        labeled_people = self.get_labeled_people(user_id)
        return person_name in labeled_people
    
    def add_person_permission(self, person_name: str, user_id: str, area_id: str, allowed: bool, 
                              conditions: List[str] = None) -> Dict[str, Any]:
        """Add access permission for a labeled person to a specific area or all areas"""
        # Load user data if not already loaded
        self._load_user_data(user_id)
        
        if area_id != "all" and area_id not in self.segments:
            return {"error": "Area not found"}
        
        # Validate that person_name is a labeled person for this user
        if not self.is_labeled_person(person_name, user_id):
            return {"error": f"'{person_name}' is not a labeled person for user {user_id}"}
        
        # Convert condition strings to enums
        condition_enums = []
        if conditions:
            for condition_str in conditions:
                try:
                    condition_enums.append(PermissionCondition(condition_str))
                except ValueError:
                    logger.warning(f"Unknown permission condition: {condition_str}")
        
        permission = ResidentPermission(
            person_name=person_name,
            user_id=user_id,
            area_id=area_id,
            allowed=allowed,
            conditions=condition_enums
        )
        
        # Remove existing permission for this person-area combination
        self.permissions = [p for p in self.permissions 
                          if not (p.person_name == person_name and p.user_id == user_id and p.area_id == area_id)]
        
        self.permissions.append(permission)
        self._save_user_data(user_id)
        
        return {"status": "success", "message": "Permission added"}
    
    def check_access(self, person_name: str, user_id: str, area_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check if a labeled person has access to a specific area"""
        # Load user data if not already loaded
        self._load_user_data(user_id)
        
        if area_id not in self.segments:
            return {"allowed": False, "reason": "Area not found"}
        
        segment = self.segments[area_id]
        if not segment.verified:
            return {"allowed": False, "reason": "Area not verified"}
        
        # Only check access for segments belonging to this user
        if segment.user_id != user_id:
            return {"allowed": False, "reason": "Area belongs to different user"}
        
        # First check for "all" areas permission
        all_permission = next((p for p in self.permissions 
                              if p.person_name == person_name and p.user_id == user_id and p.area_id == "all"), None)
        
        # Then check for specific area permission
        specific_permission = next((p for p in self.permissions 
                                  if p.person_name == person_name and p.user_id == user_id and p.area_id == area_id), None)
        
        # Use specific permission if available, otherwise use "all" permission
        permission = specific_permission or all_permission
        
        if not permission:
            return {"allowed": False, "reason": "No permission found"}
        
        if not permission.allowed:
            return {"allowed": False, "reason": "Access denied"}
        
        # Check conditions
        context = context or {}
        for condition in permission.conditions:
            if condition == PermissionCondition.ADULT_SUPERVISION_REQUIRED:
                if not context.get("adult_present", False):
                    return {"allowed": False, "reason": "Adult supervision required"}
            elif condition == PermissionCondition.DAYLIGHT_ONLY:
                if not context.get("daylight", True):
                    return {"allowed": False, "reason": "Access only allowed during daylight"}
            elif condition == PermissionCondition.WEEKENDS_ONLY:
                if not context.get("is_weekend", False):
                    return {"allowed": False, "reason": "Access only allowed on weekends"}
        
        return {"allowed": True, "reason": "Access granted"}
    
    def get_person_permissions(self, person_name: str, user_id: str) -> List[Dict[str, Any]]:
        """Get all permissions for a labeled person"""
        person_permissions = []
        for permission in self.permissions:
            if permission.person_name == person_name and permission.user_id == user_id:
                # Get area info
                area_info = None
                if permission.area_id in self.segments:
                    segment = self.segments[permission.area_id]
                    area_info = {
                        "area_type": segment.area_type.value,
                        "verified": segment.verified
                    }
                
                person_permissions.append({
                    "person_name": permission.person_name,
                    "user_id": permission.user_id,
                    "area_id": permission.area_id,
                    "allowed": permission.allowed,
                    "conditions": [condition.value for condition in permission.conditions],
                    "area_info": area_info,
                    "created_at": permission.created_at.isoformat()
                })
        
        return person_permissions
    
    def get_area_permissions(self, area_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get all permissions for a specific area"""
        area_permissions = []
        for permission in self.permissions:
            if permission.area_id == area_id and permission.user_id == user_id:
                area_permissions.append({
                    "person_name": permission.person_name,
                    "user_id": permission.user_id,
                    "area_id": permission.area_id,
                    "allowed": permission.allowed,
                    "conditions": [condition.value for condition in permission.conditions],
                    "created_at": permission.created_at.isoformat()
                })
        
        return area_permissions
    
    def remove_permission(self, person_name: str, user_id: str, area_id: str) -> Dict[str, Any]:
        """Remove a specific permission"""
        # Load user data if not already loaded
        self._load_user_data(user_id)
        
        initial_count = len(self.permissions)
        self.permissions = [p for p in self.permissions 
                          if not (p.person_name == person_name and p.user_id == user_id and p.area_id == area_id)]
        
        if len(self.permissions) < initial_count:
            self._save_user_data(user_id)
            return {"status": "success", "message": "Permission removed"}
        else:
            return {"error": "Permission not found"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_segments = len(self.segments)
        verified_segments = sum(1 for s in self.segments.values() if s.verified)
        total_permissions = len(self.permissions)
        
        # Count by area type
        area_type_counts = {}
        for segment in self.segments.values():
            area_type = segment.area_type.value
            area_type_counts[area_type] = area_type_counts.get(area_type, 0) + 1
        
        return {
            "total_segments": total_segments,
            "verified_segments": verified_segments,
            "unverified_segments": total_segments - verified_segments,
            "total_permissions": total_permissions,
            "area_type_distribution": area_type_counts,
            "unique_users": len(set(s.user_id for s in self.segments.values())),
            "unique_people_with_permissions": len(set(f"{p.person_name}_{p.user_id}" for p in self.permissions))
        }