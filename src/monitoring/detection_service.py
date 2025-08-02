"""
Real-time Detection Service
Monitors video feeds to detect known/unknown faces within defined segments
and checks access permissions.
"""

import cv2
import numpy as np
import face_recognition
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from ..face_recognition.service import load_known_faces
from ..segmentation.segment_manager import SegmentManager
from ..storage.persistent_storage import PersistentStorage
from ..face_recognition.yolo_service import yolo_model

logger = logging.getLogger(__name__)

class DetectionService:
    """Service for real-time face detection within segments"""
    
    def __init__(self):
        self.segment_manager = SegmentManager()
        self.storage = PersistentStorage()
        self.detection_history = {}  # Cache recent detections to avoid spam
        
    def _is_point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _get_face_center(self, bbox: Dict[str, int]) -> Tuple[int, int]:
        """Get center point of face bounding box"""
        center_x = (bbox["x1"] + bbox["x2"]) // 2
        center_y = (bbox["y1"] + bbox["y2"]) // 2
        return (center_x, center_y)
    
    def _is_point_in_bbox(self, point: Tuple[int, int], bbox: Dict[str, int]) -> bool:
        """Check if a point is inside a bounding box"""
        x, y = point
        return (bbox["x1"] <= x <= bbox["x2"]) and (bbox["y1"] <= y <= bbox["y2"])
    
    def _find_segment_for_face(self, face_center: Tuple[int, int], user_segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find which segment contains the face center point"""
        for segment in user_segments:
            # Check polygon first (if available), then fallback to bbox
            if "polygon" in segment and segment["polygon"]:
                if self._is_point_in_polygon(face_center, segment["polygon"]):
                    return segment
            elif "bbox" in segment and segment["bbox"]:
                if self._is_point_in_bbox(face_center, segment["bbox"]):
                    return segment
        return None
    
    def _detect_faces_in_frame(self, frame: np.ndarray, user_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect known and unknown faces in a frame"""
        known_face_encodings, known_face_names = load_known_faces(user_id)
        known_faces = []
        unknown_faces = []
        
        # Person detection first
        person_results = yolo_model(frame)
        if person_results[0].boxes is not None:
            person_boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int)
            person_classes = person_results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i, box in enumerate(person_boxes):
                if yolo_model.names[person_classes[i]] == "person":
                    x1, y1, x2, y2 = box
                    person_img = frame[y1:y2, x1:x2]
                    
                    # Face detection within person region
                    face_locations = face_recognition.face_locations(person_img)
                    face_encodings = face_recognition.face_encodings(person_img, face_locations)
                    
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        face_top, face_right, face_bottom, face_left = face_location
                        
                        # Convert relative coordinates to absolute
                        abs_bbox = {
                            "x1": x1 + face_left,
                            "y1": y1 + face_top,
                            "x2": x1 + face_right,
                            "y2": y1 + face_bottom
                        }
                        
                        face_center = self._get_face_center(abs_bbox)
                        
                        # Check if known face
                        if known_face_encodings:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                            if True in matches:
                                match_index = matches.index(True)
                                person_name = known_face_names[match_index]
                                
                                known_faces.append({
                                    "person_name": person_name,
                                    "bbox": abs_bbox,
                                    "face_center": face_center,
                                    "confidence": 1.0 - np.min(face_recognition.face_distance([known_face_encodings[match_index]], face_encoding))
                                })
                            else:
                                unknown_faces.append({
                                    "bbox": abs_bbox,
                                    "face_center": face_center,
                                    "face_encoding": face_encoding.tolist()
                                })
                        else:
                            unknown_faces.append({
                                "bbox": abs_bbox,
                                "face_center": face_center,
                                "face_encoding": face_encoding.tolist()
                            })
        
        return known_faces, unknown_faces
    
    def _check_access_violation(self, person_name: str, segment: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Check if person's presence in segment is a violation"""
        access_result = self.segment_manager.check_access(
            person_name,
            user_id,
            segment["area_id"],
            {"timestamp": datetime.now().isoformat()}
        )
        
        return {
            "is_violation": not access_result["allowed"],
            "reason": access_result["reason"],
            "area_type": segment["area_type"],
            "area_id": segment["area_id"]
        }
    
    def _should_alert(self, person_name: str, area_id: str, cooldown_minutes: int = 5) -> bool:
        """Check if we should send an alert (avoid spam with cooldown)"""
        now = datetime.now()
        key = f"{person_name}_{area_id}"
        
        if key in self.detection_history:
            last_alert = self.detection_history[key]
            time_diff = (now - last_alert).total_seconds() / 60
            if time_diff < cooldown_minutes:
                return False
        
        self.detection_history[key] = now
        return True
    
    def analyze_frame_for_violations(self, frame: np.ndarray, user_id: str) -> Dict[str, Any]:
        """
        Analyze a single frame for access violations
        
        Returns:
            Dict containing violations, unknown faces, and frame analysis
        """
        try:
            # Get user's segments
            user_segments = self.segment_manager.get_user_segments(user_id)
            if not user_segments:
                return {
                    "status": "no_segments",
                    "message": "No segments defined for user",
                    "violations": [],
                    "unknown_detections": []
                }
            
            # Detect faces in frame
            known_faces, unknown_faces = self._detect_faces_in_frame(frame, user_id)
            
            violations = []
            unknown_detections = []
            
            # Check known faces for access violations
            for face in known_faces:
                face_center = face["face_center"]
                segment = self._find_segment_for_face(face_center, user_segments)
                
                if segment:
                    violation_check = self._check_access_violation(
                        face["person_name"], 
                        segment, 
                        user_id
                    )
                    
                    if violation_check["is_violation"]:
                        # Only alert if cooldown period has passed
                        if self._should_alert(face["person_name"], segment["area_id"]):
                            violations.append({
                                "type": "known_face_violation",
                                "person_name": face["person_name"],
                                "segment": segment,
                                "face_bbox": face["bbox"],
                                "violation_reason": violation_check["reason"],
                                "confidence": face["confidence"],
                                "timestamp": datetime.now().isoformat(),
                                "should_alert": True
                            })
                        else:
                            violations.append({
                                "type": "known_face_violation",
                                "person_name": face["person_name"],
                                "segment": segment,
                                "face_bbox": face["bbox"],
                                "violation_reason": violation_check["reason"],
                                "confidence": face["confidence"],
                                "timestamp": datetime.now().isoformat(),
                                "should_alert": False
                            })
            
            # Check unknown faces in segments - all unknown persons trigger alerts (non-residents not allowed)
            for face in unknown_faces:
                face_center = face["face_center"]
                segment = self._find_segment_for_face(face_center, user_segments)
                
                if segment:
                    # Always alert for unknown persons since non-residents don't have access
                    alert_key = f"unknown_{segment['area_id']}"
                    should_alert = self._should_alert("unknown_person", segment["area_id"], cooldown_minutes=2)
                    
                    unknown_detections.append({
                        "type": "unauthorized_unknown_person",
                        "person_name": "Unknown Person",
                        "segment": segment,
                        "face_bbox": face["bbox"],
                        "violation_reason": "Non-residents not permitted - immediate action required",
                        "timestamp": datetime.now().isoformat(),
                        "face_encoding": face["face_encoding"],
                        "should_alert": should_alert
                    })
            
            return {
                "status": "success",
                "violations": violations,
                "unknown_detections": unknown_detections,
                "total_known_faces": len(known_faces),
                "total_unknown_faces": len(unknown_faces),
                "segments_checked": len(user_segments),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing frame for violations: {e}")
            return {
                "status": "error",
                "error": str(e),
                "violations": [],
                "unknown_detections": []
            }
    
    def analyze_video_for_violations(self, video_path: str, user_id: str, 
                                   frame_interval: int = 30) -> Dict[str, Any]:
        """
        Analyze video file for access violations
        
        Args:
            video_path: Path to video file
            user_id: User ID
            frame_interval: Process every Nth frame
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            all_violations = []
            all_unknown_detections = []
            frame_count = 0
            processed_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every Nth frame
                if frame_count % frame_interval == 0:
                    processed_frames += 1
                    frame_analysis = self.analyze_frame_for_violations(frame, user_id)
                    
                    if frame_analysis["status"] == "success":
                        # Add frame number to each detection
                        for violation in frame_analysis["violations"]:
                            violation["frame_number"] = frame_count
                            all_violations.append(violation)
                        
                        for detection in frame_analysis["unknown_detections"]:
                            detection["frame_number"] = frame_count
                            all_unknown_detections.append(detection)
            
            cap.release()
            
            return {
                "status": "success",
                "video_path": video_path,
                "total_frames": frame_count,
                "processed_frames": processed_frames,
                "violations": all_violations,
                "unknown_detections": all_unknown_detections,
                "summary": {
                    "total_violations": len(all_violations),
                    "total_unknown_detections": len(all_unknown_detections),
                    "unique_violators": len(set(v["person_name"] for v in all_violations if v["type"] == "known_face_violation"))
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video for violations: {e}")
            return {
                "status": "error",
                "error": str(e),
                "violations": [],
                "unknown_detections": []
            }