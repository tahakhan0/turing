from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class MonitoringRequest(BaseModel):
    video_path: str
    user_id: str
    frame_interval: int = Field(default=30, ge=1, le=300)

class ViolationEvent(BaseModel):
    type: str  # "known_face_violation" or "unknown_face_in_segment"
    person_name: Optional[str] = None
    segment: Dict[str, Any]
    face_bbox: Dict[str, int]
    violation_reason: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: str
    frame_number: Optional[int] = None
    should_alert: bool = False

class UnknownDetection(BaseModel):
    type: str  # "unknown_face_in_segment"
    segment: Dict[str, Any]
    face_bbox: Dict[str, int]
    timestamp: str
    frame_number: Optional[int] = None
    face_encoding: Optional[List[float]] = None

class MonitoringResponse(BaseModel):
    status: str
    violations: List[ViolationEvent] = Field(default_factory=list)
    unknown_detections: List[UnknownDetection] = Field(default_factory=list)
    total_known_faces: Optional[int] = None
    total_unknown_faces: Optional[int] = None
    segments_checked: Optional[int] = None
    timestamp: str
    error: Optional[str] = None

class VideoAnalysisResponse(BaseModel):
    status: str
    video_path: str
    total_frames: int
    processed_frames: int
    violations: List[ViolationEvent] = Field(default_factory=list)
    unknown_detections: List[UnknownDetection] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class NotificationRequest(BaseModel):
    user_id: str
    violation: ViolationEvent
    frame_image_path: Optional[str] = None
    analysis_summary: Optional[str] = None