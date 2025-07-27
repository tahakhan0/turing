from pydantic import BaseModel
from typing import List, Tuple, Optional

class EnrollmentPayload(BaseModel):
    user_id: str
    video_path: str = None
    folder_path: str = None

    def model_post_init(self, __context):
        if not self.video_path and not self.folder_path:
            raise ValueError("Either 'video_path' or 'folder_path' must be provided.")
        if self.video_path and self.folder_path:
            raise ValueError("Provide either 'video_path' or 'folder_path', not both.")

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    bbox: BoundingBox
    confidence: float
    class_name: str
    person_name: Optional[str] = None  # For recognized persons
    recognition_confidence: Optional[float] = None  # Body-based recognition confidence
    person_id: Optional[int] = None  # Unique ID for each person in the frame (for labeling)
    detection_type: str = "person"  # "person" for YOLO person detection, "face" for face recognition

class FrameAnalysis(BaseModel):
    frame_number: int
    detections: List[Detection]
    visualization_url: Optional[str] = None
    visualization_urls: Optional[dict] = None  # Contains separate URLs for person, face, and combined views

class VideoAnalysis(BaseModel):
    video_path: str
    analysis: List[FrameAnalysis]

class RecognizedFace(BaseModel):
    name: str
    bbox: BoundingBox

class UnrecognizedFace(BaseModel):
    encoding: List[float]
    bbox: BoundingBox

class FaceInFrame(BaseModel):
    name: str
    bbox: BoundingBox
    recognition_status: str  # "recognized" or "unrecognized"
    person_name: Optional[str] = None
    face_encoding: Optional[List[float]] = None
    face_crop_url: Optional[str] = None  # URL to individual face crop image
    person_id: Optional[int] = None
    detection_type: Optional[str] = None

class FaceRecognitionFrame(BaseModel):
    frame_number: int
    timestamp: Optional[float] = None
    detections: List[FaceInFrame]
    visualization_url: Optional[str] = None

class FaceRecognitionAnalysis(BaseModel):
    video_path: str
    total_frames: int
    processed_frames: int
    recognized_faces: int
    unrecognized_faces: int
    detections: List[FaceRecognitionFrame]

class SaveFacePayload(BaseModel):
    user_id: str
    name: str
    encoding: List[float]

class SavePersonLabelPayload(BaseModel):
    user_id: str
    video_path: str
    frame_number: int
    bbox: BoundingBox
    person_name: str
    person_id: Optional[int] = None  # For identifying which person when multiple are present
    detection_type: str = "person"  # "person" for YOLO detection, "face" for face detection

class PersonLabel(BaseModel):
    person_id: int
    person_name: str
    bbox: BoundingBox
    detection_type: str = "person"

class BulkPersonLabelPayload(BaseModel):
    user_id: str
    video_path: str
    frame_number: int
    person_labels: List[PersonLabel]  # List of labels for multiple persons in the same frame