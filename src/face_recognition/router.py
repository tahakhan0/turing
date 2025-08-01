from fastapi import APIRouter, HTTPException
import os
import glob
import cv2
from .schemas import (
    EnrollmentPayload,
    SaveFacePayload,
    SavePersonLabelPayload,
    BulkPersonLabelPayload,
    VideoAnalysis,
    FaceRecognitionAnalysis
)
from .service import (
    save_person_label,
    save_face_encoding
)
from . import yolo_service
from ..storage.persistent_storage import PersistentStorage

# Initialize persistent storage
storage = PersistentStorage()

router = APIRouter()

@router.get("/health")
def health_check():
    """
    Health check endpoint to verify service availability.
    """
    return {"status": "healthy", "service": "face-recognition"}

@router.post("/enrollment", response_model=VideoAnalysis)
def dev_enrollment(payload: EnrollmentPayload):
    """
    Process a video or a folder of videos for person detection.
    """
    if payload.video_path:
        return yolo_service.batch_process_video_for_person_detection([payload.video_path], payload.user_id)
    elif payload.folder_path:
        # Get all video files from the folder
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_paths = []
        
        if os.path.exists(payload.folder_path):
            for file in os.listdir(payload.folder_path):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_paths.append(os.path.join(payload.folder_path, file))
        
        if not video_paths:
            raise HTTPException(status_code=404, detail=f"No video files found in folder: {payload.folder_path}")
            
        return yolo_service.batch_process_video_for_person_detection(video_paths, payload.user_id)

@router.post("/analyze", response_model=FaceRecognitionAnalysis)
def analyze_video(payload: EnrollmentPayload):
    """
    Analyze a video for both person and face recognition with body-based fallback.
    Identifies known faces and returns encodings for unknown faces.
    Uses face recognition first, then falls back to body-based recognition.
"""
    if payload.video_path:
        return yolo_service.analyze_video_with_enhanced_recognition(payload.video_path, payload.user_id)
    elif payload.folder_path:
        # For folder analysis, analyze the first video found (or implement batch analysis)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_paths = []
        
        if os.path.exists(payload.folder_path):
            for file in os.listdir(payload.folder_path):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_paths.append(os.path.join(payload.folder_path, file))
        
        if not video_paths:
            raise HTTPException(status_code=404, detail=f"No video files found in folder: {payload.folder_path}")
        
        # For now, analyze the first video. Could be extended to analyze all videos
        return yolo_service.analyze_video_with_enhanced_recognition(video_paths[0], payload.user_id)
    else:
        raise HTTPException(status_code=400, detail="Either 'video_path' or 'folder_path' must be provided.")


@router.post("/save-face")
def save_face(payload: SaveFacePayload):
    """
    Save the encoding of an unrecognized face with a name.
    """
    save_face_encoding(payload.user_id, payload.name, payload.encoding)
    return {"message": f"Face for {payload.name} saved successfully."}

@router.post("/save-person-label")
def save_person_label_endpoint(payload: SavePersonLabelPayload):
    """
    Save a person label for a detected person in a specific frame.
    Include person_id and detection_type for clarity when labeling multiple people.
    """
    label_entry = save_person_label(
        payload.user_id, 
        payload.video_path, 
        payload.frame_number, 
        payload.bbox.model_dump(), 
        payload.person_name
    )
    
    # Enhanced response with clearer identification
    person_identifier = f"Person ID {payload.person_id}" if payload.person_id is not None else "Person"
    detection_type_text = f" ({payload.detection_type} detection)" if payload.detection_type else ""
    
    return {
        "message": f"Label '{payload.person_name}' saved for {person_identifier}{detection_type_text}.",
        "person_id": payload.person_id,
        "detection_type": payload.detection_type,
        "person_name": payload.person_name,
        "label": label_entry
    }

@router.post("/save-bulk-person-labels")
def save_bulk_person_labels_endpoint(payload: BulkPersonLabelPayload):
    """
    Save multiple person labels for detected persons in a single frame.
    This allows labeling all persons in one image at once.
    Example payload:
    {
        "user_id": "user123",
        "video_path": "/path/to/video.mp4", 
        "frame_number": 25,
        "person_labels": [
            {"person_id": 0, "person_name": "John", "bbox": {...}, "detection_type": "person"},
            {"person_id": 1, "person_name": "Jane", "bbox": {...}, "detection_type": "person"}
        ]
    }
    """
    saved_labels = []
    
    for person_label in payload.person_labels:
        try:
            label_entry = save_person_label(
                payload.user_id,
                payload.video_path,
                payload.frame_number,
                person_label.bbox.model_dump(),
                person_label.person_name
            )
            saved_labels.append({
                "person_id": person_label.person_id,
                "person_name": person_label.person_name,
                "detection_type": person_label.detection_type,
                "status": "success",
                "label": label_entry
            })
        except Exception as e:
            saved_labels.append({
                "person_id": person_label.person_id,
                "person_name": person_label.person_name, 
                "detection_type": person_label.detection_type,
                "status": "error",
                "error": str(e)
            })
    
    successful_saves = [l for l in saved_labels if l["status"] == "success"]
    failed_saves = [l for l in saved_labels if l["status"] == "error"]
    
    return {
        "message": f"Bulk labeling completed. {len(successful_saves)} successful, {len(failed_saves)} failed.",
        "successful_labels": successful_saves,
        "failed_labels": failed_saves,
        "total_processed": len(payload.person_labels)
    }

@router.get("/frame-detections/{user_id}/{frame_number}")
def get_frame_detections(user_id: str, frame_number: int, video_path: str = None):
    """
    Get all detections for a specific frame to help with bulk labeling.
    Returns person_ids, bounding boxes, and detection types available for labeling.
    """
    if not video_path:
        raise HTTPException(status_code=400, detail="video_path parameter is required")
    
    try:
        # For now, we'll need to re-analyze the specific frame to get detection info
        # In a production system, you'd store this information in a database
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        # Jump to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=400, detail="Could not read frame from video")
        
        # Quick detection analysis (simplified version)
        from . import yolo_service
        results = yolo_service.yolo_model(frame)
        
        detections_info = []
        if results[0].boxes is not None:
            # Convert numpy arrays to Python native types immediately
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            person_id = 0
            for i, box in enumerate(boxes):
                class_name = yolo_service.yolo_model.names[int(class_ids[i])]
                if class_name == "person":
                    detections_info.append({
                        "person_id": person_id,
                        "bbox": {
                            "x1": int(box[0]),
                            "y1": int(box[1]), 
                            "x2": int(box[2]),
                            "y2": int(box[3])
                        },
                        "confidence": float(confidences[i].item()),
                        "detection_type": "person",
                        "class_name": "person"
                    })
                    person_id += 1
        
        return {
            "user_id": user_id,
            "frame_number": frame_number,
            "video_path": video_path,
            "detections": detections_info,
            "total_persons": len(detections_info)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")

@router.get("/images")
def list_images():
    """
    List all available visualization images.
    """
    images_dir = os.path.join(storage.base_path, "visualizations")
    if not os.path.exists(images_dir):
        return {"images": []}
    
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    images = []
    
    for filepath in image_files:
        filename = os.path.basename(filepath)
        # Extract user_id and frame info from filename like "user123_frame_25.jpg"
        parts = filename.replace('.jpg', '').split('_')
        if len(parts) >= 3:
            user_id = parts[0]
            frame_number = parts[2]
            url = f"/static/visualizations/{filename}"
            images.append({
                "filename": filename,
                "user_id": user_id,
                "frame_number": int(frame_number),
                "url": url,
                "full_url": f"http://localhost:8000{url}"
            })
    
    return {"images": sorted(images, key=lambda x: (x["user_id"], x["frame_number"]))}

@router.get("/encodings")
def list_encodings():
    """
    List all stored face encodings.
    """
    encodings_file = "/app/face_encodings/encodings.json"
    if not os.path.exists(encodings_file):
        return {"encodings": [], "message": "No encodings file found"}
    
    try:
        import json
        with open(encodings_file, "r") as f:
            data = json.load(f)
        
        encodings = []
        for i, (name, encoding) in enumerate(zip(data.get("names", []), data.get("encodings", []))):
            encodings.append({
                "id": i,
                "name": name,
                "encoding_length": len(encoding),
                "encoding": encoding  # Full encoding data
            })
        
        return {
            "total_encodings": len(encodings),
            "encodings": encodings
        }
    except Exception as e:
        return {"error": f"Failed to read encodings: {str(e)}"}

@router.get("/labels/{user_id}")
def get_person_labels(user_id: str):
    """
    Get all saved person labels for a specific user.
    """
    labels_data = storage.load_face_labels(user_id)
    
    if not labels_data:
        return {"labeled_faces": [], "message": f"No labels found for user {user_id}"}
    
    return {
        "user_id": user_id,
        "total_labels": len(labels_data.get("labeled_faces", [])),
        "labeled_faces": labels_data.get("labeled_faces", [])
    }


@router.post("/train-person")
def train_person_recognition(payload: dict):
    """
    Train person recognition from labeled detections.
    Payload should contain: user_id, person_name, and list of detection data.
    """
    user_id = payload.get("user_id")
    person_name = payload.get("person_name") 
    video_path = payload.get("video_path")
    frame_numbers = payload.get("frame_numbers", [])
    bboxes = payload.get("bboxes", [])
    
    if not all([user_id, person_name, video_path, frame_numbers, bboxes]):
        raise HTTPException(status_code=400, detail="Missing required fields: user_id, person_name, video_path, frame_numbers, bboxes")
    
    # Extract person images from video frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file")
    
    detection_images = []
    
    for frame_num, bbox in zip(frame_numbers, bboxes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Extract person region
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            person_img = frame[y1:y2, x1:x2]
            if person_img.size > 0:
                detection_images.append(person_img)
    
    cap.release()
    
    if not detection_images:
        raise HTTPException(status_code=400, detail="No valid person images extracted")
    
    # Train the person recognition
    yolo_service.train_person_from_detections(user_id, person_name, detection_images)
    
    return {
        "message": f"Successfully trained person recognition for {person_name}",
        "images_used": len(detection_images)
    }

@router.get("/person-embeddings/{user_id}")
def get_person_embeddings(user_id: str):
    """
    Get information about stored person embeddings for a user.
    """
    embeddings_info = yolo_service.get_person_embeddings_info(user_id)
    return {
        "user_id": user_id,
        "total_persons": len(embeddings_info),
        "persons": embeddings_info
    }

@router.post("/extract-frame")
def extract_frame(payload: dict):
    """
    Extract a single frame from video for segmentation interface compatibility
    """
    video_path = payload.get("video_path")
    frame_number = payload.get("frame_number", 1)
    user_id = payload.get("user_id")
    
    if not all([video_path, user_id]):
        raise HTTPException(status_code=400, detail="Missing required fields: video_path, user_id")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        # Extract frame using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        # Jump to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=400, detail="Cannot read frame from video")
        
        # Save frame to static directory using persistent storage
        frames_dir = os.path.join(storage.base_path, "extracted_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        frame_filename = f"{user_id}_frame_{frame_number}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        
        return {
            "image_path": frame_path,
            "frame_number": frame_number,
            "video_path": video_path,
            "user_id": user_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {str(e)}")