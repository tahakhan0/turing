from fastapi import APIRouter, HTTPException
import os
import glob
from .schemas import (
    EnrollmentPayload,
    SaveFacePayload,
    SavePersonLabelPayload,
    VideoAnalysis,
    FaceRecognitionAnalysis
)
from .service import (
    analyze_video_with_recognition,
    save_unrecognized_face,
    save_person_label
)
from .yolo_service import batch_process_video_for_person_detection

router = APIRouter()

@router.post("/enrollment", response_model=VideoAnalysis)
def dev_enrollment(payload: EnrollmentPayload):
    """
    Process a video or a folder of videos for person detection.
    """
    if payload.video_path:
        return batch_process_video_for_person_detection([payload.video_path], payload.user_id)
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
            
        return batch_process_video_for_person_detection(video_paths, payload.user_id)

@router.post("/analyze", response_model=FaceRecognitionAnalysis)
def analyze_video(payload: EnrollmentPayload):
    """
    Analyze a video for both person and face recognition.
    Identifies known faces and returns encodings for unknown faces.
    """
    if payload.video_path:
        return analyze_video_with_recognition(payload.video_path, payload.user_id)
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
        return analyze_video_with_recognition(video_paths[0], payload.user_id)
    else:
        raise HTTPException(status_code=400, detail="Either 'video_path' or 'folder_path' must be provided.")


@router.post("/save-face")
def save_face(payload: SaveFacePayload):
    """
    Save the encoding of an unrecognized face with a name.
    """
    save_unrecognized_face(payload.user_id, payload.name, payload.encoding)
    return {"message": f"Face for {payload.name} saved successfully."}

@router.post("/save-person-label")
def save_person_label_endpoint(payload: SavePersonLabelPayload):
    """
    Save a person label for a detected person in a specific frame.
    """
    label_entry = save_person_label(
        payload.user_id, 
        payload.video_path, 
        payload.frame_number, 
        payload.bbox.model_dump(), 
        payload.person_name
    )
    return {"message": f"Person label '{payload.person_name}' saved successfully.", "label": label_entry}

@router.get("/images")
def list_images():
    """
    List all available visualization images.
    """
    images_dir = "/app/static/visualizations"
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
    labels_file = f"/app/person_labels/{user_id}_labels.json"
    if not os.path.exists(labels_file):
        return {"labels": [], "message": f"No labels found for user {user_id}"}
    
    try:
        import json
        with open(labels_file, "r") as f:
            data = json.load(f)
        
        return {
            "user_id": user_id,
            "total_labels": len(data.get("labels", [])),
            "labels": data.get("labels", [])
        }
    except Exception as e:
        return {"error": f"Failed to read labels: {str(e)}"}