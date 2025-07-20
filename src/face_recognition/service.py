# This file is a renamed version of your original face_recognition_service.py
# The content remains the same, but it's now part of a module.
# Ensure the imports within this file are correct. For example, if it uses
# yolo_service, it should be: from . import yolo_service

import face_recognition
import cv2
import os
import json
import numpy as np
from .yolo_service import yolo_model # Relative import
from .schemas import FaceRecognitionAnalysis, RecognizedFace, UnrecognizedFace, BoundingBox, FaceRecognitionFrame, FaceInFrame

# Use absolute path for face encodings file
FACE_ENCODINGS_FILE = "/app/face_encodings/encodings.json"
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
VISUALIZATIONS_DIR = "/app/static/visualizations"

def load_known_faces():
    if not os.path.exists(FACE_ENCODINGS_FILE):
        return [], []
    with open(FACE_ENCODINGS_FILE, "r") as f:
        data = json.load(f)
    known_face_encodings = [np.array(e) for e in data["encodings"]]
    known_face_names = data["names"]
    return known_face_encodings, known_face_names

def save_unrecognized_face(user_id, name, encoding):
    os.makedirs(os.path.dirname(FACE_ENCODINGS_FILE), exist_ok=True)
    try:
        with open(FACE_ENCODINGS_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"names": [], "encodings": []}

    data["names"].append(name)
    data["encodings"].append(encoding)

    with open(FACE_ENCODINGS_FILE, "w") as f:
        json.dump(data, f)

def save_person_label(user_id, video_path, frame_number, bbox, person_name):
    """
    Save a person label for a detected person in a specific frame.
    This creates a record and extracts face encoding for future recognition.
    """
    # Create labels directory
    labels_dir = "/app/person_labels"
    os.makedirs(labels_dir, exist_ok=True)
    
    # Create user-specific labels file
    labels_file = os.path.join(labels_dir, f"{user_id}_labels.json")
    
    try:
        with open(labels_file, "r") as f:
            labels_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        labels_data = {"labels": []}
    
    # Try to extract face encoding from the video frame
    face_encoding = None
    try:
        # Load the video and extract the specific frame
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Calculate frame position (frame_number is 1-indexed in our system)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            
            if ret:
                # Extract person region using bounding box
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                person_region = frame[y1:y2, x1:x2]
                
                # Try to find face in the person region
                face_locations = face_recognition.face_locations(person_region)
                if face_locations:
                    face_encodings = face_recognition.face_encodings(person_region, face_locations)
                    if face_encodings:
                        face_encoding = face_encodings[0]  # Use the first face found
                        
                        # Save the face encoding to the main encodings file
                        save_unrecognized_face(user_id, person_name, face_encoding.tolist())
                        print(f"Successfully extracted and saved face encoding for {person_name}")
            
            cap.release()
    except Exception as e:
        print(f"Failed to extract face encoding: {str(e)}")
    
    # Create label entry
    label_entry = {
        "video_path": video_path,
        "frame_number": frame_number,
        "bbox": bbox,
        "person_name": person_name,
        "timestamp": cv2.getTickCount() / cv2.getTickFrequency(),  # Current time
        "face_encoding_saved": face_encoding is not None
    }
    
    # Check if this exact detection already exists (same video, frame, bbox)
    for existing_label in labels_data["labels"]:
        if (existing_label["video_path"] == video_path and 
            existing_label["frame_number"] == frame_number and
            existing_label["bbox"] == bbox):
            # Update existing label
            existing_label["person_name"] = person_name
            existing_label["timestamp"] = label_entry["timestamp"]
            existing_label["face_encoding_saved"] = label_entry["face_encoding_saved"]
            break
    else:
        # Add new label
        labels_data["labels"].append(label_entry)
    
    # Save updated labels
    with open(labels_file, "w") as f:
        json.dump(labels_data, f, indent=2)
    
    return label_entry

def save_individual_face_crop(frame, bbox, face_id, user_id, recognition_status, person_name=None):
    """
    Save an individual face crop for detailed viewing.
    """
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Add padding around the face
    padding = 20
    x1 = max(0, bbox["x1"] - padding)
    y1 = max(0, bbox["y1"] - padding)
    x2 = min(frame.shape[1], bbox["x2"] + padding)
    y2 = min(frame.shape[0], bbox["y2"] + padding)
    
    # Crop the face with padding
    face_crop = frame[y1:y2, x1:x2]
    
    # Create filename based on recognition status
    if recognition_status == "recognized" and person_name:
        filename = f"{user_id}_face_{face_id}_known_{person_name.replace(' ', '_')}.jpg"
    else:
        filename = f"{user_id}_face_{face_id}_unknown.jpg"
    
    filepath = os.path.join(VISUALIZATIONS_DIR, filename)
    cv2.imwrite(filepath, face_crop)
    
    return f"{BASE_URL}/static/visualizations/{filename}"

def create_face_visualization(frame, recognized_faces_frame, unrecognized_faces_frame, frame_number, user_id):
    """
    Create a visualization image showing recognized and unrecognized faces.
    """
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    vis_frame = frame.copy()
    
    # Draw recognized faces in green
    for face in recognized_faces_frame:
        bbox = face["bbox"]
        name = face["name"]
        cv2.rectangle(vis_frame, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), (0, 255, 0), 3)
        cv2.putText(vis_frame, f"KNOWN: {name}", (bbox["x1"], bbox["y1"] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw unrecognized faces in red
    for face in unrecognized_faces_frame:
        bbox = face["bbox"]
        cv2.rectangle(vis_frame, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), (0, 0, 255), 3)
        cv2.putText(vis_frame, "UNKNOWN", (bbox["x1"], bbox["y1"] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    filename = f"{user_id}_recognition_frame_{frame_number}.jpg"
    filepath = os.path.join(VISUALIZATIONS_DIR, filename)
    cv2.imwrite(filepath, vis_frame)
    
    return f"{BASE_URL}/static/visualizations/{filename}"

def frames_are_similar(frame1, frame2, threshold=0.9):
    """
    Check if two frames are similar using structural similarity.
    """
    import cv2
    from skimage.metrics import structural_similarity as ssim
    
    # Resize frames for faster comparison
    small_frame1 = cv2.resize(frame1, (100, 100))
    small_frame2 = cv2.resize(frame2, (100, 100))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(small_frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(small_frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate structural similarity
    similarity = ssim(gray1, gray2)
    
    return similarity > threshold

def analyze_video_with_recognition(video_path, user_id):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    known_face_encodings, known_face_names = load_known_faces()
    frame_analyses = []
    total_frames = 0
    processed_frames = 0
    total_recognized = 0
    total_unrecognized = 0
    processed_unrecognized_faces = []
    previous_frame = None
    
    # Process every 10th frame for performance (can be adjusted)
    frame_interval = 10
    face_id_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        total_frames += 1
        
        # Only process every Nth frame
        if total_frames % frame_interval != 0:
            continue
            
        # Skip similar frames to reduce duplicates
        if previous_frame is not None and frames_are_similar(previous_frame, frame, threshold=0.9):
            continue
            
        processed_frames += 1
        frame_faces = []
        frame_recognized_faces = []
        frame_unrecognized_faces = []

        # Person detection
        person_results = yolo_model(frame)
        if person_results[0].boxes is not None:
            person_boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int)
            person_classes = person_results[0].boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(person_boxes):
                if yolo_model.names[person_classes[i]] == "person":
                    x1, y1, x2, y2 = box
                    person_img = frame[y1:y2, x1:x2]

                    # Face detection within the person bounding box
                    face_locations = face_recognition.face_locations(person_img)
                    face_encodings = face_recognition.face_encodings(person_img, face_locations)

                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_top, face_right, face_bottom, face_left = face_location
                        bbox = BoundingBox(x1=x1 + face_left, y1=y1 + face_top, x2=x1 + face_right, y2=y1 + face_bottom)

                        if True in matches:
                            # Recognized face
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]
                            
                            # Generate individual face crop
                            face_crop_url = save_individual_face_crop(
                                frame, bbox.model_dump(), face_id_counter, user_id, "recognized", name
                            )
                            
                            face_in_frame = FaceInFrame(
                                name=name,
                                bbox=bbox,
                                recognition_status="recognized",
                                person_name=name,
                                face_crop_url=face_crop_url
                            )
                            frame_faces.append(face_in_frame)
                            frame_recognized_faces.append({"name": name, "bbox": bbox.model_dump()})
                            total_recognized += 1
                            face_id_counter += 1
                        else:
                            # Unrecognized face - check if it's truly new
                            is_new = True
                            for existing_encoding in processed_unrecognized_faces:
                                if np.allclose(existing_encoding, face_encoding, atol=0.6):
                                    is_new = False
                                    break
                            
                            if is_new:
                                processed_unrecognized_faces.append(face_encoding)
                                
                                # Generate individual face crop
                                face_crop_url = save_individual_face_crop(
                                    frame, bbox.model_dump(), face_id_counter, user_id, "unrecognized"
                                )
                                
                                face_in_frame = FaceInFrame(
                                    name="Unknown",
                                    bbox=bbox,
                                    recognition_status="unrecognized",
                                    person_name="",
                                    face_encoding=face_encoding.tolist(),
                                    face_crop_url=face_crop_url
                                )
                                frame_faces.append(face_in_frame)
                                frame_unrecognized_faces.append({"bbox": bbox.model_dump()})
                                total_unrecognized += 1
                                face_id_counter += 1

        # Create visualization if there are faces in this frame
        visualization_url = None
        if frame_faces:
            visualization_url = create_face_visualization(
                frame, frame_recognized_faces, frame_unrecognized_faces, total_frames, user_id
            )

        # Create frame analysis
        frame_analysis = FaceRecognitionFrame(
            frame_number=total_frames,
            timestamp=None,
            detections=frame_faces,
            visualization_url=visualization_url
        )
        frame_analyses.append(frame_analysis)
        
        # Store this frame for similarity comparison
        previous_frame = frame.copy()

    cap.release()
    
    return FaceRecognitionAnalysis(
        video_path=video_path,
        total_frames=total_frames,
        processed_frames=processed_frames,
        recognized_faces=total_recognized,
        unrecognized_faces=total_unrecognized,
        detections=frame_analyses
    )