import face_recognition
import cv2
import os
import json
import numpy as np
from .yolo_service import yolo_model
from .schemas import FaceRecognitionAnalysis, RecognizedFace, UnrecognizedFace, BoundingBox, FaceRecognitionFrame, FaceInFrame
from ..storage.persistent_storage import PersistentStorage

# Initialize persistent storage
storage = PersistentStorage()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
VISUALIZATIONS_DIR = "/app/static/visualizations"

def load_known_faces(user_id):
    """Load known faces for a specific user from persistent storage"""
    encodings_data = storage.load_face_encodings(user_id)
    if not encodings_data:
        return [], []
    
    known_face_encodings = [np.array(e) for e in encodings_data.get("encodings", [])]
    known_face_names = encodings_data.get("names", [])
    return known_face_encodings, known_face_names

def save_face_encoding(user_id, name, encoding):
    """Save a face encoding for a specific user to persistent storage"""
    # Load existing encodings
    encodings_data = storage.load_face_encodings(user_id)
    if not encodings_data:
        encodings_data = {"names": [], "encodings": []}
    
    # Check if this person already exists (case-insensitive)
    for i, existing_name in enumerate(encodings_data["names"]):
        if existing_name.casefold() == name.casefold():
            # Update existing encoding
            encodings_data["encodings"][i] = encoding
            # Optionally, update the name to the new casing if you want to keep it consistent
            encodings_data["names"][i] = name
            break
    else:
        # Add new encoding
        encodings_data["names"].append(name)
        encodings_data["encodings"].append(encoding)
    
    # Save back to storage
    storage.save_face_encodings(user_id, encodings_data)

def save_person_label(user_id, video_path, frame_number, bbox, person_name, detection_type="person"):
    """
    Save a person or face label for a detected object in a specific frame.
    This creates a record and can extract face encoding for future recognition.
    """
    # Load existing labels from persistent storage
    labels_data = storage.load_face_labels(user_id)
    if not labels_data:
        labels_data = {"labeled_faces": []}
    
    # If it's a face detection, try to extract face encoding
    face_encoding = None
    if detection_type == "face":
        try:
            # Load the video and extract the specific frame
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                ret, frame = cap.read()
                
                if ret:
                    # Extract person region using bounding box
                    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                    face_region = frame[y1:y2, x1:x2]
                    
                    # Get face encoding from the face region
                    face_encodings = face_recognition.face_encodings(face_region)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        
                        # Save the face encoding to persistent storage
                        save_face_encoding(user_id, person_name, face_encoding.tolist())
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
        "detection_type": detection_type,
        "timestamp": cv2.getTickCount() / cv2.getTickFrequency(),  # Current time
        "face_encoding_saved": face_encoding is not None
    }
    
    # Check if this exact detection already exists (same video, frame, bbox)
    for existing_label in labels_data["labeled_faces"]:
        if (existing_label["video_path"] == video_path and 
            existing_label["frame_number"] == frame_number and
            existing_label["bbox"] == bbox):
            # Update existing label
            existing_label["person_name"] = person_name
            existing_label["detection_type"] = detection_type
            existing_label["timestamp"] = label_entry["timestamp"]
            existing_label["face_encoding_saved"] = label_entry["face_encoding_saved"]
            break
    else:
        # Add new label
        labels_data["labeled_faces"].append(label_entry)
    
    # Save updated labels to persistent storage
    storage.save_face_labels(user_id, labels_data)
    
    # Auto-update body embeddings if it's a person detection
    if detection_type == "person":
        try:
            # Extract person image for body embedding training
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                ret, frame = cap.read()
                if ret:
                    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                    person_image = frame[y1:y2, x1:x2]
                    
                    # Update body embeddings with this new person image
                    if person_image.size > 0:
                        # Import here to avoid circular import
                        from . import yolo_service
                        yolo_service.train_person_from_detections(user_id, person_name, [person_image])
                        print(f"Auto-updated body embeddings for {person_name}")
                
                cap.release()
        except Exception as e:
            print(f"Failed to auto-update body embeddings: {str(e)}")
    
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

    known_face_encodings, known_face_names = load_known_faces(user_id)
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
                                    name="",
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

def detect_faces_in_frame(frame):
    """
    Detect faces in a frame using face_recognition library.
    Returns a list of face locations.
    """
    try:
        print(f"detect_faces_in_frame: Input frame shape: {frame.shape}")
        # Convert BGR to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"detect_faces_in_frame: Found {len(face_locations)} face locations: {face_locations}")
        return face_locations
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return []

def recognize_faces_in_detections(frame, face_locations, user_id):
    """
    Recognize faces in detected face locations.
    Returns a list of Detection objects with face information.
    """
    try:
        detections = []
        if not face_locations:
            return detections
        
        # Load known faces
        known_face_encodings, known_face_names = load_known_faces(user_id)
        
        # Convert BGR to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get face encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            bbox = BoundingBox(x1=left, y1=top, x2=right, y2=bottom)
            
            # Create base detection
            detection = Detection(
                bbox=bbox,
                confidence=1.0,  # Face detection confidence is always high
                class_name="face"
            )
            
            # Try to recognize the face
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if matches and True in matches:
                    # Find the best match among only the valid matches (those that passed tolerance)
                    valid_indices = [i for i, match in enumerate(matches) if match]
                    valid_distances = [face_distances[i] for i in valid_indices]
                    
                    if valid_distances:
                        best_valid_index = valid_indices[np.argmin(valid_distances)]
                        person_name = known_face_names[best_valid_index]
                        recognition_confidence = 1.0 - face_distances[best_valid_index]  # Convert distance to confidence
                        
                        detection.person_name = person_name
                        detection.recognition_confidence = float(recognition_confidence)
                        detection.detection_type = "face"
                    else:
                        detection.person_name = None
                        detection.recognition_confidence = 0.0
                        detection.detection_type = "face"
                else:
                    detection.person_name = None
                    detection.recognition_confidence = 0.0
                    detection.detection_type = "face"
            else:
                detection.person_name = None
                detection.recognition_confidence = 0.0
                detection.detection_type = "face"
            
            detections.append(detection)
        
        return detections
        
    except Exception as e:
        print(f"Error recognizing faces: {e}")
        return []