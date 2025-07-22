from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
import face_recognition  # Import the face-recognition library
from .schemas import VideoAnalysis, FrameAnalysis, Detection, BoundingBox, FaceRecognitionAnalysis, FaceRecognitionFrame, FaceInFrame
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from .service import load_known_faces, save_individual_face_crop, create_face_visualization

# Get the base URL for serving images (can be overridden with environment variable)
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Initialize the YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# --- PERFORMANCE OPTIMIZATION ---
# Process every Nth frame to speed up video analysis.
FRAME_INTERVAL = 5

# Directory for storing visualization images
VISUALIZATIONS_DIR = "/app/static/visualizations"

# Directory for storing person embeddings (body-based features)
PERSON_EMBEDDINGS_DIR = "/app/person_embeddings"

# Temporal tracking for consistency across frames
class PersonTracker:
    def __init__(self, max_history=5, confidence_boost=0.1):
        self.person_history = {}  # frame_number -> {person_id: (name, confidence, bbox)}
        self.max_history = max_history
        self.confidence_boost = confidence_boost
    
    def add_detection(self, frame_number, person_id, person_name, confidence, bbox):
        """Add a person detection to the history."""
        if frame_number not in self.person_history:
            self.person_history[frame_number] = {}
        
        self.person_history[frame_number][person_id] = {
            'name': person_name,
            'confidence': confidence,
            'bbox': bbox
        }
        
        # Clean old history
        frames_to_remove = [f for f in self.person_history.keys() 
                           if frame_number - f > self.max_history * FRAME_INTERVAL]
        for f in frames_to_remove:
            del self.person_history[f]
    
    def get_temporal_confidence(self, frame_number, person_name, current_confidence, bbox):
        """
        Get boosted confidence based on recent detections of the same person.
        Considers spatial proximity and name consistency.
        """
        if not self.person_history:
            return current_confidence
        
        # Look at recent frames
        recent_frames = [f for f in self.person_history.keys() 
                        if 0 < frame_number - f <= self.max_history * FRAME_INTERVAL]
        
        if not recent_frames:
            return current_confidence
        
        # Count recent detections of the same person name
        recent_detections = 0
        spatial_matches = 0
        
        for frame in recent_frames:
            for detection in self.person_history[frame].values():
                if detection['name'] == person_name:
                    recent_detections += 1
                    
                    # Check spatial proximity (same person should be in similar location)
                    prev_bbox = detection['bbox']
                    bbox_overlap = calculate_bbox_overlap(bbox, prev_bbox)
                    if bbox_overlap > 0.3:  # 30% overlap threshold
                        spatial_matches += 1
        
        # Apply temporal boost based on consistency
        temporal_boost = 0
        if recent_detections > 0:
            consistency_ratio = spatial_matches / recent_detections
            temporal_boost = self.confidence_boost * consistency_ratio * min(recent_detections / 3, 1.0)
        
        return min(1.0, current_confidence + temporal_boost)


def calculate_bbox_overlap(bbox1, bbox2):
    """
    Calculate intersection over union (IoU) of two bounding boxes.
    bbox format: BoundingBox object with x1, y1, x2, y2
    """
    # Calculate intersection area
    x1 = max(bbox1.x1, bbox2.x1)
    y1 = max(bbox1.y1, bbox2.y1)
    x2 = min(bbox1.x2, bbox2.x2)
    y2 = min(bbox1.y2, bbox2.y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
    area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def extract_person_features(person_image):
    """
    Extract enhanced features from a person's body image for recognition.
    Uses histogram-based color features, shape features, texture, and body proportions.
    Designed to work better in varying lighting conditions.
    Returns a feature vector that can be used for person matching.
    """
    if person_image.size == 0:
        return np.zeros(289)  # Return zero vector for empty images
    
    # Resize image to standard size for consistent feature extraction
    resized = cv2.resize(person_image, (64, 128))
    
    # --- Color Features ---
    # Extract color histogram features (RGB)
    hist_r = cv2.calcHist([resized], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([resized], [2], None, [32], [0, 256])
    
    # Extract HSV histogram for better color representation (more robust to lighting)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    
    # --- Shape and Structure Features ---
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Enhanced edge detection with multiple parameters for robustness
    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
    
    # Body proportion features
    height, width = resized.shape[:2]
    aspect_ratio = width / height
    
    # --- Texture Features ---
    # Local Binary Pattern-like features for texture analysis
    gray_normalized = cv2.equalizeHist(gray)  # Normalize for lighting
    
    # Calculate simple texture measures in different regions
    # Upper body region (torso/clothing patterns)
    upper_region = gray_normalized[:int(height*0.6), :]
    upper_texture = np.std(upper_region) if upper_region.size > 0 else 0
    
    # Lower body region (legs/pants)
    lower_region = gray_normalized[int(height*0.6):, :]
    lower_texture = np.std(lower_region) if lower_region.size > 0 else 0
    
    # --- Lighting-Robust Features ---
    # Use relative brightness rather than absolute values
    mean_brightness = np.mean(gray)
    brightness_std = np.std(gray)
    
    # Dominant color in clothing regions (less affected by lighting than face)
    # Focus on middle torso area which is typically clothing
    torso_region = resized[int(height*0.2):int(height*0.7), :]
    if torso_region.size > 0:
        torso_hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        dominant_hue = np.median(torso_hsv[:,:,0])
        dominant_sat = np.median(torso_hsv[:,:,1])
    else:
        dominant_hue = 0
        dominant_sat = 0
    
    # --- Body Silhouette Features ---
    # Simple contour-based features for body shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (presumably the person)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape descriptors
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else 0
        
        # Bounding rectangle features
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_aspect = w / h if h > 0 else 0
        fill_ratio = area / (w * h) if (w * h) > 0 else 0
    else:
        compactness = 0
        rect_aspect = 0
        fill_ratio = 0
    
    # Combine all features into a single vector
    features = np.concatenate([
        hist_r.flatten(),      # 32 features
        hist_g.flatten(),      # 32 features 
        hist_b.flatten(),      # 32 features
        hist_h.flatten(),      # 32 features
        hist_s.flatten(),      # 32 features
        hist_v.flatten(),      # 32 features
        edge_hist.flatten(),   # 16 features
        [aspect_ratio,         # Body proportions
         upper_texture,        # Upper body texture
         lower_texture,        # Lower body texture
         mean_brightness,      # Lighting features
         brightness_std,
         dominant_hue,         # Clothing color features
         dominant_sat,
         compactness,          # Shape features
         rect_aspect,
         fill_ratio,
         area / (height * width)]  # Relative body size
    ])
    
    # Normalize features with numerical stability
    norm = np.linalg.norm(features)
    if norm > 1e-8:
        features = features / norm
    
    return features


def load_person_embeddings(user_id):
    """
    Load stored person embeddings for a user.
    Returns a dictionary with person names as keys and feature vectors as values.
    """
    embeddings_file = os.path.join(PERSON_EMBEDDINGS_DIR, f"{user_id}_person_embeddings.json")
    
    if not os.path.exists(embeddings_file):
        return {}
    
    try:
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        embeddings = {}
        for person_name, embedding_list in data.items():
            embeddings[person_name] = np.array(embedding_list)
        
        return embeddings
    except Exception as e:
        print(f"Error loading person embeddings: {e}")
        return {}


def save_person_embedding(user_id, person_name, features):
    """
    Save person embedding features for future recognition.
    """
    os.makedirs(PERSON_EMBEDDINGS_DIR, exist_ok=True)
    embeddings_file = os.path.join(PERSON_EMBEDDINGS_DIR, f"{user_id}_person_embeddings.json")
    
    # Load existing embeddings
    embeddings = load_person_embeddings(user_id)
    
    # Add or update the person's embedding
    embeddings[person_name] = features.tolist()
    
    try:
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings, f, indent=2)
        print(f"Saved person embedding for {person_name}")
    except Exception as e:
        print(f"Error saving person embedding: {e}")


def assess_lighting_quality(person_image):
    """
    Assess the lighting quality of a person image to determine reliability for recognition.
    Returns a quality score between 0 and 1, where higher is better.
    """
    if person_image.size == 0:
        return 0.0
    
    gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
    
    # Check for overexposure (too bright)
    overexposed_pixels = np.sum(gray > 240) / gray.size
    
    # Check for underexposure (too dark)
    underexposed_pixels = np.sum(gray < 15) / gray.size
    
    # Check contrast (standard deviation of intensity)
    contrast = np.std(gray)
    
    # Check brightness distribution
    mean_brightness = np.mean(gray)
    brightness_score = 1.0 - abs(mean_brightness - 128) / 128  # Prefer mid-range brightness
    
    # Calculate overall quality score
    exposure_penalty = (overexposed_pixels + underexposed_pixels) * 2
    contrast_score = min(contrast / 50.0, 1.0)  # Normalize contrast, cap at 1.0
    
    quality_score = (brightness_score * 0.3 + contrast_score * 0.5 + (1 - exposure_penalty) * 0.2)
    
    return max(0.0, min(1.0, quality_score))


def recognize_person_by_body(person_image, user_id, similarity_threshold=0.65, lighting_quality_threshold=0.3):
    """
    Recognize a person based on their body features with improved lighting handling.
    Returns (person_name, confidence) or (None, 0) if no match found.
    Now includes lighting quality assessment for better reliability.
    """
    if person_image.size == 0:
        return None, 0.0
    
    # Assess lighting quality first
    lighting_quality = assess_lighting_quality(person_image)
    
    # Adjust similarity threshold based on lighting quality
    # Lower quality lighting requires higher similarity for confidence
    adjusted_threshold = similarity_threshold + (1 - lighting_quality) * 0.15
    
    # Skip recognition if lighting is too poor
    if lighting_quality < lighting_quality_threshold:
        print(f"Lighting quality too poor for reliable recognition: {lighting_quality:.2f}")
        return None, 0.0
    
    # Extract features from the current person image
    current_features = extract_person_features(person_image)
    
    # Load known person embeddings
    known_embeddings = load_person_embeddings(user_id)
    
    if not known_embeddings:
        return None, 0.0
    
    best_match = None
    best_similarity = 0.0
    
    # Compare with all known persons
    for person_name, known_features in known_embeddings.items():
        # Calculate cosine similarity
        similarity = cosine_similarity([current_features], [known_features])[0][0]
        
        if similarity > best_similarity and similarity > adjusted_threshold:
            best_similarity = similarity
            best_match = person_name
    
    # Apply lighting quality factor to final confidence
    final_confidence = best_similarity * (0.7 + 0.3 * lighting_quality) if best_match else 0.0
    
    if best_match:
        print(f"Person recognized: {best_match} (similarity: {best_similarity:.3f}, "
              f"lighting: {lighting_quality:.2f}, final_confidence: {final_confidence:.3f})")
    
    return best_match, final_confidence


def calculate_recognition_confidence_score(similarity, lighting_quality, temporal_boost=0.0):
    """
    Calculate a comprehensive confidence score for person recognition decisions.
    Combines similarity score, lighting quality, and temporal consistency.
    
    Returns:
        confidence_score: float between 0-1
        decision_reason: string explaining the confidence level
    """
    base_confidence = similarity
    
    # Lighting quality factor (poor lighting reduces confidence)
    lighting_factor = 0.7 + 0.3 * lighting_quality
    
    # Apply lighting adjustment
    adjusted_confidence = base_confidence * lighting_factor
    
    # Apply temporal boost if available
    final_confidence = min(1.0, adjusted_confidence + temporal_boost)
    
    # Determine confidence level and reasoning
    if final_confidence >= 0.8:
        confidence_level = "HIGH"
        reason = "Strong feature match with good lighting"
    elif final_confidence >= 0.65:
        confidence_level = "MEDIUM"
        reason = "Good feature match" + (" with temporal consistency" if temporal_boost > 0 else "")
    elif final_confidence >= 0.5:
        confidence_level = "LOW"
        reason = "Weak feature match" + (" but poor lighting" if lighting_quality < 0.5 else "")
    else:
        confidence_level = "VERY_LOW"
        reason = "Insufficient feature similarity"
    
    decision_reason = f"{confidence_level}: {reason} (sim: {similarity:.2f}, lighting: {lighting_quality:.2f})"
    
    return final_confidence, decision_reason


def should_trust_recognition(confidence, lighting_quality, min_confidence=0.5, min_lighting=0.3):
    """
    Determine if a recognition result should be trusted based on multiple factors.
    
    Returns:
        trust_decision: bool
        reason: string explaining the decision
    """
    if confidence < min_confidence:
        return False, f"Confidence too low: {confidence:.2f} < {min_confidence}"
    
    if lighting_quality < min_lighting:
        return False, f"Lighting quality too poor: {lighting_quality:.2f} < {min_lighting}"
    
    # Additional checks for very high confidence with poor lighting (possible false positive)
    if confidence > 0.9 and lighting_quality < 0.4:
        return False, f"Suspiciously high confidence with poor lighting"
    
    return True, f"Recognition trusted (conf: {confidence:.2f}, lighting: {lighting_quality:.2f})"


def create_visualization_image(frame, detections, frame_number, user_id):
    """
    Create a visualization image with bounding boxes drawn on detected persons and faces.
    - People will be in GREEN.
    - Faces will be in BLUE.
    - Recognized persons will have their names displayed.
    Returns the relative URL path to the saved image.
    """
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    vis_frame = frame.copy()

    for detection in detections:
        bbox = detection.bbox

        # Determine color and label based on class name
        if detection.class_name == "person":
            color = (0, 255, 0)  # Green for person
            confidence_text = f"{detection.confidence:.2f}"
            
            # Check if person has been recognized
            person_name = getattr(detection, 'person_name', None)
            if person_name:
                label = f"{person_name}: {confidence_text}"
                color = (0, 255, 255)  # Yellow for recognized person
            else:
                label = f"Person: {confidence_text}"
                
        elif detection.class_name == "face":
            color = (255, 0, 0)  # Blue for face
            label = "Face"
        else:
            continue  # Skip other detections if any

        # Draw rectangle for the bounding box
        cv2.rectangle(vis_frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)

        # Create and draw the label with a background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_frame, (bbox.x1, bbox.y1 - label_size[1] - 10),
                      (bbox.x1 + label_size[0], bbox.y1), color, -1)
        cv2.putText(vis_frame, label, (bbox.x1, bbox.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text

    # Save the image with a unique filename
    filename = f"{user_id}_frame_{frame_number}.jpg"
    filepath = os.path.join(VISUALIZATIONS_DIR, filename)
    cv2.imwrite(filepath, vis_frame)

    return f"{BASE_URL}/static/visualizations/{filename}"


def batch_process_video_for_person_detection(video_paths: list, user_id: str) -> VideoAnalysis:
    """
    Processes the first video in a list to detect both persons (using YOLOv8)
    and faces (using face-recognition). When face detection fails, falls back
    to person recognition using body features with temporal consistency.
    Returns a VideoAnalysis object.
    """
    if not video_paths:
        return VideoAnalysis(video_path="", analysis=[])

    video_path_str = video_paths[0]
    video_path = Path(video_path_str)

    if not video_path.exists():
        # Simplified path resolution for clarity
        print(f"Video file not found: {video_path_str}")
        return VideoAnalysis(video_path=video_path_str, analysis=[])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return VideoAnalysis(video_path=video_path_str, analysis=[])

    frame_analyses = []
    frame_count = 0
    
    # Initialize temporal tracker for consistency across frames
    tracker = PersonTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            # Convert frame to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use YOLO to find all objects, we will filter for people
            results = yolo_model(frame)

            all_detections = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for i, box in enumerate(boxes):
                    class_name = yolo_model.names[class_ids[i]]
                    if class_name == "person":
                        # --- 1. Add Person Detection ---
                        person_bbox = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                        
                        # Extract person image for body-based recognition
                        person_image = frame[box[1]:box[3], box[0]:box[2]]
                        
                        # Try face recognition first
                        face_detected = False
                        person_rgb = rgb_frame[box[1]:box[3], box[0]:box[2]]
                        face_locations = face_recognition.face_locations(person_rgb)
                        
                        recognized_person_name = None
                        recognition_confidence = 0.0
                        
                        if face_locations:
                            # Face detected - add face detection
                            face_detected = True
                            for face_location in face_locations:
                                # face_location is (top, right, bottom, left) relative to person_image
                                top, right, bottom, left = face_location

                                # Convert face location to absolute coordinates in the full frame
                                abs_x1 = box[0] + left
                                abs_y1 = box[1] + top
                                abs_x2 = box[0] + right
                                abs_y2 = box[1] + bottom

                                face_bbox = BoundingBox(x1=abs_x1, y1=abs_y1, x2=abs_x2, y2=abs_y2)
                                face_detection = Detection(
                                    bbox=face_bbox,
                                    confidence=1.0,  # face_recognition doesn't provide confidence
                                    class_name="face"
                                )
                                all_detections.append(face_detection)
                        
                        # If no face detected or face recognition fails, try body-based recognition
                        if not face_detected or not recognized_person_name:
                            recognized_person_name, recognition_confidence = recognize_person_by_body(
                                person_image, user_id
                            )
                            
                            # Apply temporal consistency if person was recognized
                            if recognized_person_name:
                                # Get lighting quality for decision making
                                lighting_quality = assess_lighting_quality(person_image)
                                
                                # Get temporally-adjusted confidence
                                initial_temporal_confidence = tracker.get_temporal_confidence(
                                    frame_count, recognized_person_name, recognition_confidence, person_bbox
                                )
                                temporal_boost = initial_temporal_confidence - recognition_confidence
                                
                                # Calculate comprehensive confidence score
                                final_confidence, confidence_reason = calculate_recognition_confidence_score(
                                    recognition_confidence, lighting_quality, temporal_boost
                                )
                                
                                # Decide whether to trust this recognition
                                trust_decision, trust_reason = should_trust_recognition(
                                    final_confidence, lighting_quality
                                )
                                
                                if trust_decision:
                                    # Update recognition confidence with all factors
                                    recognition_confidence = final_confidence
                                    
                                    # Add to tracker history
                                    tracker.add_detection(frame_count, i, recognized_person_name, 
                                                        recognition_confidence, person_bbox)
                                    
                                    print(f"Recognized person by body: {recognized_person_name} - {confidence_reason}")
                                else:
                                    # Don't trust this recognition, reset it
                                    print(f"Recognition rejected: {trust_reason}")
                                    recognized_person_name = None
                                    recognition_confidence = 0.0
                        
                        # Create person detection with recognition info
                        person_detection = Detection(
                            bbox=person_bbox,
                            confidence=float(confidences[i]),
                            class_name="person"
                        )
                        
                        # Add person name if recognized
                        if recognized_person_name:
                            person_detection.person_name = recognized_person_name
                            person_detection.recognition_confidence = recognition_confidence
                        
                        all_detections.append(person_detection)

            visualization_url = None
            if all_detections:
                visualization_url = create_visualization_image(frame, all_detections, frame_count, user_id)

            frame_analysis = FrameAnalysis(
                frame_number=frame_count,
                detections=all_detections,
                visualization_url=visualization_url
            )
            frame_analyses.append(frame_analysis)

        frame_count += 1

    cap.release()
    return VideoAnalysis(video_path=video_path_str, analysis=frame_analyses)


def train_person_from_detections(user_id, person_name, detection_images):
    """
    Train person recognition model from multiple detection images.
    This function should be called when a user labels a person across multiple frames.
    
    Args:
        user_id: User identifier
        person_name: Name of the person to train
        detection_images: List of person crop images (numpy arrays)
    """
    if not detection_images:
        print("No images provided for training")
        return
    
    # Extract features from all images
    all_features = []
    for img in detection_images:
        features = extract_person_features(img)
        all_features.append(features)
    
    # Average the features for robustness
    avg_features = np.mean(all_features, axis=0)
    
    # Save the averaged features
    save_person_embedding(user_id, person_name, avg_features)
    
    print(f"Trained person recognition for {person_name} using {len(detection_images)} images")


def get_person_embeddings_info(user_id):
    """
    Get information about stored person embeddings for a user.
    Returns a dictionary with person names and metadata.
    """
    embeddings = load_person_embeddings(user_id)
    
    info = {}
    for person_name, features in embeddings.items():
        info[person_name] = {
            "feature_dimensions": len(features),
            "feature_norm": float(np.linalg.norm(features))
        }
    
    return info


def analyze_video_with_enhanced_recognition(video_path, user_id):
    """
    Analyze video with face recognition first, body-based recognition as fallback.
    Returns FaceRecognitionAnalysis format for compatibility with existing UI.
    Uses enhanced recognition that combines face detection + body matching when faces aren't found.
    """
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise ValueError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Load known face encodings 
    known_face_encodings, known_face_names = load_known_faces()
    print(f"Loaded {len(known_face_encodings)} known face encodings: {known_face_names}")
    
    # Load known person embeddings for body-based recognition
    known_person_embeddings = load_person_embeddings(user_id)
    print(f"Loaded {len(known_person_embeddings)} known person embeddings: {list(known_person_embeddings.keys())}")
    
    frame_analyses = []
    total_frames = 0
    processed_frames = 0
    total_recognized = 0
    total_unrecognized = 0
    processed_unrecognized_faces = []
    
    # Initialize temporal tracker
    tracker = PersonTracker()
    
    frame_interval = FRAME_INTERVAL  # Use same interval as enrollment
    face_id_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        total_frames += 1
        
        # Only process every Nth frame
        if total_frames % frame_interval != 0:
            continue
            
        processed_frames += 1
        frame_faces = []
        frame_recognized_faces = []
        frame_unrecognized_faces = []

        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use YOLO to find people first
        results = yolo_model(frame)
        
        if results[0].boxes is not None:
            print(f"Frame {total_frames}: YOLO detected {len(results[0].boxes)} objects")
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(boxes):
                class_name = yolo_model.names[class_ids[i]]
                if class_name == "person":
                    x1, y1, x2, y2 = box
                    person_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    person_image = frame[y1:y2, x1:x2]
                    person_rgb = rgb_frame[y1:y2, x1:x2]
                    
                    # Try face recognition first (PRIORITY)
                    face_locations = face_recognition.face_locations(person_rgb)
                    face_encodings = face_recognition.face_encodings(person_rgb, face_locations)
                    
                    person_identified = False
                    
                    if face_locations and face_encodings:
                        # Process each face found
                        for face_encoding, face_location in zip(face_encodings, face_locations):
                            face_top, face_right, face_bottom, face_left = face_location
                            
                            # Convert face coordinates to absolute frame coordinates
                            abs_face_bbox = BoundingBox(
                                x1=x1 + face_left, 
                                y1=y1 + face_top, 
                                x2=x1 + face_right, 
                                y2=y1 + face_bottom
                            )
                            
                            # Check against known faces
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                            
                            if True in matches:
                                # Face recognized!
                                first_match_index = matches.index(True)
                                person_name = known_face_names[first_match_index]
                                
                                face_crop_url = save_individual_face_crop(
                                    frame, abs_face_bbox.model_dump(), face_id_counter, user_id, "recognized", person_name
                                )
                                
                                face_in_frame = FaceInFrame(
                                    name=person_name,
                                    bbox=abs_face_bbox,
                                    recognition_status="recognized",
                                    person_name=person_name,
                                    face_crop_url=face_crop_url
                                )
                                frame_faces.append(face_in_frame)
                                frame_recognized_faces.append({"name": person_name, "bbox": abs_face_bbox.model_dump()})
                                total_recognized += 1
                                face_id_counter += 1
                                person_identified = True
                                
                                print(f"Person recognized by FACE: {person_name}")
                                
                            else:
                                # Unknown face - check if it's truly new
                                is_new = True
                                for existing_encoding in processed_unrecognized_faces:
                                    if np.allclose(existing_encoding, face_encoding, atol=0.6):
                                        is_new = False
                                        break
                                
                                if is_new:
                                    processed_unrecognized_faces.append(face_encoding)
                                    
                                    face_crop_url = save_individual_face_crop(
                                        frame, abs_face_bbox.model_dump(), face_id_counter, user_id, "unrecognized"
                                    )
                                    
                                    face_in_frame = FaceInFrame(
                                        name="Unknown",
                                        bbox=abs_face_bbox,
                                        recognition_status="unrecognized",
                                        person_name="",
                                        face_encoding=face_encoding.tolist(),
                                        face_crop_url=face_crop_url
                                    )
                                    frame_faces.append(face_in_frame)
                                    frame_unrecognized_faces.append({"bbox": abs_face_bbox.model_dump()})
                                    total_unrecognized += 1
                                    face_id_counter += 1
                    
                    # If no face was found or recognized, try BODY-BASED recognition as fallback
                    if not person_identified and known_person_embeddings:
                        print("No face detected/recognized, trying body-based recognition...")
                        
                        # Use the enhanced body recognition with lighting/confidence checks
                        recognized_person_name, recognition_confidence = recognize_person_by_body(
                            person_image, user_id
                        )
                        
                        if recognized_person_name:
                            # Get lighting quality for decision making
                            lighting_quality = assess_lighting_quality(person_image)
                            
                            # Get temporally-adjusted confidence
                            temporal_confidence = tracker.get_temporal_confidence(
                                total_frames, recognized_person_name, recognition_confidence, person_bbox
                            )
                            temporal_boost = temporal_confidence - recognition_confidence
                            
                            # Calculate comprehensive confidence score
                            final_confidence, confidence_reason = calculate_recognition_confidence_score(
                                recognition_confidence, lighting_quality, temporal_boost
                            )
                            
                            # Decide whether to trust this recognition
                            trust_decision, trust_reason = should_trust_recognition(
                                final_confidence, lighting_quality
                            )
                            
                            if trust_decision:
                                # Add to tracker history
                                tracker.add_detection(total_frames, i, recognized_person_name, 
                                                    final_confidence, person_bbox)
                                
                                # Create a "face" entry for the person (even though it's body-based)
                                # This maintains compatibility with the FaceRecognitionAnalysis format
                                face_crop_url = save_individual_face_crop(
                                    frame, person_bbox.model_dump(), face_id_counter, user_id, "recognized", recognized_person_name
                                )
                                
                                face_in_frame = FaceInFrame(
                                    name=recognized_person_name,
                                    bbox=person_bbox,  # Use person bbox since no face found
                                    recognition_status="recognized",
                                    person_name=recognized_person_name,
                                    face_crop_url=face_crop_url
                                )
                                frame_faces.append(face_in_frame)
                                frame_recognized_faces.append({"name": recognized_person_name, "bbox": person_bbox.model_dump()})
                                total_recognized += 1
                                face_id_counter += 1
                                person_identified = True
                                
                                print(f"Person recognized by BODY: {recognized_person_name} - {confidence_reason}")
                            else:
                                print(f"Body recognition rejected: {trust_reason}")

        # Create visualization if there are faces/people in this frame
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

    cap.release()
    
    return FaceRecognitionAnalysis(
        video_path=str(video_path),
        total_frames=total_frames,
        processed_frames=processed_frames,
        recognized_faces=total_recognized,
        unrecognized_faces=total_unrecognized,
        detections=frame_analyses
    )