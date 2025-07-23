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
# Import service functions only when needed to avoid circular import

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


def recognize_multiple_persons_by_body(person_images, user_id, similarity_threshold=0.65, lighting_quality_threshold=0.3):
    """
    Recognize multiple persons based on their body features with proper assignment.
    Prevents the same person from being assigned to multiple detections.
    Returns list of (person_name, confidence) tuples matching the input order.
    """
    if not person_images:
        return []
    
    # Load known person embeddings
    known_embeddings = load_person_embeddings(user_id)
    
    if not known_embeddings:
        return [(None, 0.0) for _ in person_images]
    
    # Extract features for all detected persons
    detected_features = []
    lighting_qualities = []
    
    for person_image in person_images:
        if person_image.size == 0:
            detected_features.append(None)
            lighting_qualities.append(0.0)
            continue
            
        # Assess lighting quality
        lighting_quality = assess_lighting_quality(person_image)
        lighting_qualities.append(lighting_quality)
        
        # Skip feature extraction if lighting is too poor
        if lighting_quality < lighting_quality_threshold:
            detected_features.append(None)
            continue
            
        # Extract features
        features = extract_person_features(person_image)
        detected_features.append(features)
    
    # Create similarity matrix: detected_persons x known_persons
    similarity_matrix = []
    known_person_names = list(known_embeddings.keys())
    
    for i, features in enumerate(detected_features):
        if features is None:
            similarity_matrix.append([0.0] * len(known_person_names))
            continue
            
        similarities = []
        lighting_quality = lighting_qualities[i]
        adjusted_threshold = similarity_threshold + (1 - lighting_quality) * 0.15
        
        for person_name in known_person_names:
            known_features = known_embeddings[person_name]
            similarity = cosine_similarity([features], [known_features])[0][0]
            
            # Apply threshold and lighting quality factor
            if similarity > adjusted_threshold:
                final_similarity = similarity * (0.7 + 0.3 * lighting_quality)
            else:
                final_similarity = 0.0
                
            similarities.append(final_similarity)
        
        similarity_matrix.append(similarities)
    
    # Perform greedy assignment to prevent duplicate assignments
    assignments = assign_persons_greedy(similarity_matrix, known_person_names)
    
    return assignments


def assign_persons_greedy(similarity_matrix, known_person_names):
    """
    Greedy assignment algorithm to assign detected persons to known persons.
    Each known person can only be assigned to one detected person per frame.
    """
    num_detected = len(similarity_matrix)
    num_known = len(known_person_names)
    
    assignments = [(None, 0.0) for _ in range(num_detected)]
    used_known_persons = set()
    
    # Create list of (similarity, detected_idx, known_idx) sorted by similarity
    candidates = []
    for d_idx in range(num_detected):
        for k_idx in range(num_known):
            similarity = similarity_matrix[d_idx][k_idx]
            if similarity > 0.0:
                candidates.append((similarity, d_idx, k_idx))
    
    # Sort by similarity (highest first)
    candidates.sort(reverse=True)
    
    # Greedily assign highest similarities first, avoiding duplicates
    for similarity, d_idx, k_idx in candidates:
        known_person = known_person_names[k_idx]
        
        # Skip if this known person is already assigned or detected person already assigned
        if known_person not in used_known_persons and assignments[d_idx][0] is None:
            assignments[d_idx] = (known_person, similarity)
            used_known_persons.add(known_person)
            print(f"Assigned detected person {d_idx} to '{known_person}' (confidence: {similarity:.3f})")
    
    return assignments


def recognize_person_by_body(person_image, user_id, similarity_threshold=0.65, lighting_quality_threshold=0.3):
    """
    Legacy function for single person recognition - now calls the multi-person version.
    """
    results = recognize_multiple_persons_by_body([person_image], user_id, similarity_threshold, lighting_quality_threshold)
    return results[0] if results else (None, 0.0)


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


def create_visualization_image(frame, detections, frame_number, user_id, detection_filter=None):
    """
    Create a visualization image with bounding boxes drawn on detected persons and faces.
    - Each person gets a unique color based on their person_id
    - Faces will be in BLUE.
    - Recognized persons will have their names displayed.
    
    Args:
        detection_filter: "person", "face", or None (show all)
        
    Returns the relative URL path to the saved image.
    """
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    vis_frame = frame.copy()
    
    # Define a set of distinct colors for different people (BGR format)
    person_colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue  
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
        (255, 20, 147), # Deep Pink
        (0, 206, 209),  # Dark Turquoise
    ]

    # Filter detections if specified
    filtered_detections = detections
    if detection_filter:
        filtered_detections = [d for d in detections if getattr(d, 'detection_type', 'person') == detection_filter]

    for detection in filtered_detections:
        bbox = detection.bbox
        detection_type = getattr(detection, 'detection_type', 'person')

        # Determine color and label based on detection type
        if detection_type == "person":
            # Use unique color based on person_id, default to green if no ID
            person_id = getattr(detection, 'person_id', 0)
            color = person_colors[person_id % len(person_colors)]
            
            confidence_text = f"{detection.confidence:.2f}"
            
            # Check if person has been recognized
            person_name = getattr(detection, 'person_name', None)
            if person_name:
                label = f"PERSON ID:{person_id} {person_name}: {confidence_text}"
                # Make recognized person boxes thicker
                box_thickness = 3
            else:
                label = f"PERSON ID:{person_id}: {confidence_text}"
                box_thickness = 2
                
        elif detection_type == "face":
            # Use unique color based on person_id for faces too
            person_id = getattr(detection, 'person_id', 0)
            face_color = (255, 0, 0)  # Blue base for faces
            color = face_color
            
            person_name = getattr(detection, 'person_name', None)
            confidence_text = f"{detection.confidence:.2f}"
            
            if person_name:
                label = f"FACE ID:{person_id} {person_name}: {confidence_text}"
                box_thickness = 3
            else:
                label = f"FACE ID:{person_id}: {confidence_text}"
                box_thickness = 2
        else:
            continue  # Skip other detections if any

        # Draw rectangle for the bounding box
        cv2.rectangle(vis_frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, box_thickness)

        # Create and draw the label with a background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_frame, (bbox.x1, bbox.y1 - label_size[1] - 10),
                      (bbox.x1 + label_size[0], bbox.y1), color, -1)
        cv2.putText(vis_frame, label, (bbox.x1, bbox.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text

    # Save the image with a unique filename
    suffix = f"_{detection_filter}" if detection_filter else "_combined"
    filename = f"{user_id}_frame_{frame_number}{suffix}.jpg"
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

                # Collect all person data for batch processing
                person_data = []
                for i, box in enumerate(boxes):
                    class_name = yolo_model.names[class_ids[i]]
                    if class_name == "person":
                        person_bbox = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                        person_image = frame[box[1]:box[3], box[0]:box[2]]
                        person_rgb = rgb_frame[box[1]:box[3], box[0]:box[2]]
                        
                        person_data.append({
                            'index': i,
                            'box': box,
                            'bbox': person_bbox,
                            'image': person_image,
                            'rgb_image': person_rgb,
                            'confidence': float(confidences[i])
                        })
                
                if person_data:
                    # Process faces first for all persons
                    for person_id, person in enumerate(person_data):
                        face_locations = face_recognition.face_locations(person['rgb_image'])
                        person['face_detected'] = bool(face_locations)
                        
                        if face_locations:
                            for face_location in face_locations:
                                top, right, bottom, left = face_location
                                box = person['box']
                                
                                abs_x1 = box[0] + left
                                abs_y1 = box[1] + top
                                abs_x2 = box[0] + right
                                abs_y2 = box[1] + bottom

                                face_bbox = BoundingBox(x1=abs_x1, y1=abs_y1, x2=abs_x2, y2=abs_y2)
                                face_detection = Detection(
                                    bbox=face_bbox,
                                    confidence=1.0,
                                    class_name="face",
                                    person_id=person_id,  # Link face to person
                                    detection_type="face"
                                )
                                all_detections.append(face_detection)
                    
                    # Batch body-based recognition for persons without faces or failed face recognition
                    persons_for_body_recognition = [p for p in person_data if not p['face_detected']]
                    
                    if persons_for_body_recognition:
                        person_images = [p['image'] for p in persons_for_body_recognition]
                        body_recognition_results = recognize_multiple_persons_by_body(person_images, user_id)
                        
                        # Apply results back to person data
                        for person, (recognized_name, recognition_confidence) in zip(persons_for_body_recognition, body_recognition_results):
                            person['recognized_name'] = recognized_name
                            person['recognition_confidence'] = recognition_confidence
                    
                    # Create person detections with recognition info and unique IDs
                    for person_id, person in enumerate(person_data):
                        person_detection = Detection(
                            bbox=person['bbox'],
                            confidence=person['confidence'],
                            class_name="person",
                            person_id=person_id,  # Assign unique ID for each person in the frame
                            detection_type="person"
                        )
                        
                        # Add recognition info if available
                        if person.get('recognized_name'):
                            person_detection.person_name = person['recognized_name']
                            person_detection.recognition_confidence = person['recognition_confidence']
                            
                            # Apply temporal consistency and confidence scoring
                            lighting_quality = assess_lighting_quality(person['image'])
                            
                            # Get temporally-adjusted confidence
                            initial_temporal_confidence = tracker.get_temporal_confidence(
                                frame_count, person['recognized_name'], person['recognition_confidence'], person['bbox']
                            )
                            temporal_boost = initial_temporal_confidence - person['recognition_confidence']
                            
                            # Calculate comprehensive confidence score
                            final_confidence, confidence_reason = calculate_recognition_confidence_score(
                                person['recognition_confidence'], lighting_quality, temporal_boost
                            )
                            
                            # Decide whether to trust this recognition
                            trust_decision, trust_reason = should_trust_recognition(
                                final_confidence, lighting_quality
                            )
                            
                            if trust_decision:
                                # Update recognition confidence with all factors
                                person_detection.recognition_confidence = final_confidence
                                
                                # Add to tracker history
                                tracker.add_detection(frame_count, person['index'], person['recognized_name'], 
                                                    final_confidence, person['bbox'])
                                
                                print(f"Recognized person by body: {person['recognized_name']} - {confidence_reason}")
                            else:
                                # Don't trust this recognition, remove it
                                print(f"Recognition rejected: {trust_reason}")
                                person_detection.person_name = None
                                person_detection.recognition_confidence = None
                        
                        all_detections.append(person_detection)

            # Create separate visualization images for person and face detections
            visualization_urls = {}
            if all_detections:
                # Combined view (all detections)
                visualization_urls["combined"] = create_visualization_image(frame, all_detections, frame_count, user_id)
                
                # Person-only view
                person_detections = [d for d in all_detections if d.detection_type == "person"]
                if person_detections:
                    visualization_urls["person"] = create_visualization_image(frame, all_detections, frame_count, user_id, "person")
                
                # Face-only view  
                face_detections = [d for d in all_detections if d.detection_type == "face"]
                if face_detections:
                    visualization_urls["face"] = create_visualization_image(frame, all_detections, frame_count, user_id, "face")
            
            # For backward compatibility, use combined as the main visualization URL
            visualization_url = visualization_urls.get("combined")

            frame_analysis = FrameAnalysis(
                frame_number=frame_count,
                detections=all_detections,
                visualization_url=visualization_url,
                visualization_urls=visualization_urls if visualization_urls else None
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

    # Import here to avoid circular import
    from .service import load_known_faces, save_individual_face_crop, create_face_visualization
    
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

            # Collect all person data for batch processing
            person_data = []
            for i, box in enumerate(boxes):
                class_name = yolo_model.names[class_ids[i]]
                if class_name == "person":
                    x1, y1, x2, y2 = box
                    person_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    person_image = frame[y1:y2, x1:x2]
                    person_rgb = rgb_frame[y1:y2, x1:x2]
                    
                    person_data.append({
                        'index': i,
                        'box': box,
                        'bbox': person_bbox,
                        'image': person_image,
                        'rgb_image': person_rgb,
                        'confidence': float(confidences[i])
                    })
            
            if person_data:
                # Process faces first for all persons
                for person in person_data:
                    person_identified = False  # Initialize for each person
                    face_locations = face_recognition.face_locations(person['rgb_image'])
                    face_encodings = face_recognition.face_encodings(person['rgb_image'], face_locations)
                    
                    person['face_detected'] = False
                    person['face_recognized'] = False
                    
                    if face_locations and face_encodings:
                        person['face_detected'] = True
                    
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