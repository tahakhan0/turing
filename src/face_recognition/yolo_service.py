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

# Initialize the YOLOv8 model - Using user's working version approach
import torch
import warnings

# Temporarily disable the weights_only warning for trusted model loading
try:
    # Try loading with weights_only=False (trusted source)
    import torch.serialization
    original_load = torch.load
    
    def safe_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = safe_torch_load
    
    # Try loading YOLO model
    try:
        yolo_model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Error loading YOLO model from current directory: {e}")
        # Fallback: try loading from the models directory
        yolo_model = YOLO("/app/models/yolov8n.pt")
        
    # Restore original torch.load
    torch.load = original_load
    
except Exception as e:
    print(f"Critical error initializing YOLO model: {e}")
    # Create a dummy model that will fail gracefully
    yolo_model = None

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
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
    area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


def extract_person_features(person_image):
    """
    Extract comprehensive features from a person image for recognition.
    Uses a multi-modal approach combining color, texture, and shape features.
    Returns a 289-dimensional feature vector.
    """
    try:
        # Ensure image is in the correct format
        if len(person_image.shape) == 3 and person_image.shape[2] == 3:
            # Convert BGR to RGB for consistent processing
            rgb_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = person_image

        # Resize to standard size for consistent feature extraction
        standard_size = (128, 256)  # Width x Height - typical person aspect ratio
        resized = cv2.resize(rgb_image, standard_size)
        
        # Convert to different color spaces for feature diversity
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # 1. Color Histogram Features (RGB: 3 x 16 = 48 features)
        for channel in range(3):
            hist = cv2.calcHist([resized], [channel], None, [16], [0, 256])
            features.extend(hist.flatten())
        
        # 2. HSV Histogram Features (HSV: 3 x 16 = 48 features)  
        for channel in range(3):
            hist = cv2.calcHist([hsv], [channel], None, [16], [0, 256])
            features.extend(hist.flatten())
        
        # 3. Edge and Texture Features (24 features)
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge density in different regions (8 regions = 8 features)
        h, w = edges.shape
        for i in range(2):
            for j in range(4):  # 2x4 grid for body regions
                region = edges[i*h//2:(i+1)*h//2, j*w//4:(j+1)*w//4]
                edge_density = np.sum(region) / (region.size * 255.0)
                features.append(edge_density)
        
        # Texture features using Local Binary Pattern approximation (16 features)
        kernel_3x3 = np.ones((3,3), np.uint8)
        morphological = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_3x3)
        for i in range(4):
            for j in range(4):  # 4x4 grid
                region = morphological[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
                texture_measure = np.std(region)
                features.append(texture_measure)
        
        # 4. Body Shape and Silhouette Features (32 features)
        # Binary silhouette
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Contours for shape analysis
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Hu moments (7 features)
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log transform for better numerical properties
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            features.extend(hu_moments)
            
            # Aspect ratio and solidity (2 features)
            x, y, w_cont, h_cont = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w_cont) / h_cont if h_cont > 0 else 0
            area = cv2.contourArea(largest_contour)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            features.extend([aspect_ratio, solidity])
            
            # Contour approximation features (4 features)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            features.extend([len(approx), cv2.arcLength(largest_contour, True), area, hull_area])
            
            # Body proportion features (8 features)
            # Top, middle, bottom regions analysis
            for region_idx in range(4):
                y_start = region_idx * h // 4
                y_end = (region_idx + 1) * h // 4
                region_mask = binary[y_start:y_end, :]
                region_density = np.sum(region_mask) / (region_mask.size * 255.0)
                features.append(region_density)
                
                # Width variation across height
                region_widths = []
                for row in region_mask:
                    white_pixels = np.sum(row == 255)
                    region_widths.append(white_pixels)
                avg_width = np.mean(region_widths) if region_widths else 0
                features.append(avg_width / w if w > 0 else 0)
        else:
            # If no contours found, add zero features
            features.extend([0] * 25)
        
        # 5. Gradient and Lighting Features (32 features)
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Gradient statistics in regions (16 features)
        for i in range(4):
            for j in range(4):
                region = grad_magnitude[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
                features.extend([np.mean(region), np.std(region)])
        
        # 6. Advanced Features (16 features)
        # Intensity distribution features
        hist_gray = cv2.calcHist([gray], [0], None, [8], [0, 256])
        features.extend(hist_gray.flatten())
        
        # Spatial features - center of mass, principal axes
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            features.extend([cx/w, cy/h])  # Normalized center of mass
        else:
            features.extend([0.5, 0.5])  # Default center
        
        # Lighting robustness features - relative measurements
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        features.extend([mean_intensity/255.0, std_intensity/255.0])
        
        # Symmetry features (4 features)
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        # Symmetry measure
        symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half_flipped.astype(float)))
        features.append(symmetry_diff / 255.0)
        
        # Vertical symmetry (top-bottom)
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        bottom_half_flipped = cv2.flip(bottom_half, 0)
        
        min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half_flipped = bottom_half_flipped[:min_height, :]
        
        v_symmetry_diff = np.mean(np.abs(top_half.astype(float) - bottom_half_flipped.astype(float)))
        features.append(v_symmetry_diff / 255.0)
        
        # Add padding to reach exactly 289 features if needed
        while len(features) < 289:
            features.append(0.0)
        
        # Ensure we have exactly 289 features
        features = features[:289]
        
        # Normalize features to prevent any single feature from dominating
        features = np.array(features, dtype=np.float32)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-8
        norm = np.linalg.norm(features) + epsilon
        features = features / norm
        
        return features
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        # Return zero vector if extraction fails
        return np.zeros(289, dtype=np.float32)


def load_person_embeddings(user_id):
    """Load saved person embeddings for a user."""
    try:
        embeddings_file = os.path.join(PERSON_EMBEDDINGS_DIR, f"{user_id}_person_embeddings.json")
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'r') as f:
                embeddings_data = json.load(f)
                # Convert lists back to numpy arrays
                for person_name in embeddings_data:
                    embeddings_data[person_name] = np.array(embeddings_data[person_name], dtype=np.float32)
                return embeddings_data
        return {}
    except Exception as e:
        print(f"Error loading person embeddings: {e}")
        return {}


def save_person_embedding(user_id, person_name, features):
    """Save person embedding to persistent storage."""
    try:
        # Ensure directory exists
        os.makedirs(PERSON_EMBEDDINGS_DIR, exist_ok=True)
        
        # Load existing embeddings
        embeddings = load_person_embeddings(user_id)
        
        # Add or update the person's features
        embeddings[person_name] = features.tolist()  # Convert numpy array to list for JSON serialization
        
        # Save back to file
        embeddings_file = os.path.join(PERSON_EMBEDDINGS_DIR, f"{user_id}_person_embeddings.json")
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings, f, indent=2)
        
        print(f"Saved person embedding for {person_name} (user: {user_id})")
        return True
    except Exception as e:
        print(f"Error saving person embedding: {e}")
        return False


def assess_lighting_quality(image):
    """
    Assess the lighting quality of an image for recognition reliability.
    Returns a score from 0 (poor) to 1 (excellent) and quality metrics.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calculate basic statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Brightness score (optimal range: 80-180)
        if 80 <= mean_brightness <= 180:
            brightness_score = 1.0
        elif mean_brightness < 30 or mean_brightness > 220:
            brightness_score = 0.0
        else:
            brightness_score = 0.5
        
        # Contrast score (higher std is better up to a point)
        contrast_score = min(std_brightness / 50.0, 1.0)
        
        # Check for over/under exposure
        overexposed_pixels = np.sum(gray > 240) / gray.size
        underexposed_pixels = np.sum(gray < 15) / gray.size
        exposure_score = 1.0 - min(overexposed_pixels + underexposed_pixels, 1.0)
        
        # Overall quality score
        quality_score = (brightness_score * 0.4 + contrast_score * 0.4 + exposure_score * 0.2)
        
        return quality_score, {
            'mean_brightness': mean_brightness,
            'contrast': std_brightness,
            'overexposed_ratio': overexposed_pixels,
            'underexposed_ratio': underexposed_pixels
        }
    except Exception as e:
        print(f"Error assessing lighting quality: {e}")
        return 0.5, {}


def recognize_multiple_persons_by_body(frame, person_detections, user_id, confidence_threshold=0.6):
    """
    Recognize multiple persons in detections using body-based features.
    Returns updated detections with person names and confidence scores.
    Uses assignment algorithm to prevent duplicate assignments.
    """
    try:
        if not person_detections:
            return person_detections
        
        # Load saved embeddings
        saved_embeddings = load_person_embeddings(user_id)
        
        # Also load saved labels from service.py for cross-checking
        try:
            from .service import storage
            labels_data = storage.load_face_labels(user_id)
            saved_labels = labels_data.get("labeled_faces", []) if labels_data else []
        except Exception as e:
            print(f"Could not load saved labels: {e}")
            saved_labels = []
        
        if not saved_embeddings:
            # No saved embeddings, but still check saved labels
            for detection in person_detections:
                detection.person_name = None  # Don't default to "unknown"
                detection.recognition_confidence = 0.0
                detection.detection_type = "person"
            return person_detections
        
        # Extract features for all detections
        detection_features = []
        valid_detections = []
        
        for detection in person_detections:
            try:
                bbox = detection.bbox
                person_crop = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
                
                if person_crop.size > 0:
                    features = extract_person_features(person_crop)
                    detection_features.append(features)
                    valid_detections.append(detection)
                else:
                    detection.person_name = "unknown"
                    detection.recognition_confidence = 0.0
                    detection.detection_type = "person"
            except Exception as e:
                print(f"Error processing detection: {e}")
                detection.person_name = "unknown"
                detection.recognition_confidence = 0.0
                detection.detection_type = "person"
        
        if not detection_features:
            return person_detections
        
        # Calculate similarities between detections and saved persons
        detection_features = np.array(detection_features)
        person_names = list(saved_embeddings.keys())
        saved_features = np.array([saved_embeddings[name] for name in person_names])
        
        # Compute similarity matrix
        similarities = cosine_similarity(detection_features, saved_features)
        
        # Use assignment algorithm to prevent duplicate assignments
        assignments = assign_persons_greedy(similarities, confidence_threshold)
        
        # Apply assignments
        for detection_idx, (person_idx, confidence) in assignments.items():
            detection = valid_detections[detection_idx]
            if person_idx is not None:
                detection.person_name = person_names[person_idx]
                detection.recognition_confidence = confidence
            else:
                detection.person_name = "unknown"
                detection.recognition_confidence = 0.0
            detection.detection_type = "person"
        
        return person_detections
        
    except Exception as e:
        print(f"Error in body-based recognition: {e}")
        # Fallback: mark all as unknown
        for detection in person_detections:
            detection.person_name = "unknown"
            detection.recognition_confidence = 0.0
            detection.detection_type = "person"
        return person_detections


def assign_persons_greedy(similarity_matrix, confidence_threshold):
    """
    Greedy assignment algorithm to assign detections to persons.
    Prevents multiple detections from being assigned to the same person.
    """
    assignments = {}
    used_persons = set()
    
    # Get all similarity scores with their indices
    candidates = []
    for det_idx in range(similarity_matrix.shape[0]):
        for person_idx in range(similarity_matrix.shape[1]):
            confidence = similarity_matrix[det_idx, person_idx]
            if confidence >= confidence_threshold:
                candidates.append((confidence, det_idx, person_idx))
    
    # Sort by confidence (highest first)
    candidates.sort(reverse=True)
    
    # Assign greedily
    used_detections = set()
    for confidence, det_idx, person_idx in candidates:
        if det_idx not in used_detections and person_idx not in used_persons:
            assignments[det_idx] = (person_idx, confidence)
            used_detections.add(det_idx)
            used_persons.add(person_idx)
    
    # Mark unassigned detections
    for det_idx in range(similarity_matrix.shape[0]):
        if det_idx not in assignments:
            assignments[det_idx] = (None, 0.0)
    
    return assignments


def recognize_person_by_body(frame, person_crop, user_id, confidence_threshold=0.6):
    """
    Legacy wrapper function for single person recognition.
    Maintained for backward compatibility.
    """
    # Create a dummy detection for the crop
    dummy_detection = Detection(
        bbox=BoundingBox(x1=0, y1=0, x2=person_crop.shape[1], y2=person_crop.shape[0]),
        confidence=1.0,
        class_name="person"
    )
    
    results = recognize_multiple_persons_by_body(person_crop, [dummy_detection], user_id, confidence_threshold)
    if results:
        return results[0].person_name, results[0].recognition_confidence
    return "unknown", 0.0


def calculate_recognition_confidence_score(similarity, lighting_quality, temporal_confidence=None):
    """
    Calculate a comprehensive confidence score for person recognition.
    Combines similarity score with lighting quality and temporal information.
    """
    try:
        # Base similarity contribution (60% weight)
        similarity_contribution = similarity * 0.6
        
        # Lighting quality contribution (25% weight)
        lighting_contribution = lighting_quality * 0.25
        
        # Temporal consistency contribution (15% weight)
        temporal_contribution = 0.0
        if temporal_confidence is not None:
            temporal_contribution = min(temporal_confidence, 1.0) * 0.15
        else:
            temporal_contribution = 0.15  # Assume neutral if no temporal info
        
        # Combine all factors
        final_confidence = similarity_contribution + lighting_contribution + temporal_contribution
        
        # Apply penalty for very poor lighting (reduces confidence significantly)
        if lighting_quality < 0.3:
            final_confidence *= 0.7  # 30% penalty for poor lighting
        
        # Ensure confidence is within [0, 1] range
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return final_confidence
        
    except Exception as e:
        print(f"Error calculating confidence score: {e}")
        return similarity  # Fallback to just similarity


def should_trust_recognition(similarity, lighting_quality, person_crop_size, min_crop_pixels=2000):
    """
    Determine if a recognition result should be trusted based on multiple factors.
    Returns (should_trust, reason)
    """
    try:
        crop_area = person_crop_size[0] * person_crop_size[1] if len(person_crop_size) == 2 else person_crop_size
        
        # Check minimum similarity threshold
        if similarity < 0.5:
            return False, "Low similarity score"
        
        # Check lighting quality
        if lighting_quality < 0.25:
            return False, "Poor lighting conditions"
        
        # Check crop size (person should be large enough for reliable recognition)
        if crop_area < min_crop_pixels:
            return False, "Person too small in frame"
        
        # High confidence conditions
        if similarity > 0.8 and lighting_quality > 0.7:
            return True, "High confidence match"
        
        # Medium confidence conditions
        if similarity > 0.6 and lighting_quality > 0.4:
            return True, "Medium confidence match"
        
        # Special case: very high similarity can overcome poor lighting
        if similarity > 0.9:
            return True, "Very high similarity overcomes lighting issues"
        
        return False, "Below confidence threshold"
        
    except Exception as e:
        print(f"Error in trust assessment: {e}")
        return False, "Error in assessment"


def create_visualization_image(frame, detections, frame_number, user_id, view_type="combined"):
    """
    Create a visualization image with bounding boxes drawn on detected persons.
    Returns the relative URL path to the saved image.
    
    Args:
        frame: The video frame
        detections: List of Detection objects
        frame_number: Frame number for unique filename
        user_id: User ID for organization
        view_type: "combined", "person", or "face" for different visualization styles
    """
    try:
        # Ensure the visualizations directory exists
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

        # Create a copy of the frame to draw on
        vis_frame = frame.copy()

        # Color scheme for different person IDs (BGR format for OpenCV)
        person_colors = [
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue  
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 128),    # Purple
            (0, 165, 255),    # Orange
            (255, 255, 224),  # Light Blue
            (0, 255, 0),      # Lime
            (147, 20, 255),   # Deep Pink
            (208, 224, 64),   # Dark Turquoise
        ]
        
        # Default colors for different detection types
        default_colors = {
            'face': (255, 0, 0),        # Blue for face detection
            'unknown': (0, 165, 255),   # Orange for unknown persons
            'known': (0, 255, 0),       # Green for known persons
        }

        # Track person IDs to ensure unique identification
        person_id_counter = 1

        # Draw bounding boxes for each detection
        for detection in detections:
            bbox = detection.bbox
            confidence = detection.confidence
            
            # Determine detection type and color
            detection_type = getattr(detection, 'detection_type', 'person')
            person_name = getattr(detection, 'person_name', 'unknown')
            recognition_confidence = getattr(detection, 'recognition_confidence', 0.0)
            
            # Skip face detections if view_type is "person" only
            if view_type == "person" and detection_type == "face":
                continue
            # Skip person detections if view_type is "face" only  
            if view_type == "face" and detection_type == "person":
                continue
            
            # Choose color based on person_id or recognition status
            person_id = getattr(detection, 'person_id', None)
            if detection_type == 'face':
                color = default_colors['face']
                box_thickness = 2
            elif person_id is not None and detection_type == "person":
                # Use person_id specific color
                color = person_colors[person_id % len(person_colors)]
                box_thickness = 2
            elif person_name == "unknown" or person_name is None:
                color = default_colors['unknown']
                box_thickness = 2
            else:
                color = default_colors['known']
                box_thickness = 3  # Thicker for known persons
            
            # Draw rectangle
            cv2.rectangle(vis_frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, box_thickness)

            # Prepare label text
            if detection_type == "face":
                if person_name and person_name != "unknown" and person_name.strip():
                    label = f"Label: {person_name} ({recognition_confidence:.2f})"
                else:
                    label = f"Label: Unknown ({confidence:.2f})"
            else:
                if person_name and person_name != "unknown" and person_name.strip():
                    label = f"Label: {person_name} ({recognition_confidence:.2f})"
                else:
                    label = f"Label: Unknown ({confidence:.2f})"

            # Calculate label background size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_frame, 
                         (bbox.x1, bbox.y1 - label_size[1] - 10),
                         (bbox.x1 + label_size[0] + 10, bbox.y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, 
                       (bbox.x1 + 5, bbox.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add metadata to the image
        metadata_text = f"Frame: {frame_number} | User: {user_id} | View: {view_type}"
        cv2.putText(vis_frame, metadata_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save the image with a unique filename
        filename = f"{user_id}_frame_{frame_number}_{view_type}.jpg"
        filepath = os.path.join(VISUALIZATIONS_DIR, filename)
        cv2.imwrite(filepath, vis_frame)

        # Return the absolute URL that can be accessed from anywhere
        return f"{BASE_URL}/static/visualizations/{filename}"
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None


def batch_process_video_for_person_detection(video_paths: list, user_id: str) -> VideoAnalysis:
    """
    Enhanced version of the original function - processes the first video in a list for person detection using YOLOv8,
    with frame skipping for performance. Maintains backward compatibility.
    Returns a VideoAnalysis object.
    """
    if not video_paths:
        return VideoAnalysis(video_path="", analysis=[])

    video_path_str = video_paths[0]

    # Use pathlib to handle file paths robustly
    video_path = Path(video_path_str)

    # Check if path exists, if not try to resolve it
    if not video_path.exists():
        if not video_path.is_absolute():
            resolved_path = video_path.resolve()
            if resolved_path.exists():
                video_path = resolved_path
            else:
                print(f"Video file not found: {video_path_str}")
                return VideoAnalysis(video_path=video_path_str, analysis=[])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return VideoAnalysis(video_path=video_path_str, analysis=[])

    frame_analyses = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- FRAME SKIPPING LOGIC ---
        # Only process the frame if its count is a multiple of FRAME_INTERVAL
        if frame_count % FRAME_INTERVAL == 0:
            # Perform inference on the current frame
            results = yolo_model(frame)

            # Extract detections
            detections = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for i, box in enumerate(boxes):
                    class_name = yolo_model.names[class_ids[i]]
                    if class_name == "person":
                        bbox = BoundingBox(x1=int(box[0]), y1=int(box[1]), x2=int(box[2]), y2=int(box[3]))
                        detection = Detection(
                            bbox=bbox,
                            confidence=float(confidences[i]),
                            class_name=class_name
                        )
                        # Add default values for enhanced features
                        detection.person_name = None  # Don't default to "unknown", let UI prompt
                        detection.recognition_confidence = 0.0
                        detection.detection_type = "person"
                        detection.person_id = i  # Set person_id based on detection order
                        detections.append(detection)

            # Apply body-based recognition to detections
            detections = recognize_multiple_persons_by_body(frame, detections, user_id)

            # Generate visualization image if there are detections
            visualization_url = None
            if detections:
                visualization_url = create_visualization_image(frame, detections, frame_count, user_id)

            # Only append analysis for processed frames
            frame_analysis = FrameAnalysis(
                frame_number=frame_count,
                detections=detections,
                visualization_url=visualization_url
            )
            frame_analyses.append(frame_analysis)

        # Increment frame count for every frame read
        frame_count += 1

    cap.release()

    return VideoAnalysis(video_path=video_path_str, analysis=frame_analyses)


def train_person_from_detections(user_id, person_name, detection_images):
    """
    Train person recognition from multiple detection images.
    This improves recognition accuracy by learning from multiple samples.
    """
    try:
        if not detection_images:
            return False, "No images provided"
        
        features_list = []
        for img in detection_images:
            try:
                if isinstance(img, str):
                    # If it's a file path, load the image
                    image = cv2.imread(img)
                else:
                    image = img
                
                if image is not None:
                    features = extract_person_features(image)
                    features_list.append(features)
            except Exception as e:
                print(f"Error processing training image: {e}")
                continue
        
        if not features_list:
            return False, "No valid images could be processed"
        
        # Average the features for robustness
        avg_features = np.mean(features_list, axis=0)
        
        # Save the averaged features
        success = save_person_embedding(user_id, person_name, avg_features)
        
        return success, f"Trained on {len(features_list)} images"
        
    except Exception as e:
        print(f"Error training person: {e}")
        return False, str(e)


def get_person_embeddings_info(user_id):
    """Get information about saved person embeddings for a user."""
    try:
        embeddings = load_person_embeddings(user_id)
        return {
            "user_id": user_id,
            "num_persons": len(embeddings),
            "person_names": list(embeddings.keys()) if embeddings else [],
            "storage_path": os.path.join(PERSON_EMBEDDINGS_DIR, f"{user_id}_person_embeddings.json")
        }
    except Exception as e:
        print(f"Error getting embeddings info: {e}")
        return {"error": str(e)}


def convert_numpy_to_python(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

def reconcile_detections(face_detections, person_detections, iou_threshold=0.8):
    """
    Reconcile face and person detections to merge overlapping boxes.
    - If a face is inside a person box, merge them.
    - Prioritize the person's bounding box and the face's name.
    """
    if not face_detections or not person_detections:
        return face_detections, person_detections

    unmatched_persons = list(person_detections)
    merged_detections = []
    faces_to_remove = []

    for face in face_detections:
        best_match_person = None
        highest_iou = 0

        for person in unmatched_persons:
            iou = calculate_bbox_overlap(face.bbox, person.bbox)
            if iou > highest_iou:
                highest_iou = iou
                best_match_person = person

        if highest_iou > iou_threshold:
            # Merge face and person
            merged_face = face
            merged_face.bbox = best_match_person.bbox  # Use person's larger bbox
            merged_face.detection_type = "merged"
            
            # If face was unknown but person was known, use person's name
            if (face.person_name is None or face.person_name == "unknown") and \
               (best_match_person.person_name is not None and best_match_person.person_name != "unknown"):
                merged_face.person_name = best_match_person.person_name
                merged_face.name = best_match_person.person_name
                merged_face.recognition_status = "recognized"

            merged_detections.append(merged_face)
            unmatched_persons.remove(best_match_person)
            faces_to_remove.append(face)

    # Filter out the merged faces from the original list
    remaining_faces = [f for f in face_detections if f not in faces_to_remove]
    
    # The final list of detections is the merged ones, plus any remaining faces and persons
    final_detections = merged_detections + remaining_faces + unmatched_persons
    
    return final_detections

def analyze_video_with_enhanced_recognition(video_path: str, user_id: str) -> FaceRecognitionAnalysis:
    """
    Enhanced video analysis that combines face recognition with body-based recognition.
    Returns FaceRecognitionAnalysis format for UI compatibility.
    """
    try:
        # Initialize tracker for temporal consistency
        person_tracker = PersonTracker()
        
        # Use pathlib to handle file paths robustly
        video_path_obj = Path(video_path)
        
        # Use pathlib to handle file paths robustly
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        frame_analyses = []
        frame_count = 0
        
        # Import face recognition service functions
        try:
            from .service import detect_faces_in_frame, recognize_faces_in_detections
            face_recognition_available = True
        except ImportError:
            print("Face recognition service not available, using body-based recognition only")
            face_recognition_available = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every FRAME_INTERVAL frames
            if frame_count % FRAME_INTERVAL == 0:
                all_faces = []
                face_id_counter = 0
                
                # Step 1: Try face recognition first
                if face_recognition_available:
                    try:
                        face_detections = detect_faces_in_frame(frame)
                        if face_detections:
                            recognized_faces = recognize_faces_in_detections(frame, face_detections, user_id)
                            
                            # Convert to FaceInFrame format with face IDs
                            for detection in recognized_faces:
                                person_name = getattr(detection, 'person_name', None)
                                
                                # Assign face ID for color coding
                                detection.person_id = face_id_counter
                                face_id_counter += 1
                                
                                face_in_frame = FaceInFrame(
                            name=person_name if person_name and person_name != "unknown" else "Unknown",
                            bbox=detection.bbox,
                            recognition_status="recognized" if person_name and person_name != "unknown" else "unrecognized",
                            person_name=person_name,
                            face_encoding=[],  # Not needed for display
                            person_id=getattr(detection, 'person_id', 0),
                            detection_type=getattr(detection, 'detection_type', 'face')
                        )
                        all_faces.append(face_in_frame)
                    except Exception as e:
                        print(f"Face recognition failed, using body-based fallback: {e}")
                
                # Step 2: Body-based recognition as fallback or supplement
                # Perform YOLO person detection
                results = yolo_model(frame)
                person_detections = []
                
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                    person_id_counter = face_id_counter  # Continue ID counter from face detections
                    for i, box in enumerate(boxes):
                        class_name = yolo_model.names[class_ids[i]]
                        if class_name == "person":
                            bbox = BoundingBox(x1=int(box[0]), y1=int(box[1]), x2=int(box[2]), y2=int(box[3]))
                            detection = Detection(
                                bbox=bbox,
                                confidence=float(confidences[i]),
                                class_name=class_name
                            )
                            # Assign unique person ID for color coding
                            detection.person_id = person_id_counter
                            person_id_counter += 1
                            person_detections.append(detection)

                # Apply body-based recognition
                if person_detections:
                    recognized_persons = recognize_multiple_persons_by_body(frame, person_detections, user_id)
                    
                    # Convert to FaceInFrame format and add to results
                    for detection in recognized_persons:
                        # Calculate confidence with lighting assessment
                        person_crop = frame[detection.bbox.y1:detection.bbox.y2, 
                                           detection.bbox.x1:detection.bbox.x2]
                        lighting_quality, _ = assess_lighting_quality(person_crop)
                        
                        # Use temporal tracking if person is recognized
                        final_confidence = detection.recognition_confidence
                        if hasattr(detection, 'person_name') and detection.person_name != "unknown":
                            temporal_confidence = person_tracker.get_temporal_confidence(
                                frame_count, detection.person_name, detection.recognition_confidence, detection.bbox
                            )
                            final_confidence = calculate_recognition_confidence_score(
                                detection.recognition_confidence, lighting_quality, temporal_confidence
                            )
                            
                            # Add to tracker
                            person_tracker.add_detection(
                                frame_count, len(all_faces), detection.person_name, final_confidence, detection.bbox
                            )
                        
                        person_name = getattr(detection, 'person_name', None)
                        person_id = getattr(detection, 'person_id', 0)
                        
                        face_in_frame = FaceInFrame(
                            name=person_name if person_name and person_name != "unknown" else "Unknown",
                            bbox=detection.bbox,
                            recognition_status="recognized" if person_name and person_name != "unknown" else "unrecognized",
                            person_name=person_name,
                            face_encoding=[],  # Not needed for display
                            person_id=person_id,
                            detection_type=getattr(detection, 'detection_type', 'person')
                        )
                        all_faces.append(face_in_frame)

                # Reconcile face and person detections
                if face_recognition_available and person_detections:
                    all_faces = reconcile_detections(all_faces, person_detections)

                # Create visualization with multiple views
                visualization_urls = {}
                if all_faces:
                    # Convert FaceInFrame back to Detection format for visualization
                    vis_detections = []
                    for face in all_faces:
                        detection = Detection(
                            bbox=face.bbox,
                            confidence=getattr(face, 'confidence', 0.8),  # Default confidence if not available
                            class_name=getattr(face, 'detection_type', 'person')
                        )
                        detection.person_name = face.person_name
                        detection.recognition_confidence = getattr(face, 'confidence', 0.8)
                        detection.detection_type = getattr(face, 'detection_type', 'person')
                        detection.person_id = getattr(face, 'person_id', 0)
                        vis_detections.append(detection)
                    
                    # Create different visualization views
                    visualization_urls = {
                        "combined": create_visualization_image(frame, vis_detections, frame_count, user_id, "combined"),
                        "person": create_visualization_image(frame, vis_detections, frame_count, user_id, "person"),
                        "face": create_visualization_image(frame, vis_detections, frame_count, user_id, "face")
                    }

                # Create frame analysis
                frame_analysis = FaceRecognitionFrame(
                    frame_number=frame_count,
                    detections=all_faces,
                    visualization_url=visualization_urls.get("combined")
                )
                frame_analyses.append(frame_analysis)

            frame_count += 1

        cap.release()

        # Calculate recognition statistics
        total_recognized = 0
        total_unrecognized = 0
        for frame_analysis in frame_analyses:
            for face in frame_analysis.detections:
                if face.recognition_status == "recognized":
                    total_recognized += 1
                else:
                    total_unrecognized += 1
        
        # Convert all numpy types to Python native types for JSON serialization
        converted_frame_analyses = convert_numpy_to_python(frame_analyses)
        
        result = FaceRecognitionAnalysis(
            video_path=video_path,
            total_frames=int(frame_count),
            processed_frames=int(len(frame_analyses)),
            recognized_faces=int(total_recognized),
            unrecognized_faces=int(total_unrecognized),
            detections=converted_frame_analyses
        )
        
        return result

    except Exception as e:
        print(f"Error in enhanced video analysis: {e}")
        return FaceRecognitionAnalysis(
            video_path=video_path,
            total_frames=0,
            processed_frames=0,
            recognized_faces=0,
            unrecognized_faces=0,
            detections=[]
        )