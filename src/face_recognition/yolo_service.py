from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import face_recognition  # Import the face-recognition library
from .schemas import VideoAnalysis, FrameAnalysis, Detection, BoundingBox

# Get the base URL for serving images (can be overridden with environment variable)
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Initialize the YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# --- PERFORMANCE OPTIMIZATION ---
# Process every Nth frame to speed up video analysis.
FRAME_INTERVAL = 5

# Directory for storing visualization images
VISUALIZATIONS_DIR = "/app/static/visualizations"


def create_visualization_image(frame, detections, frame_number, user_id):
    """
    Create a visualization image with bounding boxes drawn on detected persons and faces.
    - People will be in GREEN.
    - Faces will be in BLUE.
    Returns the relative URL path to the saved image.
    """
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    vis_frame = frame.copy()

    for detection in detections:
        bbox = detection.bbox

        # Determine color and label based on class name
        if detection.class_name == "person":
            color = (0, 255, 0)  # Green for person
            label = f"Person: {detection.confidence:.2f}"
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
    and faces (using face-recognition).
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
                        person_detection = Detection(
                            bbox=person_bbox,
                            confidence=float(confidences[i]),
                            class_name="person"
                        )
                        all_detections.append(person_detection)

                        # --- 2. Detect Face within the Person's BBox ---
                        # Crop the person from the frame for focused face detection
                        person_image = rgb_frame[box[1]:box[3], box[0]:box[2]]

                        # Find all face locations in the cropped person image
                        face_locations = face_recognition.face_locations(person_image)

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
