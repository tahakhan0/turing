from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from .schemas import VideoAnalysis, FrameAnalysis, Detection, BoundingBox

# Get the base URL for serving images (can be overridden with environment variable)
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Initialize the YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# --- PERFORMANCE OPTIMIZATION ---
# Process every Nth frame to speed up video analysis.
# A value of 5 means we process 1 frame and skip the next 4.
# Increase this value for more speed, or decrease it for more detailed analysis.
FRAME_INTERVAL = 5

# Directory for storing visualization images
VISUALIZATIONS_DIR = "/app/static/visualizations"


def create_visualization_image(frame, detections, frame_number, user_id):
    """
    Create a visualization image with bounding boxes drawn on detected persons.
    Returns the relative URL path to the saved image.
    """
    # Ensure the visualizations directory exists
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    # Create a copy of the frame to draw on
    vis_frame = frame.copy()

    # Draw bounding boxes for each detection
    for detection in detections:
        bbox = detection.bbox
        confidence = detection.confidence

        # Draw rectangle
        cv2.rectangle(vis_frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)

        # Add confidence label
        label = f"Person: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_frame, (bbox.x1, bbox.y1 - label_size[1] - 10),
                     (bbox.x1 + label_size[0], bbox.y1), (0, 255, 0), -1)
        cv2.putText(vis_frame, label, (bbox.x1, bbox.y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Save the image with a unique filename
    filename = f"{user_id}_frame_{frame_number}.jpg"
    filepath = os.path.join(VISUALIZATIONS_DIR, filename)
    cv2.imwrite(filepath, vis_frame)

    # Return the absolute URL that can be accessed from anywhere
    return f"{BASE_URL}/static/visualizations/{filename}"


def batch_process_video_for_person_detection(video_paths: list, user_id: str) -> VideoAnalysis:
    """
    Processes the first video in a list for person detection using YOLOv8,
    with frame skipping for performance.
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
                        detections.append(detection)

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
