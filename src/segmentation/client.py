import requests
import base64
import os
import logging
import datetime
from typing import Dict, Any, Optional
import time
from urllib.parse import urlparse
from dotenv import load_dotenv
import replicate
from pathlib import Path
from ..storage.persistent_storage import PersistentStorage

# Load environment variables from .env file
load_dotenv()

# Initialize persistent storage
storage = PersistentStorage()

logger = logging.getLogger(__name__)

SEGMENT_QUERY=""""
grass, lawn, windows, chairs, mower, lawn mower, pipe, water, pool, fence, shed, cart, pallet,
fence, garbage, car, driveway
"""

class ReplicateSegmentationClient:
    """Client for Replicate Grounding DINO API"""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN environment variable must be set")
        self.model_version = "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa"
    
    def _is_url(self, path: str) -> bool:
        """Check if the given path is a URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _encode_local_image(self, image_path: str) -> str:
        """Encode a local image file as base64 data URI"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:application/octet-stream;base64,{encoded_string}"
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise e
    
    def _save_visualization_file(self, file_output, user_id: str) -> Optional[str]:
        """Save the FileOutput visualization image locally"""
        if not file_output:
            return None
            
        try:
            # Create directory for saving visualizations using persistent storage
            viz_dir = Path(storage.base_path) / "segmentation_visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.datetime.now().isoformat()
            filename = f"segmentation_{user_id}_{timestamp}.png"
            local_path = viz_dir / filename
            
            # Write FileOutput to local file
            with open(local_path, 'wb') as f:
                f.write(file_output.read())
            
            # Return relative URL path for serving
            relative_path = f"/static/segmentation_visualizations/{filename}"
            logger.info(f"Saved visualization to {relative_path}")
            return relative_path
            
        except Exception as e:
            logger.error(f"Failed to save visualization file: {e}")
            return None
    
    def segment_image(self, 
                     image_path: str, 
                     query: str = SEGMENT_QUERY,
                     box_threshold: float = 0.3,
                     text_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Segment an image using Replicate Grounding DINO API
        
        Args:
            image_path: Path to local image file or URL
            query: Comma-separated list of objects to detect
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold

        Returns:
            API response containing detections
        """
        # Prepare image input
        if self._is_url(image_path):
            image_input = image_path
            logger.info(f"Using image URL: {image_path}")
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image_input = self._encode_local_image(image_path)
            logger.info(f"Encoded local image: {image_path}")

        # Prepare request payload
        payload = {
                "image": image_input,
                "query": query,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
            "show_visualisation":True
        }

        logger.info(f"Starting segmentation with query: {query}")

        # Create prediction
        prediction_data = replicate.run(self.model_version, input=payload)
        return prediction_data

    def process_detections(self, api_response: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Process API response into our internal format
        
        Args:
            api_response: Raw API response from Replicate
            user_id: User ID for tracking
            
        Returns:
            Processed segmentation data
        """
        try:
            detections = api_response["detections"]
            
            processed_segments = []
            
            for i, detection in enumerate(detections):
                # Extract bounding box coordinates
                bbox = detection.get("bbox", [])
                if len(bbox) != 4:
                    logger.warning(f"Invalid bbox format: {bbox}")
                    continue
                
                x1, y1, x2, y2 = bbox
                
                # Create segment data in our format
                area_id = f"area_{user_id}_{int(time.time())}_{i}"
                
                segment = {
                    "area_id": area_id,
                    "area_type": detection.get("label"),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1), 
                        "x2": int(x2),
                        "y2": int(y2)
                    },
                    "confidence": float(detection.get("confidence", 0.0)),
                    "label": detection.get("label"),
                    "user_id": user_id,
                    "verified": False,
                    "created_at": time.time(),
                    "source": "replicate_grounding_dino"
                }
                
                processed_segments.append(segment)
            
            # Handle visualization file - save FileOutput to local file
            visualization_url = None
            result_image = api_response.get("result_image")
            if result_image:
                visualization_url = self._save_visualization_file(result_image, user_id)

            print("visualization_url",visualization_url)

            result = {
                "status": "success",
                "segments": processed_segments,
                "total_detections": len(processed_segments),
                "visualization_url": visualization_url,
            }
            
            logger.info(f"Processed {len(processed_segments)} detections for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing detections: {e}")
            return {
                "status": "error",
                "error": str(e),
                "segments": []
            }
    
    def segment_and_process(self, 
                          image_path: str, 
                          user_id: str,
                          query: str = SEGMENT_QUERY,
                          box_threshold: float = 0.25,
                          text_threshold: float = 0.25,
                          show_visualisation: bool = True) -> Dict[str, Any]:
        """
        Complete workflow: segment image and process results
        
        Args:
            image_path: Path to local image file or URL
            user_id: User ID for tracking
            query: Comma-separated list of objects to detect
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            show_visualisation: Whether to generate visualization image
            
        Returns:
            Processed segmentation results
        """
        try:
            # Call segmentation API
            api_response = self.segment_image(
                image_path=image_path,
                query=query,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            
            # Process the results
            processed_results = self.process_detections(api_response, user_id)
            return processed_results
            
        except Exception as e:
            logger.error(f"Complete segmentation workflow failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "segments": []
            }

