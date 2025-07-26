"""
External Segmentation Service Client
Connects to the standalone turing-segmentation service running on RunPod
"""

import os
import requests
from fastapi import HTTPException
import tempfile
import base64
import json
import logging
from typing import Dict, Any, Optional, List
from PIL import Image
import io
import cv2
import numpy as np

from .schemas import SegmentationResponse

logger = logging.getLogger(__name__)

class ExternalSegmentationService:
    """Client for external turing-segmentation service"""
    
    def __init__(self):
        # Configuration from environment variables
        self.base_url = os.getenv("TURING_SEGMENTATION_URL", "http://localhost:8000")
        self.api_key = os.getenv("TURING_SEGMENTATION_API_KEY")
        
        if not self.api_key:
            logger.error("TURING_SEGMENTATION_API_KEY environment variable not set")
            raise ValueError("API key required for external segmentation service")
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"External segmentation service configured: {self.base_url}")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image file as base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise e
    
    def _get_default_prompts(self) -> Dict[str, str]:
        """Get default area detection prompts from external service"""
        try:
            response = requests.get(
                f"{self.base_url}/prompts/default",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("prompts", {})
        except Exception as e:
            logger.error(f"Failed to get default prompts: {e}")
            # Fallback to hardcoded defaults
            return {
                "backyard": "backyard . grass . lawn . garden . yard",
                "pool": "pool . swimming pool . water",
                "garage": "garage . garage door",
                "road": "road . street . asphalt",
                "driveway": "driveway . concrete . pavement",
                "front_yard": "front yard . entrance . lawn",
                "lawn": "lawn . grass . green space",
                "patio": "patio . deck . outdoor furniture",
                "deck": "deck . wooden deck . platform",
                "fence": "fence . fencing . boundary"
            }
    
    def process_image(self, image_path: str, user_id: str, 
                     box_threshold: float = 0.3, text_threshold: float = 0.25,
                     custom_prompts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process image using external segmentation service
        
        Args:
            image_path: Path to image file
            user_id: User ID for tracking
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            custom_prompts: Optional custom prompts, uses defaults if None
        
        Returns:
            Segmentation results in turing format
        """
        try:
            # Encode image
            image_base64 = self._encode_image(image_path)
            
            # Get prompts
            if custom_prompts is None:
                prompts = self._get_default_prompts()
            else:
                prompts = custom_prompts
            
            # Prepare request payload
            payload = {
                "image_base64": image_base64,
                "prompts": prompts,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "return_masks": True,
                "estimate_dimensions": True
            }
            
            # Make API request
            response = requests.post(
                f"{self.base_url}/segment",
                headers=self.headers,
                json=payload,
                timeout=120  # 2 minutes timeout for processing
            )
            response.raise_for_status()
            
            # Parse response
            external_result = response.json()
            
            if not external_result.get("success", False):
                raise Exception(f"External service error: {external_result.get('message', 'Unknown error')}")
            
            # Convert external format to turing format
            turing_result = self._convert_to_turing_format(
                external_result, image_path, user_id
            )
            
            logger.info(f"Successfully processed image {image_path} for user {user_id}")
            return turing_result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"External service request failed: {e}")
            raise HTTPException(status_code=503, detail=f"External segmentation service unavailable: {str(e)}")
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise e
    
    def _convert_to_turing_format(self, external_result: Dict[str, Any], 
                                 image_path: str, user_id: str) -> Dict[str, Any]:
        """Convert external service response to turing format"""
        try:
            segments = []
            
            for area in external_result.get("detected_areas", []):
                # Generate unique area ID
                area_id = f"area_{user_id}_{len(segments)}_{area['area_type']}"
                
                # Convert to turing segment format
                segment = {
                    "area_id": area_id,
                    "area_type": area["area_type"],
                    "polygon": area["polygon"],
                    "confidence": area["confidence"],
                    "dimensions": {
                        "width": area.get("estimated_width_meters", 0),
                        "height": area.get("estimated_height_meters", 0),
                        "area": area.get("estimated_area_sqm", 0)
                    },
                    "user_id": user_id,
                    "verified": False,
                    "bbox": area["bbox"],
                    "mask_encoding": area["mask_encoding"],
                    "area_pixels": area["area_pixels"],
                    "center_point": area["center_point"]
                }
                segments.append(segment)
            
            return {
                "status": "success",
                "segments": segments,
                "image_path": image_path,
                "processing_time": external_result.get("processing_time_seconds", 0),
                "total_areas_found": len(segments),
                "model_info": external_result.get("model_info", {}),
                "parameters_used": external_result.get("parameters_used", {})
            }
            
        except Exception as e:
            logger.error(f"Failed to convert external result to turing format: {e}")
            raise e
    
    def create_visualization(self, image_path: str, segments: List[Dict[str, Any]]) -> str:
        """
        Create visualization using external service
        
        Args:
            image_path: Path to original image
            segments: List of segments to visualize
        
        Returns:
            Path to visualization image
        """
        try:
            # Encode original image
            image_base64 = self._encode_image(image_path)
            
            # Prepare segments for external service
            external_segments = []
            for segment in segments:
                external_segment = {
                    "area_type": segment["area_type"],
                    "confidence": segment["confidence"],
                    "polygon": segment["polygon"],
                    "bbox": segment["bbox"],
                    "mask_encoding": segment["mask_encoding"],
                    "area_pixels": segment["area_pixels"],
                    "center_point": segment["center_point"]
                }
                external_segments.append(external_segment)
            
            # Make visualization request
            payload = {
                "image_base64": image_base64,
                "segments_json": json.dumps(external_segments)
            }
            
            response = requests.post(
                f"{self.base_url}/visualization",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # Get visualization image
            result = response.json()
            viz_base64 = result.get("visualization_base64")
            
            if not viz_base64:
                raise Exception("No visualization image returned")
            
            # Save visualization to temporary file
            viz_data = base64.b64decode(viz_base64)
            viz_filename = f"viz_{os.path.basename(image_path)}"
            viz_path = os.path.join(tempfile.gettempdir(), viz_filename)
            
            with open(viz_path, "wb") as f:
                f.write(viz_data)
            
            logger.info(f"Visualization created: {viz_path}")
            return viz_path
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            raise e
    
    def check_health(self) -> Dict[str, Any]:
        """Check health of external segmentation service"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}