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

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: scikit-image not available, using basic frame comparison")

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
    
    def _frames_are_similar(self, frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.9) -> bool:
        """Check if two frames are similar using structural similarity or basic comparison"""
        # Resize frames for faster comparison
        small_frame1 = cv2.resize(frame1, (100, 100))
        small_frame2 = cv2.resize(frame2, (100, 100))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(small_frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(small_frame2, cv2.COLOR_BGR2GRAY)
        
        if SSIM_AVAILABLE:
            # Use SSIM if available
            similarity = ssim(gray1, gray2)
        else:
            # Fallback: use normalized cross-correlation
            # Calculate mean squared difference
            diff = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
            max_diff = 255.0 ** 2  # Maximum possible difference
            similarity = 1.0 - (diff / max_diff)
        
        return similarity > threshold
    
    def _extract_unique_frames(self, video_path: str, max_frames: int = 10) -> List[np.ndarray]:
        """Extract unique frames from video, avoiding duplicates"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval to sample across the video
        frame_interval = max(1, total_frames // (max_frames * 2))
        previous_frame = None
        
        logger.info(f"Extracting frames from video with {total_frames} total frames, interval: {frame_interval}")
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Only process every Nth frame
            if frame_count % frame_interval != 0:
                continue
            
            # Skip similar frames
            if previous_frame is not None and self._frames_are_similar(previous_frame, frame, threshold=0.85):
                continue
            
            frames.append(frame.copy())
            previous_frame = frame.copy()
            
            logger.info(f"Extracted frame {len(frames)}/{max_frames}")
        
        cap.release()
        logger.info(f"Extracted {len(frames)} unique frames from video")
        return frames
    
    def process_video(self, video_path: str, user_id: str, 
                     box_threshold: float = 0.3, text_threshold: float = 0.25,
                     custom_prompts: Optional[Dict[str, str]] = None,
                     max_frames: int = 10) -> Dict[str, Any]:
        """
        Process video for segmentation by extracting unique frames
        
        Args:
            video_path: Path to video file
            user_id: User ID for tracking
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            custom_prompts: Optional custom prompts, uses defaults if None
            max_frames: Maximum number of frames to extract and process
        
        Returns:
            Combined segmentation results from all frames
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Extract unique frames from video
            frames = self._extract_unique_frames(video_path, max_frames)
            
            if not frames:
                raise ValueError("No frames could be extracted from video")
            
            all_segments = []
            processing_results = []
            
            # Process each frame for segmentation
            for i, frame in enumerate(frames):
                try:
                    # Save frame as temporary image
                    temp_image_path = os.path.join(tempfile.gettempdir(), f"frame_{user_id}_{i}.jpg")
                    cv2.imwrite(temp_image_path, frame)
                    
                    # Process frame for segmentation
                    frame_result = self.process_image(
                        temp_image_path, 
                        user_id, 
                        box_threshold, 
                        text_threshold, 
                        custom_prompts
                    )
                    
                    # Add frame info to segments
                    if frame_result.get("status") == "success":
                        for segment in frame_result.get("segments", []):
                            segment["source_frame"] = i
                            segment["source_video"] = video_path
                            all_segments.append(segment)
                    
                    processing_results.append({
                        "frame_index": i,
                        "status": frame_result.get("status", "error"),
                        "segments_found": len(frame_result.get("segments", [])),
                        "processing_time": frame_result.get("processing_time", 0)
                    })
                    
                    # Clean up temporary file
                    if os.path.exists(temp_image_path):
                        os.unlink(temp_image_path)
                        
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {e}")
                    processing_results.append({
                        "frame_index": i,
                        "status": "error",
                        "error": str(e),
                        "segments_found": 0
                    })
            
            # Merge similar segments across frames
            merged_segments = self._merge_similar_segments(all_segments)
            
            total_processing_time = sum(r.get("processing_time", 0) for r in processing_results)
            
            return {
                "status": "success",
                "video_path": video_path,
                "frames_processed": len(frames),
                "segments": merged_segments,
                "total_segments_found": len(merged_segments),
                "processing_time": total_processing_time,
                "frame_results": processing_results,
                "parameters_used": {
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                    "max_frames": max_frames
                }
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise e
    
    def _merge_similar_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar segments found across multiple frames"""
        if not segments:
            return []
        
        merged = []
        used_indices = set()
        
        for i, segment in enumerate(segments):
            if i in used_indices:
                continue
            
            # Find similar segments
            similar_segments = [segment]
            used_indices.add(i)
            
            for j, other_segment in enumerate(segments[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Check if segments are similar (same area type and overlapping location)
                if (segment["area_type"] == other_segment["area_type"] and 
                    self._segments_overlap(segment, other_segment)):
                    similar_segments.append(other_segment)
                    used_indices.add(j)
            
            # Merge the similar segments
            merged_segment = self._create_merged_segment(similar_segments)
            merged.append(merged_segment)
        
        logger.info(f"Merged {len(segments)} segments into {len(merged)} unique areas")
        return merged
    
    def _segments_overlap(self, seg1: Dict[str, Any], seg2: Dict[str, Any], threshold: float = 0.3) -> bool:
        """Check if two segments overlap significantly"""
        try:
            bbox1 = seg1.get("bbox", {})
            bbox2 = seg2.get("bbox", {})
            
            # Calculate intersection
            x1 = max(bbox1.get("x1", 0), bbox2.get("x1", 0))
            y1 = max(bbox1.get("y1", 0), bbox2.get("y1", 0))
            x2 = min(bbox1.get("x2", 0), bbox2.get("x2", 0))
            y2 = min(bbox1.get("y2", 0), bbox2.get("y2", 0))
            
            if x2 <= x1 or y2 <= y1:
                return False
            
            intersection_area = (x2 - x1) * (y2 - y1)
            
            # Calculate union
            area1 = (bbox1.get("x2", 0) - bbox1.get("x1", 0)) * (bbox1.get("y2", 0) - bbox1.get("y1", 0))
            area2 = (bbox2.get("x2", 0) - bbox2.get("x1", 0)) * (bbox2.get("y2", 0) - bbox2.get("y1", 0))
            union_area = area1 + area2 - intersection_area
            
            # Calculate IoU (Intersection over Union)
            iou = intersection_area / union_area if union_area > 0 else 0
            return iou > threshold
            
        except Exception:
            return False
    
    def _create_merged_segment(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a merged segment from multiple similar segments"""
        if len(segments) == 1:
            return segments[0]
        
        # Use the segment with highest confidence as base
        base_segment = max(segments, key=lambda s: s.get("confidence", 0))
        
        # Average confidence scores
        avg_confidence = sum(s.get("confidence", 0) for s in segments) / len(segments)
        
        # Collect source frames
        source_frames = list(set(s.get("source_frame", 0) for s in segments))
        
        merged = base_segment.copy()
        merged.update({
            "confidence": avg_confidence,
            "source_frames": source_frames,
            "merge_count": len(segments),
            "merged_from": [s.get("area_id") for s in segments]
        })
        
        return merged