import os
import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
from datetime import datetime
from PIL import Image
import logging

from groundingdino.util.inference import load_model, predict, annotate
from segment_anything import build_sam, SamPredictor
from GroundingDINO.groundingdino.util import box_ops

from .enums import AreaType
from .schemas import SegmentedArea, SegmentationResponse

logger = logging.getLogger(__name__)

class SegmentationService:
    def __init__(self, 
                 groundingdino_config_path: str = "/app/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 groundingdino_weights_path: str = "/app/segmentation_models/groundingdino_swint_ogc.pth",
                 sam_checkpoint_path: str = "/app/segmentation_models/sam_vit_h_4b8939.pth"):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.groundingdino_model = load_model(groundingdino_config_path, groundingdino_weights_path)
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint_path).to(self.device))
        
        # Area type detection prompts
        self.area_prompts = {
            AreaType.BACKYARD: "backyard . grass . lawn . garden . yard",
            AreaType.POOL: "pool . swimming pool . water",
            AreaType.GARAGE: "garage . garage door",
            AreaType.ROAD: "road . street . asphalt",
            AreaType.DRIVEWAY: "driveway . concrete . pavement",
            AreaType.FRONT_YARD: "front yard . entrance . lawn",
            AreaType.LAWN: "lawn . grass . green space",
            AreaType.PATIO: "patio . deck . outdoor furniture",
            AreaType.DECK: "deck . wooden deck . platform",
            AreaType.FENCE: "fence . fencing . boundary"
        }
        
    def detect_areas(self, image: np.ndarray, box_threshold: float = 0.3, text_threshold: float = 0.25) -> Dict[AreaType, Any]:
        """Detect different areas in the image using Grounding DINO"""
        results = {}
        
        for area_type, prompt in self.area_prompts.items():
            try:
                boxes, logits, phrases = predict(
                    model=self.groundingdino_model,
                    image=image,
                    caption=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
                
                if len(boxes) > 0:
                    results[area_type] = {
                        'boxes': boxes,
                        'logits': logits,
                        'phrases': phrases,
                        'confidence': float(torch.max(logits))
                    }
                    
            except Exception as e:
                logger.error(f"Error detecting {area_type}: {str(e)}")
                
        return results
    
    def segment_areas(self, image: np.ndarray, detected_areas: Dict[AreaType, Any]) -> Dict[AreaType, np.ndarray]:
        """Generate segmentation masks for detected areas"""
        self.sam_predictor.set_image(image)
        H, W, _ = image.shape
        segmented_areas = {}
        
        for area_type, detection in detected_areas.items():
            try:
                boxes = detection['boxes']
                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
                
                transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                    boxes_xyxy.to(self.device), image.shape[:2]
                )
                
                masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                
                segmented_areas[area_type] = masks.cpu().numpy()
                
            except Exception as e:
                logger.error(f"Error segmenting {area_type}: {str(e)}")
                
        return segmented_areas
    
    def calculate_dimensions(self, mask: np.ndarray, pixels_per_meter: float = 50.0) -> Dict[str, float]:
        """Calculate real-world dimensions from segmentation mask"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"width": 0, "height": 0, "area": 0}
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        width_meters = w / pixels_per_meter
        height_meters = h / pixels_per_meter
        area_meters = cv2.contourArea(largest_contour) / (pixels_per_meter ** 2)
        
        return {
            "width": width_meters,
            "height": height_meters,
            "area": area_meters
        }
    
    def mask_to_polygon(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Convert segmentation mask to polygon coordinates"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return [(int(point[0][0]), int(point[0][1])) for point in approx]
    
    def process_image(self, image_path: str, user_id: str, box_threshold: float = 0.3, text_threshold: float = 0.25) -> SegmentationResponse:
        """Main processing pipeline for home area segmentation"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return SegmentationResponse(status="error", error=f"Could not load image from {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            detected_areas = self.detect_areas(image_rgb, box_threshold, text_threshold)
            
            if not detected_areas:
                return SegmentationResponse(status="error", error="No areas detected in the image")
            
            segmented_areas = self.segment_areas(image_rgb, detected_areas)
            
            processed_segments = []
            for area_type, masks in segmented_areas.items():
                for i, mask in enumerate(masks):
                    mask_2d = mask[0] if len(mask.shape) > 2 else mask
                    
                    area_id = hashlib.md5(f"{user_id}_{area_type.value}_{i}_{datetime.now()}".encode()).hexdigest()
                    
                    dimensions = self.calculate_dimensions(mask_2d)
                    polygon = self.mask_to_polygon(mask_2d)
                    
                    if not polygon:
                        continue
                    
                    processed_segments.append({
                        "area_id": area_id,
                        "area_type": area_type.value,
                        "polygon": polygon,
                        "confidence": detected_areas[area_type]['confidence'],
                        "dimensions": dimensions,
                        "verified": False
                    })
            
            return SegmentationResponse(
                status="success",
                segments=processed_segments,
                image_path=image_path
            )
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return SegmentationResponse(status="error", error=str(e))
    
    def generate_visualization(self, image_path: str, segments_data: List[Dict[str, Any]]) -> np.ndarray:
        """Generate visualization image with segmented areas overlaid"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        colors = [
            (255, 0, 0, 128),    # Red
            (0, 255, 0, 128),    # Green
            (0, 0, 255, 128),    # Blue
            (255, 255, 0, 128),  # Yellow
            (255, 0, 255, 128),  # Magenta
            (0, 255, 255, 128),  # Cyan
        ]
        
        overlay = image_rgb.copy()
        
        for i, segment in enumerate(segments_data):
            color = colors[i % len(colors)]
            polygon = segment.get('polygon', [])
            
            if polygon:
                points = np.array(polygon, np.int32)
                cv2.fillPoly(overlay, [points], color[:3])
                cv2.polylines(overlay, [points], True, color[:3], 2)
                
                center_x = int(np.mean([p[0] for p in polygon]))
                center_y = int(np.mean([p[1] for p in polygon]))
                
                label = f"{segment.get('area_type', 'unknown')}"
                if segment.get('verified', False):
                    label += " ✓"
                
                cv2.putText(overlay, label, (center_x, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
        return result