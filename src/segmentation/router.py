from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import List, Dict, Any
import os
import tempfile
import logging

from .external_service import ExternalSegmentationService  
from .segment_manager import SegmentManager
from .schemas import (
    SegmentationRequest, SegmentationResponse, VerificationRequest,
    PermissionRequest, AccessCheckRequest, AccessCheckResponse,
    VisualizationRequest
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
segmentation_service = ExternalSegmentationService()
segment_manager = SegmentManager()

@router.post("/segment", response_model=SegmentationResponse)
async def segment_image(request: SegmentationRequest):
    """Segment areas in an image"""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    result = segmentation_service.process_image(
        request.image_path,
        request.user_id,
        request.box_threshold,
        request.text_threshold
    )
    
    # Store segments in manager if successful
    if result.status == "success":
        for segment_data in result.segments:
            segment_manager.add_segment(segment_data, request.user_id)
    
    return result

@router.post("/segment/upload")
async def segment_uploaded_image(user_id: str, file: UploadFile = File(...)):
    """Upload and segment an image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        result = segmentation_service.process_image(temp_path, user_id)
        
        if result.status == "success":
            for segment_data in result.segments:
                segment_manager.add_segment(segment_data, user_id)
        
        return result
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

@router.post("/segment/verify")
async def verify_segment(request: VerificationRequest):
    """Verify or reject a segmented area"""
    result = segment_manager.verify_segment(
        request.area_id,
        request.user_id,
        request.approved
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.get("/segments/user/{user_id}")
async def get_user_segments(user_id: str):
    """Get all segments for a specific user"""
    segments = segment_manager.get_user_segments(user_id)
    return {"segments": segments}

@router.post("/permission/add")
async def add_permission(request: PermissionRequest):
    """Add access permission for a labeled person to an area (use area_id='all' for all areas)"""
    result = segment_manager.add_person_permission(
        request.person_name,
        request.user_id,
        request.area_id,
        request.allowed,
        [condition.value for condition in request.conditions]
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/access/check", response_model=AccessCheckResponse)
async def check_access(request: AccessCheckRequest):
    """Check if a labeled person has access to a specific area"""
    result = segment_manager.check_access(
        request.person_name,
        request.user_id,
        request.area_id,
        request.context
    )
    
    return AccessCheckResponse(
        allowed=result["allowed"],
        reason=result["reason"]
    )

@router.get("/permissions/person/{person_name}/user/{user_id}")
async def get_person_permissions(person_name: str, user_id: str):
    """Get all permissions for a labeled person"""
    permissions = segment_manager.get_person_permissions(person_name, user_id)
    return {"permissions": permissions}

@router.get("/permissions/area/{area_id}/user/{user_id}")
async def get_area_permissions(area_id: str, user_id: str):
    """Get all permissions for a specific area"""
    permissions = segment_manager.get_area_permissions(area_id, user_id)
    return {"permissions": permissions}

@router.get("/people/{user_id}")
async def get_labeled_people(user_id: str):
    """Get all labeled people from face recognition for a user"""
    people = segment_manager.get_labeled_people(user_id)
    return {"people": people}

@router.delete("/permission/person/{person_name}/user/{user_id}/area/{area_id}")
async def remove_permission(person_name: str, user_id: str, area_id: str):
    """Remove a specific permission"""
    result = segment_manager.remove_permission(person_name, user_id, area_id)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result

@router.post("/visualization")
async def generate_visualization(request: VisualizationRequest):
    """Generate visualization image with segmented areas"""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        # Get segment data for visualization
        if request.area_ids:
            segments_data = []
            for area_id in request.area_ids:
                segment = segment_manager.get_segment(area_id)
                if segment:
                    segments_data.append(segment)
        else:
            # Use all segments from the image
            segments_data = []
        
        # Generate visualization
        result_path = segmentation_service.create_visualization(
            request.image_path,
            segments_data
        )
        
        return FileResponse(
            result_path,
            media_type="image/png",
            filename="segmentation_visualization.png"
        )
            
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating visualization")

@router.get("/health")
async def health_check():
    """Health check endpoint - includes external service status"""
    try:
        # Check external segmentation service
        external_health = segmentation_service.check_health()
        
        return {
            "status": "healthy", 
            "service": "Turing Segmentation API (Client)",
            "external_service": external_health,
            "mode": "external_api"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "Turing Segmentation API (Client)", 
            "external_service": {"status": "unhealthy", "error": str(e)},
            "mode": "external_api"
        }