from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import List, Dict, Any
import os
import tempfile
import logging
import time

from .external_service import ExternalSegmentationService  
from .segment_manager import SegmentManager
from .client import ReplicateSegmentationClient
from ..storage.persistent_storage import PersistentStorage
from .schemas import (
    SegmentationRequest, SegmentationResponse, VerificationRequest,
    PermissionRequest, AccessCheckRequest, AccessCheckResponse,
    VisualizationRequest, VideoSegmentationRequest
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
segmentation_service = ExternalSegmentationService()
segment_manager = SegmentManager()
replicate_client = ReplicateSegmentationClient()
storage = PersistentStorage()

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

@router.post("/segment/replicate", response_model=SegmentationResponse)
async def segment_image_replicate(request: SegmentationRequest):
    """Segment areas in an image using Replicate Grounding DINO API"""
    try:
        # Use Replicate client for segmentation
        result = replicate_client.segment_and_process(
            image_path=request.image_path,
            user_id=request.user_id,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold
        )
        
        if result["status"] == "success":
            # Store segmentation results using storage module
            segmentation_data = {
                "segments": result["segments"],
                "image_path": request.image_path,
                "processing_time": result.get("processing_time", 0),
                "total_detections": result.get("total_detections", 0),
                "visualization_url": result.get("visualization_url"),
                "source": "replicate_grounding_dino",
                "created_at": time.time()
            }

            # Save to persistent storage
            storage_path = storage.save_segmentation_data(request.user_id, segmentation_data)
            logger.info(f"Saved segmentation results to {storage_path}")
            
            # Also store in segment manager for API compatibility
            for segment_data in result["segments"]:
                segment_manager.add_segment(segment_data, request.user_id)
        
        return SegmentationResponse(
            status=result["status"],
            segments=result.get("segments", []),
            image_path=None,
            error=result.get("error"),
            visualization_url=result.get("visualization_url")
        )
        
    except Exception as e:
        logger.error(f"Replicate segmentation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

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

@router.post("/segment/video", response_model=SegmentationResponse)
async def segment_video(request: VideoSegmentationRequest):
    """Segment areas in a video by extracting and processing unique frames"""
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        result = segmentation_service.process_video(
            request.video_path,
            request.user_id,
            request.box_threshold,
            request.text_threshold,
            max_frames=request.max_frames
        )
        
        # Store segments in manager if successful
        if result.get("status") == "success":
            for segment_data in result.get("segments", []):
                segment_manager.add_segment(segment_data, request.user_id)
        
        return SegmentationResponse(
            status=result["status"],
            segments=result.get("segments", []),
            image_path=result.get("video_path")
        )
        
    except Exception as e:
        logger.error(f"Video segmentation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video segmentation failed: {str(e)}")

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

@router.get("/segmentation/user/{user_id}")
async def get_user_segmentation_data(user_id: str):
    """Get stored segmentation data for a specific user"""
    try:
        segmentation_data = storage.load_segmentation_data(user_id)
        return {
            "user_id": user_id,
            "segmentation_data": segmentation_data
        }
    except Exception as e:
        logger.error(f"Error loading segmentation data for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading segmentation data: {str(e)}")

@router.post("/segment/frames/{user_id}")
async def segment_extracted_frames(user_id: str):
    """Process extracted frames from face recognition for segmentation"""
    try:
        # Get user's extracted frames directory
        user_frames_dir = os.path.join(storage.face_recognition_path, user_id, "extracted_frames")
        
        if not os.path.exists(user_frames_dir):
            raise HTTPException(status_code=404, detail="No extracted frames found for user")
        
        frame_files = [f for f in os.listdir(user_frames_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not frame_files:
            raise HTTPException(status_code=404, detail="No image frames found")
        
        processed_frames = []
        total_segments = []
        
        # Process each frame
        for frame_file in frame_files:
            frame_path = os.path.join(user_frames_dir, frame_file)
            
            try:
                # Segment the frame
                result = replicate_client.segment_and_process(
                    image_path=frame_path,
                    user_id=user_id
                )
                
                if result["status"] == "success":
                    # Add frame source info to segments
                    for segment in result["segments"]:
                        segment["source_frame"] = frame_file
                        segment["source_frame_path"] = frame_path
                    
                    total_segments.extend(result["segments"])
                    processed_frames.append({
                        "frame_file": frame_file,
                        "status": "success",
                        "segments_found": len(result["segments"]),
                        "visualization_url": result.get("visualization_url")
                    })
                else:
                    processed_frames.append({
                        "frame_file": frame_file,
                        "status": "error",
                        "error": result.get("error", "Unknown error"),
                        "segments_found": 0
                    })
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_file}: {str(e)}")
                processed_frames.append({
                    "frame_file": frame_file,
                    "status": "error", 
                    "error": str(e),
                    "segments_found": 0
                })
        
        # Store combined results
        if total_segments:
            segmentation_data = {
                "segments": total_segments,
                "source": "extracted_frames",
                "frames_processed": len(frame_files),
                "successful_frames": len([f for f in processed_frames if f["status"] == "success"]),
                "total_segments_found": len(total_segments),
                "created_at": time.time(),
                "frame_details": processed_frames
            }
            
            storage_path = storage.save_segmentation_data(user_id, segmentation_data)
            logger.info(f"Saved frame segmentation results to {storage_path}")
            
            # Also store in segment manager
            for segment_data in total_segments:
                segment_manager.add_segment(segment_data, user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "frames_processed": len(frame_files),
            "successful_frames": len([f for f in processed_frames if f["status"] == "success"]),
            "total_segments_found": len(total_segments),
            "frame_details": processed_frames
        }
        
    except Exception as e:
        logger.error(f"Error processing extracted frames for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing frames: {str(e)}")

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