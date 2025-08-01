from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
import os
import tempfile
import logging
import time
import json
import asyncio
from typing import Dict

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
segment_manager = SegmentManager()
replicate_client = ReplicateSegmentationClient()
storage = PersistentStorage()

# WebSocket connection manager
class SegmentationWebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected for user {user_id}")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected for user {user_id}")
    
    async def send_frame_update(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending WebSocket message to user {user_id}: {e}")
                self.disconnect(user_id)

websocket_manager = SegmentationWebSocketManager()

@router.websocket("/ws/segmentation/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time segmentation updates"""
    await websocket_manager.connect(websocket, user_id)
    try:
        # Send initial connection confirmation
        await websocket_manager.send_frame_update(user_id, {
            "type": "connection_established",
            "user_id": user_id,
            "timestamp": time.time()
        })
        
        # Keep connection alive
        while True:
            try:
                # Wait for any client messages (ping/pong)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket_manager.send_frame_update(user_id, {
                        "type": "pong",
                        "timestamp": time.time()
                    })
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for user {user_id}: {e}")
                break
    finally:
        websocket_manager.disconnect(user_id)

@router.post("/segment", response_model=SegmentationResponse)
async def segment_image(request: SegmentationRequest):
    """Segment areas in an image using Replicate Grounding DINO API"""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
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
        logger.error(f"Segmentation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@router.post("/segment/upload")
async def segment_uploaded_image(user_id: str, file: UploadFile = File(...)):
    """Upload and segment an image using Replicate Grounding DINO API"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Use Replicate client for segmentation
        result = replicate_client.segment_and_process(
            image_path=temp_path,
            user_id=user_id
        )
        
        if result["status"] == "success":
            # Store segmentation results using storage module
            segmentation_data = {
                "segments": result["segments"],
                "image_path": temp_path,
                "processing_time": result.get("processing_time", 0),
                "total_detections": result.get("total_detections", 0),
                "visualization_url": result.get("visualization_url"),
                "source": "replicate_grounding_dino",
                "created_at": time.time()
            }

            # Save to persistent storage
            storage_path = storage.save_segmentation_data(user_id, segmentation_data)
            logger.info(f"Saved segmentation results to {storage_path}")
            
            # Also store in segment manager for API compatibility
            for segment_data in result["segments"]:
                segment_manager.add_segment(segment_data, user_id)
        
        return SegmentationResponse(
            status=result["status"],
            segments=result.get("segments", []),
            image_path=None,
            error=result.get("error"),
            visualization_url=result.get("visualization_url")
        )
        
    except Exception as e:
        logger.error(f"Upload segmentation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

@router.post("/segment/video", response_model=SegmentationResponse)
async def segment_video(request: VideoSegmentationRequest):  # noqa: ARG001
    """Segment areas in a video by extracting and processing unique frames (deprecated - use /segment/frames/{user_id})"""
    # This endpoint is deprecated in favor of the frames endpoint
    # which processes already extracted frames from face recognition
    raise HTTPException(status_code=410, detail="This endpoint is deprecated. Use /segment/frames/{user_id} instead.")

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

@router.get("/visualizations/{user_id}")
async def get_segmentation_visualizations(user_id: str):
    """Get a list of available segmentation visualization images for a user."""
    try:
        visualizations_dir = os.path.join(storage.base_path, "segmentation_visualizations")
        if not os.path.exists(visualizations_dir):
            return {"visualizations": []}

        user_visualizations = []
        for filename in os.listdir(visualizations_dir):
            if filename.startswith(f"segmentation_{user_id}") and filename.endswith(".png"):
                user_visualizations.append(f"/static/segmentation_visualizations/{filename}")
        
        return {"visualizations": user_visualizations}
    except Exception as e:
        logger.error(f"Error getting segmentation visualizations for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting visualizations")

@router.get("/user/{user_id}")
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
    """Process extracted frames from face recognition for segmentation with real-time WebSocket updates"""
    try:
        # Get user's extracted frames directory
        user_frames_dir = os.path.join(storage.face_recognition_path, user_id, "extracted_frames")
        
        if not os.path.exists(user_frames_dir):
            raise HTTPException(status_code=404, detail="No extracted frames found for user")
        
        frame_files = [f for f in os.listdir(user_frames_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not frame_files:
            raise HTTPException(status_code=404, detail="No image frames found")
        
        # Send batch started notification
        await websocket_manager.send_frame_update(user_id, {
            "type": "batch_started",
            "user_id": user_id,
            "total_frames": len(frame_files),
            "frame_files": frame_files,
            "timestamp": time.time()
        })
        
        processed_frames = []
        total_segments = []
        
        # Define WebSocket callback function
        async def ws_callback(message):
            await websocket_manager.send_frame_update(user_id, message)
        
        # Process each frame with real-time updates
        for frame_file in frame_files:
            frame_path = os.path.join(user_frames_dir, frame_file)
            
            try:
                # Segment the frame with WebSocket callback
                result = replicate_client.segment_and_process(
                    image_path=frame_path,
                    user_id=user_id,
                    websocket_callback=ws_callback,
                    frame_name=frame_file
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
        
        # Send batch completed notification
        await websocket_manager.send_frame_update(user_id, {
            "type": "batch_completed",
            "user_id": user_id,
            "frames_processed": len(frame_files),
            "successful_frames": len([f for f in processed_frames if f["status"] == "success"]),
            "total_segments_found": len(total_segments),
            "timestamp": time.time()
        })
        
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
        
        # Send batch failed notification
        await websocket_manager.send_frame_update(user_id, {
            "type": "batch_failed",
            "user_id": user_id,
            "error": str(e),
            "timestamp": time.time()
        })
        
        raise HTTPException(status_code=500, detail=f"Error processing frames: {str(e)}")

@router.post("/segment/frames/{user_id}/stream")
async def start_frame_streaming(user_id: str):
    """Start streaming segmentation of extracted frames"""
    try:
        # Get user's extracted frames directory
        user_frames_dir = os.path.join(storage.face_recognition_path, user_id, "extracted_frames")
        
        if not os.path.exists(user_frames_dir):
            raise HTTPException(status_code=404, detail="No extracted frames found for user")
        
        frame_files = [f for f in os.listdir(user_frames_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not frame_files:
            raise HTTPException(status_code=404, detail="No image frames found")
        
        # Initialize frame status storage
        frame_status_data = {
            "user_id": user_id,
            "total_frames": len(frame_files),
            "started_at": time.time(),
            "status": "processing",
            "frames": {
                frame_file: {
                    "status": "pending",
                    "frame_file": frame_file,
                    "frame_path": os.path.join(user_frames_dir, frame_file),
                    "created_at": time.time()
                } for frame_file in frame_files
            }
        }
        
        # Save initial status
        storage.save_frame_status(user_id, frame_status_data)
        
        # Start background processing (fire and forget)
        import asyncio
        asyncio.create_task(process_frames_background(user_id, frame_status_data))
        
        return {
            "status": "started",
            "user_id": user_id,
            "total_frames": len(frame_files),
            "message": "Frame processing started. Use /status endpoint to check progress."
        }
        
    except Exception as e:
        logger.error(f"Error starting frame streaming for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting frame processing: {str(e)}")

@router.get("/segment/frames/{user_id}/status")
async def get_frame_status(user_id: str):
    """Get current status of frame processing"""
    try:
        frame_status = storage.load_frame_status(user_id)
        if not frame_status:
            raise HTTPException(status_code=404, detail="No frame processing found for user")
        
        # Count statuses
        frames = frame_status.get("frames", {})
        pending_count = sum(1 for f in frames.values() if f["status"] == "pending")
        processing_count = sum(1 for f in frames.values() if f["status"] == "processing")
        completed_count = sum(1 for f in frames.values() if f["status"] == "completed")
        error_count = sum(1 for f in frames.values() if f["status"] == "error")
        
        return {
            "user_id": user_id,
            "overall_status": frame_status.get("status", "unknown"),
            "total_frames": frame_status.get("total_frames", 0),
            "pending_frames": pending_count,
            "processing_frames": processing_count,
            "completed_frames": completed_count,
            "error_frames": error_count,
            "frames": list(frames.values()),
            "started_at": frame_status.get("started_at"),
            "updated_at": frame_status.get("updated_at", time.time())
        }
        
    except Exception as e:
        logger.error(f"Error getting frame status for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting frame status: {str(e)}")

@router.post("/segment/frames/{user_id}/verify/{frame_id}")
async def verify_frame(user_id: str, frame_id: str, request: dict):
    """Verify (approve/reject) a processed frame"""
    try:
        approved = request.get("approved", True)
        
        frame_status = storage.load_frame_status(user_id)
        if not frame_status:
            raise HTTPException(status_code=404, detail="No frame processing found for user")
        
        frames = frame_status.get("frames", {})
        if frame_id not in frames:
            raise HTTPException(status_code=404, detail="Frame not found")
        
        frame_data = frames[frame_id]
        if frame_data["status"] != "completed":
            raise HTTPException(status_code=400, detail="Frame not ready for verification")
        
        # Update verification status
        frame_data["verified"] = approved
        frame_data["verified_at"] = time.time()
        
        # Save updated status
        frame_status["updated_at"] = time.time()
        storage.save_frame_status(user_id, frame_status)
        
        return {
            "status": "success",
            "frame_id": frame_id,
            "verified": approved,
            "message": f"Frame {'approved' if approved else 'rejected'}"
        }
        
    except Exception as e:
        logger.error(f"Error verifying frame {frame_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error verifying frame: {str(e)}")

async def process_frames_background(user_id: str, frame_status_data: dict):
    """Background task to process frames one by one"""
    try:
        frames = frame_status_data["frames"]
        
        for frame_id, frame_data in frames.items():
            try:
                # Update status to processing
                frame_data["status"] = "processing"
                frame_data["processing_started_at"] = time.time()
                storage.save_frame_status(user_id, frame_status_data)
                
                # Process the frame
                result = replicate_client.segment_and_process(
                    image_path=frame_data["frame_path"],
                    user_id=user_id
                )
                
                if result["status"] == "success":
                    # The visualization_url from replicate client is already a local web path
                    visualization_url = result.get("visualization_url")
                    
                    frame_data.update({
                        "status": "completed",
                        "visualization_url": visualization_url,  # This is already a local web path like /static/...
                        "web_visualization_path": visualization_url,  # Same as visualization_url
                        "segments_found": len(result.get("segments", [])),
                        "segments": result.get("segments", []),
                        "completed_at": time.time()
                    })
                else:
                    frame_data.update({
                        "status": "error",
                        "error": result.get("error", "Unknown error"),
                        "completed_at": time.time()
                    })
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_id}: {str(e)}")
                frame_data.update({
                    "status": "error",
                    "error": str(e),
                    "completed_at": time.time()
                })
            
            # Save after each frame
            frame_status_data["updated_at"] = time.time()
            storage.save_frame_status(user_id, frame_status_data)
        
        # Mark overall status as completed
        frame_status_data["status"] = "completed"
        frame_status_data["completed_at"] = time.time()
        storage.save_frame_status(user_id, frame_status_data)
        
    except Exception as e:
        logger.error(f"Error in background frame processing for user {user_id}: {str(e)}")
        # Mark as failed
        frame_status_data["status"] = "failed"
        frame_status_data["error"] = str(e)
        frame_status_data["failed_at"] = time.time()
        storage.save_frame_status(user_id, frame_status_data)

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
        # Generate visualization using replicate client
        result = replicate_client.segment_and_process(
            image_path=request.image_path,
            user_id="visualization_request"  # Temporary user for visualization
        )
        
        if result["status"] == "success" and result.get("visualization_url"):
            # Return the visualization URL as a redirect
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url=result["visualization_url"])
        else:
            raise HTTPException(status_code=500, detail="Failed to generate visualization")
            
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating visualization")

@router.get("/health")
async def health_check():
    """Health check endpoint - includes Replicate API status"""
    try:
        # Check Replicate client health (basic connectivity)
        # We can do a simple test by checking if the client is initialized
        replicate_health = {
            "status": "healthy" if replicate_client else "unhealthy",
            "service": "Replicate Grounding DINO API"
        }
        
        return {
            "status": "healthy", 
            "service": "Turing Segmentation API",
            "replicate_service": replicate_health,
            "mode": "replicate_api"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "Turing Segmentation API", 
            "replicate_service": {"status": "unhealthy", "error": str(e)},
            "mode": "replicate_api"
        }