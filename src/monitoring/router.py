from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from typing import Dict, Any
import os
import cv2
import tempfile
import logging
import asyncio

from .detection_service import DetectionService
from .notification_service import notification_service
from . import gemini_service as gemini_service_client
from .schemas import MonitoringRequest, MonitoringResponse, VideoAnalysisResponse, NotificationRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize detection service
detection_service = DetectionService()
gemini_service = gemini_service_client.GeminiAnalysisService()

@router.post("/analyze/frame", response_model=MonitoringResponse)
async def analyze_frame(user_id: str, file: UploadFile = File(...)):
    """Upload and analyze a single frame for access violations"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Read image
        frame = cv2.imread(temp_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read image file")
        
        # Analyze frame
        result = detection_service.analyze_frame_for_violations(frame, user_id)
        
        return MonitoringResponse(**result)
        
    except Exception as e:
        logger.error(f"Frame analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video(request: MonitoringRequest):
    """Analyze video file for access violations"""
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        result = detection_service.analyze_video_for_violations(
            request.video_path,
            request.user_id,
            request.frame_interval
        )
        
        return VideoAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Video analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")

@router.get("/segments/user/{user_id}")
async def get_user_monitoring_status(user_id: str):
    """Get monitoring status for a user (segments and permissions)"""
    try:
        segments = detection_service.segment_manager.get_user_segments(user_id)
        
        monitoring_status = {
            "user_id": user_id,
            "total_segments": len(segments),
            "segments": segments,
            "monitoring_enabled": len(segments) > 0,
            "last_updated": None
        }
        
        return monitoring_status
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting monitoring status: {str(e)}")

@router.websocket("/notifications/{user_id}")
async def websocket_notifications(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time notifications"""
    await notification_service.connect_user(user_id, websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            
            # Handle client messages (optional)
            if data == "ping":
                await websocket.send_text("pong")
            elif data == "test":
                await notification_service.send_test_notification(user_id)
                
    except WebSocketDisconnect:
        notification_service.disconnect_user(user_id)
        logger.info(f"WebSocket disconnected for user {user_id}")

@router.post("/notify/violation")
async def send_violation_notification(request: NotificationRequest):
    """Send a violation notification (typically called internally)"""
    try:
        # Load frame if path provided
        frame = None
        if request.frame_image_path and os.path.exists(request.frame_image_path):
            frame = cv2.imread(request.frame_image_path)
        
        await notification_service.send_violation_notification(
            request.user_id,
            request.violation.dict(),
            frame,
            request.analysis_summary
        )
        
        return {"status": "notification_sent", "user_id": request.user_id}
        
    except Exception as e:
        logger.error(f"Failed to send violation notification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send notification: {str(e)}")

@router.post("/notify/test/{user_id}")
async def send_test_notification(user_id: str):
    """Send a test notification to verify WebSocket connection"""
    try:
        success = await notification_service.send_test_notification(user_id)
        
        if success:
            return {"status": "test_notification_sent", "user_id": user_id}
        else:
            return {"status": "user_not_connected", "user_id": user_id}
            
    except Exception as e:
        logger.error(f"Failed to send test notification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send test notification: {str(e)}")

@router.get("/notifications/history/{user_id}")
async def get_notification_history(user_id: str, limit: int = 50):
    """Get notification history for a user"""
    try:
        history = notification_service.get_notification_history(user_id, limit)
        return {
            "user_id": user_id,
            "total_notifications": len(history),
            "notifications": history
        }
        
    except Exception as e:
        logger.error(f"Failed to get notification history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get notification history: {str(e)}")

@router.get("/notifications/status/{user_id}")
async def get_notification_status(user_id: str):
    """Get notification connection status for a user"""
    is_connected = user_id in notification_service.active_connections
    return {
        "user_id": user_id,
        "connected": is_connected,
        "total_active_connections": len(notification_service.active_connections)
    }

@router.post("/analyze/frame/with-notifications", response_model=MonitoringResponse)
async def analyze_frame_with_notifications(user_id: str, file: UploadFile = File(...)):
    """Upload and analyze a frame with automatic notifications for violations"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Read image
        frame = cv2.imread(temp_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read image file")
        
        # Analyze frame
        result = detection_service.analyze_frame_for_violations(frame, user_id)
        
        # Send notifications for violations that should alert
        if result.get("status") == "success":
            for violation in result.get("violations", []):
                if violation.get("should_alert", False):
                    await notification_service.send_violation_notification(
                        user_id, violation, frame
                    )
            
            # Send notifications for unknown faces
            for detection in result.get("unknown_detections", []):
                await notification_service.send_unknown_activity_notification(
                    user_id, detection, frame
                )
        
        return MonitoringResponse(**result)
        
    except Exception as e:
        logger.error(f"Frame analysis with notifications failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/analyze/frame/gemini")
async def analyze_frame_with_gemini(user_id: str, file: UploadFile = File(...)):
    """Upload and analyze a frame with Gemini AI analysis"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Read image
        frame = cv2.imread(temp_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read image file")
        
        # Analyze frame for violations
        result = detection_service.analyze_frame_for_violations(frame, user_id)
        
        # Add Gemini analysis to violations and detections
        if result.get("status") == "success":
            for violation in result.get("violations", []):
                violation["gemini_analysis"] = gemini_service.analyze_violation_frame(frame, violation)
            
            for detection in result.get("unknown_detections", []):
                detection["gemini_analysis"] = gemini_service.analyze_unknown_activity_frame(frame, detection)
        
        return MonitoringResponse(**result)
        
    except Exception as e:
        logger.error(f"Frame analysis with Gemini failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.get("/gemini/status")
async def get_gemini_status():
    """Get Google Gemini service status"""
    return gemini_service.get_service_status()

@router.get("/health")
async def health_check():
    """Health check for monitoring service"""
    gemini_status = gemini_service.get_service_status()
    return {
        "status": "healthy",
        "service": "Turing Monitoring Service",
        "components": {
            "detection_service": "active",
            "segment_manager": "active", 
            "face_recognition": "active",
            "notification_service": "active",
            "gemini_service": "active" if gemini_status["enabled"] else "disabled",
            "active_websocket_connections": len(notification_service.active_connections)
        },
        "gemini_analysis": gemini_status
    }

import base64

@router.websocket("/ws/live_feed/{user_id}")
async def websocket_live_feed(websocket: WebSocket, user_id: str):
    await websocket.accept()
    video_path = "/Users/tahakhan/Desktop/ring_camera/monitoring_video/intruder.mp4"
    if not os.path.exists(video_path):
        await websocket.send_json({"error": "Video file not found"})
        await websocket.close()
        return

    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Analyze the frame
            analysis_result = detection_service.analyze_frame_for_violations(frame, user_id)

            # Add Gemini analysis to detections for richer notifications
            if analysis_result.get("status") == "success":
                for violation in analysis_result.get("violations", []):
                    if violation.get("should_alert", False):
                        violation["gemini_analysis"] = gemini_service.analyze_violation_frame(frame, violation)
                
                for detection in analysis_result.get("unknown_detections", []):
                    if detection.get("should_alert", False):
                        detection["gemini_analysis"] = gemini_service.analyze_unknown_activity_frame(frame, detection)

            # Draw bounding boxes on frame
            frame_with_boxes = frame.copy()
            if analysis_result.get("status") == "success":
                # Draw bounding boxes for violations (red)
                for violation in analysis_result.get("violations", []):
                    bbox = violation.get("face_bbox", {})
                    if bbox:
                        cv2.rectangle(frame_with_boxes, 
                                    (bbox["x1"], bbox["y1"]), 
                                    (bbox["x2"], bbox["y2"]), 
                                    (0, 0, 255), 2)  # Red for violations
                        # Add person name label
                        person_name = violation.get("person_name", "Unknown")
                        cv2.putText(frame_with_boxes, f"{person_name} (VIOLATION)", 
                                  (bbox["x1"], bbox["y1"] - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw bounding boxes for unknown detections (yellow)
                for detection in analysis_result.get("unknown_detections", []):
                    bbox = detection.get("face_bbox", {})
                    if bbox:
                        cv2.rectangle(frame_with_boxes, 
                                    (bbox["x1"], bbox["y1"]), 
                                    (bbox["x2"], bbox["y2"]), 
                                    (0, 255, 255), 2)  # Yellow for unknown
                        cv2.putText(frame_with_boxes, "UNKNOWN PERSON", 
                                  (bbox["x1"], bbox["y1"] - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Convert frame to base64 for transmission
            _, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Send frame with analysis data in expected format
            websocket_data = {
                "image": frame_base64,
                "notifications": analysis_result
            }
            await websocket.send_json(websocket_data)

            # Send notifications for violations
            if analysis_result.get("status") == "success":
                for violation in analysis_result.get("violations", []):
                    if violation.get("should_alert", False):
                        await notification_service.send_violation_notification(
                            user_id, violation, frame
                        )
                for detection in analysis_result.get("unknown_detections", []):
                    await notification_service.send_unknown_activity_notification(
                        user_id, detection, frame
                    )

            await asyncio.sleep(0.1)  # Adjust for desired frame rate
    except WebSocketDisconnect:
        logger.info(f"Live feed WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"Error in live feed WebSocket: {e}")
    finally:
        cap.release()
        await websocket.close()
