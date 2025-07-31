"""
Web Notification Service
Handles sending web notifications for security violations and activities
"""

import os
import json
import logging
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import WebSocket
import asyncio

from . import gemini_service as gemini_service_client


logger = logging.getLogger(__name__)
gemini_service  = gemini_service_client.GeminiAnalysisService()


class NotificationService:
    """Service for handling web notifications"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.notification_history = []
        
    async def connect_user(self, user_id: str, websocket: WebSocket):
        """Connect a user to receive real-time notifications"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected for notifications")
        
    def disconnect_user(self, user_id: str):
        """Disconnect a user from notifications"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"User {user_id} disconnected from notifications")
    
    def _create_violation_message(self, violation: Dict[str, Any], 
                                frame_image_url: Optional[str] = None,
                                analysis_summary: Optional[str] = None) -> Dict[str, Any]:
        """Create a formatted notification message for a violation"""
        
        person_name = violation.get("person_name", "Unknown person")
        area_type = violation.get("segment", {}).get("area_type", "restricted area")
        reason = violation.get("violation_reason", "Access denied")
        
        # Create notification content
        notification = {
            "id": f"violation_{datetime.now().timestamp()}",
            "type": "security_violation",
            "title": f"Security Alert: {person_name} detected",
            "message": f"{person_name} detected in {area_type}. {reason}",
            "severity": "high",
            "timestamp": violation.get("timestamp", datetime.now().isoformat()),
            "data": {
                "person_name": person_name,
                "area_type": area_type,
                "area_id": violation.get("segment", {}).get("area_id"),
                "violation_reason": reason,
                "confidence": violation.get("confidence", 0),
                "frame_image_url": frame_image_url,
                "analysis_summary": analysis_summary,
                "bbox": violation.get("face_bbox", {}),
                "frame_number": violation.get("frame_number")
            },
            "actions": [
                {
                    "label": "View Details",
                    "action": "view_violation_details",
                    "data": {"violation_id": violation.get("id")}
                },
                {
                    "label": "Dismiss",
                    "action": "dismiss_notification",
                    "data": {"notification_id": f"violation_{datetime.now().timestamp()}"}
                }
            ]
        }
        
        return notification
    
    def _create_unknown_activity_message(self, detection: Dict[str, Any],
                                       frame_image_url: Optional[str] = None,
                                       analysis_summary: Optional[str] = None) -> Dict[str, Any]:
        """Create a formatted notification message for unknown activity"""
        
        area_type = detection.get("segment", {}).get("area_type", "monitored area")
        
        notification = {
            "id": f"unknown_{datetime.now().timestamp()}",
            "type": "unknown_activity",
            "title": "Unknown Person Detected",
            "message": f"Unrecognized person detected in {area_type}",
            "severity": "medium", 
            "timestamp": detection.get("timestamp", datetime.now().isoformat()),
            "data": {
                "area_type": area_type,
                "area_id": detection.get("segment", {}).get("area_id"),
                "frame_image_url": frame_image_url,
                "analysis_summary": analysis_summary,
                "bbox": detection.get("face_bbox", {}),
                "frame_number": detection.get("frame_number")
            },
            "actions": [
                {
                    "label": "Identify Person",
                    "action": "identify_unknown_person",
                    "data": {"detection_id": detection.get("id")}
                },
                {
                    "label": "Dismiss",
                    "action": "dismiss_notification", 
                    "data": {"notification_id": f"unknown_{datetime.now().timestamp()}"}
                }
            ]
        }
        
        return notification
    
    def _save_frame_for_notification(self, frame: np.ndarray, user_id: str, 
                                   violation_id: str) -> Optional[str]:
        """Save frame image for notification and return URL"""
        try:
            # Create notification frames directory
            frames_dir = f"/app/static/notification_frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Save frame image
            filename = f"{user_id}_{violation_id}_{datetime.now().timestamp()}.jpg"
            filepath = os.path.join(frames_dir, filename)
            cv2.imwrite(filepath, frame)
            
            # Return URL
            base_url = os.getenv("BASE_URL", "http://localhost:8000")
            return f"{base_url}/static/notification_frames/{filename}"
            
        except Exception as e:
            logger.error(f"Failed to save notification frame: {e}")
            return None
    
    async def send_violation_notification(self, user_id: str, violation: Dict[str, Any],
                                        frame: Optional[np.ndarray] = None,
                                        analysis_summary: Optional[str] = None):
        """Send violation notification to connected user"""
        try:
            # Generate AI analysis if frame provided and no analysis given
            if frame is not None and analysis_summary is None:
                analysis_summary = gemini_service.analyze_violation_frame(frame, violation)
            
            # Save frame if provided
            frame_url = None
            if frame is not None:
                violation_id = violation.get("id", f"violation_{datetime.now().timestamp()}")
                frame_url = self._save_frame_for_notification(frame, user_id, violation_id)
            
            # Create notification message
            notification = self._create_violation_message(violation, frame_url, analysis_summary)
            
            # Store in history
            self.notification_history.append(notification)
            
            # Send to connected user via WebSocket
            if user_id in self.active_connections:
                websocket = self.active_connections[user_id]
                try:
                    await websocket.send_json(notification)
                    logger.info(f"Sent violation notification to user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to send notification via WebSocket: {e}")
                    # Remove broken connection
                    self.disconnect_user(user_id)
            else:
                logger.warning(f"User {user_id} not connected for notifications")
                
        except Exception as e:
            logger.error(f"Error sending violation notification: {e}")
    
    async def send_unknown_activity_notification(self, user_id: str, detection: Dict[str, Any],
                                               frame: Optional[np.ndarray] = None,
                                               analysis_summary: Optional[str] = None):
        """Send unknown activity notification to connected user"""
        try:
            # Generate AI analysis if frame provided and no analysis given
            if frame is not None and analysis_summary is None:
                analysis_summary = gemini_service.analyze_unknown_activity_frame(frame, detection)
            
            # Save frame if provided
            frame_url = None
            if frame is not None:
                detection_id = detection.get("id", f"unknown_{datetime.now().timestamp()}")
                frame_url = self._save_frame_for_notification(frame, user_id, detection_id)
            
            # Create notification message
            notification = self._create_unknown_activity_message(detection, frame_url, analysis_summary)
            
            # Store in history
            self.notification_history.append(notification)
            
            # Send to connected user via WebSocket
            if user_id in self.active_connections:
                websocket = self.active_connections[user_id]
                try:
                    await websocket.send_json(notification)
                    logger.info(f"Sent unknown activity notification to user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to send notification via WebSocket: {e}")
                    # Remove broken connection
                    self.disconnect_user(user_id)
            else:
                logger.warning(f"User {user_id} not connected for notifications")
                
        except Exception as e:
            logger.error(f"Error sending unknown activity notification: {e}")
    
    def get_notification_history(self, user_id: str, limit: int = 50) -> list:
        """Get recent notification history for a user"""
        # In a real implementation, you'd filter by user_id from database
        return self.notification_history[-limit:]
    
    async def send_test_notification(self, user_id: str):
        """Send a test notification to verify connection"""
        test_notification = {
            "id": f"test_{datetime.now().timestamp()}",
            "type": "test",
            "title": "Test Notification",
            "message": "Notification system is working correctly",
            "severity": "info",
            "timestamp": datetime.now().isoformat(),
            "data": {},
            "actions": [
                {
                    "label": "Dismiss",
                    "action": "dismiss_notification",
                    "data": {"notification_id": f"test_{datetime.now().timestamp()}"}
                }
            ]
        }
        
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_json(test_notification)
                logger.info(f"Sent test notification to user {user_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to send test notification: {e}")
                self.disconnect_user(user_id)
                return False
        else:
            logger.warning(f"User {user_id} not connected for test notification")
            return False

# Global notification service instance
notification_service = NotificationService()