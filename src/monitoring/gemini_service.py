"""
Google Gemini Integration Service
Uses Google Gemini to analyze violation frames and generate descriptive summaries
"""

import os
import base64
import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional
import requests
import json
from google import genai

logger = logging.getLogger(__name__)

class GeminiAnalysisService:
    """Service for Google Gemini frame analysis"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        self.client = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

        if not self.api_key:
            logger.warning("GOOGLE_GEMINI_API_KEY not set. Gemini analysis will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Gemini analysis service initialized")
    
    def _encode_image_base64(self, frame: np.ndarray) -> str:
        """Encode frame as base64 for Gemini API"""
        try:
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            # Encode as base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return img_base64
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise e
    
    def _create_violation_analysis_prompt(self, violation: Dict[str, Any]) -> str:
        """Create a prompt for analyzing a violation"""
        person_name = violation.get("person_name", "Unknown person")
        area_type = violation.get("segment", {}).get("area_type", "restricted area")
        reason = violation.get("violation_reason", "Access denied")
        
        prompt = f"""You are a security system analyst. Please analyze this security camera image and provide a brief, professional summary of what you observe.

Context:
- Person detected: {person_name}
- Location: {area_type}
- Access status: {reason}

Please describe:
1. What the person appears to be doing
2. The setting/environment visible in the image
3. Any relevant security concerns or notable observations
4. A brief assessment of the situation
5. If access is unauthorized, you should return that it needs immediate attention. If you detect a kid, keep the tone soft
because you're alerting a parent or guardian. But still keep the level of alert so that they respond quickly.

Keep your response concise (2 sentences) and friendly, as it will be used in a security alert notification.

Focus on observable facts rather than speculation. Try not to sound like a robot. The text is read by a human. Keep your tone friendly and precise"""

        return prompt
    
    def _create_unknown_activity_prompt(self, detection: Dict[str, Any]) -> str:
        """Create a prompt for analyzing unknown person activity"""
        area_type = detection.get("segment", {}).get("area_type", "monitored area")
        
        prompt = f"""You are a security system analyst. Please analyze this security camera image showing an unrecognized person in a monitored area.

Context:
- Location: {area_type}
- Status: Unknown/unrecognized person detected

Please describe:
1. What the person appears to be doing
2. The setting/environment visible
3. Any potential security concerns
4. A brief assessment of the activity

Keep your response concise (2-3 sentences) and professional, as it will be used in a security alert. Focus on observable behavior and context.
Try not to sound like a robot. The text is read by a human. Keep your tone friendly and precise
"""

        return prompt
    
    def _call_gemini_api(self, prompt: str, image_base64: str) -> Optional[str]:
        """Make API call to Google Gemini"""
        try:
            if not self.enabled:
                return "Gemini analysis not available (API key not configured)"

            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    genai.types.Part.from_bytes(
                        data=image_base64,
                        mime_type='image/jpeg',
                    ),
                    prompt
                ]
            )
            return response.text
            # response.raise_for_status()
            # result = response.json()
            #
            # # Extract generated text
            # if "candidates" in result and len(result["candidates"]) > 0:
            #     candidate = result["candidates"][0]
            #     if "content" in candidate and "parts" in candidate["content"]:
            #         parts = candidate["content"]["parts"]
            #         if len(parts) > 0 and "text" in parts[0]:
            #             return parts[0]["text"].strip()
            #
            # logger.warning("Unexpected Gemini API response format")
            # return "Unable to analyze image - unexpected response format"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {e}")
            return f"Analysis unavailable (API error: {str(e)})"
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f"Analysis unavailable (error: {str(e)})"
    
    def analyze_violation_frame(self, frame: np.ndarray, violation: Dict[str, Any]) -> str:
        """
        Analyze a violation frame using Google Gemini
        
        Args:
            frame: The video frame as numpy array
            violation: Violation details
            
        Returns:
            Analysis summary string
        """
        try:
            if not self.enabled:
                return "AI analysis not available (Gemini API not configured)"
            
            # Encode frame
            image_base64 = self._encode_image_base64(frame)
            
            # Create prompt
            prompt = self._create_violation_analysis_prompt(violation)
            
            # Get analysis
            analysis = self._call_gemini_api(prompt, image_base64)
            
            if analysis:
                logger.info(f"Generated Gemini analysis for violation by {violation.get('person_name', 'unknown')}")
                return analysis
            else:
                return "Unable to generate analysis at this time"
                
        except Exception as e:
            logger.error(f"Error analyzing violation frame: {e}")
            return f"Analysis error: {str(e)}"
    
    def analyze_unknown_activity_frame(self, frame: np.ndarray, detection: Dict[str, Any]) -> str:
        """
        Analyze unknown person activity using Google Gemini
        
        Args:
            frame: The video frame as numpy array
            detection: Detection details
            
        Returns:
            Analysis summary string
        """
        try:
            if not self.enabled:
                return "AI analysis not available (Gemini API not configured)"
            
            # Encode frame
            image_base64 = self._encode_image_base64(frame)
            
            # Create prompt
            prompt = self._create_unknown_activity_prompt(detection)
            
            # Get analysis
            analysis = self._call_gemini_api(prompt, image_base64)
            
            if analysis:
                logger.info(f"Generated Gemini analysis for unknown activity in {detection.get('segment', {}).get('area_type', 'unknown area')}")
                return analysis
            else:
                return "Unable to generate analysis at this time"
                
        except Exception as e:
            logger.error(f"Error analyzing unknown activity frame: {e}")
            return f"Analysis error: {str(e)}"
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of Gemini service"""
        return {
            "enabled": self.enabled,
            "api_configured": self.api_key is not None,
            "service": "Google Gemini 1.5 Flash"
        }