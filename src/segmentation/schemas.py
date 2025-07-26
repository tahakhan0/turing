from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

class AreaType(str, Enum):
    """Available area types for segmentation"""
    BACKYARD = "backyard"
    POOL = "pool"
    GARAGE = "garage"
    ROAD = "road"
    DRIVEWAY = "driveway"
    FRONT_YARD = "front_yard"
    LAWN = "lawn"
    PATIO = "patio"
    DECK = "deck"
    FENCE = "fence"
    CUSTOM = "custom"

class PermissionCondition(str, Enum):
    """Conditions that can be applied to permissions"""
    ADULT_SUPERVISION_REQUIRED = "adult_supervision_required"
    DAYLIGHT_ONLY = "daylight_only"
    WEEKENDS_ONLY = "weekends_only"

class SegmentedArea(BaseModel):
    area_id: str
    area_type: AreaType
    polygon: List[Tuple[int, int]]
    confidence: float = Field(ge=0.0, le=1.0)
    dimensions: Dict[str, float]  # width, height, area in meters
    user_id: str
    verified: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True

class ResidentPermission(BaseModel):
    person_name: str  # Name from face recognition labels
    user_id: str      # Property owner's user ID
    area_id: str
    allowed: bool
    conditions: List[PermissionCondition] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True

class SegmentationRequest(BaseModel):
    image_path: str
    user_id: str
    box_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

class SegmentationResponse(BaseModel):
    status: str
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    image_path: Optional[str] = None
    error: Optional[str] = None

class VerificationRequest(BaseModel):
    area_id: str
    user_id: str
    approved: bool

class PermissionRequest(BaseModel):
    person_name: str  # Name from face recognition labels
    user_id: str      # Property owner's user ID
    area_id: str
    allowed: bool
    conditions: List[PermissionCondition] = Field(default_factory=list)

class AccessCheckRequest(BaseModel):
    person_name: str  # Name from face recognition labels
    user_id: str      # Property owner's user ID
    area_id: str
    context: Dict[str, Any] = Field(default_factory=dict)

class AccessCheckResponse(BaseModel):
    allowed: bool
    reason: str

class VisualizationRequest(BaseModel):
    image_path: str
    area_ids: Optional[List[str]] = None