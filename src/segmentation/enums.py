from enum import Enum

class AreaType(Enum):
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
    UNKNOWN = "unknown"

class PermissionCondition(Enum):
    ADULT_SUPERVISION_REQUIRED = "adult_supervision_required"
    DAYLIGHT_ONLY = "daylight_only"
    WEEKENDS_ONLY = "weekends_only"
    NO_CONDITIONS = "no_conditions"