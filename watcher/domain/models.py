"""
Domain models - Pure data structures representing business entities.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Protocol
import numpy as np

# Type aliases
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
Point = Tuple[int, int]  # (x, y)
ROI = Tuple[int, int, int, int]  # (x, y, w, h)


class EllieState(str, Enum):
    """State machine states for Ellie's behavior."""
    IDLE = "IDLE"
    POSSIVEL_DEFECACAO = "POSSIVEL_DEFECACAO"
    DEFECANDO = "DEFECANDO"
    AGUARDANDO_CONFIRMACAO = "AGUARDANDO_CONFIRMACAO"
    DEFECACAO_CONFIRMADA = "DEFECACAO_CONFIRMADA"
    COPROPHAGIA_CONFIRMADA = "COPROPHAGIA_CONFIRMADA"


@dataclass
class DogDetection:
    """Represents a dog detection event from Frigate."""
    bbox: BBox
    ratio: float
    stationary: bool
    speed: float
    motionless_count: int
    timestamp: float
    zones: list[str] = field(default_factory=list)
    
    @property
    def center(self) -> Point:
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class Blob:
    """Represents a detected blob (residue/poop)."""
    x: int
    y: int
    width: int
    height: int
    centroid: Point
    area: int
    
    @property
    def roi(self) -> ROI:
        """Return as ROI tuple."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class SquatScore:
    """Result of squat scoring heuristic."""
    score: float
    ratio_term: float
    stationary_term: float
    speed_term: float
    is_squatting: bool


@dataclass
class TrackState:
    """Main state tracker for Ellie's behavior."""
    # Dog tracking
    last_bbox: Optional[BBox] = None
    last_ts: float = 0.0
    last_seen_dog_ts: float = 0.0
    
    # Squat detection
    squat_start: float = 0.0
    in_squat: bool = False
    
    # ROI and background
    bg_roi: Optional[np.ndarray] = None
    roi_anchor: Optional[ROI] = None
    residue_bg: Optional[np.ndarray] = None
    mono_flag: bool = False
    
    # Poop presence
    poop_present: bool = False
    poop_roi: Optional[ROI] = None
    poop_centroid: Optional[Point] = None
    poop_area0: int = 0
    
    # Episode management
    episode_active: bool = False
    episode_started_ts: float = 0.0
    last_confirm_ts: float = 0.0
    last_confirm_centroid: Optional[Point] = None
    
    # Coprophagy detection
    risk_announced: bool = False
    near_since: float = 0.0
    was_near: bool = False
    visit_person_flag: bool = False
    last_person_ts: float = 0.0
    last_near_end_ts: float = 0.0
    
    # Cumulative area tracking
    cum_drop_ref_ts: float = 0.0
    cum_area_min: int = 0


@dataclass
class PoopEvent:
    """Event representing confirmed poop detection."""
    timestamp: str
    camera: str
    zone: str
    centroid: Point
    area: int
    event_id: Optional[str] = None
    clip_url: Optional[str] = None
    thumb_url: Optional[str] = None
    snapshot_url: Optional[str] = None
    ha_clip: Optional[str] = None
    ha_thumb: Optional[str] = None


@dataclass
class CoprophagyRisk:
    """Event representing coprophagy risk."""
    timestamp: str
    zone: str
    centroid: Point
    since: int
    duration_s: int


@dataclass
class CoprophagyEvent:
    """Event representing confirmed coprophagy."""
    timestamp: str
    camera: str
    zone: str
    event_id: Optional[str] = None
    manual_event_id: Optional[str] = None
    clip_url: Optional[str] = None
    thumb_url: Optional[str] = None
    snapshot_url: Optional[str] = None
    ha_clip: Optional[str] = None
    ha_thumb: Optional[str] = None


@dataclass
class ShapeParams:
    """Parameters for shape-based filtering."""
    min_blob_w: int
    min_blob_h: int
    min_circularity: float
    min_solidity: float
    hsv_s_min: int
    tex_min_mono: float
    spec_max_mono: float


@dataclass
class CoprophagyThresholds:
    """Thresholds for coprophagy detection."""
    copro_drop_frac: float
    cum_drop_frac: float
    cum_window_s: float

@dataclass
class PoopResidueScore:
    prob_poop: float

@dataclass
class CoprophagyScore:
    prob_eaten: float

class PoopResidueClassifier(Protocol):
    def predict(self, roi_bgr: np.ndarray) -> PoopResidueScore:
        ...

class CoprophagyClassifier(Protocol):
    def predict(self, roi_bgr: np.ndarray, after_bgr: np.ndarray) -> CoprophagyScore:
        ...