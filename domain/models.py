"""
Domain models - Pure data structures representing business entities.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import numpy as np

# Type aliases
Point = Tuple[int, int]  # (x, y)
ROI = Tuple[int, int, int, int]  # (x, y, w, h)

@dataclass
class FrigateEventUpdate:
    """
    Normalized Frigate MQTT event update from `frigate/events`.

    Frigate publishes event lifecycle updates with fields like:
      - type: new | update | end
      - after: { id, label, camera, entered_zones/current_zones, start_time, end_time, ... }
    """

    type: str
    camera: str
    label: str
    event_id: str
    zones: list[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    has_clip: Optional[bool] = None


class EllieState(str, Enum):
    """State machine states for Ellie's behavior."""
    IDLE = "IDLE"
    POSSIVEL_DEFECACAO = "POSSIVEL_DEFECACAO"
    DEFECANDO = "DEFECANDO"
    AGUARDANDO_CONFIRMACAO = "AGUARDANDO_CONFIRMACAO"
    DEFECACAO_CONFIRMADA = "DEFECACAO_CONFIRMADA"


class EliminationKind(str, Enum):
    """Confirmed elimination type."""
    POOP = "poop"
    PEE = "pee"


class EliminationLocation(str, Enum):
    """Whether elimination happened in the proper (pad) area."""
    PROPER = "proper"
    OUTSIDE = "outside"
    UNKNOWN = "unknown"


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
class PeeCandidate:
    """
    Represents a candidate pee region along with a "wetness mass" score.

    `mass` is an integrated (summed) pixel-level change measure intended to
    detect pee-on-pee where the affected area may grow only slightly.
    """

    blob: Blob
    mass: int


@dataclass
class TrackState:
    """Main state tracker for Ellie's behavior."""
    # Dog tracking
    last_ts: float = 0.0
    last_seen_dog_ts: float = 0.0
    last_zones: list[str] = field(default_factory=list)
    
    # Squat detection
    squat_start: float = 0.0
    in_squat: bool = False
    
    # ROI and background
    bg_roi: Optional[np.ndarray] = None
    roi_anchor: Optional[ROI] = None
    # Baseline captured at squat start (pre-elimination reference) and its ROI.
    pre_squat_bg: Optional[np.ndarray] = None
    pre_squat_roi_anchor: Optional[ROI] = None
    pre_squat_ts: float = 0.0
    residue_bg: Optional[np.ndarray] = None
    mono_flag: bool = False

    # Stable pad ROI (optional; used for long-lived wetness tracking).
    pad_roi: Optional[ROI] = None

    # Pad reference capture (e.g. after person swaps pad).
    pending_pad_ref_ts: float = 0.0
    
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
    last_confirm_kind: Optional[str] = None


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
class EliminationEvent:
    """Canonical elimination event (poop or pee) for Home Assistant analytics."""
    timestamp: str
    camera: str
    zones: list[str]
    kind: EliminationKind
    location: EliminationLocation
    centroid: Optional[Point] = None
    area: int = 0
    confidence: Optional[float] = None
    event_id: Optional[str] = None
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
class PoopResidueScore:
    prob_poop: float


@dataclass
class PeeResidueScore:
    prob_pee: float