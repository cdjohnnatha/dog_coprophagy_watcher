"""
Domain heuristics - Pure functions for scoring and detection logic.
No I/O, no side effects - only calculations.
"""
import numpy as np
from typing import Optional, Tuple
from .models import SquatScore, Blob, ShapeParams, CoprophagyThresholds, Point


def score_squat(
    ratio: float,
    stationary: bool,
    speed: float,
    motionless_count: int,
    squat_thresh: float,
    motionless_min: int,
    speed_max_still: float,
) -> SquatScore:
    """
    Calculate squat score based on dog's posture metrics.
    
    Args:
        ratio: Height/width ratio of bounding box
        stationary: Whether dog is marked as stationary
        speed: Estimated speed
        motionless_count: Number of consecutive motionless frames
        squat_thresh: Threshold for considering it a squat
        motionless_min: Minimum motionless frames required
        speed_max_still: Maximum speed to be considered still
    
    Returns:
        SquatScore with detailed scoring breakdown
    """
    # Ratio term: lower ratio (wider) = higher score
    ratio_term = max(0.0, min(1.0, (0.70 - ratio) / 0.20))
    
    # Really still check
    really_still = (
        stationary 
        and (motionless_count >= motionless_min) 
        and (speed <= speed_max_still)
    )
    stationary_term = 1.0 if really_still else 0.0
    
    # Speed term
    speed_term_raw = max(0.0, min(1.0, 1.0 - min(speed, 1.0)))
    speed_term = 0.0 if speed > speed_max_still else speed_term_raw
    
    # Weighted combination
    score = 0.55 * ratio_term + 0.30 * stationary_term + 0.15 * speed_term
    
    return SquatScore(
        score=score,
        ratio_term=ratio_term,
        stationary_term=stationary_term,
        speed_term=speed_term,
        is_squatting=(score >= squat_thresh),
    )


def calculate_circularity(area: float, perimeter: float) -> float:
    """Calculate circularity: 4πA/P²."""
    if perimeter == 0:
        return 0.0
    return 4.0 * np.pi * area / (perimeter * perimeter)


def calculate_solidity(area: float, hull_area: float) -> float:
    """Calculate solidity: A/convexHull."""
    if hull_area == 0:
        return 0.0
    return area / hull_area


def passes_shape_filter(
    area: int,
    width: int,
    height: int,
    circularity: float,
    solidity: float,
    params: ShapeParams,
) -> bool:
    """
    Check if blob passes shape-based filtering (anti-pee).
    
    Returns:
        True if blob passes all filters
    """
    if width < params.min_blob_w or height < params.min_blob_h:
        return False
    if circularity < params.min_circularity:
        return False
    if solidity < params.min_solidity:
        return False
    return True


def passes_color_filter(mean_saturation: float, hsv_s_min: int) -> bool:
    """Check if blob passes color saturation filter."""
    return mean_saturation >= hsv_s_min


def passes_texture_filter(
    laplacian_variance: float,
    bright_fraction: float,
    params: ShapeParams,
) -> bool:
    """
    Check if blob passes texture filter (for monochrome/IR mode).
    
    Args:
        laplacian_variance: Variance of Laplacian (texture measure)
        bright_fraction: Fraction of very bright pixels
        params: Shape parameters with thresholds
    
    Returns:
        True if blob passes texture filters
    """
    # Poop should have texture (not smooth like pee)
    if laplacian_variance < params.tex_min_mono:
        return False
    
    # Wet pee shines in IR (high specular highlights)
    if bright_fraction > params.spec_max_mono:
        return False
    
    return True


def is_coprophagy(
    area0: int,
    measured_area: int,
    cum_area_min: int,
    person_present: bool,
    thresholds: CoprophagyThresholds,
    window_elapsed: float,
) -> bool:
    """
    Determine if coprophagy occurred based on area drop.
    
    Args:
        area0: Initial confirmed poop area
        measured_area: Current measured area
        cum_area_min: Minimum area observed in cumulative window
        person_present: Whether a person was present during visit
        thresholds: Detection thresholds
        window_elapsed: Time elapsed since cumulative window start
    
    Returns:
        True if coprophagy is detected
    """
    if person_present:
        return False
    
    if area0 == 0:
        return False
    
    # Immediate drop fraction
    drop_frac = max(0.0, float(area0 - measured_area) / float(area0))
    
    # Cumulative drop fraction
    cum_drop = max(0.0, float(area0 - cum_area_min) / float(area0))
    
    # Check if within cumulative window
    window_ok = window_elapsed <= thresholds.cum_window_s
    
    # Decision logic
    if drop_frac >= thresholds.copro_drop_frac:
        return True
    
    if window_ok and cum_drop >= thresholds.cum_drop_frac:
        return True
    
    return False


def distance_px(point_a: Point, point_b: Point) -> float:
    """Calculate Euclidean distance between two points."""
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5


def is_near_poop(
    dog_center: Point,
    poop_centroid: Point,
    near_radius_px: int,
) -> bool:
    """Check if dog is near the poop."""
    return distance_px(dog_center, poop_centroid) <= near_radius_px


def should_merge_episode(
    time_since_last: float,
    distance: float,
    cooldown_s: float,
    merge_radius_px: int,
) -> bool:
    """
    Determine if new detection should merge with existing episode.
    
    Args:
        time_since_last: Seconds since last confirmation
        distance: Distance in pixels from last centroid
        cooldown_s: Maximum time to consider merging
        merge_radius_px: Maximum distance to consider merging
    
    Returns:
        True if should merge into same episode
    """
    return time_since_last <= cooldown_s and distance <= merge_radius_px


def calculate_roi_from_bbox(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    pad: int = 20,
) -> Tuple[int, int, int, int]:
    """
    Calculate ROI (region of interest) from bounding box.
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box
        frame_width: Width of frame
        frame_height: Height of frame
        pad: Padding around ROI
    
    Returns:
        ROI as (x, y, w, h)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    x1r = max(0, int(cx - 90) - pad)
    x2r = min(frame_width, int(cx + 90) + pad)
    y1r = max(0, int(y2) - 20)
    y2r = min(frame_height, int(y2 + 120) + pad)
    
    return (x1r, y1r, x2r - x1r, y2r - y1r)

