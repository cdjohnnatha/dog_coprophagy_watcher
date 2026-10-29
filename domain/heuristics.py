"""
Domain heuristics - Pure functions for scoring and detection logic.
No I/O, no side effects - only calculations.
"""
import numpy as np
from typing import Tuple
from .models import ShapeParams, Point, ROI


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


def point_in_roi(centroid: Point, pad_roi: ROI) -> bool:
    """
    Return True if the point (in frame coordinates) lies inside the ROI.

    pad_roi is (x, y, w, h). Point is (px, py).
    """
    if not centroid or not pad_roi or len(pad_roi) != 4:
        return False
    px, py = int(centroid[0]), int(centroid[1])
    rx, ry, rw, rh = int(pad_roi[0]), int(pad_roi[1]), int(pad_roi[2]), int(pad_roi[3])
    return rx <= px < rx + rw and ry <= py < ry + rh


def distance_px(point_a: Point, point_b: Point) -> float:
    """Calculate Euclidean distance between two points."""
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5


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

