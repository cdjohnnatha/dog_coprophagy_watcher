"""
Computer Vision operations using OpenCV.
All CV-related functionality isolated here.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from domain.models import Blob, ROI, ShapeParams
from domain import heuristics


def safe_roi(img: Optional[np.ndarray], roi: Optional[ROI]) -> Optional[np.ndarray]:
    """
    Safely extract ROI from image.
    
    Args:
        img: Source image
        roi: Region of interest as (x, y, w, h)
    
    Returns:
        Cropped image or None if invalid
    """
    if img is None or roi is None:
        return None
    
    x, y, w, h = roi
    h_img, w_img = img.shape[:2]
    
    x = max(0, int(x))
    y = max(0, int(y))
    x2 = min(x + int(w), w_img)
    y2 = min(y + int(h), h_img)
    
    if x >= x2 or y >= y2:
        return None
    
    return img[y:y2, x:x2]


def is_monochrome_roi(roi_bgr: Optional[np.ndarray], sat_thresh: int = 12) -> bool:
    """
    Check if ROI appears to be monochrome/IR (very low saturation).
    
    Args:
        roi_bgr: ROI in BGR format
        sat_thresh: Saturation threshold
    
    Returns:
        True if appears monochrome
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s_mean = float(cv2.mean(hsv[:, :, 1])[0])
    return s_mean < sat_thresh


def decode_image(buffer: bytes) -> Optional[np.ndarray]:
    """
    Decode image from bytes buffer.
    
    Args:
        buffer: Image bytes
    
    Returns:
        Decoded image or None
    """
    arr = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def diff_blob(
    bg_roi: Optional[np.ndarray],
    cur_roi: Optional[np.ndarray],
    params: ShapeParams,
    monochrome: bool = False,
    min_area: int = 140,
) -> Optional[Blob]:
    """
    Compare background vs current ROI and find best residue blob.
    
    Args:
        bg_roi: Background reference image
        cur_roi: Current image
        params: Shape filtering parameters
        monochrome: Whether in monochrome/IR mode
        min_area: Minimum blob area
    
    Returns:
        Best blob matching poop criteria, or None
    """
    if bg_roi is None or cur_roi is None or bg_roi.size == 0 or cur_roi.size == 0:
        return None
    
    try:
        # Ensure same size
        if bg_roi.shape[:2] != cur_roi.shape[:2]:
            cur_roi = cv2.resize(
                cur_roi,
                (bg_roi.shape[1], bg_roi.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        
        g0 = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return None
    
    # Preprocessing
    g0 = cv2.GaussianBlur(g0, (5, 5), 0)
    g1 = cv2.GaussianBlur(g1, (5, 5), 0)
    
    # Difference
    d = cv2.absdiff(g1, g0)
    _, th = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphology
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    
    # Find contours
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_blob: Optional[Blob] = None
    best_area = 0
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        
        # Calculate geometric properties
        per = cv2.arcLength(c, True)
        if per == 0:
            continue
        
        circ = heuristics.calculate_circularity(area, per)
        
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) if hull is not None else 0.0
        solidity = heuristics.calculate_solidity(area, hull_area)
        
        # Shape filter
        if not heuristics.passes_shape_filter(int(area), w, h, circ, solidity, params):
            continue
        
        # Create mask for this contour
        mask = np.zeros(cur_roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        
        if not monochrome:
            # Color mode: check saturation
            hsv = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2HSV)
            mean_s = cv2.mean(hsv[:, :, 1], mask=mask)[0]
            
            if not heuristics.passes_color_filter(mean_s, params.hsv_s_min):
                continue
        else:
            # Monochrome mode: check texture and specular highlights
            gray = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2GRAY)
            
            # Texture
            var_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Specular highlights
            area_mask = max(1, int(cv2.countNonZero(mask)))
            bright = cv2.countNonZero(cv2.bitwise_and((gray > 235).astype(np.uint8) * 255, mask))
            frac_bright = bright / float(area_mask)
            
            if not heuristics.passes_texture_filter(var_lap, frac_bright, params):
                continue
        
        # This blob passed all filters
        if area > best_area:
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            best_blob = Blob(
                x=x,
                y=y,
                width=w,
                height=h,
                centroid=(cx, cy),
                area=int(area),
            )
            best_area = area
    
    return best_blob


def diff_area(
    bg_roi: Optional[np.ndarray],
    cur_roi: Optional[np.ndarray],
    params: ShapeParams,
    ignore_color: bool = True,
    min_area: int = 140,
) -> int:
    """
    Calculate total area of residue differences.
    
    Args:
        bg_roi: Background reference image
        cur_roi: Current image
        params: Shape filtering parameters
        ignore_color: Skip color filtering (useful in IR mode)
        min_area: Minimum blob area
    
    Returns:
        Total area of all qualifying blobs
    """
    if bg_roi is None or cur_roi is None or bg_roi.size == 0 or cur_roi.size == 0:
        return 0
    
    try:
        if bg_roi.shape[:2] != cur_roi.shape[:2]:
            cur_roi = cv2.resize(
                cur_roi,
                (bg_roi.shape[1], bg_roi.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        
        g0 = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return 0
    
    g0 = cv2.GaussianBlur(g0, (5, 5), 0)
    g1 = cv2.GaussianBlur(g1, (5, 5), 0)
    d = cv2.absdiff(g1, g0)
    _, th = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        if w < params.min_blob_w or h < params.min_blob_h:
            continue
        
        per = cv2.arcLength(c, True)
        if per == 0:
            continue
        
        circ = heuristics.calculate_circularity(area, per)
        if circ < params.min_circularity:
            continue
        
        if not ignore_color and params.hsv_s_min > 0:
            mask = np.zeros(cur_roi.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            hsv = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2HSV)
            mean_s = cv2.mean(hsv[:, :, 1], mask=mask)[0]
            if mean_s < params.hsv_s_min:
                continue
        
        total += int(area)
    
    return total

