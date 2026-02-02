"""
Quality metrics for segmentation evaluation.
"""

import cv2
import numpy as np

def calculate_quality_score(mask):
    """
    Calculate a quality score for the segmentation mask.
    
    Combines multiple heuristics:
    - Grain count (too few or too many = bad)
    - Coverage (extreme values = bad)
    - Boundary smoothness (jagged = bad)
    
    Args:
        mask (np.ndarray): Binary segmentation mask
    
    Returns:
        float: Quality score (0-1, higher is better)
    """
    # Metric 1: Grain count
    num_labels = cv2.connectedComponents(mask)[0] - 1
    grain_score = 1.0
    if num_labels < 10:
        grain_score = num_labels / 10.0
    elif num_labels > 1000:
        grain_score = max(0, 1.0 - (num_labels - 1000) / 1000.0)
    
    # Metric 2: Coverage
    coverage = np.sum(mask > 0) / mask.size
    coverage_score = 1.0
    if coverage < 0.05 or coverage > 0.95:
        coverage_score = 0.5
    
    # Metric 3: Boundary smoothness
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
    total_area = np.sum(mask > 0)
    
    jaggedness = total_perimeter / (np.sqrt(total_area) + 1e-8) if total_area > 0 else 0
    smoothness_score = max(0, 1.0 - jaggedness / 50.0)
    
    # Combined score (weighted average)
    quality_score = (
        0.4 * grain_score +
        0.3 * coverage_score +
        0.3 * smoothness_score
    )
    
    return quality_score

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union.
    
    Args:
        pred_mask (np.ndarray): Predicted binary mask
        gt_mask (np.ndarray): Ground truth binary mask
    
    Returns:
        float: IoU score (0-1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / (union + 1e-8)

def calculate_dice(pred_mask, gt_mask):
    """
    Calculate Dice coefficient.
    
    Args:
        pred_mask (np.ndarray): Predicted binary mask
        gt_mask (np.ndarray): Ground truth binary mask
    
    Returns:
        float: Dice score (0-1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-8)
