"""
Visualization utilities for the MatSAM app.
"""

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

def create_overlay(image, mask, alpha=0.5):
    """
    Create an overlay visualization of the segmentation mask on the original image.
    
    Args:
        image (np.ndarray): Original SEM image (H, W) or (H, W, 3)
        mask (np.ndarray): Binary segmentation mask (H, W)
        alpha (float): Transparency of the overlay (0-1)
    
    Returns:
        np.ndarray: RGB image with mask overlay
    """
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Create colored mask (grain boundaries in red)
    mask_colored = np.zeros_like(image_rgb)
    mask_colored[mask > 0] = [255, 0, 0]  # Red for grain boundaries
    
    # Blend
    overlay = cv2.addWeighted(image_rgb, 1-alpha, mask_colored, alpha, 0)
    
    return overlay

def plot_grain_distribution(grain_sizes):
    """
    Create an interactive histogram of grain size distribution.
    
    Args:
        grain_sizes (list): List of grain areas in pixels
    
    Returns:
        plotly.graph_objects.Figure: Interactive histogram
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=grain_sizes,
        nbinsx=30,
        marker=dict(
            color='rgba(102, 126, 234, 0.7)',
            line=dict(color='rgba(102, 126, 234, 1)', width=1)
        ),
        name='Grain Size Distribution'
    ))
    
    fig.update_layout(
        title="Grain Size Distribution",
        xaxis_title="Grain Area (pixels)",
        yaxis_title="Count",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_comparison_view(original, mask, overlay):
    """
    Create a 3-panel comparison view.
    
    Args:
        original (np.ndarray): Original image
        mask (np.ndarray): Binary mask
        overlay (np.ndarray): Overlay image
    
    Returns:
        np.ndarray: Concatenated comparison image
    """
    # Ensure all images are RGB
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # Resize to same height if needed
    h = min(original.shape[0], mask_rgb.shape[0], overlay.shape[0])
    original = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h))
    mask_rgb = cv2.resize(mask_rgb, (int(mask_rgb.shape[1] * h / mask_rgb.shape[0]), h))
    overlay = cv2.resize(overlay, (int(overlay.shape[1] * h / overlay.shape[0]), h))
    
    # Concatenate horizontally
    comparison = np.hstack([original, mask_rgb, overlay])
    
    return comparison
