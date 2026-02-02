"""
Simple wrapper for MatSAMModel to make it compatible with the Streamlit app.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.matsam.matsam_model import MatSAMModel

class MatSAM:
    """Simplified wrapper around MatSAMModel for the Streamlit app."""
    
    def __init__(self, checkpoint=None):
        """Initialize with optional checkpoint path."""
        if checkpoint and Path(checkpoint).exists():
            self.model = MatSAMModel(checkpoint_path=checkpoint)
        else:
            # Use default base SAM
            base_checkpoint = Path(__file__).parent.parent.parent / "models/sam_weights/sam_vit_l_0b3195.pth"
            self.model = MatSAMModel(checkpoint_path=str(base_checkpoint))
    
    def segment(self, image, use_global=False):
        """
        Segment an image using either global or point-grid methodology.
        
        Args:
            image: Numpy array (H, W) or (H, W, 3)
            use_global: If True, uses the fast fine-tuned global pass.
                        If False (Default), uses the robust point-grid generator.
        
        Returns:
            Binary mask (H, W) as uint8 (0 and 255)
        """
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image
            
        # 1. Run Pipeline (Matching command line script)
        masks = self.model.generate_auto_masks(image_rgb, use_global=use_global)
        
        # 2. Aggregate Results (Matching command line script post-processing)
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        image_area = image.shape[0] * image.shape[1]
        
        if use_global:
            # Global returns connected-component style dictionaries
            # but usually we just want the union for binary display
            for m in masks:
                final_mask[m['segmentation'] > 0] = 255
        else:
            # Robust mode produces many overlapping masks - union them with erosion
            # to keep grain boundaries sharp as in test_one_image.py
            for m in sorted(masks, key=lambda x: x['area']):
                if m['area'] < 0.5 * image_area:
                    mask_data = m['segmentation'].astype(np.uint8)
                    # Use exact same kernel as command line
                    kernel = np.ones((3,3), np.uint8)
                    eroded = cv2.erode(mask_data, kernel, iterations=1)
                    final_mask[eroded > 0] = 255
                    
        return final_mask
