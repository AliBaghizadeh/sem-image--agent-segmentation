"""
apply_enhancement.py
====================

Apply optimal enhancement parameters to a batch of images.

Purpose
-------
After identifying the best enhancement parameters using `gridsearch_single_image.py`,
this script applies those parameters to all selected images to produce enhanced versions
with better grain boundary visibility.

The enhanced images can then be used for:
- Manual mask generation (SAM)
- Fine-tuning workflows
- Quality improvement of failed segmentation cases

Workflow
--------
1. Identify failed masks and move corresponding images to a separate folder
2. Run grid search to find optimal parameters (using gridsearch_single_image.py)
3. Use THIS script to apply optimal parameters to all failed images
4. Generate masks from the enhanced images using SAM

Usage
-----
Basic usage with optimal parameters:
    python apply_enhancement.py --input_dir "data/selected_tiled_images" --output_dir "data/enhanced_images" --scales 0.2 --dog_sigma_small 5.0 --dog_sigma_large 15.0 --dirt_threshold 0.6 --clip 4.0 --blend 0.9

Parameters
----------
--input_dir : str
    Directory containing images to enhance (e.g., failed segmentation cases)
--output_dir : str
    Directory where enhanced images will be saved
--scales : float
    Frangi filter scale (smaller = finer boundaries)
--dog_sigma_small : float
    DoG small sigma for fine edge detection
--dog_sigma_large : float
    DoG large sigma for coarse edge detection
--dirt_threshold : float
    Dirt filtering threshold (0.0-1.0, higher = more aggressive cleaning)
--clip : float
    CLAHE clip limit (higher = more contrast enhancement)
--blend : float
    Blending ratio (0.0 = full enhancement, 1.0 = original image)

Example
-------
# Apply conservative enhancement (recommended for rescue workflow)
python apply_enhancement.py --input_dir "data/selected_tiled_images" --output_dir "data/enhanced_images" --scales 0.2 --dog_sigma_small 5.0 --dog_sigma_large 15.0 --dirt_threshold 0.6 --clip 4.0 --blend 0.9

# Apply moderate enhancement
python apply_enhancement.py --input_dir "data/selected_tiled_images" --output_dir "data/enhanced_images" --scales 0.2 --dog_sigma_small 5.0 --dog_sigma_large 15.0 --dirt_threshold 0.8 --clip 5.0 --blend 0.5
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add Line enhancement to path
line_enhancement_path = str(Path(__file__).parent / "Line enhancement")
if line_enhancement_path not in sys.path:
    sys.path.append(line_enhancement_path)

from sem_line_enhancer.pipeline import SEMPreprocessor
from sem_line_enhancer.presets import PIPELINE_PRESETS, PREPROCESSOR_PRESETS


def apply_enhancement_batch(args):
    """
    Apply enhancement parameters to all images in input directory.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing enhancement parameters
    """
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(list(input_dir.glob(ext)))
        image_files.extend(list(input_dir.glob(ext.upper())))
    
    if not image_files:
        print(f"Error: No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Enhancement parameters:")
    print(f"  Scale: {args.scales}")
    print(f"  DoG: ({args.dog_sigma_small}, {args.dog_sigma_large})")
    print(f"  Dirt threshold: {args.dirt_threshold}")
    print(f"  CLAHE clip: {args.clip}")
    print(f"  Blend ratio: {args.blend}")
    print()
    
    # Setup preprocessor
    preprocessor = SEMPreprocessor(**PREPROCESSOR_PRESETS["boundaries"])
    
    # Prepare enhancement preset
    scales_tuple = (args.scales, args.scales * 2, args.scales * 4)
    current_preset = PIPELINE_PRESETS["boundaries"].copy()
    current_preset["frangi_scales"] = scales_tuple
    current_preset["dog_sigma_small"] = args.dog_sigma_small
    current_preset["dog_sigma_large"] = args.dog_sigma_large
    current_preset["clahe_clip"] = args.clip
    current_preset["dirt_threshold"] = args.dirt_threshold
    
    # Process each image
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing {img_path.name}...")
        
        # Load image
        image_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            print(f"  Error: Could not load image")
            continue
        
        # Apply enhancement
        _, _, i_fused, _ = preprocessor.preprocess_dual(image_gray, **current_preset)
        fused_vis = (i_fused * 255).clip(0, 255).astype(np.uint8)
        
        # Blend with original
        if args.blend > 0:
            enhanced_gray = cv2.addWeighted(image_gray, args.blend, fused_vis, 1 - args.blend, 0)
        else:
            enhanced_gray = fused_vis
        
        # Save enhanced image with _processed suffix
        output_name = img_path.stem + "_processed" + img_path.suffix
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), enhanced_gray)
        print(f"  Saved: {output_path.name}")
    
    print(f"\nCompleted! Enhanced images saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply optimal enhancement parameters to batch of images"
    )
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing images to enhance")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where enhanced images will be saved")
    parser.add_argument("--scales", type=float, default=0.2,
                        help="Frangi filter scale (default: 0.2)")
    parser.add_argument("--dog_sigma_small", type=float, default=5.0,
                        help="DoG small sigma (default: 5.0)")
    parser.add_argument("--dog_sigma_large", type=float, default=15.0,
                        help="DoG large sigma (default: 15.0)")
    parser.add_argument("--dirt_threshold", "--dirt_thresholds", type=float, default=0.6,
                        help="Dirt filtering threshold (default: 0.6)")
    parser.add_argument("--clip", "--clips", type=float, default=4.0,
                        help="CLAHE clip limit (default: 4.0)")
    parser.add_argument("--blend", "--blends", type=float, default=0.9,
                        help="Blend ratio - 0.0=full enhancement, 1.0=original (default: 0.9)")
    
    args = parser.parse_args()
    apply_enhancement_batch(args)
