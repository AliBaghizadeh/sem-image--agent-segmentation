"""
Script to generate masks from images using MatSAM with optional preprocessing.

By default, images are used AS-IS without preprocessing. This is ideal for:
- Already-preprocessed images from grid search
- Images enhanced by apply_enhancement.py
- Any images that don't need additional contrast enhancement

Use --preprocess flag only when working with raw, unprocessed images.

Usage:
    # For already-preprocessed images (DEFAULT - no preprocessing)
    python apply_rescue.py --all --input_dir "data/enhanced" --output_dir "data/masks" --overlay
    
    # For raw images that need preprocessing
    python apply_rescue.py --all --input_dir "data/raw_tiles" --output_dir "data/masks" --preprocess --blend 0.0 --overlay
    
    # Rescue specific failed cases with preprocessing
    python apply_rescue.py --preprocess --blend 0.5
"""
import os
import sys
import torch
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
from core.matsam.matsam_model import MatSAMModel

def apply_rescue(args):
    # Final Best Parameters from optimization (can be overridden by args)
    BEST_BLEND = args.blend
    BEST_SCALES = (0.3, 0.7, 1.5)
    BEST_CLAHE = 10.0
    
    # Setup directories
    input_path = Path(args.input_dir)
    final_mask_dir = Path(args.output_dir)
    final_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Identify which files to process
    if args.all:
        print(f"Mode: Processing ALL images in {input_path}")
        image_files = []
        for ext in ["*.tif", "*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(list(input_path.glob(ext)))
            image_files.extend(list(input_path.glob(ext.upper())))
        
        # Filter out existing masks/overlays/comparisons to avoid re-processing them
        files_to_process = []
        for f in image_files:
            if any(x in f.name.lower() for x in ["_mask", "_overlay", "_compare", "mask_", "overlay_"]):
                continue
            files_to_process.append(f.stem)
        
        files_to_process = sorted(list(set(files_to_process)))
        print(f"Found {len(files_to_process)} input images (skipped masks/overlays).")
    else:
        # Default to the "selected_tiled_masks" folder if not running on --all
        mask_dir = Path("data/selected_tiled_masks")
        if not mask_dir.exists():
            print(f"Error: {mask_dir} not found. Use --all and --input_dir to process a specific folder.")
            return
        failed_files = list(mask_dir.glob("*.png"))
        files_to_process = [f.stem for f in failed_files]
        print(f"Mode: Rescuing {len(files_to_process)} specific failed cases from selected_tiled_masks.")

    # 2. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MatSAM (vit_l) on {device}...")
    model = MatSAMModel(
        model_type="vit_l", 
        checkpoint_path="models/sam_weights/sam_vit_l_0b3195.pth", 
        device=device
    )
    
    # 3. Setup Preprocessor
    preprocessor = SEMPreprocessor(**PREPROCESSOR_PRESETS["boundaries"])
    current_preset = PIPELINE_PRESETS["boundaries"].copy()
    current_preset["frangi_scales"] = BEST_SCALES
    current_preset["clahe_clip"] = BEST_CLAHE

    for base_name in files_to_process:
        # Try finding the image file with various extensions in the input_dir
        img_path = None
        for ext in [".tif", ".png", ".jpg", ".jpeg"]:
            if (input_path / (base_name + ext)).exists():
                img_path = input_path / (base_name + ext)
                break
            if (input_path / (base_name + ext.upper())).exists():
                img_path = input_path / (base_name + ext.upper())
                break
        
        if not img_path:
            print(f"Skipping {base_name}: Image not found")
            continue
            
        print(f"Processing {img_path.name}...")
        
        image_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None: continue
        
        # Preprocessing: Only apply if explicitly requested
        if args.preprocess:
            # Enhance
            i_lines, i_base, i_fused, intermediates = preprocessor.preprocess_dual(
                image_gray, 
                **current_preset
            )
            
            # input_gray calculation using blend
            fused_vis = (i_fused * 255).clip(0, 255).astype(np.uint8)
            if BEST_BLEND < 1.0:
                input_gray = cv2.addWeighted(image_gray, BEST_BLEND, fused_vis, 1-BEST_BLEND, 0)
            else:
                input_gray = image_gray # Pure raw
        else:
            # No preprocessing - use image as-is
            input_gray = image_gray
            
        input_rgb = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)
        
        # Run MatSAM
        masks = model.generate_auto_masks(input_rgb, use_global=False)
        
        # Aggregate masks
        final_mask = np.zeros(image_gray.shape, dtype=np.uint8)
        masks = sorted(masks, key=lambda x: x['area'])
        
        image_area = image_gray.shape[0] * image_gray.shape[1]
        valid_masks = [m for m in masks if m['area'] < 0.5 * image_area]
        
        kernel = np.ones((3,3), np.uint8)
        for m in valid_masks:
            mask_data = m['segmentation'].astype(np.uint8)
            eroded = cv2.erode(mask_data, kernel, iterations=1)
            final_mask[eroded > 0] = 255
            
        # Save mask: Strip enhancement tags to get the 'clean' original name
        clean_name = base_name
        for tag in ["_processed", "_enhanced", "_img"]:
            if clean_name.endswith(tag):
                clean_name = clean_name[:-len(tag)]
        
        save_path = final_mask_dir / (clean_name + ".png")
        cv2.imwrite(str(save_path), final_mask)
        
        # Optional Overlay: Preprocessed image | Binary mask (side-by-side)
        if args.overlay:
            # Convert mask to RGB for consistent stacking
            mask_rgb = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
            
            # Create side-by-side: preprocessed image (left) | mask (right)
            overlay = np.hstack((input_rgb, mask_rgb))
            overlay_path = final_mask_dir / (base_name + "_overlay.png")
            cv2.imwrite(str(overlay_path), overlay)
            
            print(f"  Saved: {save_path.name}, {overlay_path.name} ({len(valid_masks)} grains)")
        else:
            print(f"  Saved mask: {save_path.name} ({len(valid_masks)} grains)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Process all images in input_dir instead of just failures")
    parser.add_argument("--input_dir", type=str, default="data/tiled_images", help="Folder containing images to process")
    parser.add_argument("--output_dir", type=str, default="data/tiled_masks", help="Folder where masks will be saved")
    parser.add_argument("--preprocess", action="store_true", help="Apply preprocessing (CLAHE, Frangi, etc.) before MatSAM. Default: False (use images as-is)")
    parser.add_argument("--blend", type=float, default=0.0, help="Blend ratio when preprocessing is enabled (1.0 = raw, 0.0 = full enhancement)")
    parser.add_argument("--overlay", action="store_true", help="Save side-by-side comparison with _compare suffix")
    args = parser.parse_args()
    
    apply_rescue(args)
