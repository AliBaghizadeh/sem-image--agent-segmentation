"""
Unified script for testing MatSAM on a single SEM image with line enhancement.
Allows manual control over input image, output directory, and enhancement parameters.
"""
import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add project paths
sys.path.append(os.getcwd())
line_enhancement_path = str(Path(__file__).parent / "Line enhancement")
if line_enhancement_path not in sys.path:
    sys.path.append(line_enhancement_path)

from sem_line_enhancer.pipeline import SEMPreprocessor
from sem_line_enhancer.presets import PIPELINE_PRESETS, PREPROCESSOR_PRESETS
from core.matsam.matsam_model import MatSAMModel

def run_single_test(input_path, output_dir, blend, clip, scales):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input image {input_path} not found.")
        return

    # 1. Load Image
    image_gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print(f"Error: Could not load {input_path}")
        return

    # 2. Enhance
    print(f"Enhancing {input_path.name}...")
    preprocessor = SEMPreprocessor(**PREPROCESSOR_PRESETS["boundaries"])
    current_preset = PIPELINE_PRESETS["boundaries"].copy()
    current_preset["frangi_scales"] = scales
    current_preset["clahe_clip"] = clip
    
    # Process
    _, _, i_fused, _ = preprocessor.preprocess_dual(image_gray, **current_preset)
    
    # Blending with raw
    fused_vis = (i_fused * 255).clip(0, 255).astype(np.uint8)
    if blend > 0:
        input_gray = cv2.addWeighted(image_gray, blend, fused_vis, 1-blend, 0)
    else:
        input_gray = fused_vis
        
    input_rgb = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)

    # 3. Predict with MatSAM
    if not args.skip_sam:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MatSAM (vit_l) on {device}...")
        model = MatSAMModel(
            model_type="vit_l", 
            checkpoint_path="models/sam_weights/sam_vit_l_0b3195.pth", 
            device=device
        )
        
        print("Generating masks (approx. 60 seconds)...")
        masks = model.generate_auto_masks(input_rgb, use_global=False)
        
        # 4. Create Binary Mask & Visual Overlay
        final_mask = np.zeros(image_gray.shape, dtype=np.uint8)
        image_area = image_gray.shape[0] * image_gray.shape[1]
        
        grain_count = 0
        for m in sorted(masks, key=lambda x: x['area']):
            if m['area'] < 0.5 * image_area:
                mask_data = m['segmentation'].astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                eroded = cv2.erode(mask_data, kernel, iterations=1)
                final_mask[eroded > 0] = 255
                grain_count += 1

        # 5. Save Results
        mask_name = input_path.stem + "_mask.png"
        overlay_name = input_path.stem + "_overlay.png"
        cv2.imwrite(str(output_dir / mask_name), final_mask)
        
        overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        overlay[final_mask > 0] = [255, 0, 0] # Red
        comparison = np.hstack((cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB), overlay))
        cv2.imwrite(str(output_dir / overlay_name), comparison)
        
        print(f"\nDone! Detected {grain_count} grains.")
        print(f"Mask saved to: {output_dir / mask_name}")
        print(f"Visual overlay saved to: {output_dir / overlay_name}")
    else:
        # Just save the enhanced image
        enhanced_name = input_path.stem + "_enhanced.png"
        cv2.imwrite(str(output_dir / enhanced_name), input_gray)
        print(f"\nDone! Enhanced image (no mask) saved to: {output_dir / enhanced_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input image file")
    parser.add_argument("--output_dir", type=str, default="data/evaluation", help="Directory for output")
    parser.add_argument("--blend", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--scales", type=float, nargs='+', default=[0.3, 0.7, 1.5])
    parser.add_argument("--skip_sam", action="store_true", help="Skip SAM masking and only save the enhanced image")
    args = parser.parse_args()
    
    run_single_test(args.input, args.output_dir, args.blend, args.clip, tuple(args.scales))
