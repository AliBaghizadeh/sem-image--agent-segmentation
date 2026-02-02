"""
Script to generate auto-masks for raw images using MatSAM.
Typical usage:
    python preprocessing/auto_label.py --input_dir "data/tiled_images" --output_dir "data/tiled_masks"

Prevention of Background Collapse:
    If your masks are covering the entire image (turning everything green), use --max_area_frac 
    to filter out giant masks. For example, --max_area_frac 0.5 will ignore any grain 
    that covers more than 50% of the image.
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add project root to path so 'core' can be found
sys.path.append(os.getcwd())

from tqdm import tqdm
import torch
from core.matsam.matsam_model import MatSAMModel

def parse_args():
    parser = argparse.ArgumentParser(description="Generate auto-masks for raw images using MatSAM.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing raw tiled images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder where generated masks will be saved.")
    parser.add_argument("--checkpoint", type=str, default="models/sam_weights/sam_vit_l_0b3195.pth")
    parser.add_argument("--model_type", type=str, default="vit_l")
    parser.add_argument("--max_area_frac", type=float, default=0.5, 
                        help="Maximum area fraction (0.0 to 1.0) for a single mask. Masks larger than this are ignored. Default 0.5.")
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MatSAM model ({args.model_type}) on {device}...")
    model = MatSAMModel(model_type=args.model_type, checkpoint_path=args.checkpoint, device=device)

    # 2. Gather images
    extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(input_path.glob(f"*{ext}")))
    
    print(f"Found {len(image_paths)} images. Generating pseudo-labels...")

    for img_p in tqdm(image_paths, desc="Auto-Labeling"):
        try:
            # 0. Check if mask already exists to skip
            save_path = output_path / (img_p.stem + ".png")
            if save_path.exists():
                # print(f"Skipping {img_p.name} (already labeled)")
                continue

            # 1. Load Image properly as RGB
            image = cv2.imread(str(img_p))
            if image is None:
                print(f"❌ Could not read {img_p}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # 2. Generate masks
            results = model.generate_auto_masks(image_rgb, use_global=False)
            
            if not results:
                # Create empty mask if nothing detected
                combined_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                h, w = results[0]['segmentation'].shape
                total_pixels = h * w
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                
                # Sort by area to process smallest first (optional, prevents large masks from eating small ones)
                results = sorted(results, key=lambda x: x['area'])
                
                valid_mask_count = 0
                for r in results:
                    mask_area = r['area']
                    # Filter out giant 'background' masks
                    if mask_area < (total_pixels * args.max_area_frac):
                        mask_data = r['segmentation']
                        # Erode each grain to keep boundaries clear
                        kernel = np.ones((3,3), np.uint8)
                        eroded = cv2.erode(mask_data.astype(np.uint8), kernel, iterations=1)
                        combined_mask[eroded > 0] = 255
                        valid_mask_count += 1
                
                if valid_mask_count == 0:
                    print(f"⚠️ Warning: All masks filtered for {img_p.name}")
            
            # Save mask
            save_path = output_path / (img_p.stem + ".png")
            cv2.imwrite(str(save_path), combined_mask)

        except Exception as e:
            print(f"❌ Error processing {img_p.name}: {e}")

    print(f"\n✅ Auto-labeling complete. Masks saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
