
"""
Verification Script: Generates visual overlays to compare MatSAM masks with original images.
Helps qualitatively evaluate the quality of autonomous/pseudo-labels.

Usage Scenarios:
1. Quick random check (Default: 50 samples):
    python preprocessing/verify_masks.py --image_dir "data/tiled_images" --mask_dir "data/tiled_masks"

2. Verify a specific number of random samples:
    python preprocessing/verify_masks.py --image_dir "data/tiled_images" --mask_dir "data/tiled_masks" --num_samples 100

3. Process ALL masks in the directory (e.g., verifying a selected subset):
    python preprocessing/verify_masks.py --image_dir "data/tiled_images" --mask_dir "data/selected_tiled_masks" --num_samples 0
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import sys
import random

# Add project root to path
sys.path.append(os.getcwd())

def parse_args():
    parser = argparse.ArgumentParser(description="Create visual overlays for mask verification.")
    parser.add_argument("--image_dir", type=str, required=True, help="Folder with raw images.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Folder with generated masks.")
    parser.add_argument("--output_dir", type=str, default="data/verification_plots")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of random samples to generate. 0 for all available matches. (Default: 50)")
    parser.add_argument("--opacity", type=float, default=0.25, help="Mask overlay opacity (0.0 to 1.0). Default: 0.25")
    return parser.parse_args()

def create_overlay(image, mask, opacity=0.4):
    """Creates a color overlay on the image using the binary mask."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    color_mask = np.zeros_like(image)
    color_mask[mask > 127] = [0, 255, 0] # Green for grains
    
    overlay = cv2.addWeighted(image, 1.0, color_mask, opacity, 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    
    return overlay

def main():
    args = parse_args()
    img_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather all masks available in the mask_dir
    mask_files = sorted(list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg")) + list(mask_dir.glob("*.tif")))
    print(f"🔍 Found {len(mask_files)} masks in {mask_dir}")

    # 2. Find matching images
    valid_pairs = []
    extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    
    print("Matching masks to images...")
    for mask_p in mask_files:
        found = False
        # Try to find corresponding image with any supported extension
        for ext in extensions:
            img_p = img_dir / (mask_p.stem + ext)
            if img_p.exists():
                valid_pairs.append((img_p, mask_p))
                found = True
                break
        if not found:
            print(f"⚠️ Warning: No image found for mask {mask_p.name}")

    if not valid_pairs:
        print("❌ Error: No matching image-mask pairs found. Check your directory paths.")
        return

    print(f"✅ Found {len(valid_pairs)} matching pairs.")

    # 3. Handle Sampling
    if args.num_samples > 0 and len(valid_pairs) > args.num_samples:
        to_process = random.sample(valid_pairs, args.num_samples)
        print(f"🎲 Randomly sampling {args.num_samples} pairs for verification.")
    else:
        to_process = valid_pairs
        print(f"📂 Processing all {len(to_process)} available pairs.")

    # 4. Generate Overlays
    for img_p, mask_p in tqdm(to_process, desc="Creating Overlays"):
        img = cv2.imread(str(img_p))
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        
        if img is not None and mask is not None:
            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            vis = create_overlay(img, mask, args.opacity)
            cv2.imwrite(str(out_dir / f"verify_{img_p.stem}.jpg"), vis)
        
    print(f"\n🎉 Finished! Created {len(to_process)} verification plots in: {out_dir}")

if __name__ == "__main__":
    main()
