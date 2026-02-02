"""
Grid search script for optimizing SEM image enhancement parameters on a single image.
Sweeps across Frangi, DoG, Dirt Filtering, and CLAHE models to find optimal visual contrast.

Example command:
python gridsearch_single_image.py --name "Ni foil (Channel 1) 018 - 00132_tile_0_0" --scales 0.3 0.7 --dog_sigmas 1.0 3.0 5.0 10.0 --dirt_thresholds 0.1 0.2 0.3 --clips 5.0 15.0 20.0 25.0 --blends 0.0 0.3 0.6 0.9
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

def resolve_image_path(image_name):
    """Finds the image file either from a direct path or by searching standard dirs."""
    img_path = Path(image_name)
    
    # 1. Check if it's a direct valid file path
    if img_path.exists() and img_path.is_file():
        return img_path
        
    # 2. Try searching in standard directories
    img_dirs = [Path("data/tiled_images"), Path("data/clean_images")]
    for d in img_dirs:
        for ext in [".tif", ".png", ".jpg", ".jpeg", ".TIF", ".PNG"]:
            test_path = d / f"{image_name}{ext}"
            if test_path.exists() and test_path.is_file():
                return test_path
                
    return None

def grid_search_visual(args):
    # 1. Resolve Image Path
    img_path = resolve_image_path(args.name)
    if not img_path:
        print(f"Error: Could not find image '{args.name}'.")
        print("Please provide a full path or a valid filename from data/tiled_images.")
        return

    print(f"Loading image: {img_path}")
    image_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print(f"Error: Could not decode image at {img_path}")
        return
        
    raw_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    
    # 2. Prepare Parameters
    # Create all combinations of DoG sigma pairs (Cartesian product)
    dog_options = [(small, large) for small in args.dog_sigma_small for large in args.dog_sigma_large]
            
    # Parse scales into tuples
    scales_options = [(s, s*2, s*4) for s in args.scales]
    
    # 3. Setup Output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = SEMPreprocessor(**PREPROCESSOR_PRESETS["boundaries"])
    
    print(f"Generating 4-model grid (Compare: {args.compare})...")
    total = len(args.blends) * len(args.clips) * len(scales_options) * len(dog_options) * len(args.dirt_thresholds)
    count = 0
    
    for scales in scales_options:
        for dog in dog_options:
            for dirt_t in args.dirt_thresholds:
                for clip in args.clips:
                    for blend in args.blends:
                        count += 1
                        current_preset = PIPELINE_PRESETS["boundaries"].copy()
                        current_preset["frangi_scales"] = scales
                        current_preset["dog_sigma_small"] = dog[0]
                        current_preset["dog_sigma_large"] = dog[1]
                        current_preset["clahe_clip"] = clip
                        current_preset["dirt_threshold"] = dirt_t
                        
                        _, _, i_fused, _ = preprocessor.preprocess_dual(image_gray, **current_preset)
                        fused_vis = (i_fused * 255).clip(0, 255).astype(np.uint8)
                        
                        if blend > 0:
                            input_gray = cv2.addWeighted(image_gray, blend, fused_vis, 1-blend, 0)
                        else:
                            input_gray = fused_vis
                        
                        # Base filename (include both small and large DoG sigmas)
                        base_filename = f"F{scales[0]}_D{dog[0]}-{dog[1]}_T{dirt_t}_C{clip}_B{blend}"
                        
                        # Always save the processed image
                        processed_path = out_dir / f"{base_filename}_processed.png"
                        cv2.imwrite(str(processed_path), input_gray)
                        
                        # If compare flag, also save comparison image
                        if args.compare:
                            input_rgb = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)
                            comparison = np.hstack((raw_rgb, input_rgb))
                            compare_path = out_dir / f"{base_filename}_compare.png"
                            cv2.imwrite(str(compare_path), comparison)
                        
                        if count % 10 == 0:
                            print(f" Progress: {count}/{total}")

    print(f"\nGrid search completed. Results saved to: {out_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Image name or full path")
    parser.add_argument("--scales", type=float, nargs='+', default=[0.3, 0.7], help="Frangi scales")
    parser.add_argument("--dog_sigma_small", type=float, nargs='+', default=[1.0], help="DoG small sigma values")
    parser.add_argument("--dog_sigma_large", type=float, nargs='+', default=[3.0], help="DoG large sigma values")
    parser.add_argument("--dirt_thresholds", type=float, nargs='+', default=[0.1, 0.2, 0.5], help="Dirt thresholds")
    parser.add_argument("--clips", type=float, nargs='+', default=[5.0, 15.0, 30.0], help="CLAHE clip limits")
    parser.add_argument("--blends", type=float, nargs='+', default=[0.0, 0.3, 0.6], help="Raw blending ratios")
    parser.add_argument("--output_dir", type=str, required=True, help="Explicit output directory for grid images")
    parser.add_argument("--compare", action="store_true", help="Side-by-side comparison with original")
    
    args = parser.parse_args()
    grid_search_visual(args)
