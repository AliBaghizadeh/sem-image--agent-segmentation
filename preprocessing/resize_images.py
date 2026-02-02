
"""
Script to resize SEM images to target dimensions (default 1024x1024) for MatSAM.

Typical usage:
    python preprocessing/resize_images.py --input_dir "data/raw_images" --output_dir "data/resized_images" --size 1024
"""

import os
import cv2
import argparse
from pathlib import Path
import glob
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Resize SEM images to target dimensions (default 1024x1024) for MatSAM.")
    
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Path to the directory containing raw SEM images.")
    
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Path to the directory where resized images will be saved.")
    
    parser.add_argument("--size", type=int, default=1024, 
                        help="Target image size (height and width). Default is 1024.")
    
    parser.add_argument("--extensions", type=str, nargs="+", default=["png", "jpg", "jpeg", "tif", "tiff"],
                        help="List of file extensions to process. Default: png jpg jpeg tif tiff")
    
    return parser.parse_args()

def resize_images(input_dir, output_dir, size, extensions):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all file paths
    files = []
    for ext in extensions:
        # Case insensitive search would be better but requires more complex globbing
        # Simple recursive glob handling
        files.extend(input_path.glob(f"**/*.{ext}"))
        files.extend(input_path.glob(f"**/*.{ext.upper()}"))
    
    print(f"[INFO] Found {len(files)} images in {input_dir}. Resizing to {size}x{size}...")
    
    for file_path in tqdm(files, desc="Processing Images"):
        try:
            # Read image
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"[WARNING] Could not read file: {file_path}")
                continue
            
            # Resize
            # Use INTER_AREA for shrinking, INTER_LINEAR/CUBIC for enlarging
            # We assume shrinking mostly for SEM (often high res)
            resized_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            
            # Construct output path (maintain relative structure if desired, 
            # but for now simpler to just flat save or strict mirror. 
            # Flat save with name collision handling is safer for simple scripts, 
            # but usually mirroring is better. 
            # Let's keep it simple: flat save to output_dir, using stem)
            
            # To avoid overwriting differing files with same name from diff subfolders:
            # We can prefix with parent folder name
            save_name = f"{file_path.stem}_resized{file_path.suffix}"
            save_path = output_path / save_name
            
            cv2.imwrite(str(save_path), resized_img)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

    print(f"[INFO] Processing complete. Resized images saved to {output_dir}")

def main():
    args = parse_args()
    resize_images(args.input_dir, args.output_dir, args.size, args.extensions)

if __name__ == "__main__":
    main()
