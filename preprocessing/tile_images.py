


"""
Script to tile large SEM images into fixed-size patches (e.g., 1024x1024 or 1024x921).
This allows handling non-square images by specifying height and width independently.

Typical usage (Square):
    python preprocessing/tile_images.py --input_dir "data/raw_images" --output_dir "data/tiled_images" --tile_height 1024 --tile_width 1024

Typical usage (Rectangular - Matching Training Data):
    python preprocessing/tile_images.py --input_dir "data/raw_images" --output_dir "data/tiled_images" --tile_height 921 --tile_width 1024
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Tile large SEM images into fixed-size patches.")
    
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Path to the directory containing raw SEM images.")
    
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Path to the directory where tiled patches will be saved.")
    
    parser.add_argument("--tile_height", type=int, default=1024, 
                        help="Target patch height. Default is 1024.")
    
    parser.add_argument("--tile_width", type=int, default=1024, 
                        help="Target patch width. Default is 1024.")
    
    parser.add_argument("--overlap", type=float, default=0.2, 
                        help="Minimum overlap fraction (0.0 to 1.0) between patches. Default 0.2.")

    parser.add_argument("--extensions", type=str, nargs="+", default=["png", "jpg", "jpeg", "tif", "tiff"],
                        help="List of file extensions to process.")
    
    return parser.parse_args()

def get_tiles(img, tile_h, tile_w, min_overlap):
    h, w = img.shape[:2]
    tiles = []
    
    # Pad if image is smaller than tile size
    if h < tile_h or w < tile_w:
        pad_h = max(0, tile_h - h)
        pad_w = max(0, tile_w - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        h, w = img.shape[:2]

    def get_coords(d_size, t_size, overlap):
        if d_size == t_size:
            return [0]
            
        stride = int(t_size * (1 - overlap))
        coords = []
        x = 0
        while x + t_size <= d_size:
            coords.append(x)
            x += stride
            
        if coords[-1] + t_size < d_size:
            coords.append(d_size - t_size)
        
        return sorted(list(set(coords)))

    y_coords = get_coords(h, tile_h, min_overlap)
    x_coords = get_coords(w, tile_w, min_overlap)

    for y in y_coords:
        for x in x_coords:
            patch = img[y:y+tile_h, x:x+tile_w]
            tiles.append((patch, y, x))
            
    return tiles

def process_images(args):
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = []
    if input_path.is_file():
        files.append(input_path)
    else:
        for ext in args.extensions:
            files.extend(input_path.glob(f"**/*.{ext}"))
            files.extend(input_path.glob(f"**/*.{ext.upper()}"))
    
    print(f"[INFO] Found {len(files)} images to tile.")
    
    count = 0
    for file_path in tqdm(files, desc="Tiling"):
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                continue
                
            tiles = get_tiles(img, args.tile_height, args.tile_width, args.overlap)
            
            for i, (patch, y, x) in enumerate(tiles):
                save_name = f"{file_path.stem}_tile_{y}_{x}{file_path.suffix}"
                cv2.imwrite(str(output_path / save_name), patch)
                count += 1
                
        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")
            
    print(f"[INFO] Created {count} tiles in {output_path}")

if __name__ == "__main__":
    args = parse_args()
    process_images(args)
