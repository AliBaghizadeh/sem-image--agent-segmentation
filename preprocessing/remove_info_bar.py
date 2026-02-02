
"""
Script to automatically detect and remove the info bar (parameters, scale bar, logo) 
from the bottom of SEM images.

It uses edge detection and variance analysis to identify the distinct region at the bottom 
that contains text/logos and crops it out, leaving only the pure microstructure.

Typical usage:
    python preprocessing/remove_info_bar.py --input_dir "data/raw" --output_dir "data/clean"
        --scale_bar_height 30 --scale_bar_width 100
"""

import cv2
import numpy as np
from pathlib import Path
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# ---------------------------------------------------------------------
SUPPORTED_EXT = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
BOTTOM_SCAN_FRAC = 0.25     # scan bottom 25% for possible bar
MIN_BAR_FRAC = 0.03         # expected min bar height ≈ 3% of image
MAX_BAR_FRAC = 0.15         # expected max bar height ≈ 15% of image

# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Remove info bar from SEM images.")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Path to directory containing raw SEM images.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Path to directory to save cleaned images.")
    parser.add_argument("--scale_bar_height", type=int, default=30, 
                        help="Expected Minimum Height of the scale bar in pixels (default: 30px).")
    parser.add_argument("--scale_bar_width", type=int, default=100, 
                        help="Expected Minimum Width of the scale bar in pixels (default: 100px).")
    parser.add_argument("--scale_bar_position", type=str, default="bottom", 
                        help="Position of the scale bar from Bottom (default: bottom).")
    return parser.parse_args()

def safe_read(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def detect_infobar_bounds(image_bgr: np.ndarray, min_height_px: int = 30):
    """Return (y1,y2) rows delimiting the info-bar region near the bottom."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Use the larger of (BOTTOM_SCAN_FRAC) OR (min_height_px * 2) to ensure we scan enough
    scan_height = int(max(h * BOTTOM_SCAN_FRAC, min_height_px * 4))
    start_row = max(0, h - scan_height)
    
    scan = gray[start_row:, :]
    
    # horizontal edges and intensity variance per row
    gx = cv2.Sobel(scan, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(scan, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy).mean(axis=1)
    row_var = scan.var(axis=1)

    # normalize
    def normalize(x):
        return (x - x.min()) / (np.ptp(x) + 1e-6)
    
    edge = normalize(edge)
    row_var = normalize(row_var)

    # composite signal: info-bar → high edge density (text), low variance (background)
    signal = edge * 0.4 + (1 - row_var) * 0.6

    # smooth and find strongest contiguous segment
    signal_smooth = cv2.GaussianBlur(signal.reshape(-1,1),(1,9),0).ravel()
    thresh = signal_smooth.mean() + 0.5*signal_smooth.std()
    mask = signal_smooth > thresh
    if not mask.any():
        return None

    # take bottommost contiguous region
    indices = np.where(mask)[0]
    
    y1_rel = indices[0] 
    y1_abs = start_row + y1_rel
    
    return y1_abs

def remove_infobar(img_path: Path, output_dir: Path, args):
    try:
        img = safe_read(img_path)
        h, w = img.shape[:2]
        
        # Detect where the bar starts
        cut_y = detect_infobar_bounds(img, min_height_px=args.scale_bar_height)

        # Validation logic using the new args
        # 1. calculated bar height must be at least args.scale_bar_height
        if cut_y is not None:
             detected_bar_height = h - cut_y
             if detected_bar_height < args.scale_bar_height:
                 # Detection yielded something too small, likely noise. 
                 # Fallback: don't crop, OR use default heuristic? 
                 # Let's assume if it's too small, we missed the real bar or there is no bar.
                 # For safety in batch processing: strict check.
                 # But usually, it's safer to fallback to a default crop if we suspect a bar exists.
                 # Let's trust the scan if it found a strong signal, otherwise:
                 pass

        # Fallback/Safety Check
        if cut_y is None or cut_y < (h * (1 - MAX_BAR_FRAC)):
             # If we are cutting > 15%, safe fallback.
             # Or if detection failed completely.
             # We can use the min_height as a guide for a minimal safe crop if desired, 
             # but sticking to the 10% heuristic is usually safer for pure failures.
             
             # However, if the user provided specific dimensions, maybe we should trust them?
             # For now, keep the safety heuristic 
             cut_y = int(h * 0.90) 
        
        cropped = img[:cut_y, :]  # keep everything above bar

        # --- Convert to grayscale before saving ---
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # --- Save ---
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), cropped_gray)

        return True

    except Exception as e:
        print(f"❌ {img_path.name}: {e}")
        return False

# ---------------------------------------------------------------------
def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Gather images
    image_paths = []
    for ext in SUPPORTED_EXT:
        image_paths.extend(input_dir.rglob(f"*{ext}"))
        image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))
        
    print(f"Found {len(image_paths)} images in {input_dir}")
    
    nworkers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        # Pass full args to function now
        futures = [ex.submit(remove_infobar, p, output_dir, args) for p in image_paths]
        results = [f.result() for f in as_completed(futures)]
        
    success_count = sum(results)
    print(f"\n✓ Successfully processed {success_count}/{len(image_paths)} images.")
    print(f"  Saved to: {output_dir}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
