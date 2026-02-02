"""
===============================================================================
SEM Image Info Bar Remover
===============================================================================

PURPOSE
--------
Read all SEM images in `data/raw/` (or `data/test/`), detect the bottom
info-bar region, and save cropped images *without* that bar.

The detection uses horizontal edge/variance analysis near the bottom of each
image, so it works even when the bar height changes across microscopes.

OUTPUT
-------
Cropped images are written to `data/cropped/` with identical filenames.

-------------------------------------------------------------------------------
"""

import cv2
import numpy as np
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
INPUT_DIR = Path("data/1000")          # where your SEM images live
OUTPUT_DIR = Path("data/cropped_1000")      # output folder for cleaned images
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXT = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
BOTTOM_SCAN_FRAC = 0.25     # scan bottom 25 % for possible bar
MIN_BAR_FRAC = 0.03         # expected min bar height ≈3 % of image
MAX_BAR_FRAC = 0.15         # expected max bar height ≈15 % of image
DEBUG = False               # True → show rectangles on screen

# ---------------------------------------------------------------------
def safe_read(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def detect_infobar_bounds(image_bgr: np.ndarray):
    """Return (y1,y2) rows delimiting the info-bar region near the bottom."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    start_row = int(h * (1 - BOTTOM_SCAN_FRAC))
    scan = gray[start_row:, :]
    sh = scan.shape[0]

    # horizontal edges and intensity variance per row
    gx = cv2.Sobel(scan, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(scan, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy).mean(axis=1)
    row_var = scan.var(axis=1)

    # normalize
    edge = (edge - edge.min()) / (np.ptp(edge) + 1e-6)
    row_var = (row_var - row_var.min()) / (np.ptp(row_var) + 1e-6)

    # composite signal: info-bar → high edge, low variance (flat background with text)
    signal = edge * 0.4 + (1 - row_var) * 0.6

    # smooth and find strongest contiguous segment
    signal_smooth = cv2.GaussianBlur(signal.reshape(-1,1),(1,9),0).ravel()
    thresh = signal_smooth.mean() + 0.5*signal_smooth.std()
    mask = signal_smooth > thresh
    if not mask.any():
        return None

    # take bottommost contiguous region
    indices = np.where(mask)[0]
    y1_rel, y2_rel = indices[0], indices[-1]
    y1_abs = start_row + y1_rel
    y2_abs = start_row + y2_rel

    # clip to plausible range
    bar_min = int(h * (1 - MAX_BAR_FRAC))
    bar_max = int(h * (1 - MIN_BAR_FRAC))
    y1_abs = max(y1_abs, bar_min)
    y2_abs = min(y2_abs, bar_max)

    return y1_abs, y2_abs

def remove_infobar(img_path: Path):
    try:
        img = safe_read(img_path)
        h, w = img.shape[:2]
        bounds = detect_infobar_bounds(img)

        if bounds is None:
            # Fallback: crop fixed 8% from bottom
            cut = int(h * 0.08)
            cropped = img[:h - cut, :]
        else:
            y1, y2 = bounds
            cropped = img[:y1, :]  # keep everything above bar

        # --- Convert to grayscale before saving ---
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # --- Save as single-channel grayscale image ---
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), cropped_gray)

        print(f"{img_path.name:30s} → cropped (gray), saved to {out_path.name}")
        return True

    except Exception as e:
        print(f"❌ {img_path.name}: {e}")
        return False


# ---------------------------------------------------------------------
def main():
    image_paths = [p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in SUPPORTED_EXT]
    print(f"Found {len(image_paths)} images in {INPUT_DIR}")
    nworkers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        futures = [ex.submit(remove_infobar, p) for p in image_paths]
        _ = [f.result() for f in as_completed(futures)]
    print(f"\n✓ Cropped images saved to {OUTPUT_DIR}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
