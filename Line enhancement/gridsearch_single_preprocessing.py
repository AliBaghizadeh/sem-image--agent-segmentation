"""
diagnose_preprocessing.py
=========================

High-Throughput Parameter Grid Search for SEM Preprocessing
-----------------------------------------------------------

Purpose
-------
Systematically evaluate and optimize preprocessing parameters for
Scanning Electron Microscopy (SEM) images. The script explores grids
of line-enhancement, artifact-removal, and contrast-restoration
parameters, computing per-image statistics to identify optimal
combinations for MatSAM segmentation and downstream classifiers.

Key Stages
----------
1. **Frangi Line Enhancement** – amplifies faint ferroelastic lines.
2. **Dirt Blob Inpainting** – removes small bright contaminants.
3. **CLAHE + Bilateral Filtering** – balances global contrast and denoises.
4. **Statistical Evaluation** – computes min, max, mean, std, p1, p99.
5. **Best-Config Selection** – chooses the most balanced, high-contrast
   settings (mean ≈ 0.5 ± 0.05, max std).

Outputs
-------
- `data/diagnostics/gridsearch_stats.csv`
- Diagnostic plots and `.npy` arrays of best configurations under
  `data/diagnostics/gridsearch_panels/best_results/`.

Usage
-----
    python diagnose_preprocessing.py
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import gc
import sys
import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2
import tifffile
import matplotlib.pyplot as plt
from skimage import filters

sys.path.append(str(Path(__file__).parent.parent))
from sem_line_enhancer.enhancers import enhance_lines_frangi, filter_dirt_blobs


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
RAW_DIR = Path("data/raw")
BASE_DIAG_DIR = Path("data/diagnostics")
GRID_DIR = BASE_DIAG_DIR / "gridsearch_panels"
BEST_DIR = GRID_DIR / "best_results"

for d in [BASE_DIAG_DIR, GRID_DIR, BEST_DIR]:
    d.mkdir(exist_ok=True, parents=True)

OUT_STATS = BASE_DIAG_DIR / "gridsearch_stats.csv"

# File selection
RAW_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
RAW_FILES = [f for f in RAW_DIR.iterdir() if f.suffix.lower() in RAW_EXTS]
N_SELECT = 50
SELECTED = random.sample(RAW_FILES, min(N_SELECT, len(RAW_FILES)))
print("SELECTED", SELECTED, "raw files", len(RAW_FILES))
# Parameter grids
FRANGI_SCALES_GRID = [(0.5, 1.0, 2.0), (0.3, 0.7, 1.5), (0.2, 0.5, 0.8)]
FRANGI_ALPHA_GRID = [0.2, 0.25, 0.3] # [0.2, 0.35, 0.45, 0.55]

DIRT_THRESHOLD_GRID = [0.08, 0.1, 0.12, 0.2] #[0.12, 0.15]
DIRT_MAX_SIZE_GRID = [20, 25, 30, 35]       #[30, 40]
CLAHE_CLIP_GRID = [20, 25, 30]
CLAHE_TILE_GRID = [4, 8]  
BILAT_D_GRID = [4, 8]
BILAT_SIGMA_COLOR_GRID = [30, 40]
BILAT_SIGMA_SPACE_GRID = [30, 40]


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def preprocess_contrast(img: np.ndarray,
                        clip: int, tile: int,
                        d: int, sig_c: int, sig_s: int) -> np.ndarray:
    """
    Apply CLAHE and bilateral filtering to improve contrast and reduce noise.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (float32 [0, 1]).
    clip, tile : int
        CLAHE parameters.
    d, sig_c, sig_s : int
        Bilateral filter parameters.

    Returns
    -------
    np.ndarray : float32 image in [0, 1].
    """
    if img.dtype != np.uint8:
        img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img8 = img

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    clahe_img = clahe.apply(img8)
    bilat = cv2.bilateralFilter(clahe_img, d, sig_c, sig_s)
    return bilat.astype(np.float32) / 255.0


def safe_dirt_filter(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    Safe wrapper around `filter_dirt_blobs` to handle dtype issues
    and GPU/Telea inpainting fallbacks.
    """
    try:
        result = filter_dirt_blobs(img, **kwargs)
    except cv2.error:
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        result = filter_dirt_blobs(img_uint8.astype(np.float32) / 255.0, **kwargs)
    if isinstance(result, tuple):
        result = next((x for x in result if isinstance(x, np.ndarray)), img)
    return result.astype(np.float32)


# ---------------------------------------------------------------------
# Core Image Evaluation
# ---------------------------------------------------------------------
def process_image(path: Path) -> list[dict]:
    """
    Execute the full parameter grid search for one image.

    Parameters
    ----------
    path : Path
        SEM image path.

    Returns
    -------
    list[dict]
        Statistics for each parameter combination.
    """
    img_name = path.stem
    results = []

    # Load grayscale & normalize
    if path.suffix.lower() in [".tif", ".tiff"]:
        img = tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = (img - img.min()) / (np.ptp(img) + 1e-8)
    img = img.astype(np.float32)

    for scales in FRANGI_SCALES_GRID:
        for alpha in FRANGI_ALPHA_GRID:
            frangi_img = enhance_lines_frangi(img, scales=scales, blend_alpha=alpha)

            for dirt_t in DIRT_THRESHOLD_GRID:
                for dirt_max in DIRT_MAX_SIZE_GRID:
                    clean = safe_dirt_filter(frangi_img,
                                             threshold=dirt_t,
                                             max_size=dirt_max,
                                             aspect_ratio_thresh=1.4,
                                             inpaint_radius=7)

                    for clip in CLAHE_CLIP_GRID:
                        for tile in CLAHE_TILE_GRID:
                            for d in BILAT_D_GRID:
                                for sig_c in BILAT_SIGMA_COLOR_GRID:
                                    for sig_s in BILAT_SIGMA_SPACE_GRID:
                                        try:
                                            proc = preprocess_contrast(clean, clip, tile, d, sig_c, sig_s)
                                            stat = dict(
                                                image=img_name,
                                                config=f"frangi{scales}_a{alpha}_"
                                                       f"dirtT{dirt_t}_M{dirt_max}_"
                                                       f"clip{clip}_tile{tile}_"
                                                       f"d{d}_col{sig_c}_sp{sig_s}",
                                                min=float(proc.min()),
                                                max=float(proc.max()),
                                                mean=float(proc.mean()),
                                                std=float(proc.std()),
                                                p1=float(np.percentile(proc, 1)),
                                                p99=float(np.percentile(proc, 99)),
                                            )
                                            results.append(stat)
                                        except Exception as e:
                                            print(f"[WARN] {img_name}: {e}")
                                            continue
    return results


# ---------------------------------------------------------------------
# Result Aggregation
# ---------------------------------------------------------------------
def select_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select best parameter set per image based on balanced mean and high std.
    """
    best = []
    for name, group in df.groupby("image"):
        subset = group[(group["mean"] > 0.45) & (group["mean"] < 0.55)]
        best_cfg = (subset if not subset.empty else group).sort_values("std", ascending=False).iloc[0]
        best.append(best_cfg)
    return pd.DataFrame(best)


def reprocess_best(df_best: pd.DataFrame):
    """
    Regenerate and save diagnostic panels for each best configuration.
    """
    for _, row in df_best.iterrows():
        name, config = row["image"], row["config"]
        print(f"Reprocessing best config for {name}...")
        parts = config.split("_")
        clahe_clip = int(parts[4][4:])
        clahe_tile = int(parts[5][4:])
        bilat_d = int(parts[6][1:])
        sig_c = int(parts[7][3:])
        sig_s = int(parts[8][2:])

        file = next(RAW_DIR.glob(f"{name}.*"), None)
        if not file:
            print(f"⚠ Missing raw file for {name}")
            continue

        img = tifffile.imread(str(file)) if file.suffix.lower() in [".tif", ".tiff"] \
            else cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        img = (img - img.min()) / (np.ptp(img) + 1e-8)
        img_prep = preprocess_contrast(img, clahe_clip, clahe_tile, bilat_d, sig_c, sig_s)

        np.save(BEST_DIR / f"{name}_{config}.npy", img_prep)
        p1, p99 = np.percentile(img_prep, [1, 99])
        renorm = np.clip((img_prep - p1) / (p99 - p1 + 1e-8), 0, 1)
        edges = filters.sobel(renorm)

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        axes[0].imshow(img, cmap="gray"); axes[0].set_title("Raw"); axes[0].axis("off")
        axes[1].imshow(img_prep, cmap="gray"); axes[1].set_title("Preprocessed"); axes[1].axis("off")
        axes[2].hist(img_prep.ravel(), bins=100, color="blue", alpha=0.7); axes[2].set_title("Histogram")
        axes[3].imshow(renorm, cmap="gray"); axes[3].imshow(edges, cmap="hot", alpha=0.5)
        axes[3].set_title("Edges Overlay"); axes[3].axis("off")
        plt.suptitle(f"{name} | {config}", fontsize=12)
        plt.savefig(BEST_DIR / f"{name}_{config}_panel.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        gc.collect()


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
def main():
    """
    Run full multi-core grid search and diagnostics.
    """
    print(f"Found {len(SELECTED)} images in {RAW_DIR}")
    all_results = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = {ex.submit(process_image, f): f for f in SELECTED}
        for i, fut in enumerate(as_completed(futures), 1):
            file = futures[fut]
            try:
                res = fut.result()
                all_results.extend(res)
                print(f"[{i}/{len(SELECTED)}] ✓ {file.stem}: {len(res)} configs")
            except Exception as e:
                print(f"[ERROR] {file.stem}: {e}")

    if not all_results:
        print("⚠ No results produced — check data or parameters.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(OUT_STATS, index=False)
    print(f"\nGrid search complete: {len(df)} parameter-image results saved to {OUT_STATS}")

    best_df = select_best_configs(df)
    best_df.to_csv(BEST_DIR / "best_results_stats.csv", index=False)
    print(f"✓ Best parameter sets saved to {BEST_DIR / 'best_results_stats.csv'}")

    reprocess_best(best_df)
    print("\n✅ Full preprocessing grid search completed successfully.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
