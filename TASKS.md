# 📋 Unified Project Tasks & Workflow

This document combines the project roadmap with the technical execution steps. Use this as your master guide for processing SEM images.

---

## 🚀 SECTION 1: Standard Pipeline (Process All Images)
Use these steps to process your entire dataset from scratch.

### 1️⃣ Tiling
Break large SEM images into 1024x1024 patches.
```powershell
python preprocessing/tile_images.py --input_dir "data/clean_images" --output_dir "data/tiled_images" --tile_height 921 --tile_width 1024
```

### 2️⃣ Auto-Labeling (Batch SAM)
Generate masks for ALL images in your tiled folder.
```powershell
python preprocessing/auto_label.py --input_dir "data/tiled_images" --output_dir "data/tiled_masks" --max_area_frac 0.5
```

### 3️⃣ Quality Check & Evaluation
Score the generated masks to find which ones failed (the "white blobs").
```powershell
python score_masks.py
```

---

## 🛠️ SECTION 2: The Rescue Workflow (For Failed Cases)
Use this when SAM fails on an image. Follow these 3 steps to find optimal parameters and rescue failed masks.

### Step 1: Generate Grid Search Images
Create enhanced versions of one failed image with different parameter combinations to test which enhancement works best.

```powershell
python gridsearch_single_image.py --name "data/selected_tiled_images/Ni foil (Channel 1) 018 - 00785_tile_0_512.tif" --output_dir "data/grid_search/Ni foil (Channel 1) 018 - 00785_tile_0_512" --scales 0.2 0.7 --dog_sigma_small 1.0 5.0 --dog_sigma_large 12.0 15.0 --dirt_thresholds 0.1 0.2 0.5 --clips 4.0 10.0 --blends 0.9 1.1 --compare

python gridsearch_single_image.py --name "data/tiled_images/Ni foil (Channel 1) 018 - 00860_tile_0_0.tif" --output_dir "data/grid_search/grid_Ni foil (Channel 1) 018 - 00860_tile_0_0" --scales 0.2 0.4 --dog_sigma_small 6.0 7.0 --dog_sigma_large 12.0 15.0 --dirt_thresholds 0.4 0.5 0.6 --clips 4.0 --blends 0.9 --compare
```

**Parameters:**
- `--name`: Path to the failed image (from `data/tiled_images`)
- `--output_dir`: Where to save the grid search results
- `--scales`: Frangi filter scales (smaller = finer boundaries, larger = thicker boundaries)
- `--dog_sigmas`: Difference of Gaussian sigma pairs (for edge detection). It has small and large (pair)
- `--dirt_thresholds`: Dirt filtering thresholds (higher = more aggressive cleaning)
- `--clips`: CLAHE clip limits (higher = more contrast enhancement)
- `--blends`: Blending ratios (0.0 = full enhancement, 1.0 = original image)
- `--compare`: Add this flag to generate comparison images (original | enhanced)

**Output:** 
- `_img.png`: Processed/enhanced image
- `_compare.png`: Side-by-side comparison (original | processed) - only if `--compare` is used

---

### Step 2: Generate Masks with Visual Overlays
Run MatSAM on all enhanced images from Step 1 and create overlays to evaluate which parameters detected boundaries correctly.

```powershell
python apply_rescue.py --all --input_dir "data/grid_search/Ni foil (Channel 1) 018 - 00785_tile_0_512" --output_dir "data/grid_search/Ni foil (Channel 1) 018 - 00785_tile_0_512" --blend 1.0 --overlay
```

**Parameters:**
- `--all`: Process all images in the input directory
- `--input_dir`: Folder containing the enhanced images from Step 1
- `--output_dir`: Where to save the generated masks (can be the same as input_dir)
- `--blend 1.0`: Use the enhanced images as-is (don't re-enhance them)
- `--overlay`: Generate overlay images for visual comparison

**Output:**
- `_mask.png`: Binary mask (white grains, black boundaries)
- `_overlay.png`: Side-by-side comparison (original raw image | binary mask)

**What to look for:** Review the `_overlay.png` files to find which parameter set produces masks that accurately trace the grain boundaries visible in the original image.

---

### Step 3: Apply Winning Parameters to All Failed Images
Once you've identified the best parameters from visual inspection in Steps 1-2, apply them to generate the training pairs.

**A. Generate Enhanced Images (Step 1)**
```powershell
python apply_enhancement.py --input_dir "data/selected_tiled_images" --output_dir "data/enhanced_images" --scales 0.2 --dog_sigma_small 5.0 --dog_sigma_large 15.0 --dirt_threshold 0.6 --clip 4.0 --blend 0.9
```
- `--blend 1.0`: Use the enhanced images as-is (don't re-enhance them)
*Output: `tile_0_0_processed.tif`*

**B. Generate Matching Masks (Step 2)**
```powershell
python apply_rescue.py --all --input_dir "data/enhanced_images" --output_dir "data/enhanced_masks" --blend 1.0
```
*Output: `tile_0_0.png`*

> [!TIP]
> **Why this naming?** 
> 1. The image gets `_processed` to avoid overwriting your originals.
> 2. The mask automatically strips `_processed` to give you a clean pair: 
>    `image: tile_0_0_processed.tif` + `mask: tile_0_0.png`. 
>    This is the professional way to manage machine learning datasets.

---

## 🎓 SECTION 3: Training & Specialized Analysis

### 1️⃣ Specialized Fine-Tuning
Re-train MatSAM using the rescued and high-quality masks.
```powershell
python finetuning/train_sam.py --train_data "data/tiled_images" --mask_data "data/tiled_masks" --epochs 25 --batch_size 4 --output_dir "finetuning/runs/exp1" --val_split 0.2
```

### 2️⃣ Roadmap Tracker
- [x] **PHASE 1**: Environment Setup & Baseline (RTX 5080 Verified)
- [x] **PHASE 2**: Core Data Pipeline (Tiling & Auto-Labeling)
- [x] **PHASE 3**: Targeted Enhancement Rescue (Current Focus)
- [ ] **PHASE 4**: Interactive Analysis UI (Agentic Queries)
- [ ] **PHASE 5**: Cloud Scaling (S3 & SageMaker)
