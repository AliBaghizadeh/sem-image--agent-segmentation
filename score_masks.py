import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def score_masks():
    mask_dir = Path("data/tiled_masks")
    output_path = Path("data/mask_quality.csv")
    
    if not mask_dir.exists():
        print(f"Error: Mask directory {mask_dir} not found.")
        return
        
    mask_files = list(mask_dir.glob("*.png"))
    print(f"Scoring {len(mask_files)} masks...")
    
    scores = []
    
    for mask_path in tqdm(mask_files):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Metric 1: Grain Count
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        grain_count = num_labels - 1 # Subtract background
        
        # Metric 2: Coverage
        coverage = np.sum(mask > 0) / mask.size
        
        # Metric 3: Boundary Smoothness (Jaggedness)
        # We calculate the perimeter to area ratio
        # Low ratio = smooth, High ratio = jagged/noisy
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
        total_area = np.sum(mask > 0)
        
        # normalize jaggedness by area (sqrt to keep units comparable)
        jaggedness = total_perimeter / (np.sqrt(total_area) + 1e-8) if total_area > 0 else 0
        
        # Metric 4: Average Grain Size
        avg_grain_size = total_area / grain_count if grain_count > 0 else 0
        
        scores.append({
            "image": mask_path.stem,
            "grain_count": grain_count,
            "coverage": coverage,
            "jaggedness": jaggedness,
            "avg_grain_size": avg_grain_size
        })
        
    df = pd.DataFrame(scores)
    
    # Define a "Failure Probability" or "Quality Score"
    # Low quality if grain count is extremely high or low, or high jaggedness
    # Heuristic for bad masks:
    # 1. grain_count < 10 or > 1000
    # 2. coverage < 0.05 or > 0.95
    # 3. jaggedness > 30 (very noisy)
    
    df["is_bad_heuristic"] = (
        (df["grain_count"] < 10) | 
        (df["grain_count"] > 1000) | 
        (df["coverage"] < 0.05) | 
        (df["coverage"] > 0.95) |
        (df["jaggedness"] > 40)
    )
    
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"Identified {df['is_bad_heuristic'].sum()} potential bad masks.")

if __name__ == "__main__":
    score_masks()
