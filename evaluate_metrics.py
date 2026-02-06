
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

def calculate_iou(pred_mask, gt_mask):
    """Intersection over Union."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / (union + 1e-8)

def calculate_dice(pred_mask, gt_mask):
    """Dice coefficient (F1-score)."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-8)

def calculate_precision_recall(pred_mask, gt_mask):
    """Pixel-wise precision and recall."""
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision, recall

def calculate_hausdorff(pred_mask, gt_mask):
    """Hausdorff distance between boundaries (Optional)."""
    try:
        from scipy.spatial.distance import directed_hausdorff
    except ImportError:
        return np.nan
    
    pred_points = np.argwhere(pred_mask)
    gt_points = np.argwhere(gt_mask)
    
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')
    
    d1 = directed_hausdorff(pred_points, gt_points)[0]
    d2 = directed_hausdorff(gt_points, pred_points)[0]
    return max(d1, d2)

def evaluate_all_metrics(pred_mask, gt_mask):
    """Compute all metrics for a single image."""
    prec, rec = calculate_precision_recall(pred_mask, gt_mask)
    return {
        'iou': calculate_iou(pred_mask, gt_mask),
        'dice': calculate_dice(pred_mask, gt_mask),
        'precision': prec,
        'recall': rec,
        'hausdorff': calculate_hausdorff(pred_mask, gt_mask)
    }

def evaluate_dataset(pred_dir, gt_dir, output_csv):
    """Evaluate all predictions against ground truth."""
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    if not gt_dir.exists():
        print(f"Error: Ground truth directory {gt_dir} not found.")
        return
        
    results = []
    
    print(f"Evaluating predictions in {pred_dir} against {gt_dir}...")
    
    for gt_path in gt_dir.glob("*.png"):
        # Match by filename
        pred_path = pred_dir / gt_path.name
        
        if not pred_path.exists():
            # Try png -> jpg mapping just in case
            pred_path_jpg = pred_dir / (gt_path.stem + ".jpg")
            if pred_path_jpg.exists():
                pred_path = pred_path_jpg
            else:
                print(f"Warning: No prediction for {gt_path.name}")
                continue
        
        # Load and binarize
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        
        if gt_mask is None or pred_mask is None:
            print(f"Error reading {gt_path.name}")
            continue
            
        gt_mask = gt_mask > 127
        pred_mask = pred_mask > 127
        
        # Resize if necessary (robustness)
        if gt_mask.shape != pred_mask.shape:
             pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        
        metrics = evaluate_all_metrics(pred_mask, gt_mask)
        metrics['image'] = gt_path.stem
        results.append(metrics)
    
    if not results:
        print("No matches found for evaluation.")
        return

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Images Evaluated: {len(df)}")
    print(f"Mean IoU:       {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
    print(f"Mean Dice:      {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
    print(f"Mean Precision: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
    print(f"Mean Recall:    {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
    
    # Only print Hausdorff if it's available (not NaN)
    if not df['hausdorff'].isna().all():
        print(f"Mean Hausdorff: {df['hausdorff'].mean():.2f} ± {df['hausdorff'].std():.2f} px")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing prediction masks")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth masks")
    parser.add_argument("--output", type=str, default="data/evaluation/metrics.csv", help="Path to save results CSV")
    
    args = parser.parse_args()
    
    evaluate_dataset(args.pred_dir, args.gt_dir, args.output)
