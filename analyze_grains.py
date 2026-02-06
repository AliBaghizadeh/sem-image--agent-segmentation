"""
Grain Statistics Analysis Script

Extracts comprehensive microstructural features from binary segmentation masks:
- Grain count and size distributions
- Boundary metrics (length, density)
- Spatial statistics (nearest-neighbor distances)
- Morphological features (aspect ratio, circularity)

Usage:
    python analyze_grains.py --mask_dir "data/masks" --output_csv "data/grain_stats.csv" --pixel_size 0.5

Arguments:
    --mask_dir: Directory containing binary mask images (.png)
    --output_csv: Path to save the results CSV file
    --pixel_size: Physical size of one pixel in micrometers (default: 1.0)
    --min_area: Minimum grain area in pixels to include (default: 50)

Output CSV columns:
    - image_name: Filename of the mask
    - grain_count: Total number of grains detected
    - mean_area_um2: Mean grain area in μm²
    - median_area_um2: Median grain area in μm²
    - std_area_um2: Standard deviation of grain areas
    - mean_diameter_um: Mean equivalent diameter in μm
    - mean_aspect_ratio: Mean aspect ratio (major/minor axis)
    - mean_circularity: Mean circularity (4π·Area/Perimeter²)
    - total_boundary_length_um: Total grain boundary length in μm
    - boundary_density_um_per_um2: Boundary length per unit area
    - coverage_fraction: Fraction of image covered by grains
    - mean_nearest_neighbor_um: Mean distance to nearest neighboring grain
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from core.matsam.matsam_model import MatSAMModel
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
from scipy.spatial import distance_matrix


def extract_grain_features(mask, pixel_size=1.0, min_area=50):
    """
    Extract grain statistics from a binary mask.
    
    Args:
        mask (np.ndarray): Binary mask (uint8, 0=boundary, 255=grain)
        pixel_size (float): Physical size of one pixel in micrometers
        min_area (int): Minimum grain area in pixels to include
    
    Returns:
        dict: Dictionary containing grain statistics
    """
    # Find connected components (grains)
    num_grains, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    # Filter out background (label 0) and small grains
    valid_grains = []
    for i in range(1, num_grains):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_grains.append(i)
    
    if len(valid_grains) == 0:
        return {
            'grain_count': 0,
            'mean_area_um2': 0,
            'median_area_um2': 0,
            'std_area_um2': 0,
            'mean_diameter_um': 0,
            'mean_aspect_ratio': 0,
            'mean_circularity': 0,
            'total_boundary_length_um': 0,
            'boundary_density_um_per_um2': 0,
            'coverage_fraction': 0,
            'mean_nearest_neighbor_um': 0
        }
    
    # Extract features for each grain
    areas = []
    diameters = []
    aspect_ratios = []
    circularities = []
    grain_centroids = []
    
    for grain_id in valid_grains:
        # Create single-grain mask
        grain_mask = (labels == grain_id).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        
        contour = contours[0]
        
        # Area
        area_px = cv2.contourArea(contour)
        area_um2 = area_px * (pixel_size ** 2)
        areas.append(area_um2)
        
        # Equivalent diameter
        diameter_um = 2 * np.sqrt(area_um2 / np.pi)
        diameters.append(diameter_um)
        
        # Aspect ratio (from fitted ellipse)
        if len(contour) >= 5:  # Need at least 5 points to fit ellipse
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            aspect_ratio = major_axis / (minor_axis + 1e-8)
            aspect_ratios.append(aspect_ratio)
        else:
            aspect_ratios.append(1.0)
        
        # Circularity (4π·Area/Perimeter²)
        perimeter_px = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area_px) / (perimeter_px ** 2 + 1e-8)
        circularities.append(circularity)
        
        # Centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            grain_centroids.append([cx, cy])
    
    # Boundary metrics
    # Find all grain boundaries (edges between grains)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundaries = mask - eroded
    boundary_length_px = np.sum(boundaries > 0)
    boundary_length_um = boundary_length_px * pixel_size
    
    # Image area
    image_area_um2 = mask.shape[0] * mask.shape[1] * (pixel_size ** 2)
    boundary_density = boundary_length_um / image_area_um2
    
    # Coverage fraction
    grain_area_px = np.sum(mask > 0)
    coverage_fraction = grain_area_px / (mask.shape[0] * mask.shape[1])
    
    # Nearest neighbor distances
    mean_nn_distance = 0
    if len(grain_centroids) > 1:
        centroids_array = np.array(grain_centroids)
        dist_matrix = distance_matrix(centroids_array, centroids_array)
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(dist_matrix, np.inf)
        # Find minimum distance for each grain
        min_distances = np.min(dist_matrix, axis=1)
        mean_nn_distance = np.mean(min_distances) * pixel_size
    
    return {
        'grain_count': len(valid_grains),
        'mean_area_um2': np.mean(areas) if areas else 0,
        'median_area_um2': np.median(areas) if areas else 0,
        'std_area_um2': np.std(areas) if areas else 0,
        'mean_diameter_um': np.mean(diameters) if diameters else 0,
        'mean_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 0,
        'mean_circularity': np.mean(circularities) if circularities else 0,
        'total_boundary_length_um': boundary_length_um,
        'boundary_density_um_per_um2': boundary_density,
        'coverage_fraction': coverage_fraction,
        'mean_nearest_neighbor_um': mean_nn_distance
    }


def analyze_dataset(mask_dir, output_csv, pixel_size=1.0, min_area=50):
    """
    Analyze all masks in a directory and save statistics to CSV.
    
    Args:
        mask_dir (str): Directory containing binary mask images
        output_csv (str): Path to save the results CSV
        pixel_size (float): Physical size of one pixel in micrometers
        min_area (int): Minimum grain area in pixels to include
    """
    mask_dir = Path(mask_dir)
    
    if not mask_dir.exists():
        print(f"Error: Mask directory {mask_dir} not found.")
        return
    
    results = []
    
    print(f"Analyzing masks in {mask_dir}...")
    print(f"Pixel size: {pixel_size} um/pixel")
    print(f"Minimum grain area: {min_area} pixels")
    
    # Find all mask files
    mask_files = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.tif"))
    
    for mask_path in tqdm(mask_files, desc="Processing masks"):
        # Skip overlay/comparison images
        if any(x in mask_path.name.lower() for x in ["overlay", "compare", "comparison"]):
            continue
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Could not read {mask_path.name}")
            continue
        
        # Binarize (in case it's not already binary)
        mask = (mask > 127).astype(np.uint8) * 255
        
        # Extract features
        features = extract_grain_features(mask, pixel_size, min_area)
        features['image_name'] = mask_path.stem
        results.append(features)
    
    if not results:
        print("No valid masks found.")
        return
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'image_name', 'grain_count', 'mean_area_um2', 'median_area_um2', 
        'std_area_um2', 'mean_diameter_um', 'mean_aspect_ratio', 
        'mean_circularity', 'total_boundary_length_um', 
        'boundary_density_um_per_um2', 'coverage_fraction', 
        'mean_nearest_neighbor_um'
    ]
    df = df[column_order]
    
    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Analysis Complete - {len(df)} images processed")
    print(f"{'='*60}")
    print(f"Results saved to: {output_csv}")
    print(f"\n--- Summary Statistics ---")
    print(f"Total grains analyzed: {df['grain_count'].sum():.0f}")
    print(f"Mean grain count per image: {df['grain_count'].mean():.1f} ± {df['grain_count'].std():.1f}")
    print(f"Mean grain area: {df['mean_area_um2'].mean():.2f} ± {df['mean_area_um2'].std():.2f} um^2")
    print(f"Mean grain diameter: {df['mean_diameter_um'].mean():.2f} ± {df['mean_diameter_um'].std():.2f} um")
    print(f"Mean aspect ratio: {df['mean_aspect_ratio'].mean():.3f} ± {df['mean_aspect_ratio'].std():.3f}")
    print(f"Mean circularity: {df['mean_circularity'].mean():.3f} ± {df['mean_circularity'].std():.3f}")
    print(f"Mean coverage fraction: {df['coverage_fraction'].mean():.3f} ± {df['coverage_fraction'].std():.3f}")
    print(f"Mean boundary density: {df['boundary_density_um_per_um2'].mean():.4f} ± {df['boundary_density_um_per_um2'].std():.4f} um/um^2")
    print(f"{'='*60}\n")
    
    return df


def plot_distributions(csv_path, output_dir="data/plots"):
    """
    Generate publication-quality distribution plots from grain statistics CSV.
    
    Args:
        csv_path (str): Path to the grain statistics CSV file
        output_dir (str): Directory to save the plots
    """
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Grain Microstructure Analysis', fontsize=16, fontweight='bold')
    
    # 1. Grain Size Distribution (Diameter)
    ax1 = axes[0, 0]
    ax1.hist(df['mean_diameter_um'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Grain Diameter (μm)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('(a) Grain Size Distribution')
    ax1.axvline(df['mean_diameter_um'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["mean_diameter_um"].mean():.1f} μm')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Grain Area Distribution
    ax2 = axes[0, 1]
    ax2.hist(df['mean_area_um2'], bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Grain Area (μm²)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('(b) Grain Area Distribution')
    ax2.axvline(df['mean_area_um2'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["mean_area_um2"].mean():.0f} μm²')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Aspect Ratio Distribution
    ax3 = axes[1, 0]
    ax3.hist(df['mean_aspect_ratio'], bins=30, edgecolor='black', alpha=0.7, color='mediumseagreen')
    ax3.set_xlabel('Aspect Ratio', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('(c) Grain Aspect Ratio Distribution')
    ax3.axvline(df['mean_aspect_ratio'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["mean_aspect_ratio"].mean():.2f}')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Circularity Distribution
    ax4 = axes[1, 1]
    ax4.hist(df['mean_circularity'], bins=30, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax4.set_xlabel('Circularity', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('(d) Grain Circularity Distribution')
    ax4.axvline(df['mean_circularity'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["mean_circularity"].mean():.2f}')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_path / "grain_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Distribution plots saved to: {plot_path}")
    
    # Create cumulative distribution plot for grain size
    fig2, ax = plt.subplots(figsize=(8, 6))
    sorted_diameters = np.sort(df['mean_diameter_um'])
    cumulative = np.arange(1, len(sorted_diameters) + 1) / len(sorted_diameters) * 100
    
    ax.plot(sorted_diameters, cumulative, linewidth=2, color='steelblue')
    ax.set_xlabel('Grain Diameter (μm)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cumulative Frequency (%)', fontweight='bold', fontsize=12)
    ax.set_title('Cumulative Grain Size Distribution', fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Median')
    ax.axvline(np.median(sorted_diameters), color='red', linestyle='--', alpha=0.5)
    ax.legend()
    
    cumulative_path = output_path / "grain_cumulative_distribution.png"
    plt.savefig(cumulative_path, dpi=300, bbox_inches='tight')
    print(f"📊 Cumulative distribution plot saved to: {cumulative_path}")
    
    plt.close('all')
    print(f"\n✅ All plots generated successfully!")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract grain statistics from binary segmentation masks"
    )
    parser.add_argument(
        "--mask_dir", 
        type=str, 
        required=True, 
        help="Directory containing binary mask images (.png or .tif)"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="data/grain_statistics.csv", 
        help="Path to save the results CSV file"
    )
    parser.add_argument(
        "--pixel_size", 
        type=float, 
        default=1.0, 
        help="Physical size of one pixel in micrometers (default: 1.0)"
    )
    parser.add_argument(
        "--min_area", 
        type=int, 
        default=50, 
        help="Minimum grain area in pixels to include (default: 50)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate distribution plots (saved to data/plots/)"
    )
    
    args = parser.parse_args()
    
    # Analyze dataset
    df = analyze_dataset(
        mask_dir=args.mask_dir,
        output_csv=args.output_csv,
        pixel_size=args.pixel_size,
        min_area=args.min_area
    )
    
    # Generate plots if requested
    if args.plot and df is not None:
        plot_distributions(args.output_csv)
