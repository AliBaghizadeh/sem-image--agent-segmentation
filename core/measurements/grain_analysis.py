
"""
Grain measurement and morphology analysis tools for materials science.
Calculates properties like area, diameter, perimeter, and aspect ratio from segmentation masks.

Example usage:
    analyzer = GrainAnalyzer(pixel_to_micron_ratio=0.1)
    df = analyzer.analyze_masks(masks)
"""

import numpy as np
import cv2
import pandas as pd
from skimage import measure

class GrainAnalyzer:
    def __init__(self, pixel_to_micron_ratio=1.0):
        self.ratio = pixel_to_micron_ratio

    def analyze_masks(self, masks):
        """
        Analyze a list of MatSAM masks (dictionaries with 'segmentation', 'area', etc.).
        """
        results = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation'].astype(np.uint8)
            
            # Calculate properties
            props = measure.regionprops(mask)[0]
            
            area_um2 = props.area * (self.ratio ** 2)
            equiv_diameter = props.equivalent_diameter * self.ratio
            perimeter = props.perimeter * self.ratio
            aspect_ratio = props.major_axis_length / (props.minor_axis_length + 1e-6)
            
            results.append({
                'id': i,
                'area_um2': area_um2,
                'diameter_um': equiv_diameter,
                'perimeter_um': perimeter,
                'aspect_ratio': aspect_ratio,
                'centroid_y': props.centroid[0],
                'centroid_x': props.centroid[1],
                'bbox': props.bbox
            })
            
        return pd.DataFrame(results)

    def filter_grains(self, df, query):
        """
        Dynamic filtering of grains based on a natural language query (simplified for now).
        Actually, we can use pandas query for this.
        """
        # Example query: "area_um2 > 50 and aspect_ratio < 1.5"
        try:
            filtered_df = df.query(query)
            return filtered_df
        except Exception as e:
            print(f"Error filtering grains: {e}")
            return df
