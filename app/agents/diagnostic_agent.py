"""
Diagnostic Agent for SEM Segmentation Quality Analysis

This agent analyzes segmentation quality and suggests rescue parameters
when the initial segmentation fails or produces poor results.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional

class SegmentationDiagnosticAgent:
    """
    AI agent that diagnoses segmentation failures and recommends solutions.
    
    The agent analyzes quality metrics and suggests optimal preprocessing
    parameters from the rescue workflow.
    """
    
    def __init__(self):
        """Initialize the diagnostic agent with quality thresholds."""
        self.quality_thresholds = {
            'grain_count_min': 10,
            'grain_count_max': 1000,
            'coverage_min': 0.05,
            'coverage_max': 0.95,
            'jaggedness_max': 40,
            'quality_score_min': 0.6
        }
        
        # Rescue parameter presets based on failure modes
        self.rescue_presets = {
            'low_contrast': {
                'blend': 1.1,
                'clip': 4.0,
                'sigma_small': 5.0,
                'sigma_large': 15.0,
                'scale': 0.2,
                'reason': 'Low contrast detected - enhancing boundaries'
            },
            'high_noise': {
                'blend': 0.9,
                'clip': 10.0,
                'sigma_small': 7.0,
                'sigma_large': 12.0,
                'scale': 0.4,
                'reason': 'High noise detected - applying stronger filtering'
            },
            'over_segmentation': {
                'blend': 0.9,
                'clip': 4.0,
                'sigma_small': 6.0,
                'sigma_large': 15.0,
                'scale': 0.4,
                'reason': 'Over-segmentation detected - using larger scales'
            },
            'under_segmentation': {
                'blend': 1.1,
                'clip': 4.0,
                'sigma_small': 1.0,
                'sigma_large': 12.0,
                'scale': 0.2,
                'reason': 'Under-segmentation detected - using finer scales'
            }
        }
    
    def analyze_quality(self, mask: np.ndarray) -> Dict:
        """
        Analyze segmentation quality and compute metrics.
        
        Args:
            mask: Binary segmentation mask
        
        Returns:
            Dictionary with quality metrics and analysis
        """
        # Calculate metrics
        num_labels = cv2.connectedComponents(mask)[0] - 1
        coverage = np.sum(mask > 0) / mask.size
        
        # Boundary smoothness
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
        total_area = np.sum(mask > 0)
        jaggedness = total_perimeter / (np.sqrt(total_area) + 1e-8) if total_area > 0 else 0
        
        # Quality score
        grain_score = 1.0
        if num_labels < self.quality_thresholds['grain_count_min']:
            grain_score = num_labels / self.quality_thresholds['grain_count_min']
        elif num_labels > self.quality_thresholds['grain_count_max']:
            grain_score = max(0, 1.0 - (num_labels - self.quality_thresholds['grain_count_max']) / 1000.0)
        
        coverage_score = 1.0 if 0.05 <= coverage <= 0.95 else 0.5
        smoothness_score = max(0, 1.0 - jaggedness / 50.0)
        
        quality_score = 0.4 * grain_score + 0.3 * coverage_score + 0.3 * smoothness_score
        
        return {
            'grain_count': num_labels,
            'coverage': coverage * 100,
            'jaggedness': jaggedness,
            'quality_score': quality_score,
            'is_good': quality_score >= self.quality_thresholds['quality_score_min']
        }
    
    def diagnose_failure(self, metrics: Dict) -> Tuple[str, str]:
        """
        Diagnose the type of segmentation failure.
        
        Args:
            metrics: Quality metrics from analyze_quality()
        
        Returns:
            Tuple of (failure_type, diagnosis_message)
        """
        grain_count = metrics['grain_count']
        coverage = metrics['coverage'] / 100
        jaggedness = metrics['jaggedness']
        
        # Diagnose failure mode
        if grain_count < self.quality_thresholds['grain_count_min']:
            failure_type = 'under_segmentation'
            message = f"⚠️ **Under-segmentation detected**: Only {grain_count} grains found. The model may be missing fine grain boundaries."
        
        elif grain_count > self.quality_thresholds['grain_count_max']:
            failure_type = 'over_segmentation'
            message = f"⚠️ **Over-segmentation detected**: {grain_count} grains found (likely noise). The model is detecting too many false boundaries."
        
        elif coverage < self.quality_thresholds['coverage_min']:
            failure_type = 'low_contrast'
            message = f"⚠️ **Low coverage detected**: Only {coverage*100:.1f}% of image segmented. The image may have low contrast."
        
        elif coverage > self.quality_thresholds['coverage_max']:
            failure_type = 'low_contrast'
            message = f"⚠️ **High coverage detected**: {coverage*100:.1f}% of image segmented. Boundaries may be unclear."
        
        elif jaggedness > self.quality_thresholds['jaggedness_max']:
            failure_type = 'high_noise'
            message = f"⚠️ **Noisy boundaries detected**: Jaggedness score {jaggedness:.1f}. The segmentation has irregular edges."
        
        else:
            failure_type = 'low_contrast'
            message = "⚠️ **Quality below threshold**: The segmentation quality is suboptimal."
        
        return failure_type, message
    
    def suggest_rescue_parameters(self, failure_type: str) -> Dict:
        """
        Suggest optimal rescue parameters based on failure type.
        
        Args:
            failure_type: Type of failure from diagnose_failure()
        
        Returns:
            Dictionary with suggested parameters and explanation
        """
        preset = self.rescue_presets.get(failure_type, self.rescue_presets['low_contrast'])
        
        suggestion = {
            'parameters': {
                'blend': preset['blend'],
                'clip': preset['clip'],
                'sigma_small': preset['sigma_small'],
                'sigma_large': preset['sigma_large'],
                'scale': preset['scale']
            },
            'reason': preset['reason'],
            'explanation': self._generate_explanation(preset)
        }
        
        return suggestion
    
    def _generate_explanation(self, preset: Dict) -> str:
        """Generate human-readable explanation of parameters."""
        explanation = f"""
**Recommended Rescue Parameters:**

- **Blend (B)**: {preset['blend']} - {'Enhance boundaries' if preset['blend'] > 1.0 else 'Reduce noise'}
- **CLAHE Clip**: {preset['clip']} - {'Strong' if preset['clip'] > 5 else 'Moderate'} contrast enhancement
- **DoG σ_small**: {preset['sigma_small']} - {'Fine' if preset['sigma_small'] < 3 else 'Coarse'} edge detection
- **DoG σ_large**: {preset['sigma_large']} - Background suppression
- **Frangi Scale**: {preset['scale']} - {'Fine' if preset['scale'] < 0.3 else 'Thick'} boundary detection

**Why these parameters?**
{preset['reason']}
        """
        return explanation.strip()
    
    def generate_report(self, metrics: Dict, failure_type: Optional[str] = None) -> str:
        """
        Generate a comprehensive diagnostic report.
        
        Args:
            metrics: Quality metrics
            failure_type: Optional failure type
        
        Returns:
            Formatted diagnostic report
        """
        if metrics['is_good']:
            report = f"""
✅ **Segmentation Quality: GOOD**

**Metrics:**
- Grain Count: {metrics['grain_count']}
- Coverage: {metrics['coverage']:.1f}%
- Quality Score: {metrics['quality_score']:.2f}/1.00

The segmentation looks good! No rescue workflow needed.
            """
        else:
            failure_type, diagnosis = self.diagnose_failure(metrics)
            suggestion = self.suggest_rescue_parameters(failure_type)
            
            report = f"""
{diagnosis}

**Current Metrics:**
- Grain Count: {metrics['grain_count']}
- Coverage: {metrics['coverage']:.1f}%
- Boundary Smoothness: {'Smooth' if metrics['jaggedness'] < 30 else 'Jagged'}
- Quality Score: {metrics['quality_score']:.2f}/1.00

---

🔧 **Agent Recommendation:**

{suggestion['explanation']}

**Click "Apply Rescue Workflow" below to automatically enhance the image with these parameters.**
            """
        
        return report.strip()
