"""
Shared preprocessing presets for different SEM image classes.
"""

from __future__ import annotations

from typing import Dict


PIPELINE_PRESETS: Dict[str, Dict[str, object]] = {
    "lines": {
        "use_frangi": True,
        "frangi_scales": (0.5, 1.0, 2.0),
        "frangi_blend_alpha": 0.2,
        "w_frangi": 0.4,
        "use_dog": True,
        "dog_sigma_small": 1.0,
        "dog_sigma_large": 4.0,
        "w_dog": 1.0,
        "use_clahe": True,
        "clahe_clip": 25.0,
        "clahe_tile": 4,
        "w_clahe": 1.0,
        "use_unsharp": False,
        "unsharp_amount": 0.3,
        "unsharp_radius": 2.0,
        "w_unsharp": 0.0,
        "dirt_threshold": 0.2,
        "dirt_min_size": 10,
        "dirt_max_size": 35,
        "dirt_aspect_ratio_thresh": 1.3,
        "smooth_sigma": 2,
        "blend_alpha_fused": 0.6,
    },
    "boundaries": {
        "use_frangi": True,
        "frangi_scales": (0.3, 0.7, 1.5),
        "frangi_blend_alpha": 0.2,
        "w_frangi": 0.2,
        "use_dog": False,
        "dog_sigma_small": 1.0,
        "dog_sigma_large": 4.0,
        "w_dog": 0.0,
        "use_clahe": True,
        "clahe_clip": 20.0,
        "clahe_tile": 4,
        "w_clahe": 1.0,
        "use_unsharp": False,
        "unsharp_amount": 0.3,
        "unsharp_radius": 2.0,
        "w_unsharp": 0.0,
        "dirt_threshold": 0.1,
        "dirt_min_size": 5,
        "dirt_max_size": 25,
        "dirt_aspect_ratio_thresh": 1.2,
        "smooth_sigma": 2,
        "blend_alpha_fused": 0.5,
    },
}

PREPROCESSOR_PRESETS: Dict[str, Dict[str, object]] = {
    "lines": {
        "clahe_clip_limit": 3.0,
        "clahe_tile_size": 16,
        "bilateral_d": 4,
        "bilateral_sigma_color": 30,
        "bilateral_sigma_space": 30,
    },
    "boundaries": {
        "clahe_clip_limit": 2.0,
        "clahe_tile_size": 16,
        "bilateral_d": 4,
        "bilateral_sigma_color": 30,
        "bilateral_sigma_space": 30,
    },
}

DEFAULT_PRESET = "lines"
