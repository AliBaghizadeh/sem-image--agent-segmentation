"""
Preprocessing pipeline utilities for SEM line enhancement workflows.

This module exposes :class:`SEMPreprocessor`, a std-aware preprocessing
pipeline combining Frangi/DoG enhancements, dirt removal, CLAHE, and bilateral
filtering. It mirrors the behavior used throughout the original scripts.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from .enhancers import enhance_lines_frangi, enhance_lines_modular, filter_dirt_blobs


class SEMPreprocessor:
    """
    Std-aware preprocessing pipeline for SEM images with artifacts.

    Each method exposes a concept demonstrated in the educational notebooks:
    local standard deviation detection, selective CLAHE, inpainting, and dual
    output generation (lines/base/fused).

    Parameters
    ----------
    prefer_gpu : bool
        Whether to run GPU-friendly kernels (CuPy/OpenCL) when available.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 3.0,
        clahe_tile_size: int = 16,
        bilateral_d: int = 7,
        bilateral_sigma_color: float = 75,
        bilateral_sigma_space: float = 75,
        std_threshold: float = 8.0,
        std_window_size: int = 20,
        prefer_gpu: bool = True,
    ) -> None:
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.std_threshold = std_threshold
        self.std_window_size = std_window_size
        self.prefer_gpu = prefer_gpu

    # --- Helper maps -------------------------------------------------
    def compute_local_std(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the local standard deviation map for artifact detection.
        """
        img = image.astype(np.float64)
        window_size = self.std_window_size
        kernel = np.ones((window_size, window_size), dtype=np.float64) / (
            window_size**2
        )
        local_mean = cv2.filter2D(img, -1, kernel)
        local_mean_sq = cv2.filter2D(img**2, -1, kernel)
        local_var = np.maximum(local_mean_sq - local_mean**2, 0)
        return np.sqrt(local_var).astype(np.float32)

    def create_artifact_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a binary mask of high-variance regions and return it alongside the std map.
        """
        std_map = self.compute_local_std(image)
        artifact_mask = (std_map > self.std_threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        artifact_mask = cv2.morphologyEx(artifact_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        artifact_mask = cv2.morphologyEx(artifact_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        artifact_mask = cv2.dilate(artifact_mask, kernel, iterations=1)
        return artifact_mask, std_map

    def inpaint_artifacts(self, image: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
        """
        Telea-inpaint artifact regions to prevent dirt from influencing later filters.
        """
        image_uint8 = (image * 255).astype(np.uint8)
        inpainted = cv2.inpaint(
            image_uint8,
            artifact_mask,
            inpaintRadius=7,
            flags=cv2.INPAINT_TELEA,
        )
        return inpainted.astype(np.float32) / 255.0

    def selective_clahe(self, image: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE only where the artifact mask is zero, preserving noisy regions.
        """
        image_uint8 = (image * 255).astype(np.uint8)
        enhanced = self.clahe.apply(image_uint8)
        clean_mask = cv2.bitwise_not(artifact_mask)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=clean_mask)
        original_masked = cv2.bitwise_and(image_uint8, image_uint8, mask=artifact_mask)
        enhanced = cv2.add(enhanced, original_masked)
        return enhanced.astype(np.float32) / 255.0

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Standard CLAHE operation used when artifact masking is disabled.
        """
        image_uint8 = (image * 255).astype(np.uint8)
        enhanced = self.clahe.apply(image_uint8)
        return enhanced.astype(np.float32) / 255.0

    def denoise_bilateral(self, image: np.ndarray) -> np.ndarray:
        """
        Bilateral filter helper shared by both base and fused paths.
        """
        image_uint8 = (image * 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(
            image_uint8,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
        )
        return denoised.astype(np.float32) / 255.0

    def normalize_intensity(self, image: np.ndarray, method: str = "percentile") -> np.ndarray:
        """
        Normalize intensities via percentile clipping (default) or min/max scaling.
        """
        if method == "percentile":
            p1 = np.percentile(image, 1)
            p99 = np.percentile(image, 99)
            image_clipped = np.clip(image, p1, p99)
            normalized = (image_clipped - p1) / (p99 - p1 + 1e-8)
        elif method == "minmax":
            min_val = np.min(image)
            max_val = np.max(image)
            normalized = (image - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        return normalized

    # --- Pipelines ---------------------------------------------------
    def preprocess_step1_debug(
        self,
        image: np.ndarray,
        frangi_scales=(0.5, 1.0, 2.0),
        frangi_blend_alpha: float = 0.3,
        dirt_threshold: float = 0.15,
        dirt_min_size: int = 10,
        dirt_max_size: int = 30,
        dirt_aspect_ratio_thresh: float = 1.3,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Educational step-by-step pipeline used for visualizing each intermediate output.
        """
        intermediates: Dict[str, np.ndarray] = {}
        img = image.copy()
        intermediates["00_original_input"] = img.copy()

        if frangi_scales is not None and frangi_blend_alpha > 0:
            img = enhance_lines_frangi(
                img,
                scales=frangi_scales,
                blend_alpha=frangi_blend_alpha,
                prefer_gpu=self.prefer_gpu,
            )
        intermediates["01_frangi_enhanced"] = img.copy()

        img = filter_dirt_blobs(
            img,
            threshold=dirt_threshold,
            min_size=dirt_min_size,
            max_size=dirt_max_size,
            aspect_ratio_thresh=dirt_aspect_ratio_thresh,
            inpaint_radius=9,
            inpaint_method="telea",
            prefer_gpu=self.prefer_gpu,
        )
        intermediates["02_dirt_filtered"] = img.copy()

        std_map = self.compute_local_std(img)
        intermediates["03_std_map"] = std_map.copy()
        artifact_mask, _ = self.create_artifact_mask(img)
        intermediates["04_artifact_mask"] = artifact_mask.copy()

        inpainted = self.inpaint_artifacts(img, artifact_mask)
        intermediates["05_inpainted"] = inpainted.copy()

        return inpainted, intermediates

    def preprocess(
        self,
        image: np.ndarray,
        save_intermediates: bool = False,
        use_std_masking: bool = False,
        line_enhancement: bool = True,
        dirt_filtering: bool = True,
        frangi_scales=(0.5, 1.0, 2.0),
        frangi_blend_alpha: float = 0.3,
        dirt_threshold: float = 0.15,
        dirt_min_size: int = 10,
        dirt_max_size: int = 30,
        dirt_aspect_ratio_thresh: float = 1.3,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Single-path preprocessing used for legacy workflows (with optional std masking).
        """
        intermediates: Dict[str, np.ndarray] = {}
        img = image.copy()
        if save_intermediates:
            intermediates["00_original_input"] = img.copy()

        if line_enhancement:
            img = enhance_lines_frangi(
                img,
                scales=frangi_scales,
                blend_alpha=frangi_blend_alpha,
                prefer_gpu=self.prefer_gpu,
            )
            if save_intermediates:
                intermediates["01_frangi_enhanced"] = img.copy()

        if dirt_filtering:
            img = filter_dirt_blobs(
                img,
                threshold=dirt_threshold,
                min_size=dirt_min_size,
                max_size=dirt_max_size,
                aspect_ratio_thresh=dirt_aspect_ratio_thresh,
                prefer_gpu=self.prefer_gpu,
            )
            if save_intermediates:
                intermediates["02_dirt_filtered"] = img.copy()

        if use_std_masking:
            artifact_mask, std_map = self.create_artifact_mask(img)
            if save_intermediates:
                intermediates["04_std_map"] = std_map.copy()
                intermediates["05_artifact_mask"] = artifact_mask.copy()
            cleaned = self.inpaint_artifacts(img, artifact_mask)
            if save_intermediates:
                intermediates["06_inpainted"] = cleaned.copy()
            enhanced = self.selective_clahe(cleaned, artifact_mask)
        else:
            enhanced = self.enhance_contrast(img)
        if save_intermediates:
            intermediates["07_clahe"] = enhanced.copy()

        denoised = self.denoise_bilateral(enhanced)
        if save_intermediates:
            intermediates["08_bilateral"] = denoised.copy()

        normalized = self.normalize_intensity(denoised, method="percentile")
        if save_intermediates:
            intermediates["09_normalized"] = normalized.copy()
        return normalized, intermediates

    def preprocess_dual(
        self,
        image: np.ndarray,
        use_frangi: bool = True,
        frangi_scales=(0.5, 1.0, 2.0),
        frangi_blend_alpha: float = 0.3,
        w_frangi: float = 0.4,
        use_dog: bool = True,
        dog_sigma_small: float = 1.0,
        dog_sigma_large: float = 4.0,
        w_dog: float = 1.0,
        use_clahe: bool = False,
        clahe_clip: float = 2.0,
        clahe_tile: int = 16,
        w_clahe: float = 0.0,
        use_unsharp: bool = False,
        unsharp_amount: float = 0.3,
        unsharp_radius: float = 2.0,
        w_unsharp: float = 0.0,
        dirt_threshold: float = 0.15,
        dirt_min_size: int = 10,
        dirt_max_size: int = 40,
        dirt_aspect_ratio_thresh: float = 1.3,
        smooth_sigma: int = 2,
        blend_alpha_fused: float = 0.6,
        save_intermediates: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Dual-path preprocessing that returns line, base, and fused variants.
        """
        intermediates: Dict[str, np.ndarray] = {}
        img = image.copy()

        i_lines = enhance_lines_modular(
            img,
            use_frangi=use_frangi,
            frangi_scales=frangi_scales,
            frangi_alpha=frangi_blend_alpha,
            w_frangi=w_frangi,
            use_dog=use_dog,
            dog_sigma_small=dog_sigma_small,
            dog_sigma_large=dog_sigma_large,
            w_dog=w_dog,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            clahe_tile=clahe_tile,
            w_clahe=w_clahe,
            use_unsharp=use_unsharp,
            unsharp_amount=unsharp_amount,
            unsharp_radius=unsharp_radius,
            w_unsharp=w_unsharp,
            prefer_gpu=self.prefer_gpu,
        )

        i_lines = filter_dirt_blobs(
            i_lines,
            threshold=dirt_threshold,
            min_size=dirt_min_size,
            max_size=dirt_max_size,
            aspect_ratio_thresh=dirt_aspect_ratio_thresh,
            prefer_gpu=self.prefer_gpu,
        )
        artifact_mask, _ = self.create_artifact_mask(i_lines)
        i_lines = self.inpaint_artifacts(i_lines, artifact_mask)
        i_lines = self.normalize_intensity(i_lines, method="percentile")

        i_base = cv2.GaussianBlur(img, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
        i_base = self.enhance_contrast(i_base)
        i_base = self.denoise_bilateral(i_base)
        i_base = self.normalize_intensity(i_base, method="percentile")

        i_fused = np.clip(
            blend_alpha_fused * i_lines + (1 - blend_alpha_fused) * i_base,
            0,
            1,
        )

        if save_intermediates:
            intermediates["i_lines"] = i_lines
            intermediates["i_base"] = i_base
            intermediates["i_fused"] = i_fused

        return i_lines, i_base, i_fused, intermediates
