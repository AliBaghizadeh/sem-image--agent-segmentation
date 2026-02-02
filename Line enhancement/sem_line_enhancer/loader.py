"""
Lightweight SEM image loader utilities.

This module exposes :class:`SEMImageLoader`, the canonical way to scan
directories of microscopy images while standardizing their dtype, color space,
and normalization into float32 [0, 1] arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple
import warnings

import cv2
import numpy as np
import tifffile


class SEMImageLoader:
    """
    Robust image loader for SEM microscopy images in multiple formats.

    Parameters
    ----------
    input_dir:
        Directory containing raw SEM images.
    supported_formats:
        Iterable of file extensions to include (defaults to common TIFF/JPEG/PNG).
    """

    def __init__(
        self,
        input_dir: Path | str,
        supported_formats: Sequence[str] | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.supported_formats: Tuple[str, ...] = tuple(
            supported_formats or (".tif", ".tiff", ".jpg", ".jpeg", ".png")
        )

        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

    def load_image(self, image_path: Path | str) -> tuple[np.ndarray, dict]:
        """
        Load image from any supported format with optional metadata.
        """

        path = Path(image_path)
        extension = path.suffix.lower()

        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported format: {extension}")

        metadata = {
            "filename": path.name,
            "format": extension,
            "path": str(path),
        }

        if extension in [".tif", ".tiff"]:
            image = tifffile.imread(str(path))
            metadata["original_dtype"] = str(image.dtype)
            metadata["bit_depth"] = 16 if image.dtype == np.uint16 else 8
        else:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise IOError(f"Failed to load image: {path}")
            metadata["original_dtype"] = str(image.dtype)
            metadata["bit_depth"] = 16 if image.dtype == np.uint16 else 8

        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                warnings.warn(f"Converted RGB to grayscale: {path.name}")
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                warnings.warn(f"Converted RGBA to grayscale: {path.name}")

        metadata["original_shape"] = image.shape
        metadata["shape"] = image.shape

        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = np.clip(image, 0, 1).astype(np.float32)

        return image, metadata

    def scan_directory(self) -> List[Path]:
        """
        Recursively scan input directory for supported images.
        """

        image_paths: list[Path] = []
        for ext in self.supported_formats:
            image_paths.extend(self.input_dir.rglob(f"*{ext}"))

        image_paths = sorted(image_paths)
        print(f"[LOADER] Found {len(image_paths)} images in {self.input_dir}")
        return image_paths
