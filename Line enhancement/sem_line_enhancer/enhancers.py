"""Educational SEM enhancement primitives used throughout the toolkit.

The functions below match the steps taught in the curriculum:

1. ``enhance_lines_frangi`` boosts faint ferroelastic ridges via multi-scale Hessian analysis.
2. ``filter_dirt_blobs`` removes round bright contaminants using region measurements and OpenCV inpainting.
3. ``enhance_lines_modular`` blends Frangi, DoG, CLAHE, and unsharp components with user-controlled weights.
4. ``make_prompt_map_from_lines`` converts the line-confidence map into prompt coordinates for downstream segmenters.

Each function carries a detailed docstring so learners can experiment with the same parameters showcased in the app."""

import numpy as np
import cv2
from skimage.filters import frangi
from skimage.measure import label, regionprops
from joblib import Parallel, delayed
import os, inspect
from skimage.morphology import skeletonize
import warnings

print("[TRACE] filter_dirt_blobs loaded from:", inspect.getfile(inspect.currentframe()))
print("[TRACE] Current working dir:", os.getcwd())
from concurrent.futures import ThreadPoolExecutor

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter

    HAS_CUPY = True
except Exception:  # pragma: no cover - GPU optional
    HAS_CUPY = False


def _gaussian_blur(image: np.ndarray, sigma: float, prefer_gpu: bool = True) -> np.ndarray:
    """
    Gaussian blur helper that uses CuPy when available, otherwise OpenCV CPU.
    """
    global HAS_CUPY
    if prefer_gpu and HAS_CUPY:
        try:
            arr = cp.asarray(image, dtype=cp.float32)
            blurred = cp_gaussian_filter(arr, sigma=sigma)
            return cp.asnumpy(blurred)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"CuPy gaussian filter failed ({exc}); falling back to CPU.")
            HAS_CUPY = False
    return cv2.GaussianBlur(image, (0, 0), sigma)

def enhance_lines_frangi(
    img,
    scales=(0.5, 1.0, 2.0, 4.0, 6.0, 8.0),
    blend_alpha=0.5,
    preblur_sigma=0.4,
    percentile_norm=(1, 99.8),
    sharpen_gain=1.4,
    prefer_gpu: bool = True,
):
    """
    High-performance multi-scale Frangi line enhancement.

    Accelerations
    -------------
    - Multi-threaded Frangi evaluation (ThreadPoolExecutor)
    - CuPy gaussian pre-blur when available (falls back to OpenCV CPU)
    - OpenCV OpenCL acceleration (cv2.UMat)
    - Vectorized percentile normalization (avoids Python loops)

    Parameters
    ----------
    img : ndarray (float32 [0,1])
        Input normalized SEM image.
    scales : tuple
        Gaussian sigmas for multi-scale Frangi.
    blend_alpha : float
        Blend weight for combining Frangi map with original image.
    preblur_sigma : float
        Light pre-smoothing to suppress pixel noise before filtering.
    percentile_norm : (low, high)
        Percentiles for robust normalization of Frangi response.
    sharpen_gain : float
        Weight of high-frequency boost (1.2-1.6 typical).
    prefer_gpu : bool
        Try GPU-backed operations first (CuPy/OpenCL), otherwise use CPU.

    Returns
    -------
    img_enhanced : ndarray (float32 [0,1])
        Line-enhanced image, same shape as input.
    """

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(True)

    img = img.astype(np.float32)
    if img.max() > 1.0 or img.min() < 0.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    if preblur_sigma > 0:
        img = _gaussian_blur(img, sigma=preblur_sigma, prefer_gpu=prefer_gpu)

    def _frangi_single(s):
        return frangi(img, scale_range=(s, s), scale_step=1, black_ridges=False)

    with ThreadPoolExecutor(max_workers=min(len(scales), 8)) as ex:
        frangi_maps = list(ex.map(_frangi_single, scales))

    frangi_map = np.max(np.stack(frangi_maps, axis=0), axis=0)

    p_low, p_high = np.percentile(frangi_map, percentile_norm)
    frangi_norm = np.clip((frangi_map - p_low) / (p_high - p_low + 1e-8), 0, 1)

    umat = cv2.UMat(frangi_norm.astype(np.float32))
    blurred = cv2.GaussianBlur(umat, (0, 0), 1.0)
    line_sharp = cv2.addWeighted(umat, sharpen_gain, blurred, -0.4, 0)
    line_sharp = cv2.UMat.get(line_sharp)
    line_sharp = np.clip(line_sharp, 0, 1)

    img_enhanced = np.clip((1.0 - blend_alpha) * img + blend_alpha * line_sharp, 0, 1)
    return img_enhanced

def filter_dirt_blobs(
    img,
    threshold=0.15,
    min_size=10,
    max_size=40,
    aspect_ratio_thresh=1.4,
    inpaint_radius=7,
    inpaint_method="telea",
    n_jobs=8,
    prefer_gpu: bool = True,
):
    """
    Parallel dirt-blob removal with GPU-accelerated inpainting.

    Accelerations
    -------------
    - Joblib parallel region filtering (CPU)
    - OpenCL-enabled cv2.inpaint (GPU)
    - Early mask pruning for large images

    Parameters
    ----------
    img : ndarray (float32 [0,1])
        Input SEM image.
    threshold : float
        Intensity threshold for bright blob detection.
    min_size, max_size : int
        Pixel area range of blobs to remove.
    aspect_ratio_thresh : float
        Roundness threshold (smaller = stricter).
    inpaint_radius : int
        Inpainting radius in pixels.
    inpaint_method : str
        'telea' or 'ns'.
    n_jobs : int
        Parallel CPU jobs for region filtering.
    prefer_gpu : bool
        Use CuPy for gaussian/threshold steps when available, else CPU.

    Returns
    -------
    img_clean : ndarray (float32 [0,1])
        Image with dirt blobs removed and inpainted.
    """

    global HAS_CUPY
    cv2.ocl.setUseOpenCL(True)

    smoothed = _gaussian_blur(img, sigma=0.5, prefer_gpu=prefer_gpu)
    if prefer_gpu and HAS_CUPY:
        try:
            bin_img = cp.asnumpy(cp.asarray(smoothed) > threshold)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"CuPy thresholding failed ({exc}); falling back to CPU.")
            HAS_CUPY = False
            bin_img = smoothed > threshold
    else:
        bin_img = smoothed > threshold
    labeled = label(bin_img)
    props = regionprops(labeled)

    mask = np.zeros(img.shape, dtype=np.uint8)
    for p in props:
        if min_size <= p.area <= max_size:
            minr, minc, maxr, maxc = p.bbox
            h, w = maxr - minr, maxc - minc
            aspect = max(h / w, w / h)
            if aspect < aspect_ratio_thresh:
                mask[labeled == p.label] = 255


    # --- Prepare 8-bit mask and image for OpenCV inpainting ---
    mask = (mask > 0).astype(np.uint8)
    img_uint8 = (img * 255).astype(np.uint8)

    # --- Choose inpainting algorithm ---
    flags = cv2.INPAINT_TELEA if inpaint_method.lower() == "telea" else cv2.INPAINT_NS

    # --- Perform inpainting (CPU-safe, 1-channel mask) ---
    img_inpainted = cv2.inpaint(img_uint8, mask, inpaintRadius=inpaint_radius, flags=flags)

    # --- Return normalized float32 image ---
    return img_inpainted.astype(np.float32) / 255.0
    """
    img_uint8 = (img * 255).astype(np.uint8)
    flags = cv2.INPAINT_TELEA if inpaint_method.lower() == "telea" else cv2.INPAINT_NS
    mask = (mask > 0).astype(np.uint8) #added experimental


    img_inpainted = cv2.inpaint(cv2.UMat(img_uint8), cv2.UMat(mask),
                                inpaintRadius=inpaint_radius, flags=flags)
    img_inpainted = cv2.UMat.get(img_inpainted)
    return img_inpainted.astype(np.float32) / 255.0
    """

def enhance_dog(img, sigma_small=1.5, sigma_large=7.0, prefer_gpu: bool = True):
    """
    Difference-of-Gaussians (DoG) enhancer.
    Highlights mid-frequency structures such as ferroelastic lines.

    Parameters
    ----------
    prefer_gpu : bool
        Use CuPy-based gaussian filtering when available before falling back to CPU.
    """
    base = img.astype(np.float32)
    base = (base - base.min()) / (np.ptp(base) + 1e-8)

    g_small = _gaussian_blur(base, sigma=sigma_small, prefer_gpu=prefer_gpu)
    g_large = _gaussian_blur(base, sigma=sigma_large, prefer_gpu=prefer_gpu)
    dog = g_small - g_large
    dog = (dog - dog.min()) / (np.ptp(dog) + 1e-8)

    return dog.astype(np.float32)


def enhance_clahe(img, clip=4.0, tile=8):
    """
    CLAHE local contrast enhancement.
    """
    base = img.astype(np.float32)
    base = (base - base.min()) / (np.ptp(base) + 1e-8)

    img8 = np.clip(base * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    out = clahe.apply(img8).astype(np.float32) / 255.0
    out = (out - out.min()) / (np.ptp(out) + 1e-8)
    return out


def enhance_unsharp(img, amount=0.3, radius=2.0):
    """
    Unsharp mask to boost edges.
    """
    base = img.astype(np.float32)
    base = (base - base.min()) / (np.ptp(base) + 1e-8)

    blur = cv2.GaussianBlur(base, (0, 0), sigmaX=radius, sigmaY=radius)
    sharp = base + amount * (base - blur)
    sharp = np.clip(sharp, 0, 1)
    return sharp.astype(np.float32)


def enhance_lines_modular(
    img,
    # Frangi
    use_frangi=True,
    frangi_scales=(0.5, 1.0, 1.5),
    frangi_alpha=0.2,
    w_frangi=0.4,
    # DoG
    use_dog=True,
    dog_sigma_small=1.0,
    dog_sigma_large=4.0,
    w_dog=1.0,
    # CLAHE
    use_clahe=False,
    clahe_clip=2.0,
    clahe_tile=16,
    w_clahe=0.0,
    # Unsharp
    use_unsharp=False,
    unsharp_amount=0.3,
    unsharp_radius=2.0,
    w_unsharp=0.0,
    prefer_gpu: bool = True,
):
    """
    Modular line enhancer combining:
      - Frangi (existing enhance_lines_frangi)
      - DoG
      - (optionally) CLAHE, Unsharp

    Default config matches your final grid-search:
      LINES = Normalize( DoG(1,4)*1.0 + Frangi(0.5,1,1.5)*0.4 )

    Parameters
    ----------
    prefer_gpu : bool
        Propagate GPU preference down to the component filters.
    """
    base = img.astype(np.float32)
    base = (base - base.min()) / (np.ptp(base) + 1e-8)

    components = []

    # --- Frangi component (reuses your optimized function) ---
    if use_frangi and w_frangi > 0:
        fr = enhance_lines_frangi(
            base,
            scales=frangi_scales,
            blend_alpha=frangi_alpha,
            prefer_gpu=prefer_gpu,
        )
        components.append(w_frangi * fr)

    # --- DoG component ---
    if use_dog and w_dog > 0:
        dog = enhance_dog(
            base,
            sigma_small=dog_sigma_small,
            sigma_large=dog_sigma_large,
            prefer_gpu=prefer_gpu,
        )
        components.append(w_dog * dog)

    # --- CLAHE component (usually off for LINES) ---
    if use_clahe and w_clahe > 0:
        cl = enhance_clahe(base, clip=clahe_clip, tile=clahe_tile)
        components.append(w_clahe * cl)

    # --- Unsharp component (usually off for LINES) ---
    if use_unsharp and w_unsharp > 0:
        us = enhance_unsharp(base, amount=unsharp_amount, radius=unsharp_radius)
        components.append(w_unsharp * us)

    if not components:
        out = base
    else:
        out = np.sum(np.stack(components, axis=0), axis=0)

    # Robust percentile normalization
    p1, p99 = np.percentile(out, [1, 99])
    out = np.clip((out - p1) / (p99 - p1 + 1e-8), 0, 1)

    return out.astype(np.float32)


from skimage.measure import label, regionprops
import numpy as np

def make_prompt_map_from_lines(
    img_lines: np.ndarray,
    thr: float = 0.8,
    stride: int = 12,
    min_area: int = 150,
    min_aspect: float = 3.0,
    max_width: int = 25,
    max_points: int = 2000,
):
    """
    1) Threshold the line map.
    2) Keep only large, elongated, thin components (ferroelastic slabs).
    3) Sample points inside those components, with an overall cap on prompts.
    """

    h, w = img_lines.shape

    # --- 1. Fixed high threshold (simple & stable) ---
    bin_map = (img_lines >= thr).astype(np.uint8)
    mask_frac = bin_map.mean()
    print(f"[PROMPTS] thr={thr:.3f}, mask_frac={mask_frac:.4f}")

    if bin_map.sum() == 0:
        return np.empty((0, 2), dtype=np.int32), np.zeros_like(bin_map)

    # --- 2. Connected components: keep long, thin slabs ---
    labeled = label(bin_map)
    props = regionprops(labeled)

    mask_filtered = np.zeros_like(bin_map, dtype=np.uint8)

    for p in props:
        area = p.area
        minr, minc, maxr, maxc = p.bbox
        h_box, w_box = maxr - minr, maxc - minc

        aspect = max(h_box, w_box) / max(1, min(h_box, w_box))
        thin_width = min(h_box, w_box)

        # Strict ferroelastic-line filter
        if (
            150 <= area <= 30000      # slab-sized, not grain-sized
            and aspect >= 12          # very elongated shape
            and thin_width <= 20      # thin structure
        ):
            mask_filtered[labeled == p.label] = 1

    # --- 3. Fallback: if filtering killed everything, use the raw threshold map ---
    if mask_filtered.sum() == 0:
        print("[PROMPTS] component filter removed everything, falling back to bin_map")
        mask_filtered = bin_map

    # --- 4. Sample prompt coordinates with stride + global cap ---
    ys, xs = np.nonzero(mask_filtered)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32), mask_filtered

    xs_s = xs[::stride]
    ys_s = ys[::stride]
    coords = np.stack([xs_s, ys_s], axis=1).astype(np.int32)

    # Global cap to prevent 100k prompts
    if coords.shape[0] > max_points:
        idx = np.linspace(0, coords.shape[0] - 1, max_points, dtype=int)
        coords = coords[idx]

    print(f"[PROMPTS] final coords: {coords.shape[0]}")
    return coords, mask_filtered

"""

def make_prompt_map_from_lines(img_lines: np.ndarray, thr: float = 0.6, stride: int = 8):

    h, w = img_lines.shape
    # threshold
    bin_map = (img_lines >= thr).astype(np.uint8)
    if bin_map.sum() == 0:
        return np.empty((0, 2), dtype=np.int32), np.zeros_like(bin_map)

    # thin to 1-px lines
    skel = skeletonize(bin_map.astype(bool)).astype(np.uint8)

    # sample points on a grid to limit prompts
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32), skel

    # stride sampling (cheap Poisson-ish thinning)
    xs_s = xs[::stride]
    ys_s = ys[::stride]
    coords = np.stack([xs_s, ys_s], axis=1).astype(np.int32)
    return coords, skel
"""
