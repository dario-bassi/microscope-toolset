"""Texture analysis for tissue classification and surface characterization.

Computes texture descriptors from grayscale images using GLCM,
local binary patterns, and sliding-window statistics.

Functions:
    glcm_features     -- Haralick texture features from GLCM
    local_binary_pattern -- LBP histogram descriptor
    texture_map       -- Sliding-window texture feature map
    entropy_map       -- Local Shannon entropy
    color_deconvolution -- Stain separation (H&E, etc.)
"""

import numpy as np
from scipy import ndimage


def glcm_features(image, distances=None, angles=None, levels=32):
    """Compute texture features from gray-level co-occurrence matrix.

    Args:
        image: 2D grayscale image.
        distances: List of pixel distances (default: [1]).
        angles: List of angles in radians (default: [0]).
        levels: Number of gray levels for quantization.

    Returns:
        dict with:
            contrast: Intensity contrast between neighboring pixels.
            correlation: Linear dependency of gray levels.
            energy: Sum of squared GLCM elements (uniformity).
            homogeneity: Closeness of distribution to GLCM diagonal.
            entropy: Randomness of gray-level distribution.
            dissimilarity: Mean absolute difference of co-occurring pairs.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("Image must be 2D")

    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0]

    # Quantize to [0, levels-1]
    lo, hi = img.min(), img.max()
    if hi > lo:
        quantized = ((img - lo) / (hi - lo) * (levels - 1)).astype(np.int32)
    else:
        quantized = np.zeros_like(img, dtype=np.int32)

    h, w = quantized.shape

    # Accumulate GLCM over all distance/angle combinations
    glcm = np.zeros((levels, levels), dtype=np.float64)

    for d in distances:
        for angle in angles:
            dy = int(round(-d * np.sin(angle)))
            dx = int(round(d * np.cos(angle)))

            # Valid region
            y0 = max(0, -dy)
            y1 = min(h, h - dy)
            x0 = max(0, -dx)
            x1 = min(w, w - dx)

            if y1 <= y0 or x1 <= x0:
                continue

            rows = quantized[y0:y1, x0:x1].ravel()
            cols = quantized[y0 + dy : y1 + dy, x0 + dx : x1 + dx].ravel()

            for i, j in zip(rows, cols, strict=False):
                glcm[i, j] += 1

    # Symmetrize and normalize
    glcm = glcm + glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total

    # Compute features
    i_idx, j_idx = np.meshgrid(range(levels), range(levels), indexing="ij")
    i_idx = i_idx.astype(np.float64)
    j_idx = j_idx.astype(np.float64)

    # Means and stds
    mu_i = np.sum(i_idx * glcm)
    mu_j = np.sum(j_idx * glcm)
    sig_i = np.sqrt(np.sum((i_idx - mu_i) ** 2 * glcm))
    sig_j = np.sqrt(np.sum((j_idx - mu_j) ** 2 * glcm))

    contrast = float(np.sum((i_idx - j_idx) ** 2 * glcm))
    dissimilarity = float(np.sum(np.abs(i_idx - j_idx) * glcm))
    homogeneity = float(np.sum(glcm / (1 + (i_idx - j_idx) ** 2)))
    energy = float(np.sum(glcm**2))

    # Correlation
    if sig_i > 0 and sig_j > 0:
        correlation = float(np.sum((i_idx - mu_i) * (j_idx - mu_j) * glcm) / (sig_i * sig_j))
    else:
        correlation = 0.0

    # Entropy
    nonzero = glcm > 0
    entropy = -float(np.sum(glcm[nonzero] * np.log2(glcm[nonzero])))

    return {
        "contrast": contrast,
        "correlation": correlation,
        "energy": energy,
        "homogeneity": homogeneity,
        "entropy": entropy,
        "dissimilarity": dissimilarity,
    }


def local_binary_pattern(image, radius=1, n_points=8):
    """Compute local binary pattern histogram.

    For each pixel, compare with n_points neighbors at given radius.
    Build a histogram of the resulting binary codes.

    Args:
        image: 2D grayscale image.
        radius: Radius of circular neighborhood.
        n_points: Number of sample points on the circle.

    Returns:
        dict with:
            histogram: Normalized histogram of LBP codes (2^n_points bins
                if n_points <= 8, else 256 bins using uniform patterns).
            n_patterns: Number of unique LBP patterns found.
            uniformity: Fraction of uniform patterns (at most 2 bit transitions).
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("Image must be 2D")

    h, w = img.shape
    margin = radius + 1

    if h < 2 * margin + 1 or w < 2 * margin + 1:
        return {"histogram": np.array([1.0]), "n_patterns": 1, "uniformity": 1.0}

    # Sample points on circle
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    dy = -radius * np.sin(angles)  # row offset
    dx = radius * np.cos(angles)  # col offset

    # Compute LBP for interior pixels
    interior = img[margin:-margin, margin:-margin]
    codes = np.zeros_like(interior, dtype=np.int32)

    for k in range(n_points):
        # Bilinear interpolation at neighbor position
        ny = np.arange(margin, h - margin) + dy[k]
        nx = np.arange(margin, w - margin) + dx[k]
        ny_grid, nx_grid = np.meshgrid(ny, nx, indexing="ij")

        y0 = np.floor(ny_grid).astype(int)
        x0 = np.floor(nx_grid).astype(int)
        fy = ny_grid - y0
        fx = nx_grid - x0

        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)

        neighbor = (
            img[y0, x0] * (1 - fy) * (1 - fx)
            + img[y1, x0] * fy * (1 - fx)
            + img[y0, x1] * (1 - fy) * fx
            + img[y1, x1] * fy * fx
        )

        codes += (neighbor >= interior).astype(np.int32) << k

    # Histogram
    n_bins = min(2**n_points, 256)
    hist, _ = np.histogram(codes.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total

    # Count uniform patterns (at most 2 bit transitions)
    unique_codes = np.unique(codes)
    n_uniform = 0
    for code in unique_codes:
        bits = format(code, f"0{n_points}b")
        transitions = sum(1 for a, b in zip(bits, bits[1:] + bits[0], strict=False) if a != b)
        if transitions <= 2:
            n_uniform += 1

    uniformity = n_uniform / max(len(unique_codes), 1)

    return {
        "histogram": hist,
        "n_patterns": len(unique_codes),
        "uniformity": round(float(uniformity), 3),
    }


def texture_map(image, window_size=15, feature="std"):
    """Compute a sliding-window texture feature map.

    Args:
        image: 2D grayscale image.
        window_size: Side length of square window (must be odd).
        feature: One of 'std', 'entropy', 'range', 'contrast'.

    Returns:
        2D array same shape as input with local texture values.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("Image must be 2D")

    if window_size % 2 == 0:
        window_size += 1

    if feature == "std":
        # Local standard deviation via uniform filter
        mean = ndimage.uniform_filter(img, size=window_size)
        mean_sq = ndimage.uniform_filter(img**2, size=window_size)
        variance = np.maximum(mean_sq - mean**2, 0)
        return np.sqrt(variance)

    elif feature == "entropy":
        return _local_entropy(img, window_size)

    elif feature == "range":
        max_filt = ndimage.maximum_filter(img, size=window_size)
        min_filt = ndimage.minimum_filter(img, size=window_size)
        return max_filt - min_filt

    elif feature == "contrast":
        # Difference of local max and min normalized by sum
        max_filt = ndimage.maximum_filter(img, size=window_size)
        min_filt = ndimage.minimum_filter(img, size=window_size)
        denom = max_filt + min_filt
        denom[denom < 1e-10] = 1e-10
        return (max_filt - min_filt) / denom

    else:
        raise ValueError(
            f"Unknown feature: {feature}. " f"Use 'std', 'entropy', 'range', or 'contrast'."
        )


def entropy_map(image, window_size=9, n_bins=32):
    """Compute local Shannon entropy map.

    Args:
        image: 2D grayscale image.
        window_size: Side length of local window.
        n_bins: Number of bins for local histograms.

    Returns:
        2D array of local entropy values (bits).
    """
    return _local_entropy(np.asarray(image, dtype=np.float64), window_size, n_bins)


def color_deconvolution(rgb, stain="hematoxylin_eosin"):
    """Separate stains from an RGB histology image.

    Implements color deconvolution using a stain matrix in optical
    density space. Supports H&E and custom stain vectors.

    Args:
        rgb: (H, W, 3) RGB image (uint8 or float [0,255]).
        stain: Stain name ('hematoxylin_eosin') or custom (3,3) matrix
            where each row is a normalized stain vector in OD space.

    Returns:
        dict with:
            channel_0: First stain channel (e.g., hematoxylin).
            channel_1: Second stain channel (e.g., eosin).
            channel_2: Residual channel.
            stain_matrix: The stain matrix used.
    """
    img = np.asarray(rgb, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input must be (H, W, 3) RGB image")

    # Stain matrices (rows = normalized stain vectors in OD space)
    STAIN_MATRICES = {
        "hematoxylin_eosin": np.array(
            [
                [0.6500, 0.7040, 0.2860],  # Hematoxylin
                [0.0720, 0.9900, 0.1050],  # Eosin
                [0.2680, 0.5700, 0.7780],  # Residual/DAB
            ]
        ),
    }

    if isinstance(stain, str):
        if stain not in STAIN_MATRICES:
            raise ValueError(
                f"Unknown stain: {stain}. " f"Available: {list(STAIN_MATRICES.keys())}"
            )
        M = STAIN_MATRICES[stain]
    else:
        M = np.asarray(stain, dtype=np.float64)
        if M.shape != (3, 3):
            raise ValueError("Custom stain matrix must be (3, 3)")

    # Normalize rows
    for i in range(3):
        norm = np.linalg.norm(M[i])
        if norm > 0:
            M[i] /= norm

    # Convert to optical density
    img_clipped = np.clip(img, 1, 255)
    od = -np.log10(img_clipped / 255.0)

    # Deconvolve: OD = concentration * stain_matrix
    # concentration = OD * inv(stain_matrix)
    M_inv = np.linalg.inv(M)
    h, w, _ = od.shape
    od_flat = od.reshape(-1, 3)
    concentrations = od_flat @ M_inv.T
    concentrations = np.clip(concentrations, 0, None)

    channels = concentrations.reshape(h, w, 3)

    return {
        "channel_0": channels[:, :, 0],
        "channel_1": channels[:, :, 1],
        "channel_2": channels[:, :, 2],
        "stain_matrix": M,
    }


def _local_entropy(img, window_size, n_bins=32):
    """Compute local entropy using a sliding window approach."""
    h, w = img.shape
    result = np.zeros_like(img)
    half = window_size // 2

    # Quantize
    lo, hi = img.min(), img.max()
    if hi > lo:
        quantized = ((img - lo) / (hi - lo) * (n_bins - 1)).astype(np.int32)
    else:
        return result

    padded = np.pad(quantized, half, mode="reflect")

    # Use integral histogram approach for efficiency on small windows
    for y in range(h):
        for x in range(w):
            patch = padded[y : y + window_size, x : x + window_size].ravel()
            hist = np.bincount(patch, minlength=n_bins).astype(np.float64)
            hist /= hist.sum()
            nonzero = hist > 0
            result[y, x] = -np.sum(hist[nonzero] * np.log2(hist[nonzero]))

    return result
