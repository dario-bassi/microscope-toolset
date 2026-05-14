"""First-contact sample classifier.

Cheap heuristic routing primitive: takes a fresh snap from an unknown
sample and returns a confidence-ranked list of guesses. Pre-recipe,
pre-segmentation; runs in < 200 ms on a 512×512 frame so it can be
the first call after ``connect()``.

Recognised classes:
  - ``plate_reader_grid``     : 8x12 RLU array (96-well plate reader)
  - ``voronoi_monolayer``     : tessellated tissue with bright nuclei
  - ``phase_contrast_tessellation``: tessellated tissue, mid-grey
  - ``sparse_cells``          : few isolated bright spots on dark bg
  - ``single_bright_spot``    : 1-3 detections, mostly empty
  - ``wide_dynamic_range_field``: continuous gradient / speckle field
  - ``unknown``               : nothing matched

API:
    extract_features(image) -> dict[str, float]
    classify_sample(image, *, channel=None) -> list[(class, confidence)]

Sprint #17 (2026-04-26).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


CLASSES = (
    "plate_reader_grid",
    "voronoi_monolayer",
    "phase_contrast_tessellation",
    "sparse_cells",
    "single_bright_spot",
    "wide_dynamic_range_field",
    "unknown",
)


def _to_gray(image: np.ndarray, channel: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3:
        if channel is not None:
            arr = arr[..., channel] if arr.shape[-1] <= 4 else arr[channel]
        else:
            arr = arr.mean(axis=-1) if arr.shape[-1] <= 4 else arr.mean(axis=0)
    if arr.size > 256 * 256:
        # Downsample by stride for speed.
        sy = max(1, arr.shape[0] // 256)
        sx = max(1, arr.shape[1] // 256)
        arr = arr[::sy, ::sx]
    return arr


def extract_features(image: np.ndarray, channel: Optional[int] = None) -> dict:
    """Compute the 5-feature vector used by the classifier.

    All features are O(N) on the downsampled view.
    """
    from scipy import ndimage

    img = _to_gray(image, channel=channel)
    if img.size == 0:
        return {
            "entropy_binned": 0.0, "hist_bimodality": 0.0,
            "fft_grid_score": 0.0, "edge_density": 0.0,
            "n_components": 0, "median_area": 0.0, "area_cv": 0.0,
            "sparsity": 1.0,
        }

    # 1. Bin-mass count.
    hist, _ = np.histogram(img.ravel(), bins=256)
    total = float(hist.sum())
    occupied = int((hist / max(total, 1.0) > 0.005).sum())

    # 2. Histogram bimodality (top-2 / total on a 64-bin Gaussian-smoothed hist).
    h64, _ = np.histogram(img.ravel(), bins=64)
    h64s = ndimage.gaussian_filter1d(h64.astype(float), sigma=1.5)
    if h64s.sum() > 0:
        # Find the two largest peaks.
        diffs = np.diff(np.sign(np.diff(h64s)))
        peaks_idx = np.where(diffs < 0)[0] + 1
        peak_amps = sorted(h64s[peaks_idx], reverse=True) if len(peaks_idx) > 0 else [0.0]
        top2 = sum(peak_amps[:2])
        bimodal = top2 / float(h64s.sum())
    else:
        bimodal = 0.0

    # 3. FFT periodicity (peak / median, ignoring DC).
    spec = np.abs(np.fft.rfft2(img - img.mean()))
    if spec.size > 1:
        spec_flat = spec.ravel()
        spec_flat[0] = 0.0
        nz = spec_flat[spec_flat > 0]
        if nz.size > 0:
            fft_score = float(spec_flat.max() / max(np.median(nz), 1e-9))
        else:
            fft_score = 0.0
    else:
        fft_score = 0.0

    # 4. Edge density (Sobel above threshold).
    sobel = np.hypot(
        ndimage.sobel(img, axis=0), ndimage.sobel(img, axis=1)
    )
    sobel_thresh = sobel.mean() + 0.5 * sobel.std()
    edge_density = float((sobel > sobel_thresh).mean())

    # 5. CC stats above mean+1σ.
    mask = img > (img.mean() + img.std())
    labels, n_components = ndimage.label(mask)
    if n_components > 0:
        sizes = np.bincount(labels.ravel(), minlength=n_components + 1)[1:]
        median_area = float(np.median(sizes))
        area_cv = float(sizes.std() / max(sizes.mean(), 1e-9))
    else:
        median_area = 0.0
        area_cv = 0.0

    sparsity = 1.0 - float(mask.mean())

    return {
        "entropy_binned": float(occupied),
        "hist_bimodality": float(bimodal),
        "fft_grid_score": float(fft_score),
        "edge_density": float(edge_density),
        "n_components": int(n_components),
        "median_area": float(median_area),
        "area_cv": float(area_cv),
        "sparsity": float(sparsity),
    }


def _soft_confidence(margin: float, k: float = 1.5) -> float:
    if margin <= 0:
        return 0.0
    return float(1.0 - np.exp(-k * margin))


def classify_sample(image: np.ndarray, *, channel: Optional[int] = None) -> list:
    """Return a confidence-ranked list of (class_name, confidence) tuples."""
    f = extract_features(image, channel=channel)

    # Decision rules → base scores in [0, 1]. Tuned on synthetic generators
    # in tests/test_sample_classifier.py — see the feature table in the
    # commit message for ranges per class.
    scores: dict[str, float] = {c: 0.0 for c in CLASSES}

    # Single bright spot: 1-3 CCs + extreme sparsity + huge FFT peak.
    if f["n_components"] <= 3 and f["sparsity"] > 0.95:
        scores["single_bright_spot"] = _soft_confidence(
            (f["sparsity"] - 0.95) * 20 + (3 - f["n_components"]) * 0.5
            + min((f["fft_grid_score"] - 1000) / 1000.0, 2.0)
        )

    # Plate-reader grid: moderate entropy (~60-100), strong periodicity
    # (FFT score 500-3000), few large CCs.
    if (60 < f["entropy_binned"] < 130 and 300 < f["fft_grid_score"] < 5000
            and f["n_components"] < 50 and f["edge_density"] < 0.18):
        scores["plate_reader_grid"] = _soft_confidence(
            (f["fft_grid_score"] - 300) / 600.0 + (130 - f["entropy_binned"]) / 70.0
        )

    # Tessellated tissue (Voronoi or phase): edge_density > 0.20 and
    # 30 < n_components < 500.
    if f["edge_density"] > 0.20 and 30 < f["n_components"] < 500:
        # Voronoi has a high-amplitude periodic signature (cells bright on
        # dark bg → strong FFT peak); phase contrast has dim halos and
        # a much lower FFT score.
        if f["fft_grid_score"] > 1000:
            scores["voronoi_monolayer"] = _soft_confidence(
                (f["edge_density"] - 0.20) * 3 + (f["n_components"] - 30) / 200.0
                + min((f["fft_grid_score"] - 1000) / 5000.0, 1.0)
            )
        else:
            scores["phase_contrast_tessellation"] = _soft_confidence(
                (f["edge_density"] - 0.20) * 3 + (f["n_components"] - 30) / 200.0
            )

    # Sparse cells: high sparsity, low entropy (few intensity bins),
    # moderate CC count.
    if (f["sparsity"] > 0.85 and f["entropy_binned"] < 30
            and 5 <= f["n_components"] <= 200):
        scores["sparse_cells"] = _soft_confidence(
            (f["sparsity"] - 0.85) * 5 + (30 - f["entropy_binned"]) / 30.0
        )

    # Wide-dynamic-range field: high entropy AND moderate FFT score
    # (some structure) AND many CCs (continuous gradient → 1000+).
    if (f["entropy_binned"] > 60 and 50 < f["fft_grid_score"] < 1000
            and f["n_components"] > 500):
        scores["wide_dynamic_range_field"] = _soft_confidence(
            (f["entropy_binned"] - 60) / 60.0 + (f["fft_grid_score"] - 50) / 200.0
        )

    # Unknown / flat noise: high entropy, very low FFT score (no structure).
    if f["fft_grid_score"] < 20 and f["n_components"] > 500:
        scores["unknown"] = _soft_confidence(
            (20 - f["fft_grid_score"]) / 10.0 + (f["n_components"] - 500) / 1000.0
        )

    # If nothing scored, raise unknown floor.
    max_other = max(v for k, v in scores.items() if k != "unknown")
    if max_other < 0.1:
        scores["unknown"] = max(scores["unknown"], 0.6)
    elif scores["unknown"] == 0.0:
        scores["unknown"] = 0.05

    # Sort and return non-zero (plus unknown floor).
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(name, score) for name, score in ranked if score > 0]
