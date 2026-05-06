"""Tissue segmentation via membrane channel inversion.

The core primitive for confluent tissue analysis: threshold membrane
boundaries, invert to get cell interiors, then connected-components
to count and locate individual cells.

Two strategies:
- Otsu (default, robust to optical artifacts like PSF blur, vignetting, debris)
- Sigma-based (mean + sigma*std, for clean images)

When tissue_mode=True, applies adaptive area filtering:
- Interior cells: keep if area >= 20% of median (catches junction fragments)
- Edge cells: keep if area >= 50% of median (catches partial boundary cells)
"""

import cv2
import numpy as np


def segment_tissue(
    image,
    use_otsu=True,
    threshold_sigma=1.5,
    open_kernel=5,
    dilate_iter=1,
    min_area=50,
    tissue_mode=True,
    connectivity=4,
):
    """Segment tissue cells from a membrane channel image.

    Args:
        image: 2D array, membrane channel (bright boundaries, dark interiors).
        use_otsu: Use Otsu threshold (recommended). If False, uses sigma-based.
        threshold_sigma: Sigma for sigma-based threshold (ignored if use_otsu).
        open_kernel: Morphological opening kernel size (removes thin junctions).
        dilate_iter: Dilation iterations before inversion.
        min_area: Absolute minimum component area in pixels.
        tissue_mode: Apply adaptive area filtering based on median cell size.
        connectivity: 4 or 8 for connected components.

    Returns:
        dict with keys:
            cells: list of dicts with px, py, area, label, touches_edge
            labels: 2D label array from connectedComponentsWithStats
            stats: CC stats array
            centroids: CC centroids array
            n_labels: number of labels (excluding background)
            binary: thresholded membrane mask (before inversion)
    """
    if use_otsu:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (binary > 0).astype(np.uint8)
    else:
        m, s = float(image.mean()), float(image.std())
        thresh = m + threshold_sigma * s
        binary = (image > thresh).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=dilate_iter)
    inv = 1 - dilated

    if tissue_mode and open_kernel > 0:
        ok = np.ones((open_kernel, open_kernel), np.uint8)
        inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, ok)

    nl, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inv.astype(np.uint8), connectivity=connectivity
    )

    h_img, w_img = image.shape[:2]

    raw_cells = []
    for i in range(1, nl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx = float(centroids[i][0])
        cy = float(centroids[i][1])
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        touches_edge = x0 == 0 or y0 == 0 or x0 + cw >= w_img or y0 + ch >= h_img
        raw_cells.append(
            {
                "px": round(cx, 1),
                "py": round(cy, 1),
                "area": area,
                "label": i,
                "touches_edge": touches_edge,
            }
        )

    if tissue_mode and len(raw_cells) > 3:
        areas = sorted(c["area"] for c in raw_cells)
        median_area = areas[len(areas) // 2]
        interior_thresh = max(min_area, int(median_area * 0.2))
        edge_thresh = max(min_area, int(median_area * 0.5))
        raw_cells = [
            c
            for c in raw_cells
            if c["area"] >= (edge_thresh if c["touches_edge"] else interior_thresh)
        ]

    return {
        "cells": raw_cells,
        "labels": labels,
        "stats": stats,
        "centroids": centroids,
        "n_labels": nl - 1,
        "binary": binary,
    }


def measure_wound_closure(images, threshold=None, direction="horizontal"):
    """Measure wound gap closure over a time series.

    Detects the wound (cell-free region) and tracks its width over time.
    Works with BF or fluorescence images of confluent tissue with a scratch.

    Strategy: project each image along the wound direction to find the gap
    where cell density is minimal.

    Args:
        images: List of 2D arrays (time series of wound images).
        threshold: Intensity threshold to separate cells from background.
            If None, uses Otsu on each frame.
        direction: 'horizontal' (wound runs left-right, measure vertical gap)
            or 'vertical' (wound runs top-bottom, measure horizontal gap).

    Returns:
        dict with:
            gap_widths: List of gap width in pixels at each timepoint.
            closure_rate: Linear closure rate (pixels/frame) from fit.
            closure_pct: Percentage closed (1 - final_gap/initial_gap) * 100.
            leading_edges: List of (edge1, edge2) positions per frame.
            time_to_close: Frame index where gap reaches 0 (or None).
    """
    n = len(images)
    if n == 0:
        return {
            "gap_widths": [],
            "closure_rate": 0.0,
            "closure_pct": 0.0,
            "leading_edges": [],
            "time_to_close": None,
        }

    gap_widths = []
    leading_edges = []

    for img in images:
        img = np.asarray(img, dtype=np.float64)

        # Binarize: cells = bright/textured, gap = dark/uniform
        if threshold is not None:
            binary = (img > threshold).astype(np.uint8)
        else:
            img8 = np.clip(img, 0, 255).astype(np.uint8)
            _, binary = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = (binary > 0).astype(np.uint8)

        # Project along wound direction to get 1D density profile
        if direction == "horizontal":
            # Wound runs horizontally → measure vertical gap
            profile = binary.mean(axis=1)  # average across columns
        else:
            # Wound runs vertically → measure horizontal gap
            profile = binary.mean(axis=0)  # average across rows

        # Find the gap: contiguous region where density is below 50%
        gap_mask = profile < 0.5
        if not gap_mask.any():
            gap_widths.append(0)
            leading_edges.append((0, 0))
            continue

        # Find largest contiguous gap
        labeled_gaps, n_gaps = _label_runs(gap_mask)
        if n_gaps == 0:
            gap_widths.append(0)
            leading_edges.append((0, 0))
            continue

        # Find the widest gap (usually the wound)
        best_gap = 0
        best_start = 0
        best_end = 0
        for gap_id in range(1, n_gaps + 1):
            indices = np.where(labeled_gaps == gap_id)[0]
            if len(indices) > best_gap:
                best_gap = len(indices)
                best_start = indices[0]
                best_end = indices[-1]

        gap_widths.append(best_gap)
        leading_edges.append((int(best_start), int(best_end)))

    # Compute closure rate from linear fit
    if n >= 2 and gap_widths[0] > 0:
        t = np.arange(n, dtype=float)
        coeffs = np.polyfit(t, gap_widths, 1)
        closure_rate = float(-coeffs[0])  # negative slope = closing
        closure_pct = (1.0 - gap_widths[-1] / max(gap_widths[0], 1)) * 100
    else:
        closure_rate = 0.0
        closure_pct = 0.0

    # Find time to close
    time_to_close = None
    for i, gw in enumerate(gap_widths):
        if gw == 0:
            time_to_close = i
            break

    return {
        "gap_widths": gap_widths,
        "closure_rate": round(closure_rate, 2),
        "closure_pct": round(closure_pct, 1),
        "leading_edges": leading_edges,
        "time_to_close": time_to_close,
    }


def _label_runs(mask):
    """Label contiguous runs of True values in a 1D boolean array."""
    labels = np.zeros(len(mask), dtype=int)
    current_label = 0
    in_run = False
    for i in range(len(mask)):
        if mask[i]:
            if not in_run:
                current_label += 1
                in_run = True
            labels[i] = current_label
        else:
            in_run = False
    return labels, current_label


def contact_graph(labels, cells, max_boundary_px=20):
    """Build cell-cell adjacency graph from a label map.

    Two cells are neighbors if their segmented regions are separated by
    at most `max_boundary_px` pixels (bridgeable by dilation).

    Args:
        labels: 2D int array from segment_tissue (0 = background/membrane).
        cells: list of cell dicts with 'label' key (from segment_tissue).
        max_boundary_px: max membrane gap (pixels) to count as contact.

    Returns:
        dict with:
            edges: set of (label_a, label_b) tuples (a < b)
            degree: dict label -> int (number of neighbors)
            n_edges: int
            mean_degree: float
            max_degree: int
    """
    kernel = np.ones((3, 3), np.uint8)
    max_iter = max(1, max_boundary_px // 2)

    contact_dist = {}  # (min_label, max_label) -> min dilation iterations
    for c in cells:
        lbl = c["label"]
        mask = (labels == lbl).astype(np.uint8)
        for d in range(1, max_iter + 1):
            dilated = cv2.dilate(mask, kernel, iterations=d)
            overlap = set(labels[(dilated > 0) & (labels > 0) & (labels != lbl)])
            for nl in overlap:
                pair = (min(lbl, int(nl)), max(lbl, int(nl)))
                if pair not in contact_dist:
                    contact_dist[pair] = d

    edges = {pair for pair, d in contact_dist.items() if d <= max_iter}

    degree = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    for c in cells:
        if c["label"] not in degree:
            degree[c["label"]] = 0

    degrees = list(degree.values())
    return {
        "edges": edges,
        "degree": degree,
        "n_edges": len(edges),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees)) if degrees else 0,
    }
