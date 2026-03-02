"""Z-stack acquisition and analysis for smart microscopy.

Handles cells distributed across different focal planes.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from useq import MDAEvent

from ..detection.cells import detect_cells


def acquire_zstack(mmc, z_range=(-30, 30), z_step=3.0, exposure=None):
    """Acquire a Z-stack at the current XY position via MDA events.

    Generates one MDAEvent per Z position and executes them with
    ``run_events()``.

    Args:
        mmc: pymmcore-plus core instance.
        z_range: (z_min, z_max) in microns.
        z_step: Step size in microns.
        exposure: Optional exposure time in ms.

    Returns:
        (stack, z_positions): stack is (N, H, W) array, z_positions is 1D array.
    """
    from .core import run_events

    z_positions = np.arange(z_range[0], z_range[1] + z_step / 2, z_step)

    events = []
    for i, z in enumerate(z_positions):
        kwargs = {"z_pos": float(z), "index": {"z": i}}
        if exposure is not None:
            kwargs["exposure"] = float(exposure)
        events.append(MDAEvent(**kwargs))

    result_frames = run_events(mmc, events)
    frames = [img for img, ev in result_frames]

    return np.array(frames), z_positions


def detect_cells_zstack(stack, z_positions, threshold_sigma=2.5,
                        min_area_px=15, fill_holes=False,
                        dedup_dist=15):
    """Detect cells across a Z-stack and merge across planes.

    Uses global stats for consistent thresholding, then clusters
    detections across Z-planes. Returns centroid from best-focus slice.

    IMPORTANT: Use best-focus centroid, not averaged centroid.
    Out-of-focus cells have blurred signals that shift apparent position.

    Args:
        stack: (N, H, W) array of Z-stack images.
        z_positions: 1D array of Z positions.
        threshold_sigma: Detection threshold.
        min_area_px: Minimum area for detection.
        fill_holes: Whether to fill holes.
        dedup_dist: Clustering distance for merging across Z.

    Returns:
        List of dicts with:
        {x, y, best_z, peak, n_slices, confidence}
        confidence is 'high' (10+ slices, peak >= 15),
        'medium' (5-9 slices), or 'low' (<5 slices).
    """
    global_mean = float(stack.mean())
    global_std = float(stack.std())

    all_detections = []
    for zi, z in enumerate(z_positions):
        cells = detect_cells(stack[zi], threshold_sigma=threshold_sigma,
                             min_area_px=min_area_px, fill_holes=fill_holes,
                             global_stats=(global_mean, global_std))
        for c in cells:
            all_detections.append({
                'x': c['centroid_px'][0],
                'y': c['centroid_px'][1],
                'z': float(z),
                'peak': c['peak'],
                'area': c['area_px'],
            })

    if not all_detections:
        return []

    # Cluster XY positions across Z
    pts = np.array([(d['x'], d['y']) for d in all_detections])

    if len(pts) < 2:
        d = all_detections[0]
        return [{
            'x': round(d['x']), 'y': round(d['y']),
            'best_z': d['z'], 'peak': d['peak'],
            'n_slices': 1, 'confidence': 'low',
        }]

    Z_linkage = linkage(pts, method='average')
    clusters = fcluster(Z_linkage, t=dedup_dist, criterion='distance')

    results = []
    for cid in sorted(set(clusters)):
        mask = clusters == cid
        cluster_dets = [d for d, m in zip(all_detections, mask) if m]

        # Use centroid from BEST FOCUS slice only (highest peak)
        # Don't average — out-of-focus cells have shifted centroids
        best = max(cluster_dets, key=lambda d: d['peak'])
        n_slices = len(cluster_dets)

        if n_slices >= 10 and best['peak'] >= 15:
            confidence = 'high'
        elif n_slices >= 5:
            confidence = 'medium'
        else:
            confidence = 'low'

        results.append({
            'x': round(best['x']),
            'y': round(best['y']),
            'best_z': best['z'],
            'peak': best['peak'],
            'n_slices': n_slices,
            'confidence': confidence,
        })

    return results
