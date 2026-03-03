"""Multi-position scanning module for smart microscopy.

Provides grid scanning and cell deduplication for imaging
fields larger than the camera FOV.

Provides grid_positions for tiling, scan_and_detect_mda for
MDA-native scanning with per-frame detection callbacks, and
deduplicate_cells for merging overlapping detections.

For standard multi-position scans without callbacks, use
pymmcore-plus MDASequence directly.
"""

from typing import Any, Generator, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from useq import MDAEvent

from ..detection.cells import detect_cells


def grid_positions(world_width, world_height, fov_size, overlap_frac=0.3):
    """Generate grid scan positions to tile a world with overlap.

    Args:
        world_width: World width in world units.
        world_height: World height in world units.
        fov_size: FOV size in world units (assumes square).
        overlap_frac: Fraction of FOV to overlap (0.3 = 30%).

    Returns:
        List of (x, y) stage center positions.
    """
    step = fov_size * (1 - overlap_frac)
    half_fov = fov_size / 2

    # Start so first FOV's left edge is at 0, last FOV's right edge >= world_width
    xs = []
    x = half_fov
    while x - half_fov < world_width:
        xs.append(min(x, world_width - half_fov))
        x += step
        if xs[-1] >= world_width - half_fov:
            break

    ys = []
    y = half_fov
    while y - half_fov < world_height:
        ys.append(min(y, world_height - half_fov))
        y += step
        if ys[-1] >= world_height - half_fov:
            break

    return [(float(x), float(y)) for y in ys for x in xs]


def scan_and_detect_mda(
    positions: Sequence[tuple[float, float]],
    pixel_size: float = 1.0,
    threshold_sigma: float = 2.5,
    min_area_px: int = 20,
    fill_holes: bool = False,
    channels: Optional[Sequence[str]] = None,
    exposure: float = 50.0,
    metadata: Optional[dict[str, Any]] = None,
):
    """Create an MDA-native grid scan with cell detection.

    Returns a generator + on_frame callback + shared state, following
    the same pattern as multiscale_mda() and autofocus_mda().

    Usage::

        gen, on_frame, state = scan_and_detect_mda(positions, pixel_size=0.5)
        mmc.mda.events.frameReady.connect(on_frame)
        mmc.mda.run(gen())
        cells = state['world_positions']

    Args:
        positions: List of (x, y) stage positions.
        pixel_size: World units per pixel.
        threshold_sigma: Detection threshold.
        min_area_px: Minimum detection area.
        fill_holes: Whether to fill holes in detections.
        channels: Optional channel config names.
        exposure: Exposure time in ms.
        metadata: Extra metadata.

    Returns:
        Tuple of (event_generator, on_frame_callback, shared_state).
        shared_state contains 'world_positions' (list of (wx, wy)).
    """
    state: dict[str, Any] = {
        'world_positions': [],
        'per_tile_counts': [],
    }

    def event_generator():
        ch_list = channels or [None]
        extra_meta = metadata or {}
        for p_idx, (x, y) in enumerate(positions):
            for c_idx, ch in enumerate(ch_list):
                idx = {"p": p_idx}
                if len(ch_list) > 1:
                    idx["c"] = c_idx
                meta = {"scan_position": p_idx, **extra_meta}
                kwargs = {
                    "index": idx,
                    "exposure": exposure,
                    "x_pos": x,
                    "y_pos": y,
                    "metadata": meta,
                }
                if ch is not None:
                    kwargs["channel"] = {"config": ch}
                yield MDAEvent(**kwargs)

    def on_frame(image, event, meta=None):
        H, W = image.shape[:2]
        sx = event.x_pos if event.x_pos is not None else 0.0
        sy = event.y_pos if event.y_pos is not None else 0.0

        cells = detect_cells(
            image, threshold_sigma=threshold_sigma,
            min_area_px=min_area_px, fill_holes=fill_holes,
        )
        state['per_tile_counts'].append(len(cells))

        for c in cells:
            col, row = c['centroid_px']
            wx = sx + (col - W / 2) * pixel_size
            wy = sy + (row - H / 2) * pixel_size
            state['world_positions'].append((wx, wy))

    return event_generator, on_frame, state


def deduplicate_cells(world_positions, min_dist=5):
    """Merge duplicate detections from overlapping FOVs.

    Uses hierarchical clustering with complete linkage so that
    all detections in a cluster are within ``min_dist`` of each
    other. This avoids transitive merging of distant cells.

    A small default ``min_dist=5`` works for most scans because
    the same cell in overlapping tiles produces centroids that
    differ by only 1-3 world units. Increase for noisier
    detections or lower magnifications.

    Args:
        world_positions: List of (x, y) world coordinates.
        min_dist: Maximum distance to consider as same cell.

    Returns:
        List of (x, y) unique cell positions (cluster centroids).
    """
    if len(world_positions) < 2:
        return list(world_positions)

    pts = np.array(world_positions, dtype=float)
    Z = linkage(pts, method='complete')
    clusters = fcluster(Z, t=min_dist, criterion='distance')

    unique = []
    for cid in sorted(set(clusters)):
        mask = clusters == cid
        cluster_pts = pts[mask]
        unique.append((float(cluster_pts[:, 0].mean()),
                        float(cluster_pts[:, 1].mean())))
    return unique
