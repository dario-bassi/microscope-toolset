"""Adaptive acquisition controller for survey->decide->zoom->measure workflows.

Provides the core loop for intelligent microscopy: scan at low magnification,
identify regions of interest (bright cells, clusters, anomalies), zoom in
at high magnification, and measure with precision.

Includes both imperative (survey_cells, zoom_and_measure, adaptive_survey)
and MDA-native (adaptive_survey_mda) interfaces.

Key functions:
    pixel_to_world  -- coordinate conversion at any magnification
    survey_cells    -- multi-tile survey with per-cell feature measurement
    find_clusters   -- spatial clustering of cells by proximity
    rank_by_feature -- rank cells/clusters by any measured feature
    zoom_and_measure -- move to ROI, switch objective, detect and measure
    adaptive_survey -- full pipeline: survey -> rank -> zoom -> measure
    adaptive_survey_mda -- MDA-native: generator + on_frame + shared_state
"""

from typing import Any, Generator, Optional, Sequence

import numpy as np
import cv2
from useq import MDAEvent, CustomAction

from ..hardware import core as hw
from ..hardware.config import get_config
from ..detection.cells import detect_cells, find_bright_centroid
from ..detection.tissue import segment_tissue
from .scanning import deduplicate_cells


# ---------------------------------------------------------------------------
# Coordinate conversion — re-exports of canonical implementations in
# hardware/core.py. These wrappers accept mag= instead of core= for
# convenience in workflows that don't hold a core reference.
# ---------------------------------------------------------------------------

def pixel_to_world(px, py, stage_x, stage_y, mag=10):
    """Convert pixel coordinates to world coordinates.

    Thin wrapper around hw.pixel_to_world for backward compatibility.
    Prefers config-based conversion when a core is available.

    Args:
        px, py: Pixel coordinates in the image.
        stage_x, stage_y: Current stage position (world coords).
        mag: Current objective magnification (fallback).

    Returns:
        (world_x, world_y) tuple.
    """
    return hw.pixel_to_world(px, py, stage_x, stage_y, mag=mag)


def world_to_pixel(wx, wy, stage_x, stage_y, mag=10):
    """Convert world coordinates to pixel coordinates.

    Thin wrapper around hw.world_to_pixel for backward compatibility.

    Args:
        wx, wy: World coordinates.
        stage_x, stage_y: Current stage position.
        mag: Current magnification (fallback).

    Returns:
        (px, py) tuple.
    """
    return hw.world_to_pixel(wx, wy, stage_x, stage_y, mag=mag)


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------

def survey_cells(core, world_size=(512, 512), channels=('brightfield',),
                 detect_method='threshold', detect_kwargs=None,
                 dedup_dist=35):
    """Multi-tile survey: scan world, detect cells, return world-coord list.

    Works at current objective. Automatically tiles based on world size and FOV.

    Args:
        core: Microscope core.
        world_size: (width, height) of sample in world pixels.
        channels: Channels to snap at each tile. First channel used for detection.
        detect_method: 'threshold' (detect_cells) or 'membrane' (segment_tissue).
        detect_kwargs: Extra kwargs for detection function.
        dedup_dist: Deduplication distance (0 to skip dedup).

    Returns:
        dict with:
            cells: list of dicts with world_x, world_y, and detection features
            images: dict of {(sx,sy): {channel: image}}
            positions: list of (sx, sy) stage positions used
            n_tiles: number of tiles acquired
    """
    cfg = get_config(core)
    mag = hw.get_objective(core)
    fov = hw.fov_size(core)
    img_w, img_h = cfg.image_width, cfg.image_height
    w, h = world_size
    dk = detect_kwargs or {}

    # Generate tile positions
    step = fov  # non-overlapping tiles for simplicity
    positions = []
    y = fov // 2
    while y <= h - fov // 2 + 1:
        x = fov // 2
        while x <= w - fov // 2 + 1:
            positions.append((x, y))
            x += step
        y += step

    if not positions:
        positions = [(w // 2, h // 2)]

    all_cells = []
    images = {}

    for sx, sy in positions:
        hw.move_to(core, sx, sy)

        tile_images = {}
        detect_img = None

        for ch in channels:
            img = hw.snap(core, channel=ch)
            tile_images[ch] = img
            if detect_img is None:
                detect_img = img  # first channel used for detection

        images[(sx, sy)] = tile_images

        # Detect cells
        if detect_method == 'membrane':
            seg = segment_tissue(detect_img, use_otsu=True,
                                 tissue_mode=dk.get('tissue_mode', True),
                                 min_area=dk.get('min_area', 50))
            for c in seg['cells']:
                wx, wy = pixel_to_world(c['px'], c['py'], sx, sy, mag)
                cell = {
                    'world_x': wx, 'world_y': wy,
                    'area': c['area'], 'label': c['label'],
                    'touches_edge': c['touches_edge'],
                    'tile': (sx, sy),
                }
                all_cells.append(cell)
        else:
            cells = detect_cells(detect_img,
                                 threshold_sigma=dk.get('threshold_sigma', 2.5),
                                 min_area_px=dk.get('min_area_px', 20),
                                 fill_holes=dk.get('fill_holes', True))
            for c in cells:
                cx_px, cy_px = c['centroid_px']  # (col, row)
                wx, wy = pixel_to_world(cx_px, cy_px, sx, sy, mag)
                cell = {
                    'world_x': wx, 'world_y': wy,
                    'area_px': c['area_px'], 'peak': c['peak'],
                    'mean_intensity': c['mean_intensity'],
                    'circularity': c.get('circularity', 0),
                    'tile': (sx, sy),
                }
                all_cells.append(cell)

        # Measure features in additional channels
        for ch in channels[1:]:
            ch_img = tile_images[ch]
            for cell in all_cells:
                if cell['tile'] != (sx, sy):
                    continue
                # Get pixel coords in this tile
                px, py = world_to_pixel(cell['world_x'], cell['world_y'],
                                        sx, sy, mag)
                px, py = int(round(px)), int(round(py))
                r = 15
                y0 = max(0, py - r)
                y1 = min(img_h, py + r)
                x0 = max(0, px - r)
                x1 = min(img_w, px + r)
                if y1 > y0 and x1 > x0:
                    region = ch_img[y0:y1, x0:x1]
                    key = ch.replace('-', '_') + '_mean'
                    cell[key] = float(region.mean())

    # Deduplicate
    if dedup_dist > 0 and len(all_cells) > 1:
        coords = [(c['world_x'], c['world_y']) for c in all_cells]
        unique = deduplicate_cells(coords, min_dist=dedup_dist)
        # Match back
        deduped = []
        used = set()
        for ux, uy in unique:
            best_i, best_d = -1, float('inf')
            for i, c in enumerate(all_cells):
                if i in used:
                    continue
                d = ((c['world_x'] - ux)**2 + (c['world_y'] - uy)**2)**0.5
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                used.add(best_i)
                deduped.append(all_cells[best_i])
        all_cells = deduped

    return {
        'cells': all_cells,
        'images': images,
        'positions': positions,
        'n_tiles': len(positions),
    }


# ---------------------------------------------------------------------------
# Clustering and ranking
# ---------------------------------------------------------------------------

def find_clusters(cells, radius=300, min_size=2, feature_key=None,
                  feature_threshold=None):
    """Find spatial clusters of cells using neighbor counting.

    For each cell, counts neighbors within `radius`. Cells with the most
    neighbors form cluster cores. Connected cells are grouped.

    Args:
        cells: List of dicts with 'world_x', 'world_y'.
        radius: Max distance (world px) to count as neighbor.
        min_size: Minimum cells to form a cluster.
        feature_key: If given, only cluster cells with this feature above threshold.
        feature_threshold: Threshold for feature_key filtering.

    Returns:
        list of cluster dicts, each with:
            cells: list of cell dicts in this cluster
            center_x, center_y: cluster centroid
            n_cells: number of cells
    """
    # Optionally filter by feature
    if feature_key and feature_threshold is not None:
        candidates = [c for c in cells if c.get(feature_key, 0) > feature_threshold]
    else:
        candidates = list(cells)

    if len(candidates) < min_size:
        return []

    # Build adjacency via distance
    coords = np.array([[c['world_x'], c['world_y']] for c in candidates])
    from scipy.spatial.distance import cdist
    D = cdist(coords, coords)

    # Simple connected-component clustering within radius
    visited = set()
    clusters = []

    for seed in range(len(candidates)):
        if seed in visited:
            continue
        # BFS
        queue = [seed]
        cluster_idx = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster_idx.append(node)
            for j in range(len(candidates)):
                if j not in visited and D[node, j] <= radius:
                    queue.append(j)

        if len(cluster_idx) >= min_size:
            cluster_cells = [candidates[i] for i in cluster_idx]
            cx = float(np.mean([c['world_x'] for c in cluster_cells]))
            cy = float(np.mean([c['world_y'] for c in cluster_cells]))
            clusters.append({
                'cells': cluster_cells,
                'center_x': round(cx, 1),
                'center_y': round(cy, 1),
                'n_cells': len(cluster_cells),
            })

    # Sort by size descending
    clusters.sort(key=lambda cl: cl['n_cells'], reverse=True)
    return clusters


def rank_by_feature(cells, feature_key, descending=True, top_n=None):
    """Rank cells by a numeric feature.

    Args:
        cells: List of cell dicts.
        feature_key: Key to sort by (e.g., 'peak', 'mean_intensity', 'area_px').
        descending: If True, highest values first.
        top_n: If given, return only top N cells.

    Returns:
        List of cell dicts, sorted by feature.
    """
    ranked = sorted(cells, key=lambda c: c.get(feature_key, 0),
                    reverse=descending)
    if top_n:
        ranked = ranked[:top_n]
    return ranked


# ---------------------------------------------------------------------------
# Zoom and measure
# ---------------------------------------------------------------------------

def zoom_and_measure(core, target_x, target_y, mag=40,
                     channels=('brightfield',),
                     detect_method='threshold', detect_kwargs=None,
                     filter_edge=True):
    """Move to target, switch to high mag, detect and measure cells.

    Args:
        core: Microscope core.
        target_x, target_y: World coordinates to center on.
        mag: Target magnification (20 or 40).
        channels: Channels to acquire. First is used for detection.
        detect_method: 'threshold' or 'membrane'.
        detect_kwargs: Extra kwargs for detection.
        filter_edge: If True, exclude cells touching the image edge.

    Returns:
        dict with:
            cells: list of cell dicts (world coords, features)
            images: dict of {channel: image}
            n_cells: count (after edge filtering)
            mag: magnification used
            stage: (x, y) actual stage position
    """
    dk = detect_kwargs or {}

    hw.move_to(core, target_x, target_y)
    hw.set_objective(core, mag)

    cfg = get_config(core)
    img_w, img_h = cfg.image_width, cfg.image_height
    sx, sy = hw.get_position(core)
    tile_images = {}
    detect_img = None

    for ch in channels:
        img = hw.snap(core, channel=ch)
        tile_images[ch] = img
        if detect_img is None:
            detect_img = img

    # Detect
    cells = []
    if detect_method == 'membrane':
        seg = segment_tissue(detect_img, use_otsu=True,
                             tissue_mode=dk.get('tissue_mode', True),
                             min_area=dk.get('min_area', 100))
        for c in seg['cells']:
            wx, wy = pixel_to_world(c['px'], c['py'], sx, sy, mag)
            cell = {
                'world_x': wx, 'world_y': wy,
                'area': c['area'], 'touches_edge': c['touches_edge'],
            }
            cells.append(cell)
    else:
        dets = detect_cells(detect_img,
                            threshold_sigma=dk.get('threshold_sigma', 2.5),
                            min_area_px=dk.get('min_area_px', 200),
                            fill_holes=dk.get('fill_holes', True))
        for c in dets:
            cx_px, cy_px = c['centroid_px']  # (col, row)
            wx, wy = pixel_to_world(cx_px, cy_px, sx, sy, mag)
            # Edge check: centroid > 50px from boundary
            edge = (cx_px < 50 or cx_px > img_w - 50 or cy_px < 50 or cy_px > img_h - 50)
            cell = {
                'world_x': wx, 'world_y': wy,
                'area_px': c['area_px'], 'peak': c['peak'],
                'mean_intensity': c['mean_intensity'],
                'touches_edge': edge,
            }
            cells.append(cell)

    # Measure in additional channels
    for ch in channels[1:]:
        ch_img = tile_images[ch]
        for cell in cells:
            px, py = world_to_pixel(cell['world_x'], cell['world_y'],
                                    sx, sy, mag)
            px, py = int(round(px)), int(round(py))
            r = int(30 * (mag / 10))  # scale measurement window with mag
            y0 = max(0, py - r)
            y1 = min(img_h, py + r)
            x0 = max(0, px - r)
            x1 = min(img_w, px + r)
            if y1 > y0 and x1 > x0:
                region = ch_img[y0:y1, x0:x1]
                key = ch.replace('-', '_') + '_mean'
                cell[key] = float(region.mean())

    # Filter edge cells
    if filter_edge:
        cells = [c for c in cells if not c.get('touches_edge', False)]

    return {
        'cells': cells,
        'images': tile_images,
        'n_cells': len(cells),
        'mag': mag,
        'stage': (sx, sy),
    }


# ---------------------------------------------------------------------------
# Full adaptive pipeline
# ---------------------------------------------------------------------------

def adaptive_survey(core, world_size=(1024, 1024),
                    survey_channels=('nucleus-channel',),
                    feature_key='mean_intensity',
                    zoom_mag=40, zoom_channels=('brightfield',),
                    zoom_detect='threshold', zoom_detect_kwargs=None,
                    cluster_radius=300, min_cluster_size=2,
                    survey_detect_kwargs=None):
    """Full adaptive pipeline: survey at current mag -> find bright ROI -> zoom -> measure.

    1. Survey at current magnification across all tiles
    2. Rank cells by feature_key, find spatial clusters of high-feature cells
    3. Move to best cluster center
    4. Zoom to zoom_mag, detect and measure

    Args:
        core: Microscope core.
        world_size: Sample extent in world pixels.
        survey_channels: Channels for survey. First used for detection.
        feature_key: Cell feature to rank by.
        zoom_mag: Magnification for detailed measurement.
        zoom_channels: Channels to acquire at high mag.
        zoom_detect: Detection method at high mag.
        zoom_detect_kwargs: Detection kwargs at high mag.
        cluster_radius: Spatial clustering radius (world px).
        min_cluster_size: Minimum cluster size.
        survey_detect_kwargs: Detection kwargs for survey.

    Returns:
        dict with:
            survey: survey_cells result
            clusters: list of cluster dicts
            best_cluster: the chosen cluster dict (or None)
            zoom: zoom_and_measure result (or None)
    """
    # Step 1: Survey
    survey = survey_cells(core, world_size=world_size,
                          channels=survey_channels,
                          detect_kwargs=survey_detect_kwargs)

    cells = survey['cells']
    if not cells:
        return {'survey': survey, 'clusters': [], 'best_cluster': None, 'zoom': None}

    # Step 2: Find intensity threshold via gap detection
    values = np.array([c.get(feature_key, 0) for c in cells])
    sorted_vals = np.sort(values)
    if len(sorted_vals) > 1:
        gaps = np.diff(sorted_vals)
        gap_idx = int(np.argmax(gaps))
        threshold = (sorted_vals[gap_idx] + sorted_vals[gap_idx + 1]) / 2
    else:
        threshold = float(sorted_vals[0]) - 1

    # Step 3: Cluster bright cells
    clusters = find_clusters(cells, radius=cluster_radius,
                             min_size=min_cluster_size,
                             feature_key=feature_key,
                             feature_threshold=threshold)

    # If no cluster, fall back to single brightest cell
    if not clusters:
        best = rank_by_feature(cells, feature_key, top_n=1)
        if best:
            clusters = [{
                'cells': best,
                'center_x': best[0]['world_x'],
                'center_y': best[0]['world_y'],
                'n_cells': 1,
            }]

    best_cluster = clusters[0] if clusters else None

    # Step 4: Zoom to best cluster
    zoom_result = None
    if best_cluster:
        zoom_result = zoom_and_measure(
            core, best_cluster['center_x'], best_cluster['center_y'],
            mag=zoom_mag, channels=zoom_channels,
            detect_method=zoom_detect, detect_kwargs=zoom_detect_kwargs,
        )

    # Restore 10x
    hw.set_objective(core, 10)

    return {
        'survey': survey,
        'clusters': clusters,
        'best_cluster': best_cluster,
        'zoom': zoom_result,
    }


# ---------------------------------------------------------------------------
# MDA-native adaptive survey
# ---------------------------------------------------------------------------

def adaptive_survey_mda(
    survey_positions: Sequence[tuple[float, float]],
    survey_mag: int = 10,
    zoom_mag: int = 40,
    survey_exposure: float = 50.0,
    zoom_exposure: float = 50.0,
    survey_channels: Optional[Sequence[str]] = None,
    zoom_channels: Optional[Sequence[str]] = None,
    feature_key: str = 'mean_intensity',
    cluster_radius: float = 300.0,
    min_cluster_size: int = 2,
    threshold_sigma: float = 2.5,
    min_area_px: int = 20,
    fill_holes: bool = True,
    dedup_dist: float = 35.0,
    filter_edge: bool = True,
    metadata: Optional[dict[str, Any]] = None,
):
    """Create an MDA-native adaptive survey workflow.

    Phase 1: Survey — yields events for each tile position. The on_frame
    callback detects cells and accumulates world-coordinate detections.

    Between phases: The generator analyzes accumulated detections,
    finds clusters, and selects the best ROI.

    Phase 2: Zoom — yields events at the selected ROI position with
    an objective switch (via CustomAction). The on_frame callback
    detects and measures cells at high magnification.

    Usage::

        positions = grid_positions(1024, 1024, 512, overlap_frac=0.0)
        gen, on_frame, state = adaptive_survey_mda(positions)
        mmc.mda.events.frameReady.connect(on_frame)
        mmc.mda.set_engine(MicroscopyEngine(mmc))
        mmc.mda.run(gen())
        # Results in state['zoom_cells'], state['best_cluster'], etc.

    Args:
        survey_positions: List of (x, y) positions for the survey.
        survey_mag: Magnification for survey phase.
        zoom_mag: Magnification for zoom phase.
        survey_exposure: Exposure for survey.
        zoom_exposure: Exposure for zoom.
        survey_channels: Channels for survey (first used for detection).
        zoom_channels: Channels for zoom acquisition.
        feature_key: Feature to rank cells by.
        cluster_radius: Clustering radius in world pixels.
        min_cluster_size: Minimum cluster size.
        threshold_sigma: Detection threshold.
        min_area_px: Minimum detection area.
        fill_holes: Fill holes in detections.
        dedup_dist: Deduplication distance (0 to skip).
        filter_edge: Filter edge cells at zoom mag.
        metadata: Extra metadata.

    Returns:
        Tuple of (event_generator, on_frame_callback, shared_state).
    """
    survey_channels = survey_channels or [None]
    zoom_channels = zoom_channels or [None]
    extra_meta = metadata or {}

    state: dict[str, Any] = {
        'phase': 'survey',
        'survey_cells': [],
        'survey_images_count': 0,
        'clusters': [],
        'best_cluster': None,
        'zoom_cells': [],
        'zoom_images_count': 0,
    }

    pixel_size_survey = 10.0 / survey_mag  # fallback; overridden by image shape in on_frame

    def event_generator():
        # Phase 1: Survey tiles
        for p_idx, (x, y) in enumerate(survey_positions):
            for c_idx, ch in enumerate(survey_channels):
                idx = {"p": p_idx, "phase": 0}
                if len(survey_channels) > 1:
                    idx["c"] = c_idx
                meta = {"phase": "survey", "mag": survey_mag, **extra_meta}
                kwargs = {
                    "index": idx, "exposure": survey_exposure,
                    "x_pos": x, "y_pos": y, "metadata": meta,
                }
                yield MDAEvent(**kwargs)

        # Between phases: analyze survey results
        cells = state['survey_cells']

        # Deduplicate
        if dedup_dist > 0 and len(cells) > 1:
            coords = [(c['world_x'], c['world_y']) for c in cells]
            unique = deduplicate_cells(coords, min_dist=dedup_dist)
            deduped = []
            used = set()
            for ux, uy in unique:
                best_i, best_d = -1, float('inf')
                for i, c in enumerate(cells):
                    if i in used:
                        continue
                    d = ((c['world_x'] - ux)**2 + (c['world_y'] - uy)**2)**0.5
                    if d < best_d:
                        best_d = d
                        best_i = i
                if best_i >= 0:
                    used.add(best_i)
                    deduped.append(cells[best_i])
            cells = deduped
            state['survey_cells'] = cells

        if not cells:
            return

        # Find clusters
        values = np.array([c.get(feature_key, 0) for c in cells])
        sorted_vals = np.sort(values)
        if len(sorted_vals) > 1:
            gaps = np.diff(sorted_vals)
            gap_idx = int(np.argmax(gaps))
            threshold = (sorted_vals[gap_idx] + sorted_vals[gap_idx + 1]) / 2
        else:
            threshold = float(sorted_vals[0]) - 1

        clusters = find_clusters(
            cells, radius=cluster_radius,
            min_size=min_cluster_size,
            feature_key=feature_key,
            feature_threshold=threshold,
        )

        if not clusters:
            ranked = rank_by_feature(cells, feature_key, top_n=1)
            if ranked:
                clusters = [{
                    'cells': ranked,
                    'center_x': ranked[0]['world_x'],
                    'center_y': ranked[0]['world_y'],
                    'n_cells': 1,
                }]

        state['clusters'] = clusters
        best = clusters[0] if clusters else None
        state['best_cluster'] = best

        if not best:
            return

        # Phase 2: Switch objective + zoom
        state['phase'] = 'zoom'

        # Emit objective switch action
        yield MDAEvent(
            action=CustomAction(
                name='switch_objective',
                data={'mag': zoom_mag},
            ),
            metadata={'action': 'switch_objective', 'mag': zoom_mag},
        )

        # Emit zoom acquisition events
        for c_idx, ch in enumerate(zoom_channels):
            idx = {"p": 0, "phase": 1}
            if len(zoom_channels) > 1:
                idx["c"] = c_idx
            meta = {"phase": "zoom", "mag": zoom_mag, **extra_meta}
            yield MDAEvent(
                index=idx, exposure=zoom_exposure,
                x_pos=best['center_x'], y_pos=best['center_y'],
                metadata=meta,
            )

    pixel_size_zoom = 10.0 / zoom_mag  # fallback; overridden by image shape in on_frame

    def on_frame(image, event, meta=None):
        if state['phase'] == 'survey':
            H, W = image.shape[:2]
            sx = event.x_pos if event.x_pos is not None else 0.0
            sy = event.y_pos if event.y_pos is not None else 0.0

            cells = detect_cells(
                image, threshold_sigma=threshold_sigma,
                min_area_px=min_area_px, fill_holes=fill_holes,
            )
            state['survey_images_count'] += 1

            for c in cells:
                cx_px, cy_px = c['centroid_px']
                wx = sx + (cx_px - W / 2) * pixel_size_survey
                wy = sy + (cy_px - H / 2) * pixel_size_survey
                cell = {
                    'world_x': round(wx, 1),
                    'world_y': round(wy, 1),
                    'area_px': c['area_px'],
                    'peak': c['peak'],
                    'mean_intensity': c['mean_intensity'],
                }
                state['survey_cells'].append(cell)

        elif state['phase'] == 'zoom':
            H, W = image.shape[:2]
            sx = event.x_pos if event.x_pos is not None else 0.0
            sy = event.y_pos if event.y_pos is not None else 0.0

            cells = detect_cells(
                image, threshold_sigma=threshold_sigma,
                min_area_px=max(min_area_px, 200),
                fill_holes=fill_holes,
            )
            state['zoom_images_count'] += 1

            for c in cells:
                cx_px, cy_px = c['centroid_px']
                wx = sx + (cx_px - W / 2) * pixel_size_zoom
                wy = sy + (cy_px - H / 2) * pixel_size_zoom
                edge = (cx_px < 50 or cx_px > W - 50
                        or cy_px < 50 or cy_px > H - 50)
                if filter_edge and edge:
                    continue
                cell = {
                    'world_x': round(wx, 1),
                    'world_y': round(wy, 1),
                    'area_px': c['area_px'],
                    'peak': c['peak'],
                    'mean_intensity': c['mean_intensity'],
                    'touches_edge': edge,
                }
                state['zoom_cells'].append(cell)

    return event_generator, on_frame, state
