"""Whole-area mosaic scanning with boundary search, focus, and cell stats.

End-to-end large-area tile scan for a real microscope driven through
pymmcore-plus / pymmcore-proxy:

    1. ``measure_illumination_footprint`` -- find the illuminated camera sub-region
       (e.g. a DMD/Mosaic field that does not fill the whole sensor).
    2. ``find_well_bounds``  -- search outward for the well wall in +X/-X/+Y/-Y
       and return a bounding box of the well interior.
    3. ``mosaic_scan``       -- tile the bbox, focus each tile (Nikon PFS with a
       software-autofocus fallback), snap one channel, crop to the illuminated
       footprint, and **stream each raw tile to disk** (memory-safe for hundreds
       of tiles).
    4. ``build_display_mosaic`` -- assemble a (optionally downscaled) mosaic from
       the on-disk tiles.
    5. ``analyze_mosaic``    -- segment every tile with a consistent global
       threshold, map per-tile centroids to mosaic-pixel and world-µm
       coordinates, de-duplicate cells in tile overlaps, and compute statistics
       (total count, cells per FOV, biggest/smallest cell + coordinates).

Design notes
------------
* Pixel size is passed **explicitly** everywhere — on rigs where the running
  config has no pixel-size calibration ``core.getPixelSizeUm()`` returns 0, so
  ``get_config``/``fov_size``/``pixel_to_world`` cannot be trusted for geometry.
* Tiles may be cropped to the illuminated footprint, which is generally NOT
  centred on the sensor. World-µm mapping therefore uses the crop origin and the
  full-sensor centre, while mosaic-pixel coordinates are self-consistent in the
  cropped frame.
* Acquisition order is serpentine (minimal stage travel) but tiles are written
  to disk named by their **row-major** index, so file index == row-major index
  == ``positions`` index == ``tile_origins`` index throughout analysis.
* The DMD/SLM must be lit (full field) before every snap on rigs where it gates
  the illumination — callers light it; ``mosaic_scan`` refreshes it per tile
  because the hold expires after the device ``ExposureTime``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..detection.cells import detect_cells
from ..hardware import core as hw
from ..hardware.config import get_config
from ..utils.image import auto_contrast
from .focus_map import predict_z, refine_focus_map
from .tiling import tile_positions

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _as_gray_float(img):
    """Return a 2D float64 view of an image (collapses RGB if needed)."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        from ..utils.image import to_grayscale

        return to_grayscale(arr)
    return arr.astype(np.float64)


def _crop(img, crop):
    """Crop ``img`` to ``crop=(x0, y0, w, h)``; return as-is if crop is None."""
    if crop is None:
        return img
    x0, y0, w, h = crop
    return img[y0 : y0 + h, x0 : x0 + w]


def _snap(core, channel, exposure, crop=None, *, retries=4, refresh_dmd=True):
    """Snap one (optionally cropped) frame, retrying transient camera errors.

    This rig occasionally returns "Camera image buffer read failed"; a short
    pause + DMD refresh + re-snap recovers it.
    """
    last = None
    for attempt in range(retries):
        try:
            if refresh_dmd:
                hw.dmd_on(core)
            return _crop(hw.snap(core, channel=channel, exposure=exposure), crop)
        except Exception as e:  # noqa: BLE001 — transient hardware/proxy errors
            last = e
            time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"snap failed after {retries} retries: {last}")


def _fov_wh(fov_um):
    """Normalise a scalar or (w, h) FOV spec to (fov_w, fov_h) floats."""
    if isinstance(fov_um, int | float):
        return float(fov_um), float(fov_um)
    return float(fov_um[0]), float(fov_um[1])


def _edge_energy(g):
    """Mean Sobel gradient magnitude — a texture/edge sharpness proxy."""
    from scipy.ndimage import sobel

    gx = sobel(g, axis=1)
    gy = sobel(g, axis=0)
    return float(np.sqrt(gx * gx + gy * gy).mean())


# ---------------------------------------------------------------------------
# Illumination footprint (DMD/Mosaic field smaller than the sensor)
# ---------------------------------------------------------------------------


def measure_illumination_footprint(
    core, channel, *, exposure=None, n=11, jitter_um=250.0, frac=0.2, blur_sigma=15.0, pfs=True
):
    """Find the illuminated camera sub-region by per-pixel variance over moves.

    Snaps ``n`` frames while jittering XY; pixels inside the illuminated field
    vary as cells come and go (high std), pixels outside stay at baseline
    (~zero std). Returns the bounding box where the smoothed column/row std
    profiles exceed ``frac`` of their peak.

    Returns:
        dict: crop=(x0, y0, w, h), bbox=(x0, y0, x1, y1), full_shape=(H, W),
        std (the projection, for inspection).
    """
    import cv2

    x0c, y0c = hw.get_position(core)
    offs = [(0, 0)]
    r = jitter_um
    ring = [
        (-r, -r),
        (0, -r),
        (r, -r),
        (-r, 0),
        (r, 0),
        (-r, r),
        (0, r),
        (r, r),
        (r / 2, -r / 2),
        (-r / 2, r / 2),
    ]
    offs += ring[: max(0, n - 1)]

    stack = []
    for dx, dy in offs:
        core.setXYPosition(x0c + dx, y0c + dy)
        core.waitForDevice(core.getXYStageDevice())
        if pfs:
            pfs_settle(core, 1.5)
        stack.append(_snap(core, channel, exposure).astype(np.float32))
    core.setXYPosition(x0c, y0c)
    core.waitForDevice(core.getXYStageDevice())
    if pfs:
        pfs_settle(core, 1.5)

    std = np.stack(stack).std(0)
    sm = cv2.GaussianBlur(std, (0, 0), blur_sigma)
    col, row = sm.mean(0), sm.mean(1)
    xs = np.where(col > frac * col.max())[0]
    ys = np.where(row > frac * row.max())[0]
    H, W = std.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {
        "crop": (x0, y0, x1 - x0, y1 - y0),
        "bbox": (x0, y0, x1, y1),
        "full_shape": (H, W),
        "std": std,
    }


# ---------------------------------------------------------------------------
# Well-wall detection
# ---------------------------------------------------------------------------


def wall_baseline(images):
    """Characterise interior (cell-covered) tiles for wall comparison."""
    meds, p99s, edges = [], [], []
    for im in images:
        g = _as_gray_float(im)
        meds.append(float(np.median(g)))
        p99s.append(float(np.percentile(g, 99)))
        edges.append(_edge_energy(g))
    return {
        "median": float(np.median(meds)),
        "p99": float(np.median(p99s)),
        "edge_energy": float(np.median(edges)),
    }


def detect_well_wall(img, baseline, *, bright_frac_thresh=0.04, edge_ratio_thresh=3.0):
    """Heuristically decide whether an image shows a well wall / edge artifact.

    Combines two cues vs an interior ``baseline``:
      * ``bright_frac`` -- fraction of pixels far brighter than the interior
        99th percentile (a bright plastic wall floods part of the FOV).
      * ``edge_ratio``  -- Sobel edge energy relative to baseline (a sharp wall
        edge spikes gradient energy).

    NOTE: pass a footprint-cropped image; the dark illumination edge would
    otherwise read as a wall.
    """
    g = _as_gray_float(img)
    hi = max(baseline.get("p99", 0.0), 1.0) * 1.8
    bright_frac = float((g > hi).mean())
    edge_ratio = _edge_energy(g) / max(baseline.get("edge_energy", 1e-6), 1e-6)

    reasons = []
    is_wall = False
    if bright_frac > bright_frac_thresh:
        is_wall = True
        reasons.append(f"bright_frac={bright_frac:.3f}>{bright_frac_thresh}")
    if edge_ratio > edge_ratio_thresh:
        is_wall = True
        reasons.append(f"edge_ratio={edge_ratio:.2f}>{edge_ratio_thresh}")

    conf = min(
        1.0, 0.5 * (bright_frac / bright_frac_thresh) + 0.5 * (edge_ratio / edge_ratio_thresh)
    )
    return {
        "is_wall": is_wall,
        "confidence": round(float(conf), 3),
        "reason": "; ".join(reasons) or "interior",
        "bright_frac": round(bright_frac, 4),
        "edge_ratio": round(edge_ratio, 3),
        "median": round(float(np.median(g)), 2),
    }


# ---------------------------------------------------------------------------
# Perfect Focus System (PFS) helpers
# ---------------------------------------------------------------------------


def pfs_available(core):
    """True if the core exposes a hardware continuous-autofocus device."""
    try:
        return bool(core.getAutoFocusDevice())
    except Exception:
        return False


def pfs_settle(core, timeout_s=1.5, poll_s=0.05):
    """Poll until continuous focus reports locked, or timeout. Returns bool."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if core.isContinuousFocusLocked():
                return True
        except Exception:
            return False
        time.sleep(poll_s)
    try:
        return bool(core.isContinuousFocusLocked())
    except Exception:
        return False


def _software_autofocus(core, channel, exposure, *, z_range=15.0, coarse_step=2.0, fine_step=0.5):
    """Relative coarse+fine software autofocus around the current Z.

    Sweeps relative to the current Z (not absolute [-z_range, +z_range] like
    ``autofocus.coarse_fine_focus``), so it works on real stands where Z is e.g.
    ~4600 µm. Continuous focus is temporarily disabled during the sweep so
    manual Z moves take hold, then re-enabled.
    """
    from .autofocus import sweep_focus

    cf_was_on = False
    try:
        cf_was_on = bool(core.isContinuousFocusEnabled())
        if cf_was_on:
            core.enableContinuousFocus(False)
    except Exception:
        cf_was_on = False

    if exposure is not None:
        core.setExposure(float(exposure))
    hw.dmd_on(core)
    z0 = hw.get_z(core)
    coarse = sweep_focus(
        core, z0 - z_range, z0 + z_range, coarse_step, channel=channel, method="brenner"
    )
    zc = coarse["best_z"]
    hw.dmd_on(core)
    fine_half = coarse_step * 1.5
    fine = sweep_focus(
        core, zc - fine_half, zc + fine_half, fine_step, channel=channel, method="brenner"
    )
    hw.set_z(core, fine["best_z"])

    if cf_was_on:
        try:
            core.enableContinuousFocus(True)
            pfs_settle(core, timeout_s=1.0)
        except Exception:
            pass
    return float(fine["best_z"])


def _focus_tile(
    core, use_pfs, channel, exposure, settle_s, af_fallback=True, focus_map=None, xy=None
):
    """Focus the current tile. Returns (method, z).

    Order of preference:
      1. ``pfs``       -- PFS hardware lock. If a ``focus_map`` is given, the
         locked (x, y, z) is fed back via ``refine_focus_map`` so the fitted
         plane tracks the coverslip tilt live.
      2. ``predicted`` -- where PFS won't lock (well edge / off the coverslip),
         drive Z to the focus-map plane prediction instead of holding a stale Z.
      3. ``software``  -- relative software-AF sweep (only if ``af_fallback``).
      4. ``noaf``      -- keep the current Z (last resort).

    The stale-Z hold that wrecked the 2026-05-21 scan corresponds to
    ``focus_map=None, af_fallback=False`` (``noaf``); pass a calibrated
    ``focus_map`` so unlocked tiles get a predicted plane Z instead.
    """
    if use_pfs:
        # A previous "predicted" tile may have switched continuous focus off to
        # move Z manually; make sure it is back on before trying to lock.
        try:
            if not core.isContinuousFocusEnabled():
                core.enableContinuousFocus(True)
        except Exception:
            pass
        if pfs_settle(core, timeout_s=settle_s):
            z = float(hw.get_z(core))
            if focus_map is not None and xy is not None:
                try:
                    refine_focus_map(focus_map, xy[0], xy[1], z)
                except Exception:
                    pass
            return "pfs", z
    if focus_map is not None and xy is not None and focus_map.get("fit_type"):
        try:
            zp = float(predict_z(focus_map, xy[0], xy[1]))
            if use_pfs:
                try:
                    core.enableContinuousFocus(False)
                except Exception:
                    pass
            hw.set_z(core, zp)
            return "predicted", zp
        except Exception:
            pass
    if af_fallback:
        return "software", _software_autofocus(core, channel, exposure)
    return "noaf", float(hw.get_z(core))


# ---------------------------------------------------------------------------
# Boundary search
# ---------------------------------------------------------------------------


def find_well_bounds(
    core,
    *,
    fov_um,
    channel,
    exposure=None,
    max_search_um=9000.0,
    settle_s=1.0,
    use_pfs=True,
    crop=None,
    save_dir=None,
    wall_kwargs=None,
):
    """Search outward for the well wall in 4 directions; return interior bbox.

    Steps one (footprint) FOV at a time from the current stage center along +X,
    -X (by fov_w), +Y, -Y (by fov_h) until ``detect_well_wall`` fires (or
    ``max_search_um`` is reached). The bbox is formed from the last *interior*
    probe center per direction. Probe images (footprint-cropped) are saved if
    ``save_dir`` is given.

    Returns:
        dict: bbox, edges, center, baseline, probe_log.
    """
    wall_kwargs = wall_kwargs or {}
    cfg = get_config(core)
    xy = cfg.xy_device
    cx, cy = hw.get_position(core)
    fov_w, fov_h = _fov_wh(fov_um)

    if use_pfs:
        pfs_settle(core, timeout_s=settle_s)
    base_img = _snap(core, channel, exposure, crop)
    baseline = wall_baseline([base_img])

    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        from ..utils.diagnostics import save_snapshot

    directions = {
        "+x": (1, 0, fov_w),
        "-x": (-1, 0, fov_w),
        "+y": (0, 1, fov_h),
        "-y": (0, -1, fov_h),
    }
    edges = {"+x": cx, "-x": cx, "+y": cy, "-y": cy}
    probe_log = []

    for name, (dx, dy, step) in directions.items():
        d = step
        last_interior = (cx, cy)
        while d <= max_search_um:
            x, y = cx + dx * d, cy + dy * d
            core.setXYPosition(x, y)
            if xy:
                core.waitForDevice(xy)
            locked = pfs_settle(core, timeout_s=settle_s) if use_pfs else False
            img = _snap(core, channel, exposure, crop)
            wall = detect_well_wall(img, baseline, **wall_kwargs)
            probe_log.append(
                {
                    "dir": name,
                    "dist_um": round(d, 1),
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "pfs_locked": bool(locked),
                    **wall,
                }
            )
            if save_dir:
                save_snapshot(img, f"probe_{name}_{int(round(d))}um", save_dir=save_dir)
            if wall["is_wall"]:
                break
            last_interior = (x, y)
            d += step
        edges[name] = last_interior[0] if name in ("+x", "-x") else last_interior[1]

    core.setXYPosition(cx, cy)
    if xy:
        core.waitForDevice(xy)
    if use_pfs:
        pfs_settle(core, timeout_s=settle_s)

    x_min, x_max = min(edges["-x"], edges["+x"]), max(edges["-x"], edges["+x"])
    y_min, y_max = min(edges["-y"], edges["+y"]), max(edges["-y"], edges["+y"])
    return {
        "bbox": (x_min, y_min, x_max, y_max),
        "edges": edges,
        "center": (cx, cy),
        "baseline": baseline,
        "probe_log": probe_log,
    }


# ---------------------------------------------------------------------------
# Mosaic acquisition (streams cropped tiles to disk)
# ---------------------------------------------------------------------------


def _serpentine_order(n_cols, n_rows):
    """Row-major indices in serpentine acquisition order (minimal travel)."""
    order = []
    for r in range(n_rows):
        cols = range(n_cols) if r % 2 == 0 else reversed(range(n_cols))
        for c in cols:
            order.append(r * n_cols + c)
    return order


def mosaic_scan(
    core,
    bbox,
    *,
    pixel_size,
    fov_um,
    channel,
    group=None,
    exposure=None,
    overlap=0.1,
    use_pfs=True,
    settle_s=1.0,
    tiles_dir,
    crop=None,
    dmd_refresh=True,
    af_fallback=True,
    focus_map=None,
    on_tile=None,
):
    """Tile ``bbox``, focus + snap each tile via MDA, crop, and stream to disk.

    Acquisition is MDA-native: an adaptive generator moves to each tile, focuses
    it (PFS / focus-map predicted-Z / software-AF — see ``_focus_tile``), lights
    the DMD, then yields an ``MDAEvent`` (channel + exposure, no position) that
    ``run_events`` acquires. Each cropped tile is written to
    ``tiles_dir/tile_####.tif`` by the ``on_frame`` callback and freed
    immediately (``collect=False`` keeps RAM flat for hundreds of tiles); files
    are named by **row-major** index == ``positions`` index == ``tile_origins``
    index.

    ``group`` is the channel config group, embedded in each event's channel spec.
    Pass it explicitly — auto-resolution is unreliable on rigs whose first config
    group is e.g. 'Binning'.

    Returns:
        dict: positions, grid_shape=(n_cols,n_rows), n_tiles, tile_origins,
        tile_shape=(H,W) of cropped tiles, step_px=(step_x,step_y),
        mosaic_full_shape, crop, focus_log, plus scan parameters.
    """
    import tifffile
    from useq import MDAEvent

    tiles_dir = Path(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    fov_w, fov_h = _fov_wh(fov_um)

    grid = tile_positions(bbox, (fov_w, fov_h), overlap=overlap, serpentine=False)
    positions = grid["positions"]
    n_cols, n_rows = grid["grid_shape"]
    n_tiles = grid["n_tiles"]

    cfg = get_config(core)
    xy = cfg.xy_device
    focus_log = []
    shape = {"hw": None}

    ch_spec = None
    if channel is not None:
        ch_spec = {"config": channel}
        if group is not None:
            ch_spec["group"] = group

    def _write_frame(image, event):
        md = dict(getattr(event, "metadata", {}) or {})
        idx = int(md["idx"])
        img = _crop(np.asarray(image), crop)
        shape["hw"] = img.shape[:2]
        tifffile.imwrite(tiles_dir / f"tile_{idx:04d}.tif", img, compression="zlib")
        focus_log.append(
            {
                "idx": idx,
                "method": md["method"],
                "z": round(float(md["z"]), 3),
                "xy": (round(md["x"], 2), round(md["y"], 2)),
            }
        )
        if on_tile is not None:
            on_tile(int(md["k"]), idx, img, md["x"], md["y"])

    def _events():
        for k, idx in enumerate(_serpentine_order(n_cols, n_rows)):
            x, y = positions[idx]
            core.setXYPosition(x, y)
            if xy:
                core.waitForDevice(xy)
            method, z = _focus_tile(
                core,
                use_pfs,
                channel,
                exposure,
                settle_s,
                af_fallback=af_fallback,
                focus_map=focus_map,
                xy=(x, y),
            )
            if dmd_refresh:
                hw.dmd_on(core)
            yield MDAEvent(
                channel=ch_spec,
                exposure=exposure,
                metadata={
                    "idx": int(idx),
                    "k": int(k),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "method": method,
                },
            )

    hw.run_events(core, _events(), on_frame=_write_frame, collect=False)

    H, W = shape["hw"]
    step_x = int(round(W * (1 - overlap)))
    step_y = int(round(H * (1 - overlap)))
    tile_origins = [(c * step_x, r * step_y) for r in range(n_rows) for c in range(n_cols)]
    full_w = (n_cols - 1) * step_x + W
    full_h = (n_rows - 1) * step_y + H

    return {
        "positions": positions,
        "grid_shape": (n_cols, n_rows),
        "n_tiles": n_tiles,
        "tile_origins": tile_origins,
        "tile_shape": (H, W),
        "step_px": (step_x, step_y),
        "mosaic_full_shape": (full_h, full_w),
        "crop": crop,
        "focus_log": focus_log,
        "tiles_dir": str(tiles_dir),
        "pixel_size": pixel_size,
        "fov_um": (fov_w, fov_h),
        "overlap": overlap,
    }


# ---------------------------------------------------------------------------
# Display mosaic
# ---------------------------------------------------------------------------


def build_display_mosaic(
    tiles_dir, grid_shape, tile_shape, step_px, *, max_dim=12000, row_order="topdown"
):
    """Stitch on-disk tiles into a (optionally downscaled) mosaic.

    Placement uses per-axis ``step_px`` (so rectangular tiles / per-axis overlap
    work). Overlaps are blended with ``np.maximum``.

    ``row_order`` controls the vertical placement of grid rows (tiles are NEVER
    flipped internally — that would break seams):
      * ``"topdown"``  -- row 0 at the top (image-array convention).
      * ``"bottomup"`` -- row 0 at the bottom, i.e. world-Y increasing upward.
        Use this when ``tile_positions`` row 0 is the smallest world-Y and you
        want a true spatial map (correct for the Nikon Ti rig).

    Returns:
        dict: mosaic (2D, tile dtype), scale (<=1.0), full_shape (H, W).
    """
    import cv2
    import tifffile

    tiles_dir = Path(tiles_dir)
    n_cols, n_rows = grid_shape
    H, W = tile_shape
    step_x, step_y = step_px
    full_w = (n_cols - 1) * step_x + W
    full_h = (n_rows - 1) * step_y + H

    scale = min(1.0, float(max_dim) / float(max(full_w, full_h)))
    dW, dH = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    dsx, dsy = int(round(step_x * scale)), int(round(step_y * scale))
    mos_w = (n_cols - 1) * dsx + dW
    mos_h = (n_rows - 1) * dsy + dH

    dtype = tifffile.imread(tiles_dir / "tile_0000.tif").dtype
    mosaic = np.zeros((mos_h, mos_w), dtype=dtype)
    for idx in range(n_cols * n_rows):
        t = tifffile.imread(tiles_dir / f"tile_{idx:04d}.tif")
        if scale < 1.0:
            t = cv2.resize(t, (dW, dH), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx, n_cols)
        rr = (n_rows - 1 - r) if row_order == "bottomup" else r
        oy, ox = rr * dsy, c * dsx
        region = mosaic[oy : oy + dH, ox : ox + dW]
        mosaic[oy : oy + dH, ox : ox + dW] = np.maximum(
            region, t[: region.shape[0], : region.shape[1]]
        )
    return {"mosaic": mosaic, "scale": scale, "full_shape": (full_h, full_w)}


# ---------------------------------------------------------------------------
# Analysis: segmentation, coordinate mapping, dedup, statistics
# ---------------------------------------------------------------------------


def _dedup_cells(all_cells, tile_shape, min_dist_um):
    """Cluster cells within ``min_dist_um`` (world µm); keep a representative.

    KDTree + union-find (O(n log n)). Representative kept per cluster is the
    detection whose centroid is closest to its own tile center (least edge
    distortion), preserving its measured area.
    """
    if len(all_cells) < 2:
        return list(all_cells)

    from scipy.spatial import cKDTree

    H, W = tile_shape
    cx0, cy0 = W / 2.0, H / 2.0
    pts = np.array([c["world_um"] for c in all_cells], dtype=float)
    parent = list(range(len(all_cells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in cKDTree(pts).query_pairs(r=float(min_dist_um)):
        union(a, b)

    clusters = {}
    for i in range(len(all_cells)):
        clusters.setdefault(find(i), []).append(i)

    deduped = []
    for members in clusters.values():
        if len(members) == 1:
            deduped.append(all_cells[members[0]])
            continue
        best, best_d = None, None
        for i in members:
            ccx, ccy = all_cells[i]["centroid_px"]
            d = (ccx - cx0) ** 2 + (ccy - cy0) ** 2
            if best_d is None or d < best_d:
                best_d, best = d, i
        deduped.append(all_cells[best])
    return deduped


def _nonoverlap_count(all_cells, grid_shape, step_px):
    """Cross-check: cells in each tile's non-overlapping owned region."""
    n_cols, n_rows = grid_shape
    step_x, step_y = step_px
    count = 0
    for c in all_cells:
        r, col = divmod(c["tile_idx"], n_cols)
        cx, cy = c["centroid_px"]
        ok_x = (col == n_cols - 1) or (cx < step_x)
        ok_y = (r == n_rows - 1) or (cy < step_y)
        if ok_x and ok_y:
            count += 1
    return count


def analyze_mosaic(
    tiles_dir,
    positions,
    grid_shape,
    tile_origins,
    *,
    pixel_size,
    step_px,
    tile_shape,
    crop_origin=(0, 0),
    cam_center=None,
    threshold_sigma=2.5,
    min_area_px=30,
    max_area_px=None,
    dedup_world_dist_um=8.0,
    border_margin_px=3,
    detect_fn=None,
):
    """Segment every tile, map coordinates, de-duplicate, and compute stats.

    Two passes over the on-disk tiles keep memory flat: pass 1 accumulates a
    pooled (mean, std) for a consistent detection threshold; pass 2 runs
    ``detect_cells`` with that ``global_stats`` and maps centroids to:
      * mosaic-pixel = tile_origin + centroid (cropped frame, self-consistent),
      * world-µm     = stage_xy + (crop_origin + centroid - cam_center)*px.

    ``cam_center`` defaults to the cropped-tile center (i.e. crop_origin=(0,0));
    pass the full-sensor center (e.g. (1024,1024)) with the real crop_origin to
    get absolute world coordinates when tiles are cropped off-center.

    Returns:
        dict with cells (deduped), n_total, raw_detections, nonoverlap_count,
        per_tile_counts, cells_per_fov_{mean,std,min,max}, biggest, smallest,
        global_stats, n_tiles.
    """
    import tifffile

    tiles_dir = Path(tiles_dir)
    n = len(positions)
    H, W = tile_shape
    crop_x0, crop_y0 = crop_origin
    cam_cx, cam_cy = cam_center if cam_center is not None else (W / 2.0, H / 2.0)

    # Pass 1: pooled mean/std (running sums) for a consistent sigma threshold.
    # Skipped when a custom ``detect_fn`` is supplied (e.g. cellpose), which does
    # not need a global threshold.
    gmean = gstd = 0.0
    if detect_fn is None:
        s = s2 = 0.0
        cnt = 0
        for idx in range(n):
            t = tifffile.imread(tiles_dir / f"tile_{idx:04d}.tif").astype(np.float64)
            s += float(t.sum())
            s2 += float((t * t).sum())
            cnt += t.size
            del t
        gmean = s / cnt
        gstd = float(np.sqrt(max(s2 / cnt - gmean * gmean, 0.0)))

    # Pass 2: detect + map.
    all_cells = []
    per_tile_counts = [0] * n
    for idx in range(n):
        t = tifffile.imread(tiles_dir / f"tile_{idx:04d}.tif")
        if detect_fn is not None:
            cells = detect_fn(t)
        else:
            cells = detect_cells(
                t,
                threshold_sigma=threshold_sigma,
                min_area_px=min_area_px,
                max_area_px=max_area_px,
                pixel_size_um=pixel_size,
                fill_holes=True,
                global_stats=(gmean, gstd),
            )
        per_tile_counts[idx] = len(cells)
        ox, oy = tile_origins[idx]
        sx, sy = positions[idx]
        for c in cells:
            cx, cy = c["centroid_px"]
            bx, by, bw, bh = c["bbox"]
            is_border = (
                bx <= border_margin_px
                or by <= border_margin_px
                or bx + bw >= W - border_margin_px
                or by + bh >= H - border_margin_px
            )
            all_cells.append(
                {
                    "tile_idx": int(idx),
                    "centroid_px": (float(cx), float(cy)),
                    "mosaic_px": (float(ox + cx), float(oy + cy)),
                    "world_um": (
                        float(sx + (crop_x0 + cx - cam_cx) * pixel_size),
                        float(sy + (crop_y0 + cy - cam_cy) * pixel_size),
                    ),
                    "area_px": int(c["area_px"]),
                    "area_um2": float(c["area_um2"]),
                    "bbox": (int(bx), int(by), int(bw), int(bh)),
                    "circularity": float(c["circularity"]),
                    "solidity": float(c["solidity"]),
                    "eccentricity": float(c["eccentricity"]),
                    "is_border": bool(is_border),
                }
            )
        del t

    deduped = _dedup_cells(all_cells, tile_shape, dedup_world_dist_um)
    nonoverlap = _nonoverlap_count(all_cells, grid_shape, step_px)

    counts = np.array(per_tile_counts, dtype=float)
    pool = [c for c in deduped if not c["is_border"]] or deduped
    biggest = max(pool, key=lambda c: c["area_um2"]) if pool else None
    smallest = min(pool, key=lambda c: c["area_um2"]) if pool else None

    return {
        "cells": deduped,
        "n_total": len(deduped),
        "raw_detections": len(all_cells),
        "nonoverlap_count": int(nonoverlap),
        "per_tile_counts": per_tile_counts,
        "cells_per_fov_mean": float(counts.mean()) if len(counts) else 0.0,
        "cells_per_fov_std": float(counts.std()) if len(counts) else 0.0,
        "cells_per_fov_min": int(counts.min()) if len(counts) else 0,
        "cells_per_fov_max": int(counts.max()) if len(counts) else 0,
        "biggest": biggest,
        "smallest": smallest,
        "global_stats": (round(gmean, 3), round(gstd, 3)),
        "n_tiles": n,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_mosaic_outputs(mosaic, out_dir, *, basename="mosaic"):
    """Save a raw TIF and an auto-contrast PNG of the mosaic. Returns paths."""
    import cv2
    import tifffile

    out_dir = Path(out_dir)
    tif_path = out_dir / f"{basename}.tif"
    png_path = out_dir / f"{basename}.png"
    tifffile.imwrite(tif_path, mosaic)
    cv2.imwrite(str(png_path), auto_contrast(mosaic))
    return {"tif": tif_path, "png": png_path}
