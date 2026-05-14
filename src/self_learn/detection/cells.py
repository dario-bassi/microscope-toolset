"""Cell detection module for smart microscopy.

Provides detection functions that work through standard pymmcore-plus API.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import binary_fill_holes


def detect_cells(image, threshold_sigma=2.5, min_area_px=30, max_area_px=None,
                 pixel_size_um=1.0, fill_holes=True, global_stats=None,
                 backend="sigma"):
    """Detect cells in a grayscale image using adaptive thresholding.

    Args:
        image: 2D grayscale numpy array.
        threshold_sigma: Std devs above mean for threshold.
        min_area_px: Minimum connected component area.
        max_area_px: Maximum connected component area. Use to filter out
            diffuse glow (e.g., gradient fields) when detecting small objects.
        pixel_size_um: Pixel size for area conversion.
        fill_holes: Fill holes in detections. Essential for brightfield
            where cells appear as rings (bright membrane, dark interior).
        global_stats: Optional (mean, std) for consistent thresholding
            across a stack. ALWAYS use this for multi-frame analysis.
        backend: ``"sigma"`` (default — preserves all existing behaviour),
            ``"cellpose"``, ``"stardist"``, ``"auto"``, or any backend
            registered via ``segmentation_backend.register_backend``.
            Non-sigma backends route through
            ``segmentation_backend.segment(...)``; the centroid-dict
            output shape is preserved by the
            ``labels_to_centroid_dicts`` adapter, but recipe-specific
            keys (``circularity``, ``solidity``, ``eccentricity``,
            ``intensity_std``) are populated only on the sigma path
            for now. Sprint-#10 migration target.

    Returns:
        List of dicts sorted by area (largest first):
        {centroid_px, area_px, peak, mean_intensity, bbox,
         circularity, solidity, eccentricity, intensity_std}
    """
    if backend != "sigma":
        # Route through the pluggable backend dispatcher and adapt the
        # output shape. Falls back to sigma when the requested backend
        # is unavailable (auto-mode); cleanest opt-in for recipes that
        # want Cellpose / StarDist without the migration risk of
        # changing the rest of the function.
        from .segmentation_backend import labels_to_centroid_dicts, segment

        mask = segment(np.asarray(image, dtype=float), backend=backend,
                       sigma=threshold_sigma, min_area=min_area_px)
        cells = labels_to_centroid_dicts(mask, image=np.asarray(image, dtype=float))
        # Carry over the per-cell area filter even when the backend
        # ignored ``min_area``.
        cells = [c for c in cells if c["area_px"] >= int(min_area_px)]
        if max_area_px is not None:
            cells = [c for c in cells if c["area_px"] <= int(max_area_px)]
        return sorted(cells, key=lambda c: c["area_px"], reverse=True)

    img = image.astype(np.float32)

    if global_stats is not None:
        mean_val, std_val = global_stats
    else:
        mean_val = img.mean()
        std_val = img.std()

    if std_val < 1e-6:
        return []

    thresh = mean_val + threshold_sigma * std_val
    binary = (img > thresh).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cells = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        if max_area_px is not None and area > max_area_px:
            continue

        region_mask = labels == i
        if fill_holes:
            region_mask = binary_fill_holes(region_mask)
            area = int(region_mask.sum())

        cx, cy = centroids[i]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        region_vals = img[region_mask]

        # Shape descriptors for morphology classification
        contours, _ = cv2.findContours(
            region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        circularity = 0.0
        solidity = 0.0
        eccentricity = 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                axes = ellipse[1]
                if axes[0] > 0:
                    eccentricity = 1.0 - min(axes) / max(axes)

        cells.append({
            "centroid_px": (float(cx), float(cy)),
            "area_px": int(area),
            "area_um2": float(area * pixel_size_um ** 2),
            "peak": float(region_vals.max()),
            "mean_intensity": float(region_vals.mean()),
            "intensity_std": float(region_vals.std()),
            "bbox": (int(x), int(y), int(w), int(h)),
            "circularity": float(circularity),
            "solidity": float(solidity),
            "eccentricity": float(eccentricity),
        })

    cells.sort(key=lambda c: c["area_px"], reverse=True)
    return cells


def detect_cells_multichannel(core, brightfield_config="brightfield",
                              nucleus_config="nucleus-channel",
                              config_group=None,
                              threshold_sigma=2.5, min_area_px=20):
    """Detect cells using nucleus fluorescence for clean separation.

    Nucleus channel gives one bright spot per cell — no rings, no clumps.
    This is the preferred method when fluorescence is available.

    Args:
        core: pymmcore-plus core instance.
        brightfield_config: Config name for brightfield channel.
        nucleus_config: Config name for nucleus fluorescence.
        config_group: Config group name. If None, auto-discovers from core.
        threshold_sigma: Detection threshold.
        min_area_px: Minimum detection area.

    Returns:
        List of cell dicts (same format as detect_cells).
    """
    if config_group is None:
        from ..hardware.config import get_config
        cfg = get_config(core)
        config_group = cfg.channel_group
    if config_group is not None:
        core.setConfig(config_group, nucleus_config)
    core.snapImage()
    img = core.getImage()
    return detect_cells(img, threshold_sigma=threshold_sigma,
                        min_area_px=min_area_px, fill_holes=False)


def find_bright_centroid(image, threshold_sigma=2.5, window=64):
    """Find centroid of brightest region in image.

    Uses local windowed search around peak pixel for robustness.

    Returns:
        (cy, cx, area_px, peak) or (None, None, 0, peak) if nothing found.
    """
    img = image.astype(np.float32)
    mean_val = img.mean()
    std_val = img.std()
    peak = float(img.max())

    thresh = mean_val + threshold_sigma * std_val
    if peak <= thresh:
        return None, None, 0, peak

    h, w = img.shape[:2]
    peak_idx = img.argmax()
    peak_y, peak_x = divmod(int(peak_idx), w)

    y0 = max(0, peak_y - window)
    y1 = min(h, peak_y + window)
    x0 = max(0, peak_x - window)
    x1 = min(w, peak_x + window)
    patch = img[y0:y1, x0:x1]

    local_mask = patch > thresh
    area_px = int(local_mask.sum())
    if area_px == 0:
        return None, None, 0, peak

    local_ys, local_xs = np.where(local_mask)
    weights = patch[local_mask] - thresh
    total_w = weights.sum()
    if total_w <= 0:
        return None, None, 0, peak

    cy = float((local_ys * weights).sum() / total_w) + y0
    cx = float((local_xs * weights).sum() / total_w) + x0
    return cy, cx, area_px, peak


def count_blobs_log(image, sigma=4, peak_thresh=1.0, min_intensity=0):
    """Count blobs using Laplacian of Gaussian (LoG) detection.

    Better than connected-component counting for dense/touching nuclei
    because LoG finds individual peaks even when blobs merge at threshold.

    Args:
        image: 2D grayscale array.
        sigma: LoG scale parameter. Match to blob radius:
            - Compact nuclei (r~8px): sigma=4
            - Diffuse nuclei (r~12px): sigma=6
        peak_thresh: Minimum LoG response for a peak. Higher = fewer blobs.
        min_intensity: Minimum raw intensity at peak location.

    Returns:
        n_blobs: Number of detected blobs.
    """
    img = image.astype(np.float64)
    log = -ndimage.gaussian_laplace(img, sigma=sigma)
    local_max = ndimage.maximum_filter(log, size=2 * sigma + 1)
    peaks = (log == local_max) & (log > peak_thresh)
    if min_intensity > 0:
        peaks = peaks & (img > min_intensity)
    _, n = ndimage.label(peaks)
    return n


def detect_blobs_log(image, sigma=4, peak_thresh=1.0, min_intensity=0):
    """Detect blobs using LoG and return their centroids.

    Args:
        image: 2D grayscale array.
        sigma: LoG scale parameter (see count_blobs_log).
        peak_thresh: Minimum LoG response for a peak.
        min_intensity: Minimum raw intensity at peak location.

    Returns:
        List of dicts with 'centroid_px' (cx, cy) and 'log_response'.
    """
    img = image.astype(np.float64)
    log = -ndimage.gaussian_laplace(img, sigma=sigma)
    local_max = ndimage.maximum_filter(log, size=2 * sigma + 1)
    peaks = (log == local_max) & (log > peak_thresh)
    if min_intensity > 0:
        peaks = peaks & (img > min_intensity)
    labeled, n = ndimage.label(peaks)
    blobs = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        cy = float(ys.mean())
        cx = float(xs.mean())
        resp = float(log[ys[0], xs[0]])
        blobs.append({
            'centroid_px': (cx, cy),
            'log_response': resp,
        })
    return blobs


def extract_hematoxylin(image):
    """Extract hematoxylin channel from an RGB H&E image using LAB color space.

    Hematoxylin stains nuclei blue-purple. In LAB space, this corresponds
    to low B values (blue end of blue-yellow axis). We invert the B channel
    so nuclei appear bright.

    Args:
        image: RGB uint8 image (H, W, 3).

    Returns:
        2D float array where hematoxylin-stained regions are bright.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3)")

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # B channel: 0=blue, 128=neutral, 255=yellow
    # Hematoxylin (blue-purple) has low B values
    b_channel = lab[:, :, 2].astype(np.float32)

    # Invert so nuclei are bright
    hematoxylin = 255.0 - b_channel

    return hematoxylin


def detect_nuclei_hae(image, min_area_px=8, sigma=4, peak_thresh=0.5):
    """Detect nuclei in an RGB H&E stained image.

    Uses LAB color space to isolate hematoxylin (blue-purple nuclei),
    then applies LoG blob detection for robust counting even with
    touching nuclei.

    Args:
        image: RGB uint8 image (H, W, 3).
        min_area_px: Minimum nucleus area in pixels.
        sigma: LoG scale parameter (match to nucleus radius).
        peak_thresh: Minimum LoG response for detection.

    Returns:
        List of dicts with 'centroid_px' (cx, cy) and 'log_response'.
    """
    hema = extract_hematoxylin(image)
    return detect_blobs_log(hema, sigma=sigma, peak_thresh=peak_thresh,
                            min_intensity=0)


def detect_point_source(image, smooth_sigma=3):
    """Find the position of a point source (peak intensity) in an image.

    Uses Gaussian smoothing followed by sub-pixel peak localization via
    centroid of the top region around the argmax. More robust than raw
    argmax (handles noise) or weighted centroid (biased by large dim regions).

    Args:
        image: 2D grayscale array.
        smooth_sigma: Gaussian smoothing sigma for noise suppression.

    Returns:
        (cx, cy, peak_value) — sub-pixel center of the brightest source.
    """
    img = image.astype(np.float64)
    smoothed = ndimage.gaussian_filter(img, sigma=smooth_sigma)

    # Find peak in smoothed image
    peak_idx = smoothed.argmax()
    peak_y, peak_x = np.unravel_index(peak_idx, smoothed.shape)
    peak_val = float(smoothed[peak_y, peak_x])

    # Sub-pixel refinement: centroid of pixels above 80% of peak in local window
    window = max(5, int(smooth_sigma * 4))
    h, w = img.shape
    y0 = max(0, peak_y - window)
    y1 = min(h, peak_y + window + 1)
    x0 = max(0, peak_x - window)
    x1 = min(w, peak_x + window + 1)

    patch = smoothed[y0:y1, x0:x1]
    mask = patch > peak_val * 0.8
    if mask.sum() == 0:
        return float(peak_x), float(peak_y), peak_val

    ys, xs = np.where(mask)
    weights = patch[mask] - peak_val * 0.8
    total_w = weights.sum()
    if total_w <= 0:
        return float(peak_x), float(peak_y), peak_val

    cy = float((ys * weights).sum() / total_w) + y0
    cx = float((xs * weights).sum() / total_w) + x0
    return cx, cy, peak_val


def merge_nearby_centroids(centroids, threshold_px=6):
    """Merge centroids that are within threshold_px of each other.

    Uses greedy clustering: iterates through centroids, merging any
    within threshold distance into the first unmerged centroid's group.

    Args:
        centroids: List of (cx, cy) tuples.
        threshold_px: Maximum distance to merge.

    Returns:
        List of merged (cx, cy) tuples.
    """
    if not centroids:
        return []

    used = [False] * len(centroids)
    merged = []

    for i in range(len(centroids)):
        if used[i]:
            continue
        group = [centroids[i]]
        used[i] = True
        for j in range(i + 1, len(centroids)):
            if used[j]:
                continue
            dx = centroids[j][0] - centroids[i][0]
            dy = centroids[j][1] - centroids[i][1]
            if (dx * dx + dy * dy) <= threshold_px * threshold_px:
                group.append(centroids[j])
                used[j] = True
        cx = float(np.mean([c[0] for c in group]))
        cy = float(np.mean([c[1] for c in group]))
        merged.append((cx, cy))

    return merged


def count_objects_dt(mask, filter_size=5, dt_threshold=1.5,
                     fill_holes=True, return_centroids=False):
    """Count objects in a binary mask using distance transform local maxima.

    Splits touching/overlapping objects by finding peaks in the distance
    transform. Works well for circular/convex objects like cells, nuclei, RBCs.

    Args:
        mask: 2D boolean array (True = object pixels).
        filter_size: Window size for local maxima detection. Smaller values
            split more aggressively (3=fine, 5=standard, 7-9=coarse).
        dt_threshold: Minimum distance transform value for a peak. Controls
            minimum object size that gets counted.
        fill_holes: Fill holes in mask before processing.
        return_centroids: If True, return (count, list_of_centroids).

    Returns:
        int: Number of objects detected.
        If return_centroids=True: (int, list of (cx, cy) tuples).
    """
    from scipy.ndimage import distance_transform_edt, maximum_filter

    mask = np.asarray(mask, dtype=bool)
    if fill_holes:
        mask = binary_fill_holes(mask)

    dt = distance_transform_edt(mask)
    local_max = (maximum_filter(dt, size=filter_size) == dt) & (dt > dt_threshold)
    labeled, n_objects = ndimage.label(local_max)

    if not return_centroids:
        return int(n_objects)

    centroids = []
    for i in range(1, n_objects + 1):
        ys, xs = np.where(labeled == i)
        centroids.append((float(xs.mean()), float(ys.mean())))
    return int(n_objects), centroids


def watershed_split(mask, min_distance=7, dt_threshold=2.0, fill_holes=True):
    """Split touching objects in a binary mask using watershed segmentation.

    Standard approach for separating clumped cells (e.g., hemocytometer,
    dense tissue). Finds seeds via distance transform local maxima, then
    applies watershed to assign each pixel to the nearest seed.

    Args:
        mask: 2D boolean array (True = object pixels).
        min_distance: Minimum distance between watershed seeds, or
            ``'auto'`` to estimate from the mask using
            :func:`estimate_min_distance`. Controls splitting
            aggressiveness. Smaller = split more.
            Match to typical cell radius: min_distance ≈ 0.7 * cell_radius.
        dt_threshold: Minimum distance-transform value for a seed.
            Filters out small noise fragments.
        fill_holes: Fill holes in mask before processing.

    Returns:
        dict with:
            labeled: 2D int array, each object labeled 1..N.
            n_objects: Number of split objects.
            centroids: List of (cx, cy) tuples.
            areas: List of areas in pixels.
    """
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask, dtype=bool)
    if fill_holes:
        mask = binary_fill_holes(mask)

    if not mask.any():
        return {
            'labeled': np.zeros_like(mask, dtype=int),
            'n_objects': 0,
            'centroids': [],
            'areas': [],
        }

    if min_distance == 'auto':
        min_distance = estimate_min_distance(mask)

    dt = distance_transform_edt(mask)

    # Find seeds as local maxima of distance transform
    coords = peak_local_max(dt, min_distance=min_distance,
                            threshold_abs=dt_threshold, labels=mask)
    if len(coords) == 0:
        # No peaks found — treat entire mask as one object
        labeled, n = ndimage.label(mask)
        centroids = []
        areas = []
        for i in range(1, n + 1):
            ys, xs = np.where(labeled == i)
            centroids.append((float(xs.mean()), float(ys.mean())))
            areas.append(int(len(xs)))
        return {
            'labeled': labeled,
            'n_objects': n,
            'centroids': centroids,
            'areas': areas,
        }

    # Create seed markers
    markers = np.zeros_like(mask, dtype=int)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i

    # Watershed: expand seeds to fill mask
    labeled = watershed(-dt, markers, mask=mask)

    n_objects = int(labeled.max())
    centroids = []
    areas = []
    for i in range(1, n_objects + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) == 0:
            continue
        centroids.append((float(xs.mean()), float(ys.mean())))
        areas.append(int(len(xs)))

    return {
        'labeled': labeled,
        'n_objects': len(centroids),
        'centroids': centroids,
        'areas': areas,
    }


def estimate_min_distance(mask, percentile=25):
    """Estimate watershed min_distance from object spacing in a binary mask.

    Measures the distance transform at local maxima and uses the lower
    percentile as a conservative estimate of the smallest object radius.
    Good for setting ``watershed_split(min_distance=...)`` adaptively.

    Args:
        mask: 2D boolean array (True = object pixels).
        percentile: Percentile of peak distances to use (25 = conservative,
            handles dense packing). Use 50 for typical, 10 for very crowded.

    Returns:
        int: Recommended min_distance for watershed_split/peak_local_max.
            Minimum 2, maximum capped at 20.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.feature import peak_local_max

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 5  # safe default

    dt = distance_transform_edt(mask)
    # Find all peaks with a very small min_distance to get raw spacing
    coords = peak_local_max(dt, min_distance=2, threshold_abs=1.5)
    if len(coords) < 3:
        return 5

    peak_dists = dt[coords[:, 0], coords[:, 1]]
    est = float(np.percentile(peak_dists, percentile))
    return max(2, min(int(round(est)), 20))


def count_nuclei_fluorescence(image, threshold=12, min_area=10, min_mean=15,
                               edge_margin=3, return_details=False,
                               split_touching=False, min_distance='auto'):
    """Count fluorescent nuclei (bright on dark) with proper edge handling.

    Pipeline: threshold → connected components → filter by area and
    mean intensity → inclusive edge handling with small margin.

    Args:
        image: 2D grayscale array (nucleus/DAPI channel).
        threshold: Minimum pixel intensity for nuclear mask.
        min_area: Minimum CC area in pixels.
        min_mean: Minimum mean intensity within CC.
        edge_margin: Pixels from image edge to classify as "edge cell".
            Use 3 for inclusive counting (default). Edge cells are still
            counted in total but flagged separately.
        return_details: If True, return (total, interior, details_list).
        split_touching: If True, apply watershed to split touching nuclei.
            Essential for densely packed tissue (organoid walls, epithelia).
        min_distance: Watershed seed min_distance. Use ``'auto'`` (default)
            to estimate from mask, or int for manual control.

    Returns:
        int: Total cell count (including edge cells).
        If return_details: (total, interior_count, list of dicts with
            centroid_px, area, mean_intensity, max_intensity, is_edge).
    """
    from skimage.measure import regionprops

    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape

    mask = img > threshold

    if split_touching:
        ws = watershed_split(mask, min_distance=min_distance,
                             dt_threshold=1.5, fill_holes=True)
        labeled = ws['labeled']
        n = ws['n_objects']
    else:
        labeled, n = ndimage.label(mask)

    if n == 0:
        return (0, 0, []) if return_details else 0

    props = regionprops(labeled, intensity_image=img.astype(int))

    cells = []
    for p in props:
        if p.area < min_area:
            continue
        if p.intensity_mean < min_mean:
            continue

        cy, cx = p.centroid
        is_edge = (cy < edge_margin or cy >= h - edge_margin or
                   cx < edge_margin or cx >= w - edge_margin)

        cells.append({
            'centroid_px': (float(cx), float(cy)),
            'area': p.area,
            'mean_intensity': float(p.intensity_mean),
            'max_intensity': int(p.intensity_max),
            'eccentricity': float(p.eccentricity),
            'is_edge': is_edge,
        })

    total = len(cells)
    interior = sum(1 for c in cells if not c['is_edge'])

    if return_details:
        return total, interior, cells
    return total


def count_bacteria_bf(image, threshold=10, magnification=40, fov_px=512,
                      return_centroids=False):
    """Count bacteria in a brightfield image with magnification-aware size filtering.

    Bacteria are DARK on gray background. Detects by inverting (median - pixel)
    then applying size filters tuned to expected bacteria size at the given
    magnification.

    Includes a density sanity check: raises ValueError if count seems
    unreasonably high for the FOV size.

    Args:
        image: 2D grayscale brightfield array.
        threshold: Inversion threshold (median_bg - pixel > threshold).
        magnification: Objective magnification (10, 20, 40, 100).
            Controls expected bacteria size in pixels.
        fov_px: FOV width in pixels (default 512).
        return_centroids: If True, return (count, centroids) where centroids
            is an (N, 2) array of (x, y) positions in camera pixels.
            Useful for tracking workflows with population_speeds().

    Returns:
        int: Number of bacteria detected. Or (int, ndarray) if return_centroids.

    Raises:
        ValueError: If detected count exceeds sanity threshold
            (> FOV_area / 50, which would mean < 7px per bacterium).
    """
    img = np.asarray(image, dtype=np.float64)
    median_bg = np.median(img)
    inv = median_bg - img
    mask = inv > threshold

    # Magnification-dependent size filtering:
    # Bacteria are ~1×3 µm. Pixel size ≈ 10/magnification µm/px.
    # At 10x: 1µm/px → bacteria = 1×3 px = ~3 px area
    # At 20x: 0.5µm/px → bacteria = 2×6 px = ~12 px area
    # At 40x: 0.25µm/px → bacteria = 4×12 px = ~48 px area
    # At 100x: 0.1µm/px → bacteria = 10×30 px = ~300 px area
    size_table = {
        10: (2, 20),
        20: (5, 80),
        40: (15, 300),
        100: (50, 2000),
    }
    min_size, max_size = size_table.get(magnification, (3, 300))

    labeled, n = ndimage.label(mask)
    if n == 0:
        if return_centroids:
            return 0, np.empty((0, 2))
        return 0

    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    valid_labels = [i + 1 for i, s in enumerate(sizes) if min_size <= s <= max_size]
    count = len(valid_labels)

    # Sanity check: count should not exceed FOV_area / 50
    max_reasonable = (fov_px * fov_px) // 50
    if count > max_reasonable:
        raise ValueError(
            f"Bacteria count {count} exceeds sanity limit {max_reasonable} "
            f"for {fov_px}px FOV at {magnification}x. Check segmentation."
        )

    if return_centroids:
        centroids = []
        for lbl in valid_labels:
            ys, xs = np.where(labeled == lbl)
            centroids.append((float(xs.mean()), float(ys.mean())))
        centroids = np.array(centroids) if centroids else np.empty((0, 2))
        return count, centroids

    return count


def count_nuclei_adaptive(image, block_size=31, offset=-2, min_area=8,
                          watershed_min_dist=3, dt_threshold=1.0,
                          log_sigma=2.5, log_thresh=0.5,
                          edge_margin=3, return_details=False):
    """Count densely packed nuclei using adaptive threshold + watershed + LoG.

    Designed for cases where nuclei are small (5-10 px diameter) and closely
    packed with ~1px gaps — typical for 10x DAPI imaging of organoids or
    dense tissue. Uses three complementary methods:

    1. Adaptive local threshold captures dim nuclei missed by global threshold
    2. Watershed with tight min_distance splits merged nuclei
    3. LoG blob detection cross-validates the count

    Args:
        image: 2D grayscale array (nucleus/DAPI channel).
        block_size: Local threshold neighborhood (odd int, ~2x nucleus size).
        offset: Subtracted from local threshold (negative = more permissive).
        min_area: Minimum nucleus area in pixels.
        watershed_min_dist: Minimum distance between watershed seeds.
            For 5-6px nuclei, use 3. For 10-12px nuclei, use 5-7.
        dt_threshold: Min distance-transform value for watershed seeds.
        log_sigma: LoG scale parameter (~0.4 * nucleus radius).
        log_thresh: Minimum LoG response for peak detection.
        edge_margin: Pixels from image border for edge classification.
        return_details: If True, return full details dict.

    Returns:
        int: Consensus nuclei count.
        If return_details: dict with count_adaptive, count_log,
            count_consensus, centroids_watershed, centroids_log.
    """
    from skimage import filters, morphology, measure
    from skimage.segmentation import watershed as ws_func
    from skimage.feature import peak_local_max

    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape

    # Step 1: Adaptive local threshold
    thresh_map = filters.threshold_local(img, block_size,
                                          method='gaussian', offset=offset)
    binary = img > thresh_map

    # Clean small objects
    if min_area > 0:
        binary = morphology.remove_small_objects(binary, max_size=min_area)

    # Step 2: Watershed splitting
    if binary.any():
        dt = ndimage.distance_transform_edt(binary)
        coords = peak_local_max(dt, min_distance=watershed_min_dist,
                                threshold_abs=dt_threshold, labels=binary)
        if len(coords) > 0:
            markers = np.zeros_like(binary, dtype=int)
            for i, (y, x) in enumerate(coords, 1):
                markers[y, x] = i
            labeled_ws = ws_func(-dt, markers, mask=binary)
        else:
            labeled_ws = measure.label(binary)
    else:
        labeled_ws = np.zeros_like(binary, dtype=int)

    # Extract watershed centroids
    props_ws = measure.regionprops(labeled_ws)
    centroids_ws = []
    for p in props_ws:
        if p.area < min_area:
            continue
        cy, cx = p.centroid
        centroids_ws.append((float(cx), float(cy)))
    count_ws = len(centroids_ws)

    # Step 3: LoG cross-validation
    log_response = -ndimage.gaussian_laplace(img, sigma=log_sigma)
    log_max = ndimage.maximum_filter(log_response, size=int(2 * log_sigma + 1))
    peaks = (log_response == log_max) & (log_response > log_thresh)
    # Filter peaks to within the adaptive mask (avoid background noise)
    if binary.any():
        dilated = ndimage.binary_dilation(binary, iterations=2)
        peaks = peaks & dilated
    labeled_log, count_log = ndimage.label(peaks)

    centroids_log = []
    for i in range(1, count_log + 1):
        ys, xs = np.where(labeled_log == i)
        centroids_log.append((float(xs.mean()), float(ys.mean())))

    # Consensus: take the higher count (LoG catches what watershed misses and vice versa)
    # but cap at 1.3× the lower count to avoid noise-driven overcounting
    count_min = min(count_ws, count_log)
    count_max = max(count_ws, count_log)
    if count_min > 0 and count_max > 1.3 * count_min:
        # Large disagreement — trust the method that's closer to morphology
        # Watershed is more reliable when there's clear structure
        count_consensus = count_ws
    else:
        count_consensus = count_max

    if return_details:
        return {
            'count_adaptive': count_ws,
            'count_log': count_log,
            'count_consensus': count_consensus,
            'centroids_watershed': centroids_ws,
            'centroids_log': centroids_log,
            'binary_mask': binary,
            'labeled_watershed': labeled_ws,
        }
    return count_consensus


def detect_fluorescent_centroids(image, threshold=None, min_area=3, max_area=500):
    """Detect bright fluorescent objects and return centroids as numpy array.

    Bright spots on dark background — cleaner than BF inversion for
    quantitative tracking. Preferred for speed measurement over BF
    (which creates noise false-positives with zero displacement).

    Args:
        image: 2D grayscale fluorescence image.
        threshold: Intensity threshold. If None, uses mean + 3*std of
            non-zero pixels (auto).
        min_area: Minimum CC area in pixels.
        max_area: Maximum CC area in pixels.

    Returns:
        (N, 2) array of (x, y) centroids in camera pixels.
        Returns empty (0, 2) array if no objects detected.
    """
    img = np.asarray(image, dtype=np.float64)

    if threshold is None:
        # Auto-threshold: background is dark, objects are bright
        nz = img[img > 0]
        if len(nz) == 0:
            return np.empty((0, 2))
        threshold = float(nz.mean() + 3 * nz.std())

    mask = img > threshold
    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.empty((0, 2))

    centroids = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        area = len(ys)
        if min_area <= area <= max_area:
            centroids.append((float(xs.mean()), float(ys.mean())))

    return np.array(centroids) if centroids else np.empty((0, 2))
