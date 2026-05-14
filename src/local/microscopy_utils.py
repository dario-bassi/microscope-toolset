"""Microscopy utility functions for smart acquisition pipelines.

Available in the execute_python_code namespace for MCP agent use.
"""

import logging

import numpy as np

logger = logging.getLogger("MicroscopyUtils")


def find_bright_centroid(image: np.ndarray, threshold_sigma: float = 2.5, window: int = 64):
    """Find the centroid of the brightest region in a grayscale image.

    Strategy: find the peak pixel, then compute the intensity-weighted
    centroid of above-threshold pixels within a local window around the peak.
    This avoids noise pixels far from the signal biasing the result.

    Args:
        image: 2D grayscale image (uint8 or float).
        threshold_sigma: Number of standard deviations above mean to threshold.
        window: Half-size of the search window around the peak pixel.

    Returns:
        (cy, cx, area_px, peak) — centroid row/col (full-image coords),
        number of bright pixels in window, and peak intensity.
        Returns (None, None, 0, peak) if no significant bright region found.
    """
    img = image.astype(np.float32)
    mean_val = img.mean()
    std_val = img.std()
    peak = float(img.max())

    thresh = mean_val + threshold_sigma * std_val
    if peak <= thresh:
        return None, None, 0, peak

    # Find peak pixel location
    h, w = img.shape[:2]
    peak_idx = img.argmax()
    peak_y, peak_x = divmod(int(peak_idx), w)

    # Extract local window around peak
    y0 = max(0, peak_y - window)
    y1 = min(h, peak_y + window)
    x0 = max(0, peak_x - window)
    x1 = min(w, peak_x + window)
    patch = img[y0:y1, x0:x1]

    # Threshold within the local window
    local_mask = patch > thresh
    area_px = int(local_mask.sum())
    if area_px == 0:
        return None, None, 0, peak

    # Intensity-weighted centroid within the window
    local_ys, local_xs = np.where(local_mask)
    weights = patch[local_mask] - thresh
    total_w = weights.sum()
    if total_w <= 0:
        return None, None, 0, peak

    # Convert back to full-image coordinates
    cy = float((local_ys * weights).sum() / total_w) + y0
    cx = float((local_xs * weights).sum() / total_w) + x0
    return cy, cx, area_px, peak


def center_on_cell(
    mmc,
    pixel_size_um: float = 0.25,
    threshold_sigma: float = 2.5,
    min_peak_above_bg: float = 30.0,
    max_iterations: int = 2,
):
    """Snap an image, find the brightest region, and re-center the stage on it.

    Iteratively adjusts the stage position so the bright region is centered
    in the field of view.  Works at any magnification.

    Args:
        mmc: CMMCorePlus / UniMMCore instance (raw, not GatekeeperCore).
        pixel_size_um: Pixel size in microns for the current objective.
        threshold_sigma: σ above mean for bright-pixel detection.
        min_peak_above_bg: Minimum (peak - mean) to consider real signal present.
        max_iterations: Maximum re-centering attempts.

    Returns:
        dict with keys:
            image: final snapped image (2D ndarray)
            centered: bool — whether a bright region was found and centered
            peak: peak intensity in final image
            offset_um: (dx, dy) total stage offset applied
            centroid_px: (cx, cy) centroid in final image (pixel coords)
    """
    xy_device = mmc.getXYStageDevice()
    total_dx, total_dy = 0.0, 0.0

    for _iteration in range(max_iterations):
        mmc.snapImage()
        img = mmc.getImage()

        cy, cx, area_px, peak = find_bright_centroid(img, threshold_sigma)
        mean_val = float(img.astype(np.float32).mean())

        # Check if there's real signal
        if cy is None or (peak - mean_val) < min_peak_above_bg:
            return {
                "image": img,
                "centered": False,
                "peak": peak,
                "offset_um": (total_dx, total_dy),
                "centroid_px": (None, None),
            }

        h, w = img.shape[:2]
        center_y, center_x = h / 2.0, w / 2.0

        # Offset in pixels from image center
        dx_px = cx - center_x
        dy_px = cy - center_y

        # Convert to microns
        dx_um = dx_px * pixel_size_um
        dy_um = dy_px * pixel_size_um

        # If already close to center (within 5% of FOV), done
        fov = w * pixel_size_um
        if abs(dx_um) < fov * 0.05 and abs(dy_um) < fov * 0.05:
            return {
                "image": img,
                "centered": True,
                "peak": peak,
                "offset_um": (total_dx, total_dy),
                "centroid_px": (cx, cy),
            }

        # Move stage
        cur_x = mmc.getXPosition()
        cur_y = mmc.getYPosition()
        mmc.setXYPosition(cur_x + dx_um, cur_y + dy_um)
        mmc.waitForDevice(xy_device)

        total_dx += dx_um
        total_dy += dy_um

    # Final snap after last repositioning
    mmc.snapImage()
    img = mmc.getImage()
    cy, cx, area_px, peak = find_bright_centroid(img, threshold_sigma)

    return {
        "image": img,
        "centered": (cx is not None),
        "peak": peak,
        "offset_um": (total_dx, total_dy),
        "centroid_px": (cx, cy),
    }


def detect_cells(
    image: np.ndarray,
    threshold_sigma: float = 2.5,
    min_area_px: int = 50,
    pixel_size_um: float = 1.0,
    fill_holes: bool = True,
    global_stats: tuple[float, float] | None = None,
):
    """Detect cells in a grayscale image using thresholding + hole filling.

    Args:
        image: 2D grayscale image.
        threshold_sigma: sigma above mean for threshold.
        min_area_px: Minimum connected component area in pixels.
        pixel_size_um: Pixel size for area conversion to um2.
        fill_holes: If True, fill holes inside detected cells (recommended
            for brightfield where cells have bright membrane + darker interior).
        global_stats: Optional (mean, std) tuple computed across multiple
            frames. When processing a stack, compute stats once over all
            frames and pass here to avoid per-frame threshold adaptation
            (which fails on noise-only frames).

    Returns:
        List of dicts, each with keys: centroid_px (x, y), centroid_um (x, y),
        area_px, area_um2, peak, mean_intensity, bbox (x, y, w, h).
    """
    import cv2
    from scipy.ndimage import binary_fill_holes

    img = image.astype(np.float32)

    if global_stats is not None:
        mean_val, std_val = global_stats
    else:
        mean_val = img.mean()
        std_val = img.std()

    thresh = mean_val + threshold_sigma * std_val

    binary = (img > thresh).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cells = []
    for i in range(1, num_labels):  # skip background (label 0)
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < min_area_px:
            continue

        # Optionally fill holes inside the cell
        region_mask = labels == i
        if fill_holes:
            region_mask = binary_fill_holes(region_mask)
            area_px = int(region_mask.sum())

        cx, cy = centroids[i]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        region_vals = img[region_mask]

        cells.append(
            {
                "centroid_px": (float(cx), float(cy)),
                "centroid_um": (float(cx * pixel_size_um), float(cy * pixel_size_um)),
                "area_px": int(area_px),
                "area_um2": float(area_px * pixel_size_um**2),
                "peak": float(region_vals.max()),
                "mean_intensity": float(region_vals.mean()),
                "bbox": (int(x), int(y), int(w), int(h)),
            }
        )

    # Sort by area descending
    cells.sort(key=lambda c: c["area_px"], reverse=True)
    return cells
