"""Multi-position tiled imaging and stitching.

Functions for computing tile grids, acquiring tiled images, and
assembling them into a single stitched mosaic. Includes phase-correlation
based sub-pixel alignment for precise stitching.

Functions:
    tile_positions         -- Compute grid of stage positions for a bounding box
    stitch_tiles           -- Assemble overlapping tiles into a mosaic
    phase_correlation      -- Compute sub-pixel shift between two overlapping images
    align_tile_pair        -- Align two adjacent tiles using phase correlation
    stitch_tiles_aligned   -- Stitch tiles with phase-correlation refinement
"""

import numpy as np


def tile_positions(bbox, fov_size, overlap=0.1):
    """Compute grid of stage positions covering a bounding box.

    Args:
        bbox: (x_min, y_min, x_max, y_max) in world coordinates.
        fov_size: (width, height) of one field of view in world coords.
            Or a single number for square FOV.
        overlap: Fractional overlap between adjacent tiles (0.0-0.5).

    Returns:
        dict with:
            positions: List of (x, y) stage center positions.
            grid_shape: (n_cols, n_rows) of the tile grid.
            step: (step_x, step_y) spacing between tile centers.
            n_tiles: Total number of tiles.
    """
    x_min, y_min, x_max, y_max = bbox

    if isinstance(fov_size, (int, float)):
        fov_w = fov_h = float(fov_size)
    else:
        fov_w, fov_h = float(fov_size[0]), float(fov_size[1])

    step_x = fov_w * (1 - overlap)
    step_y = fov_h * (1 - overlap)

    # Number of tiles needed
    width = x_max - x_min
    height = y_max - y_min
    n_cols = max(1, int(np.ceil(width / step_x)))
    n_rows = max(1, int(np.ceil(height / step_y)))

    # Center the grid so tiles extend symmetrically
    total_w = (n_cols - 1) * step_x + fov_w
    total_h = (n_rows - 1) * step_y + fov_h
    x_start = x_min + fov_w / 2 - (total_w - width) / 2
    y_start = y_min + fov_h / 2 - (total_h - height) / 2

    positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = x_start + col * step_x
            y = y_start + row * step_y
            positions.append((float(x), float(y)))

    return {
        "positions": positions,
        "grid_shape": (n_cols, n_rows),
        "step": (float(step_x), float(step_y)),
        "n_tiles": len(positions),
    }


def stitch_tiles(tiles, grid_shape, overlap_px=0):
    """Assemble tiles into a mosaic.

    Simple placement-based stitching (no phase correlation).
    Tiles are placed left-to-right, top-to-bottom.

    Args:
        tiles: List of 2D arrays, ordered left-to-right then top-to-bottom.
            Length must equal grid_shape[0] * grid_shape[1].
        grid_shape: (n_cols, n_rows) of the tile grid.
        overlap_px: Pixel overlap between adjacent tiles. Overlapping
            regions use the maximum of both tiles.

    Returns:
        dict with:
            mosaic: 2D stitched image.
            tile_origins: List of (x, y) pixel origins of each tile.
    """
    n_cols, n_rows = grid_shape
    assert len(tiles) == n_cols * n_rows, f"Expected {n_cols * n_rows} tiles, got {len(tiles)}"

    tile_h, tile_w = tiles[0].shape[:2]

    step_x = tile_w - overlap_px
    step_y = tile_h - overlap_px

    mosaic_w = (n_cols - 1) * step_x + tile_w
    mosaic_h = (n_rows - 1) * step_y + tile_h

    mosaic = np.zeros((mosaic_h, mosaic_w), dtype=tiles[0].dtype)
    origins = []

    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            ox = col * step_x
            oy = row * step_y
            origins.append((ox, oy))

            region = mosaic[oy : oy + tile_h, ox : ox + tile_w]
            mosaic[oy : oy + tile_h, ox : ox + tile_w] = np.maximum(region, tiles[idx])

    return {
        "mosaic": mosaic,
        "tile_origins": origins,
    }


def phase_correlation(img1, img2):
    """Compute translation between two images using phase correlation.

    Uses FFT-based cross-correlation to find the sub-pixel shift that
    aligns img2 to img1.

    Args:
        img1: Reference image (2D array).
        img2: Image to align (2D array, same shape as img1).

    Returns:
        dict with:
            shift_y: Vertical shift (pixels, positive = img2 is shifted down).
            shift_x: Horizontal shift (pixels, positive = img2 is shifted right).
            confidence: Peak-to-noise ratio (higher = more reliable).
    """
    f1 = np.asarray(img1, dtype=np.float64)
    f2 = np.asarray(img2, dtype=np.float64)

    if f1.shape != f2.shape:
        # Crop to common size
        h = min(f1.shape[0], f2.shape[0])
        w = min(f1.shape[1], f2.shape[1])
        f1 = f1[:h, :w]
        f2 = f2[:h, :w]

    # FFT-based cross-correlation
    F1 = np.fft.fft2(f1)
    F2 = np.fft.fft2(f2)

    # Cross-power spectrum
    cross = F1 * np.conj(F2)
    denom = np.abs(cross)
    denom[denom < 1e-10] = 1e-10
    normalized = cross / denom

    # Inverse FFT to get correlation surface
    corr = np.real(np.fft.ifft2(normalized))

    # Find peak
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = float(corr[peak_idx])

    # Convert to signed shift (handle wrap-around)
    h, w = corr.shape
    dy = peak_idx[0]
    dx = peak_idx[1]
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w

    # Confidence: ratio of peak to mean correlation
    mean_corr = float(np.mean(np.abs(corr)))
    confidence = peak_val / max(mean_corr, 1e-10)

    return {
        "shift_y": int(dy),
        "shift_x": int(dx),
        "confidence": round(float(confidence), 2),
    }


def align_tile_pair(tile1, tile2, overlap_px, direction="horizontal"):
    """Align two adjacent tiles using phase correlation on the overlap region.

    Args:
        tile1: First tile (2D array, left or top).
        tile2: Second tile (2D array, right or bottom).
        overlap_px: Expected overlap in pixels.
        direction: 'horizontal' (tile2 is to the right of tile1) or
            'vertical' (tile2 is below tile1).

    Returns:
        dict with:
            shift_y: Refined vertical offset between tiles.
            shift_x: Refined horizontal offset between tiles.
            confidence: Alignment confidence.
    """
    t1 = np.asarray(tile1, dtype=np.float64)
    t2 = np.asarray(tile2, dtype=np.float64)

    if overlap_px < 4:
        return {"shift_y": 0, "shift_x": 0, "confidence": 0.0}

    if direction == "horizontal":
        # Overlap region: right edge of tile1, left edge of tile2
        strip1 = t1[:, -overlap_px:]
        strip2 = t2[:, :overlap_px]
    else:
        # Overlap region: bottom edge of tile1, top edge of tile2
        strip1 = t1[-overlap_px:, :]
        strip2 = t2[:overlap_px, :]

    result = phase_correlation(strip1, strip2)
    return result


def stitch_tiles_aligned(tiles, grid_shape, overlap_px, max_correction=None):
    """Stitch tiles with phase-correlation alignment.

    First computes nominal positions from the grid, then refines each
    tile position using phase correlation with its left/top neighbor.

    Args:
        tiles: List of 2D arrays, ordered left-to-right, top-to-bottom.
        grid_shape: (n_cols, n_rows) of the tile grid.
        overlap_px: Nominal overlap in pixels.
        max_correction: Maximum allowed correction (px). If None, uses
            overlap_px // 4.

    Returns:
        dict with:
            mosaic: 2D stitched image.
            tile_origins: List of (x, y) refined pixel origins.
            corrections: List of (dx, dy) corrections applied.
            confidences: List of alignment confidence values.
    """
    n_cols, n_rows = grid_shape
    assert len(tiles) == n_cols * n_rows

    if max_correction is None:
        max_correction = max(overlap_px // 4, 1)

    tile_h, tile_w = tiles[0].shape[:2]
    step_x = tile_w - overlap_px
    step_y = tile_h - overlap_px

    # Start with nominal positions
    origins = []
    corrections = []
    confidences = []

    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            nom_x = col * step_x
            nom_y = row * step_y

            dx, dy = 0, 0
            conf = 0.0

            # Align with left neighbor
            if col > 0 and overlap_px >= 4:
                left_idx = row * n_cols + (col - 1)
                align = align_tile_pair(tiles[left_idx], tiles[idx], overlap_px, "horizontal")
                if (
                    abs(align["shift_x"]) <= max_correction
                    and abs(align["shift_y"]) <= max_correction
                ):
                    dx = align["shift_x"]
                    dy = align["shift_y"]
                    conf = align["confidence"]

            # Align with top neighbor (use average if both available)
            if row > 0 and overlap_px >= 4:
                top_idx = (row - 1) * n_cols + col
                align = align_tile_pair(tiles[top_idx], tiles[idx], overlap_px, "vertical")
                if (
                    abs(align["shift_x"]) <= max_correction
                    and abs(align["shift_y"]) <= max_correction
                ):
                    if conf > 0:
                        # Average horizontal and vertical alignments
                        dx = (dx + align["shift_x"]) // 2
                        dy = (dy + align["shift_y"]) // 2
                        conf = (conf + align["confidence"]) / 2
                    else:
                        dx = align["shift_x"]
                        dy = align["shift_y"]
                        conf = align["confidence"]

            # Accumulate corrections from previous tiles
            if col > 0:
                prev_ox = origins[idx - 1][0]
                nom_x = prev_ox + step_x + dx
            if row > 0:
                prev_oy = origins[idx - n_cols][1]
                if col == 0:
                    nom_y = prev_oy + step_y + dy

            origins.append((int(nom_x), int(nom_y)))
            corrections.append((int(dx), int(dy)))
            confidences.append(round(float(conf), 2))

    # Shift all origins so minimum is (0, 0)
    min_x = min(o[0] for o in origins)
    min_y = min(o[1] for o in origins)
    origins = [(o[0] - min_x, o[1] - min_y) for o in origins]

    # Compute mosaic size
    max_x = max(o[0] + tile_w for o in origins)
    max_y = max(o[1] + tile_h for o in origins)

    mosaic = np.zeros((max_y, max_x), dtype=tiles[0].dtype)

    for idx, (ox, oy) in enumerate(origins):
        region = mosaic[oy : oy + tile_h, ox : ox + tile_w]
        mosaic[oy : oy + tile_h, ox : ox + tile_w] = np.maximum(region, tiles[idx])

    return {
        "mosaic": mosaic,
        "tile_origins": origins,
        "corrections": corrections,
        "confidences": confidences,
    }
