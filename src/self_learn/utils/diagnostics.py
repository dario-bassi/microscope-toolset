"""Diagnostic image saving for visual verification.

Save snapshots and overlays to disk so the agent can read them with its
vision capability and confirm the analysis before acting on the result.

See `knowledge/Core/Approach/Visual verification.md` for the full protocol:
render an overlay, read the PNG, verify the marks match what you plan to
report — then proceed.

Usage:
    from self_learn.utils.diagnostics import save_snapshot, save_overlay

    tmp = Path(tempfile.gettempdir())
    save_snapshot(img, 'membrane_raw', save_dir=tmp)
    save_overlay(img, cells, 'detected_cells', save_dir=tmp)
"""

import tempfile
import numpy as np
import cv2
from pathlib import Path


def save_snapshot(img, label='snapshot', save_dir=None):
    """Save a grayscale image as PNG with auto-scaling.

    Args:
        img: 2D numpy array (any dtype, any bit depth — auto-scaled to uint8).
        label: Descriptive label used in the filename.
        save_dir: Directory to save into (Path or str). Defaults to the
            system temp directory (`tempfile.gettempdir()`).

    Returns:
        Path: Path to the saved PNG file.
    """
    out = Path(save_dir) if save_dir else Path(tempfile.gettempdir())
    path = out / f'{label}.png'
    img_f = img.astype(float)
    if img_f.max() > img_f.min():
        scaled = ((img_f - img_f.min()) / (img_f.max() - img_f.min()) * 255)
    else:
        scaled = np.zeros_like(img_f)
    cv2.imwrite(str(path), scaled.astype(np.uint8))
    return path


def save_overlay(img, cells, label='overlay', save_dir=None,
                 marker_color=(0, 255, 0), marker_radius=8):
    """Save image with cell centroids marked as circles.

    Args:
        img: 2D grayscale image (any dtype — auto-scaled to uint8).
        cells: List of dicts with 'x', 'y' keys (pixel coordinates, image
            space) or list of (x, y) tuples. Coordinates must be in pixel
            space (not world/µm coordinates).
        label: Descriptive label used in the filename.
        save_dir: Directory to save into (Path or str). Defaults to the
            system temp directory.
        marker_color: BGR tuple for the circle color. Default: green (0,255,0).
        marker_radius: Circle radius in pixels. Default: 8.

    Returns:
        Path: Path to the saved PNG file.
    """
    out = Path(save_dir) if save_dir else Path(tempfile.gettempdir())
    path = out / f'{label}.png'

    # Auto-scale to 8-bit
    img_f = img.astype(float)
    if img_f.max() > img_f.min():
        scaled = ((img_f - img_f.min()) / (img_f.max() - img_f.min()) * 255)
    else:
        scaled = np.zeros_like(img_f)
    gray = scaled.astype(np.uint8)

    # Convert to BGR for colored markers
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for c in cells:
        if isinstance(c, dict):
            cx = int(c.get('x', c.get('px', 0)))
            cy = int(c.get('y', c.get('py', 0)))
        elif isinstance(c, (tuple, list)):
            cx, cy = int(c[0]), int(c[1])
        else:
            continue
        cv2.circle(bgr, (cx, cy), marker_radius, marker_color, 2)

    cv2.imwrite(str(path), bgr)
    return path


def save_composite(images, titles, label='composite', save_dir=None, cols=3):
    """Save a composite image showing multiple channels side by side.

    Args:
        images: List of 2D arrays (any dtype — each auto-scaled to uint8).
        titles: List of title strings, same length as images.
        label: Descriptive label used in the filename.
        save_dir: Directory to save into (Path or str). Defaults to the
            system temp directory.
        cols: Number of columns in the grid layout. Default: 3.

    Returns:
        Path: Path to the saved PNG file.
    """
    out = Path(save_dir) if save_dir else Path(tempfile.gettempdir())
    path = out / f'{label}.png'

    n = len(images)
    rows = (n + cols - 1) // cols
    h, w = images[0].shape[:2]
    title_h = 30  # pixels for title bar

    canvas = np.zeros((rows * (h + title_h), cols * w, 3), dtype=np.uint8)

    for i, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(i, cols)
        y0 = r * (h + title_h)
        x0 = c * w

        # Scale image
        img_f = img.astype(float)
        if img_f.max() > img_f.min():
            scaled = ((img_f - img_f.min()) / (img_f.max() - img_f.min()) * 255)
        else:
            scaled = np.zeros_like(img_f)
        gray = scaled.astype(np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Title bar
        canvas[y0:y0 + title_h, x0:x0 + w] = (40, 40, 40)
        cv2.putText(canvas, title, (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Image
        canvas[y0 + title_h:y0 + title_h + h, x0:x0 + w] = bgr

    cv2.imwrite(str(path), canvas)
    return path
