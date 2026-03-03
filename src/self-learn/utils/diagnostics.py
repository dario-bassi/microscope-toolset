"""Diagnostic image saving for challenge submissions.

Every submission must include at least one saved snapshot showing what
the microscope acquired and what the analysis found. This is non-negotiable
per orchestrator directive.

Usage:
    from src.diagnostics import save_snapshot, save_overlay

    # Save raw channel images
    save_snapshot(img, 153, 'membrane_raw')

    # Save image with detected cell centroids overlaid
    save_overlay(img, cells, 153, 'detected_cells')
"""

import numpy as np
import cv2
import os


def save_snapshot(img, challenge_id, label='snapshot'):
    """Save a grayscale image as PNG with auto-scaling.

    Args:
        img: 2D numpy array (any dtype).
        challenge_id: Challenge number for filename.
        label: Descriptive label for the image.

    Returns:
        str: Path to saved file.
    """
    path = f'/tmp/ch{challenge_id}_{label}.png'
    img_f = img.astype(float)
    if img_f.max() > img_f.min():
        scaled = ((img_f - img_f.min()) / (img_f.max() - img_f.min()) * 255)
    else:
        scaled = np.zeros_like(img_f)
    cv2.imwrite(path, scaled.astype(np.uint8))
    return path


def save_overlay(img, cells, challenge_id, label='overlay',
                 marker_color=(0, 255, 0), marker_radius=8):
    """Save image with cell centroids marked as circles.

    Args:
        img: 2D grayscale image.
        cells: List of dicts with 'x', 'y' (pixel coords within image)
            or list of (x, y) tuples.
        challenge_id: Challenge number for filename.
        label: Descriptive label.
        marker_color: BGR color for markers.
        marker_radius: Circle radius in pixels.

    Returns:
        str: Path to saved file.
    """
    path = f'/tmp/ch{challenge_id}_{label}.png'

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

    cv2.imwrite(path, bgr)
    return path


def save_composite(images, titles, challenge_id, label='composite', cols=3):
    """Save a composite image showing multiple channels side by side.

    Args:
        images: List of 2D arrays.
        titles: List of title strings (same length as images).
        challenge_id: Challenge number.
        label: Descriptive label.
        cols: Number of columns in the grid.

    Returns:
        str: Path to saved file.
    """
    path = f'/tmp/ch{challenge_id}_{label}.png'

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

    cv2.imwrite(path, canvas)
    return path
