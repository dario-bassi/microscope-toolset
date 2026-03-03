"""Showcase figure generator for experiment results.

Creates multi-panel publication-style figures with annotations,
scale bars, and result text. Ideal for documenting microscopy
experiments and analysis results.

Functions:
    make_showcase -- Build and save a multi-panel showcase figure
    add_scalebar  -- Add a scale bar to an image
    annotate_image -- Add markers, contours, or text to an image
    get_showcase_dir -- Get the output directory for showcase figures
"""

import os
import numpy as np
import cv2


def get_showcase_dir():
    """Get the showcase output directory.

    Checks in order:
    1. MICROSCOPE_SHOWCASE_DIR environment variable
    2. ~/.microscope/showcase directory
    3. ./showcase in current working directory (fallback)

    Returns:
        str: Path to showcase directory (will be created if doesn't exist).
    """
    # Check environment variable first
    if 'MICROSCOPE_SHOWCASE_DIR' in os.environ:
        return os.environ['MICROSCOPE_SHOWCASE_DIR']

    # Default to user's home directory
    default_dir = os.path.expanduser('~/.microscope/showcase')
    return default_dir


# Legacy global for backward compatibility
SHOWCASE_DIR = get_showcase_dir()


def _scale_image(img):
    """Scale any image to uint8 for display."""
    img_f = img.astype(np.float64)
    lo, hi = img_f.min(), img_f.max()
    if hi > lo:
        return ((img_f - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)


def _to_bgr(img):
    """Convert grayscale or RGB to BGR for OpenCV."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def add_scalebar(img, pixel_size_um, bar_um=None, position='bottom_right',
                 color=(255, 255, 255), thickness=3):
    """Add a scale bar to an image.

    Args:
        img: 2D or 3D (BGR) uint8 image.
        pixel_size_um: Micrometers per pixel.
        bar_um: Length of scale bar in µm. If None, auto-selects a
            round number that fits ~1/5 of image width.
        position: 'bottom_right', 'bottom_left', 'top_right', 'top_left'.
        color: BGR color tuple.
        thickness: Line thickness in pixels.

    Returns:
        Annotated BGR image (copy).
    """
    bgr = _to_bgr(img)
    h, w = bgr.shape[:2]

    if bar_um is None:
        target_px = w // 5
        target_um = target_px * pixel_size_um
        # Round to nice number
        for nice in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
            if nice >= target_um * 0.5:
                bar_um = nice
                break
        else:
            bar_um = int(target_um)

    bar_px = int(bar_um / pixel_size_um)
    margin = 15

    if 'right' in position:
        x1 = w - margin
        x0 = x1 - bar_px
    else:
        x0 = margin
        x1 = x0 + bar_px

    if 'bottom' in position:
        y = h - margin
    else:
        y = margin + 10

    cv2.line(bgr, (x0, y), (x1, y), color, thickness)
    label = f'{bar_um} um'
    cv2.putText(bgr, label, (x0, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return bgr


def annotate_image(img, centroids=None, contours=None, text=None,
                   marker_color=(0, 255, 0), marker_radius=6,
                   contour_color=(0, 255, 255), text_color=(255, 255, 255)):
    """Add markers, contours, and text to an image.

    Args:
        img: 2D or BGR image.
        centroids: List of (x, y) or dicts with 'x','y' keys.
        contours: List of contour arrays (OpenCV format).
        text: List of (x, y, string) tuples for text labels.
        marker_color, contour_color, text_color: BGR colors.
        marker_radius: Circle radius for centroids.

    Returns:
        Annotated BGR image (copy).
    """
    bgr = _to_bgr(_scale_image(img) if img.dtype != np.uint8 else img)

    if centroids:
        for c in centroids:
            if isinstance(c, dict):
                cx, cy = int(c.get('x', 0)), int(c.get('y', 0))
            else:
                cx, cy = int(c[0]), int(c[1])
            cv2.circle(bgr, (cx, cy), marker_radius, marker_color, 2)

    if contours:
        cv2.drawContours(bgr, contours, -1, contour_color, 1)

    if text:
        for tx, ty, label in text:
            cv2.putText(bgr, str(label), (int(tx), int(ty)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    return bgr


def make_showcase(panels, experiment_id, description='', cols=None,
                  panel_size=(256, 256), results_text=None, output_dir=None):
    """Build and save a multi-panel showcase figure.

    Args:
        panels: List of dicts, each with:
            'image': 2D or BGR array.
            'title': Panel title string.
            Optional 'centroids': list for annotation.
            Optional 'contours': list for annotation.
            Optional 'scalebar': pixel_size_um for auto scale bar.
        experiment_id: Experiment identifier (e.g., 'exp_001' or descriptive name).
        description: Short description for filename.
        cols: Number of columns (default: auto based on panel count).
        panel_size: (width, height) to resize each panel.
        results_text: List of strings to show in a results panel at
            the bottom (e.g., ["Count: 42", "Diameter: 15.3 um"]).
        output_dir: Directory to save showcase figure. If None, uses
            get_showcase_dir() (respects MICROSCOPE_SHOWCASE_DIR env var).

    Returns:
        str: Path to saved showcase file.
    """
    if output_dir is None:
        output_dir = get_showcase_dir()
    os.makedirs(output_dir, exist_ok=True)

    n = len(panels)
    if cols is None:
        cols = min(n, 3)

    pw, ph = panel_size
    title_h = 25
    rows = (n + cols - 1) // cols

    # Results panel at bottom
    results_h = 0
    if results_text:
        results_h = 20 * len(results_text) + 15

    canvas_w = cols * pw
    canvas_h = rows * (ph + title_h) + results_h

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        x0 = c * pw
        y0 = r * (ph + title_h)

        # Process image
        img = panel['image']
        display = _scale_image(img)
        bgr = _to_bgr(display)

        # Annotate if needed
        if panel.get('centroids'):
            for pt in panel['centroids']:
                if isinstance(pt, dict):
                    cx, cy = int(pt.get('x', 0)), int(pt.get('y', 0))
                else:
                    cx, cy = int(pt[0]), int(pt[1])
                # Scale coordinates to panel size
                sx = cx * pw // img.shape[1] if img.shape[1] > 0 else cx
                sy = cy * ph // img.shape[0] if img.shape[0] > 0 else cy
                cv2.circle(bgr, (sx, sy), 4, (0, 255, 0), 1)

        # Resize to panel size
        resized = cv2.resize(bgr, (pw, ph))

        # Scale bar
        if panel.get('scalebar'):
            pxsz = panel['scalebar']
            scale_factor = img.shape[1] / pw
            resized = add_scalebar(resized, pxsz * scale_factor)

        # Title bar
        canvas[y0:y0 + title_h, x0:x0 + pw] = (50, 50, 50)
        title = panel.get('title', f'Panel {i}')
        cv2.putText(canvas, title, (x0 + 5, y0 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Image
        canvas[y0 + title_h:y0 + title_h + ph, x0:x0 + pw] = resized

    # Results text at bottom
    if results_text:
        results_y0 = rows * (ph + title_h)
        canvas[results_y0:, :] = (30, 30, 30)
        for j, line in enumerate(results_text):
            y = results_y0 + 15 + j * 20
            cv2.putText(canvas, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Save
    desc = description.replace(' ', '_')[:40] if description else 'result'
    filename = f'{experiment_id}_{desc}.png'
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, canvas)
    return path

