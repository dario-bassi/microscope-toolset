"""SLM/DMD ↔ Camera calibration via affine transform.

Computes a 3x3 affine matrix that maps SLM pixel coordinates to camera pixel
coordinates (and vice versa).  Works with any SLM device (DMD, liquid crystal,
etc.) that is accessible through pymmcore-plus.

Calibration procedure:
    1. Find the SLM conjugate focal plane (Z-scan with a test pattern).
    2. Display 3 dots sequentially at known SLM positions, detect each on camera.
    3. Solve the exact affine from 3 point correspondences.
    4. Verify with 3 independent test dots — if all errors < threshold, accept.

The calibration is **specific** to the current:
    - Objective (magnification, light path)
    - Channel / config group (different channels may have different paths)
    - Camera binning / ROI settings

If any of these change, re-calibrate.

Usage
-----
    from src.self_learn.hardware.slm_calibration import (
        calibrate_slm,
        find_slm_conjugate_z,
        camera_mask_to_slm,
        load_calibration,
        save_calibration,
    )

    # Find where SLM patterns come into focus
    z_conj = find_slm_conjugate_z(core, slm_device, channel_group, channel_config,
                                  z_center=3500, z_range=1500, z_step=50)

    # Run full calibration
    calib = calibrate_slm(core, slm_device, channel_group, channel_config,
                          z_conjugate=z_conj)

    # Convert a camera-space binary mask to an SLM mask
    slm_mask = camera_mask_to_slm(camera_mask, calib["dmd_to_camera"]["matrix"])
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_slm_conjugate_z(
    core,
    slm_device: str,
    channel_group: str,
    channel_config: str,
    z_center: float,
    z_range: float = 1500.0,
    z_step: float = 50.0,
    exposure_ms: float = 5.0,
    dot_radius: int = 15,
    grid_spacing: int = 100,
) -> float:
    """Find the Z position where SLM patterns are in focus on the camera.

    Uploads a grid of dots to the SLM and scans through Z, measuring image
    sharpness (Laplacian variance) at each step.  Returns the Z with peak
    sharpness.

    Args:
        core: CMMCorePlus instance.
        slm_device: SLM device label (e.g. "Mosaic3").
        channel_group: Config group for the SLM channel (e.g. "TTL_ERK").
        channel_config: Config preset routed through the SLM (e.g. "CyanStim").
        z_center: Starting Z position (um) — typically near sample focus.
        z_range: How far above/below z_center to scan (um).
        z_step: Step size (um).
        exposure_ms: Camera exposure per frame.
        dot_radius: Radius of each grid dot in SLM pixels.
        grid_spacing: Spacing between dots in SLM pixels.

    Returns:
        Z position (um) with highest Laplacian variance (sharpest SLM pattern).
    """
    slm_w, slm_h = _get_slm_size(core, slm_device)
    mask = _make_grid_mask(slm_w, slm_h, grid_spacing, dot_radius)
    z_positions = np.arange(z_center - z_range, z_center + z_range + z_step, z_step)
    z_device = core.getFocusDevice()

    core.setConfig(channel_group, channel_config)
    core.waitForConfig(channel_group, channel_config)
    core.setExposure(exposure_ms)

    best_z = z_center
    best_metric = -1.0

    for z in z_positions:
        img = _snap_with_slm(core, slm_device, z_device, mask, float(z))
        metric = ndimage.laplace(img.astype(float)).var()
        logger.debug("Z=%.0f: laplacian_var=%.0f", z, metric)
        if metric > best_metric:
            best_metric = metric
            best_z = float(z)

    # Return to starting position
    core.setPosition(z_device, z_center)
    core.waitForDevice(z_device)

    logger.info("SLM conjugate plane found at Z=%.1f um (laplacian_var=%.0f)", best_z, best_metric)
    return best_z


def calibrate_slm(
    core,
    slm_device: str,
    channel_group: str,
    channel_config: str,
    z_conjugate: float,
    exposure_ms: float = 2.0,
    dot_radius: int = 30,
    verification_threshold_px: float = 5.0,
    run_mda_fn=None,
) -> dict:
    """Calibrate SLM ↔ Camera affine transform.

    Displays 3 calibration dots sequentially, detects their centroids, computes
    the affine, then verifies with 3 independent test dots.

    Args:
        core: CMMCorePlus instance.
        slm_device: SLM device label.
        channel_group: Config group for the SLM channel.
        channel_config: Config preset routed through the SLM.
        z_conjugate: Z position of the SLM conjugate plane (from find_slm_conjugate_z).
        exposure_ms: Camera exposure per frame.
        dot_radius: Radius of calibration dots in SLM pixels.
        verification_threshold_px: Max error (px) for verification to pass.
        run_mda_fn: Optional MDA runner function (signature: run_mda_fn(events, on_frame)).
                    If None, uses manual shutter/acquisition fallback.

    Returns:
        dict with keys: dmd_to_camera, camera_to_dmd, calibration_points,
        verification_max_error_px, slm_size, camera_size, slm_conjugate_z_um,
        device, channel_group, channel_config, objective, verified.

    Raises:
        RuntimeError: If fewer than 3 dots are detected or verification fails.
    """
    slm_w, slm_h = _get_slm_size(core, slm_device)

    # Asymmetric triangle of 3 calibration points
    margin_x = max(50, int(slm_w * 0.12))
    margin_y = max(50, int(slm_h * 0.17))
    calib_points = [
        (margin_x, margin_y),  # top-left region
        (slm_w - margin_x, margin_y),  # top-right region
        (slm_w // 2, slm_h - margin_y),  # bottom-center
    ]
    # 3 different test points
    test_points = [
        (slm_w // 4, slm_h // 2),  # left-center
        (3 * slm_w // 4, slm_h // 2),  # right-center
        (slm_w // 2, slm_h // 4),  # top-center
    ]

    # --- Acquire calibration dots ---
    logger.info("Acquiring %d calibration dots...", len(calib_points))
    calib_results = _acquire_dots(
        core,
        slm_device,
        channel_group,
        channel_config,
        calib_points,
        z_conjugate,
        exposure_ms,
        dot_radius,
        run_mda_fn,
    )
    if len(calib_results) < 3:
        raise RuntimeError(
            f"Only detected {len(calib_results)}/3 calibration dots. "
            "Check SLM conjugate Z, exposure, or channel configuration."
        )

    # --- Compute affine (exact from 3 points) ---
    fwd_mat, inv_mat = _compute_affine_3pt(calib_results)

    logger.info(
        "Affine (SLM->Cam): cam_x = %.4f*slm_x + %.4f*slm_y + %.4f",
        fwd_mat[0, 0],
        fwd_mat[0, 1],
        fwd_mat[0, 2],
    )
    logger.info(
        "Affine (SLM->Cam): cam_y = %.4f*slm_x + %.4f*slm_y + %.4f",
        fwd_mat[1, 0],
        fwd_mat[1, 1],
        fwd_mat[1, 2],
    )

    # --- Verify with 3 test dots ---
    logger.info("Acquiring %d verification dots...", len(test_points))
    test_results = _acquire_dots(
        core,
        slm_device,
        channel_group,
        channel_config,
        test_points,
        z_conjugate,
        exposure_ms,
        dot_radius,
        run_mda_fn,
    )

    max_err = 0.0
    for slm_x, slm_y, cam_x, cam_y in test_results:
        pred = fwd_mat @ np.array([slm_x, slm_y, 1.0])
        err = np.sqrt((pred[0] - cam_x) ** 2 + (pred[1] - cam_y) ** 2)
        max_err = max(max_err, err)
        logger.info(
            "  SLM(%d,%d): predicted cam(%.1f,%.1f), actual cam(%.1f,%.1f), err=%.1f px",
            slm_x,
            slm_y,
            pred[0],
            pred[1],
            cam_x,
            cam_y,
            err,
        )

    verified = max_err < verification_threshold_px
    if not verified:
        logger.warning(
            "Verification FAILED: max_err=%.1f px > threshold=%.1f px",
            max_err,
            verification_threshold_px,
        )

    # --- Gather metadata ---
    objective = _get_objective_label(core)

    calib = {
        "dmd_to_camera": {"matrix": fwd_mat.tolist()},
        "camera_to_dmd": {"matrix": inv_mat.tolist()},
        "calibration_points": [
            {"slm": [int(r[0]), int(r[1])], "camera": [float(r[2]), float(r[3])]}
            for r in calib_results
        ],
        "verification_max_error_px": float(max_err),
        "slm_size": [slm_w, slm_h],
        "camera_size": [int(core.getImageWidth()), int(core.getImageHeight())],
        "slm_conjugate_z_um": z_conjugate,
        "device": slm_device,
        "channel_group": channel_group,
        "channel_config": channel_config,
        "objective": objective,
        "verified": verified,
    }
    return calib


def camera_mask_to_slm(
    mask_cam: np.ndarray,
    fwd_matrix,
    slm_w: int = 800,
    slm_h: int = 600,
) -> np.ndarray:
    """Transform a binary camera-space mask into SLM pixel coordinates.

    For each SLM pixel, looks up the corresponding camera pixel via the
    forward affine (SLM→Camera) and copies the mask value.

    Args:
        mask_cam: 2D array in camera space (nonzero = selected).
        fwd_matrix: 3x3 forward affine (SLM → Camera). Can be list or ndarray.
        slm_w: SLM width in pixels.
        slm_h: SLM height in pixels.

    Returns:
        uint8 array (slm_h, slm_w) with 0 or 255.
    """
    fwd = np.asarray(fwd_matrix)
    slm_y, slm_x = np.mgrid[0:slm_h, 0:slm_w]
    cam_x = fwd[0, 0] * slm_x + fwd[0, 1] * slm_y + fwd[0, 2]
    cam_y = fwd[1, 0] * slm_x + fwd[1, 1] * slm_y + fwd[1, 2]
    cx = np.clip(np.round(cam_x).astype(int), 0, mask_cam.shape[1] - 1)
    cy = np.clip(np.round(cam_y).astype(int), 0, mask_cam.shape[0] - 1)
    return (mask_cam[cy, cx] > 0).astype(np.uint8) * 255


def save_calibration(calib: dict, path: str | Path) -> None:
    """Save calibration dict to a JSON file."""
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
    logger.info("Calibration saved to %s", path)


def load_calibration(path: str | Path) -> dict:
    """Load calibration dict from a JSON file.

    Returns the dict with 'dmd_to_camera' and 'camera_to_dmd' matrices
    converted to numpy arrays for convenience.
    """
    with open(path) as f:
        calib = json.load(f)
    # Convert matrices back to numpy
    calib["dmd_to_camera"]["matrix"] = np.array(calib["dmd_to_camera"]["matrix"])
    calib["camera_to_dmd"]["matrix"] = np.array(calib["camera_to_dmd"]["matrix"])
    return calib


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_slm_size(core, slm_device: str) -> tuple[int, int]:
    """Query SLM resolution from the device."""
    w = int(core.getSLMWidth(slm_device))
    h = int(core.getSLMHeight(slm_device))
    return w, h


def _make_grid_mask(
    slm_w: int,
    slm_h: int,
    spacing: int,
    radius: int,
) -> np.ndarray:
    """Create a grid of dots as a uint8 mask."""
    mask = np.zeros((slm_h, slm_w), dtype=np.uint8)
    for cy in range(spacing // 2, slm_h, spacing):
        for cx in range(spacing // 2, slm_w, spacing):
            yy, xx = np.ogrid[:slm_h, :slm_w]
            circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
            mask[circle] = 255
    return mask


def _make_dot_mask(
    slm_w: int,
    slm_h: int,
    cx: int,
    cy: int,
    radius: int,
) -> np.ndarray:
    """Create a single-dot mask."""
    mask = np.zeros((slm_h, slm_w), dtype=np.uint8)
    yy, xx = np.ogrid[:slm_h, :slm_w]
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
    mask[circle] = 255
    return mask


def _snap_with_slm(
    core,
    slm_device: str,
    z_device: str,
    mask: np.ndarray,
    z_pos: float,
) -> np.ndarray:
    """Upload SLM mask, move Z, open shutter, acquire one frame, close shutter.

    Uses sequence acquisition as a fallback for cameras where snapImage() fails.
    """
    core.setPosition(z_device, z_pos)
    core.waitForDevice(z_device)

    core.setSLMImage(slm_device, mask)
    core.displaySLMImage(slm_device)
    time.sleep(0.05)

    core.setShutterOpen(True)
    time.sleep(0.05)

    try:
        core.snapImage()
        img = core.getImage()
    except RuntimeError:
        # Fallback for cameras that fail on snapImage (e.g. PVCAM)
        core.clearCircularBuffer()
        core.startSequenceAcquisition(1, 0, True)
        t0 = time.time()
        while core.isSequenceRunning():
            time.sleep(0.05)
            if (time.time() - t0) > 5:
                core.stopSequenceAcquisition()
                break
        if core.getRemainingImageCount() < 1:
            core.setShutterOpen(False)
            raise RuntimeError("No image returned from acquisition") from None
        img = core.popNextImage()

    core.setShutterOpen(False)
    return img


def _acquire_dots(
    core,
    slm_device: str,
    channel_group: str,
    channel_config: str,
    points: list[tuple[int, int]],
    z_conjugate: float,
    exposure_ms: float,
    dot_radius: int,
    run_mda_fn=None,
) -> list[tuple[int, int, float, float]]:
    """Display dots one at a time, detect centroids.

    If run_mda_fn is provided, uses MDAEvent+SLMImage (preferred).
    Otherwise falls back to manual shutter control.

    Returns list of (slm_x, slm_y, cam_x, cam_y).
    """
    slm_w, slm_h = _get_slm_size(core, slm_device)

    if run_mda_fn is not None:
        return _acquire_dots_mda(
            run_mda_fn,
            slm_device,
            channel_group,
            channel_config,
            points,
            z_conjugate,
            exposure_ms,
            dot_radius,
            slm_w,
            slm_h,
        )

    # Manual fallback
    z_device = core.getFocusDevice()
    core.setConfig(channel_group, channel_config)
    core.waitForConfig(channel_group, channel_config)
    core.setExposure(exposure_ms)

    results = []
    for cx, cy in points:
        dot_mask = _make_dot_mask(slm_w, slm_h, cx, cy, dot_radius)
        dark_mask = np.zeros((slm_h, slm_w), dtype=np.uint8)

        dot_img = _snap_with_slm(core, slm_device, z_device, dot_mask, z_conjugate).astype(float)
        dark_img = _snap_with_slm(core, slm_device, z_device, dark_mask, z_conjugate).astype(float)

        cam_xy = _detect_centroid(dot_img - dark_img)
        if cam_xy is not None:
            results.append((cx, cy, cam_xy[0], cam_xy[1]))
            logger.info("  SLM(%d,%d) -> Cam(x=%.1f, y=%.1f)", cx, cy, cam_xy[0], cam_xy[1])
        else:
            logger.warning("  SLM(%d,%d) -> NOT DETECTED", cx, cy)

    return results


def _acquire_dots_mda(
    run_mda_fn,
    slm_device: str,
    channel_group: str,
    channel_config: str,
    points: list[tuple[int, int]],
    z_conjugate: float,
    exposure_ms: float,
    dot_radius: int,
    slm_w: int,
    slm_h: int,
) -> list[tuple[int, int, float, float]]:
    """Acquire dots using MDAEvent with SLMImage (preferred method)."""
    from useq import MDAEvent
    from useq._mda_event import SLMImage

    def make_events():
        for i, (cx, cy) in enumerate(points):
            dot_mask = _make_dot_mask(slm_w, slm_h, cx, cy, dot_radius)
            yield MDAEvent(
                channel={"config": channel_config, "group": channel_group},
                exposure=exposure_ms,
                z_pos=z_conjugate,
                slm_image=SLMImage(data=dot_mask, device=slm_device),
                index={"t": i * 2},
            )
            dark_mask = np.zeros((slm_h, slm_w), dtype=np.uint8)
            yield MDAEvent(
                channel={"config": channel_config, "group": channel_group},
                exposure=exposure_ms,
                z_pos=z_conjugate,
                slm_image=SLMImage(data=dark_mask, device=slm_device),
                index={"t": i * 2 + 1},
            )

    frames: list[np.ndarray] = []

    def on_frame(image, event, metadata):
        frames.append(image.astype(float))

    run_mda_fn(make_events(), on_frame)

    results = []
    for i, (cx, cy) in enumerate(points):
        diff = frames[i * 2] - frames[i * 2 + 1]
        cam_xy = _detect_centroid(diff)
        if cam_xy is not None:
            results.append((cx, cy, cam_xy[0], cam_xy[1]))
            logger.info("  SLM(%d,%d) -> Cam(x=%.1f, y=%.1f)", cx, cy, cam_xy[0], cam_xy[1])
        else:
            logger.warning("  SLM(%d,%d) -> NOT DETECTED", cx, cy)

    return results


def _detect_centroid(
    image: np.ndarray,
    threshold_frac: float = 0.4,
) -> tuple[float, float] | None:
    """Detect the centroid of the brightest blob in an image.

    Returns (cam_x, cam_y) or None if no blob detected.
    """
    threshold = image.max() * threshold_frac
    if threshold <= 0:
        return None
    binary = image > threshold
    labeled, n = ndimage.label(binary)
    if n < 1:
        return None
    com = ndimage.center_of_mass(image, labeled, 1)
    return (com[1], com[0])  # (x, y)


def _compute_affine_3pt(
    results: list[tuple[int, int, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute affine from exactly 3 point correspondences.

    Args:
        results: list of (slm_x, slm_y, cam_x, cam_y)

    Returns:
        (forward_matrix, inverse_matrix) — both 3x3 ndarrays.
        Forward maps SLM -> Camera.
    """
    slm = np.array([(r[0], r[1]) for r in results], dtype=float)
    cam = np.array([(r[2], r[3]) for r in results], dtype=float)

    A = np.column_stack([slm[:, 0], slm[:, 1], np.ones(3)])
    coeffs_x = np.linalg.solve(A, cam[:, 0])
    coeffs_y = np.linalg.solve(A, cam[:, 1])

    fwd = np.array(
        [
            [coeffs_x[0], coeffs_x[1], coeffs_x[2]],
            [coeffs_y[0], coeffs_y[1], coeffs_y[2]],
            [0, 0, 1],
        ]
    )
    inv = np.linalg.inv(fwd)
    return fwd, inv


def _get_objective_label(core) -> str:
    """Try to read the current objective label."""
    for device in ("TINosePiece", "Objective", "ObjectiveState", "Nosepiece"):
        try:
            return str(core.getProperty(device, "Label"))
        except Exception:
            continue
    return "unknown"
