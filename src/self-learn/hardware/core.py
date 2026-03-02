"""Hardware abstraction for pymmcore-plus microscope control.

Provides thin wrappers with device-wait, state-check, and coordinate
conversion. All functions take ``core`` as their first argument and
discover hardware capabilities at runtime via MicroscopeConfig.

Works with any pymmcore-plus microscope (real or simulated).
"""

import warnings

import numpy as np

from .config import get_config, refresh_config, _mag_to_pixel_size


# ---------------------------------------------------------------------------
# Image acquisition
# ---------------------------------------------------------------------------

def snap(core, channel=None, exposure=None):
    """Snap an image, optionally switching channel/exposure first.

    Returns:
        np.ndarray: The captured image.
    """
    if channel is not None:
        cfg = get_config(core)
        if cfg.channel_group is not None:
            core.setConfig(cfg.channel_group, channel)
    if exposure is not None:
        core.setExposure(float(exposure))
    core.snapImage()
    return core.getImage().copy()


# ---------------------------------------------------------------------------
# Stage movement
# ---------------------------------------------------------------------------

def move_to(core, x, y, wait=True):
    """Move stage to (x, y) in world coordinates."""
    cfg = get_config(core)
    core.setXYPosition(float(x), float(y))
    if wait and cfg.xy_device:
        core.waitForDevice(cfg.xy_device)


def get_position(core):
    """Return current stage position as (x, y).

    Returns (0.0, 0.0) if no XY stage is available.
    """
    cfg = get_config(core)
    if cfg.xy_device is None:
        return 0.0, 0.0
    return core.getXPosition(), core.getYPosition()


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def set_objective(core, mag):
    """Set objective by magnification (e.g. 10, 20, 40, 100).

    Discovers available objectives at runtime and matches by parsing
    magnification from objective labels. Falls back to setting the
    "Label" property directly (e.g. "20x") if state-based lookup fails.

    Args:
        mag: Target magnification integer.
    """
    cfg = get_config(core)
    if cfg.objective_device is None:
        return
    state_idx = cfg.state_for_mag(mag)
    if state_idx is not None:
        core.setState(cfg.objective_device, state_idx)
        core.waitForDevice(cfg.objective_device)
        refresh_config(core)  # pixel_size changes with objective
        return
    # Fallback: try setting label directly (handles proxy quirks)
    label = f"{mag}x"
    try:
        allowed = core.getAllowedPropertyValues(cfg.objective_device, "Label")
        if label in allowed:
            core.setProperty(cfg.objective_device, "Label", label)
            core.waitForDevice(cfg.objective_device)
            refresh_config(core)  # pixel_size changes with objective
            return
    except Exception:
        pass
    available = [lbl for _, lbl in cfg.objective_labels]
    raise ValueError(
        f"No objective with magnification {mag}x found. "
        f"Available: {available}"
    )


def get_objective(core):
    """Return current objective magnification (e.g. 10, 20, 40).

    Returns None if no objective device is available.
    """
    cfg = get_config(core)
    return cfg.current_magnification(core)


def get_pixel_size(core) -> float:
    """Return current pixel size in micrometers.

    Always reflects the current objective — ``set_objective()``
    auto-refreshes the config, so this is always up to date.
    """
    cfg = get_config(core)
    return cfg.pixel_size_um


def fov_size(core):
    """Return current FOV size in world units (micrometers).

    Uses the config's pixel_size_um and image_width. After switching
    objectives, call ``refresh_config(core)`` if the pixel size changes.
    """
    cfg = get_config(core)
    return cfg.fov_width_um


# ---------------------------------------------------------------------------
# Z / Focus
# ---------------------------------------------------------------------------

def set_z(core, z, wait=True):
    """Set Z position."""
    cfg = get_config(core)
    if cfg.z_device is None:
        return
    core.setPosition(cfg.z_device, float(z))
    if wait:
        core.waitForDevice(cfg.z_device)


def get_z(core):
    """Return current Z position. Returns 0.0 if no Z device."""
    cfg = get_config(core)
    if cfg.z_device is None:
        return 0.0
    return core.getPosition(cfg.z_device)


# ---------------------------------------------------------------------------
# SLM
# ---------------------------------------------------------------------------

def make_slm_circle(center, radius, size=None, intensity=255, core=None):
    """Create a circular SLM mask in viewport coordinates.

    Args:
        center: (vx, vy) center of the circle in viewport pixels.
        radius: Radius in viewport pixels.
        size: SLM mask size. If None and core is provided, uses image width.
        intensity: Mask intensity (0-255).
        core: Optional core to auto-detect size.

    Returns:
        np.ndarray: uint8 mask of shape (size, size).
    """
    if size is None:
        if core is not None:
            cfg = get_config(core)
            size = cfg.image_width
        else:
            size = 512
    mask = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    dist_sq = (xx - center[0])**2 + (yy - center[1])**2
    mask[dist_sq <= radius**2] = intensity
    return mask


def apply_slm(core, mask, device=None):
    """Apply an SLM mask to the microscope.

    Args:
        core: pymmcore-plus core instance.
        mask: uint8 SLM array.
        device: SLM device name. If None, auto-discovers from config.
    """
    if device is None:
        cfg = get_config(core)
        device = cfg.slm_device
        if device is None:
            warnings.warn("No SLM device available")
            return
    core.setSLMImage(device, mask)
    core.displaySLMImage(device)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def world_to_viewport(world_x, world_y, stage_x, stage_y, viewport_size=512):
    """Convert world coordinates to viewport pixel coordinates.

    The viewport is centered on the current stage position.

    Args:
        world_x, world_y: World coordinates.
        stage_x, stage_y: Current stage position (center of viewport).
        viewport_size: Viewport size in pixels (default 512).

    Returns:
        (vx, vy): Viewport pixel coordinates.
    """
    half = viewport_size / 2
    vx = (world_x - stage_x) + half
    vy = (world_y - stage_y) + half
    return vx, vy


def pixel_to_world(px, py, stage_x, stage_y, config=None, core=None, mag=None):
    """Convert pixel coordinates to world coordinates.

    Uses config for image center and pixel size. Falls back to legacy
    formula (10/mag, center=256) if neither config nor core is provided.

    Args:
        px, py: Pixel coordinates in the image.
        stage_x, stage_y: Current stage position (world coords).
        config: MicroscopeConfig instance (preferred).
        core: Core instance (used to get config if config is None).
        mag: Magnification override (legacy fallback).

    Returns:
        (world_x, world_y) tuple.
    """
    if config is not None:
        cx, cy = config.image_center_px
        pixel_size = config.pixel_size_um
    elif core is not None:
        cfg = get_config(core)
        cx, cy = cfg.image_center_px
        pixel_size = cfg.pixel_size_um
    else:
        # Legacy fallback
        cx, cy = 256.0, 256.0
        pixel_size = _mag_to_pixel_size(mag) if mag else 1.0

    wx = stage_x + (px - cx) * pixel_size
    wy = stage_y + (py - cy) * pixel_size
    return round(wx, 1), round(wy, 1)


def world_to_pixel(wx, wy, stage_x, stage_y, config=None, core=None, mag=None):
    """Convert world coordinates to pixel coordinates.

    Args:
        wx, wy: World coordinates.
        stage_x, stage_y: Current stage position.
        config: MicroscopeConfig instance (preferred).
        core: Core instance (used to get config if config is None).
        mag: Magnification override (legacy fallback).

    Returns:
        (px, py) tuple.
    """
    if config is not None:
        cx, cy = config.image_center_px
        pixel_size = config.pixel_size_um
    elif core is not None:
        cfg = get_config(core)
        cx, cy = cfg.image_center_px
        pixel_size = cfg.pixel_size_um
    else:
        cx, cy = 256.0, 256.0
        pixel_size = _mag_to_pixel_size(mag) if mag else 1.0

    px_x = (wx - stage_x) / pixel_size + cx
    py_y = (wy - stage_y) / pixel_size + cy
    return round(px_x, 1), round(py_y, 1)


# ---------------------------------------------------------------------------
# MDA helpers
# ---------------------------------------------------------------------------

def run_events(core, events, on_frame=None):
    """Execute MDAEvent generators via core.mda.run() with optional frame callback.

    Works with both local CMMCorePlus and remote pymmcore-proxy. The MDA
    engine on the microscope (or server) handles all hardware sequencing,
    timing, stage moves, channel switching, and exposure. Frames are
    delivered locally via the frameReady signal (psygnal for local,
    WebSocket for proxy).

    CustomAction events are intercepted and handled locally before being
    passed to the engine, since they encode microscope-side logic (e.g.
    objective switches) that the MDA engine does not know about.
    Currently handled:
        CustomAction(name='switch_objective', data={'mag': <int>})

    Args:
        core: Microscope core (CMMCorePlus or RemoteMMCore proxy).
        events: Iterable of MDAEvent objects (generator, list, or MDASequence).
        on_frame: Optional callback(image, event) called after each frame.

    Returns:
        list of (image, event) tuples for all acquired frames.
    """
    frames = []

    def _handler(image, event, meta=None):
        frames.append((image, event))
        if on_frame is not None:
            on_frame(image, event)

    def _filtered(events):
        """Strip CustomAction events and execute them inline."""
        for event in events:
            action = getattr(event, 'action', None)
            if action is not None and type(action).__name__ == 'CustomAction':
                if action.name == 'switch_objective':
                    set_objective(core, action.data.get('mag', 10))
                # Don't forward to the engine — no image to acquire
            else:
                yield event

    core.mda.events.frameReady.connect(_handler)
    try:
        core.mda.run(_filtered(events))
    finally:
        core.mda.events.frameReady.disconnect(_handler)
    return frames
