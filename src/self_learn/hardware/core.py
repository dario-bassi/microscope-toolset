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


def snap_all_channels(core, exposure=None):
    """Snap an image in every available channel.

    Discovers channels from the microscope config and returns a dict
    mapping channel name to image array.  Useful for the initial
    step "snap ALL channels, look at each".

    Args:
        core: CMMCorePlus (or proxy) instance.
        exposure: Optional exposure time (ms) to use for all channels.
            If None, keeps the current exposure.

    Returns:
        dict mapping channel_name (str) → np.ndarray.
        Returns empty dict if no channel group is configured.
    """
    cfg = get_config(core)
    if cfg.channel_group is None or not cfg.available_channels:
        # No channel group — just snap a single image
        return {'default': snap(core, exposure=exposure)}

    images = {}
    for ch in cfg.available_channels:
        images[ch] = snap(core, channel=ch, exposure=exposure)
    return images


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


def set_objective_verified(core, mag, max_wait=2.0):
    """Set objective and verify the pixel size changed correctly.

    Calls :func:`set_objective` then polls ``get_pixel_size`` to confirm
    the expected pixel size is active. This catches silent failures where
    the hardware command completes but the objective didn't actually switch
    (e.g. asynchronous proxy, device busy).

    Args:
        mag: Target magnification (10, 20, 40, 100).
        max_wait: Maximum time in seconds to wait for verification.

    Raises:
        RuntimeError: If pixel size doesn't match expected value within
            *max_wait* seconds.
    """
    import time

    expected_ps = _mag_to_pixel_size(mag)
    set_objective(core, mag)

    deadline = time.time() + max_wait
    while time.time() < deadline:
        actual_ps = get_pixel_size(core)
        if abs(actual_ps - expected_ps) < 0.01:
            return
        time.sleep(0.1)

    actual_ps = get_pixel_size(core)
    if abs(actual_ps - expected_ps) < 0.01:
        return

    raise RuntimeError(
        f"Objective switch to {mag}x failed verification: "
        f"expected pixel_size={expected_ps:.3f} µm/px, "
        f"got {actual_ps:.3f} µm/px after {max_wait}s"
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


def dmd_on(core, mask=None, device=None):
    """Light the DMD/SLM so the sample can be imaged.

    A displayed pattern is held only for the device's ``ExposureTime``,
    then the DMD drops it and frames come back as pure camera noise. On the
    Andor Mosaic3 (``TriggerMode='InternalExpose'``) that is a single
    exposure of ``ExposureTime`` seconds — e.g. ~2 s, or ~120 s in some
    configs. So call this just before acquiring (snap within the hold
    window); to keep it lit continuously, refresh it at < ``ExposureTime``
    (the ``keep_dmd_alive`` MCP tool does this automatically).

    Args:
        core: pymmcore-plus core instance.
        mask: uint8 SLM array to display. If None, a full-bright mask
            (all mirrors on) sized to the device is used.
        device: SLM device name. If None, auto-discovers from config.

    Returns:
        str | None: the SLM device label that was lit, or None if no SLM
        device is available.
    """
    if device is None:
        device = get_config(core).slm_device
        if device is None:
            warnings.warn("No SLM/DMD device available")
            return None
    if mask is None:
        w, h = core.getSLMWidth(device), core.getSLMHeight(device)
        mask = np.full((h, w), 255, dtype=np.uint8)
    core.setSLMImage(device, mask)
    core.displaySLMImage(device)
    return device


def dmd_off(core, device=None):
    """Blank the DMD/SLM (all mirrors off) — display an all-zero mask.

    Args:
        core: pymmcore-plus core instance.
        device: SLM device name. If None, auto-discovers from config.

    Returns:
        str | None: the SLM device label, or None if no SLM is available.
    """
    if device is None:
        device = get_config(core).slm_device
        if device is None:
            warnings.warn("No SLM/DMD device available")
            return None
    w, h = core.getSLMWidth(device), core.getSLMHeight(device)
    core.setSLMImage(device, np.zeros((h, w), dtype=np.uint8))
    core.displaySLMImage(device)
    return device


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


def pixel_to_world(px, py, stage_x, stage_y, config=None, core=None,
                    mag=None, pixel_size=None):
    """Convert pixel coordinates to world coordinates.

    Uses config for image center and pixel size. Falls back to legacy
    formula (10/mag, center=256) if neither config nor core is provided.

    Args:
        px, py: Pixel coordinates in the image.
        stage_x, stage_y: Current stage position (world coords).
        config: MicroscopeConfig instance (preferred).
        core: Core instance (used to get config if config is None).
        mag: Magnification override (legacy fallback).
        pixel_size: Direct pixel size in µm/px (overrides mag).

    Returns:
        (world_x, world_y) tuple.
    """
    if config is not None:
        cx, cy = config.image_center_px
        ps = config.pixel_size_um
    elif core is not None:
        cfg = get_config(core)
        cx, cy = cfg.image_center_px
        ps = cfg.pixel_size_um
    else:
        # Legacy fallback
        cx, cy = 256.0, 256.0
        if pixel_size is not None:
            ps = pixel_size
        elif mag is not None:
            ps = _mag_to_pixel_size(mag)
        else:
            ps = 1.0

    wx = stage_x + (px - cx) * ps
    wy = stage_y + (py - cy) * ps
    return round(wx, 1), round(wy, 1)


def world_to_pixel(wx, wy, stage_x, stage_y, config=None, core=None,
                    mag=None, pixel_size=None):
    """Convert world coordinates to pixel coordinates.

    Args:
        wx, wy: World coordinates.
        stage_x, stage_y: Current stage position.
        config: MicroscopeConfig instance (preferred).
        core: Core instance (used to get config if config is None).
        mag: Magnification override (legacy fallback).
        pixel_size: Direct pixel size in µm/px (overrides mag).

    Returns:
        (px, py) tuple.
    """
    if config is not None:
        cx, cy = config.image_center_px
        ps = config.pixel_size_um
    elif core is not None:
        cfg = get_config(core)
        cx, cy = cfg.image_center_px
        ps = cfg.pixel_size_um
    else:
        cx, cy = 256.0, 256.0
        if pixel_size is not None:
            ps = pixel_size
        elif mag is not None:
            ps = _mag_to_pixel_size(mag)
        else:
            ps = 1.0

    px_x = (wx - stage_x) / ps + cx
    py_y = (wy - stage_y) / ps + cy
    return round(px_x, 1), round(py_y, 1)


# ---------------------------------------------------------------------------
# MDA helpers
# ---------------------------------------------------------------------------

def run_events(core, events, on_frame=None, collect=True):
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

    If MDA engine fails (e.g. device configuration issues on the server),
    automatically falls back to snap-based acquisition with manual channel
    switching and timing.

    **Generator support:** Generators are processed lazily (one event at a
    time) to preserve side effects between yields (e.g., SLM device
    control). Lists and other iterables are materialized so they can be
    retried with the MDA engine fallback.

    Args:
        core: Microscope core (CMMCorePlus or RemoteMMCore proxy).
        events: Iterable of MDAEvent objects (generator, list, or MDASequence).
        on_frame: Optional callback(image, event) called after each frame.
        collect: If True (default), accumulate every (image, event) and return
            them. Set False for large scans that stream each frame to disk in
            ``on_frame`` — avoids holding hundreds of full frames in RAM.

    Returns:
        list of (image, event) tuples for all acquired frames (empty if
        ``collect`` is False).
    """
    import types

    def _filtered(events):
        """Strip CustomAction events and execute them inline."""
        for event in events:
            action = getattr(event, 'action', None)
            if action is not None and type(action).__name__ == 'CustomAction':
                if action.name == 'switch_objective':
                    set_objective(core, action.data.get('mag', 10))
            else:
                yield event

    # Generators may have side effects between yields (e.g., SLM control).
    # Process them lazily via manual acquisition to preserve those effects.
    is_generator = isinstance(events, types.GeneratorType)

    if is_generator:
        return _manual_run(core, _filtered(events), on_frame, collect=collect)

    # For lists/tuples/MDASequence: use the MDA engine.
    event_list = list(events)
    frames = []

    def _handler(image, event, meta=None):
        if collect:
            frames.append((image, event))
        if on_frame is not None:
            on_frame(image, event)

    try:
        core.mda.events.frameReady.connect(_handler)
        try:
            core.mda.run(_filtered(event_list))
        finally:
            core.mda.events.frameReady.disconnect(_handler)
        return frames
    except Exception as e:
        err = str(e)
        _retriable = (
            "No device with label" in err
            or "websocket" in err.lower()
            or "connection" in err.lower()
            or "websockets" in type(e).__module__
            or type(e).__name__ in ("ConnectionClosedError", "ConnectionClosedOK",
                                    "WebSocketDisconnect", "TimeoutError")
        )
        if not _retriable:
            raise
        import warnings
        warnings.warn(f"MDA engine error ({type(e).__name__}: {e}) — retrying with manual acquisition")

    return _manual_run(core, _filtered(event_list), on_frame, collect=collect)


def _manual_run(core, events, on_frame=None, collect=True):
    """Manual snap-based acquisition fallback.

    Processes events lazily (one at a time), preserving any side effects
    the caller may have between event yields (e.g., SLM device control).
    """
    import time
    frames = []
    t0 = time.time()
    for event in events:
        # Apply timing
        min_start = getattr(event, 'min_start_time', None)
        if min_start is not None:
            target = t0 + min_start
            now = time.time()
            if now < target:
                time.sleep(target - now)

        # Apply channel
        ch = getattr(event, 'channel', None)
        if ch is not None:
            group = getattr(ch, 'group', None)
            config = getattr(ch, 'config', None)
            if group and config:
                core.setConfig(group, config)
                core.waitForConfig(group, config)

        # Apply exposure
        exposure = getattr(event, 'exposure', None)
        if exposure is not None:
            core.setExposure(exposure)

        # Apply stage position
        x_pos = getattr(event, 'x_pos', None)
        y_pos = getattr(event, 'y_pos', None)
        if x_pos is not None and y_pos is not None:
            core.setXYPosition(x_pos, y_pos)
            core.waitForDevice(core.getXYStageDevice())

        z_pos = getattr(event, 'z_pos', None)
        if z_pos is not None:
            core.setPosition(z_pos)
            core.waitForDevice(core.getFocusDevice())

        # Snap and collect
        core.snapImage()
        image = core.getImage()
        if collect:
            frames.append((image.copy(), event))
        if on_frame is not None:
            on_frame(image, event)

    return frames


def timelapse(core, n_frames, interval_s=1.0, channel=None, exposure=None,
              on_frame=None):
    """Acquire a timelapse via the MDA engine.

    Convenience wrapper around :func:`run_events` that handles the most
    common acquisition pattern: N frames at fixed interval in one channel.
    Returns a 3D numpy stack (T, H, W) ready for analysis.

    Args:
        core: Microscope core (CMMCorePlus or proxy).
        n_frames: Number of frames to acquire.
        interval_s: Interval between frames in seconds.
        channel: Channel config name (e.g. 'GFP', 'brightfield').
            If None, uses whatever is currently set.
        exposure: Exposure time in ms. If None, uses current.
        on_frame: Optional callback(image, event) per frame.

    Returns:
        np.ndarray: 3D array (T, H, W) of acquired frames.
    """
    from useq import MDASequence

    seq_kwargs = {
        'time_plan': {'loops': int(n_frames), 'interval': float(interval_s)},
    }
    if channel is not None:
        cfg = get_config(core)
        from .config import resolve_channel_group
        group = cfg.channel_group or resolve_channel_group(core, None)
        seq_kwargs['channels'] = [{'config': channel, 'group': group}]
    if exposure is not None:
        if 'channels' in seq_kwargs:
            seq_kwargs['channels'][0]['exposure'] = float(exposure)

    seq = MDASequence(**seq_kwargs)
    frames = run_events(core, list(seq), on_frame=on_frame)
    return np.array([f[0] for f in frames])
