"""Multi-phase experiment protocol builder.

Constructs MDA-based acquisition plans for common experimental designs:
baseline → intervention → recovery patterns with device state changes
between phases.

Functions:
    phase_timelapse     -- Run a multi-phase timelapse with device changes
    baseline_treatment  -- Standard baseline → treatment → observation protocol
    wash_experiment     -- Perfusion/wash-in/wash-out protocol
    temperature_shift   -- Temperature change experiment
"""

from useq import MDAEvent


def phase_timelapse(phases, channels, on_frame=None):
    """Generate MDA events for a multi-phase timelapse experiment.

    Each phase specifies duration, interval, and optional device state
    changes (temperature, perfusion, SLM, etc.).

    Args:
        phases: list of dicts, each with:
            name: str, phase label (e.g. 'baseline', 'treatment').
            n_frames: int, number of timepoints in this phase.
            interval: float, seconds between frames.
            devices: dict, optional device state changes to apply
                at phase start. Keys are (device, property) tuples or
                state device names, values are the settings.
                Example: {'Temperature': '37'} or
                         {('Camera', 'Gain'): 4.0}
            wait_before: float, seconds to wait before starting phase.
        channels: list of channel dicts for MDAEvent, e.g.
            [{'config': 'GFP', 'group': 'Fake', 'exposure': 50}]
        on_frame: optional callback(image, event, phase_info) for
            real-time analysis. phase_info is dict with phase name
            and frame index within phase.

    Yields:
        MDAEvent objects for the full experiment.

    Returns metadata in each event:
        event.metadata['phase'] = phase name
        event.metadata['phase_index'] = index in phases list
        event.metadata['frame_in_phase'] = frame number within phase
    """
    global_time = 0.0

    for phase_idx, phase in enumerate(phases):
        name = phase["name"]
        n_frames = phase["n_frames"]
        interval = phase.get("interval", 1.0)
        wait_before = phase.get("wait_before", 0.0)

        global_time += wait_before

        # Device state changes at phase start — converted to native event.properties
        devices = phase.get("devices", {})
        phase_properties = []  # list of (device, prop, value) tuples for first frame
        for key, val in devices.items():
            if isinstance(key, tuple):
                # (device, property) → set property
                phase_properties.append((key[0], key[1], str(val)))
            else:
                # State device — set via Label property
                phase_properties.append((key, "Label", str(val)))

        for frame in range(n_frames):
            for ch in channels:
                event_kwargs = {
                    "min_start_time": global_time,
                    "metadata": {
                        "phase": name,
                        "phase_index": phase_idx,
                        "frame_in_phase": frame,
                    },
                }
                # Apply device state changes only on the first frame of each phase
                if frame == 0 and phase_properties:
                    event_kwargs["properties"] = phase_properties
                if "config" in ch:
                    event_kwargs["channel"] = ch
                if "exposure" in ch and "channel" not in event_kwargs:
                    event_kwargs["exposure"] = ch["exposure"]

                yield MDAEvent(**event_kwargs)

            global_time += interval


def baseline_treatment(
    n_baseline,
    n_treatment,
    interval=1.0,
    channels=None,
    treatment_devices=None,
    wait_after_treatment=0.0,
):
    """Standard baseline → treatment experiment protocol.

    Args:
        n_baseline: int, number of baseline frames.
        n_treatment: int, number of post-treatment frames.
        interval: float, seconds between frames.
        channels: list of channel dicts. Default: [{'config': 'brightfield'}]
        treatment_devices: dict of device changes at treatment start.
        wait_after_treatment: float, seconds to wait after applying
            treatment before starting observation.

    Returns:
        list of phases suitable for phase_timelapse().
    """
    if channels is None:
        channels = [{"config": "brightfield"}]

    phases = [
        {
            "name": "baseline",
            "n_frames": n_baseline,
            "interval": interval,
        },
        {
            "name": "treatment",
            "n_frames": n_treatment,
            "interval": interval,
            "devices": treatment_devices or {},
            "wait_before": wait_after_treatment,
        },
    ]
    return phases


def wash_experiment(
    n_baseline,
    n_wash_in,
    n_wash_out,
    interval=1.0,
    channels=None,
    wash_in_devices=None,
    wash_out_devices=None,
):
    """Perfusion wash-in / wash-out experiment protocol.

    Three phases: baseline → wash-in (apply drug/compound) → wash-out
    (remove drug/compound and observe recovery).

    Args:
        n_baseline: int, number of baseline frames.
        n_wash_in: int, frames during drug application.
        n_wash_out: int, frames during washout/recovery.
        interval: float, seconds between frames.
        channels: list of channel dicts.
        wash_in_devices: dict of device changes for wash-in.
        wash_out_devices: dict of device changes for wash-out.

    Returns:
        list of phases suitable for phase_timelapse().
    """
    if channels is None:
        channels = [{"config": "brightfield"}]

    phases = [
        {
            "name": "baseline",
            "n_frames": n_baseline,
            "interval": interval,
        },
        {
            "name": "wash_in",
            "n_frames": n_wash_in,
            "interval": interval,
            "devices": wash_in_devices or {},
        },
        {
            "name": "wash_out",
            "n_frames": n_wash_out,
            "interval": interval,
            "devices": wash_out_devices or {},
        },
    ]
    return phases


def temperature_shift(
    n_baseline,
    n_shifted,
    n_recovery=0,
    interval=1.0,
    baseline_temp="37",
    shift_temp="4",
    recovery_temp=None,
    channels=None,
):
    """Temperature shift experiment protocol.

    Args:
        n_baseline: int, frames at baseline temperature.
        n_shifted: int, frames at shifted temperature.
        n_recovery: int, frames during recovery (0 to skip).
        interval: float, seconds between frames.
        baseline_temp: str, Temperature state label for baseline.
        shift_temp: str, Temperature state label for shift.
        recovery_temp: str or None, label for recovery (default=baseline).
        channels: list of channel dicts.

    Returns:
        list of phases suitable for phase_timelapse().
    """
    if recovery_temp is None:
        recovery_temp = baseline_temp
    if channels is None:
        channels = [{"config": "brightfield"}]

    phases = [
        {
            "name": "baseline",
            "n_frames": n_baseline,
            "interval": interval,
            "devices": {"Temperature": baseline_temp},
        },
        {
            "name": "temperature_shift",
            "n_frames": n_shifted,
            "interval": interval,
            "devices": {"Temperature": shift_temp},
        },
    ]

    if n_recovery > 0:
        phases.append(
            {
                "name": "recovery",
                "n_frames": n_recovery,
                "interval": interval,
                "devices": {"Temperature": recovery_temp},
            }
        )

    return phases


def extract_phase_data(results, events, phase_name):
    """Extract images and events from a specific phase.

    Args:
        results: list of (image, event) tuples from run_events.
        events: list of MDAEvents (with metadata).
        phase_name: str, phase to extract.

    Returns:
        dict with:
            images: list of np.ndarray.
            frame_indices: list of int (frame_in_phase).
            times: list of float (min_start_time).
            n_frames: int.
    """
    images = []
    frame_indices = []
    times = []

    for img, evt in results:
        md = getattr(evt, "metadata", {}) or {}
        if md.get("phase") == phase_name:
            images.append(img)
            frame_indices.append(md.get("frame_in_phase", 0))
            times.append(getattr(evt, "min_start_time", 0))

    return {
        "images": images,
        "frame_indices": frame_indices,
        "times": times,
        "n_frames": len(images),
    }
