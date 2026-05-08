"""MDA utilities for adaptive microscopy experiments.

Provides adaptive_phase_events() for phase-based generators with
early-stop support.

For all acquisitions (fixed or adaptive), use run_events() from
src.core.hardware.core — it delegates to core.mda.run() which works
identically for local CMMCorePlus and remote pymmcore-proxy.
"""

from typing import Any, Generator, Optional

from useq import MDAEvent


def adaptive_phase_events(
    name: str,
    max_frames: int,
    interval: float = 1.0,
    exposure: float = 50.0,
    shared_state: Optional[dict] = None,
    stop_key: str = "stop",
    t_offset: int = 0,
    metadata: Optional[dict[str, Any]] = None,
) -> Generator[MDAEvent, None, None]:
    """Yield MDAEvents for a phase that can be stopped early.

    The generator checks shared_state[stop_key] before each event.
    An on_frame callback should set shared_state[stop_key] = True
    when the phase should end.

    Args:
        name: Phase name.
        max_frames: Maximum number of frames.
        interval: Seconds between frames.
        exposure: Exposure time.
        shared_state: Mutable dict checked for stop condition.
        stop_key: Key in shared_state that signals stop.
        t_offset: Starting timepoint index.
        metadata: Extra metadata.

    Yields:
        MDAEvent until max_frames or stop condition.
    """
    state = shared_state if shared_state is not None else {}
    extra_meta = metadata or {}

    for t in range(max_frames):
        if state.get(stop_key, False):
            return

        yield MDAEvent(
            index={"t": t + t_offset},
            exposure=exposure,
            min_start_time=t * interval,
            metadata={"phase": name, **extra_meta},
        )


