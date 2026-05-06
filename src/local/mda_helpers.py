"""Helpers for generator-based MDA (Multi-Dimensional Acquisition) with feedback.

Provides `run_mda_with_feedback()` which wraps pymmcore-plus's `run_mda()` with
a per-frame callback for synchronous delivery on the MDA thread.

napari-micromanager's _mda_handler natively handles generator-based sequences
(GeneratorMDASequence skip in _on_mda_started, preview layer in _on_mda_frame).
"""

import contextlib
import logging
from collections.abc import Callable, Iterable

import numpy as np
from useq import MDAEvent

logger = logging.getLogger("MDAHelpers")


def run_mda_with_feedback(
    mmc,
    events: Iterable[MDAEvent],
    on_frame: Callable[[np.ndarray, MDAEvent, dict], None] | None = None,
) -> list[tuple[np.ndarray, MDAEvent, dict]]:
    """Run MDA with per-frame callback.

    Args:
        mmc: CMMCorePlus / UniMMCore instance (the raw core, not GatekeeperCore).
        events: Iterable of MDAEvent — can be a list or a generator.
            Generators enable feedback loops: the on_frame callback can modify
            shared state that the generator reads when yielding the next event.
        on_frame: Optional callback(image, event, metadata) called synchronously
            on the MDA thread for each acquired frame. If None, frames are
            collected into a list and returned.

    Returns:
        If on_frame is None: list of (image, event, metadata) tuples.
        If on_frame is provided: empty list (callback handles data).
    """
    frames: list[tuple[np.ndarray, MDAEvent, dict]] = []

    def _collector(img, event, meta):
        frames.append((img.copy(), event, meta))

    callback = on_frame or _collector

    # psygnal's default connect() fires synchronously on the emitting thread.
    # CMMCorePlus wraps signals with Qt psygnal that also accepts a
    # Qt.ConnectionType — try DirectConnection first, fall back to plain
    # connect() for UniMMCore (pure psygnal).
    try:
        from PyQt6.QtCore import Qt

        mmc.mda.events.frameReady.connect(callback, Qt.ConnectionType.DirectConnection)
    except TypeError:
        mmc.mda.events.frameReady.connect(callback)

    try:
        mmc.run_mda(events, block=True)
    finally:
        with contextlib.suppress(TypeError, RuntimeError):
            mmc.mda.events.frameReady.disconnect(callback)

    return frames
