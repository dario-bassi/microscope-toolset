"""Helpers for generator-based MDA (Multi-Dimensional Acquisition) with feedback.

Provides `run_mda_with_feedback()` which wraps pymmcore-plus's `run_mda()` with
a per-frame callback connected via Qt.DirectConnection for synchronous delivery
on the MDA thread.

Note: napari-micromanager's _mda_handler has been patched to gracefully skip
generator-based sequences (GeneratorMDASequence check in _on_mda_started,
event.sequence is None check in _on_mda_frame). If running with an unpatched
version, this helper falls back to disconnecting/reconnecting the handler.
"""

import gc
import contextlib
import logging
from typing import Callable, Iterable, Optional

import numpy as np
from useq import MDAEvent

logger = logging.getLogger("MDAHelpers")


def run_mda_with_feedback(
    mmc,
    events: Iterable[MDAEvent],
    on_frame: Optional[Callable[[np.ndarray, MDAEvent, dict], None]] = None,
) -> list[tuple[np.ndarray, MDAEvent, dict]]:
    """Run MDA with per-frame callback.

    Connects a frameReady callback with Qt.DirectConnection for synchronous
    delivery on the MDA thread, runs the MDA, then disconnects.

    The napari-micromanager handler is compatible with generator-based MDA
    (it skips frames with event.sequence is None). If running with an unpatched
    napari-micromanager, falls back to disconnecting/reconnecting the handler.

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
    from PyQt6.QtCore import Qt

    # Check if napari-micromanager handler needs fallback disconnect
    handler = None
    if not _handler_is_patched():
        handler = _find_napari_handler()
        if handler:
            handler._cleanup()
            logger.info("Disconnected unpatched napari-micromanager MDA handler")

    frames: list[tuple[np.ndarray, MDAEvent, dict]] = []

    def _collector(img, event, meta):
        frames.append((img.copy(), event, meta))

    callback = on_frame or _collector

    mmc.mda.events.frameReady.connect(
        callback, Qt.ConnectionType.DirectConnection
    )

    try:
        mmc.run_mda(events, block=True)
    finally:
        with contextlib.suppress(TypeError, RuntimeError):
            mmc.mda.events.frameReady.disconnect(callback)

        if handler:
            _reconnect_napari_handler(handler)
            logger.info("Reconnected napari-micromanager MDA handler")

    return frames


def _handler_is_patched() -> bool:
    """Check if napari-micromanager's _on_mda_frame handles event.sequence is None."""
    try:
        import inspect
        from napari_micromanager._mda_handler import _NapariMDAHandler
        source = inspect.getsource(_NapariMDAHandler._on_mda_frame)
        return "event.sequence is None" in source
    except Exception:
        return False


def _find_napari_handler():
    """Find the napari-micromanager MDA handler instance via gc."""
    try:
        from napari_micromanager._mda_handler import _NapariMDAHandler
    except ImportError:
        return None

    for obj in gc.get_objects():
        if isinstance(obj, _NapariMDAHandler):
            return obj
    return None


def _reconnect_napari_handler(handler):
    """Reconnect all signal-slot pairs on the napari-micromanager handler."""
    for signal, slot in handler._connections:
        with contextlib.suppress(TypeError, RuntimeError):
            signal.connect(slot)
