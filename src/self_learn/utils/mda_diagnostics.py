"""MDA-acquisition diagnostics — detect silent frame truncation.

ch348 r2 (8/10) lost 1/6 fields because the proxy returned 19 frames
from a 30-frame ``MDASequence(loops=30, interval=0.0)`` request and
the recipe didn't notice. ``analyze_frap`` produced a corrupt
recovery curve from the truncated stack and submitted a
mobile-fraction value that under-estimated the asymptote because
the curve hadn't saturated.

This module provides:

  - ``check_frame_count(results, expected, *, raise_on_mismatch=False)``
    — pure-numpy validator the caller invokes after ``run_events``.
  - ``run_events_checked(core, events, *, expected_frames, ...)`` —
    thin wrapper around ``src.core.hardware.core.run_events`` that
    raises ``MDAFrameTruncation`` when the result is short of the
    request, or returns the validated list unchanged when complete.

Both helpers explicit-fail rather than silent-degrade, so the recipe
can decide whether to retry, downscale, or escalate.

Sprint #15 (2026-04-26).
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Optional


class MDAFrameTruncation(RuntimeError):
    """Raised when an MDA result has fewer frames than requested.

    Attributes:
        expected: number of frames the caller asked for.
        actual: number of frames the engine returned.
        ratio: actual / expected.
    """

    def __init__(self, expected: int, actual: int):
        self.expected = int(expected)
        self.actual = int(actual)
        self.ratio = float(actual) / max(float(expected), 1.0)
        super().__init__(
            f"MDA returned {actual}/{expected} frames "
            f"({self.ratio:.0%}); expected stack truncated"
        )


def check_frame_count(
    results: Iterable[Any],
    expected: int,
    *,
    raise_on_mismatch: bool = False,
    warn_threshold: float = 0.95,
) -> dict:
    """Validate that an MDA returned the requested number of frames.

    Args:
        results: the iterable returned by ``run_events`` (a list of
            ``(image, event)`` tuples).
        expected: the number of frames the caller asked for.
        raise_on_mismatch: when True, raise ``MDAFrameTruncation`` on
            truncation; when False (default), emit a ``UserWarning``
            and return a structured report.
        warn_threshold: emit a warning even when not raising if the
            return ratio is below this fraction (default 0.95).

    Returns:
        ``{actual, expected, ratio, complete, warned}``.
    """
    results_list = list(results) if not isinstance(results, list) else results
    actual = len(results_list)
    expected_int = int(expected)
    ratio = float(actual) / max(float(expected_int), 1.0)
    complete = actual >= expected_int
    warned = False

    if not complete and raise_on_mismatch:
        raise MDAFrameTruncation(expected=expected_int, actual=actual)

    if ratio < float(warn_threshold):
        warnings.warn(
            f"MDA returned {actual}/{expected_int} frames "
            f"({ratio:.0%}). The recipe may be running on a truncated "
            f"stack; check for engine idle-pause, snap-cache fallthrough, "
            f"or proxy timeout.",
            UserWarning,
            stacklevel=2,
        )
        warned = True

    return {
        "actual": int(actual),
        "expected": int(expected_int),
        "ratio": float(ratio),
        "complete": bool(complete),
        "warned": bool(warned),
    }


def run_events_checked(
    core,
    events,
    *,
    expected_frames: int,
    on_frame: Optional[Any] = None,
    raise_on_mismatch: bool = False,
    warn_threshold: float = 0.95,
):
    """Wrapper around ``run_events`` that validates the frame count.

    Args:
        core: the microscope core.
        events: iterable / generator / list of ``MDAEvent`` objects.
        expected_frames: the number of frames the caller expects (the
            generator's ``loops`` parameter, or ``len(list(events))``
            for a fully-materialised event list).
        on_frame: optional callback forwarded to ``run_events``.
        raise_on_mismatch: see ``check_frame_count``.
        warn_threshold: see ``check_frame_count``.

    Returns:
        ``(results, report)`` tuple where ``results`` is the
        ``(image, event)`` tuple list from ``run_events`` (unchanged)
        and ``report`` is the dict from ``check_frame_count``.
    """
    from ..hardware.core import run_events

    results = run_events(core, events, on_frame=on_frame)
    report = check_frame_count(
        results, expected_frames,
        raise_on_mismatch=raise_on_mismatch,
        warn_threshold=warn_threshold,
    )
    return results, report
