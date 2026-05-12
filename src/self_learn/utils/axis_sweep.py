"""useq-MDA-driven axis sweep + coordinate descent for state-device alignment.

Sprint #46 (2026-04-28). Lifts the ch657 / ch658 light-sheet AutoPilot
shape into a transferable primitive that sits cleanly above the
2026-04-27 transferability contract: every snap is delivered through a
:class:`useq.MDAEvent`, with state-device set-points carried in
``MDAEvent.properties`` rather than via inline ``setState`` + ``snap``
loops.

Why this exists separate from
:func:`src.core.utils.sensorless_ao.sweep_state_device`:

- ``sensorless_ao.sweep_state_device`` is the legacy AO sweep — inline
  ``setState`` → ``waitForDevice`` → ``snap``. Trips the submission gate
  (NON_NEGOTIABLES rule 4, ``logs/comms/submission_gate.py``) on any
  newly-authored solve, so callers who reach for it from inside a
  fresh solve hit a ``SubmissionRejected`` raise.
- ``axis_sweep.sweep_axis_mda`` builds a useq event sequence; one
  event per state, with ``MDAEvent.properties = [(axis, "State", k)]``.
  ``run_events`` dispatches them, and on a real microscope the same
  sequence object lands on the hardware MDA runner.
- The shape generalises to: filter-wheel sweep, objective-turret
  sweep, dichroic sweep, electronic-galvo offset sweep, anything
  that's a categorical state-device with N positions.

Sister to:
- :mod:`src.core.utils.sensorless_ao` — same coordinate-descent shape,
  inline-snap variant.
- :mod:`src.core.utils.fft_peak`, :mod:`src.core.utils.fluorophore_brightness` —
  reduce-on-snap utilities that compose cleanly with the per-state
  reduce hook here.

Composes with:
- :func:`src.core.hardware.core.run_events`
- :class:`useq.MDAEvent` ``properties`` field

Tested on:
- ch657 r1 = 10/10 (3 axes × 5 states independent sweep)
- ch658 r1 = 10/10 (3 axes × 5 states coordinate descent)

Both shipped inline before extraction; this module is the agent-portable
substrate so future alignment / categorical-sweep challenges auto-route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

# Default reducers. Use mean signal for fluorescence-alignment use cases
# (Royer 2016 AutoPilot); use Brenner gradient for sharpness-driven
# focus / DM sweeps (delegated to :func:`focus_metric` if requested).
_REDUCERS: dict[str, Callable] = {
    "mean":  lambda img: float(np.asarray(img).mean()),
    "max":   lambda img: float(np.asarray(img).max()),
    "std":   lambda img: float(np.asarray(img).std()),
    "p95":   lambda img: float(np.percentile(np.asarray(img), 95)),
}


def _resolve_reduce(reduce):
    """Accept a string key from :data:`_REDUCERS`, a callable, or None.

    None defaults to ``"mean"`` — the canonical alignment metric.
    """
    if reduce is None:
        return _REDUCERS["mean"]
    if isinstance(reduce, str):
        if reduce not in _REDUCERS:
            raise ValueError(
                f"reduce={reduce!r} not in built-ins {sorted(_REDUCERS)}; "
                f"pass a callable for custom metrics."
            )
        return _REDUCERS[reduce]
    if callable(reduce):
        return reduce
    raise TypeError(
        f"reduce must be None, str, or callable; got {type(reduce).__name__}"
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisSweepResult:
    """Output of :func:`sweep_axis_mda`.

    Attributes:
        axis: The state-device that was swept.
        signals: ``{state_idx: reduced_value}`` over ``range(n_states)``.
        argmax_state: ``int`` — state index with the highest reduced value.
        argmax_value: The reduced value at ``argmax_state``.
        images: Per-state raw frames, keyed by state index. Indexed by the
            same keys as ``signals`` so callers can re-reduce or showcase.
        n_events: Number of useq events dispatched (== ``n_states``).
    """
    axis: str
    signals: dict[int, float]
    argmax_state: int
    argmax_value: float
    images: dict[int, np.ndarray] = field(default_factory=dict)
    n_events: int = 0


@dataclass(frozen=True)
class DescentResult:
    """Output of :func:`coordinate_descent_mda`.

    Attributes:
        per_axis: ``{axis: AxisSweepResult}`` — one per axis.
        final_state_map: ``{axis: argmax_state}`` for every axis after
            the descent. Use this to rebuild the optimal device state.
        n_events: Total useq events across all axes (``axes × n_states``
            for a single-pass descent).
        initial_state_map: device state at descent start.
    """
    per_axis: dict[str, AxisSweepResult]
    final_state_map: dict[str, int]
    n_events: int
    initial_state_map: dict[str, int]


# ---------------------------------------------------------------------------
# Pure-data helpers (testable without a live core)
# ---------------------------------------------------------------------------

def build_sweep_events(
    *,
    axis: str,
    n_states: int,
    channel: str,
    exposure: float,
    pin: Optional[Mapping[str, int]] = None,
    metadata_extra: Optional[dict] = None,
):
    """Build a list of :class:`useq.MDAEvent` for one axis sweep.

    Each event sets ``axis`` to ``s`` for ``s in range(n_states)``, and
    pins every other axis listed in ``pin`` to its mapped state.

    Pure-data: no core touched. Easy to unit-test the property tuples.

    Args:
        axis: state-device name to sweep.
        n_states: number of states to traverse (0..n_states-1).
        channel: channel-config name (group ``"Channel"``).
        exposure: per-event exposure (ms).
        pin: optional ``{other_axis: state}`` mapping. The pinned axes
            are written into every event so the OTHER axes hold the
            reference set-point during the sweep.
        metadata_extra: optional dict merged into every event's metadata.

    Returns:
        List of :class:`useq.MDAEvent`.
    """
    from useq import MDAEvent

    pin_dict = dict(pin or {})
    base_meta = dict(metadata_extra or {})
    events: list = []
    for s in range(int(n_states)):
        props = [(axis, "State", int(s))]
        for other, value in pin_dict.items():
            if other == axis:
                continue
            props.append((other, "State", int(value)))
        meta = {"axis": axis, "state": int(s), **base_meta}
        events.append(MDAEvent(
            channel={"config": channel, "group": "Channel"},
            exposure=float(exposure),
            properties=props,
            metadata=meta,
        ))
    return events


def argmax_state(signals: Mapping[int, float]) -> tuple[int, float]:
    """Return ``(state, value)`` of the highest entry in ``signals``.

    Tie-break: smallest state index wins (deterministic).
    """
    if not signals:
        raise ValueError("signals is empty")
    best_state = min(signals)
    best_value = signals[best_state]
    for s, v in signals.items():
        if v > best_value or (v == best_value and s < best_state):
            best_state = s
            best_value = v
    return int(best_state), float(best_value)


# ---------------------------------------------------------------------------
# Live-core entry points
# ---------------------------------------------------------------------------

def sweep_axis_mda(
    core,
    axis: str,
    *,
    n_states: int = 5,
    channel: str,
    exposure: float = 50.0,
    pin: Optional[Mapping[str, int]] = None,
    reduce=None,
    keep_images: bool = True,
) -> AxisSweepResult:
    """Sweep one state-device through ``range(n_states)`` via useq events.

    Each event is dispatched through :func:`run_events`; the on-frame
    callback reduces the frame to a scalar per the ``reduce`` hook.

    Use cases:
    - Light-sheet alignment axis sweep (Royer 2016 AutoPilot single-pass)
    - Filter-wheel argmax over candidate dichroics
    - Sensorless-AO state sweep where the metric is mean / std / argmax
      rather than Brenner sharpness

    Args:
        core: pymmcore-plus / -proxy instance.
        axis: state-device to sweep (e.g. ``"LightSheetY"``).
        n_states: number of states ``0..n_states-1``. Default 5
            matches the 5-position light-sheet alignment encoders.
        channel: channel-config name (set via the per-event channel).
        exposure: per-event exposure in ms.
        pin: optional ``{other_axis: state}`` to hold during the sweep.
        reduce: per-frame reducer. Accepts string keys from
            :data:`_REDUCERS`, a callable ``(image) -> float``, or None
            (defaults to ``"mean"``). For Brenner-style sensorless AO
            pass ``reduce=lambda img: focus_metric(img, "brenner")``.
        keep_images: if False, do NOT retain frames in the result
            (saves memory on long sweeps).

    Returns:
        :class:`AxisSweepResult` with per-state signals + argmax.
    """
    from ..hardware.core import run_events

    reducer = _resolve_reduce(reduce)
    events = build_sweep_events(
        axis=axis, n_states=n_states, channel=channel, exposure=exposure,
        pin=pin,
    )
    signals: dict[int, float] = {}
    images: dict[int, np.ndarray] = {}

    def _on_frame(img, event):
        meta = dict(event.metadata)
        s = int(meta["state"])
        arr = np.asarray(img)
        signals[s] = float(reducer(arr))
        if keep_images:
            images[s] = arr.copy()

    run_events(core, events, on_frame=_on_frame)

    best_state, best_value = argmax_state(signals)
    return AxisSweepResult(
        axis=axis,
        signals=signals,
        argmax_state=best_state,
        argmax_value=best_value,
        images=images,
        n_events=len(events),
    )


def coordinate_descent_mda(
    core,
    axes: Sequence[str],
    *,
    n_states_per_axis: int = 5,
    channel: str,
    exposure: float = 50.0,
    reduce=None,
    n_passes: int = 1,
    initial_state_map: Optional[Mapping[str, int]] = None,
    keep_images: bool = False,
) -> DescentResult:
    """Coordinate-descent across N axes via successive :func:`sweep_axis_mda` calls.

    Each axis is swept with the OTHER axes pinned at their CURRENT-best
    state (= argmax from the previous axis's sweep, or
    ``initial_state_map`` for the first axis).

    Use cases:
    - Royer 2016 AutoPilot 3-axis (or 5-axis) descent
    - General categorical-state argmax search where each axis is
      independently sweepable but their argmaxes interact

    Multi-pass (``n_passes > 1``) re-sweeps each axis with the latest
    state map; useful when axis interactions don't converge in one pass.

    Args:
        core: pymmcore-plus / -proxy instance.
        axes: sequence of state-device names. Order matters — the first
            axis is swept first.
        n_states_per_axis: states 0..n-1 per axis. Default 5.
        channel: channel-config name.
        exposure: per-event exposure in ms.
        reduce: per-frame reducer (see :func:`sweep_axis_mda`).
        n_passes: number of full passes through ``axes``. Default 1.
        initial_state_map: optional ``{axis: state}`` to seed the descent.
            Defaults to reading ``core.getProperty(axis, "State")`` for
            each axis.
        keep_images: forwarded to :func:`sweep_axis_mda`.

    Returns:
        :class:`DescentResult` with the final ``state_map`` and per-axis
        sweep results from the LAST pass.
    """
    if initial_state_map is None:
        state_map = {a: int(core.getProperty(a, "State")) for a in axes}
    else:
        state_map = {a: int(initial_state_map[a]) for a in axes}

    initial = dict(state_map)
    per_axis: dict[str, AxisSweepResult] = {}
    n_events = 0

    for _pass in range(int(n_passes)):
        for axis in axes:
            pin = {a: s for a, s in state_map.items() if a != axis}
            res = sweep_axis_mda(
                core, axis,
                n_states=n_states_per_axis,
                channel=channel,
                exposure=exposure,
                pin=pin,
                reduce=reduce,
                keep_images=keep_images,
            )
            per_axis[axis] = res
            state_map[axis] = res.argmax_state
            n_events += res.n_events

    return DescentResult(
        per_axis=per_axis,
        final_state_map=state_map,
        n_events=n_events,
        initial_state_map=initial,
    )
