"""Rate-limited drive — forward-model + early-stop primitive.

Forward-model + early-stop pattern for monotonic saturating rate laws:
    N_predict = ln(1 / (1 - target/max)) / rate
    early_stop when measurement enters [target_lo, target_lo + slack]

Applies to any process following ``state(N) = max·(1 - exp(-rate·N))``:
photoconversion, ablation dose accumulation, FUCCI phase progression.

Two functions, no scenario-specific assumptions:

  - :func:`predict_n_steps` — closed-form forward model for the
    monotonic-saturating rate law ``state(N) = max·(1 - exp(-rate·N))``.
  - :func:`drive_to_band` — closed-loop wrap that takes a
    ``step_fn`` actuator + ``measure_fn`` sensor and stops the
    moment the measurement enters ``[band_lo, band_lo + slack]``.

Composes nothing — pure math + a generic callback loop. The caller
wires up how to step the system (e.g. setProperty + setConfig +
snapImage) and how to read its state (image analysis on the
returned frame).

Single-setpoint by design. Multi-waypoint reference-trajectory MPC
(pulsed stim, no mid-trajectory early-stop) is a different shape —
the early-stop semantics break when you have to overshoot
intermediate setpoints — and lives in
:mod:`self_learn.utils.pulsed_schedule` (sprint #37, lifted from
ch624 r1).

This module is intentionally NOT registered in
:mod:`self_learn.utils.auto_recipe`. Rate-limited-drive availability
is a scenario-level signal (rate constant + setpoint target both
named in the brief) — same rationale as the other utility primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Literal, Tuple


StoppedReason = Literal["band", "max_steps"]


@dataclass(frozen=True)
class DriveResult:
    """Outcome of a closed-loop drive.

    ``trace`` is ``[(0, pre_value), (1, v_after_step_1), ...]`` — the
    pre-read is recorded at step 0 so callers can plot the full
    trajectory; ``len(trace) - 1 == n_steps``. ``stopped_reason``
    distinguishes a band-success from a max-steps cutoff.
    """

    n_steps: int
    final_value: float
    trace: List[Tuple[int, float]]
    stopped_reason: StoppedReason


def predict_n_steps(target: float, *, rate: float, max: float = 1.0) -> int:
    """Closed-form: minimum N with ``max·(1 - exp(-rate·N)) >= target``.

    Solves ``target = max·(1 - exp(-rate·N))`` for N and returns
    ``ceil``. The recommended use is to predict against the BAND FLOOR
    (not the band midpoint) — that gives the smallest N that REACHES
    the band, leaving the early-stop wrap to absorb residual model
    error. ch621 r1: ``predict_n_steps(0.40, rate=0.2, max=1.0)`` →
    ``ceil(ln(1/0.60)/0.2)`` = ``ceil(2.554)`` = 3.

    Raises ``ValueError`` for ``target <= 0``, ``target >= max``,
    ``rate <= 0``, or ``max <= 0``.
    """
    if max <= 0:
        raise ValueError(f"max must be > 0; got {max}")
    if target <= 0:
        raise ValueError(f"target must be > 0; got {target}")
    if target >= max:
        raise ValueError(
            f"target must be < max ({max}); got {target} (asymptote unreachable)"
        )
    if rate <= 0:
        raise ValueError(f"rate must be > 0; got {rate}")
    n_real = math.log(1.0 / (1.0 - target / max)) / rate
    return int(math.ceil(n_real))


def drive_to_band(
    measure_fn: Callable[[], float],
    step_fn: Callable[[], None],
    *,
    band: Tuple[float, float],
    slack: float = 0.05,
    max_steps: int = 10,
) -> DriveResult:
    """Closed-loop drive: step until measurement enters early-stop window.

    Algorithm:

    1. Read ``v0 = measure_fn()`` once (no step yet); ``trace = [(0, v0)]``.
       If ``v0`` is already in ``[band[0], band[0] + slack]``, return
       immediately with ``n_steps=0`` and ``stopped_reason="band"``.
    2. For ``k`` in ``1..max_steps``: call ``step_fn()``, then
       ``v_k = measure_fn()``; append ``(k, v_k)`` to trace. If
       ``band[0] <= v_k <= band[0] + slack``, stop with
       ``stopped_reason="band"``.
    3. If the loop exits without entering the window,
       ``stopped_reason="max_steps"`` (the value may have overshot
       past ``band[0] + slack`` — early-stop is a one-sided
       floor + slack, not an overshoot-recovery controller).

    Validates ``band[0] < band[1]``, ``0 < slack <= band_width``,
    ``max_steps >= 1``. Does NOT validate that the early-stop window
    is reachable — the caller's forward model is theirs.
    """
    band_lo, band_hi = band
    if band_lo >= band_hi:
        raise ValueError(f"band[0] must be < band[1]; got {band}")
    if slack <= 0:
        raise ValueError(f"slack must be > 0; got {slack}")
    band_width = band_hi - band_lo
    if slack > band_width:
        raise ValueError(
            f"slack {slack} exceeds band width {band_width} "
            f"(early-stop ceiling > band ceiling)"
        )
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1; got {max_steps}")

    early_stop_hi = band_lo + slack

    v = float(measure_fn())
    trace: List[Tuple[int, float]] = [(0, v)]
    if band_lo <= v <= early_stop_hi:
        return DriveResult(
            n_steps=0, final_value=v, trace=trace, stopped_reason="band"
        )

    n_steps = 0
    stopped_reason: StoppedReason = "max_steps"
    for k in range(1, max_steps + 1):
        step_fn()
        v = float(measure_fn())
        trace.append((k, v))
        n_steps = k
        if band_lo <= v <= early_stop_hi:
            stopped_reason = "band"
            break

    return DriveResult(
        n_steps=n_steps,
        final_value=v,
        trace=trace,
        stopped_reason=stopped_reason,
    )
