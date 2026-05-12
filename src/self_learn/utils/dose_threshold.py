"""Dose-threshold sweep + bisection — minimum-effective-dose detection.

The third leg of dose-aware control, alongside
:mod:`self_learn.utils.rate_limited_drive` (single-setpoint drive to a
band) and :mod:`self_learn.utils.pulsed_schedule` (multi-waypoint
trajectory MPC):

  - **rate_limited_drive** answers "how many ON pulses to reach the
    setpoint?" given a known forward rate.
  - **pulsed_schedule** answers "how should I split ON/OFF segments
    to track a reference trajectory?"
  - **dose_threshold** answers "what is the minimum dose at which the
    response crosses a binary threshold?" — i.e. the *clinical*
    minimum-effective-dose problem, not the full dose-response curve.

This primitive is dose-agnostic (SLM mask amplitude, laser mW, drug
concentration, temperature). Caller supplies:

  * ``apply_dose(value: float) -> None``  — set the dose on the rig.
  * ``measure_response() -> float``       — read a scalar response.
  * ``response_threshold: float``         — the binary cut-off.

Optional: ``reset()`` callback fired between trials (e.g. clear FHN
refractory state, refresh perfusion, reset state device).

Composes nothing — pure-data primitives. Lifted from
``scratch/solve_632.py`` (264 inline lines, ch632 r1 = 10/10) where
the geometric sweep + nucleation-detection pattern was the core of
the win. Sister win shape to ``sweep_state_device`` (sprint #25):
that one ranks by scalar metric; this one ranks by binary outcome.

Sprint #43 (2026-04-28) extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThresholdSweepTrial:
    """One probed dose with its measured response."""
    dose: float
    response: float
    above: bool


@dataclass(frozen=True)
class ThresholdSweepResult:
    """Outcome of :func:`sweep_threshold` / :func:`bisect_threshold`.

    Attributes:
        trials: Per-dose outcomes, in the order they were probed.
        crossing_low: The highest probed dose with ``above=False``,
            or ``None`` if every probe was super-threshold.
        crossing_high: The lowest probed dose with ``above=True``,
            or ``None`` if no probe crossed.
        threshold_estimate: Geometric mean of ``crossing_low`` and
            ``crossing_high`` when both exist; else ``crossing_high``
            alone (or ``None`` if no super-threshold probe). This is
            the conservative ``min-effective-dose`` estimate — geometric
            mean is the natural mid-point on log-spaced sweeps.
        bracketed: ``True`` iff at least one sub-threshold AND one
            super-threshold trial were observed. The threshold sits
            inside the ``(crossing_low, crossing_high)`` interval only
            when bracketed.
    """
    trials: Tuple[ThresholdSweepTrial, ...]
    crossing_low: Optional[float]
    crossing_high: Optional[float]
    threshold_estimate: Optional[float]
    bracketed: bool

    def min_effective_dose(self) -> Optional[float]:
        """Smallest probed dose with ``above=True``, or ``None``.

        Conservative answer for the "minimum effective dose" submit
        shape: returns the smallest dose that *empirically* crossed
        the threshold, not the geometric-mean estimate. Use
        ``threshold_estimate`` when reporting a bracketed midpoint.
        """
        return self.crossing_high


# ---------------------------------------------------------------------------
# Dose grid helpers
# ---------------------------------------------------------------------------

def geometric_doses(low: float, high: float, n: int) -> List[float]:
    """Geometric series from ``low`` to ``high`` inclusive, length ``n``.

    Equivalent to ``np.geomspace(low, high, n).tolist()`` with explicit
    validation. Geometric (log-uniform) is the right grid when the
    threshold's order of magnitude is unknown and you want equal
    log-resolution above and below — typical for dose-response work.
    """
    if low <= 0 or high <= 0:
        raise ValueError(
            f"geometric grid requires positive bounds; got low={low}, high={high}"
        )
    if low >= high:
        raise ValueError(f"low must be < high; got {low} ≥ {high}")
    if n < 2:
        raise ValueError(f"n must be ≥ 2; got {n}")
    return [float(x) for x in np.geomspace(low, high, n)]


# ---------------------------------------------------------------------------
# Sweep + bisection
# ---------------------------------------------------------------------------

def sweep_threshold(
    *,
    doses: Sequence[float],
    apply_dose: Callable[[float], None],
    measure_response: Callable[[], float],
    response_threshold: float,
    reset: Optional[Callable[[], None]] = None,
) -> ThresholdSweepResult:
    """Sweep ``doses`` once each; record per-dose response + threshold cross.

    For every dose ``d`` in order:
      1. ``reset()`` if provided.
      2. ``apply_dose(d)``.
      3. ``measure_response()`` → scalar.
      4. ``above = response > response_threshold``.

    Returns :class:`ThresholdSweepResult`. Order of ``doses`` is
    preserved in ``trials``; ``crossing_low`` / ``crossing_high`` are
    derived from the dose values themselves, not their probe order.

    Raises ``ValueError`` on empty ``doses``.
    """
    if not doses:
        raise ValueError("doses must be non-empty")

    trials: List[ThresholdSweepTrial] = []
    for d in doses:
        if reset is not None:
            reset()
        apply_dose(float(d))
        r = float(measure_response())
        trials.append(ThresholdSweepTrial(
            dose=float(d), response=r, above=bool(r > response_threshold),
        ))

    return _summarise(tuple(trials))


def bisect_threshold(
    *,
    low: float,
    high: float,
    apply_dose: Callable[[float], None],
    measure_response: Callable[[], float],
    response_threshold: float,
    n_iters: int = 5,
    reset: Optional[Callable[[], None]] = None,
    log_space: bool = True,
    seed_trials: Optional[Sequence[ThresholdSweepTrial]] = None,
) -> ThresholdSweepResult:
    """Bisect between known ``low`` (sub) and ``high`` (super) doses.

    Caller is responsible for ensuring ``low`` is sub-threshold and
    ``high`` is super-threshold; typically you'd run
    :func:`sweep_threshold` first to bracket, then refine. The function
    does not re-probe the bracket endpoints — pass them in via
    ``seed_trials`` if you want them in the result.

    Each iteration:
      1. Pick midpoint — geometric (sqrt(low*high)) when ``log_space``,
         arithmetic ((low+high)/2) otherwise.
      2. ``reset()`` if provided.
      3. ``apply_dose(mid)``; ``measure_response()``.
      4. If above: tighten ``high = mid``; else tighten ``low = mid``.

    Result reports the final bracket plus all probed midpoints. The
    threshold estimate is the geometric / arithmetic mean of the
    final ``crossing_low`` / ``crossing_high``.

    Raises ``ValueError`` on invalid bracket or non-positive bounds
    when ``log_space=True``.
    """
    if low >= high:
        raise ValueError(f"low must be < high; got {low} ≥ {high}")
    if log_space and (low <= 0 or high <= 0):
        raise ValueError(
            f"log_space=True requires positive bounds; got low={low}, high={high}"
        )
    if n_iters < 1:
        raise ValueError(f"n_iters must be ≥ 1; got {n_iters}")

    trials: List[ThresholdSweepTrial] = list(seed_trials or ())

    # Track the live bracket independently from trials.
    cur_low, cur_high = float(low), float(high)
    for _ in range(n_iters):
        if log_space:
            mid = float(np.sqrt(cur_low * cur_high))
        else:
            mid = 0.5 * (cur_low + cur_high)
        if reset is not None:
            reset()
        apply_dose(mid)
        r = float(measure_response())
        above = bool(r > response_threshold)
        trials.append(ThresholdSweepTrial(dose=mid, response=r, above=above))
        if above:
            cur_high = mid
        else:
            cur_low = mid

    return _summarise(tuple(trials))


# ---------------------------------------------------------------------------
# Internal summarisation
# ---------------------------------------------------------------------------

def _summarise(trials: Tuple[ThresholdSweepTrial, ...]) -> ThresholdSweepResult:
    """Reduce a list of probed trials to a :class:`ThresholdSweepResult`."""
    sub_doses = [t.dose for t in trials if not t.above]
    sup_doses = [t.dose for t in trials if t.above]
    crossing_low = max(sub_doses) if sub_doses else None
    crossing_high = min(sup_doses) if sup_doses else None
    bracketed = crossing_low is not None and crossing_high is not None

    if crossing_low is not None and crossing_high is not None:
        if crossing_low > 0 and crossing_high > 0:
            estimate = float(np.sqrt(crossing_low * crossing_high))
        else:
            estimate = 0.5 * (crossing_low + crossing_high)
    elif crossing_high is not None:
        estimate = crossing_high
    else:
        estimate = None

    return ThresholdSweepResult(
        trials=trials,
        crossing_low=crossing_low,
        crossing_high=crossing_high,
        threshold_estimate=estimate,
        bracketed=bracketed,
    )
