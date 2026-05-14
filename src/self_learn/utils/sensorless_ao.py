"""Sensorless-AO primitives: state-device sweep + sharpness + per-axis inference.

Pure-data utilities (except :func:`sweep_state_device` which touches
``core``). The argmax / baseline-axes / residual functions take only
dict-shaped inputs so they can be tested with synthetic data — no live
core required.

Composes from:
- :func:`self_learn.hardware.core.snap`
- :func:`self_learn.workflows.autofocus.focus_metric`

Extracted from the ch607 / ch608 / ch610 wins (sprint #25, counter=87).
The three patterns this module operationalises:

1. **DM-sweep + argmax** — sweep all N states of a categorical
   state-device, compute Brenner gradient at each, pick argmax.
2. **Per-axis-presence inference** — for each axis, check whether
   either of its ± states exceeds flat by ≥ ``margin``. Recovers
   the baseline axes from the curve shape alone.
3. **Residual axis** — set difference between baseline axes and the
   axis that the picked-best state addresses. The uncancelled
   component when no single discrete state cancels everything.

Pattern 3 from the autofocus iteration arc — *calibrate-around-focus*
— is **not** in this module. It's a different scaffolding (Z sweep +
``phase_cross_correlation``); see ``scratch/solve_609_r3.py`` and
``knowledge/recipes/Sensorless AO.md`` § 3 for why and where.
"""

from __future__ import annotations

from typing import Callable, Iterable


DEFAULT_AXES: tuple[str, ...] = (
    "defocus", "astig_x", "astig_y", "coma_x", "coma_y",
)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

_AXIS_PREFIXES: tuple[tuple[str, str], ...] = (
    ("defocus_", "defocus"),
    ("astig_x_", "astig_x"),
    ("astig_y_", "astig_y"),
    ("coma_x_",  "coma_x"),
    ("coma_y_",  "coma_y"),
)


def default_axis_for_label(label: str) -> tuple[str, int]:
    """Map a DM state label to ``(axis_name, sign)``.

    Examples:
        ``"flat"`` → ``("flat", 0)``
        ``"defocus_+0.5"`` → ``("defocus", +1)``
        ``"astig_x_-0.5"`` → ``("astig_x", -1)``
        ``"unknown_label"`` → ``("other", 0)``

    Extend the prefix table when a future experiment uses non-Zernike
    action sets; don't generalise pre-emptively.
    """
    if label == "flat":
        return ("flat", 0)
    for prefix, axis in _AXIS_PREFIXES:
        if label.startswith(prefix):
            sign = +1 if "+" in label else -1 if "-" in label else 0
            return (axis, sign)
    return ("other", 0)


# ---------------------------------------------------------------------------
# State-device sweep
# ---------------------------------------------------------------------------

def sweep_state_device(
    core,
    device: str,
    *,
    channel: str | None = None,
    metric: str = "brenner",
    reduce=None,
) -> dict:
    """Sweep all N states of ``device``, snap at each, return ``{label: value}``.

    Order is 0..N-1; labels come from ``core.getStateLabels(device)``.
    Each iteration: ``setState`` → ``waitForDevice`` → ``snap`` → reduce.
    The pre-sweep state is **restored on exit** so callers don't observe
    a side-effect.

    Args:
        device: state-device name.
        channel: optional channel config to set before each snap.
        metric: forwarded verbatim to ``focus_metric`` when ``reduce``
            is None. Default ``"brenner"`` matches Hu 2023 / Royer 2016
            / Zhang 2023 closed-loop AO conventions.
        reduce: optional ``(image) -> any`` callable. When provided, it
            overrides ``metric`` and the dict values become whatever
            ``reduce`` returns — a scalar (e.g. mean intensity), a
            tuple, or the raw frame itself. Use this for non-AO sweeps
            that need richer per-state state (e.g. SIM 3-phase
            demodulation: ``reduce=lambda img: np.asarray(img).copy()``
            keeps every frame for downstream summation).

    Returns:
        ``{label: float}`` when ``reduce`` is None (legacy AO use),
        ``{label: reduce(image)}`` otherwise.
    """
    from ..hardware.core import snap
    from ..workflows.autofocus import focus_metric

    n_states = int(core.getNumberOfStates(device))
    labels = list(core.getStateLabels(device))
    pre_state = int(core.getState(device))

    scores: dict = {}
    try:
        for k in range(n_states):
            core.setState(device, k)
            core.waitForDevice(device)
            img = snap(core, channel=channel)
            if reduce is not None:
                scores[labels[k]] = reduce(img)
            else:
                scores[labels[k]] = float(focus_metric(img, method=metric))
    finally:
        # Restore pre-sweep state so callers don't observe a side-effect.
        try:
            core.setState(device, pre_state)
            core.waitForDevice(device)
        except Exception:  # noqa: BLE001
            pass
    return scores


# ---------------------------------------------------------------------------
# Argmax + margin
# ---------------------------------------------------------------------------

def argmax_state(scores: dict[str, float]) -> tuple[str, float, float]:
    """Return ``(best_label, best_score, margin_over_runner_up)``.

    ``margin = (best - second_best) / second_best``, or ``float('inf')``
    when ``second_best <= 0`` or there's only one entry.

    Tie-breaking on ``best_score``: first-inserted label wins (Python
    dict insertion order; deterministic on Python 3.7+).
    """
    if not scores:
        raise ValueError("scores is empty")
    items = list(scores.items())
    best_label, best_score = items[0]
    for label, score in items[1:]:
        if score > best_score:
            best_label = label
            best_score = score
    if len(items) < 2:
        return best_label, float(best_score), float("inf")
    second = max((s for lbl, s in items if lbl != best_label), default=0.0)
    if second <= 0:
        return best_label, float(best_score), float("inf")
    margin = (best_score - second) / second
    return best_label, float(best_score), float(margin)


# ---------------------------------------------------------------------------
# Per-axis-presence inference
# ---------------------------------------------------------------------------

def infer_baseline_axes(
    scores: dict[str, float],
    *,
    axis_for_label: Callable[[str], tuple[str, int]] = default_axis_for_label,
    flat_label: str = "flat",
    margin: float = 0.03,
    axes: Iterable[str] | None = None,
) -> list[str]:
    """Per-axis-presence test (ch610 diagnostic).

    For each axis (default: :data:`DEFAULT_AXES`), find ``max(score)``
    across its ± states; the axis is **in baseline** iff
    ``max > scores[flat_label] * (1 + margin)``.

    Returns the list of baseline axes in ``axes`` iteration order.
    Returns ``[]`` when nothing exceeds flat by ``margin`` — the
    legitimate ch607-shape no-aberration case.

    An axis is in the baseline if its best ± state exceeds flat by
    ``margin`` Brenner — a sound a-priori test for which axes have
    non-zero baseline contribution.
    """
    flat_score = scores.get(flat_label)
    if flat_score is None:
        raise KeyError(f"flat_label {flat_label!r} not in scores")
    threshold = flat_score * (1.0 + float(margin))

    by_axis: dict[str, float] = {}
    for label, score in scores.items():
        ax, _sign = axis_for_label(label)
        if ax in ("flat", "other"):
            continue
        prev = by_axis.get(ax, float("-inf"))
        if score > prev:
            by_axis[ax] = score

    axes_iter = list(axes) if axes is not None else list(DEFAULT_AXES)
    return [ax for ax in axes_iter if by_axis.get(ax, float("-inf")) > threshold]


def residual_axis(
    baseline_axes: list[str],
    best_axis: str,
) -> list[str]:
    """Return baseline axes the picked-best state did NOT address.

    Edge cases:
      - ``best_axis == "flat"`` (no DM correction won): residual is
        the full ``baseline_axes`` (legitimate when no single ±0.5
        state addresses the baseline strongly enough to beat flat).
      - ``best_axis`` not in baseline_axes: residual is full
        ``baseline_axes`` (best state addressed an axis that wasn't
        in the baseline at all — improbable given how
        ``infer_baseline_axes`` works, but handled).
    """
    return [a for a in baseline_axes if a != best_axis]
