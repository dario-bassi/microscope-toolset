"""Fluorophore-brightness primitives: predict ranking + measure + compare.

Pure-data utilities — no live core required for the predict / compare
half. Mirrors :mod:`src.core.utils.fft_peak` and
:mod:`src.core.utils.spectral_leak` (utility-only, no recipe layer
this sprint).

Lifted from :mod:`scratch.solve_617` (counter=116) where the inline
(predict ranking from registry → snap each channel → verify ratios)
block was the core of the ch617 r1 → 10/10 win.

Three primitives:

1. :func:`predict_brightness_ranking` — pure data: dict of
   ``{channel: brightness}`` → :class:`BrightnessRanking` with
   sorted order, ratios normalised to dimmest=1.0.
2. :func:`measure_brightness_ranking` — snap each channel at a
   matched exposure, reduce to one scalar per channel
   (default: FOV mean), wrap in the same dataclass.
3. :func:`compare_predictions` — verification step: ranking-match
   bool + per-channel rel-err + max rel-err + tolerance check.

This module is intentionally NOT registered in
:mod:`src.core.utils.auto_recipe`. Brightness ranking is a
brief-level signal ("brightness", "ε × Φ", "FPbase") — same
rationale as :mod:`src.core.utils.spectral_leak`,
:mod:`src.core.utils.fft_peak`.

REUSABLE per the ch617 grader:
- Multi-channel bleach-rate ranking via the registry's
  ``bleach_kx`` field.
- Filter-set SNR design.
- Bleed-through severity (the brightness ratio sets which channel
  the leak from :mod:`src.core.utils.spectral_leak` is most visible
  in).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_REDUCE = lambda im: float(np.mean(im))


@dataclass(frozen=True)
class BrightnessRanking:
    """Per-channel brightness ranking + ratios.

    ``ranking`` is brightest → dimmest; ``ratios`` is normalised so
    the dimmest channel is exactly 1.0; ``brightnesses`` is the raw
    input (predict: ε × Φ; measure: FOV reducer values).

    Tie-break on ``brightnesses``: lexicographic on channel name —
    deterministic, no surprise re-orderings between rounds.
    """

    ranking: List[str]
    ratios: Dict[str, float]
    brightnesses: Dict[str, float]


def predict_brightness_ranking(
    channel_brightnesses: Dict[str, float],
) -> BrightnessRanking:
    """Closed-form Phase A: ε × Φ → ranking + ratios.

    Lex tie-break on equal brightness so the order is reproducible.
    """
    if not channel_brightnesses:
        raise ValueError("channel_brightnesses is empty")
    items = list(channel_brightnesses.items())
    for ch, b in items:
        if not np.isfinite(b):
            raise ValueError(f"brightness for {ch!r} is not finite: {b!r}")
    items.sort(key=lambda kv: (-float(kv[1]), kv[0]))
    ranking = [ch for ch, _ in items]
    dim = min(float(b) for _, b in items)
    if dim <= 0:
        raise ValueError(
            f"min brightness must be > 0 to normalise ratios; got {dim}"
        )
    ratios = {ch: float(b) / dim for ch, b in items}
    brightnesses = {ch: float(b) for ch, b in items}
    return BrightnessRanking(
        ranking=ranking, ratios=ratios, brightnesses=brightnesses,
    )


def measure_brightness_ranking(
    core,
    channels: List[str],
    *,
    exposure_ms: Optional[float] = None,
    reduce: Callable[[np.ndarray], float] = _DEFAULT_REDUCE,
) -> BrightnessRanking:
    """Phase B: snap each channel at matched exposure → reducer per channel.

    ``exposure_ms`` if provided is set per snap and the prior
    exposure is restored on exit (try/finally). ``reduce`` defaults
    to FOV mean (matches ch617); pass e.g.
    ``lambda im: float(np.percentile(im, 95))`` for noise-robust or
    ``lambda im: float(im.mean()/im.std())`` for SNR-proper.

    Raises ``ValueError`` if any channel's reduced value is non-finite.
    """
    if not channels:
        raise ValueError("channels is empty")

    prior_exposure: Optional[float] = None
    if exposure_ms is not None:
        try:
            prior_exposure = float(core.getExposure())
        except Exception:  # noqa: BLE001
            prior_exposure = None

    measured: Dict[str, float] = {}
    try:
        for ch in channels:
            core.setConfig("Channel", ch)
            try:
                core.waitForConfig("Channel", ch)
            except Exception:  # noqa: BLE001 — older proxies
                pass
            if exposure_ms is not None:
                core.setExposure(float(exposure_ms))
            core.snapImage()
            img = np.asarray(core.getImage())
            value = reduce(img)
            if not np.isfinite(value):
                raise ValueError(
                    f"reducer returned non-finite value {value!r} for "
                    f"channel {ch!r}"
                )
            measured[ch] = float(value)
    finally:
        if exposure_ms is not None and prior_exposure is not None:
            try:
                core.setExposure(prior_exposure)
            except Exception:  # noqa: BLE001
                pass

    return predict_brightness_ranking(measured)  # same shape; lex tie-break


def compare_predictions(
    predicted: BrightnessRanking,
    measured: BrightnessRanking,
    *,
    ratio_tol: float = 0.5,
) -> dict:
    """Verification step: ranking match + per-channel rel-err.

    ``per_channel_rel_err[ch] = |predicted_ratio / measured_ratio - 1|``
    for each channel; the dimmest entry is 0.0 by construction
    (both ratios are 1.0). ``max_rel_err`` is the max over
    *non-dimmest* channels. ``within_tolerance`` is
    ``ranking_match AND max_rel_err < ratio_tol``.

    Returns a dict (not a dataclass) so callers can ``**unpack``
    into ``submit_solution(answer={...})`` payloads.
    """
    ranking_match = predicted.ranking == measured.ranking

    # Rel-err per channel; dimmest excluded from max via separate tracking.
    pred_dim = predicted.ranking[-1] if predicted.ranking else None
    per_err: Dict[str, float] = {}
    max_rel_err = 0.0
    for ch in predicted.ratios:
        if ch == pred_dim:
            per_err[ch] = 0.0
            continue
        m = measured.ratios.get(ch, float("nan"))
        if m <= 0 or not np.isfinite(m):
            per_err[ch] = float("inf")
            max_rel_err = float("inf")
            continue
        e = abs(float(predicted.ratios[ch]) / m - 1.0)
        per_err[ch] = float(e)
        if e > max_rel_err:
            max_rel_err = float(e)

    within_tolerance = bool(ranking_match and max_rel_err < float(ratio_tol))

    return {
        "ranking_match": bool(ranking_match),
        "per_channel_rel_err": per_err,
        "max_rel_err": float(max_rel_err),
        "within_tolerance": within_tolerance,
    }


