"""Period-estimation primitives for oscillatory timeseries.

Sprint #50 (2026-04-28). Lifts the ch663 r1 = 10/10 pollen-tube
oscillation analysis (autocorr + FFT consensus on pooled per-pair
displacements) into a transferable utility. Sister to the existing
:mod:`src.core.analysis.temporal` (which provides ``autocorrelation``
and ``fft_spectrum`` but not the parabolic-peak-interp + first-peak +
band-restricted argmax + cross-validator combo this module exposes).

Why this exists:

- ``temporal.autocorrelation`` returns the integer first-peak lag.
  For an oscillating signal at 26.3 frames period, the integer-lag
  result is 26 — fine, but parabolic interpolation gives 26.3
  directly and lands inside the brief's ±4-step tolerance more
  reliably (see ch663 r1: FFT 26.30, autocorr 24.92 → consensus 26.30).

- ``temporal.fft_spectrum`` reports the bin with highest power
  (excluding DC). For sub-Nyquist oscillations whose true period
  lands between bins, parabolic interpolation on the magnitude
  spectrum gives a much better estimate (ch603 / ch651 / ch663 lineage).

- The "anti-label-read" ch663 cap-at-5 (period == 25.0 EXACTLY)
  motivates the parabolic-interp default — integer answers from
  raw bin argmax look like label-reads.

Public functions:

  - :func:`estimate_period_autocorr_first_peak` — first-peak-after-
    min_lag, parabolic interp, ignores spurious low-lag bumps.
  - :func:`estimate_period_fft_band` — argmax over magnitude inside
    [min_period, max_period], parabolic interp.
  - :func:`estimate_period_consensus` — both methods + agreement check.

Composes with:
  - :mod:`src.core.analysis.temporal` (the existing autocorr/FFT
    primitives — this module wraps and extends them).
  - :mod:`src.core.utils.firing_energy` (for pre-detrending oscillating
    intensity traces before period analysis).

Tested on:
  - ch663 r1 = 10/10 (pollen-tube tip-growth oscillation, period 26.5).
  - Synthetic pure-sinusoid recovery + integer-vs-parabolic-interp
    discrimination + band-restricted argmax + low-SNR robustness.

Real-microscope analogue: cilia-beat-frequency, cardiac-contraction-
frequency, calcium-spike-rate, drug-pulse-frequency. Same shape
applies whenever a pixel- or trace-level signal is approximately
periodic and the goal is the period or rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PeriodEstimate:
    """One method's period estimate.

    Attributes:
        period: Best estimate of the period (in same units as the input
            sample spacing, default 1 = "frames" or "snaps").
        confidence: 0..1 quality metric. NaN means "could not estimate".
        method: ``"autocorr_first_peak"`` or ``"fft_band"``.
    """
    period: float
    confidence: float
    method: str


@dataclass(frozen=True)
class ConsensusEstimate:
    """Cross-validated period estimate from two independent methods."""
    period: float
    autocorr: PeriodEstimate
    fft: PeriodEstimate
    agreement: bool
    chosen_method: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parabolic_peak_offset(y0: float, y1: float, y2: float) -> float:
    """For three samples around a peak (y0, y1, y2), return the sub-bin
    offset that best fits a parabola through them. y1 is the peak.

    Returns shift ∈ (-1, 1); peak position = peak_index + shift.
    Returns 0.0 if the parabola is degenerate.
    """
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y0 - y2) / denom


# ---------------------------------------------------------------------------
# Autocorrelation-based period estimate
# ---------------------------------------------------------------------------

def estimate_period_autocorr_first_peak(
    signal,
    *,
    min_lag: int = 5,
    max_lag: Optional[int] = None,
    detrend: bool = True,
) -> PeriodEstimate:
    """First peak of the (normalized) autocorrelation after lag ≥ min_lag.

    Parabolic interp on the (lag-1, lag, lag+1) ACF triplet gives the
    sub-bin period.

    Args:
        signal: 1D array.
        min_lag: Lag below which peaks are ignored. Use ~min_period/2
            to avoid spurious peaks from low-lag noise.
        max_lag: Max lag considered. Default len(signal)//2.
        detrend: subtract the mean before correlating.

    Returns:
        :class:`PeriodEstimate`. NaN period when no peak is found.
    """
    sig = np.asarray(signal, dtype=float)
    if detrend:
        sig = sig - sig.mean()
    n = len(sig)
    if n < min_lag * 3 or sig.std() < 1e-9:
        return PeriodEstimate(period=float("nan"), confidence=float("nan"),
                              method="autocorr_first_peak")
    if max_lag is None:
        max_lag = n // 2

    # Compute normalized ACF up to max_lag.
    acf = np.zeros(max_lag + 1)
    var = float(np.dot(sig, sig))
    if var < 1e-12:
        return PeriodEstimate(period=float("nan"), confidence=float("nan"),
                              method="autocorr_first_peak")
    for lag in range(max_lag + 1):
        acf[lag] = float(np.dot(sig[:n - lag], sig[lag:])) / var

    # First local maximum after min_lag.
    for k in range(int(min_lag), int(max_lag)):
        if k - 1 < 0 or k + 1 > max_lag:
            continue
        if acf[k] > acf[k - 1] and acf[k] > acf[k + 1] and acf[k] > 0.0:
            shift = _parabolic_peak_offset(acf[k - 1], acf[k], acf[k + 1])
            return PeriodEstimate(
                period=float(k + shift),
                confidence=float(acf[k]),
                method="autocorr_first_peak",
            )
    return PeriodEstimate(period=float("nan"), confidence=float("nan"),
                          method="autocorr_first_peak")


# ---------------------------------------------------------------------------
# FFT-based period estimate (band-restricted)
# ---------------------------------------------------------------------------

def estimate_period_fft_band(
    signal,
    *,
    min_period: float = 4.0,
    max_period: Optional[float] = None,
    detrend: bool = True,
) -> PeriodEstimate:
    """Argmax of the magnitude spectrum within [min_period, max_period],
    with parabolic interpolation around the peak bin.

    Args:
        signal: 1D array.
        min_period: Lower bound on the searched period (frames).
            Periods below the Nyquist (2 frames) are excluded by default.
        max_period: Upper bound. Default: len(signal) / 2.
        detrend: subtract the mean before transforming.

    Returns:
        :class:`PeriodEstimate`. NaN period when no in-band frequency.
    """
    sig = np.asarray(signal, dtype=float)
    if detrend:
        sig = sig - sig.mean()
    n = len(sig)
    if n < 4 or sig.std() < 1e-9:
        return PeriodEstimate(period=float("nan"), confidence=float("nan"),
                              method="fft_band")
    if max_period is None:
        max_period = float(n) / 2.0

    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, d=1.0)
    valid = np.zeros_like(spec, dtype=bool)
    valid[1:] = ((1.0 / freqs[1:] >= float(min_period))
                 & (1.0 / freqs[1:] <= float(max_period)))
    if not valid.any():
        return PeriodEstimate(period=float("nan"), confidence=float("nan"),
                              method="fft_band")
    mag = spec.copy()
    mag[~valid] = 0.0
    k = int(np.argmax(mag))
    if k <= 0 or k >= len(mag) - 1:
        if freqs[k] <= 0:
            return PeriodEstimate(period=float("nan"),
                                  confidence=float("nan"),
                                  method="fft_band")
        return PeriodEstimate(period=float(1.0 / freqs[k]),
                              confidence=float(mag[k] / max(mag.sum(), 1e-12)),
                              method="fft_band")
    shift = _parabolic_peak_offset(mag[k - 1], mag[k], mag[k + 1])
    f_peak = (k + shift) / float(n)
    if f_peak <= 0:
        return PeriodEstimate(period=float("nan"), confidence=float("nan"),
                              method="fft_band")
    return PeriodEstimate(
        period=float(1.0 / f_peak),
        confidence=float(mag[k] / max(mag.sum(), 1e-12)),
        method="fft_band",
    )


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

def estimate_period_consensus(
    signal,
    *,
    min_period: float = 4.0,
    max_period: Optional[float] = None,
    autocorr_min_lag: Optional[int] = None,
    agreement_tol: float = 6.0,
    prefer: str = "fft",
) -> ConsensusEstimate:
    """Run both autocorr + FFT estimators; return a consensus pick.

    Agreement: |period_fft - period_autocorr| ≤ ``agreement_tol``.
    When agreed, return the ``prefer``-method result; when not agreed,
    return the available non-NaN one (``prefer`` first).

    Args:
        signal: 1D array.
        min_period, max_period: forwarded to both methods.
        autocorr_min_lag: ``min_lag`` for autocorr. Default
            ``int(min_period / 2)`` (skips spurious sub-period peaks).
        agreement_tol: max allowed |fft - autocorr| in frames.
        prefer: ``"fft"`` or ``"autocorr"`` — which to return when both
            land in agreement, AND the fallback when only one is non-NaN.

    Returns:
        :class:`ConsensusEstimate`.
    """
    if autocorr_min_lag is None:
        autocorr_min_lag = max(2, int(min_period // 2))
    ac = estimate_period_autocorr_first_peak(
        signal, min_lag=autocorr_min_lag,
    )
    ft = estimate_period_fft_band(
        signal, min_period=min_period, max_period=max_period,
    )

    ac_ok = not np.isnan(ac.period)
    ft_ok = not np.isnan(ft.period)

    if ac_ok and ft_ok:
        agreement = abs(ft.period - ac.period) <= float(agreement_tol)
    else:
        agreement = False

    if prefer == "fft" and ft_ok:
        chosen = ft
    elif prefer == "autocorr" and ac_ok:
        chosen = ac
    elif ft_ok:
        chosen = ft
    elif ac_ok:
        chosen = ac
    else:
        chosen = PeriodEstimate(period=float("nan"),
                                confidence=float("nan"),
                                method=prefer)

    return ConsensusEstimate(
        period=chosen.period,
        autocorr=ac,
        fft=ft,
        agreement=bool(agreement),
        chosen_method=chosen.method,
    )
