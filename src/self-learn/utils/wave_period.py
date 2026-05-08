"""Preview-based wave-period estimator for excitable-tissue scouts.

Auto-tunes the ``n_burst`` parameter of
``src.recipes.event_driven_modality_switch.modality_switch_pipeline``
from a brief preview burst, replacing the GCaMP-tuned default of 25
frames that doesn't transfer to slower physics (cardio AP, cAMP /
Dictyostelium spirals).

The recipe lesson (ch601 r1 → r2): σ × mean's
constitutive-vs-transient asymmetry develops only when the burst
spans ≥ 4 wave cycles. For an unknown sample the agent has no a
priori period estimate; this estimator runs a tiny preview burst,
locates the dominant FFT peak in the brightest-pixel time series,
and recommends ``n_burst ≈ 4 × period_in_frames``.

Sprint #14 (2026-04-26).

Functions:
    estimate_wave_period(core, channel, ...) -> dict
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


DEFAULT_N_BURST = 25
DEFAULT_PREVIEW_FRAMES = 10
DEFAULT_PREVIEW_EXPOSURE_MS = 20.0
BURST_MULTIPLIER = 4
BRIGHT_PIXEL_FRACTION = 0.05
FLAT_SPECTRUM_PEAK_RATIO = 3.0


def _bright_pixel_trace(
    stack: np.ndarray, *, top_fraction: float = BRIGHT_PIXEL_FRACTION
) -> np.ndarray:
    """Per-frame mean intensity of the brightest ``top_fraction`` pixels.

    Picks pixels by their *mean across time* (so the same pixel set is
    used for every frame) — that way the time series reflects the
    illumination of the high-signal cells, not whichever pixel
    happened to be brightest in a given frame.
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3D (T, H, W), got shape {stack.shape}")
    pix_mean = stack.mean(axis=0).ravel()
    threshold = np.quantile(pix_mean, 1.0 - float(top_fraction))
    bright = pix_mean >= threshold
    n_pixels = max(int(bright.sum()), 1)
    return stack.reshape(stack.shape[0], -1)[:, bright].mean(axis=1) if n_pixels else stack.mean(axis=(1, 2))


def _fft_dominant(
    series: np.ndarray, fs_hz: float
) -> tuple[float, float, float, bool]:
    """Locate the dominant frequency in a 1-D series.

    Returns ``(peak_hz, peak_amp, confidence, aliased)`` where
    ``confidence`` is ``peak_amp / sum(non_dc_amps)`` (in [0, 1]) and
    ``aliased`` is True when the peak falls in the Nyquist bin.
    """
    n = series.size
    if n < 4:
        return 0.0, 0.0, 0.0, False
    detrended = np.asarray(series, dtype=float) - float(series.mean())
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(detrended * window))
    if spec.size <= 1:
        return 0.0, 0.0, 0.0, False
    spec_no_dc = spec[1:]
    peak_idx_no_dc = int(np.argmax(spec_no_dc))
    peak_idx = peak_idx_no_dc + 1
    peak_amp = float(spec[peak_idx])
    total_no_dc = float(spec_no_dc.sum())
    confidence = peak_amp / total_no_dc if total_no_dc > 1e-9 else 0.0

    median_amp = float(np.median(spec_no_dc))
    if median_amp > 1e-9 and peak_amp / median_amp < FLAT_SPECTRUM_PEAK_RATIO:
        confidence = 0.0
        return 0.0, peak_amp, confidence, False

    freqs_hz = np.fft.rfftfreq(n, d=1.0 / float(fs_hz))
    peak_hz = float(freqs_hz[peak_idx])
    aliased = peak_idx == spec.size - 1
    return peak_hz, peak_amp, float(confidence), aliased


def estimate_wave_period(
    core,
    *,
    channel: str,
    n_preview: int = DEFAULT_PREVIEW_FRAMES,
    exposure_ms: float = DEFAULT_PREVIEW_EXPOSURE_MS,
    burst_multiplier: int = BURST_MULTIPLIER,
    fallback_n_burst: int = DEFAULT_N_BURST,
    burst_fn: Optional[object] = None,
) -> dict:
    """Estimate the dominant wave period and recommend ``n_burst``.

    Args:
        core: a connected microscope-core (CMMCorePlus or pymmcore-proxy).
        channel: Channel preset to acquire on.
        n_preview: number of preview frames (small, typically 10).
        exposure_ms: per-frame exposure for the preview.
        burst_multiplier: ``n_burst_recommended = round(M × period_dt)``.
        fallback_n_burst: returned when the spectrum is flat (no
            detectable oscillation).
        burst_fn: optional callable matching
            ``event_driven_modality_switch.burst_scout``'s signature
            ``(core, *, channel, n_frames, exposure_ms) -> (stack, wall_s)``.
            Lets unit tests inject a synthetic sequence-acquirer
            without needing a live core. Defaults to lazy-importing
            ``event_driven_modality_switch.burst_scout``.

    Returns dict with:
        period_dt: estimated period in frame units (or 0 if undetectable)
        n_burst_recommended: int (≥ ``fallback_n_burst``)
        confidence: float in [0, 1] (0 means fall back to default)
        fft_peak_hz: float
        fft_peak_amp: float
        aliased: bool
        n_preview: int (frames actually returned)
        wall_s: float (preview wall-clock duration)
    """
    if burst_fn is None:
        from src.recipes.event_driven_modality_switch import burst_scout
        burst_fn = burst_scout

    stack, wall_s = burst_fn(
        core, channel=channel, n_frames=int(n_preview),
        exposure_ms=float(exposure_ms),
    )
    n_actual = stack.shape[0]
    fs_hz = n_actual / max(float(wall_s), 1e-6)

    series = _bright_pixel_trace(stack)
    peak_hz, peak_amp, confidence, aliased = _fft_dominant(series, fs_hz)

    if confidence == 0.0 or peak_hz <= 0.0:
        return {
            "period_dt": 0.0,
            "n_burst_recommended": int(fallback_n_burst),
            "confidence": 0.0,
            "fft_peak_hz": peak_hz,
            "fft_peak_amp": peak_amp,
            "aliased": aliased,
            "n_preview": int(n_actual),
            "wall_s": float(wall_s),
        }

    period_s = 1.0 / peak_hz
    period_dt = period_s * fs_hz
    n_burst_recommended = max(
        int(fallback_n_burst), int(round(burst_multiplier * period_dt))
    )
    if aliased:
        warnings.warn(
            f"FFT peak at Nyquist ({peak_hz:.3f} Hz) — preview may be undersampled; "
            f"n_burst_recommended={n_burst_recommended} likely under-estimates.",
            stacklevel=2,
        )

    return {
        "period_dt": float(period_dt),
        "n_burst_recommended": int(n_burst_recommended),
        "confidence": float(confidence),
        "fft_peak_hz": float(peak_hz),
        "fft_peak_amp": float(peak_amp),
        "aliased": bool(aliased),
        "n_preview": int(n_actual),
        "wall_s": float(wall_s),
    }
