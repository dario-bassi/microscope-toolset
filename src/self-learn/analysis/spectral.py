"""Spectral unmixing and bleedthrough correction.

Separates overlapping fluorophore signals in multi-channel
fluorescence microscopy. Handles linear unmixing, simple
bleedthrough correction, and autofluorescence removal.

Functions:
    bleedthrough_correct  -- Correct channel cross-talk
    linear_unmix          -- NNLS spectral unmixing with reference spectra
    estimate_bleedthrough -- Estimate cross-talk from single-label controls
    ratio_image           -- Compute ratiometric image from two channels
    autofluorescence_subtract -- Remove autofluorescence component
"""

import numpy as np


def bleedthrough_correct(target, source, coefficient):
    """Correct bleedthrough from source channel into target channel.

    corrected = target - coefficient * source

    Args:
        target: 2D array, channel with bleedthrough contamination.
        source: 2D array, channel causing bleedthrough.
        coefficient: float, bleedthrough fraction (0-1).
            Fraction of source signal that leaks into target.

    Returns:
        dict with:
            corrected: 2D array, bleedthrough-corrected image.
            bleedthrough_image: 2D array, estimated bleedthrough signal.
            correction_fraction: float, fraction of target signal removed.
    """
    target = np.asarray(target, dtype=float)
    source = np.asarray(source, dtype=float)

    bleed = coefficient * source
    corrected = target - bleed
    corrected = np.clip(corrected, 0, None)

    target_sum = float(target.sum())
    bleed_sum = float(bleed.sum())
    frac = bleed_sum / target_sum if target_sum > 0 else 0.0

    return {
        "corrected": corrected,
        "bleedthrough_image": bleed,
        "correction_fraction": round(frac, 4),
    }


def linear_unmix(channels, reference_spectra):
    """Linear spectral unmixing using non-negative least squares.

    Decomposes multi-channel image into individual fluorophore
    contributions using reference emission spectra.

    Each pixel: channels = reference_spectra @ abundances (NNLS)

    Args:
        channels: 3D array (C, H, W), multi-channel image stack.
        reference_spectra: 2D array (C, N), reference emission spectra
            for N fluorophores measured across C channels.

    Returns:
        dict with:
            abundances: 3D array (N, H, W), unmixed fluorophore maps.
            residual: 2D array (H, W), per-pixel fitting residual.
            r_squared: float, overall goodness of fit.
    """
    from scipy.optimize import nnls

    channels = np.asarray(channels, dtype=float)
    reference_spectra = np.asarray(reference_spectra, dtype=float)

    n_channels, h, w = channels.shape
    n_fluors = reference_spectra.shape[1]

    # Reshape for vectorized NNLS
    pixel_data = channels.reshape(n_channels, -1).T  # (H*W, C)
    abundances = np.zeros((n_fluors, h * w))
    residuals = np.zeros(h * w)

    for i in range(h * w):
        spectrum = pixel_data[i]
        if np.sum(spectrum) < 1e-6:
            continue
        x, res = nnls(reference_spectra, spectrum)
        abundances[:, i] = x
        fitted = reference_spectra @ x
        residuals[i] = np.sqrt(np.sum((spectrum - fitted) ** 2))

    abundances = abundances.reshape(n_fluors, h, w)
    residual_map = residuals.reshape(h, w)

    # R² calculation
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((pixel_data - pixel_data.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "abundances": abundances,
        "residual": residual_map,
        "r_squared": round(float(max(r2, 0.0)), 4),
    }


def estimate_bleedthrough(source_control, target_control, mask=None):
    """Estimate bleedthrough coefficient from single-label controls.

    Uses a control sample where only the source fluorophore is present.
    The signal appearing in the target channel is bleedthrough.

    coefficient = mean(target[mask]) / mean(source[mask])

    Args:
        source_control: 2D array, source channel from single-label control.
        target_control: 2D array, target channel from same control.
        mask: 2D bool array or None, region where fluorophore is present.
            If None, uses pixels where source > Otsu threshold.

    Returns:
        dict with:
            coefficient: float, bleedthrough fraction.
            source_mean: float, mean source signal.
            target_mean: float, mean target (bleedthrough) signal.
            n_pixels: int, pixels used for estimation.
    """
    source = np.asarray(source_control, dtype=float)
    target = np.asarray(target_control, dtype=float)

    if mask is None:
        # Auto-threshold: use pixels above median + 2*std
        threshold = np.median(source) + 2 * np.std(source)
        mask = source > max(threshold, 1.0)

    mask = np.asarray(mask, dtype=bool)
    n_px = int(mask.sum())

    if n_px < 10:
        return {
            "coefficient": 0.0,
            "source_mean": 0.0,
            "target_mean": 0.0,
            "n_pixels": n_px,
        }

    src_mean = float(source[mask].mean())
    tgt_mean = float(target[mask].mean())
    coeff = tgt_mean / src_mean if src_mean > 0 else 0.0

    return {
        "coefficient": round(max(coeff, 0.0), 4),
        "source_mean": round(src_mean, 4),
        "target_mean": round(tgt_mean, 4),
        "n_pixels": n_px,
    }


def ratio_image(channel1, channel2, background1=0, background2=0, clip_range=None):
    """Compute ratiometric image from two channels.

    ratio = (ch1 - bg1) / (ch2 - bg2)

    Used for FRET, Fura-2 calcium imaging, pH indicators, etc.

    Args:
        channel1: 2D array, numerator channel.
        channel2: 2D array, denominator channel.
        background1: float, background for channel 1.
        background2: float, background for channel 2.
        clip_range: tuple (min, max) or None, clip output ratio range.

    Returns:
        dict with:
            ratio: 2D array, ratio image.
            mean_ratio: float, mean ratio in signal region.
            mask: 2D bool array, pixels where denominator is above threshold.
    """
    ch1 = np.asarray(channel1, dtype=float) - background1
    ch2 = np.asarray(channel2, dtype=float) - background2

    # Only compute ratio where denominator is significant
    # Use 10% of the mean as threshold to avoid dividing by near-zero
    ch2_mean = float(ch2.mean())
    denom_threshold = max(ch2_mean * 0.1, 1.0) if ch2_mean > 0 else 1.0
    mask = ch2 >= denom_threshold

    ratio = np.zeros_like(ch1)
    ratio[mask] = ch1[mask] / ch2[mask]

    if clip_range is not None:
        ratio = np.clip(ratio, clip_range[0], clip_range[1])

    mean_r = float(ratio[mask].mean()) if mask.sum() > 0 else 0.0

    return {
        "ratio": ratio,
        "mean_ratio": round(mean_r, 4),
        "mask": mask,
    }


def autofluorescence_subtract(image, autofluor_reference, scale=1.0):
    """Subtract autofluorescence using a reference channel/image.

    corrected = image - scale * autofluor_reference

    Args:
        image: 2D array, fluorescence image with autofluorescence.
        autofluor_reference: 2D array, autofluorescence reference
            (e.g., unstained control or separate AF channel).
        scale: float, scaling factor for autofluorescence.

    Returns:
        dict with:
            corrected: 2D array, autofluorescence-corrected image.
            af_contribution: float, fraction of signal from AF.
    """
    image = np.asarray(image, dtype=float)
    af_ref = np.asarray(autofluor_reference, dtype=float) * scale

    corrected = image - af_ref
    corrected = np.clip(corrected, 0, None)

    img_sum = float(image.sum())
    af_sum = float(af_ref.sum())
    af_frac = af_sum / img_sum if img_sum > 0 else 0.0

    return {
        "corrected": corrected,
        "af_contribution": round(min(af_frac, 1.0), 4),
    }
