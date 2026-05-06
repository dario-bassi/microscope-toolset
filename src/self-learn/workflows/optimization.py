"""Imaging parameter optimization workflows.

Systematic search for optimal exposure, gain, and other acquisition
parameters. Uses MDA-based parameter sweeps with SNR quantification.

Functions:
    optimize_exposure   -- Find best exposure time for a channel
    optimize_gain       -- Find best gain setting
    parameter_sweep     -- Full exposure × gain sweep with SNR matrix
    suggest_parameters  -- Quick recommendation from a few test shots
"""

import numpy as np
from useq import MDAEvent

from ..analysis.image_quality import assess_quality
from ..analysis.intensity import compute_snr


def optimize_exposure(images, exposures, target_snr=None):
    """Find optimal exposure from a set of test images.

    Evaluates SNR and saturation for each exposure and selects the best.
    The optimal exposure maximizes SNR without saturating.

    Args:
        images: list of 2D arrays, one per exposure setting.
        exposures: list of float, exposure times in ms.
        target_snr: float, optional. If provided, pick the shortest
            exposure that achieves this SNR.

    Returns:
        dict with:
            best_exposure: float, recommended exposure time.
            best_index: int, index into the input list.
            best_snr: float, SNR at the best exposure.
            results: list of dicts with per-exposure metrics.
            method: str, selection method used.
    """
    if len(images) != len(exposures):
        raise ValueError("images and exposures must have same length")
    if len(images) == 0:
        raise ValueError("At least one image required")

    results = []
    for img, exp in zip(images, exposures, strict=False):
        img_f = np.asarray(img, dtype=np.float64)
        snr_info = compute_snr(img_f)
        quality = assess_quality(img_f)

        results.append(
            {
                "exposure": exp,
                "snr": snr_info["snr"],
                "signal_mean": snr_info["signal_mean"],
                "bg_mean": snr_info["bg_mean"],
                "bg_std": snr_info["bg_std"],
                "saturation": quality["saturation_fraction"],
                "dynamic_range": quality["dynamic_range_fraction"],
                "focus": quality["focus_score"],
                "warnings": quality["warnings"],
            }
        )

    # Selection logic
    if target_snr is not None:
        # Shortest exposure that meets target SNR without saturation
        for i, r in enumerate(results):
            if r["snr"] >= target_snr and r["saturation"] < 0.01:
                return {
                    "best_exposure": exposures[i],
                    "best_index": i,
                    "best_snr": r["snr"],
                    "results": results,
                    "method": f"target_snr>={target_snr}",
                }

    # Default: best SNR among unsaturated images
    valid = [(i, r) for i, r in enumerate(results) if r["saturation"] < 0.01]
    if not valid:
        # All saturated — pick least saturated
        best_i = int(np.argmin([r["saturation"] for r in results]))
        method = "least_saturated"
    else:
        best_i = max(valid, key=lambda x: x[1]["snr"])[0]
        method = "max_snr_unsaturated"

    return {
        "best_exposure": exposures[best_i],
        "best_index": best_i,
        "best_snr": results[best_i]["snr"],
        "results": results,
        "method": method,
    }


def optimize_gain(images, gains):
    """Find optimal gain from test images at different gain settings.

    In most cameras, gain amplifies signal AND noise equally, so SNR
    doesn't improve with gain. Optimal gain is usually the lowest that
    gives sufficient signal for detection.

    Args:
        images: list of 2D arrays, one per gain setting.
        gains: list of float, gain values tested.

    Returns:
        dict with:
            best_gain: float, recommended gain.
            best_index: int, index into input list.
            results: list of per-gain metric dicts.
            snr_trend: str, 'improving', 'flat', or 'degrading'.
    """
    if len(images) != len(gains):
        raise ValueError("images and gains must have same length")
    if len(images) == 0:
        raise ValueError("At least one image required")

    results = []
    for img, gain in zip(images, gains, strict=False):
        img_f = np.asarray(img, dtype=np.float64)
        snr_info = compute_snr(img_f)
        quality = assess_quality(img_f)

        results.append(
            {
                "gain": gain,
                "snr": snr_info["snr"],
                "signal_mean": snr_info["signal_mean"],
                "bg_std": snr_info["bg_std"],
                "saturation": quality["saturation_fraction"],
                "dynamic_range": quality["dynamic_range_fraction"],
            }
        )

    # Check SNR trend
    snrs = [r["snr"] for r in results]
    if len(snrs) >= 2:
        slope = np.polyfit(range(len(snrs)), snrs, 1)[0]
        if slope > 0.5:
            snr_trend = "improving"
        elif slope < -0.5:
            snr_trend = "degrading"
        else:
            snr_trend = "flat"
    else:
        snr_trend = "unknown"

    # Pick lowest gain with acceptable SNR (>3) and no saturation
    for i, r in enumerate(results):
        if r["snr"] > 3 and r["saturation"] < 0.01:
            return {
                "best_gain": gains[i],
                "best_index": i,
                "results": results,
                "snr_trend": snr_trend,
            }

    # Fallback: best SNR
    best_i = int(np.argmax(snrs))
    return {
        "best_gain": gains[best_i],
        "best_index": best_i,
        "results": results,
        "snr_trend": snr_trend,
    }


def parameter_sweep(images, exposures, gains):
    """Evaluate a grid of exposure × gain combinations.

    Args:
        images: 2D list of images, images[i][j] = exposure_i, gain_j.
            Shape: (n_exposures, n_gains).
        exposures: list of float, exposure times.
        gains: list of float, gain values.

    Returns:
        dict with:
            snr_matrix: 2D array (n_exposures, n_gains) of SNR values.
            best_exposure: float.
            best_gain: float.
            best_snr: float.
            best_indices: (exp_idx, gain_idx).
            results: 2D list of per-combination metric dicts.
    """
    n_exp = len(exposures)
    n_gain = len(gains)

    if len(images) != n_exp:
        raise ValueError(f"Expected {n_exp} exposure rows, got {len(images)}")
    for i, row in enumerate(images):
        if len(row) != n_gain:
            raise ValueError(f"Row {i}: expected {n_gain} gain cols, got {len(row)}")

    snr_matrix = np.zeros((n_exp, n_gain))
    all_results = []

    for i, exp in enumerate(exposures):
        row_results = []
        for j, gain in enumerate(gains):
            img_f = np.asarray(images[i][j], dtype=np.float64)
            snr_info = compute_snr(img_f)
            quality = assess_quality(img_f)

            snr_matrix[i, j] = snr_info["snr"]
            row_results.append(
                {
                    "exposure": exp,
                    "gain": gain,
                    "snr": snr_info["snr"],
                    "signal_mean": snr_info["signal_mean"],
                    "bg_std": snr_info["bg_std"],
                    "saturation": quality["saturation_fraction"],
                }
            )
        all_results.append(row_results)

    # Find best: max SNR among unsaturated
    best_snr = -1
    best_idx = (0, 0)
    for i in range(n_exp):
        for j in range(n_gain):
            r = all_results[i][j]
            if r["saturation"] < 0.01 and r["snr"] > best_snr:
                best_snr = r["snr"]
                best_idx = (i, j)

    if best_snr < 0:
        # All saturated — pick least saturated
        min_sat = float("inf")
        for i in range(n_exp):
            for j in range(n_gain):
                if all_results[i][j]["saturation"] < min_sat:
                    min_sat = all_results[i][j]["saturation"]
                    best_idx = (i, j)
        best_snr = snr_matrix[best_idx]

    return {
        "snr_matrix": snr_matrix,
        "best_exposure": exposures[best_idx[0]],
        "best_gain": gains[best_idx[1]],
        "best_snr": float(best_snr),
        "best_indices": best_idx,
        "results": all_results,
    }


def suggest_parameters(image, current_exposure, current_gain=1.0, bit_depth=None):
    """Quick parameter suggestion from a single test image.

    Analyzes the current image and suggests adjustments to exposure
    and gain based on SNR, saturation, and dynamic range.

    Args:
        image: 2D array from a test acquisition.
        current_exposure: float, current exposure time in ms.
        current_gain: float, current gain setting.
        bit_depth: int, expected bit depth. Auto-detected if None.

    Returns:
        dict with:
            current_snr: float.
            suggested_exposure: float, recommended exposure.
            suggested_gain: float, recommended gain.
            exposure_factor: float, ratio of suggested/current.
            reasoning: list of str, explanation of suggestions.
            quality: dict from assess_quality().
    """
    img_f = np.asarray(image, dtype=np.float64)
    snr_info = compute_snr(img_f)
    quality = assess_quality(img_f, bit_depth=bit_depth)

    reasoning = []
    suggested_exposure = current_exposure
    suggested_gain = current_gain

    # Check saturation
    if quality["saturation_fraction"] > 0.01:
        factor = 0.5
        suggested_exposure = current_exposure * factor
        reasoning.append(
            f"Image saturated ({quality['saturation_fraction']:.1%}). "
            f"Reduce exposure from {current_exposure} to {suggested_exposure} ms."
        )
    elif quality["saturation_fraction"] > 0.001:
        reasoning.append("Near saturation — exposure is at upper limit.")
    else:
        # Check if we have headroom to increase exposure
        max_val = img_f.max()
        if bit_depth is not None:
            full_range = 2**bit_depth - 1
        elif max_val > 4095:
            full_range = 65535
        elif max_val > 255:
            full_range = 4095
        else:
            full_range = 255

        usage = max_val / full_range
        if usage < 0.3 and snr_info["snr"] < 10:
            # Underexposed — increase exposure
            factor = min(0.7 / max(usage, 0.01), 4.0)
            suggested_exposure = current_exposure * factor
            reasoning.append(
                f"Underexposed ({usage:.0%} of range used, SNR={snr_info['snr']:.1f}). "
                f"Increase exposure to {suggested_exposure:.1f} ms."
            )
        elif usage < 0.5 and snr_info["snr"] < 5:
            factor = min(0.6 / max(usage, 0.01), 2.0)
            suggested_exposure = current_exposure * factor
            reasoning.append(
                f"Low SNR ({snr_info['snr']:.1f}). "
                f"Increase exposure to {suggested_exposure:.1f} ms."
            )

    # Check if gain could help (only if exposure is already at limit)
    if snr_info["snr"] < 3 and suggested_exposure == current_exposure:
        # SNR is very low and we didn't suggest exposure change
        reasoning.append(
            f"Low SNR ({snr_info['snr']:.1f}). Consider increasing exposure "
            f"rather than gain — gain amplifies noise too."
        )

    if not reasoning:
        reasoning.append(
            f"Parameters look good (SNR={snr_info['snr']:.1f}, "
            f"no saturation, {quality['dynamic_range_fraction']:.0%} dynamic range)."
        )

    return {
        "current_snr": snr_info["snr"],
        "suggested_exposure": suggested_exposure,
        "suggested_gain": suggested_gain,
        "exposure_factor": suggested_exposure / max(current_exposure, 1e-10),
        "reasoning": reasoning,
        "quality": quality,
    }


def make_sweep_events(channel, exposures, gains=None, group=None):
    """Generate MDA events for a parameter sweep.

    Creates one MDAEvent per exposure/gain combination, suitable for
    run_events(). Each event includes the parameters in its metadata
    for later matching to results.

    Args:
        channel: str, channel config name.
        exposures: list of float, exposure times to test.
        gains: list of float or None. If None, uses current gain.
        group: str, config group name.

    Returns:
        list of MDAEvent objects ready for run_events().
    """
    events = []
    gains_list = gains if gains is not None else [None]

    for exp in exposures:
        for gain in gains_list:
            evt_kwargs = {
                "channel": {"config": channel, "group": group} if group else {"config": channel},
                "exposure": exp,
            }
            if gain is not None:
                evt_kwargs["properties"] = [("Camera", "Gain", str(gain))]

            events.append(MDAEvent(**evt_kwargs))

    return events
