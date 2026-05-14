"""Pre-recording preflight checks.

Code-callable guard that enforces parameter advisor usage, validates
result dicts, and catches common pitfalls before recording or reporting
any experiment result.

Functions:
    run_preflight  -- Run all preflight checks on a result dict
    check_nc_psf   -- Warn about PSF compression at low magnification
    check_tracking -- Warn about track fragmentation or low track count
"""

import numpy as np

from .validation import validate_answer


def run_preflight(answer, pixel_size_um=None,
                  rules=None, required_keys=None, sample_type=None,
                  n_visible_objects=None):
    """Run all preflight checks before recording an experiment result.

    Combines validate_answer checks with domain-specific warnings:
    - NaN / Inf / numpy type checks
    - Required key completeness
    - Range plausibility checks
    - N:C ratio PSF compression warnings
    - Tracking fragmentation warnings
    - Parameter advisor reminder

    Args:
        answer: dict of result key-value pairs.
        pixel_size_um: float or None, current pixel size (for PSF checks).
        rules: dict mapping keys to (min, max) ranges.
        required_keys: list of required result keys.
        sample_type: str or None (e.g. 'bacterium', 'nucleus', 'c_elegans').
        n_visible_objects: int or None, approximate number of visible objects
            in the image (for tracking sanity checks).

    Returns:
        dict with:
            passed: bool, True if no errors (warnings OK).
            errors: list of error strings.
            warnings: list of warning strings.
            advisor_used: bool, False (reminder to call parameter_advisor).
    """
    errors = []
    warnings = []

    # 1. Core validation
    result = validate_answer(answer, rules=rules, required_keys=required_keys)
    errors.extend(result['errors'])
    warnings.extend(result['warnings'])

    # 2. N:C PSF compression check
    nc_warnings = check_nc_psf(answer, pixel_size_um)
    warnings.extend(nc_warnings)

    # 3. Tracking fragmentation check
    track_warnings = check_tracking(answer, n_visible_objects)
    warnings.extend(track_warnings)

    # 4. Half-time plausibility
    ht_warnings = _check_half_time(answer)
    warnings.extend(ht_warnings)

    # 5. Count sanity
    count_warnings = _check_counts(answer, sample_type)
    warnings.extend(count_warnings)

    # 6. Parameter advisor reminder
    advisor_used = False  # caller should set this
    if pixel_size_um is not None and sample_type is not None:
        warnings.append(
            f"Reminder: call parameter_report() with pixel_size={pixel_size_um}, "
            f"object_type='{sample_type}' before analysis"
        )

    passed = len(errors) == 0
    return {
        'passed': passed,
        'errors': errors,
        'warnings': warnings,
        'advisor_used': advisor_used,
    }


def check_nc_psf(answer, pixel_size_um=None):
    """Check for PSF compression artifacts in N:C ratio measurements.

    At low magnification, PSF spreading dilutes nuclear signal into
    cytoplasm, compressing the measured N:C dynamic range by ~2x.

    Args:
        answer: dict with answer keys.
        pixel_size_um: float or None.

    Returns:
        list of warning strings.
    """
    warnings = []
    if pixel_size_um is None:
        return warnings

    psf_fwhm_um = 0.5  # typical visible-light PSF
    psf_px = psf_fwhm_um / pixel_size_um

    nc_keys = [k for k in answer if 'nc_ratio' in k.lower() or 'n_c_ratio' in k.lower()
               or k.lower().startswith('nc') or k.lower().endswith('nc')]

    for key in nc_keys:
        val = answer[key]
        if not isinstance(val, (int, float)):
            continue

        # At 20x (0.5 µm/px), PSF ~1px — borderline
        # At 10x (1.0 µm/px), PSF ~0.5px — but guard band helps
        if pixel_size_um >= 0.5 and val > 3.0:
            warnings.append(
                f"'{key}' = {val:.2f} at pixel_size={pixel_size_um}µm/px "
                f"(PSF={psf_px:.1f}px). High N:C at this magnification may "
                f"indicate PSF compression artifact. Use guard_band={max(1, int(round(psf_px)))} "
                f"in compute_nc_ratio() or switch to 40x."
            )
        if pixel_size_um >= 1.0 and 'max' in key.lower():
            warnings.append(
                f"'{key}' measured at 10x — PSF compression likely reduces "
                f"dynamic range by ~2x. Consider 40x for quantitative N:C."
            )

    return warnings


def check_tracking(answer, n_visible_objects=None):
    """Check for track fragmentation or implausible track counts.

    Args:
        answer: dict with answer keys.
        n_visible_objects: int or None, approximate visible object count.

    Returns:
        list of warning strings.
    """
    warnings = []

    track_keys = [k for k in answer
                  if 'n_tracked' in k.lower() or 'track_count' in k.lower()
                  or 'num_tracks' in k.lower()]

    for key in track_keys:
        val = answer[key]
        if not isinstance(val, (int, float)):
            continue

        if n_visible_objects is not None and val > 3 * n_visible_objects:
            warnings.append(
                f"'{key}' = {val} is {val/n_visible_objects:.1f}x the "
                f"visible object count ({n_visible_objects}). "
                f"Likely track fragmentation — increase max_dist or "
                f"decrease min_track_length."
            )

        if n_visible_objects is not None and val < 0.3 * n_visible_objects:
            warnings.append(
                f"'{key}' = {val} is only {val/n_visible_objects:.0%} of "
                f"visible objects ({n_visible_objects}). "
                f"Possible tracking failure — check detection threshold."
            )

    return warnings


def _check_half_time(answer):
    """Check half-time values for common pitfalls."""
    warnings = []

    ht_keys = [k for k in answer
               if 'half_time' in k.lower() or 'halftime' in k.lower()
               or 't_half' in k.lower() or 't50' in k.lower()]

    for key in ht_keys:
        val = answer[key]
        if not isinstance(val, (int, float)):
            continue
        if val <= 0:
            continue  # validate_answer catches this
        if val < 0.5:
            warnings.append(
                f"'{key}' = {val:.3f}s is very fast — check units "
                f"(seconds vs frames?)."
            )
        if val > 300:
            warnings.append(
                f"'{key}' = {val:.1f}s is >5 min — ensure this is within "
                f"the experiment duration."
            )

    return warnings


def _check_counts(answer, sample_type=None):
    """Check count values for plausibility."""
    warnings = []

    count_keys = [k for k in answer
                  if 'count' in k.lower() or 'n_cells' in k.lower()
                  or 'n_bacteria' in k.lower() or 'n_nuclei' in k.lower()]

    for key in count_keys:
        val = answer[key]
        if not isinstance(val, (int, float)):
            continue
        # NaN/Inf can't be cast to int — let validate_answer's NaN error
        # surface separately; skip the fractional check here.
        if isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))):
            continue

        # Fractional count warning
        if isinstance(val, float) and val != int(val):
            warnings.append(
                f"'{key}' = {val} is a non-integer count. "
                f"Cast to int before submitting."
            )

    return warnings
