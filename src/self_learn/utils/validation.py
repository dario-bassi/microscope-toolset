"""Pre-submission result validation.

Catches common errors by checking values against expected biological ranges
before submitting challenge solutions. This prevents submitting technically
computed but biologically implausible values.

Functions:
    validate_answer    -- Check answer dict values against expected ranges
    validate_nc_ratio  -- N:C ratio sanity checks
    validate_kinetics  -- Half-time / rate constant sanity checks
    validate_count     -- Cell/object count sanity checks
"""

import warnings

import numpy as np


def validate_answer(answer, rules=None, required_keys=None):
    """Check answer values against expected ranges and completeness.

    Args:
        answer: dict of key-value pairs to validate.
        rules: dict mapping answer keys to (min, max) tuples.
            If None, skips range checking (only does type checks).
        required_keys: list of key names that must be present.
            If None, skips completeness checking.

    Returns:
        dict with:
            valid: bool, True if all checks pass.
            warnings: list of warning strings.
            errors: list of error strings.
    """
    warns = []
    errors = []

    # Completeness check
    if required_keys is not None:
        missing = [k for k in required_keys if k not in answer]
        if missing:
            errors.append(f"Missing required keys: {missing}")

    for key, value in answer.items():
        # Type checks
        if isinstance(value, (np.integer, np.floating)):
            warns.append(f"'{key}' is numpy type {type(value).__name__}; "
                        f"cast to Python float/int for JSON serialization")

        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            errors.append(f"'{key}' = {value} (NaN or Inf)")

        # Range checks
        if rules and key in rules:
            lo, hi = rules[key]
            if lo is not None and value < lo:
                warns.append(f"'{key}' = {value} is below expected minimum {lo}")
            if hi is not None and value > hi:
                warns.append(f"'{key}' = {value} is above expected maximum {hi}")

    valid = len(errors) == 0
    if warns:
        for w in warns:
            warnings.warn(w)

    return {
        'valid': valid,
        'warnings': warns,
        'errors': errors,
    }


def validate_nc_ratio(ratio, context='resting'):
    """Validate a nuclear-to-cytoplasmic ratio.

    Args:
        ratio: float, N:C ratio value.
        context: 'resting' (expect 0.2-1.0) or 'stimulated' (expect 1.0-10.0).

    Returns:
        dict with valid, warnings, errors.
    """
    warns = []
    errors = []

    if np.isnan(ratio) or np.isinf(ratio):
        errors.append(f"N:C ratio is {ratio}")
        return {'valid': False, 'warnings': warns, 'errors': errors}

    if ratio <= 0:
        errors.append(f"N:C ratio = {ratio} (must be positive)")
    elif ratio > 50:
        warns.append(f"N:C ratio = {ratio} is extremely high; check cytoplasm mask")

    if context == 'resting' and ratio > 2.0:
        warns.append(f"Resting N:C = {ratio:.3f} is high for cytoplasmic protein; "
                     "check if nuclear/cytoplasm masks are swapped")
    if context == 'stimulated' and ratio < 0.5:
        warns.append(f"Stimulated N:C = {ratio:.3f} is low; "
                     "drug may not have taken effect or masks are wrong")

    return {
        'valid': len(errors) == 0,
        'warnings': warns,
        'errors': errors,
    }


def validate_kinetics(half_time_s, experiment_duration_s=None,
                      expected_range=None):
    """Validate kinetic parameters.

    Args:
        half_time_s: float, half-time in seconds.
        experiment_duration_s: float or None, total duration of measurement.
        expected_range: (min_s, max_s) tuple or None.

    Returns:
        dict with valid, warnings, errors.
    """
    warns = []
    errors = []

    if np.isnan(half_time_s) or np.isinf(half_time_s):
        errors.append(f"Half-time is {half_time_s}")
        return {'valid': False, 'warnings': warns, 'errors': errors}

    if half_time_s <= 0:
        errors.append(f"Half-time = {half_time_s}s (must be positive)")

    if experiment_duration_s is not None:
        if half_time_s > experiment_duration_s:
            warns.append(f"Half-time ({half_time_s:.1f}s) exceeds experiment "
                        f"duration ({experiment_duration_s:.1f}s); "
                        f"response may not have reached 50%")
        if half_time_s < 0.01 * experiment_duration_s:
            warns.append(f"Half-time ({half_time_s:.1f}s) is <1% of experiment "
                        f"duration; may indicate instantaneous response or error")

    if expected_range is not None:
        lo, hi = expected_range
        if half_time_s < lo:
            warns.append(f"Half-time {half_time_s:.1f}s below expected range "
                        f"[{lo}, {hi}]s")
        if half_time_s > hi:
            warns.append(f"Half-time {half_time_s:.1f}s above expected range "
                        f"[{lo}, {hi}]s")

    return {
        'valid': len(errors) == 0,
        'warnings': warns,
        'errors': errors,
    }


def extract_answer_keys(description):
    """Extract required answer keys from a challenge description.

    Parses the challenge text for lines matching the pattern
    ``- key_name: description`` which is the standard format for
    listing required answer fields.

    Args:
        description: str, challenge description text.

    Returns:
        list of str, extracted key names (e.g. ['baseline_nc_ratio',
            'max_nc_ratio', ...]).
    """
    import re
    # Match "- key_name: description" or "• key_name: description"
    matches = re.findall(r'[-•]\s*(\w+):\s*.+', description)
    # Filter out common non-answer keys (channel descriptions, notes)
    skip = {'brightfield', 'channel', 'hint', 'note', 'example', 'gfp',
            'dapi', 'task', 'step', 'warning', 'config', 'default'}
    return [m for m in matches if m.lower() not in skip
            and not m.startswith(('ch', 'Ch'))]


def validate_count(count, image_shape=None, expected_density=None,
                   min_count=0, max_count=None):
    """Validate object/cell count.

    Args:
        count: int, detected count.
        image_shape: (H, W) or None, for density checks.
        expected_density: float or None, expected objects per 1000 px^2.
        min_count: int, minimum expected count.
        max_count: int or None, maximum expected count.

    Returns:
        dict with valid, warnings, errors.
    """
    warns = []
    errors = []

    if count < 0:
        errors.append(f"Count = {count} (must be non-negative)")

    if count < min_count:
        warns.append(f"Count = {count} below minimum expected {min_count}")

    if max_count is not None and count > max_count:
        warns.append(f"Count = {count} above maximum expected {max_count}")

    if image_shape is not None and expected_density is not None:
        area = image_shape[0] * image_shape[1] / 1000.0
        expected = expected_density * area
        if count > 5 * expected:
            warns.append(f"Count = {count} is {count/expected:.1f}x the expected "
                        f"density ({expected:.0f} expected in this FOV)")
        elif count < 0.2 * expected:
            warns.append(f"Count = {count} is only {count/expected:.1%} of expected "
                        f"density ({expected:.0f} expected in this FOV)")

    return {
        'valid': len(errors) == 0,
        'warnings': warns,
        'errors': errors,
    }
