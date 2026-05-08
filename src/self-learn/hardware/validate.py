"""Pre-experiment configuration validation.

Checks that the microscope hardware has the channels, objectives, SLM,
and devices required for a particular experiment before it starts.
Catches configuration mismatches early rather than mid-experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import get_config, MicroscopeConfig


@dataclass
class ValidationResult:
    """Result of a hardware validation check."""
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation: {status}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def validate_channels(core, required: list[str]) -> ValidationResult:
    """Check that all required channels are available.

    Args:
        core: pymmcore-plus core instance.
        required: List of channel names needed (e.g. ["brightfield", "GFP"]).

    Returns:
        ValidationResult with errors for missing channels.
    """
    result = ValidationResult()
    cfg = get_config(core)

    if not cfg.channel_group:
        result.add_error("No channel config group found on microscope")
        return result

    available = set(cfg.available_channels)
    for ch in required:
        if ch not in available:
            result.add_error(
                f"Channel '{ch}' not available. "
                f"Available: {sorted(available)}"
            )

    return result


def validate_objectives(core, required_mags: list[int]) -> ValidationResult:
    """Check that required objective magnifications are available.

    Args:
        core: pymmcore-plus core instance.
        required_mags: List of magnifications needed (e.g. [10, 40]).

    Returns:
        ValidationResult with errors for missing objectives.
    """
    result = ValidationResult()
    cfg = get_config(core)

    if not cfg.objective_device:
        if required_mags:
            result.add_warning("No objective device found — cannot verify magnifications")
        return result

    available_mags = set()
    for _, label in cfg.objective_labels:
        mag = cfg.magnification_for_label(label)
        if mag:
            available_mags.add(mag)

    for mag in required_mags:
        if mag not in available_mags:
            result.add_error(
                f"Objective {mag}x not available. "
                f"Available: {sorted(available_mags)}"
            )

    return result


def validate_slm(core) -> ValidationResult:
    """Check that SLM device is available.

    Returns:
        ValidationResult with error if no SLM found.
    """
    result = ValidationResult()
    cfg = get_config(core)

    if not cfg.slm_device:
        result.add_error("No SLM device found on microscope")

    return result


def validate_stage(core) -> ValidationResult:
    """Check that XY stage is available.

    Returns:
        ValidationResult with error if no stage found.
    """
    result = ValidationResult()
    cfg = get_config(core)

    if not cfg.xy_device:
        result.add_error("No XY stage device found")

    return result


def validate_z(core) -> ValidationResult:
    """Check that Z/focus device is available.

    Returns:
        ValidationResult with error if no Z device found.
    """
    result = ValidationResult()
    cfg = get_config(core)

    if not cfg.z_device:
        result.add_error("No Z/focus device found")

    return result


def validate_experiment(
    core,
    channels: Optional[list[str]] = None,
    objectives: Optional[list[int]] = None,
    needs_slm: bool = False,
    needs_stage: bool = False,
    needs_z: bool = False,
) -> ValidationResult:
    """Comprehensive pre-experiment validation.

    Checks all requested hardware requirements in one call.

    Args:
        core: pymmcore-plus core instance.
        channels: Required channel names, or None to skip check.
        objectives: Required magnifications, or None to skip check.
        needs_slm: Whether experiment requires SLM.
        needs_stage: Whether experiment requires XY stage.
        needs_z: Whether experiment requires Z/focus device.

    Returns:
        Combined ValidationResult with all errors and warnings.
    """
    result = ValidationResult()

    if channels:
        ch_result = validate_channels(core, channels)
        result.errors.extend(ch_result.errors)
        result.warnings.extend(ch_result.warnings)

    if objectives:
        obj_result = validate_objectives(core, objectives)
        result.errors.extend(obj_result.errors)
        result.warnings.extend(obj_result.warnings)

    if needs_slm:
        slm_result = validate_slm(core)
        result.errors.extend(slm_result.errors)

    if needs_stage:
        stage_result = validate_stage(core)
        result.errors.extend(stage_result.errors)

    if needs_z:
        z_result = validate_z(core)
        result.errors.extend(z_result.errors)

    if result.errors:
        result.passed = False

    return result
