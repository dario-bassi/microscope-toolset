"""Utility modules.

General utilities:
    diagnostics    -- Save snapshots / overlays to disk for visual verification.
                      save_snapshot(img, label, save_dir) →  Path
                      save_overlay(img, cells, label, save_dir) → Path
                      save_composite(images, titles, label, save_dir) → Path
    image          -- Grayscale conversion, normalization, contrast
    showcase       -- Matplotlib-native multi-panel showcase figures (Panel + register)
    experiment_log -- Structured JSON experiment logging (ExperimentLog)
    report         -- Experiment report generation with statistics
    validation     -- Result validation and sanity checks
    preflight      -- Code-callable preflight checks
    sample_classifier -- Cheap-feature image → sample-class label (7 classes)
    auto_recipe    -- Image + core → RecipeSuggestion dispatcher
    mda_diagnostics -- run_events_checked() — MDA frame-count guard
    agree_or_flag  -- Two-estimator agreement check (cross-validate, don't average)
    axis_sweep     -- Generic hardware-axis sweep + argmax

Image-only measurement primitives (pure numpy/scipy — no hardware calls):
    sensorless_ao          -- DM-state sweep + argmax + per-axis-presence inference
    slm_masks              -- circle / gaussian / ring SLM mask builders
    spectral_leak          -- N×N leak matrix + unmix(K, observed)
    fft_peak               -- FFTPeak dataclass + find_fft_peak + find_top_n_fft_peaks
    fluorophore_brightness -- BrightnessRanking: predict ε×Φ, measure, compare
    rate_limited_drive     -- predict_n_steps + drive_to_band closed-loop
    pulsed_schedule        -- predict_segment_end + plan_trajectory (multi-waypoint MPC)
    dose_threshold         -- geometric_doses + sweep_threshold + bisect_threshold
    firing_energy          -- Bleach-immune per-pixel firing-rate score
                              (firing_energy + detrended_sigma)
    wave_period            -- estimate_wave_period(core, burst_fn) → recommended n_burst

Note: utility primitives are NOT re-exported below — callers should use
``from self_learn.utils.<module> import <name>`` directly.
"""

from .diagnostics import save_snapshot, save_overlay, save_composite
from .image import to_grayscale, normalize, auto_contrast
from .showcase import make_showcase, Panel, register
from .experiment_log import ExperimentLog
from .report import (
    generate_report, format_markdown, measurement_table,
    phase_comparison, experiment_timeline,
)
from .validation import (
    validate_answer, validate_nc_ratio, validate_kinetics, validate_count,
    extract_answer_keys,
)
from .preflight import run_preflight, check_nc_psf, check_tracking
