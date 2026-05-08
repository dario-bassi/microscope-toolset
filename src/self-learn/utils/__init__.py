"""Utility modules.

Existing utilities:
    diagnostics    -- Image saving for challenge submissions
    image          -- Grayscale conversion, normalization, contrast
    showcase       -- Matplotlib-native multi-panel showcase figures (Panel + register)
    experiment_log -- Structured JSON experiment logging
    report         -- Experiment report generation with statistics
    validation     -- Pre-submission result validation and sanity checks
    preflight      -- Code-callable pre-submission preflight checks
    sample_classifier -- Cheap-feature image classifier (7 sample classes)
    auto_recipe    -- Image+core → RecipeSuggestion (dispatcher)

Image-only measurement primitives (camera + standard device props):
    sensorless_ao          -- DM-state sweep + argmax + per-axis-presence
                              inference (sprint #25)
    slm_masks              -- circle / gaussian / ring mask builders
                              vectorised over centre count (sprint #28)
    spectral_leak          -- top_percentile_mask + measure_leak (N×N matrix)
                              + unmix (K^-1 @ observed) (sprint #31)
    fft_peak               -- FFTPeak dataclass + find_fft_peak +
                              find_top_n_fft_peaks (±k suppression)
                              (sprint #32)
    fluorophore_brightness -- BrightnessRanking + predict + measure +
                              compare; ε × Φ closed-form prediction
                              (sprint #33)
    rate_limited_drive     -- predict_n_steps closed-form + drive_to_band
                              closed-loop early-stop (sprint #35; pure
                              math + caller-supplied step_fn / measure_fn)
    pulsed_schedule        -- predict_segment_end + plan_n_on_for_waypoint
                              + plan_trajectory for multi-waypoint
                              reference-trajectory MPC (sprint #37)
    dose_threshold         -- geometric_doses + sweep_threshold +
                              bisect_threshold for binary-cut-off /
                              minimum-effective-dose detection (sprint
                              #43; pure callbacks — apply_dose +
                              measure_response — so dose-agnostic over
                              SLM amplitude / laser mW / drug conc /
                              temperature / voltage). Third leg of the
                              dose-aware-control trio with
                              rate_limited_drive + pulsed_schedule.
    firing_energy          -- bleach-immune per-pixel firing-rate score:
                              firing_energy (np.diff().clip(0).sum()) +
                              detrended_sigma (σ after linear-trend
                              subtract). For pacemaker / wave-source
                              localisation in any channel with a global
                              decay envelope (GCaMP, voltage indicators,
                              FRAP, photoconversion) — ch651 r3 lesson.
    brief_parse            -- Structured-data extraction from
                              challenge.json briefs. parse_brief +
                              find_disclosed_coord_near +
                              validate_against_brief. ALWAYS run on
                              the brief at solve start — disclosed
                              priors / submit shapes / method-summary
                              gates are sitting in plain text and have
                              cost grades when ignored (ch651, ch623,
                              ch606 lessons).
    session_audit          -- Per-recipe + per-utils + per-core-module
                              score-distribution mining (sprint #23/#30)

Note: utility primitives are NOT re-exported below — callers should
``from src.core.utils.<module> import <name>`` directly.

REMOVED 2026-04-27 (bridge RPC surface closed by virtual-env):
    bridge_state, bridge_preflight, dose_budget, first_contact,
    per_cell_drop_out — all wrapped ``core._rpc("bridge.*")`` reads
    or removed Camera virtual properties (LastFrameBleachDose,
    LastFrameExposureMs, LastFramePixelCoverage, EmissionWavelengthNm,
    SimTime). On a real microscope these quantities must come from
    image analysis on camera frames — the helpers were transferability
    cheats. See virtual-env/skills/transferability-audit.md and the
    2026-04-27 message from virtual-env.
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
