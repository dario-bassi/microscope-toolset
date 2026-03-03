"""Experiment workflows and acquisition protocols.

For standard fixed acquisitions (timelapse, Z-stack, multi-position), use
pymmcore-plus MDASequence directly. This package provides adaptive and
decision-making workflows that go beyond what MDASequence covers.

Modules:
    adaptive       -- Survey->decide->zoom->measure (+ adaptive_survey_mda)
    scanning       -- Multi-position scanning + dedup (+ scan_and_detect_mda)
    stage_tracking -- Closed-loop target tracking with motion prediction
    mda            -- adaptive_phase_events()
    autofocus      -- Software autofocus (+ autofocus_mda)
    optogenetics   -- SLM-targeted stimulation (soma detection, ΔF/F)
    scouting       -- Pre-acquisition characterization (channel_scout, sample_survey)
    tiling         -- Multi-position tiled imaging and stitching
    dose_response  -- Plate layout, normalization, and IC50/EC50 fitting
    multi_scale    -- Multi-scale coordinator (overview_first, zoom, params)
    batch          -- Multi-position batch acquisition and analysis
    experiment     -- Multi-phase experiment protocol builder
    optimization   -- SNR optimization and parameter sweep workflows
    organoid       -- Organoid Z-scan, morphometry, and volume estimation
"""

from .adaptive import (
    pixel_to_world, world_to_pixel, survey_cells,
    find_clusters, rank_by_feature, zoom_and_measure, adaptive_survey,
    adaptive_survey_mda,
)
from .scanning import (
    grid_positions, deduplicate_cells,
    scan_and_detect_mda,
)
from .stage_tracking import (
    make_tracker, predict_position, update_velocity,
    locate_target, center_on_target, spiral_search, track_target,
    track_multiple,
)
from .mda import (
    adaptive_phase_events,
)
from .autofocus import (
    focus_metric, sweep_focus, coarse_fine_focus,
    make_focus_state, check_and_correct_focus, autofocus_mda,
    drift_corrected_timelapse,
)
from .optogenetics import (
    detect_somata, detect_somata_fluorescence,
    make_soma_rois, analyze_stimulation,
    slm_stimulation_experiment, connectivity_mapping,
)
from .scouting import (
    channel_scout, sample_survey, experiment_protocol,
)
from .tiling import (
    tile_positions, stitch_tiles,
    phase_correlation, align_tile_pair, stitch_tiles_aligned,
)
from .dose_response import (
    make_plate_layout, measure_plate, normalize_responses,
    auto_ec50, dose_response_pipeline,
)
from .multi_scale import (
    suggest_magnification, scale_params, overview_first,
    multi_scale_measure, validate_object_size,
)
from .batch import (
    tile_and_analyze, multi_position_measure, aggregate_results,
    measure_nuclear_expression, identify_hotspot,
)
from .experiment import (
    phase_timelapse, baseline_treatment, wash_experiment,
    temperature_shift, extract_phase_data,
)
from .optimization import (
    optimize_exposure, optimize_gain, parameter_sweep,
    suggest_parameters, make_sweep_events,
)
from .organoid import (
    find_equatorial_z, measure_organoid, organoid_z_profile,
)
from .bacteria_trap import (
    detect_bacteria_gfp, detect_bacteria_area_threshold,
    count_bacteria_in_circle, compute_enrichment, measure_bacteria_intensity,
)
