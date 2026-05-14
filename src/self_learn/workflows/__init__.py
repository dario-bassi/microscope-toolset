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
    optogenetics   -- SLM-targeted stimulation (soma detection, ΔF/F, connectivity_mapping_events)
    scouting       -- Pre-acquisition characterization (channel_scout, sample_survey, find_sample_events)
    tiling         -- Multi-position tiled imaging and stitching
    dose_response  -- Plate layout, normalization, and IC50/EC50 fitting
    multi_scale    -- Multi-scale coordinator (overview_first, zoom, params)
    batch          -- Multi-position batch acquisition and analysis
    experiment     -- Multi-phase experiment protocol builder
    optimization   -- SNR optimization and parameter sweep workflows
    organoid       -- Organoid Z-scan, morphometry, and volume estimation
    adaptive_monitor -- Multi-position patrol with event-triggered zoom
    focus_map        -- Multi-position Z calibration and interpolation
    well_plate       -- Multi-well plate imaging, analysis, and comparison
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
    focus_metric, parabolic_peak_interp,
    sweep_focus, coarse_fine_focus,
    make_focus_state, check_and_correct_focus, autofocus_mda,
    drift_corrected_timelapse,
)
from .optogenetics import (
    detect_somata, detect_somata_fluorescence,
    make_soma_rois, analyze_stimulation,
    slm_stimulation_experiment, connectivity_mapping,
    connectivity_mapping_events, infer_connectivity_dff,
)
from .scouting import (
    channel_scout, sample_survey, find_sample_events, experiment_protocol,
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
# bacteria_trap moved to src.recipes (sim-specific physics); import directly if needed.
from .adaptive_monitor import (
    adaptive_monitor, adaptive_monitor_events,
    make_position_states, default_event_scorer,
    make_bright_region_scorer, make_count_change_scorer,
    PositionState, CapturedEvent,
)
from .focus_map import (
    make_focus_map, predict_z, build_focus_map, refine_focus_map,
    apply_focus_corrections, focus_map_events,
)
from .well_plate import (
    well_layout, well_name, parse_well_name,
    well_scan_events, per_well_summary, compare_wells,
    assign_conditions,
)
from .two_pass_scan import (
    build_position_events, two_pass_mda, detect_nuclei_log,
    survey_nuclei_pass, analysis_pass,
)
from .phototaxis_steering import (
    steering_generator, make_tracking_callback, clear_slm_event,
)
