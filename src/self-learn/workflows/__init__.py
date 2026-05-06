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
    adaptive_survey,
    adaptive_survey_mda,
    find_clusters,
    pixel_to_world,
    rank_by_feature,
    survey_cells,
    world_to_pixel,
    zoom_and_measure,
)
from .autofocus import (
    autofocus_mda,
    check_and_correct_focus,
    coarse_fine_focus,
    drift_corrected_timelapse,
    focus_metric,
    make_focus_state,
    sweep_focus,
)
from .bacteria_trap import (
    compute_enrichment,
    count_bacteria_in_circle,
    detect_bacteria_area_threshold,
    detect_bacteria_gfp,
    measure_bacteria_intensity,
)
from .batch import (
    aggregate_results,
    identify_hotspot,
    measure_nuclear_expression,
    multi_position_measure,
    tile_and_analyze,
)
from .dose_response import (
    auto_ec50,
    dose_response_pipeline,
    make_plate_layout,
    measure_plate,
    normalize_responses,
)
from .experiment import (
    baseline_treatment,
    extract_phase_data,
    phase_timelapse,
    temperature_shift,
    wash_experiment,
)
from .mda import (
    adaptive_phase_events,
)
from .multi_scale import (
    multi_scale_measure,
    overview_first,
    scale_params,
    suggest_magnification,
    validate_object_size,
)
from .optimization import (
    make_sweep_events,
    optimize_exposure,
    optimize_gain,
    parameter_sweep,
    suggest_parameters,
)
from .optogenetics import (
    analyze_stimulation,
    connectivity_mapping,
    detect_somata,
    detect_somata_fluorescence,
    make_soma_rois,
    slm_stimulation_experiment,
)
from .organoid import (
    find_equatorial_z,
    measure_organoid,
    organoid_z_profile,
)
from .scanning import (
    deduplicate_cells,
    grid_positions,
    scan_and_detect_mda,
)
from .scouting import (
    channel_scout,
    experiment_protocol,
    sample_survey,
)
from .stage_tracking import (
    center_on_target,
    locate_target,
    make_tracker,
    predict_position,
    spiral_search,
    track_multiple,
    track_target,
    update_velocity,
)
from .tiling import (
    align_tile_pair,
    phase_correlation,
    stitch_tiles,
    stitch_tiles_aligned,
    tile_positions,
)
