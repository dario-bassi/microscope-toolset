"""Analysis and quantification.

Modules:
    intensity    -- Multi-class intensity classification (gap, k-means, rank), SNR
    calcium      -- ROI trace extraction, ΔF/F₀, calcium transient detection
    orientation  -- Circular statistics for axial data (angles mod 180°)
    tracking     -- Cell tracking (Hungarian matching, trajectory analysis)
    kinetics     -- Exponential decay/recovery, Michaelis-Menten fitting
    temporal     -- FFT, autocorrelation, peak detection for time series
    morphometry  -- Object segmentation, size measurement, population analysis
    ring         -- Ring/annular structure analysis (organoids, ZOI)
    motion       -- Kymograph, optical flow, contraction, migration front
    classifier   -- Unsupervised (k-means) and rule-based cell classification
    spatial      -- Spatial statistics (NN distances, Ripley's K, Clark-Evans, Voronoi)
    zstack       -- Z-stack projections, cross-sections, volumetric analysis
    events       -- Time-series event detection (division, arrival, changepoint)
    colocalization -- Multi-channel overlap analysis (Pearson, Manders, Costes)
    diffusion      -- MSD analysis, diffusion fitting, motion classification
    registration   -- Timelapse drift correction via phase correlation
    texture        -- GLCM, LBP, texture maps, color deconvolution
    fluorescence   -- Background subtraction, bleach correction, ROI measurement
    statistics     -- Hypothesis testing, confidence intervals, effect sizes
    contour        -- Shape descriptors, Fourier descriptors, shape classification
    image_quality  -- Focus, noise, saturation, dynamic range assessment
    profile        -- Line profiles, path profiles, edge detection, FWHM
    optical_mapping -- Cardiac/neural wave propagation analysis (frequency, phase, CV)
    sample_characterizer -- Automated sample type detection and workflow recommendation
    measurement_validator -- Sanity checks for counts, sizes, speeds, frequencies
    histology     -- H&E tissue analysis: deconvolution, grading, necrosis
    treatment     -- Treatment response: fold change, classification, temporal analysis
    viability     -- Cell viability: live/dead counting, trypan blue, morphology
    wound_healing -- Scratch assay: wound detection, closure rate, migration speed
    confluency   -- Cell confluency, growth curves, doubling time
    size_distribution -- Particle size stats, percentiles, distribution fitting
    electrophysiology -- AP detection, firing rate, ISI, burst analysis
    plate_reader -- Plate reader data processing, background correction, outlier detection
    chemotaxis   -- Directed migration analysis, directionality, persistence, chemotactic index
    network      -- Tubular network analysis: skeletonize, junctions, branches, meshes
    cell_cycle   -- Cell cycle phase analysis: DNA content, G1/S/G2/M classification
    apoptosis    -- Cell death detection: morphological classification, temporal progression
    frap         -- FRAP analysis: recovery kinetics, mobile fraction, diffusion coefficient
    nuclear_cytoplasmic -- N:C ratio analysis, translocation, localization classification
    flow         -- Bulk flow analysis: velocity fields, profiles, vorticity, uniformity
    spectral     -- Spectral unmixing, bleedthrough correction, ratiometric imaging
    colony       -- Colony formation assay: detection, measurement, plating efficiency
    photoconversion -- Photoactivation/conversion: signal tracking, spread, half-life
    phase_segmentation -- Automatic experimental phase detection and classification
    gradient       -- Radial gradient edge detection for ZOI, wavefronts, boundaries
    confidence     -- Measurement confidence, CV, error propagation, sample sufficiency
    temporal_constraints -- Monotonicity enforcement, jump detection, constrained smoothing
    reaction_diffusion   -- Excitable-media pattern classification (waves, spirals, spots, etc.)
    aggregation          -- Collective cell migration, streaming, mound detection
    morphological_dynamics -- Track shape changes over time (spreading, heterogeneity)
    functional_connectivity -- Neural circuit mapping (correlation, cascade, graph metrics)
"""

from .aggregation import (
    aggregation_fraction,
    classify_migration_state,
    collective_order,
    detect_aggregation_centers,
    detect_mounds,
    detect_onset,
    detect_streaming,
)
from .apoptosis import (
    apoptotic_index,
    detect_apoptotic,
    morphology_score,
    temporal_death_progression,
)
from .calcium import (
    compute_dff,
    detect_calcium_transients,
    extract_roi_traces,
)
from .cell_cycle import (
    classify_cycle_phase,
    cycle_phase_summary,
    dna_content_histogram,
    mitotic_index,
    proliferation_markers,
)
from .chemotaxis import (
    analyze_migration,
    angular_histogram,
    chemotactic_index,
    directionality_index,
    migration_angles,
    persistence_time,
)
from .classifier import (
    classify_kmeans,
    classify_rules,
    classify_size_bins,
    extract_features,
    feature_importance,
)
from .colocalization import (
    colocalization_map,
    costes_threshold,
    intensity_scatter,
    manders_coefficients,
    pearson_r,
)
from .colony import (
    colony_size_classes,
    detect_colonies,
    measure_colonies,
    plating_efficiency,
    surviving_fraction,
)
from .confidence import (
    measurement_cv,
    measurement_summary,
    propagate_uncertainty,
    recount_variability,
    sufficient_samples,
)
from .confluency import (
    confluency_timecourse,
    doubling_time,
    growth_curve,
    measure_confluency,
)
from .contour import (
    classify_shape,
    contour_from_mask,
    fourier_descriptors,
    match_shape,
    shape_descriptors,
)
from .diffusion import (
    classify_motion,
    compute_msd,
    ensemble_msd,
    fit_diffusion,
)
from .electrophysiology import (
    action_potential_shape,
    burst_detection,
    detect_action_potentials,
    firing_rate,
    interspike_intervals,
)
from .events import (
    classify_trajectory,
    detect_arrival,
    detect_changepoint,
    detect_division,
    detect_morphology_change,
)
from .flow import (
    compute_flow_field,
    flow_statistics,
    flow_uniformity,
    velocity_profile,
    vorticity_map,
)
from .fluorescence import (
    bleach_correct,
    correct_illumination,
    measure_roi,
    normalize_intensity,
    subtract_background,
)
from .frap import (
    diffusion_coefficient,
    fit_frap_recovery,
    measure_frap_curve,
    mobile_fraction,
    normalize_frap,
)
from .functional_connectivity import (
    build_adjacency,
    cascade_connectivity,
    correlation_matrix,
    graph_metrics,
    lagged_correlation,
)
from .gradient import (
    annular_statistics,
    half_max_radius,
    radial_density_profile,
    steepest_gradient,
    wavefront_radius,
)
from .histology import (
    assess_glands,
    deconvolve_hae,
    detect_mitoses,
    detect_necrosis,
    grade_tumor,
    measure_pleomorphism,
    segment_nuclei_hae,
)
from .image_quality import (
    assess_quality,
    check_saturation,
    dynamic_range,
    focus_score,
    noise_estimate,
)
from .intensity import (
    classify_intensities,
    compute_snr,
    local_density,
    radial_profile,
)
from .kinetics import (
    fit_beer_lambert,
    fit_exponential_decay,
    fit_exponential_recovery,
    fit_hill,
    fit_michaelis_menten,
    fit_q10,
    fit_q10_with_ci,
    measure_bleaching_trajectory,
    measure_growth_rate_series,
    rank_by_bleach_rate,
)
from .measurement_validator import (
    run_sanity_checks,
    validate_cell_count,
    validate_concentration,
    validate_frequency,
    validate_size,
    validate_speed,
)
from .morphological_dynamics import (
    detect_shape_change,
    morphology_timecourse,
    shape_heterogeneity,
    spreading_index,
    track_morphology,
)
from .morphometry import (
    identify_outliers,
    measure_objects,
    population_stats,
    segment_nuclei,
    to_world_coords,
)
from .motion import (
    contraction_amplitude,
    kymograph,
    migration_front,
    optical_flow,
)
from .network import (
    detect_endpoints,
    detect_junctions,
    measure_network,
    segment_branches,
    skeletonize_network,
)
from .nuclear_cytoplasmic import (
    classify_localization,
    compute_nc_ratio,
    population_nc_stats,
    segment_cytoplasm,
    translocation_timecourse,
)
from .optical_mapping import (
    activation_map,
    conduction_velocity,
    detect_pacemaker,
    frequency_map,
    phase_map,
)
from .orientation import (
    circular_mean,
    circular_std,
    nematic_order_parameter,
)
from .phase_segmentation import (
    classify_phases,
    detect_steady_state,
    experiment_phases,
    phase_metrics,
    segment_phases,
)
from .photoconversion import (
    activation_efficiency,
    half_life,
    measure_activation,
    signal_spread,
    transport_rate,
)
from .plate_reader import (
    background_correct,
    detect_outliers,
    edge_correction,
    read_plate,
    well_statistics,
)
from .profile import (
    find_edges,
    line_profile,
    measure_width,
    path_profile,
)
from .quality import (
    check_bimodal,
    check_density,
    check_edge_bias,
    check_size_range,
    validate_count,
)
from .reaction_diffusion import (
    classify_rd_pattern,
    component_dynamics,
    detect_spiral_arms,
    static_features,
    temporal_features,
)
from .registration import (
    apply_shift,
    compute_drift,
    register_stack,
    register_translation,
    stabilize_stack,
)
from .ring import (
    find_best_z_plane,
    find_ring_center,
    measure_ring,
    ring_radii,
)
from .sample_characterizer import (
    characterize_image,
    characterize_sample,
    check_completeness,
    suggest_workflow,
)
from .size_distribution import (
    detect_subpopulations,
    fit_distribution,
    size_filter,
    size_percentiles,
    size_stats,
)
from .spatial import (
    clark_evans_index,
    density_map,
    detect_density_peaks,
    nearest_neighbor_distances,
    quadrat_count,
    ripleys_k,
    ripleys_l,
    voronoi_areas,
)
from .spectral import (
    autofluorescence_subtract,
    bleedthrough_correct,
    estimate_bleedthrough,
    linear_unmix,
    ratio_image,
)
from .statistics import (
    bootstrap_ci,
    compare_multiple,
    compare_two,
    confidence_interval,
    effect_size,
)
from .temporal import (
    autocorrelation,
    detect_peaks,
    detrend,
    fft_spectrum,
    measure_periodic_rate,
    measure_response_time,
    measure_wave_speed,
)
from .temporal_constraints import (
    detect_jumps,
    enforce_monotonic,
    interpolate_gaps,
    smooth_constrained,
    validate_timecourse,
)
from .texture import (
    color_deconvolution,
    entropy_map,
    glcm_features,
    local_binary_pattern,
    texture_map,
)
from .tracking import (
    cell_dispersion,
    detect_clusters,
    displacement_vectors,
    match_centroids,
    match_frames,
    population_speeds,
    smooth_trajectory,
    track_multiframe,
    trajectory_curvature,
    trajectory_speed,
)
from .treatment import (
    classify_response,
    compare_conditions,
    compute_change,
    temporal_response,
)
from .viability import (
    live_dead_count,
    morphology_viability,
    pixel_viability,
    trypan_blue_count,
    viability_index,
    viability_timecourse,
    volume_corrected_viability,
)
from .wound_healing import (
    analyze_scratch_assay,
    analyze_wound_healing,
    count_cells_in_region,
    define_wound_region,
    detect_wound,
    measure_wound_gap,
    migration_speed,
    segment_cells_bf,
    track_wound_repopulation,
    wound_closure_rate,
)
from .zstack import (
    find_focus_plane,
    measure_volume,
    project_mean,
    project_mip,
    project_std,
    slice_orthogonal,
    z_profile,
)
