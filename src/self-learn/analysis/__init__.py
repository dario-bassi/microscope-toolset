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

from .intensity import (
    classify_intensities, compute_snr, radial_profile, local_density,
)
from .calcium import (
    extract_roi_traces, compute_dff, detect_calcium_transients,
)
from .orientation import (
    circular_mean, circular_std, nematic_order_parameter,
)
from .temporal import (
    fft_spectrum, autocorrelation, detect_peaks, detrend,
    measure_periodic_rate, measure_wave_speed, measure_response_time,
)
from .tracking import (
    match_frames, track_multiframe, displacement_vectors,
    smooth_trajectory, trajectory_speed, trajectory_curvature,
    cell_dispersion, detect_clusters,
    match_centroids, population_speeds,
)
from .kinetics import (
    fit_exponential_decay, fit_exponential_recovery,
    fit_michaelis_menten, fit_beer_lambert, fit_hill, fit_q10, fit_q10_with_ci, measure_growth_rate_series,
    measure_bleaching_trajectory, rank_by_bleach_rate,
)
from .morphometry import (
    segment_nuclei, measure_objects, population_stats,
    identify_outliers, to_world_coords,
)
from .ring import (
    ring_radii, measure_ring, find_ring_center, find_best_z_plane,
)
from .motion import (
    kymograph, optical_flow, contraction_amplitude, migration_front,
)
from .classifier import (
    extract_features, classify_kmeans, classify_rules,
    classify_size_bins, feature_importance,
)
from .spatial import (
    nearest_neighbor_distances, clark_evans_index, ripleys_k,
    ripleys_l, voronoi_areas, quadrat_count,
    density_map, detect_density_peaks,
)
from .zstack import (
    project_mip, project_mean, project_std, slice_orthogonal,
    measure_volume, find_focus_plane, z_profile,
)
from .events import (
    detect_division, detect_arrival, detect_morphology_change,
    detect_changepoint, classify_trajectory,
)
from .quality import (
    check_density, check_size_range, check_edge_bias,
    check_bimodal, validate_count,
)
from .colocalization import (
    pearson_r, manders_coefficients, costes_threshold,
    intensity_scatter, colocalization_map,
)
from .diffusion import (
    compute_msd, fit_diffusion, classify_motion, ensemble_msd,
)
from .registration import (
    register_translation, register_stack, apply_shift,
    compute_drift, stabilize_stack,
)
from .texture import (
    glcm_features, local_binary_pattern, texture_map,
    entropy_map, color_deconvolution,
)
from .fluorescence import (
    subtract_background, correct_illumination, measure_roi,
    bleach_correct, normalize_intensity,
)
from .statistics import (
    compare_two, compare_multiple, confidence_interval,
    effect_size, bootstrap_ci,
)
from .contour import (
    shape_descriptors, fourier_descriptors, match_shape,
    contour_from_mask, classify_shape,
)
from .image_quality import (
    assess_quality, focus_score, noise_estimate,
    check_saturation, dynamic_range,
)
from .profile import (
    line_profile, path_profile, find_edges, measure_width,
)
from .optical_mapping import (
    frequency_map, phase_map, conduction_velocity,
    detect_pacemaker, activation_map,
)
from .sample_characterizer import (
    characterize_image, characterize_sample,
    suggest_workflow, check_completeness,
)
from .measurement_validator import (
    validate_cell_count, validate_size, validate_speed,
    validate_frequency, validate_concentration, run_sanity_checks,
)
from .histology import (
    deconvolve_hae, segment_nuclei_hae, detect_necrosis,
    assess_glands, measure_pleomorphism, detect_mitoses, grade_tumor,
)
from .treatment import (
    compute_change, classify_response, compare_conditions,
    temporal_response,
)
from .viability import (
    live_dead_count, viability_index, trypan_blue_count,
    morphology_viability, viability_timecourse,
    pixel_viability, volume_corrected_viability,
)
from .wound_healing import (
    detect_wound, measure_wound_gap, wound_closure_rate,
    migration_speed, analyze_scratch_assay,
    segment_cells_bf, define_wound_region, count_cells_in_region,
    track_wound_repopulation, analyze_wound_healing,
)
from .confluency import (
    measure_confluency, confluency_timecourse,
    growth_curve, doubling_time,
)
from .size_distribution import (
    size_stats, size_percentiles, fit_distribution,
    detect_subpopulations, size_filter,
)
from .electrophysiology import (
    detect_action_potentials, firing_rate, interspike_intervals,
    action_potential_shape, burst_detection,
)
from .plate_reader import (
    read_plate, background_correct, detect_outliers,
    edge_correction, well_statistics,
)
from .chemotaxis import (
    directionality_index, migration_angles, angular_histogram,
    persistence_time, chemotactic_index, analyze_migration,
)
from .network import (
    skeletonize_network, detect_junctions, detect_endpoints,
    measure_network, segment_branches,
)
from .cell_cycle import (
    dna_content_histogram, classify_cycle_phase, mitotic_index,
    proliferation_markers, cycle_phase_summary,
)
from .apoptosis import (
    detect_apoptotic, apoptotic_index, morphology_score,
    temporal_death_progression,
)
from .frap import (
    measure_frap_curve, normalize_frap, fit_frap_recovery,
    mobile_fraction, diffusion_coefficient,
)
from .nuclear_cytoplasmic import (
    segment_cytoplasm, compute_nc_ratio, classify_localization,
    translocation_timecourse, population_nc_stats,
)
from .flow import (
    compute_flow_field, flow_statistics, velocity_profile,
    vorticity_map, flow_uniformity,
)
from .spectral import (
    bleedthrough_correct, linear_unmix, estimate_bleedthrough,
    ratio_image, autofluorescence_subtract,
)
from .colony import (
    detect_colonies, measure_colonies, plating_efficiency,
    surviving_fraction, colony_size_classes,
)
from .photoconversion import (
    measure_activation, signal_spread, half_life,
    transport_rate, activation_efficiency,
)
from .phase_segmentation import (
    segment_phases, classify_phases, detect_steady_state,
    phase_metrics, experiment_phases,
)
from .gradient import (
    radial_density_profile, steepest_gradient, half_max_radius,
    wavefront_radius, annular_statistics,
)
from .confidence import (
    measurement_cv, recount_variability, sufficient_samples,
    propagate_uncertainty, measurement_summary,
)
from .temporal_constraints import (
    enforce_monotonic, detect_jumps, smooth_constrained,
    validate_timecourse, interpolate_gaps,
)
from .reaction_diffusion import (
    classify_rd_pattern, static_features, temporal_features,
    component_dynamics, detect_spiral_arms,
)
from .aggregation import (
    detect_aggregation_centers, detect_streaming,
    aggregation_fraction, classify_migration_state, collective_order,
    detect_mounds, detect_onset,
)
from .morphological_dynamics import (
    track_morphology, detect_shape_change, morphology_timecourse,
    spreading_index, shape_heterogeneity,
)
from .functional_connectivity import (
    correlation_matrix, lagged_correlation, build_adjacency,
    cascade_connectivity, graph_metrics,
)
