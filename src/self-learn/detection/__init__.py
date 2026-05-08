"""Cell and tissue detection.

Modules:
    cells    -- BF threshold detection, blob detection, morphology descriptors
    tissue   -- Membrane Otsu segmentation, contact graph
    neurons       -- Soma detection, puncta counting, neurite length, classification
    segmentation  -- Watershed, adaptive threshold, marker-controlled, erosion splitting
    threshold     -- Adaptive threshold selection, sweep, retry
    consensus     -- Multi-method detection consensus, cross-validation
"""

from .cells import (
    detect_cells, detect_cells_multichannel, find_bright_centroid,
    count_blobs_log, detect_blobs_log, count_objects_dt,
    merge_nearby_centroids, detect_point_source,
    extract_hematoxylin, detect_nuclei_hae,
    count_nuclei_fluorescence, count_nuclei_adaptive,
    count_bacteria_bf, watershed_split,
    detect_fluorescent_centroids, estimate_min_distance,
)
from .tissue import segment_tissue, contact_graph, measure_wound_closure
from .neurons import (
    detect_somata, count_puncta, estimate_neurite_length,
    count_primary_processes, classify_neuron, count_branch_points,
)
from .segmentation import (
    adaptive_threshold, segment_by_markers, separate_touching,
    expand_labels_voronoi,
)
from .threshold import (
    auto_threshold, threshold_sweep, detect_with_retry,
    estimate_noise_floor,
)
from .consensus import (
    consensus_count, merge_detections, validate_detections,
)
from .segmentation_backend import (
    Backend, LabeledMask, segment, register_backend, get_backend,
    list_backends, available_backends, labels_to_centroid_dicts,
)
