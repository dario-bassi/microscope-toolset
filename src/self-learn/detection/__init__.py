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
    count_bacteria_bf,
    count_blobs_log,
    count_nuclei_adaptive,
    count_nuclei_fluorescence,
    count_objects_dt,
    detect_blobs_log,
    detect_cells,
    detect_cells_multichannel,
    detect_fluorescent_centroids,
    detect_nuclei_hae,
    detect_point_source,
    extract_hematoxylin,
    find_bright_centroid,
    merge_nearby_centroids,
    watershed_split,
)
from .consensus import (
    consensus_count,
    merge_detections,
    validate_detections,
)
from .neurons import (
    classify_neuron,
    count_branch_points,
    count_primary_processes,
    count_puncta,
    detect_somata,
    estimate_neurite_length,
)
from .segmentation import (
    adaptive_threshold,
    segment_by_markers,
    separate_touching,
)
from .segmentation import (
    watershed_split as watershed_split_advanced,
)
from .threshold import (
    auto_threshold,
    detect_with_retry,
    threshold_sweep,
)
from .tissue import contact_graph, measure_wound_closure, segment_tissue
