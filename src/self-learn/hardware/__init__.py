"""Hardware abstraction layer.

Modules:
    config          -- runtime microscope configuration discovery
    core            -- microscope control (snap, move, objectives, autofocus)
    quality         -- image quality assessment, SNR, focus metrics
    zstack          -- Z-stack acquisition and focal-plane analysis
    drift           -- drift detection and correction (phase/FFT correlation, centroid tracking)
    slm_calibration -- SLM/DMD ↔ camera affine calibration
"""

from .config import MicroscopeConfig, clear_config_cache, get_config, refresh_config
from .core import (
    fov_size,
    get_objective,
    get_position,
    get_z,
    move_to,
    pixel_to_world,
    set_objective,
    set_z,
    snap,
    world_to_pixel,
)
from .drift import (
    centroid_drift,
    correct_drift,
    fft_cross_correlate,
    measure_drift_incremental,
    measure_drift_timelapse,
    phase_correlate,
)
from .quality import assess_image, check_focus_quality, estimate_snr, validate_acquisition
from .slm_calibration import (
    calibrate_slm,
    camera_mask_to_slm,
    find_slm_conjugate_z,
    load_calibration,
    save_calibration,
)
from .validate import (
    ValidationResult,
    validate_channels,
    validate_experiment,
    validate_objectives,
    validate_slm,
    validate_stage,
    validate_z,
)
from .zstack import acquire_zstack, detect_cells_zstack
