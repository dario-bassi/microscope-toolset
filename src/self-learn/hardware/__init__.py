"""Hardware abstraction layer.

Modules:
    config   -- runtime microscope configuration discovery
    core     -- microscope control (snap, move, objectives, autofocus)
    quality  -- image quality assessment, SNR, focus metrics
    zstack   -- Z-stack acquisition and focal-plane analysis
    drift    -- drift detection and correction (phase/FFT correlation, centroid tracking)
"""

from .config import MicroscopeConfig, get_config, refresh_config, clear_config_cache
from .core import (
    snap, move_to, get_position, set_objective, get_objective,
    fov_size, set_z, get_z,
    pixel_to_world, world_to_pixel,
)
from .quality import assess_image, check_focus_quality, estimate_snr, validate_acquisition
from .zstack import acquire_zstack, detect_cells_zstack
from .drift import (
    phase_correlate, fft_cross_correlate, centroid_drift,
    measure_drift_timelapse, measure_drift_incremental, correct_drift,
)
from .validate import (
    ValidationResult, validate_channels, validate_objectives,
    validate_slm, validate_stage, validate_z, validate_experiment,
)