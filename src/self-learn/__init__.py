# Smart Microscope Agent Toolkit
#
# hardware/   -- Microscope control, image quality, Z-stacks, drift
# detection/  -- Cell detection (BF threshold, membrane Otsu, neurons)
# analysis/   -- Intensity classification, tracking, kinetics
# workflows/  -- Adaptive acquisition, scanning, stage tracking, MDA, autofocus
# utils/      -- Diagnostics (image saving), path configuration

# Re-export commonly used functions for convenience
from .utils.diagnostics import save_snapshot, save_overlay, save_composite
from .detection.tissue import segment_tissue
from .analysis.intensity import classify_intensities