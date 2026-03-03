"""Utility modules.

Modules:
    diagnostics    -- Image saving for challenge submissions
    image          -- Grayscale conversion, normalization, contrast
    showcase       -- Multi-panel showcase figures with annotations
    experiment_log -- Structured JSON experiment logging
    report         -- Experiment report generation with statistics
"""

from .diagnostics import save_snapshot, save_overlay, save_composite
from .image import to_grayscale, normalize, auto_contrast
from .showcase import make_showcase, add_scalebar, annotate_image
from .experiment_log import ExperimentLog
from .report import (
    generate_report, format_markdown, measurement_table,
    phase_comparison, experiment_timeline,
)