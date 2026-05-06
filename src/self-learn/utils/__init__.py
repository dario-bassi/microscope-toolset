"""Utility modules for experiment analysis and documentation.

Modules:
    diagnostics    -- Image saving and validation for analysis results
    image          -- Grayscale conversion, normalization, contrast
    showcase       -- Multi-panel showcase figures with annotations
    experiment_log -- Structured JSON experiment logging
    report         -- Experiment report generation with statistics
"""

from .diagnostics import save_composite, save_overlay, save_snapshot
from .experiment_log import ExperimentLog
from .image import auto_contrast, normalize, to_grayscale
from .report import (
    experiment_timeline,
    format_markdown,
    generate_report,
    measurement_table,
    phase_comparison,
)
from .showcase import add_scalebar, annotate_image, make_showcase
