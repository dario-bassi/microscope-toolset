# Smart Microscope — Operational Guide

This guide explains how to use the Smart Microscope toolkit (`src/self-learn/`) to design and execute intelligent microscopy experiments on **real hardware**.

## Quick Start: Running an Experiment

### 1. Connect to Microscope

```python
from pymmcore_plus import CMMCorePlus

# Local microscope
core = CMMCorePlus.instance()
core.loadSystemConfiguration()

# OR: Remote microscope (via pymmcore-proxy)
from pymmcore_proxy import connect
core = connect("http://127.0.0.1:8081")  # adjust port/IP
```

### 2. Plan Your Experiment

Identify your sample type and read the relevant playbook:

```python
from pathlib import Path

# Example: FUCCI cell cycle detection
playbook_path = Path(__file__).parent / "self-learn" / "knowledge" / "playbooks" / "fucci.md"
with open(playbook_path) as f:
    print(f.read())  # Read step-by-step protocol
```

### 3. Import and Use Self-Learn Modules

```python
# Hardware & acquisition
from src.hardware.core import (
    snap, move_to, set_objective, run_events,
    pixel_to_world, world_to_pixel
)

# Detection
from src.detection.cells import detect_cells

# Analysis
from src.analysis.cell_cycle import detect_fucci_g2m, detect_fucci_division
from src.analysis.morphometry import measure_morphometry

# Workflows (high-level protocols)
from src.workflows.adaptive import adaptive_survey_mda
from src.workflows.autofocus import autofocus_mda

# Utilities
from src.utils.showcase import make_showcase
from src.utils.experiment_log import ExperimentLog
```

### 4. Execute Protocol with MDA

```python
from useq import MDASequence

# 10x overview survey
set_objective(core, 10)
move_to(core, 0, 0)

# Acquire 20 frames
seq = MDASequence(
    time_plan={"loops": 20, "interval": 2.0},
    channels=[{"config": "nucleus-channel"}],
)
results = run_events(core, list(seq))

# Analyze
for img, event in results:
    cells = detect_cells(img)
    print(f"Detected {len(cells)} cells")
```

### 5. Save Results

```python
from src.utils.showcase import make_showcase
from src.utils.experiment_log import ExperimentLog

# Log experiment for reproducibility
log = ExperimentLog(
    experiment_id="sample_001",
    description="FUCCI cell cycle monitoring"
)
log.add_phase("acquisition")
log.save("/data/experiments/sample_001.json")

# Generate showcase figure
make_showcase(
    panels=[{"image": results[0], "title": "Overview"}],
    experiment_id="sample_001",
    description="FUCCI monitoring",
    results_text=["Found 47 G2/M cells", "Division observed: True"]
)
```

## pymmcore-plus First

**Use pymmcore-plus natively. Don't wrap what the framework already provides.** If you need a device that doesn't exist yet (e.g. temperature controller, electrode array, perfusion pump), propose it to virtual-env — anything that fits the pymmcore-plus device API can be added.

### Fixed acquisitions → `MDASequence` + `run_events`
```python
from useq import MDASequence
from src.hardware.core import run_events

seq = MDASequence(
    time_plan={"loops": 10, "interval": 1.0},
    channels=[{"config": "GFP", "exposure": 50}],
    stage_positions=[{"x": 100, "y": 200}],
)
results = run_events(core, list(seq))
# run_events() delegates to core.mda.run() via frameReady signal — works for
# both local CMMCorePlus and remote pymmcore-proxy.
```

### Multi-position timelapse → `MDASequence`
```python
from useq import MDASequence
from src.hardware.core import run_events

seq = MDASequence(
    time_plan={"loops": 20, "interval": 2.0},
    stage_positions=[{"x": 100, "y": 200}, {"x": 300, "y": 400}],
    channels=[{"config": "brightfield", "group": "Fake"}],
    axis_order="tpc",  # time → position → channel
)
results = run_events(core, list(seq))
```

### Adaptive/closed-loop → generator with feedback
```python
from useq import MDAEvent
from src.hardware.core import run_events

shared = {"cell_count": 0}

def on_frame(img, event):
    cells = detect_cells(img)
    shared["cell_count"] = len(cells)

def my_generator():
    for i in range(50):
        if shared["cell_count"] > 100:  # early stop condition
            return
        yield MDAEvent(channel={"config": "BF", "group": "Fake"})

results = run_events(core, my_generator(), on_frame=on_frame)
```

## Common Experiment Patterns

### Pattern 1: Multi-Scale Survey → Zoom

```python
# Low-mag overview to identify regions
set_objective(core, 10)
overview = snap(core, "brightfield")
cells = detect_cells(overview)

# Navigate to brightest cluster
brightest = max(cells, key=lambda c: c.get("intensity", 0))
target_x = pixel_to_world(brightest["cx"], core, 10)[0]
target_y = pixel_to_world(brightest["cy"], core, 10)[1]

# Zoom and focus
set_objective(core, 40)
move_to(core, target_x, target_y)
autofocus_mda(core, channel="nucleus-channel")

# High-mag timelapse
seq = MDASequence(time_plan={"loops": 20, "interval": 1.0},
                  channels=[{"config": "geminin-channel"}])
results = run_events(core, list(seq))
```

### Pattern 2: Adaptive Acquisition with Feedback

```python
from useq import MDAEvent

shared_state = {"stop": False, "count": 0}

def on_frame(img, event):
    cells = detect_cells(img)
    shared_state["count"] = len(cells)
    if len(cells) > 100:
        shared_state["stop"] = True
        print(f"Reached target: {len(cells)} cells")

def my_generator():
    for i in range(100):
        if shared_state["stop"]:
            return
        yield MDAEvent(channel={"config": "nucleus-channel"})

run_events(core, my_generator(), on_frame=on_frame)
```

### Pattern 3: Real Hardware Considerations

**Each module in `src/self-learn/` includes ⚠️ "Real Microscope Considerations" sections** explaining:
- Simulation assumptions that don't hold on real hardware
- Hardware timing requirements (settling times, thermal drift, focus drift)
- Calibration steps needed before use
- Suggested parameter adjustments

**Always read these sections before using a module on real hardware.**

Example from `analysis/confluency.py`:
```markdown
## ⚠️ Real Microscope Considerations
Real cells have halos, shadows, and uneven illumination.
Adjustments for real hardware:
- Add morphological preprocessing (erosion/dilation)
- Use adaptive thresholding instead of global Otsu
- Validate threshold direction on actual samples first
```

## Self-Learn Modules Reference

All modules in `src/self-learn/` include detailed docstrings and **⚠️ Real Microscope Considerations** sections. Always explore functions before using them!

### `src/self-learn/analysis/` (50 modules)
Feature extraction and quantification from images.

**Cell Biology:**
- `cell_cycle.py` — FUCCI phase classification, division detection
- `calcium.py` — Calcium transient analysis, ΔF/F computation
- `apoptosis.py` — Apoptotic cell detection
- `viability.py` — Live/dead staining analysis
- `confluency.py` — Monolayer density, growth curves

**Morphology & Structure:**
- `morphometry.py` — Cell size, shape, roundness metrics
- `contour.py` — Detailed shape analysis
- `network.py` — Branching, tubular, vascular structures
- `lipid_droplet.py` — Lipid droplet segmentation

**Motion & Tracking:**
- `tracking.py` — Cell trajectory analysis
- `motion.py` — Optical flow, kymographs
- `chemotaxis.py` — Directed migration metrics
- `diffusion.py` — MSD analysis

**Signal Processing:**
- `intensity.py` — Multi-class intensity classification
- `fluorescence.py` — Quantitative fluorescence measurement
- `spectral.py` — Spectral unmixing
- `kinetics.py` — Growth rates, Q10, doubling time

**See `ARCHITECTURE.md`** for complete list of all 50 modules.

### `src/self-learn/detection/` (5 modules)
Object segmentation and localization.

- `cells.py` — General cell detection (brightfield, fluorescence)
- `neurons.py` — Soma, neurite, puncta detection
- `tissue.py` — Tissue/membrane segmentation
- `segmentation.py` — Generic watershed, adaptive threshold strategies
- `threshold.py` — Adaptive threshold selection with quality checks

### `src/self-learn/hardware/` (1 module)
Microscope-agnostic acquisition helpers.

- `core.py` — Hardware functions that work with any pymmcore-plus setup:
  - `snap()`, `move_to()`, `set_objective()`, `get_pixel_size()`
  - `run_events()` — Execute MDA sequences (fixed or adaptive)
  - `pixel_to_world()`, `world_to_pixel()` — Coordinate conversion
  - `apply_slm()`, `make_slm_circle()` — SLM control
  - `get_z()`, `set_z()` — Focus control

### `src/self-learn/workflows/` (20 modules)
High-level acquisition protocols combining hardware + analysis.

**Acquisition Strategies:**
- `adaptive.py` — Scout → cluster → zoom → measure pipeline
- `scanning.py` — Multi-position grid scanning
- `autofocus.py` — Software autofocus
- `two_pass_scan.py` — Low-mag survey + high-mag zoom
- `tiling.py` — Multi-position stitching

**Closed-Loop Control:**
- `bacteria_trap.py` — SLM bacterial light trap (baseline → illumination → measurement)
- `phototaxis_steering.py` — Real-time organism steering via SLM
- `optogenetics.py` — SLM-targeted stimulation protocols
- `stage_tracking.py` — Target tracking with motion prediction

**Specialized Protocols:**
- `temperature_experiment.py` — Q10 growth curves across temperatures
- `dose_response.py` — Multi-dose timelapse experiments
- `organoid.py` — Z-navigation for spheroids
- `experiment.py` — Multi-phase workflow builder

### `src/self-learn/utils/` (5 modules)
Diagnostics, logging, and reporting.

- `showcase.py` — Multi-panel publication figures with annotations
- `experiment_log.py` — Structured JSON logging for reproducibility
- `diagnostics.py` — Image snapshots, overlays, composites
- `report.py` — Statistical summaries and reports
- `image.py` — Preprocessing utilities

## How to Discover Functions

### 1. Browse Module Docstrings
```python
import src.self_learn.analysis.morphometry as morph
help(morph)  # View module docstring
help(morph.measure_morphometry)  # View function docstring
```

### 2. Search ARCHITECTURE.md
Use `ARCHITECTURE.md` as a quick reference for what exists and where:
```
Section "src/analysis/" → Lists all 50 modules + what they measure
Section "src/workflows/" → Lists all 20 protocols + what they do
```

### 3. Read the Real Microscope Considerations
Every module includes ⚠️ sections with:
- What assumptions don't hold on real hardware
- Hardware timing requirements
- Calibration steps needed
- Suggested parameter adjustments

**Always read these before using a module!**

## Knowledge Base

Your `knowledge/` directory survives across sessions. Use it to:

1. **Before each experiment:** Read the relevant playbook
   ```
   knowledge/playbooks/     — per-sample-type step-by-step protocols
   knowledge/strategies/    — universal microscopy principles
   knowledge/patterns/      — acquisition strategy decision trees
   knowledge/failures/      — post-mortem analysis of common mistakes
   ```

2. **Document learnings:**
   - Add insights about what works/doesn't work on your hardware
   - Note hardware-specific calibrations (focus metrics, threshold values)
   - Record troubleshooting solutions for reproducibility
   - Update playbooks if you discover improvements

**Critical:** Only document knowledge that generalizes to real microscopes. If something only works due to hardware quirks, fix the hardware setup instead. 

## Your Codebase

Two complementary resources — **code** for computation, **knowledge** for reasoning:

```
src/                  — hardware control, detection, analysis, workflows
tests/                — all tests (pytest tests/ -v), synthetic numpy data
knowledge/
  strategies/         — universal microscopy strategies (read first)
  workflows/          — category-specific workflow patterns
  pymmcore/           — pymmcore-plus API reference and MDA patterns
  playbooks/          — per-sample-type step-by-step guides (read BEFORE solving)
  prompts/            — reusable LLM vision prompts
  patterns/           — decision trees for detection/acquisition strategy
  failures/           — post-mortem analysis (read to avoid repeating mistakes)
scratch/              — solve scripts (may reference removed functions)
```

**Before each challenge:** identify sample type → read relevant playbook/source code. What can we re-use?

**You are an LLM with vision.** Not everything needs code. Save images and look at them. Use `knowledge/prompts/` for visual classification. Code handles computation; knowledge handles reasoning; vision handles understanding.

## Core Design Principles

1. **Use pymmcore-plus natively** — Don't wrap what the framework already provides
2. **All modules are microscope-agnostic** — Work with any pymmcore-plus setup
3. **World coordinates everywhere** — Use `pixel_to_world()` / `world_to_pixel()` from `src/hardware/core.py`
4. **MDA for all multi-frame acquisitions** — Use `MDASequence` for fixed protocols, generators for adaptive
5. **Stateless, composable functions** — Functions take `core` as first arg, return dicts
6. **Layers only call downward** — Workflows use analysis/detection; analysis doesn't call workflows
7. **Import from `src/`** — No inline analysis >30 lines; make it reusable

## Configuration

Set output directory for showcase figures:

```bash
export MICROSCOPE_SHOWCASE_DIR=/data/experiments/my_project/figures
```

Or specify per-call:
```python
make_showcase(panels, "exp_001", output_dir="/data/results/")
```

## Troubleshooting Real Hardware Issues

### Focus Drift During Timelapse
Use autofocus between frames:
```python
from src.workflows.autofocus import autofocus_mda
gen, on_frame, state = autofocus_mda(core, channel="nucleus-channel")
results = run_events(core, gen())
best_z = state["best_z"]
```

### Photobleaching Corrupts Measurements
Use reference ROI method:
```python
from src.analysis.fluorescence import correct_photobleaching
corrected = correct_photobleaching(target_frames, baseline_reference)
```

### Spectral Bleedthrough Confounds Analysis
Apply unmixing:
```python
from src.analysis.spectral import unmix_channels
gem_corrected, nuc_corrected = unmix_channels(
    gem_raw, nuc_raw, bleedthrough_matrix
)
```

## Rules for Real Microscopy

1. **Never skip hardware validation** — Test on known controls before real experiments
2. **Always read ⚠️ Real Microscope Considerations** in each module before use
3. **Commit reproducible scripts** — Save `experiment_*.py` scripts to git for traceability
4. **Log all experiments** — Use `ExperimentLog` for metadata and timestamps
5. **Validate visually first** — Always look at images before code decisions
6. **Measure hardware properties** — Determine pixel size, focus sensitivity, thermal drift on your setup
7. **Test detection on pilot data** — Never run full experiment with untested detection parameters
8. **Document calibrations** — Record objective turret positions, z-offset corrections, etc.
