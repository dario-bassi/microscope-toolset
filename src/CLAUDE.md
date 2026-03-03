# Smart Microscope — Operational Guide

This guide explains how to use the Smart Microscope toolkit as an LLM agent controlled via MCP tools.

**Architecture:** The workflow initializes a `CMMCorePlus` singleton and provides an agent with MCP tools to:
1. Query the microscope state and documentation
2. Execute Python code for acquisition and analysis
3. Visualize results in napari
4. Ask for user clarification when needed

## How the Agent Works

Your requests are handled by an agent that has access to:

**Knowledge Tools** (answer questions without code):
- `pymmcore_api_database` — Search pymmcore-plus API documentation
- `micromanager_device_database` — Search Micro-Manager device documentation
- `pdfs_publication_database` — Search scientific publications
- `reformulate_user_query` — Rephrase queries for better database matching

**Microscope Status Tools** (inspect current state):
- `get_microscope_settings` — Current objective, channels, device properties
- `get_microscope_events` — Recent hardware activity and status changes
- `get_last_microscope_event` — Most recent action status
- `snap_image` — Capture a single frame and display in napari

**Execution Tools** (run code on real hardware):
- `execute_python_code` — Run Python with pre-configured `mmc` instance
  - **Modes:** `buffered` (atomic/single-shot) or `live` (sequential hardware calls)
  - **Pre-configured helpers:** `run_mda_with_feedback()`, `center_on_cell()`, `detect_cells()`, `find_bright_centroid()`
  - **MDAEvent patterns:** Generators with feedback for adaptive acquisition

**Hardware Control Tools** (simple direct operations):
- `move_stage(x, y, relative=False)` — Move XY stage
- `set_objective(label)` — Switch objective lens
- `get_stage_position()` — Query current position

**Napari Viewer Tools** (visualize and interact):
- `viewer_add_image()`, `viewer_add_labels()`, `viewer_add_points()`, `viewer_add_tracks()` — Display results
- `viewer_screenshot()`, `viewer_layer_screenshot()` — Capture for visual inspection
- `view_image()` — Load and display arbitrary image files
- `viewer_set_layer_properties()`, `viewer_set_camera()`, etc. — Control visualization

## Typical Workflow

When you submit a request:

```
User Query
    ↓
Agent receives: microscope singleton + MCP tools available
    ↓
Agent chooses strategy:
    • Question about microscopy? → Use database tools
    • Need microscope state? → Use get_microscope_settings, snap_image
    • Ready to acquire/analyze? → Use execute_python_code (with execution_mode)
    • Need to visualize? → Use viewer_* tools
    ↓
Results appear in napari + console output
```

## Before Each Experiment

Read the relevant playbook for your sample type:

```
knowledge/playbooks/          — step-by-step protocols per sample type
knowledge/strategies/         — universal microscopy principles
knowledge/patterns/           — decision trees for detection/acquisition
knowledge/failures/           — post-mortem analysis of common mistakes
```

Use `knowledge/prompts/` for visual classification if you have images.

## MDA (Multi-Dimensional Acquisition) Patterns

The agent has access to Python code execution via `execute_python_code`. Use these patterns to define acquisition strategies:

### Fixed Acquisitions with MDASequence

```python
from useq import MDASequence
from src.hardware.core import run_events

seq = MDASequence(
    time_plan={"loops": <NUM_FRAMES>, "interval": <INTERVAL_SEC>},
    channels=[{"config": "<CHANNEL_NAME>", "exposure": <EXPOSURE_MS>}],
    stage_positions=[{"x": <X_UM>, "y": <Y_UM>}],
)
results = run_events(mmc, list(seq))
```

### Adaptive/Closed-Loop with Generators

```python
from useq import MDAEvent

state = {"measurement": 0}

def on_frame(img, event):
    state["measurement"] = <analyze>(img)

def my_generator():
    for i in range(<MAX_FRAMES>):
        if state["measurement"] > <THRESHOLD>:
            return
        yield MDAEvent(channel={"config": "<CHANNEL>"})

results = run_mda_with_feedback(my_generator(), on_frame=on_frame)
```

## Code Execution Modes

When using `execute_python_code`:

**`buffered` mode:**
- Hardware calls are collected and replayed atomically
- All `snapImage()` calls produce the same image (captured at the same moment)
- Use for: single-shot analysis, non-hardware code
- ⚠️ **Do NOT use for:** move-then-snap, timelapse, tracking, multi-position workflows

**`live` mode:**
- Each hardware call executes immediately in sequence
- Stage position changes happen before the next snap
- Required for: timelapse, tracking, adaptive acquisition, `run_mda_with_feedback()`
- Slower but necessary for dependent operations

**Rule:** If your code has `move() then snap()` or any loop with hardware calls, use `live` mode.

### Fixed acquisitions → `MDASequence` + `run_events`
```python
from useq import MDASequence
from src.hardware.core import run_events

seq = MDASequence(
    time_plan={"loops": <NUM_FRAMES>, "interval": <INTERVAL_SEC>},
    channels=[{"config": "<CHANNEL>", "exposure": <EXPOSURE_MS>}],
    stage_positions=[{"x": <X_UM>, "y": <Y_UM>}],
)
results = run_events(core, list(seq))
# run_events() delegates to core.mda.run() via frameReady signal
# Works for both local CMMCorePlus and remote pymmcore-proxy
```

### Multi-position timelapse → `MDASequence`
```python
from useq import MDASequence
from src.hardware.core import run_events

positions = [{"x": <X1_UM>, "y": <Y1_UM>}, {"x": <X2_UM>, "y": <Y2_UM>}]
seq = MDASequence(
    time_plan={"loops": <NUM_FRAMES>, "interval": <INTERVAL_SEC>},
    stage_positions=positions,
    channels=[{"config": "<CHANNEL>"}],
    axis_order="tpc",  # time → position → channel
)
results = run_events(core, list(seq))
```

### Adaptive/closed-loop → generator with feedback
```python
from useq import MDAEvent
from src.hardware.core import run_events

state = {"measurement": 0}

def on_frame(img, event):
    # Update state based on image analysis
    result = <your_analysis_function>(img)
    state["measurement"] = result

def my_generator():
    for i in range(<MAX_FRAMES>):
        if state["measurement"] > <STOP_THRESHOLD>:
            return  # Early termination condition
        yield MDAEvent(channel={"config": "<CHANNEL>"})

results = run_events(core, my_generator(), on_frame=on_frame)
```

## Common Agent Workflow Patterns

### Pattern 1: Inspect State → Decide Action

```
Agent workflow:
1. get_microscope_settings() — Check objectives, channels, current config
2. snap_image() — Capture frame to examine
3. viewer_screenshot() — Show result
4. Based on visual inspection, decide next action
```

### Pattern 2: Multi-Scale Survey → Zoom → Measure

```python
# Low-mag overview
Execute: set current objective to 10x, snap_image()
Visualize with viewer_screenshot()

# Analyze overview
Execute Python code (buffered mode):
  - Load snapshot, detect features
  - Identify region of interest (brightest, largest, etc.)
  - Compute target stage position from pixel location

# Zoom and focus
Execute: set objective to 40x, move_stage to target position
Execute: run_mda_with_feedback() for autofocus on high-contrast channel

# High-mag measurement
Execute Python (live mode) with MDASequence for timelapse
Visualize results in napari with viewer_add_image()
```

### Pattern 3: Adaptive Closed-Loop Acquisition

```python
Execute Python (live mode) with run_mda_with_feedback():
  - Define my_generator() yielding MDAEvent objects
  - Define on_frame(img, event) callback for real-time analysis
  - Generator reads shared state updated by on_frame
  - Terminates when condition met (e.g., cell count > threshold)
```

This is the preferred approach over manual `time.sleep()` loops — uses hardware timing.

### Pattern 4: Real Hardware Considerations

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

## Available Modules in Code Execution

When using `execute_python_code`, you have access to:

### Analysis Modules (`src/analysis/`)
Feature extraction and quantification. See `ARCHITECTURE.md` for complete list.

Import in your code:
```python
from src.analysis.<module> import <function>

# Examples:
from src.analysis.morphometry import measure_morphometry
from src.analysis.tracking import track_cells
from src.analysis.intensity import classify_intensity
```

Categories available:
- Morphological analysis (size, shape, contours)
- Intensity-based measurements (fluorescence, classification)
- Motion analysis (tracking, flow, migration)
- Signal processing (spectral unmixing, kinetics)

### Detection Modules (`src/detection/`)
Object segmentation and localization.

```python
from src.detection.cells import detect_cells
from src.detection.threshold import adaptive_threshold
from src.detection.segmentation import watershed_segment
```

### Hardware Module (`src/hardware/core.py`)
Pre-configured via `mmc` instance (already available):
```python
from src.hardware.core import snap, move_to, set_objective, run_events
from src.hardware.core import pixel_to_world, world_to_pixel
from src.hardware.core import get_pixel_size, get_z, set_z
```

### Workflow Modules (`src/workflows/`)
High-level acquisition protocols.

```python
from src.workflows.autofocus import autofocus_mda
from src.workflows.adaptive import adaptive_survey_mda
# See ARCHITECTURE.md for full list
```

### Utility Modules (`src/utils/`)
Logging and visualization.

```python
from src.utils.showcase import make_showcase
from src.utils.experiment_log import ExperimentLog
from src.utils.diagnostics import save_snapshot
```

**All modules include ⚠️ "Real Microscope Considerations" sections.** Review before using on real hardware.

## Discovering Available Functions

The agent can discover and use functions in three ways:

### 1. Direct Code Execution
Ask the agent to explore via `execute_python_code`:
```python
# Agent executes this code:
import src.analysis.morphometry as morph
help(morph)  # Prints module docstring with available functions
```

### 2. Search ARCHITECTURE.md
Use this as a reference for:
- All available analysis modules and what they measure
- All available detection methods
- All available workflows and their purposes

### 3. Look at Real Microscope Considerations
Before asking the agent to use a function on real hardware, they should review the module's ⚠️ sections which document:
- Hardware assumptions and limitations
- Calibration steps required
- Timing requirements
- Parameter adjustments for real data

## Smart Acquisition Helpers (Pre-configured)

These are available in the `execute_python_code` namespace (no import needed):

```python
# Auto-center on brightest region, iteratively refining
result = center_on_cell(pixel_size_um=0.25, threshold_sigma=2.5, max_iterations=2)
# Returns: {image, centered (bool), peak, offset_um, centroid_px}

# Find bright region centroid in an image
cy, cx, area_px, peak = find_bright_centroid(image, threshold_sigma=2.5)

# General cell/object detection
cells = detect_cells(image, threshold_sigma=2.5, min_area_px=50,
                     pixel_size_um=1.0, fill_holes=True)
# Returns: list of {centroid_px, area_um2, peak, bbox}

# Feedback-based MDA execution
results = run_mda_with_feedback(events, on_frame=callback_func)
```

Use these in your code instead of reimplementing common operations.

## Knowledge Base

The `knowledge/` directory survives across sessions. Use it to:

1. **Before each experiment:**
   - Read the relevant playbook for your sample type
   - Review universal strategies in `knowledge/strategies/`
   - Check `knowledge/failures/` for common pitfalls to avoid

2. **Document learnings after each experiment:**
   - What parameters worked well on your hardware?
   - What thresholds did you need to adjust?
   - Did you discover a new detection strategy?
   - What caused failures, and how did you fix them?
   - Update playbooks if you discover improvements

**Critical:** Only document knowledge that generalizes to real microscopes. If something only works due to hardware quirks, document the workaround but also consider fixing the hardware setup.

## MCP Tools Quick Reference

| Tool | Purpose | When to Use |
|------|---------|-----------|
| `get_microscope_settings()` | Inspect current config | Before each experiment |
| `snap_image()` | Capture single frame | Visual inspection |
| `execute_python_code(code, mode='buffered')` | Run analysis | Pure image processing |
| `execute_python_code(code, mode='live')` | Hardware operations | Timelapse, tracking, adaptive |
| `move_stage(x, y)` | Navigate stage | Simple positioning |
| `set_objective(label)` | Change magnification | Before acquisition |
| `viewer_add_image(path)` | Display results | Show timelapse stacks |
| `viewer_screenshot()` | Capture napari view | Document results |
| `pymmcore_api_database(query)` | Search API docs | "How do I...?" questions |
| `pdfs_publication_database(query)` | Search papers | Scientific background |
| `request_user_clarification(message)` | Ask user | Need human input | 

## Codebase Structure

**Code + Knowledge working together:**

```
src/                  — Computation modules for hardware, detection, analysis, workflows
knowledge/
  playbooks/          — Step-by-step protocols per sample type (START HERE)
  strategies/         — Universal microscopy principles (read first)
  patterns/           — Decision trees for detection/acquisition strategy
  failures/           — Post-mortem analysis (learn from mistakes)
  pymmcore/           — pymmcore-plus API reference and patterns
  prompts/            — LLM vision prompts for image classification
tests/                — Unit tests (pytest tests/ -v with synthetic data)
scratch/              — Archived solve scripts (may reference old functions)
```

**Agent workflow:**
1. User submits request
2. Agent reads relevant `knowledge/playbooks/` for sample type
3. Agent checks `ARCHITECTURE.md` for available modules
4. Agent uses MCP tools to inspect microscope state
5. Agent writes Python code to execute via `execute_python_code`
6. Agent visualizes results in napari
7. Agent learns and updates knowledge base

## Core Design Principles

1. **pymmcore-plus is the foundation** — All hardware control flows through `mmc` (CMMCorePlus singleton)
2. **Microscope-agnostic modules** — All functions work with any pymmcore-plus setup
3. **World coordinates** — Use `pixel_to_world()` / `world_to_pixel()` for stage navigation
4. **MDA for all acquisitions** — `MDASequence` for fixed protocols, generators for adaptive
5. **Functions are stateless** — Take data in, return results as dicts
6. **Layered architecture** — Workflows use detection/analysis; analysis doesn't call workflows
7. **Reusable, importable code** — No inline analysis >30 lines; put it in `src/`
8. **MCP tools are the interface** — Agent interacts with hardware exclusively through defined tools

## Configuration

The agent has access to:

- **Microscope instance (`mmc`)** — Already initialized CMMCorePlus singleton
- **Napari viewer** — napari-micromanager for live visualization
- **File system** — Save/load data with absolute paths

Set environment variables before running (or ask agent to set them in code):

```bash
export MICROSCOPE_SHOWCASE_DIR=/data/experiments/my_project/figures
```

Agent can specify output paths directly in code:
```python
from src.utils.showcase import make_showcase

make_showcase(
    panels=[...],
    experiment_id="exp_001",
    output_dir="/data/results/"
)
```

## Common Real Hardware Issues & Solutions

### Focus Drift During Timelapse
**Problem:** Z-position drifts over time, corrupting measurements
**Solution:** Use autofocus between acquisition frames
```python
from src.workflows.autofocus import autofocus_mda
# Agent asks you to enable autofocus, runs this code in live mode
gen, on_frame, state = autofocus_mda(mmc, channel="<FOCUS_CHANNEL>")
results = run_mda_with_feedback(gen(), on_frame=on_frame)
best_z = state["best_z"]
```

### Photobleaching Corrupts Intensity Measurements
**Problem:** Signal intensity decreases over time due to fluorophore photodestruction
**Solution:** Correct each frame using a reference ROI that doesn't move
```python
from src.analysis.fluorescence import correct_photobleaching
corrected = correct_photobleaching(target_frames, baseline_reference_frames)
```

### Spectral Bleedthrough Between Channels
**Problem:** Signal from one channel leaks into another, confounding analysis
**Solution:** Apply unmixing using a pre-computed bleedthrough matrix
```python
from src.analysis.spectral import unmix_channels
ch1_corrected, ch2_corrected = unmix_channels(
    ch1_raw, ch2_raw, bleedthrough_matrix
)
```

### Low Contrast or Uneven Illumination
**Problem:** Threshold-based detection fails with variable image quality
**Solution:** Use adaptive thresholding and morphological preprocessing
```python
from src.detection.threshold import adaptive_threshold
from src.analysis.image import preprocess

img_prep = preprocess(img, method="clahe")  # Contrast-limited histogram equalization
binary = adaptive_threshold(img_prep, sigma=2.5)
```

## When to Ask for Help

If the agent encounters issues:
1. **Visual inspection** — Ask agent to use `viewer_screenshot()` or `view_image()` to show you the data
2. **Hardware validation** — Ask agent to run `get_microscope_settings()` and compare to expected configuration
3. **Parameter tuning** — Ask agent to test parameters on a small pilot dataset before full acquisition
4. **Documentation** — Agent can search `pymmcore_api_database` or `pdfs_publication_database` for solutions

## Best Practices for Real Microscopy Experiments

### Before Each Experiment
1. **Ask for microscope state** — Agent should use `get_microscope_settings()` to confirm configuration
2. **Inspect pilot data** — Use `snap_image()` and `viewer_screenshot()` to visualize sample
3. **Test detection parameters** — Run analysis on small dataset before full acquisition
4. **Read relevant playbook** — Check `knowledge/playbooks/` for your sample type

### During Execution
1. **Use `live` mode for hardware sequences** — Buffered mode is only for analysis
2. **Enable feedback-based termination** — Use `run_mda_with_feedback()` to avoid unnecessary exposure
3. **Save intermediate results** — Use `ExperimentLog` to track progress
4. **Monitor for drift/artifacts** — Ask agent to check images periodically with `snap_image()`

### After Acquisition
1. **Visualize in napari** — Use `viewer_add_image()`, `viewer_add_labels()` to inspect results
2. **Log metadata** — Save configuration, timing, and parameter values
3. **Document learnings** — Update `knowledge/playbooks/` if you discover improvements
4. **Version control** — Save any custom analysis scripts to git for reproducibility

### Hardware-Specific Calibration
Before using modules on real hardware, agent should:
- Determine `pixel_size_um` for your objective/camera combination
- Measure `focus_sensitivity` (Z displacement per motor step)
- Record objective turret positions and Z-offset corrections
- Verify `thermal_drift` rate if doing long timelapses
- Document `settling_times` for your stage motor
