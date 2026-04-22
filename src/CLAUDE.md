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

## MDA (Multi-Dimensional Acquisition) Patterns

The agent has access to Python code execution via `execute_python_code`. Use these patterns to define acquisition strategies:

### Fixed Acquisitions with MDASequence

```python
from useq import MDASequence

seq = MDASequence(
    time_plan={"loops": <NUM_FRAMES>, "interval": <INTERVAL_SEC>},
    channels=[{"config": "<CHANNEL_NAME>", "exposure": <EXPOSURE_MS>}],
    stage_positions=[{"x": <X_UM>, "y": <Y_UM>}],
)
# mmc is pre-configured in the execution namespace
results = run_mda_with_feedback(iter(seq))
```

### Adaptive/Closed-Loop with Generators

```python
from useq import MDAEvent

state = {"measurement": 0}

def on_frame(img, event, metadata):
    state["measurement"] = <analyze>(img)

def my_generator():
    for i in range(<MAX_FRAMES>):
        if state["measurement"] > <THRESHOLD>:
            return
        yield MDAEvent(channel={"config": "<CHANNEL>"})

# NOTE: In the MCP sandbox, mmc is pre-bound. Do NOT pass mmc as first arg.
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

### Fixed acquisitions → `MDASequence` + `run_mda_with_feedback`
```python
from useq import MDASequence

seq = MDASequence(
    time_plan={"loops": <NUM_FRAMES>, "interval": <INTERVAL_SEC>},
    channels=[{"config": "<CHANNEL>", "exposure": <EXPOSURE_MS>}],
    stage_positions=[{"x": <X_UM>, "y": <Y_UM>}],
)
results = run_mda_with_feedback(iter(seq))
```

### Multi-position timelapse → `MDASequence`
```python
from useq import MDASequence

positions = [{"x": <X1_UM>, "y": <Y1_UM>}, {"x": <X2_UM>, "y": <Y2_UM>}]
seq = MDASequence(
    time_plan={"loops": <NUM_FRAMES>, "interval": <INTERVAL_SEC>},
    stage_positions=positions,
    channels=[{"config": "<CHANNEL>"}],
    axis_order="tpc",  # time → position → channel
)
results = run_mda_with_feedback(iter(seq))
```

### Adaptive/closed-loop → generator with feedback
```python
from useq import MDAEvent

state = {"measurement": 0}

def on_frame(img, event, metadata):
    # Update state based on image analysis
    result = <your_analysis_function>(img)
    state["measurement"] = result

def my_generator():
    for i in range(<MAX_FRAMES>):
        if state["measurement"] > <STOP_THRESHOLD>:
            return  # Early termination condition
        yield MDAEvent(channel={"config": "<CHANNEL>"})

# NOTE: In the MCP sandbox, mmc is pre-bound. Do NOT pass mmc as first arg.
results = run_mda_with_feedback(my_generator(), on_frame=on_frame)
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
Execute Python (live mode) with run_mda_with_feedback(...):
  - Define my_generator() yielding MDAEvent objects
  - Define on_frame(img, event, metadata) callback for real-time analysis
  - In MCP sandbox, mmc is pre-bound: run_mda_with_feedback(generator, on_frame=callback)
  - Do NOT pass mmc as first arg — it causes "multiple values for on_frame" error
  - Generator reads shared state updated by on_frame
  - Terminates when condition met (e.g., cell count > threshold)
```

This is the preferred approach over manual `time.sleep()` loops — uses hardware timing.

### Pattern 4: Real Hardware Considerations

When deploying code on real hardware, account for:
- Simulation assumptions that don't hold on real hardware
- Hardware timing requirements (settling times, thermal drift, focus drift)
- Calibration steps needed before use
- Suggested parameter adjustments for your specific setup

**Always validate acquisition parameters on a small pilot dataset before full acquisition.**

## Smart Acquisition Helpers (Pre-configured)

These are available in the `execute_python_code` namespace (no import needed):

```python
# Auto-center on brightest region, iteratively refining
result = center_on_cell(mmc, pixel_size_um=0.25, threshold_sigma=2.5, max_iterations=2)
# Returns: {image, centered (bool), peak, offset_um, centroid_px}

# Find bright region centroid in an image
cy, cx, area_px, peak = find_bright_centroid(image, threshold_sigma=2.5)

# General cell/object detection
cells = detect_cells(image, threshold_sigma=2.5, min_area_px=50,
                     pixel_size_um=1.0, fill_holes=True)
# Returns: list of {centroid_px, area_um2, peak, bbox}
```

**Feedback-based MDA Execution** — For adaptive acquisition loops:

```python
from useq import MDAEvent

def on_frame(img, event, metadata):
    # Analyze image, update shared state for adaptive control
    pass

def my_generator():
    for i in range(max_frames):
        yield MDAEvent(index={'t': i}, exposure=50)

# In MCP sandbox, mmc is pre-bound — do NOT pass mmc as first arg
results = run_mda_with_feedback(my_generator(), on_frame=on_frame)
```

Use these in your code instead of reimplementing common operations.

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

```
src/                  — Computation modules for hardware, detection, analysis, workflows
tests/                — Unit tests (pytest tests/ -v with synthetic data)
scratch/              — Archived solve scripts (may reference old functions)
```

**Agent workflow:**
1. User submits request
2. Agent uses MCP tools to inspect microscope state
3. Agent writes Python code to execute via `execute_python_code`
4. Agent visualizes results in napari

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

## Common Real Hardware Issues & Solutions

### Focus Drift During Timelapse
**Problem:** Z-position drifts over time, corrupting measurements
**Solution:** Use autofocus between acquisition frames — implement an `on_frame` callback that measures focus quality and adjusts Z before yielding the next event in your generator.

### Photobleaching Corrupts Intensity Measurements
**Problem:** Signal intensity decreases over time due to fluorophore photodestruction
**Solution:** Correct each frame using a reference ROI that doesn't move. Measure the mean intensity of a stable reference region and normalize subsequent frames to the first frame's reference value.

### Spectral Bleedthrough Between Channels
**Problem:** Signal from one channel leaks into another, confounding analysis
**Solution:** Apply linear unmixing using a pre-computed bleedthrough matrix. Measure pure single-channel controls to determine bleedthrough coefficients, then solve the linear system per pixel.

### Low Contrast or Uneven Illumination
**Problem:** Threshold-based detection fails with variable image quality
**Solution:** Use adaptive thresholding and morphological preprocessing:
```python
import skimage.filters as sf
import skimage.morphology as sm

# CLAHE for contrast normalization
img_eq = sf.rank.equalize(img, footprint=sm.disk(50))
# Local threshold instead of global Otsu
thresh = sf.threshold_local(img_eq, block_size=51)
binary = img_eq > thresh
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

### During Execution
1. **Use `live` mode for hardware sequences** — Buffered mode is only for analysis
2. **Enable feedback-based termination** — Use `run_mda_with_feedback()` to avoid unnecessary exposure
3. **Monitor for drift/artifacts** — Ask agent to check images periodically with `snap_image()`

### After Acquisition
1. **Visualize in napari** — Use `viewer_add_image()`, `viewer_add_labels()` to inspect results
2. **Log metadata** — Save configuration, timing, and parameter values
3. **Version control** — Save any custom analysis scripts to git for reproducibility

### Hardware-Specific Calibration
Before using modules on real hardware, agent should:
- Determine `pixel_size_um` for your objective/camera combination
- Measure `focus_sensitivity` (Z displacement per motor step)
- Record objective turret positions and Z-offset corrections
- Verify `thermal_drift` rate if doing long timelapses
- Document `settling_times` for your stage motor
