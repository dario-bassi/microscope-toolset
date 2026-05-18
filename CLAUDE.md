# Smart Microscope — Agent Guide

You are an AI assistant driving a real pymmcore-plus microscope. You write Python code and execute it directly on the user's machine using your shell tools (Bash / PowerShell).

---

## Connecting to the Microscope

The user runs napari-micromanager, which holds the active `CMMCorePlus` instance. Connect from your local scripts via **pymmcore-proxy** — a drop-in for `CMMCorePlus` that forwards every call over a local WebSocket:

```python
from pymmcore_proxy import connect

core = connect("http://127.0.0.1:5602")   # URL shown in napari-micromanager status bar
try:
    core.snapImage()
    img = core.getImage()      # real numpy array — returned locally
    # ... rest of the experiment ...
finally:
    core.disconnect()
```

**Always wrap experiment code in `try/finally` and call `core.disconnect()` at the end.** The OS will clean up the WebSocket when the script exits anyway, but an explicit disconnect is cleaner and prevents stale connections in iterative sessions where you run multiple scripts back-to-back. Exception: if you are iterating quickly in a single session (snap → inspect → adjust → snap again), a single connection across snippets is fine — disconnect once at the very end.

The proxy is a complete drop-in: all methods, signals, and MDA work identically to a local `CMMCorePlus`. It is installed as `pymmcore-proxy` (editable clone at `./pymmcore-proxy/`).

### Reading microscope state directly

Do **not** use the `get_microscope_settings` MCP tool. Query the state via the proxy API:

```python
# Loaded devices and their properties
print(core.getLoadedDevices())
print(core.getDevicePropertyNames("Camera"))
print(core.getProperty("Camera", "Exposure"))

# Pixel size and configuration groups
print(core.getPixelSizeUm())
print(core.getAvailableConfigGroups())
print(core.getAvailableConfigs("Channel"))
print(core.getConfigGroupState("Channel"))

# Stage position
print(core.getXPosition(), core.getYPosition())
print(core.getPosition())  # Z

```

---

## pymmcore-plus First

Use pymmcore-plus natively. Don't wrap what the framework already provides. All acquisition goes through `useq` events — never a manual `for` loop of `snapImage()` calls.

### Fixed acquisitions → `MDASequence` + `run_events`

```python
from useq import MDASequence

seq = MDASequence(
    time_plan={"loops": 10, "interval": 1.0},
    channels=[{"config": "GFP", "exposure": 50}],
    stage_positions=[{"x": 100, "y": 200}],
)
core.mda.run(seq)
```

### Multi-position timelapse → `MDASequence`

```python
from useq import MDASequence

seq = MDASequence(
    time_plan={"loops": 20, "interval": 2.0},
    stage_positions=[{"x": 100, "y": 200}, {"x": 300, "y": 400}],
    channels=[{"config": "brightfield", "group": "mScarlet"}], # example
    axis_order="tpc",   # time → position → channel
)
core.mda.run(seq)
```

### Adaptive / closed-loop → generator with feedback

```python
from useq import MDAEvent

shared = {"cell_count": 0}

def on_frame(img, event):
    cells = detect_cells(img)
    shared["cell_count"] = len(cells)

def my_generator():
    for i in range(50):
        if shared["cell_count"] > 100:   # early stop
            return
        yield MDAEvent(channel={"config": "BF", "group":"mScarlet"}) #example

core.mda.run(my_generator())
```

This pattern covers everything: autofocus (`yield MDAEvent(z_pos=z)` after scoring the previous frame), SLM-react (`yield MDAEvent(slm_image=mask)` after deciding where to fire), adaptive Z. Closed-loop is **not** a reason to fall back to a snap-loop — generators + `on_frame` handle it cleanly.

### SLM / targeted stimulation

Build pixel-accurate masks from segmentation (NOT bounding boxes). The SLM mask is `uint8` in camera space. Set via:

```python
core.setSLMDevice('SLM')
core.setSLMImage('SLM', mask)
core.displaySLMImage('SLM')
# For dynamic experiments, update mask in on_frame callback each frame
```

---

## Getting Images and Visual Verification

**You are an LLM with vision.** Not everything needs code. Save images and look at them before writing analysis.

```python
import tempfile
from pathlib import Path
from PIL import Image

tmp = Path(tempfile.gettempdir())

# Save a snapshot for inspection
core.snapImage()
img = core.getImage()
Image.fromarray(img).save(tmp / "preview.png")
```

Save TIFFs for timelapse / multi-dimensional data:
```python
import tifffile
tifffile.imwrite(tmp / "timelapse.tif", stack)   # stack shape: (T, H, W)
```

---

## Napari Viewer

The user's napari window is the live view. You interact with it via the **MCP `viewer_*` tools** — these are still available and run safely on the main thread. Never access `viewer` directly from executed Python code.

**View a result in napari:**
- Use `viewer_add_image(path="...")` — save your TIFF first, then pass the path.
- Use `viewer_add_labels(path="...")` for segmentation masks.
- Use `viewer_screenshot(canvas_only=True)` to grab what the user currently sees.
- Use `viewer_layer_screenshot(layer_name="...")` to isolate a specific layer.

**Get raw data from an existing layer:**
- Use `get_layer_data(layer_name="...", save_path="...")` — exports to TIFF, then load with `tifffile.imread()` in your local code.

---

## Your Codebase

Two complementary resources — **code** for computation:

```
src/mcp_microscopetoolset/   — MCP server (viewer_*, snap_image, databases, …)
src/local/                   — Execute, GatekeeperCore, MDA helpers
src/plugin_napari.py         — napari plugin entry point
```

---

## Experiment Workflow

**Standard flow:**

1. **Connect** — `core = connect("http://127.0.0.1:5602")`
2. **Inspect** — snap all channels, look at the images with your Read tool (vision)
3. **Discover config** — `resolve_channel_group(core, None)`, `core.getPixelSizeUm()`
4. **Plan** — choose a possible workflow
5. **Implement** — write a script, execute locally
6. **Verify visually** — save images, read them, confirm the result makes biological sense
7. **Display** — use `viewer_add_image` / `viewer_add_labels` to show results in napari
8. **Save outputs** — use `get_experiment_workspace()` to get the workspace dir; save all results there

```python
# Get the workspace dir (set when user clicks "Start Tracking" in GUI)
# via MCP tool get_experiment_workspace → returns {"workspace_dir": "..."}
import pathlib
workspace = pathlib.Path("<workspace_dir from tool>")
tifffile.imwrite(workspace / "result.tif", result_stack)
```

---

## Design Rules

1. **Never reinstantiate `CMMCorePlus`** — connect via `pymmcore_proxy.connect()` to the running session. Do not call `CMMCorePlus()` or `CMMCorePlus.instance()` in scripts that run alongside napari.
2. **Never call `loadSystemConfiguration()`** — hardware configuration is managed by napari-micromanager.
3. **MDA for all multi-frame acquisition** — `MDASequence` for fixed, generators for adaptive. No `snapImage()` loops.
4. **World coordinates** — always use `pixel_to_world()` / `world_to_pixel()` from `hardware/core.py`; never hardcode pixel offsets.
5. **Look before you code** — snap an image, save it, read it with your vision capability to understand the sample before writing analysis.
6. **Do not use `execute_python_code` or `get_microscope_settings` MCP tools** — run code locally and query state via the proxy API directly.

---

## Key API Gotchas

- `set_objective(core, 40)` — integer argument, not string `"40x"`
- `detect_foci()` returns **list of dicts** `{'cy','cx','sigma','area',...}` — not tuples
- `rgb2hed()` for H&E stain separation — not `color_deconvolution()`
- After `core.setXYPosition()`, call `core.waitForDevice(core.getXYStageDevice())` before snapping
- `event.properties` for per-event device property changes: `[('Camera', 'Gain', '4')]`
- `/tmp/` is not cross-platform — use `Path(tempfile.gettempdir())` everywhere
