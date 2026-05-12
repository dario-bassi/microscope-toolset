---
title: Automatic real-time three-dimensional cell tracking by fluorescence microscopy
authors: G Rabut, J Ellenberg
year: 2004
venue: Journal of Microscopy 216(Pt 2):131-137
doi: 10.1111/j.0022-2720.2004.01404.x
url: https://pubmed.ncbi.nlm.nih.gov/15516224/
researched: 2026-04-24
---

## Abstract

Live cell imaging has become an indispensable technique for cell biologists. However, when grown on coverslip glass used for live cell imaging many cultured cells move even during relatively short observation times and focus can drift as a result of mechanical instabilities and/or temperature fluctuations. Time-lapse imaging therefore requires constant adjustment of the imaging field and focus position to keep the cell of interest centred in the imaged volume. We show here that this limitation can be overcome by tracking cells in a fully automated way using the mass centre of cellular fluorescence. Combined with automated multiple location revisiting, this method dramatically increases the throughput of high-resolution live cell imaging experiments.

## Smart microscopy principle

Rabut and Ellenberg describe the canonical single-cell target-tracking microscope: the field of view is a closed-loop setpoint controlled by the cell, not by the stage coordinate set at t=0. At each time point the microscope snaps a fluorescence image, computes the intensity-weighted mass centre of the labelled cell inside the current FOV, converts the pixel offset between that centroid and the image centre into a world-space XY (and Z from a short focus sweep) correction, and updates the stage before the next acquisition. Multiple cells at different stage positions are time-shared: the loop runs per location in a revisit schedule, so one microscope follows many independently drifting cells for hours without manual intervention.

The transferable contribution is **target-driven acquisition** — the reference frame of the experiment is attached to the biology, not to the instrument. This predates the light-sheet specimen-tracking work of McDole 2018 by more than a decade and is the general pattern every long live-cell time-lapse on motile or drifting samples needs: cell translation (mitotic rounding, migration), stage thermal drift, and coverslip relaxation all look the same to the control loop — an offset to be measured from the last image and subtracted from the next setpoint. The innovation compared with manual re-centring is that the *criterion* (keep the cell centroid on-axis) and the *authority* (update the stage) close within a single acquisition-analysis cycle.

## Implementation on pymmcore-plus

The loop maps cleanly onto `run_events` + an `on_frame` callback that writes the next stage setpoint. The core primitive is already implemented in `src/core/workflows/stage_tracking.py` as `track_target` / `track_target_mda` — this paper is that module's conceptual parent. The snippet below shows the minimum closed-loop version for a single cell; multi-cell revisiting is a loop over `stage_positions` with per-position state.

```python
import numpy as np
from useq import MDAEvent
from src.core.hardware.core import run_events
from src.core.detection.cells import find_bright_centroid

INTERVAL_S = 60.0           # time between revisits per cell
N_FRAMES = 180              # 3 h at 1 min / frame
MAX_OFFSET_UM = 30.0        # safety clamp on per-step correction

def track_single_cell(core, pixel_size_um, channel="GFP"):
    """Keep a fluorescent cell at the FOV centre across a long timelapse.

    Each frame:
      1. snap fluorescence image
      2. find the bright centroid (mass centre of labelled cell)
      3. convert pixel offset from FOV centre -> world-space dx, dy
      4. update the stage setpoint for the next frame
    The next MDAEvent reads the updated setpoint via shared state.
    """
    h, w = core.getImageHeight(), core.getImageWidth()
    fov_cx_px, fov_cy_px = w / 2, h / 2
    state = {
        "x_um": core.getXPosition(),
        "y_um": core.getYPosition(),
        "lost": 0,
    }

    def on_frame(img, event, meta=None):
        result = find_bright_centroid(img, threshold_sigma=2.5, window=64)
        if result is None:
            state["lost"] += 1
            return
        cy, cx, _area, _peak = result
        # Pixel -> world offset. y axis sign depends on camera orientation.
        dx_um = (cx - fov_cx_px) * pixel_size_um
        dy_um = (cy - fov_cy_px) * pixel_size_um
        # Clamp: reject large jumps (likely a neighbour, not the tracked cell)
        if abs(dx_um) > MAX_OFFSET_UM or abs(dy_um) > MAX_OFFSET_UM:
            state["lost"] += 1
            return
        state["x_um"] += dx_um
        state["y_um"] += dy_um
        state["lost"] = 0

    def gen():
        for i in range(N_FRAMES):
            yield MDAEvent(
                x_pos=state["x_um"],
                y_pos=state["y_um"],
                channel={"config": channel},
                min_start_time=i * INTERVAL_S,
                metadata={"i": i},
            )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- [[../../../src/core/workflows/stage_tracking.py]] — `make_tracker`, `locate_target`, `center_on_target`, `spiral_search`, `track_target`, `track_target_mda` already implement the full loop including target-loss recovery (spiral search) and linear-velocity prediction between frames. Rabut & Ellenberg is the 2004 paradigm this module instantiates.
- [[../../../src/core/detection/cells.py]] — `find_bright_centroid` gives the mass-centre primitive the paper names as its target estimator. For confluent cultures where a single bright centroid is ambiguous, use `detect_cells` and match to the tracked identity by nearest neighbour / Hungarian assignment between frames.
- [[../../../src/core/workflows/autofocus.py]] — Rabut & Ellenberg also close the Z axis each revisit (short focus sweep + sharpness metric); `sweep_focus` + `focus_metric` cover this. On a modern rig `check_and_correct_focus` or DL autofocus (Pinkard 2019) replaces the sweep.
- Multi-cell revisit: a `stage_positions` loop with per-position `state` dictionaries. The paper's throughput gain comes from time-sharing one microscope across N independently tracked cells — the same pattern as Micropilot's multi-site FRAP but with tracking instead of phenotype scoring driving the revisit decision.

Calibration considerations:
- **Motion budget**: pick INTERVAL_S so that expected cell motion between revisits is well below the FOV radius (e.g. 1 µm / min migration × 60 s / revisit = 1 µm; at 40× with 0.16 µm/pixel and 512×512 FOV this is ~1% of the FOV — safe). If motion approaches the FOV radius, centroid will jump to the wrong cell.
- **Max-offset clamp** prevents a neighbouring bright cell from capturing the track when the target dims transiently.
- **Z tracking** is the failure mode the paper flags first — XY is easy, focus drift is what kills long live-cell runs. Always close both axes, not just XY.

## Cited by

- [[Core/Strategies/Adaptive acquisition]] — target-driven acquisition: the FOV is a closed-loop setpoint attached to the cell, not to the stage. Single-cell analogue of McDole 2018's specimen-tracking; the canonical early closed-loop-tracking microscope and the conceptual parent of `src/core/workflows/stage_tracking.py`.
