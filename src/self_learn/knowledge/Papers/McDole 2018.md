---
title: "In Toto Imaging and Reconstruction of Post-Implantation Mouse Development at the Single-Cell Level"
authors: Katie McDole, Léo Guignard, Fernando Amat, Andrew Berger, Grégoire Malandain, Loïc A. Royer, Srinivas C. Turaga, Kristin Branson, Philipp J. Keller
year: 2018
venue: Cell 175(3):859-876.e33
doi: 10.1016/j.cell.2018.09.031
url: https://pubmed.ncbi.nlm.nih.gov/30318151/
researched: 2026-04-24
---

## Abstract

The mouse embryo has long been central to the study of mammalian development; however, elucidating the cell behaviors governing gastrulation and the formation of tissues and organs remains a fundamental challenge. A major obstacle is the lack of live imaging and image analysis technologies capable of systematically following cellular dynamics across the developing embryo. We developed a light-sheet microscope that adapts itself to the dramatic changes in size, shape, and optical properties of the post-implantation mouse embryo and captures its development from gastrulation to early organogenesis at the cellular level. We furthermore developed a computational framework for reconstructing long-term cell tracks, cell divisions, dynamic fate maps, and maps of tissue morphogenesis across the entire embryo. By jointly analyzing cellular dynamics in multiple embryos registered in space and time, we built a dynamic atlas of post-implantation mouse development that, together with our microscopy and computational methods, is provided as a resource.

## Smart microscopy principle

McDole et al. solve a problem Royer's AutoPilot can't: the sample *itself* moves, grows, and changes shape on the scale of the field of view. Between E6.5 and E8.5 the mouse embryo translates, rotates, and roughly doubles in volume while forming a cup-shaped gastrulating structure. A fixed imaging volume programmed at t=0 clips the embryo by t=12 h; a naively large volume wastes photon and time budget on empty space. The paper's microscope **measures where the embryo is at each time point** (centroid + bounding geometry from a low-resolution scout scan) and **reconfigures the acquisition volume** — stage centre, Z extent, XY tiles, light-sheet and detection-plane geometry — so the embryo stays centred and fully enclosed as it develops. The loop runs every time point for ~48 h.

The transferable smart-microscopy contribution is **specimen-tracking acquisition**: treat the imaged volume as a closed-loop setpoint that follows the specimen, not as a fixed rectangle set at t=0. This applies far beyond light-sheet or embryos — any long timelapse on a sample that translates (organoid, developing zebrafish, motile cell aggregate, spheroid on soft gel) benefits from re-measuring the sample's bounding region each time point and updating the tile grid / Z range accordingly. It sits alongside AutoPilot on the "adapt the instrument to the sample" axis — AutoPilot adapts optical alignment, McDole adapts the imaging volume.

## Implementation on pymmcore-plus

The pattern is a two-level generator: an outer loop over time points, and inside each time point (i) a low-resolution scout scan that estimates the specimen's current centroid and bounding box, (ii) a recomputed tile grid + Z range that covers the new bounding box with a configurable margin, (iii) the full-resolution acquisition on the updated volume. An `on_frame` callback on the scout scan updates shared state that the tile generator reads.

```python
import numpy as np
from useq import MDAEvent
from src.core.hardware.core import run_events
from src.core.detection.tissue import segment_tissue

SCOUT_OBJECTIVE = 4       # low-mag scout: survey the whole specimen cheaply
MAIN_OBJECTIVE = 20       # full-resolution acquisition
MARGIN_UM = 50.0          # padding around estimated bounding box
TIME_POINTS = 120         # e.g. 48 h at 24 min/timepoint
SCOUT_GRID = (3, 3)       # coarse 3x3 tiles to localise the specimen

def specimen_tracking_timelapse(core, pixel_size_um):
    """Each time point: scout -> estimate bbox -> update volume -> acquire.

    The imaging volume follows the specimen centroid and bounding box,
    re-computed at every time point. The full-resolution tile grid is
    regenerated from the updated bbox plus a fixed margin.
    """
    state = {"bbox_um": None, "centroid_um": None, "scout_accum": []}

    def update_bbox():
        """Fuse scout tiles into a sample mask, extract bbox + centroid."""
        tiles = state["scout_accum"]
        # Project stage-registered tiles into a common mosaic; segment
        # specimen vs background (brightfield or membrane channel).
        mask = segment_tissue(_mosaic(tiles), method="otsu")
        ys, xs = np.where(mask)
        if ys.size == 0:
            return  # keep previous bbox
        cy_px, cx_px = ys.mean(), xs.mean()
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        # Convert pixel bbox -> world-space bbox using tile stage origins
        state["centroid_um"] = _pixel_to_world(cy_px, cx_px, tiles, pixel_size_um)
        state["bbox_um"] = _pixel_bbox_to_world(
            (y0, x0, y1, x1), tiles, pixel_size_um, margin=MARGIN_UM,
        )
        state["scout_accum"] = []

    def on_frame(img, event, meta=None):
        phase = (event.metadata or {}).get("phase")
        if phase == "scout":
            state["scout_accum"].append((img, event.x_pos, event.y_pos))
            if len(state["scout_accum"]) == SCOUT_GRID[0] * SCOUT_GRID[1]:
                update_bbox()

    def gen():
        for t in range(TIME_POINTS):
            # 1. Scout scan at low magnification
            for (sx, sy) in _scout_positions(state["centroid_um"], SCOUT_GRID):
                yield MDAEvent(
                    x_pos=sx, y_pos=sy,
                    channel={"config": "BF"},
                    metadata={"phase": "scout", "t": t},
                )
            # 2. Reconfigure the main acquisition volume from the new bbox
            if state["bbox_um"] is None:
                continue
            tiles = _tile_grid(state["bbox_um"], MAIN_OBJECTIVE)
            z_range = _z_range(state["bbox_um"])
            # 3. Full-resolution tiles + Z-stack on the updated volume
            for (x, y) in tiles:
                for z in z_range:
                    yield MDAEvent(
                        x_pos=x, y_pos=y, z_pos=z,
                        channel={"config": "GFP"},
                        index={"t": t},
                        metadata={"phase": "main", "t": t},
                    )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- [[../../../src/core/workflows/stage_tracking.py]] — already provides `make_tracker`, `locate_target`, `center_on_target`, `track_target` for the single-cell analogue of this loop. McDole generalises this from "keep one cell in the FOV" to "keep an entire specimen inside the imaged volume"; the transferable primitive is the same (measure centroid + bounds, update setpoint, re-plan next acquisition).
- [[../../../src/core/workflows/scanning.py]] — tile-grid generation. An adaptive tile-grid helper that takes a world-space bounding box and returns the minimal tile set covering it with overlap does not yet exist — candidate for `src/core/workflows/adaptive_volume.py`.
- [[../../../src/core/detection/tissue.py]] — `segment_tissue` gives the per-tile specimen mask that feeds the bounding-box estimator.
- The light-sheet-specific degrees of freedom (illumination/detection geometry adapted to local embryo shape) collapse to Z range + tile grid on a widefield/confocal rig; the generator pattern is unchanged.

Calibration on a real prep: the scout scan must be fast and gentle (brightfield or low-exposure fluorescence at low magnification) so its photon and time overhead is small relative to the main acquisition — a 3×3 scout grid every 24 min is typical when the main volume is a 7×7 tile grid with a 30-plane Z-stack. The margin around the measured bbox should exceed the specimen's expected displacement between scout-and-acquire to prevent clipping if the specimen moves during the main acquisition.

## Cited by

- [[Core/Strategies/Adaptive acquisition]] — specimen-tracking acquisition: the imaged volume itself is a closed-loop setpoint that follows a growing, moving specimen across long experiments. Sibling to AutoPilot on the "instrument adapts to sample" axis.
- [[Core/Strategies/Timelapse design]] — for long timelapses on motile / growing specimens, the imaging volume is not fixed at t=0; re-measure the specimen's bounding box per time point and update the tile grid and Z range accordingly.
