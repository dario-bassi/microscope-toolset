---
title: Imaging of cellular dynamics from a whole organism to subcellular scale with self-driving, multiscale microscopy
authors: Stephan Daetwyler, Hanieh Mazloom-Farsibaf, Felix Y. Zhou, Dagan Segal, Etai Sapoznik, Bingying Chen, Jill M. Westcott, Rolf A. Brekken, Gaudenz Danuser, Reto Fiolka
year: 2025
venue: Nature Methods 22(3):569-578
doi: 10.1038/s41592-025-02598-2
url: https://www.nature.com/articles/s41592-025-02598-2
researched: 2026-04-25
---

## Abstract

Most biological processes, from development to pathogenesis, span multiple time and length scales. While light-sheet fluorescence microscopy has become a fast and efficient method for imaging organisms, cells and subcellular dynamics, simultaneous observations across all these scales have remained challenging. Moreover, continuous high-resolution imaging inside living organisms has mostly been limited to a few hours, as regions of interest quickly move out of view due to sample movement and growth. Here, we present a self-driving, multiresolution light-sheet microscope platform controlled by custom Python-based software, to simultaneously observe and quantify subcellular dynamics in the context of entire organisms in vitro and in vivo over hours of imaging. We apply the platform to the study of developmental processes, cancer invasion and metastasis, and we provide quantitative multiscale analysis of immune-cancer cell interactions in zebrafish xenografts.

## Smart microscopy principle

Daetwyler et al. fuse two distinct light-sheet modalities on the same chassis — a low-resolution multidirectional SPIM (mSPIM) overview and a high-resolution axially-swept light-sheet (ASLM) capture — and run a controller that uses the overview to **continuously re-target the ASLM volume at a moving region of interest**. The contribution is the fusion of two ideas that earlier papers handled separately: McDole 2018 keeps a developing specimen inside a fixed-resolution acquisition volume by re-measuring its bounding box each time point; Shi 2024 (smartLLSM) switches modality between a cheap scout and an expensive capture at detected events. Daetwyler is both at once — two modalities live concurrently, the cheap one runs *continuously* and feeds a 3D region-tracking algorithm that updates the high-resolution stage / Z setpoint of the expensive one *during* the experiment, so an organism-scale view and a subcellular-scale view are recorded simultaneously over hours.

The transferable lesson is the **dual-rate multiscale acquisition pattern**: don't pick between organism-scale and subcellular-scale — run a slow, low-dose, wide-FOV channel to localise the moving target, and a fast, high-dose, narrow-FOV channel that follows it, with the wide channel's analysis driving the narrow channel's setpoint. This generalises beyond light-sheet to any rig with an objective turret or a digital ROI: a 4× / 10× scout updates a 40× / 60× capture region every N main frames. Cancer invasion in zebrafish xenografts is the paper's headline demonstration — an immune-cancer interaction that wanders out of a static high-mag FOV in minutes is held inside the ASLM volume for hours by closing the loop on the mSPIM overview.

## Implementation on pymmcore-plus

A widefield/confocal rig doesn't have mSPIM and ASLM, but the dual-rate pattern collapses to: (i) a low-mag overview at coarse cadence, (ii) a 3D centroid / bbox extraction on each overview, (iii) a high-mag stack centred on the updated target, repeated until the experiment ends. The same generator + on-frame callback pattern McDole 2018 uses for specimen tracking applies, with a higher overview cadence and a per-overview update of the high-mag setpoint.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events, set_objective
from src.core.detection.tissue import segment_tissue

LOW_MAG = 4
HIGH_MAG = 40
OVERVIEW_EVERY_N = 10        # one mSPIM-equivalent overview per 10 ASLM stacks
HIGH_MAG_Z_RANGE = (-15, 15) # um, narrow stack around the tracked target
HIGH_MAG_Z_STEP = 0.5
N_HIGH_MAG_FRAMES = 600

state = {"target_xy_um": None, "target_z_um": None}

def on_frame(img, event, meta=None):
    phase = (event.metadata or {}).get("phase")
    if phase != "overview":
        return
    mask = segment_tissue(img, method="otsu")
    if not mask.any():
        return
    # Per-tile centroid -> world coords (use src/core/workflows/adaptive.pixel_to_world)
    state["target_xy_um"] = _centroid_world(mask, event)
    state["target_z_um"] = _focus_z_from_overview(img, event)

def gen():
    for i in range(N_HIGH_MAG_FRAMES):
        if i % OVERVIEW_EVERY_N == 0:
            # Switch to low-mag, take a scout, return to high-mag
            yield MDAEvent(action="set_objective", data={"mag": LOW_MAG},
                           metadata={"phase": "switch"})
            yield MDAEvent(channel={"config": "BF"},
                           metadata={"phase": "overview", "i": i})
            yield MDAEvent(action="set_objective", data={"mag": HIGH_MAG},
                           metadata={"phase": "switch"})
        if state["target_xy_um"] is None:
            continue
        x, y = state["target_xy_um"]
        z0 = state["target_z_um"] or 0.0
        for dz in range(*HIGH_MAG_Z_RANGE, int(HIGH_MAG_Z_STEP * 10)):
            yield MDAEvent(x_pos=x, y_pos=y, z_pos=z0 + dz / 10,
                           channel={"config": "GFP"},
                           metadata={"phase": "main", "i": i})

run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- [[../../../src/core/workflows/multiscale.py]] — `select_rois`, `acquire_at_level`, `multiscale_acquire` already do hierarchical N-level acquisition; what is missing is the *dual-rate concurrent* form (overview keeps running while capture runs) — currently the pattern is run-overview-then-run-capture sequentially. A `dual_rate_multiscale(core, scout_every_n=10, ...)` helper would fit alongside `multiscale_acquire`.
- [[../../../src/core/workflows/stage_tracking.py]] — `make_tracker`, `track_target` give the closed-loop centroid-following primitive; Daetwyler's "3D region tracking algorithm" is the volumetric generalisation (track an XYZ centroid + bbox, not just an XY one).
- [[../../../src/core/workflows/engine.py]] — `MicroscopyEngine` custom action `switch_objective` is the per-event mode switch the dual-rate generator interleaves.
- [[../../../src/core/detection/tissue.py]] — `segment_tissue` provides the per-overview mask the centroid extraction operates on.

Differences from the paper to keep in mind:

- mSPIM and ASLM share the sample but use different illumination geometries; on a single-objective rig the modality switch collapses to an objective change (4× ↔ 40×), which costs ~1 s of stabilisation per switch — budget it against your acquisition cadence.
- The paper's "3D region tracking algorithm" runs on a true volumetric overview; on a 2D widefield scout you track XY centroid only and must autofocus the high-mag capture separately ([[Papers/Royer 2016]]).
- Hours-long imaging compounds drift: combine the overview-driven tracking with periodic AutoPilot-style alignment maintenance ([[Core/Strategies/Closed-loop autofocus]]) and per-cell exposure adaptation ([[Core/Strategies/Adaptive acquisition]]).

## Cited by

- [[Core/Strategies/Multi-scale morphometry]] — dual-rate multiscale: organism-scale overview *concurrently* drives subcellular-scale capture; generalises survey→zoom from a one-shot pipeline to a continuous control loop. Sibling to [[Papers/Shi 2024]] (modality-switching at events) and [[Papers/Conrad 2011]] (offline survey then per-hit capture).
- [[Core/Strategies/Adaptive acquisition]] — region-tracking acquisition: the high-resolution imaging volume is a continuously-updated setpoint driven by a parallel low-resolution channel; sibling to [[Papers/McDole 2018]] (whole-specimen tracking) and [[Papers/Royer 2016]] (alignment tracking) on the "instrument adapts to sample" axis.
- [[Core/Strategies/Timelapse design]] — for hour-scale timelapses on motile / growing samples (organoids, xenografts, developing embryos), a static high-mag FOV loses the target in minutes; co-acquire a low-mag scout and use it to update the capture setpoint each scout cycle.
