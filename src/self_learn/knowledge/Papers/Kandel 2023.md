---
title: "Demonstration of an AI-driven workflow for autonomous high-resolution scanning microscopy"
authors: Saugat Kandel, Tao Zhou, Anakha V. Babu, Zichao Di, Xinxin Li, Xuedan Ma, Martin Holt, Antonino Miceli, Charudatta Phatak, Mathew J. Cherukara
year: 2023
venue: Nature Communications 14:5501
doi: 10.1038/s41467-023-40339-1
url: https://www.nature.com/articles/s41467-023-40339-1
researched: 2026-04-24
---

## Abstract

Modern scanning microscopes can image materials with up to sub-atomic spatial and sub-picosecond time resolutions, but these capabilities come with large volumes of data, which can be difficult to store and analyze. We report the Fast Autonomous Scanning Toolkit (FAST) that addresses this challenge by combining a neural network, route optimization, and efficient hardware controls to enable a self-driving experiment that actively identifies and measures a sparse but representative data subset in lieu of the full dataset. FAST requires no prior information about the sample, is computationally efficient, and uses generic hardware controls with minimal experiment-specific wrapping. We test FAST in simulations and a dark-field X-ray microscopy experiment of a WSe2 film. Our studies show that a FAST scan of <25% is sufficient to accurately image and analyze the sample. FAST is easy to adapt for any scanning microscope; its broad adoption will empower general multi-level studies of materials evolution with respect to time, temperature, or other parameters.

## Smart microscopy principle

FAST recasts "which tile should I scan next?" as an online active-learning problem: an inpainting-style neural network (SLADS-Net) predicts, for every unmeasured position, the Expected Reduction in Distortion (ERD) that a new measurement there would bring to the current reconstruction. Each iteration the controller picks the top-K ERD points, hands them to a travelling-salesman route optimiser (OR-Tools) so the stage/galvo trajectory is short, scans that batch, updates the reconstruction, and repeats. The loop starts from ~1 % quasi-random coverage and typically converges to a publication-quality image at <25 % total coverage.

Three things separate this from earlier survey→zoom paradigms. First, **no prior training on the sample class** — the network is trained once on a generic image (the classic "cameraman") and generalises because ERD is a reconstruction-quality metric, not a phenotype classifier. Second, **sampling is sparse-point, not two-tier**: there is no low-mag scout followed by high-mag revisit; the instrument acquires at its native resolution and the model decides which native-resolution points are worth the photons. Third, **route optimisation is a first-class scheduling step** — because dead time between arbitrary stage moves often dominates total experiment time, picking informative points without planning the path wastes the savings.

The paradigm transfers directly to whole-slide pathology and whole-well HCS: replace "next stage point" with "next tile", and ERD with any reconstruction- or classifier-confidence-based score. The instrument spends its photon and time budget on the tiles whose content the model can't yet predict from tiles already seen.

## Implementation on pymmcore-plus

The FAST loop is naturally an `MDAEvent` generator driven by a shared "scan state" that holds the current reconstruction, the ERD scorer, and the route optimiser. Each iteration emits a batch of K events along a route-optimised path; `on_frame` updates the reconstruction and ERD; the generator yields the next batch.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

BATCH = 50            # points per iteration
STOP_COVERAGE = 0.25  # stop once 25 % of grid has been measured

def fast_acquisition(core, grid_xy, scorer, router, channel="BF"):
    """FAST-style adaptive scan.

    scorer(reconstruction, measured_mask) -> ERD array over unmeasured points.
    router(points) -> ordered list (shortest path through the batch).
    """
    measured = {}          # (x, y) -> image
    mask = np.zeros(len(grid_xy), dtype=bool)

    # Seed: 1 % quasi-random coverage.
    seed_idx = quasi_random_indices(len(grid_xy), frac=0.01)
    batch = [grid_xy[i] for i in seed_idx]

    def on_frame(img, event, meta=None):
        measured[(event.x_pos, event.y_pos)] = img

    def gen():
        nonlocal batch
        while mask.mean() < STOP_COVERAGE:
            # Acquire along the route-optimised path.
            for (x, y) in router(batch):
                yield MDAEvent(x_pos=x, y_pos=y,
                               channel={"config": channel})
            # Update reconstruction + pick next batch.
            recon = inpaint(grid_xy, measured)
            erd = scorer(recon, mask)
            top_k = np.argpartition(-erd, BATCH)[:BATCH]
            batch = [grid_xy[i] for i in top_k]
            mask[top_k] = True

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/scanning.py` — `grid_positions` and `scan_events` provide the full-grid primitive; FAST replaces the "acquire every point" scheduling with a selective batch-loop.
- `../../../src/core/workflows/multiscale.py` — `multiscale_acquire` already implements hierarchical N-level acquisition with a score function; FAST's ERD would plug in as that score in a flat-grid configuration (no magnification switch, just sparse sampling).
- A dedicated sparse-sampling workflow module (`src/core/workflows/sparse_scan.py`) does not yet exist; this paper motivates one whose distinguishing features are (a) a pluggable information-gain scorer and (b) a route optimiser for the inter-point path. For WSI/HCS adaptation, `scorer` becomes a tile-level diagnostic-content classifier and "route" reduces to plate-well visit order.

Calibration on a real prep: the ERD surrogate in FAST is pre-trained on a generic image, but on a biological sample the reconstruction task (inpainting a brightfield tile mosaic, reconstructing a sparse fluorescence map, etc.) needs its own scorer. Hold back a fully-scanned reference slide, compute the ERD scorer's ranked selections against that ground truth, and tune `BATCH` / `STOP_COVERAGE` so the coverage-vs-reconstruction-error curve plateaus at an acceptable fidelity before the experiment runs.

## Cited by

- [[Core/Strategies/Adaptive acquisition]] — FAST formalises "where to scan next" as an online active-learning problem with an information-gain (ERD) score, extending survey→rank→zoom to flat-grid sparse sampling without a mag switch.
- [[Core/Strategies/Multi-position survey]] — route-optimised batch scheduling is the natural multi-position pattern when the grid is only partially visited; FAST shows dead-time between arbitrary stage moves is as important to optimise as which points to pick.
