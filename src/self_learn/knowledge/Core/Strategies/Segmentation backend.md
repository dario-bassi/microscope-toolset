# Segmentation backend (Cellpose / StarDist / sigma)

How to choose between deep-learning instance segmentation and
sigma-tuned thresholding inside a smart-microscope recipe — and the
pluggable abstraction that lets a single recipe call site serve both
without forking the code path.

## When to reach for which

**Deep-learning instance segmentation (Cellpose, StarDist):**
- Tessellated tissue with shared cell boundaries — adjacent cells
  cannot be separated by intensity threshold alone (the ch599 Kaede
  failure mode).
- Sample geometry where morphology carries the cell-vs-background
  signal, not intensity (phase-contrast nuclei, brightfield WBCs,
  histology).
- Count-style submissions where a per-instance label is the right
  primitive (ground-truth records `sim._converted.sum()`-style
  per-object state).

**Sigma-tuned thresholding (the platform's existing path):**
- Sub-diffraction puncta (single molecules, FISH dots, synaptic
  vesicles) — Cellpose models aren't trained on point-source data.
- Fast-burst closed loops where wall-clock is critical — Cellpose-on-
  CPU is 5–30 s per 512×512 frame; sigma is < 5 ms.
- Bacteria-in-brightfield, plant cell walls, coral colonies — sample
  classes outside the standard Cellpose / StarDist training
  distribution.

## The pluggable abstraction

`src/core/detection/segmentation_backend.py` (sprint #10) exposes:

```python
from self_learn.detection.segmentation_backend import (
    Backend, LabeledMask, segment, register_backend,
    available_backends, labels_to_centroid_dicts,
)

mask = segment(image, backend="auto", sigma=2.5, min_area=20)
# → LabeledMask(labels, backend_used, inference_ms, ...)
cells = labels_to_centroid_dicts(mask, image=image)
# → [{centroid_px, area_px, mean_intensity, max_intensity, label_id}, ...]
```

`backend="auto"` picks the heaviest available backend that fits a
caller-supplied `budget_s`; `auto` with `budget_s < 2.0` falls
through to sigma. A `LabeledMask` carries `backend_used` and
`inference_ms` so the agent can record which path actually ran.

Recipes opt in at the boundary they care about. The `detect_cells`
function in `src/core/detection/cells.py` accepts a `backend=`
kwarg; default `"sigma"` preserves all existing behaviour.

## The four-tier fallback ladder

Inside `segment(backend="auto")`:

1. GPU + Cellpose / StarDist installed + budget ≥ 2 s → run the
   heavy backend on the full frame.
2. CPU + ROI candidates supplied (`roi_centers=[(x, y), ...]`) →
   crop the image around each candidate, run the heavy backend on
   each ~128 px patch (cuts 30 s × 1 → 0.3 s × N).
3. `budget_s < 2 s` → drop to sigma immediately, log
   `backend_used="sigma_budget_fallback"`.
4. Streaming timelapse → first frame runs the heavy backend; later
   frames reuse the previous label map as a prior, only re-running
   when registration drift > `max_drift_px` or count delta > 20%.

Tiers 2 and 4 are not yet implemented — sigma fallback (tier 3) and
full-frame heavy run (tier 1) are. The ROI and streaming tiers are
sprint-#10 follow-ups.

## Adapter contract

`labels_to_centroid_dicts(mask, image=...)` reproduces the dict shape
that the existing `src/core/detection/cells.detect_cells` returns
(centroid_px, area_px, plus optional mean / max intensity). Recipes
that work on the dict shape don't notice the backend swap. Sigma-
specific extras (`circularity`, `solidity`, `eccentricity`) are not
populated by non-sigma backends; recipes that depend on those
descriptors stay on the sigma path.

## Composing in a recipe

Pilot migration: `src/recipes/two_population_stain.detect_all_cells`
took a `backend=` kwarg without changing its existing call sites.
The non-default backends invert the phase-contrast image first
(Cellpose / StarDist / sigma all expect bright-on-dark), then the
recipe re-applies its `pc_dark_max` validation on each label's
centroid — preserving the ch597 lesson that real nuclei sit darker
than cell-body artefacts.

The same migration shape will work for `blood_smear_wbc`,
`hemocytometer`, and `fibroblast_focal_adhesions` — all
phase-contrast / brightfield + per-cell morphology workflows.

## Anti-patterns

- **Don't loop sigma → Cellpose → StarDist trying to find a count
  that matches GT.** That's the trap `[[Core/Pitfalls/Sim-state vs rendered count asymmetry]]`
  documents — when the rendering decoheres from the sim's per-object
  state, no segmentation method recovers the per-object index.
- **Don't use Cellpose for sub-diffraction puncta detection.** The
  default `cyto` model is trained on cell-body morphology; small
  bright spots are below its diameter range. Use `blob_log` or LoG
  thresholding instead (`src/core/analysis/puncta.py`).

## See also

- [[../../Recipes/Two population stain]] — pilot migration target.
- [[Core/Pitfalls/Sim-state vs rendered count asymmetry]] — when no
  backend can close the gap.
- [[Core/Approach/Detection strategy]] — choosing a primitive for the
  sample at hand.
