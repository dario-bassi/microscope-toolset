# Apparent vs Underlying Count Asymmetry

> **When to use:** When segmentation count never matches the expected object count regardless of threshold tuning.

## What goes wrong

You set up a clean segmentation pipeline (threshold → connected components,
or peak_local_max, or watershed). You apply it to your camera frames. The
output count never matches the expected value, no matter how you tune the
thresholds — sometimes high, sometimes low, never inside the tolerance band.

## When it bites

Whenever the **expected count** comes from a per-object boolean state
(e.g., number of cells that converted, number of objects that crossed a
threshold) but the *rendered camera image* has a fundamentally different
visual structure:

- **Shared cell membranes**: cadherin / ZO-1 / E-cadherin / any membrane
  fluorophore in a close-packed epithelial tessellation. Each cell carries
  the label, but the rendered membrane signal is shared with all its
  neighbours. Adjacent cells fuse into one connected component; widely-separated
  cells render as several blobs apiece.
- **Shared cytoplasm via gap junctions**: any reporter that diffuses freely
  between coupled cells. The sample has N labelled cells; the rendered image
  shows one continuous bright domain.
- **Sub-cell features render at sub-pixel scale**: vertex artifacts, centriole-pair
  markers (~2 px), nucleolar foci. Peak-finding will sometimes pick up these
  features alongside cells, but they aren't individual objects.
- **Cell objects span multiple bright regions per cell**: e.g. a nucleus + a
  cytoplasmic punctum on the same cell would render as two connected-component
  blobs but count as one object in the expected answer.

## Why it's a pitfall, not a bug

Rendered images have *spatial structure*. An expected count does not.
When the image's spatial structure (shared membranes, gap junctions,
sub-pixel artifacts) decoheres from per-object accounting,
**no segmentation pipeline can recover the exact count without explicit
prior knowledge of the boundary geometry**. CC under-counts when objects
fuse; peak_local_max over-counts when each object renders as several local
maxima; watershed over-segments when the distance transform peaks inside a
single object.

This is a *methodology pitfall*, not a bug to fix in your code: the
rendered evidence simply does not contain the per-object indexing the
underlying state has.

## Signature symptoms

- Any combination of segmentation parameters either over-counts or
  under-counts; there's no setting that lands within tolerance.
- The same pipeline that worked beautifully on a *sparse* version of the
  sample (one cell per FOV region) fails on a dense version.
- Increasing/decreasing density of the sample changes the apparent count
  *non-linearly* even when the per-cell physics is identical.
- Feedback says something like "your pipeline is correct, but the count
  is measured from the rendered image, not the state."

## How to detect it before reporting

1. **Inspect the raw image overlaid with your detections.** If you find
   yourself flicking between "this looks like 10 cells" and "this looks
   like 30 fragments" depending on threshold, you're in the asymmetry zone.
2. **Compare segmentation count vs. distance-transform-area count.**
   If `n_cells_distance_transform = total_mask_area / (median cell area)`
   differs from your CC count by a factor of 3+, the rendering has fused
   or fragmented at the boundaries.
3. **Clarify whether the expected count comes from per-object state or
   rendered-image segmentation.** This is the single most diagnostic
   question; the answer determines whether you can close the gap by
   tuning or whether you need an "apparent count" approach.

## Right fix (on the data/GT side)

When possible, define ground truth as the segmentation result on the
rendered image:

```
GT_apparent = canonical_segmentation(rendered_image)
GT_underlying = per_object_state_count   # kept for transparency
```

The reported answer is `GT_apparent`. The analysis pipeline can then
match exactly, removing the asymmetry.

## Right fix (agent side, when GT definition is fixed)

When the underlying count is fixed and won't change:
**you can't reliably hit the count target, but you can still do the right
experimental design and report the apparent count.** The methodology
(acquisition, segmentation strategy, quality checks) is independent of
the count miss. Document the asymmetry in your method description so
collaborators understand the structural issue.

## Detecting it programmatically (`render_vs_submit_check`)

`utils/render_vs_submit.py` exposes `render_vs_submit_check` — a
deterministic numerical guard that re-runs your detector on the rendered
image and compares before reporting. Pass
`(image, submitted_value, recipe_re_detect_fn)`. Severity ladder:

- `ok`: relative delta ≤ `rel_tol_flag` (default 5%).
- `flag`: delta in `(rel_tol_flag, rel_tol_block]` (default 5–20%) →
  log in method description and proceed.
- `block`: delta > `rel_tol_block` (20%) → re-derive the answer.
- `error`: detector raised → surfaced as a structured field, never silenced.

Position-list answers are Hungarian-matched and compared as a fraction of
the image diagonal, with a recall-floor that catches "same N but different
locations" cases.

This is a numerical complement to visual review — `render_vs_submit_check`
looks at the scalar/list/dict the recipe is about to report; visual review
looks at the overlay. The two are designed to compose: visual verification
+ numerical agreement.

## See also

- [[Core/Pitfalls/Reaction-diffusion classification]] — sister pitfall where
  the rendered intensities decohere through dynamics.
- [[Core/Approach/Visual verification]] — the render-and-look habit catches
  this before you report.
- `utils/render_vs_submit.py` — the agent-side numerical guard described above.
- `utils/presubmit.py` — `run_presubmit_guards()` orchestrator that wraps
  preflight + render_vs_submit + optional review-prep.
- [[Core/Approach/Pre-submit guard architecture]] — the *why* of the
  3-tier guard pattern this pitfall motivates.
