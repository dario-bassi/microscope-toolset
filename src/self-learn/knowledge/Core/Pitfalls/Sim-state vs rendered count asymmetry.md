# Sim-state vs rendered-count asymmetry

## What goes wrong

You set up a clean segmentation pipeline (threshold → connected components,
or peak_local_max, or watershed). You apply it to your camera frames. The
output count never matches ground truth, no matter how you tune the
thresholds — sometimes high, sometimes low, never inside the tolerance
band.

## When it bites

Whenever the grader reports `sim._converted.sum()` (or any other
**per-object boolean state** counted directly from the simulator's data
structures) but the *rendered camera image* has a fundamentally different
visual structure:

- **Shared cell membranes**: Kaede / cadherin / ZO-1 / E-cadherin / any
  membrane fluorophore in a close-packed epithelial tessellation. Each
  cell carries the label, but the rendered membrane signal is shared
  with all its neighbours. Adjacent cells fuse into one connected
  component; widely-separated cells render as several blobs apiece.
- **Shared cytoplasm via gap junctions**: any reporter that diffuses
  freely between coupled cells. Sim has N labelled cells; renderer
  shows one continuous bright domain.
- **Sub-cell features render at sub-pixel scale**: vertex artifacts in
  Voronoi-renderer backends (~5–15 px each), centriole-pair markers
  (~2 px), nucleolar foci. peak-finding will sometimes pick up these
  features alongside cells, but they aren't counted in the sim state.
- **Cell objects span multiple bright regions per cell**: e.g. a nucleus
  + a cytoplasmic punctum on the same cell would render as two CC
  blobs but count as one in the sim.

## Why it's a pitfall, not a bug

Rendered images have *spatial structure*. `sim._counter.sum()` does not.
When the renderer's spatial structure (shared membranes, gap junctions,
sub-pixel artifacts) decoheres from the sim's per-object accounting,
**no segmentation pipeline can recover the sim count without explicit
prior knowledge of the boundary geometry**. CC under-counts when objects
fuse; peak_local_max over-counts when each object renders as several
local maxima; watershed over-segments when the distance transform peaks
inside a single object.

This is a *methodology pitfall*, not a bug to fix in your code: the
rendered evidence simply does not contain the per-object indexing the
sim state has.

## Signature symptoms

- Any combination of segmentation parameters either over-counts or
  under-counts; there's no setting that lands within tolerance.
- The same pipeline that worked beautifully on a *sparse* version of
  the sample (one cell per FOV region) fails on a dense version.
- Increasing/decreasing density of the sample changes the apparent
  count *non-linearly* even when the per-cell physics is identical.
- The grader feedback says something like "your pipeline is correct,
  but GT counts the simulator state, not the rendered blobs."

## How to detect it before submitting

1. **Inspect the diff/raw image overlaid with your detections.** If you
   find yourself flicking between "this looks like 10 cells" and "this
   looks like 30 fragments" depending on threshold, you're in the
   asymmetry zone.
2. **Compare segmentation count vs. distance-transform-area count.**
   If `n_cells_distance_transform = total_mask_area / (median cell
   area)` differs from your CC count by a factor of 3+, the rendering
   has fused or fragmented at the boundaries.
3. **Ask the grader / sim author whether GT comes from sim state or
   from a rendered-image-segmentation reference.** This is the single
   most diagnostic question; the answer determines whether you can
   close the gap by tuning or whether you need a published "apparent
   count" GT.

## Right fix (sim author side)

Republish ground truth as the segmentation result on the rendered
image — analogous to ch593's apparent-IC50 fix where the Hill fit on
the rendered plate becomes the graded value. The pattern:

```
GT_apparent = canonical_segmentation(rendered_image)
GT_underlying = sim._state_count            # kept for transparency
```

The graded answer is `GT_apparent`. The agent's pipeline can then
match the grader's pipeline exactly, removing the asymmetry.

## Right fix (agent side, no GT republication)

When the grader insists on `sim._state.sum()` GT and won't republish:
**you can't reliably hit the count target, but you can still do the
right experimental design and report the apparent count.** The
methodology score (acquisition, scout, modality switch, sign-checks)
is independent of the count miss. Document the asymmetry in the
method_summary so the grader knows you understood the structural
issue.

## Detecting it programmatically (`render_vs_submit_check`)

`src/core/utils/render_vs_submit.py` exposes `render_vs_submit_check`
— a deterministic numerical guard that runs in-process right before
`submit_solution`. Pass `(image, submitted_value, recipe_re_detect_fn)`;
the utility re-runs the recipe's own detector on the rendered image
and compares. Severity ladder:

- `ok`: relative delta ≤ `rel_tol_flag` (default 5%).
- `flag`: delta in `(rel_tol_flag, rel_tol_block]` (default 5–20%) →
  log it in the method_summary and ship anyway.
- `block`: delta > `rel_tol_block` (20%) → abort and re-derive.
- `error`: detector raised → surfaced as a structured field, never
  silenced.

Position-list answers (NN rule "Position-based grading") are
Hungarian-matched and compared as a fraction of the image diagonal,
with a recall-floor that demotes "same N but different locations"
to at-least `flag`. An `abs_tol` escape hatch suppresses spurious
relative-error blocks on low-count integer answers (3 vs 4 is 33%
relative but only 1 absolute).

This is a numerical complement to the LLM-driven `pre-submit-review`
skill — `pre-submit-review` looks at the overlay, `render_vs_submit_check`
looks at the scalar/list/dict the recipe is about to submit. The two
are designed to compose: visual verification + numerical agreement.
See ch593 r3, ch599 r1–r7, ch348 r2 for the failure-mode trace.

## See also

- [[Recipes/Photoconversion]] — the active-conversion variant
  hits this on Kaede + close-packed Voronoi tissue (ch599 r1–r7,
  closed at 5/10 by grader as "GT vs render asymmetry on my side").
- [[Core/Pitfalls/Reaction-diffusion classification]] — sister pitfall where sim
  state and rendered intensities decohere through dynamics.
- [[Core/Approach/Visual verification]] — the render-and-look habit
  catches this *before* you submit.
- `src/core/utils/render_vs_submit.py` — the agent-side numerical
  guard described above; sister to `pre-submit-review` skill.
- `src/core/utils/presubmit.py` — `run_presubmit_guards()` orchestrator
  that wraps preflight + render_vs_submit + optional review-prep into
  a single `PresubmitResult` with early-exit-on-block (sprint #22).
- [[Core/Approach/Pre-submit guard architecture]] — the *why* of the
  3-tier guard pattern this pitfall motivates.
