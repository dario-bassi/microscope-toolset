# Pitfall: Stage drift exceeding autofocus range

> **When to use:** When autofocus corrections are applied but images progressively blur anyway over a long timelapse.

## Symptom

Autofocus correction is applied every frame, but images progressively blur anyway. The autofocus score keeps improving (it finds the best Z within its search range) but the focus quality measured on the image keeps dropping. At late timepoints the image is visibly out of focus even immediately after a correction step.

## Cause

The autofocus loop searches within a fixed Z window (`z_range`). If the sample drifts faster than the frame interval × drift rate, the true focus plane exits the search window before the next correction can capture it. The autofocus then finds the best Z *within the window* — which is not the true focus — and reports a non-zero sharpness score that looks like success.

A secondary cause: Z-scan correction itself deposits dose and may cause thermal drift, compounding the problem (see [[Core/Pitfalls/Z-drift autofocus]]).

## How to detect

1. **Plot the autofocus correction magnitude per frame.** If corrections hit the edge of the search window consistently (`abs(z_correction) ≈ z_range / 2`), the drift is exceeding the range.
2. **Compare sharpness before and after correction.** If post-correction sharpness is less than 90% of the initial baseline, the loop is losing ground.
3. **Check focus score trend.** A monotonically declining Brenner/Laplacian variance across the whole timelapse indicates the autofocus is not tracking.

## Fixes

### Widen the search range
Increase `z_range` to cover ≥ 2× the expected drift per frame interval:
```python
from self_learn.workflows.autofocus import run_autofocus_loop

run_autofocus_loop(core, channel="BF", z_range=10.0, n_steps=11, interval_s=30.0)
```
Tradeoff: more Z steps = more dose per correction = faster bleaching. Balance with longer correction intervals.

### Increase correction frequency
Reduce the timelapse interval so drift is smaller between corrections. If the biology allows a longer frame interval, use `min_start_time` on `MDAEvent` to pace the loop.

### Separate tracking from measurement channel
Use a brightfield or a robust nuclear stain for focus tracking (fewer bleaching constraints) and reserve the experiment channel for quantification only.

### Check for thermal drift
If drift is fast at the start of a session and slows after 30–60 min, it is likely thermal. Pre-warm the objective and stage for 30 min before a critical timelapse. The correction loop will then need a smaller search range for the stable period.

## When autofocus is not the right tool

If the sample is actively moving (migrating cells leaving the FOV, organoids rotating, zebrafish developing), autofocus in Z cannot help. Use a multi-position scan instead and track objects across positions.

## See also

- [[Core/Strategies/Closed-loop autofocus]] — the active per-frame Z-correction strategy.
- [[Core/Pitfalls/Z-drift autofocus]] — the related pitfall where Z-scan itself causes drift.
- [[Core/Strategies/Timelapse design]] — frame interval and exposure budget choices.
