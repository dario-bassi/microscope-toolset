# Pitfall: FOV does not cover the whole well

## The symptom

You submit a count-based answer (total cells, fraction of X, etc.) and
the grader reports cell counts well below the ground truth. Hill fits
flatten, dose-response curves compress, EC50 estimates drift.

## The cause

The challenge spec (or the simulator) seeds cells across a physical
well (e.g. 500 × 500 µm). Your microscope's field of view at the
current objective covers only part of that well — typical 512 × 512
camera at 20× (px = 0.5 µm/px) sees only 256 × 256 µm, or **a quarter
of the well by area**.

Cells beyond the FOV go unseen. When you compute a fraction, the
numerator/denominator are both from the FOV, which is fine for local
metrics — but when the GT answer is a *whole-well* quantity that
assumes all seeded cells are counted, the local fraction can be
biased (a corner of the well may over- or under-represent the treated
population).

## How to detect

Before acquiring, mentally (or explicitly) compute:

```
FOV_area_um2 = (cfg.image_width * cfg.pixel_size_um) * (cfg.image_height * cfg.pixel_size_um)
expected_cell_count_per_FOV = expected_seed_count * FOV_area_um2 / well_area_um2
```

If the challenge hints at ~40 cells per well, and you see ~10 at the
current mag, you are seeing ~1/4 of the well.

Also: after segmenting, compare `n_cells` across wells. Variable
counts that don't track dose (e.g. 12, 21, 16, 30, 17 for 5 wells)
are suspicious — seeded wells should have similar counts under
homogeneous seeding.

## The fix

For counting tasks, default to **10×** (the widest commonly-available
FOV). Higher mags are for morphology, not counting:

```python
from src.core.hardware.core import set_objective
from src.core.hardware.config import refresh_config

set_objective(core, 10)
cfg = refresh_config(core)
```

If the sample has features that need high resolution AND need whole-well
counting, do a two-pass scan: 10× survey for the count, then 20× or
40× at the same stage positions for per-cell morphology.

## Related pitfalls

- **Droplet fusion at 20×** (ch569 r1): when detecting bright sub-cellular
  features (lipid droplets, puncta) in cells that pack them densely,
  individual features may fuse into a single large blob at sub-Nyquist
  sampling. LoG detection then undercounts. Either go to 40× to resolve
  individuals, or use an **intensity-based proxy** (mean fluorescence per
  cell; fraction of cell area above a BODIPY threshold) that is
  fusion-robust.
- **Non-monotonic dose-response**: a dip in the middle of a dose titration
  is almost always a segmentation bug (over-counting in one well,
  under-counting in another), not a biological effect — unless the GT
  itself is biphasic.

## References

- ch569 round 1 (20× default, EC50 fit = 12.6 µM vs GT 1.5 µM → 6/10).
- ch569 round 2 (10× + retuned LoG params, EC50 fit = 1.35 µM, R² = 0.995).
