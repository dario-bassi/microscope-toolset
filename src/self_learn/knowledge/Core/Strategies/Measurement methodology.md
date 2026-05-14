# Measurement Methodology

> **When to use:** When designing what and how to measure to ensure statistical validity and the right methodology for the biological question.

## Principle
Detection accuracy alone doesn't determine score. **HOW** you measure is as important
as **WHAT** you detect. Session 15 showed perfect detection with 5-6/10 scores due to
measurement methodology errors.

## Rules

### 1. Match Measurement to FOV
- Before measuring, check if the object fits in the FOV
- Object length > 0.5 × FOV → use lower magnification
- Only include objects whose bounding box is >10px from all FOV edges
- If ALL objects touch the border → wrong magnification for this measurement

### 2. Instantaneous vs Cumulative
- **Instantaneous**: current state at measurement time (FHN wave coverage, live cell count)
- **Cumulative**: all-time total (cells ever converted, total area swept)
- Default to **instantaneous** unless explicitly asked for cumulative
- Wave/excitable media: active front is a ring, behind it tissue recovers

### 3. Temporal Dynamics Before Classification
- Never classify a dynamic system from a single snapshot
- Compare 3-4 timepoints for:
  - Feature splitting (mitosis) vs tip growth (coral)
  - Rotating arms (spirals) vs traveling fronts (waves)
  - Static features (spots/stripes) vs dynamic (everything else)

### 4. Report What's Asked
- "growth rate" might mean rate constant (k) or ratio (N_final/N_initial)
- "count" might mean all visible or only fully-contained or only alive
- "coverage" might mean currently active or ever activated
- Read the expected output format carefully, match units and conventions

### 5. Morphological Validation
- Don't classify objects by intensity alone
- Require shape (AR, eccentricity), size (area), and intensity together
- Background noise passes intensity thresholds but fails shape/size checks
- Ghost/dead cells have specific morphology, not just dim signal

### 6. 2D Projections of 3D Structures
- Microscope images are 2D projections — for 3D objects, scale appropriately
- **Spheroid viability**: necrotic core diameter d, total D → viable fraction = 1 - (d/D)^3 (volume)
- NOT (d/D)^2 (area) or d/D (linear) — common trap in microscopy
- Organoid volume ∝ d^3, cross-section area ∝ d^2, diameter ∝ d

### 7. Temporal Signal Analysis
- For oscillatory signals (GCaMP, cell cycle reporters): use peak detection + frequency extraction
- Ensure adequate temporal sampling (Nyquist: >2× the signal frequency)
- Distinguish signal periodicity from noise — require consistent peak-to-peak intervals

### 8. Start with Low-Mag Overview (Multi-Scale Rule)
- **Before detailed acquisition, snap a low-magnification overview** to count total objects and map positions
- At high magnification, the FOV shrinks — you may only see a fraction of the sample
- Multi-scale workflow:
  1. **Low-mag overview**: Count all cells/structures, record positions
  2. **High-mag detail**: Visit each cell for subcellular counts (fibers, FA, organelles)
  3. **Timelapse**: Either track globally at low-mag OR pick representative high-mag position
- Report overview counts for total cell/object numbers, high-mag counts for per-cell detail
- Intensity ratios from a single high-mag FOV are reliable; total counts need the full-field view

### 9. Fixed vs Adaptive Thresholds for Temporal Tracking
- When counting structures across a timelapse, use a FIXED threshold from the baseline
- Adaptive thresholds (per-frame percentile) track the signal down → artificially stable counts
- The fixed approach reveals the true extent of structural change over time

### 10. PSF-Aware Boundary Measurement (ch417 lesson)
- Fluorescence boundaries appear 30-50% wider than physical structures (PSF spread)
- For thin structures (walls, membranes): fluorescence thickness >> true thickness
- **Prefer brightfield edges** for wall/boundary thickness (BF has sharper gradients)
- Alternative: measure at intensity half-maximum (FWHM), not visible boundary
- Ring structures: use BF gradient for outer/inner boundaries, fluorescence for center-finding

### 15. FFT-Peak Orientation Extraction (ch615 lesson — sprint #32)
- When a frame carries a visible periodic modulation (SIM grating, sarcomere
  banding, focal-adhesion stripes, microvilli), the orientation question is
  cleanest in Fourier space: **bg-subtract → fft2 → fftshift → bandpass →
  argmax → atan2(ky, kx) mod π**.
- Bandpass excludes DC (`r_lo`) and corner / edge artifacts (`r_hi`); without
  it the DC peak dominates and the bandpass mask has to be huge to beat it.
- ALWAYS sanity-check `wavelength_px = max(shape) / k_mag` against the
  expected fringe period — confirms you found the real peak, not a noise
  lobe.
- Modulo-π folds the ±k symmetry of the real-valued FFT (every real grating
  has peaks at both +k and -k); orientations naturally live in [0°, 180°).
- Use `utils.fft_peak.find_fft_peak(image)` for single-orientation
  detection; `find_top_n_fft_peaks(image, n)` for multi-component decomposition
  (with ±k symmetric suppression). Transfers cleanly to the SIM 9-frame
  collection / SR-SIM reconstruction / sarcomere-and-adhesion detection
  family the ch615 grader called out as REUSABLE.

### 16. Fluorophore-Brightness Prediction (ch617 lesson — sprint #33)
- When channel SNR / brightness ranking is asked, the closed-form
  prediction is **ε × Φ from the fluorophore registry** (FPbase /
  Lambert 2019 framing), not measurement-only.
- Pattern: build channel→fluorophore→brightness map, normalise to
  dimmest, rank, then snap each channel at a common exposure to
  verify. Tolerance ±50% on the non-dimmest ratio reflects sample-
  inhomogeneity floor (voronoi nuc-only vs mem-only render densities
  differ substantially).
- Use `utils.fluorophore_brightness.predict_brightness_ranking`
  for Phase A (closed-form), `measure_brightness_ranking(core, channels,
  exposure_ms=...)` for Phase B (snap each channel + reducer over FOV);
  `compare_predictions(predicted, measured, ratio_tol=0.5)` is the
  verification step (returns ranking_match + per-channel rel-err
  + max_rel_err + within_tolerance).
- Same primitive transfers to bleach-rate ranking (registry's
  `bleach_kx`), filter-set SNR design, and bleed-through severity —
  REUSABLE per ch617 grader. Compose with `utils.spectral_leak`
  once challenges land.

## Checklist Before Submitting
- [ ] Am I measuring what's asked? (instantaneous/cumulative, rate/ratio, count/density)
- [ ] Are all measured objects fully within the FOV?
- [ ] **Did I snap a 10x overview first?** (for cell count / spatial coverage)
- [ ] Did I validate detections with morphological criteria?
- [ ] Did I check temporal dynamics if classification is required?
- [ ] Do my units/conventions match the expected output?
- [ ] Am I using fixed baseline thresholds for temporal comparisons?

### 11. Intensity vs Counting for Population Measurements
- When measuring POPULATION ENRICHMENT or DENSITY RATIOS, prefer intensity over counting
- Summed fluorescence ∝ biomass regardless of shape, fragmentation, or detection parameters
- Counting is subject to systematic errors from: shape assumptions, area filters, noise
- Key insight (ch539): area-threshold counting was 3× inflated (rod fragmentation),
  but intensity-based enrichment was correct because: `intensity_in/intensity_total / area_frac`
  gives the true density ratio even when absolute counts are wrong
- Use `measure_bacteria_intensity()` from `src.recipes.bacteria_trap` for enrichment
- Use counting when absolute cell number is required (not just relative concentration)

### 12. Growth Rate: Use Log-Linear Regression, Not Endpoints
- `(n_end - n_start) / (n_snaps * n_start)` is noisy for small n and short observation windows
- Log-linear regression over all snaps: more robust (noise averages out)
- Returns R-squared to quantify fit quality — if R² < 0.5, growth is not exponential
- Use `measure_growth_rate_series(counts)` from `analysis.kinetics`
- More snaps = better regression (use N=20+ for reliable Q10 measurements)

### 13. Report Positions, Not Just Counts

When a method is described by a count (cells, puncta, infected objects), the real measurement is the **set of positions**, and the count is derivative. A count collapses localisation to a number and hides two failure modes:

- **Calibrated-to-fail detectors** can hit the right count by picking up the wrong objects. Tune a threshold to land near an expected value and the submitted number looks correct even when no localisation is happening.
- **Compensating errors** hide themselves. Detect 10× too few objects in each class and the *ratio* is still correct. Only per-object matching surfaces this.

Position-based validation pairs submitted objects to ground truth by nearest-neighbour matching inside a tolerance radius, then scores **precision** (submitted objects that matched a GT object) and **recall** (GT objects that matched a submission). Precision ≈ recall ≈ 1.0 means the detector is actually finding the right things; a count-only grade can reward PR ≈ 0.5.

**How to apply:**
- For any count/density task: submit position lists (`[[x, y], ...]`) as the primary payload. Counts derived from the lists are fine as auxiliary fields.
- Pick `match_radius_px` ≈ object radius — generous enough that small centroid jitter doesn't miss, tight enough that unrelated near-neighbours don't match.
- Verify on the overlay, not on the count. A PR-graded detector with the right count but a ~2× offset centroid fails recall even though the scalar looks fine. [[Core/Approach/Visual verification]].
- This transfers straight to real validation: precision/recall on a held-out manually-annotated FOV is how a real detector gets published.

### 14. Rare-Event Sampling
- If the event rate you're measuring is **< 20%** of the population (rare mitoses, low parasitaemia, sparse transformants), a single FOV is statistically insufficient — the per-FOV count variance swamps the signal.
- Rule of thumb: **≥ 10 FOVs** when reporting a rare-event fraction; far more if the CI needs to be tight.
- Estimate the needed N from the binomial CI: for a fraction *p*, standard error is `√(p(1-p)/N)`. A 15 % fraction with 10 FOVs of 50 cells each has SE ≈ 1.6 %; with 2 FOVs it's ≈ 3.5 %.
- If survey time is bounded and the event is rare: use a **low-mag overview** first to spot-sample the whole well, then zoom to measure rate only in regions with confirmed events.

## Code patterns

The cross-cutting rules above are easier to apply as runnable sketches.

**Rule 13 — Position-based, not count-only.** Submit positions; let the
grader compute precision/recall.

```python
from self_learn.detection.cells import detect_cells

cells = detect_cells(img, threshold_sigma=2.5, min_area_px=50)
positions = [list(c['centroid_px']) for c in cells]   # [[x, y], ...]
answer = {
    "n_cells": len(positions),
    "positions": positions,                           # ← primary payload
}
```

**Rule 6 — 2D projection ⇒ cube the diameter.** Spheroid / organoid
viability is a *volume* fraction:

```python
viable_fraction = 1.0 - (d_necrotic_um / d_total_um) ** 3
# NOT (d_necrotic / d_total) ** 2  (area)
# NOT  d_necrotic / d_total        (linear)
```

**Rule 11 (implicit) — Gate kinetic answers on R².** A regression with
R² < 0.5 is not exponential growth; emit a `flag` rather than a confident
rate.

```python
from self_learn.analysis.kinetics import measure_growth_rate_series

reg = measure_growth_rate_series(counts)
if reg["r_squared"] < 0.5:
    method_description += f"  [flag: R²={reg['r_squared']:.2f} — not exponential]"
rate = reg["rate_per_frame"] if reg["r_squared"] >= 0.5 else float("nan")
```

**Rule 13 — Verify the answer matches the rendered image** before you
ship — this is what `render_vs_submit_check` is for. See
[[Core/Approach/Pre-submission checklist]] § 2b.
