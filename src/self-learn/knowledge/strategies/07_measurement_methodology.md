# Strategy 07: Measurement Methodology

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
- Use `measure_bacteria_intensity()` from `src.workflows.bacteria_trap` for enrichment
- Use counting when absolute cell number is required (not just relative concentration)

### 12. Growth Rate: Use Log-Linear Regression, Not Endpoints
- `(n_end - n_start) / (n_snaps * n_start)` is noisy for small n and short observation windows
- Log-linear regression over all snaps: more robust (noise averages out)
- Returns R-squared to quantify fit quality — if R² < 0.5, growth is not exponential
- Use `measure_growth_rate_series(counts)` from `src.analysis.kinetics`
- More snaps = better regression (use N=20+ for reliable Q10 measurements)
