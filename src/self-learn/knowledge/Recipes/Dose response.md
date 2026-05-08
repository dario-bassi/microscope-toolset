# Dose-Response Experiment Playbook

## When to Use
Multi-well experiments where different wells contain different drug concentrations. Goal: measure fluorescence suppression and fit Hill equation for IC50.

## Workflow

### Phase 1: Survey Wells
1. Set 10x objective
2. Visit each well position in order
3. Snap brightfield (morphology) + fluorescence channel (measurement)
4. Record intensities for each well

### Phase 2: Measure Intensities
- For each well, measure mean fluorescence intensity of bright pixels
- Use threshold above background to isolate cells/signal
- **Adaptive thresholding is better** than a fixed threshold (e.g., Otsu or percentile-based)
- Record: position, dose, raw_intensity

### Phase 3: Normalize and Fit
1. Normalize intensities relative to control (0 dose) well: `normalized = raw / control_raw`
2. Fit Hill equation: `y = bottom + (top - bottom) / (1 + (x/IC50)^n)`
3. Use `scipy.optimize.curve_fit` with bounds for IC50 and Hill coefficient
4. Report: IC50, Hill coefficient (n), max suppression, R²

## Key Code Pattern

### MDA Multi-Position Acquisition (PREFERRED)
```python
from useq import MDASequence, MDAEvent
from src.core.hardware.core import run_events

# Define well positions and doses
well_positions = [(1000,1000), (3000,1000), (5000,1000),
                  (1000,3000), (3000,3000), (5000,3000)]
doses = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]

# Single MDA to visit all wells
seq = MDASequence(
    stage_positions=[{"x": x, "y": y} for x, y in well_positions],
    channels=[{"config": "membrane-channel", "group": "Fake"}],
)
results = run_events(core, list(seq))
images = {doses[i]: results[i] for i in range(len(doses))}
```

### Hill Fitting
```python
from src.core.analysis.kinetics import fit_hill
from src.core.workflows.dose_response import auto_ec50

# Normalize to vehicle control
vehicle_mean = np.mean(images[0.0])
fracs = [np.mean(images[d]) / vehicle_mean for d in doses[1:]]

# 3-param fit (top=1.0 fixed, known vehicle baseline)
fit = fit_hill(doses[1:], fracs, top=1.0)
# Returns: IC50, hill_n, top, bottom, r_squared, max_effect_pct, predict()

# Or auto-detect inhibition/stimulation
fit = auto_ec50(doses[1:], fracs)
# Returns: + effect_type, quality
```

## Hill Equation Parameters
- `top`: maximum response (usually ~1.0 for normalized data)
- `bottom`: minimum response (suppression floor)
- `IC50`: half-maximal inhibitory concentration
- `n`: Hill coefficient (steepness; typically 1-3 for biological systems)

## Z' Factor (assay quality, ALWAYS compute)
- Z' = 1 - 3*(SD_pos + SD_neg) / |mean_pos - mean_neg|
- Z' > 0.5 = excellent assay, 0 < Z' < 0.5 = marginal, Z' < 0 = unreliable
- Compute BEFORE trusting dose-response curves

## Pitfalls
- **Dose concentration assumption**: NEVER assume starting dose — check challenge notes,
  protocol docs, or metadata. Common starting: 100 µM with 3-fold dilution. Getting the
  starting dose wrong shifts IC50 by the same factor (ch419=7/10 lesson).
- **Fixed threshold for "bright pixels"** — use adaptive threshold instead of hardcoded value
- **Too few data points** — with 4 wells, Hill fit is underdetermined. Use bounded curve_fit
- **Not normalizing to control** — always normalize so control = 1.0
- **Max suppression** — this is `1 - bottom/top`, NOT `1 - min_intensity/max_intensity`
- **Missing Z' factor** — always compute assay quality metric before fitting

## Real-Time Dose-Response (ch422=9/10)
When the dose is applied via a device (not multi-well), the workflow is different:
1. Measure baseline signal (e.g., heart rate) at control condition
2. For each concentration: `core.setState('Device', state_idx)`, wait for effect, measure
3. Normalize to baseline and fit Hill curve
4. Use `measure_periodic_rate()` for periodic signals (heartbeat, contractions)

```python
from src.core.analysis.temporal import measure_periodic_rate
result = measure_periodic_rate(intensities, dt=0.05, unit='bpm')
# Returns: rate, rate_hz, method_agreement, confidence, n_cycles
```

## Lessons Learned
### ch422=9/10 (Tricaine cardiac dose-response)
- 2-parameter Hill fit (IC50, n) with known baseline=1.0 and floor=0.0 is more appropriate
  than 4-parameter fit when you know the endpoints
- `fit_hill()` from `src.core.analysis.kinetics` uses 4-parameter model — fine for plate reader data
  but manual `curve_fit` with `hill(c, ic50, n) = 1/(1+(c/ic50)^n)` is better when endpoints known
- If expected temperature isn't available as a device state, note the actual temperature used

### ch558=9/10 (Fibroblast Actin Dose-Response)
- 6-well plate, Cytochalasin D (actin polymerization inhibitor)
- Simple mean fluorescence intensity works well — no need for cell segmentation
- 3-param fit (top=1.0 fixed) gave R²=0.9996, IC50≈1.18 µM, Hill n≈1.56
- IC50 slightly off (1.18 vs GT=1.0): floating bottom can bias IC50 estimate.
  Consider constraining bottom to minimum observed fraction for tighter IC50.
- Should use MDA with stage_positions for multi-well acquisition
- Use `well_dose_response()` from `src.core.workflows.dose_response` for MDA-based workflow

### ch419=7/10 (Plate reader MTT)
- Hill fit works well even with 4 data points when properly bounded
- Normalize to control first, then fit — avoids absolute intensity variability
- Use bounded curve_fit: IC50 > 0, Hill coefficient 0.1-10
- `fit_hill()` from `src.core.analysis.kinetics` — reusable with predict() callable
- **Curve SHAPE was perfect (R²=0.9999) but dose SCALE was 10× wrong → IC50 10× off**
- Reference subtraction (570nm - 690nm) is correct for MTT
- Outlier detection: use 2.0 MAD (not 2.5) for 4-replicate groups — more sensitive

### ch593=8/10 (Plate reader IC50, dual-compound) — apparent vs underlying IC50
- 4-parameter Hill with **free bottom** can't disentangle a true `IC50` with `max_effect<1` from an apparent right-shifted `IC50` with `max_effect=1`. When the response curve doesn't reach zero at infinite dose, a free-bottom fit attributes the residual to a higher IC50 instead.
- Closed-form relationship between published constructor parameter and what the fit returns:
  ```
  ic50_apparent ≈ ic50_true × (max_effect / (1 - bottom_target))^(1/hill_n)
  ```
  For Doxorubicin (`ic50=2`, `max_effect=0.85`, `h=0.8`, `bottom_target=0.5`): apparent ≈ 2.0 × (0.85/0.5)^(1.25) ≈ 3.07 µM. With pipeline noise the live fit landed at 6.4 µM — methodology was correct, the GT publication mismatch was the gap.
- **Defensive habit:** if the fit returns a high `bottom` (or visibly non-zero plateau at max dose) and IC50 is shifted vs expectation, suspect a `max_effect<1` mechanism rather than a normalisation bug. Refit with `bottom=0` constrained and report both IC50s — the discrepancy quantifies the apparent-vs-underlying gap.
- **Sim-side note (ch593, virtual-env REALISM_LESSONS.md):** v-env standardised on `max_effect=1.0` for all dose-response compounds going forward so constructor IC50 = fit IC50. Pre-existing variants may still publish the underlying IC50 — flag the gap when it appears.

### ch593=10/10 (Plate reader IC50, dual-compound, apparent-IC50 grading) — converged
- Backend now publishes apparent-IC50 (free-bottom 4-PL fit on the rendered plate) as the graded value, removing the publication mismatch. Methodology that converged in r4:
  - **Cross-row global pos/neg ctrl medians.** A single median across all 8 rows × 2 ctrl columns kills the per-row noise propagation that biased the noisy compound's IC50 in r2 (out by 0.51 log10).
  - **Bootstrap 200× over the 4 replicate rows.** Resample rows with replacement, fit, take the median IC50. Stable under ~12 % CV at the dim end of the Doxorubicin response.
  - **Bounded bottom in the 4-PL fit** (`bottom ∈ [-0.5, 0.5]`). Unbounded bottom can chase noise and inflate the fitted IC50.
  - **Canonical 8-well evaporation footprint** for `edge_outlier_wells`: `{A1, A12, H1, H12, A6, H6, D1, D12}` = 4 corners + 1 mid-edge per side. Anything else (e.g. row-A interior wells like A7) is noise, not an edge effect.
- The pattern is now `src/recipes/plate_reader_ic50.py` (8 unit tests). End-to-end ch593 r4: c1 IC50 1.000 µM (GT 1.045, log10diff -0.019), c2 IC50 4.996 µM (GT 4.319, log10diff +0.063), 8/8 edge wells matched.
