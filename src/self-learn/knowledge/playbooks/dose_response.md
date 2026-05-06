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
```python
from src.workflows.templates import dose_response_experiment, DoseResponseResult
from src.analysis.plot_reading import fit_hill, hill_equation

# Or manual approach:
from scipy.optimize import curve_fit
popt, _ = curve_fit(hill_equation, doses, normalized,
                     p0=[1.0, 0.0, median_dose, 1.0],
                     bounds=([0, 0, 0, 0.1], [2, 2, max_dose*10, 10]))
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
from src.analysis.temporal import measure_periodic_rate
result = measure_periodic_rate(intensities, dt=0.05, unit='bpm')
# Returns: rate, rate_hz, method_agreement, confidence, n_cycles
```

## Lessons Learned
### ch422=9/10 (Tricaine cardiac dose-response)
- 2-parameter Hill fit (IC50, n) with known baseline=1.0 and floor=0.0 is more appropriate
  than 4-parameter fit when you know the endpoints
- `fit_hill()` from `src.analysis.kinetics` uses 4-parameter model — fine for plate reader data
  but manual `curve_fit` with `hill(c, ic50, n) = 1/(1+(c/ic50)^n)` is better when endpoints known
- If expected temperature isn't available as a device state, note the actual temperature used

### ch419=7/10 (Plate reader MTT)
- Hill fit works well even with 4 data points when properly bounded
- Normalize to control first, then fit — avoids absolute intensity variability
- Use bounded curve_fit: IC50 > 0, Hill coefficient 0.1-10
- `fit_hill()` from `src.analysis.kinetics` — reusable with predict() callable
- **Curve SHAPE was perfect (R²=0.9999) but dose SCALE was 10× wrong → IC50 10× off**
- Reference subtraction (570nm - 690nm) is correct for MTT
- Outlier detection: use 2.0 MAD (not 2.5) for 4-replicate groups — more sensitive
