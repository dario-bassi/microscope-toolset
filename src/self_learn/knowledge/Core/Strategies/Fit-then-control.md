# Fit-then-control

A two-phase pattern for control problems where the **target law is
unknown** but the **law family** is disclosed. Sister to
[[Core/Strategies/Rate-limited drive]] (forward-model + early-stop) and
[[Core/Strategies/Pulsed schedule trajectory]] (multi-waypoint trajectory MPC); the
new abstraction here is the **system-identification phase** that
runs *before* the control loop.

## Shape

```
Phase 1: SYSTEM ID
    1a. read disclosed (input, output) calibration samples from brief
    1b. fit law-family parameters: numpy.polyfit / scipy.curve_fit
    1c. validate: predicted-vs-disclosed residual ≈ machine epsilon
        (exact-determined fit) or below tolerance (over-determined)

Phase 2: CONTROL
    2a. apply fitted law to all (or unknown) inputs → per-cell targets
    2b. closed-form per-cell N: N_i = -ln(1 - target_i) / rate
        (or whatever the kinetic model dictates)
    2c. execute via per-cell drop-out scheduling, MPC, or open-loop
        (image-derivation pending — bridge RPC surface closed 2026-04-27;
        per-cell scheduling now has to come from image-segmented
        per-cell tracking)
```

## Validated on

- **ch642 — quadratic per-cell reference law inference** (counter=196,
  10/10). Brief disclosed 3 (cell_index, cx, target) calibration
  pairs in scoring text + family `target = a·u² + b·u + c, u = cx/W`.
  `numpy.polyfit(u_cal, target_cal, 2)` with 3 points exactly
  determined the quadratic — fitted `(a, b, c) = (0.5999, 0.0001,
  0.2000)` against canonical `(0.6, 0.0, 0.2)`, all 3 errors at
  0.0001 (tolerance 0.05). Per-cell drop-out scheduling with ROUND-N
  rounding (backend-specific — wound_healing needs ROUND, ch637's
  VoronoiGeneInduction needed CEIL) → 5/5 test cells in tolerance.
  Grader: *"REUSABLE — Fit-then-control
  pattern: pre-flight system identification phase before MPC
  execution. Generalizes to any control problem where the target
  law is unknown but the law family is."*

## When to use

The trigger is the brief shape:
1. A **family** is disclosed (linear / quadratic / exponential / Hill / sinusoidal).
2. The **coefficients are hidden**.
3. The brief discloses **at least N calibration samples** for an N-coefficient family.

If all three hold, fit-then-control. If only #1 + #2 (no calibration samples), this isn't fit-then-control — it's an estimation problem requiring active probing.

## Family fitters

| Family                    | Coefs | Method                                            |
|---------------------------|-------|---------------------------------------------------|
| Polynomial degree N       | N+1   | `numpy.polyfit(x, y, N)` returns `[a_N, …, a_0]`. |
| Linear                    | 2     | Special case of polyfit deg 1.                    |
| Exponential `a·exp(b·x)+c`| 3     | `scipy.optimize.curve_fit(lambda x,a,b,c: a*np.exp(b*x)+c, x, y)`. |
| Hill `V·xⁿ/(Kⁿ+xⁿ)`       | 3     | `analysis.plot_reading.fit_hill`.        |
| Sinusoidal `A·sin(2πfx+φ)+c` | 4 | `scipy.optimize.curve_fit` with bounds.           |

Exact-determined (N samples for N coefficients) gives a unique
solution. Over-determined (N+1+ samples) gives a least-squares fit;
expect non-zero residual on the calibration points themselves.

## Validation step (always)

After fitting, **reproduce the calibration points**:

```python
for u_cal, t_cal in calibration_samples:
    t_pred = fitted_law(u_cal)
    assert abs(t_pred - t_cal) < ftol, f"residual {abs(t_pred - t_cal)} ≥ {ftol}"
```

For exact-determined fits, residual ≤ machine epsilon (`1e-10`). For
over-determined, residual ≤ tolerance. ch642 r1 verified all 3
calibration points reproduced at residual=0.0 — that's the
exact-determination signature.

## Composition

The execution phase composes with whatever per-cell control primitive
the kinetics dictate:

- **Continuous spatial law** + **gene induction** → per-cell drop-out
  scheduling. Each cell exits the SLM mask after its `N_i` stim
  steps. (Execution primitive image-derivation pending — bridge RPC
  surface closed 2026-04-27; ground-truth per-cell counters are no
  longer addressable, so the scheduler now has to read post-stim
  per-cell intensity from the camera and decide drop-out from that.)
- **Single-cell setpoint** → forward-model + early-stop
  ([[Core/Strategies/Rate-limited drive]]).
- **Multi-waypoint trajectory** → closed-form per-segment planner
  ([[Core/Strategies/Pulsed schedule trajectory]]).
- **Combined-budget MPC** with phase transitions → per-PHASE
  remaining-obligation decomposition (ch639 binding-flip pattern).

The fit-then-control split is orthogonal to all of them — it just
*provides the targets* the control phase consumes.

## When NOT to use

- **Brief discloses the formula entirely** (no hidden coefficients).
  Just apply the formula. ch637 (linear ramp `target = 0.2 +
  0.6·cx/W`) was the disclosed-formula sister.
- **No calibration samples** (or fewer than N for an N-coef family).
  Have to actively probe instead — incompatible with this pattern.
- **The "law" is non-parametric** (e.g., a 2D heatmap of values).
  Use a lookup table or interpolation, not a fitted law.

## See also

- [[Core/Strategies/Rate-limited drive]] — sister: forward-model + early-stop, but
  with the rate parameter known (not fitted).
- [[Core/Strategies/Pulsed schedule trajectory]] — sister: multi-waypoint trajectory
  MPC, with the kinetics known.
- [[Core/Strategies/Operating the solve loop]] § Forward-model — the broader rule
  this instantiates: when the brief cites a formula, prefer the
  formula over matching. Fit-then-control is the same shape with
  inferred coefficients.
## Future variants

- **Sinusoidal target law** — same shape with a 4-coef family.
  `scipy.optimize.curve_fit` with bounds; need ≥4 calibration
  samples or use prior bounds for under-determined fit.
- **Active probing** when the brief doesn't disclose enough
  samples — use the agent's snap budget to test 1-3 conditions
  itself, then fit. (Image-derived dose tracking required — bridge
  RPC surface closed 2026-04-27.)
- **Time-varying laws** (the same family but coefficients drift).
  Re-fit at each step using a sliding window of the most recent
  observations.
