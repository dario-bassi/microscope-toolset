# Rate-limited drive

> **When to use:** When driving a population to a single target band under a monotonic saturating rate law — use `predict_n_steps` for the open-loop burst then `drive_to_band` for closed-loop fine control.

Drive a population to a SETPOINT BAND under a monotonic-saturating
rate law. Two functions in `self_learn.utils.rate_limited_drive`
(sprint #35, lifted from ch621 r1 → 10/10):

- `predict_n_steps(target, *, rate, max=1.0)` — closed-form forward
  model. Solves `target = max·(1 − exp(−rate·N))` for N and returns
  `ceil(N)`. Predict against the **band floor**, not the midpoint —
  that gives the smallest N that REACHES the band, leaving the
  closed-loop wrap to absorb residual model error from decay terms
  or rate uncertainty.
- `drive_to_band(measure_fn, step_fn, *, band, slack, max_steps)` —
  closed-loop wrap. Reads pre-state, then calls `step_fn()` and
  `measure_fn()` in a loop, stopping the moment the measurement
  enters `[band[0], band[0] + slack]`. Returns a `DriveResult` with
  `n_steps`, `final_value`, `trace`, `stopped_reason ∈ {"band",
  "max_steps"}`.

## When to reach for it

Any monotonic, saturating, rate-limited drive where the brief gives
a rate constant and a SETPOINT BAND (not just a "≥" floor):

- Gene-expression induction (ch621 origin).
- Photoconversion (PA-GFP / mEos / Dendra2 conversion fraction → setpoint).
- Ablation damage accumulation (target lethal-dose without overshoot).
- FUCCI phase progression (drive a fraction of cells into S-phase).
- Drug-conc ramps when the actuator has saturating delivery kinetics.

The grader's verbatim composition rule (ch621):

> "The forward-model + early-stop pattern composes with any rate-limited
> drive: photoconversion, ablation damage accumulation, FUCCI phase
> progression. Same shape:
>   `N_predict = ln(1/(1−target/max)) / rate`
>   `early_stop` when measurement enters `[target_lo, target_lo+slack]`
> Save this as a planning primitive."

## When NOT to use it

- **Multi-waypoint reference trajectories** (drive cells along a curve:
  0.3 @ t=5, 0.6 @ t=10, 0.8 @ t=15). The early-stop semantics break
  mid-trajectory because you have to overshoot intermediate setpoints.
  Different shape — pulsed stim with rest periods. See
  [[Core/Strategies/Pulsed schedule trajectory]] (sprint #37, lifted from ch624 r1).
- **MAX-out drives** (just exceed a floor — ch613 / ch614 territory).
  Historically `induce_targets(n_stim_steps=N)` from `targeted_gene_induction`
  shipped this case; the recipe and its bridge RPC dependency were
  removed 2026-04-27 (see [[Core/Approach/Transferability contract]]).
  A future MAX-out challenge would compose `slm_masks` + a fixed-N MDA
  generator over standard device-property reads.
- **Non-monotonic drives** (oscillators, bistable switches). Different
  control problem; this primitive will silently early-stop on the first
  upcrossing of the band.

## See also

- [[Core/Strategies/Closed-loop state device]] — the generic discrete-state-device wrap
  this composes with when the actuator is an SLM mask.
- [[Core/Strategies/Operating the solve loop]] § Forward-model when the brief cites
  a formula — the meta-rule that says reach for closed-form math when
  the brief has it.
