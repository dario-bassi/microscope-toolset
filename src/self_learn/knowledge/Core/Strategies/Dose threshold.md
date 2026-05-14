# Dose threshold

> **When to use:** When finding the minimum effective dose at which a binary biological response (wave nucleates, cell dies, reporter activates) first occurs — not a full Hill fit, just the threshold.

Find the **minimum effective dose** — the dose at which a binary
response crosses a threshold — without fitting the full dose-response
curve. Two functions in `self_learn.utils.dose_threshold`
(sprint #43, lifted from ch632 r1 → 10/10 / 264 inline lines):

- `sweep_threshold(*, doses, apply_dose, measure_response, response_threshold, reset=None)`
  — visit each dose once, record the binary `above`/`below` outcome,
  return a `ThresholdSweepResult` with `crossing_low` (highest sub),
  `crossing_high` (lowest super), and `threshold_estimate` (geometric
  mean of the bracket).
- `bisect_threshold(*, low, high, apply_dose, measure_response, response_threshold, n_iters, log_space=True, seed_trials=None)`
  — refine a known sub/super bracket via log-space bisection. Each
  iteration probes the geometric (or arithmetic) midpoint and
  tightens the bracket on the matching side.
- Companion: `geometric_doses(low, high, n)` — log-uniform grid for
  the initial sweep; the right grid when the threshold's order of
  magnitude is unknown.

## When to reach for it

Any **clinical minimum-effective-dose** problem where the brief
asks "what's the smallest X that produces effect Y?" rather than
"fit the full dose-response curve":

- SLM mask amplitude → wave nucleates / does not (ch632 origin).
- Laser mW → bleach observable / not.
- Drug concentration → kill / no kill.
- Temperature step → trigger heat-shock response or not.
- Voltage step → action potential or sub-threshold.

If the brief asks for a **Hill / sigmoid fit** instead, reach for
`analysis.population.dose_response_curve` — that's a
different shape (full curve, EC50, Hill coefficient).

## The canonical idiom (ch632 win shape)

```python
from self_learn.utils.dose_threshold import (
    geometric_doses, sweep_threshold, bisect_threshold,
)

# 1. Bracket with a coarse geometric sweep.
sweep = sweep_threshold(
    doses=geometric_doses(0.001, 0.04, 7),
    apply_dose=lambda v: arm_slm_disc(core, mask_value=v),
    measure_response=lambda: peak_dFF_in_disc(core, baseline_F0),
    response_threshold=0.5,        # nucleation iff ΔF/F > 0.5
    reset=lambda: reset_fhn_field(core),
)
assert sweep.bracketed, f"sweep too narrow: {sweep}"

# 2. Optionally refine via log-space bisection.
refined = bisect_threshold(
    low=sweep.crossing_low,
    high=sweep.crossing_high,
    apply_dose=...,
    measure_response=...,
    response_threshold=0.5,
    n_iters=8,
    seed_trials=sweep.trials,      # keeps coarse-sweep history in result
)

answer = {
    "min_effective_mask_value": refined.min_effective_dose(),
    "threshold_estimate": refined.threshold_estimate,
    "n_sweep_points": len(refined.trials),
}
```

The submit shape often wants `min_effective_dose` (the empirically
observed smallest super-threshold dose), not the geometric-mean
estimate — `result.min_effective_dose()` returns the conservative
answer; `result.threshold_estimate` returns the bracket midpoint.

## Why log-uniform first, then bisect

Log-uniform sweep gives equal log-resolution above and below the
threshold without committing to a guess. Once the bracket is found,
bisection converges to log-distance ~0.03 in 8 iterations — well
inside the typical "log10 distance < 0.30" tolerance graders use
for biological dose questions.

## Sister utilities

`dose_threshold` rounds out the dose-aware control trio:

| Utility               | Question                                    |
|-----------------------|---------------------------------------------|
| `rate_limited_drive`  | "How many ON pulses to reach the setpoint?" |
| `pulsed_schedule`     | "How to track a multi-waypoint trajectory?" |
| `dose_threshold`      | "What's the minimum dose to cross threshold?" |

Same dose-agnostic shape across all three: caller supplies setter +
reader callbacks; the utility owns the scheduling logic. Composes
into any biology — SLM, perfusion, temperature, drug — without
knowing the actuator name.

## Tested on

- ch632 r1 = 10/10 (FHN wave-nucleation SLM mask amplitude); the
  264-inline-line solve is the source code for the recipe shape.
- `tests/test_dose_threshold.py` — 20 synthetic-step-responder
  tests covering sweep, bisect, all-sub/all-super edge cases, and
  the canonical sweep→bisect composition.

## See also

- `self_learn.utils.sensorless_ao: sweep_state_device` — same
  primitive shape, ranks by **scalar metric** instead of binary
  outcome. Use when you have a continuous quality metric (Brenner,
  intensity), `dose_threshold` when you have a hard yes/no.
- `self_learn.workflows.dose_response` — for full-curve dose-response
  with Hill fits.
