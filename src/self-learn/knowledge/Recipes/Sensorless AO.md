# Sensorless AO

**Assumes:** a categorical state device (`DeformableMirror` / `Modality` /
`Objective`) where each state encodes a discrete wavefront correction
(typically `flat` + 5 Zernike axes × ±0.5 rad-rms = 11 states), plus a
Brenner-friendly fluorescent sample. Paired with
`src/recipes/sensorless_ao.py` (`solve_dm_sweep` — tested on ch607, ch608,
ch610). Channel default `DAPI`.

## Sample types

This recipe applies whenever the question is **which discrete wavefront
correction maximises sharpness?**:

- DM-based widefield AO with a fixed action set (the sim's 11-state DM).
- Pre-defined SLM mode libraries.
- Categorical objective-correction-collar dials.
- Any sensorless-AO regime where the action space is *quantised* (no
  continuous gradient, no inner Gauss-Newton step). Iterative
  walking — DAOSM-style "apply best move, re-measure, walk toward
  residual" — composes on top of this recipe; the kernel below is
  one-shot.

## Workflow

```python
from src.recipes.sensorless_ao import solve_dm_sweep

result = solve_dm_sweep(
    core,
    device="DeformableMirror",
    channel="DAPI",
    metric="brenner",
    want_residual_axes=True,    # ch610 diagnostic; off by default
    presence_margin=0.03,       # 3% above flat = "axis is in baseline"
)
# result["best_state_label"]      → "flat" | "defocus_-0.5" | ...
# result["best_state_index"]      → int (use with core.setState)
# result["sharpness_values"]      → {label: float} curve
# result["margin_over_runner_up"] → relative margin vs second-best
# result["baseline_axes"]         → e.g. ["defocus", "astig_x"]   (with want_residual_axes)
# result["residual_axes"]         → axes the picked-best did NOT address
```

The recipe composes 5 building blocks from `src/core/utils/sensorless_ao.py`,
each separately exposed:

1. `sweep_state_device(core, device, channel, metric)` — sweep all N
   states, snap at each, return `{label: sharpness}`. Restores the
   pre-sweep state on exit so callers don't observe a side-effect.
2. `default_axis_for_label(label)` — parse `"defocus_+0.5"` → `("defocus", +1)`.
3. `argmax_state(scores)` — returns `(best_label, best_score, margin)`
   with margin = `(best - runner_up) / runner_up`.
4. `infer_baseline_axes(scores, margin=0.03)` — for each axis, check
   whether either of its ± states exceeds flat by ≥`margin`. Returns
   the list of axes whose baseline is non-zero.
5. `residual_axis(baseline_axes, best_axis)` — set difference. The
   uncancelled component when no single discrete state addresses
   everything.

## The three patterns (ch607 / ch608 / ch610 arc)

1. **DM-sweep + argmax** (ch607 r1 → 10/10): sweep all 11 states, compute
   Brenner gradient at each, pick argmax. On a defocus-dominated backend,
   defocus shows ~−31% Brenner vs flat; astigmatism ~intermediate; coma
   ~−8% (near the noise floor — coma is a centroid shift more than a
   sharpness loss). When sharpness must *rank* aberrations rather than
   just find the optimum, defocus dominates.

2. **Don't hardcode "answer = flat"** (ch608 r1 → 10/10): the sim
   pre-loaded a sample-induced defocus +0.5 baseline so the DM state that
   wins is the one that *cancels* it (`defocus_-0.5`), not flat. The
   argmax-driven sweep handles this without code changes — the lesson is
   the *reflex*: let argmax find whichever state has the highest
   sharpness on the *current* sample. The 11-state curve is itself a
   signature: `defocus_-0.5` wins big AND `defocus_+0.5` is among the
   worst → sample is biased toward defocus +.

3. **Per-axis-presence inference** (ch610 → 10/10): when the sim bakes in
   TWO baseline axes (defocus +0.4 + astig_x +0.3) but the DM only
   carries single-axis ±0.5 corrections, the closest single state wins
   (defocus_-0.5 here) but the sweep curve also reveals which *other*
   axes had non-zero baseline. Diagnostic: each axis's best ± state
   exceeds flat by ≥3% Brenner ⟺ that axis is in the baseline. Recovers
   `[defocus, astig_x]` exactly. Residual = `baseline_axes \ {best_axis}`
   = `[astig_x]` — the uncancelled component.

   Grader endorsement: *"BIG win on inference: an axis is in the baseline
   iff its best ± state exceeds flat by >3% Brenner — that's a sound
   a-priori test for which axes have non-zero baseline contribution.
   Identified [defocus, astig_x] from the curve — exactly the
   ground-truth baseline."*

## When NOT to use

- **ch609-style off-axis-LED single-shot AF** is a *Z* sweep with
  `phase_cross_correlation`, not a state-device sweep. Different device
  (focus stage, not DM), different metric (image shift, not Brenner),
  different geometry (calibrate around focus — see
  [[Core/Strategies/Closed-loop autofocus]] § Validated on ch609).
  `scratch/solve_609_r3.py` preserves that pattern; do **not** retrofit
  into `solve_dm_sweep`.
- **DAOSM-style iterative walking** — apply best move, re-measure, walk
  toward residual. `solve_dm_sweep` is one-shot. For iterative regimes
  use it as the inner loop, with the residual axis driving the next
  iteration's action choice. (Out of scope for this recipe.)
- **Continuous-action AO** (analog DM voltages, full Gauss-Newton modal
  optimisation) — the argmax shape doesn't apply. Use Hu 2023 / Zhang
  2023 ML-AO or a real modal optimiser.

## Validated on

- **ch607 — flat-wins baseline** (counter=63): 11-state DM sweep, flat
  scored 25.45 Brenner vs next-best 23.42 → 8.7% margin. Submitted
  `best_state_index=0`. **10/10**.
- **ch608 — sample-bias cancellation** (counter=68): same code, sim
  pre-loaded defocus + baseline. `defocus_-0.5` won by ~12% margin,
  `defocus_+0.5` at the bottom of the curve. Submitted state 2. **10/10**.
- **ch610 — multi-axis baseline + residual inference** (counter=85): same
  code with `want_residual_axes=True`. Best=`defocus_-0.5` at 22.5%
  margin. Per-axis-presence test recovered `baseline_axes=[defocus, astig_x]`
  exactly; residual=`[astig_x]`. **10/10**.
- **ch650 — uncalibrated DM (label-trust forbidden)** (2026-04-28): same
  recipe shape, different baseline (astig_y dominant + coma_x
  secondary), method-summary gate forbids citing the device label as
  the *basis* of inference. Sweep argmax = state 6 (8.5% margin over
  state 10 / coma_y_-0.5; tolerance ≥7%). The index → axis-pair
  lookup is correctly described as a *post-hoc finishing step*; the
  decision basis is the Brenner argmax over the sweep curve, not the
  device's pre-coded labels. Validates the recipe in the DAOSM /
  Royer 2016 first-contact regime where the DM-channel-to-Zernike
  mapping isn't yet calibrated. **10/10 r1**.

## Cross-archetype reuse

`sweep_state_device` has been reused 3 times across distinct
archetypes — the primitive is generic over device name + scoring
metric, not tied to AO. ch631 grader: *"sweep_state_device + scoring
metric is now a generic primitive that has been reused 3x. Save the
pattern: any sweep over a state device with a per-state metric
reduces to this one abstraction shape."*

| Archetype | Device | Metric | Decision shape | Challenge |
|-----------|--------|--------|----------------|-----------|
| Wavefront-correction | `DeformableMirror` | Brenner | argmax → flat | ch607-610 |
| Scalar-per-channel ranking | `Channel` (via `BrightnessRanking`) | mean / percentile | rank by ε×Φ or kx | ch617 / ch619 / ch627 |
| Sub-diffraction depletion | `STEDDepletion` | Brenner | endpoint contract (off=lowest) + ratio | ch631 |
| FHN binary threshold | (no state device — sweep over SLM mask values) | per-trial peak ΔF/F → bool | smallest mask value where wave_nucleated=True | ch632 |

The [[Core/Strategies/Operating the solve loop]] § Same-loop REUSABLE table
tracks the cross-archetype reuses; this recipe note is the
canonical entry-point.

## See also

- [[Core/Strategies/Closed-loop autofocus]] — the parent strategy
  family. Per-axis-presence inference and argmax are the
  *quantised-action* siblings of continuous parabolic-peak interp; the
  two-estimator cross-check pattern in § ch609 r3 generalises across
  both.
- [[Core/Strategies/Closed-loop state device]] — generic state-device
  pattern this recipe instantiates with `DeformableMirror` as the state.
- [[Recipes/Event-driven modality switch]] — sibling state-device recipe; differs
  in *which* device (Objective / Modality vs DM) and *what* the decision
  optimises (where to point vs what shape to apply).
- [[Papers/Hu 2023]] —
  generalises this to continuous Zernike-vector inference; the recipe
  here is the categorical/discrete reduction.
- [[Papers/Zhang 2023]] — sibling in the
  SMLM-AO domain; uses the experiment's own data (single-molecule PSFs)
  as the wavefront probe — no separate sensing phase, unlike our explicit
  sweep.
- [[Papers/Royer 2016]] — continuous
  multi-axis closed-loop AO predecessor; same survey-then-pick shape on
  optical alignment rather than wavefront correction.
