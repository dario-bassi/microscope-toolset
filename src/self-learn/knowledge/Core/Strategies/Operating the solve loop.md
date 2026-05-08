# Operating the solve loop

A meta-strategy note distilling what consistently works across the
post-restart sprint #25–#34 arc (counter ≈ 87→131; 14 of 15 scored
solves at 10/10). Content-level lessons; sibling of
[[Core/Strategies/Session retrospective]] which covers the `session_audit`
machinery instead.

## The five reflexes

### 1. First-contact composition

On every challenge, before any acquisition decision, snap once and
run [[Core/Strategies/Auto recipe selection]] / `auto_recipe(image, core)`
to pick a recipe + parameter prefill from the image alone.

```python
core.snapImage()
preview = core.getImage()
suggestion = auto_recipe(preview, core=core)
```

Scenario-level signals (dose budgets, per-cell state) used to come
from a sibling `bridge_preflight` / `first_contact` probe over the
bridge RPC surface, but that surface was closed 2026-04-27. Read
the brief and the camera frames; do not invent a replacement that
peeks at sim-internal state. See [[Core/Approach/Transferability contract]].

### 2. Forward-model when the brief cites a formula

[[Core/Concepts/Chromatic aberration]] § How to
calibrate captures the rule: when the brief points at an analytic
expression (achromat lateral_coef × Δλ × radius for LCA, ε × Φ for
brightness, exp(-rate · t) for gene induction, exp(-x/λ) for
diffusion gradients), prefer **forward-model + spot-check via
measurement** over pairwise matching / Hungarian / cross-correlation.
ch603 r1 (5/10) → r3 (10/10) was the canonical recovery: matching
silently collapsed at the FOV edge where chromatic shift approached
voronoi cell-spacing scales; the registry-driven analytic recovered
without that failure mode. Also captured in
`feedback_forward_model_over_matching.md`.

The pattern is **predict → measure → verify**:

1. Phase A: read registry / brief / per-channel metadata exposed via
   standard `Camera.<prop>` reads — produce predicted ratios + ranking.
2. Phase B: snap each channel / timepoint at matched conditions —
   reduce per-FOV / per-cell to scalars.
3. `compare_predictions(pred, meas, ratio_tol=...)` — verify
   ranking match + per-channel rel-err.

Lives in [`src.core.utils.fluorophore_brightness`](../../../src/core/utils/fluorophore_brightness.py)
for ε × Φ; trivially reused for `bleach_kx` (ch619), and works
for any per-fluorophore scalar.

### 3. Same-loop REUSABLE extraction

When a 10/10 grade explicitly says "REUSABLE: this pattern
transfers to…", lift the win shape into a sprint within the same
session loop. Sprints #25–#35 carried the streak this way:

| Win    | Extracted as                          | First reuse                |
|--------|---------------------------------------|----------------------------|
| ch607-610 | `sensorless_ao` (sprint #25)        | (covers itself)            |
| ch611  | `dose_budget` (sprint #26 — REMOVED 2026-04-27, bridge RPC closed) | ch612 r1 used `set_pixel_mask` + ch620 r1 used `dose_report` |
| ch613  | `slm_masks` (sprint #28) + `targeted_gene_induction` (REMOVED 2026-04-27, bridge RPC closed) | **ch621 r1** |
| ch614  | `spectral_leak` + `spectral_unmix` (sprint #31, parallel) | ch616 (queued) |
| ch615  | `fft_peak` (sprint #32, parallel)     | (covers itself)            |
| ch617  | `fluorophore_brightness` (sprint #33, parallel) | **ch619 r1** |
| ch613/617/618 | `bridge_state` (sprint #34) + `bridge_preflight.has_per_cell_state` (sprint #36) — both REMOVED 2026-04-27 (bridge RPC closed) | **ch621 r1 + ch622 r1** |
| ch621  | `rate_limited_drive` (sprint #35, parallel) | future rate-limited drive (photoconv, ablation, FUCCI) |
| ch622  | `intracellular_drug_conc_b` canonical CellState field (sprint #34 refresh) + dual-field-filter feedback memory | future multi-reservoir / chemotaxis / ratio imaging |
| ch623  | calibrate-σ-from-empirical extraction (sprint TBD) + import-paths-in-brief feedback memory | future spectral / dichroic / FRET overlap-bounded mechanics |
| ch624  | `pulsed_schedule` (sprint #37, parallel) — closed-form per-segment ON+OFF model + plan_trajectory | future fluorophore activation / dose ramping / cell-cycle entrainment |
| ch625  | asymmetric-predicate generalization in `feedback_dual_field_filter.md` | future 1-target-1-distractor / therapeutic AND-NOT toxicity / activate-and-not-inhibit |
| ch626  | `predict_n_more_exposure` twin (sprint #38) | future combined-budget (min(bleach, exposure)) / temperature / illumination-power budgets |
| ch627  | (no new utility — `BrightnessRanking` from sprint #33 reused for kx) | future scalar-per-channel rankings (lifetime, ε, Φ, brightness×kx) |
| ch628  | (no new utility — `pulsed_schedule.predict_segment_end` reused for two-phase mask differentiation) | future N-group ramping, ensemble-spread MPC |
| ch629/630 | (no new utility — 4-utility composition validated) + `feedback_snap_vs_control_budget.md` | future stim-step budget / combined-budget MPC |
| ch631  | `sweep_state_device` from sprint #25 reused for STED (3rd cross-archetype reuse) + `feedback_state_device_sweep_generic.md` | future quantitative FWHM / STED + budget composition |
| ch632  | `sweep_state_device` 4th reuse — sweep+ranking generalised from scalar-metric to BINARY OUTCOME (FHN threshold detection) | future min-effective-drug-conc / min-exposure-for-AF / min-SLM-for-ChR2 |
| ch633/634 | (no new utility — local-maxima MAP2 detection + count-from-positions on disclosed disc) | grader flagged 20x-FOV-too-small as a tracked scenario-design pitfall; future opto scenarios will use 10x |
| ch635  | "decompose cum_dose into per-dimension components" (snap×N_snap + stim_step×N_stim + ...) becomes a planning primitive | future combined-budget MPC where agent identifies binding ceiling |
| ch636  | binding-constraint identification + per-dimension decomposition realised on a 2-ceiling MPC (cum = stim + snap, residual at machine epsilon) | future mid-run replan when binding flips |
| ch637  | per-cell drop-out mask scheduling for continuum spatial law (each cell exits mask once N_i reached); ceil(N_i) compensates decay-during-post-stim drift | extracted as `per_cell_drop_out` (sprint #39 — REMOVED 2026-04-27, bridge RPC closed); first reuse on **ch642 r1** |
| ch638  | min(H_dim1, H_dim2, ..., H_dimN) compound primitive over predict_n_more (bleach) + predict_n_more_exposure (exposure) — twin utilities co-exercised for the first time. Closes idle-ping #3 | future N-dim conserved-quantity budgets (temperature ms, illumination power) reduce to same min-horizon shape |
| ch639  | per-PHASE REMAINING obligation decomposition (binding flips when phase transitions consume one component faster); 3-deep nested composition over ch634+ch636+ch637. Closes idle-ping #1 | future multi-phase MPC where dose rates differ per phase: binding moves with dominant remaining-obligation component |
| ch642  | "Fit-then-control" pattern: system-identification phase (numpy.polyfit on 3 disclosed calibration samples) BEFORE control execution. Strategy note `Fit-then-control.md` survives; the `per_cell_drop_out` execution primitive was REMOVED 2026-04-27 (bridge RPC closed). Closes idle-ping #2 | future sinusoidal/exponential/Hill law inference with parametric family disclosed but coefficients hidden |
| ch643  | distribution-trajectory MPC (unimodal at t=5 → bimodal at t=10) via two-phase SLM mask switch + ROUND-N empirical rate calibration; cross-product of ch628 two-phase-mask × ch624 pulsed-stim-per-phase. Closes idle-ping #6. Grader explicitly cited bidirectional feedback-memory loop (agent applied wound_healing pitfall captured earlier) | future trimodal trajectory / arbitrary-N continuous waypoints / mid-run REPLAN combined with shape change |
| ch646 + ch649 | `sim_protocol` recipe (sprint #42) — 3-phase demod + FFT-orientation legs; auto_recipe(brief=) routes Archetype: sim with mode auto-picked from submit-shape. Bonus side-fix: extract_submit_shape now accepts `Submission:` / `Submission shape:` headers | future SR-SIM reconstruction / non-square-grid SIM / SIM at different magnifications |
| ch632  | `dose_threshold` utility (sprint #43) — geometric_doses + sweep_threshold + bisect_threshold; dose-agnostic via apply_dose + measure_response callbacks. Third leg of dose-aware-control trio (rate_limited_drive setpoint + pulsed_schedule trajectory + dose_threshold cut-off) | future laser-mW bleach onset / drug-conc cell-death / temperature heat-shock / voltage AP threshold |

Captured in `feedback_grader_reusable_to_sprint.md`. Pattern:

1. Plan agent in background on the extraction (spawn with
   `run_in_background=true`).
2. Solve the next challenge inline while Plan runs.
3. When Plan returns, execute its file-by-file diff plan.

The parallel-extraction precedent (sprints #31, #32, #33) keeps
the loop responsive while still doing serious infrastructure work.

### 4. Sign-check before submit

Print first-vs-last frame, sign of derived rates/ratios,
boundary-condition assertions BEFORE `submit_solution`. If any
check fails, **do not submit** — iterate offline. The recent
solve scripts (ch603 r2, ch614, ch617, ch619) all have explicit
`assert` calls between the answer build and the submit call;
ch603 r2's sign-check correctly aborted a degenerate run that
would have shipped at 5/10 again. Captured in
`feedback_sign_check_before_submit.md`.

### 5. Read-and-filter for state-exposed challenges (HISTORICAL)

Up to 2026-04-27, several challenges (ch613, ch617, ch618, ch619)
were solvable as `read bridge state → filter by predicate → submit`
with no acquisition cost, via `bridge.get_cell_state` and the
`src.core.utils.bridge_state` typed wrapper. **That surface is now
closed.** Per-cell quantities have to be re-derived from camera
frames + standard segmentation. The reflex still applies in shape
(filter on a predicate over per-cell measurements) — only the
source of the measurements changes. See
[[Core/Approach/Transferability contract]].

## When the queue is dry

After 30+ wakeups of pure idle work, productivity drops. Anti-
patterns to avoid:

- **Don't write tests 15× in a row** (`feedback_repetitive_cycles.md`).
  Rotate to audits, knowledge cross-links, cross-cutting refactors.
- **Don't speculatively pre-build for hypothetical follow-ons.**
  ch611's `take_to_budget(adaptive=True)` was built into the
  recipe but no challenge has exercised it yet. The grader's
  REUSABLE flags are reliable signals; speculative code without
  one is over-fitting.
- **Don't skip the sign-check rule when "the math is obvious."**
  ch603 r1 auto-submitted past sign-check despite my own durable
  rule because the script didn't `assert`. Result: 5/10. r2 + r3
  both `assert`-gated — 10/10.
- **Do ping virtual-env when truly idle** with a list of specific
  REUSABLE follow-ons that map to existing utilities. The
  counter=128 ping with 5 named candidates yielded ch619 within
  ~2 wakeups.

## What this *isn't*

- **Not an implementation guide.** See
  [[Core/Strategies/Auto recipe selection]], [[Core/Strategies/Closed-loop autofocus]],
  [[Core/Strategies/Closed-loop state device]], [[Core/Strategies/Gentle imaging]] for the
  workflow-level patterns this note generalizes over.
- **Not a paper survey.** See `Papers candidates.md`
  for the verified-citation pipeline.
- **Not the audit machinery.** See [[Core/Strategies/Session retrospective]] for
  `session_audit.py` usage.

## Validation

The five reflexes above were each validated on multiple challenges
this session — see the per-feedback-memory files for the specific
challenges and grader quotes. The win-shape REUSABLE → sprint
pattern is the meta-loop the rest of the workflow runs inside.
