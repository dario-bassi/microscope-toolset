# Pulsed schedule trajectory

Drive an ensemble along a multi-waypoint REFERENCE TRAJECTORY under
a saturating rate law (rate) + first-order decay. Three functions
in [`src.core.utils.pulsed_schedule`](../../../src/core/utils/pulsed_schedule.py)
(sprint #37, lifted from ch624 r1 → 10/10):

- `predict_segment_end(prev, *, n_on, n_off, rate, decay, max=1.0)`
  — closed-form per-segment forward model:
  `expr_after_ON = max - (max - prev)·exp(-rate·n_on)`, then
  `expr_after_OFF = expr_after_ON · exp(-decay·n_off)`.
- `plan_n_on_for_waypoint(prev, target_band, *, segment_len, rate,
  decay, max=1.0)` — inverse: smallest integer N_on in
  `[0, segment_len]` whose predicted end-of-segment lands inside
  `target_band`; tiebreak toward band midpoint. Raises with
  `closest_n_on` + `closest_end` if unreachable.
- `plan_trajectory(waypoint_bands, *, segment_len, rate, decay,
  max=1.0, prev=0.0)` — orchestrator. Walks bands sequentially,
  propagating predicted end-of-segment as the prev for the next
  segment. Returns a frozen `PulsedSchedule` dataclass with
  `n_on_per_segment` + `predicted_trajectory` + `predicted_end_state`.

## When to reach for it

The brief gives a list of **waypoint bands** (not a single floor or
single band):

- Lugagne 2024 deep-MPC for gene expression (ch624 origin).
- FUCCI staging across G1 → S → G2 → M (drive a fraction of cells
  through cell-cycle checkpoints in sequence).
- Controlled drug-conc curves with washout (perfusion + delay
  intervals; the OFF segments are the wash phases).
- Photoconversion ramps (target a specific conversion fraction
  curve, not just a floor).
- Any constant-decay rate-limited system where the agent must
  OVERSHOOT early to satisfy a later setpoint, then let decay
  carry the system into the next target band.

The grader's verbatim composition rule (ch624):

> "The pulsed-stim multi-segment shape generalizes: fluorophore
> activation curves, dose ramping with stop-and-cool intervals,
> cell-cycle entrainment via SLM-pulsed checkpoints, any constant-
> decay system where the agent must overshoot early to satisfy a
> later setpoint. Save the closed-form per-segment forward model
> as a planning primitive — composes with any rate-limited drive."

## When NOT to use it

- **Single setpoint band** — use [[Core/Strategies/Rate-limited drive]]'s
  `drive_to_band` (early-stop is correct because there's no later
  waypoint to reach).
- **Max-out drives** (just exceed a floor — ch613 / ch614 territory)
  — historically `induce_targets` from the (deleted) targeted_gene_induction
  recipe handled the open-loop fixed-N saturation case. Both the recipe and
  its `bridge.dose_report` dependency were removed 2026-04-27 (bridge RPC
  closed; see [[Core/Approach/Transferability contract]]). For a max-out
  drive on a future challenge, compose `slm_masks` + a fixed-N MDA
  generator over standard device-property reads — no bridge RPC.
- **Closed-loop with mid-segment replan** (sample mid-trajectory
  and adjust remaining schedule based on noisy reality). Different
  shape — needs per-segment callbacks. Deferred to a future ch625-
  class scenario; do not pre-build.
- **Per-cell ensemble-spread MPC** (drive cells to a target
  *distribution* rather than a mean). ch624 grader flagged this as
  the natural next abstraction; needs per-cell SLM mask
  phase-shifting. Different shape — different module.

## See also

- [[Core/Strategies/Rate-limited drive]] — single-setpoint sibling primitive
  (sprint #35); use `predict_n_steps` for the open-loop horizon and
  `drive_to_band` for the closed-loop early-stop. The grader on
  ch621 flagged the multi-waypoint extension as "next week"; this
  note is that extension.
- ~~[[Recipes/Targeted gene induction]]~~ — REMOVED 2026-04-27 with
  the rest of the bridge RPC surface. The historical contract was
  open-loop fixed-N saturation via `induce_targets`; pulsed_schedule's
  closed-form per-segment forward model survives because it only uses
  camera frames + standard device-property reads.
- [[Core/Strategies/Closed-loop state device]] — the generic discrete-state-device
  pattern this composes with when the actuator is an SLM mask.
- [[Core/Strategies/Operating the solve loop]] § Forward-model when the brief cites
  a formula — meta-rule that says reach for closed-form math when
  the brief discloses rate / decay / segment_len / waypoints.
