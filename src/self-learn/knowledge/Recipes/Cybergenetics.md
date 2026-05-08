# Cybergenetics — per-cell SLM-feedback control

Closing the loop from live single-cell measurements to per-cell light
delivery, so each cell in a heterogeneous population is driven to a
shared target reporter level.

## When to use

- A single-cell reporter (mScarlet, mNeonGreen, GCaMP) coupled to a
  light-driven actuator (EL222, ChR2-derived TFs, optoLat).
- Heterogeneous responses across the population — some cells respond
  faster to the same dose than others, so an open-loop schedule
  underdoses some cells and overdoses others.
- A dose-actuator with at least one decade of dynamic range (DMD,
  galvo-scanned laser, SLM with intensity modulation per pixel).

If the population is homogeneous and the dose-response is well
characterised, an open-loop schedule (single uniform stim → measure)
gives equivalent average behaviour at far less complexity. Closed-loop
shines when *individual-cell variance* matters.

## Recipe layout

`src/recipes/cybergenetic_per_cell_control.py` (12 unit tests). The
recipe exports:

- `CellState` — per-cell controller state.
- `update_integrator(cell, measured, target, gain_i, dose_min, dose_max)`
  — one PI step with anti-windup.
- `build_per_cell_mask(centroids, doses, slm_size, radius)` — per-cell
  SLM circles via `np.maximum`-blend.
- `step_population(state, image, centroids, cfg)` — one closed-loop
  step (detect → match → measure → integrator → mask).
- `feedback_generator(cfg, state, channel, ...)` — lazy MDAEvent
  generator wrapping the loop in useq events.
- `make_feedback_callback(cfg, state, detect_centroids=...)` — the
  `on_frame` factory.
- `run_per_cell_feedback(core, cfg, channel, detect_centroids, ...)`
  — end-to-end driver.

## Lesson stack

### 1. Integral-with-anti-windup is the right pilot controller

Rullan 2018 (`[[Papers/Rullan 2018]]`)
demonstrates `I(t_k) = K_I · Σ e(t_n)` with single-cell yeast yields
~3× variance collapse vs open-loop. PID is overkill for a slow
biological actuator — derivative on top of fluorescent shot noise
amplifies high-frequency noise into the SLM mask.

**Anti-windup is essential**: when the integrator saturates dose at
`dose_max`, *don't* keep accumulating error in the same direction —
otherwise the integrator will overshoot when the cell finally
catches up. Allow accumulation when error sign flips so the
integrator releases.

### 2. Hungarian matching, not greedy NN, for per-cell tracking

Cells drift between frames; a greedy nearest-neighbour matcher swaps
identities under crossings (a known pitfall — see
`[[Core/Strategies/Connectivity mapping]]`). Use
`scipy.optimize.linear_sum_assignment` against a squared-distance
matrix with a `max_distance` gate so impossible matches are rejected.

### 3. Track-loss-grace before freezing the integrator

Cells dip below detection threshold for a frame or two then come
back. Don't immediately discard the integrator — that loses the
controller's accumulated context. Freeze the integrator (no
accumulation, no light delivery) for `track_loss_grace` consecutive
unmatched frames; only then mark as `edge_or_lost`. If >50 % of the
baseline cells are lost simultaneously, abort: something has gone
wrong with the FOV (drift, focus, bleach catastrophe).

### 4. Lazy generator + `on_frame`, not threading

useq's `MDAEvent` generator pattern lets you yield events whose
`slm_image` is computed from the previous frame's `on_frame`
callback. The mask lives in the recipe's `state` dict; the generator
reads `state['next_mask']` at each yield. This is the pattern
documented in `agent/CLAUDE.md` "Adaptive/closed-loop" example and
in `[[Core/Strategies/Feedback control]]`. Don't reach for
threading or async — useq's iterator protocol composes cleanly with
`run_events` and works identically on local CMMCorePlus and the
remote pymmcore-proxy.

### 5. Per-cell mask intensity, not temporal multiplexing (for v1)

To deliver per-cell doses you can either (a) modulate per-cell SLM
*pixel intensity* (each cell's circle drawn at its own dose) or (b)
temporal multiplexing (binary mask, vary dwell time across
sub-cycles). Option (a) is simpler and survives proxy latency; (b)
needs >1 SLM update per control step and may not honour `min_start_time`
pacing. The `[[Core/Strategies/SLM optogenetics]]` separation
principle (geometry vs dose) is satisfied at the recipe boundary —
geometry comes from the detected centroids, dose comes from the
integrator state — even though the SLM mask combines them.

## Composes from core

- `src/core/hardware/core.py`: `run_events`, `make_slm_circle`.
- `src/core/detection/cells.py` (or any caller-supplied
  `detect_centroids` callable): cell centres per frame.
- `scipy.optimize.linear_sum_assignment`: Hungarian matcher.

## Test harness without a live server

`tests/test_recipe_cybergenetic_per_cell_control.py` ships a
`FakePlant` one-pole simulator: each cell has its own
`α, β` so `I_{t+1} = α·I_t + β·dose_t + noise`. Drive 40 cycles via
`step_population` and assert the population RMSE drops by ≥ 50 %.
This validates the controller's stability and convergence without
any pymmcore proxy or useq engine.

## Sample-tuned burst window for σ × mean

The cardio recipe-validation (ch601 r1, 4/10) revealed that the
default 25-frame burst was implicitly tuned to GCaMP / FHN calcium
dynamics. For excitable media with slower wave periods or full-
depolarization wavefront brightness (cardio AP, cAMP / Dictyostelium
spirals), the burst window has to span ≥ 4 wave cycles before σ ×
mean's constitutive-vs-transient asymmetry develops enough to
separate the pacemaker from wave-front pixels. Rule of thumb: set
``n_burst ≈ 4 × period_in_dt_units``. Calcium FHN: 32 frames. Cardio
AP: 60 frames. Dictyostelium cAMP spirals: 120+ frames.

The recipe's ``modality_switch_pipeline`` exposes ``n_burst`` as a
caller-supplied hyper-parameter; new excitable-tissue samples should
calibrate it from a brief preview before the production scout.

## See also

- [[Core/Strategies/Feedback control]] — generic closed-loop
  patterns, sample equilibration, sample-vs-microscope timescales.
- [[Core/Strategies/SLM optogenetics]] — geometry-vs-dose
  separation principle.
- [[Papers/Rullan 2018]]
- [[Papers/Hinderling 2025]]
- [[Papers/Lugagne 2024]] — the deep-MPC
  successor; swap of the control law only.
