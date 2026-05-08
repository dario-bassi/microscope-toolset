# `strategies/` — workflow applications of smart microscopy

Literature-backed smart-microscopy applications, each implemented (or sketched) as a pymmcore-plus workflow. The pattern: name the application, cite the paper or class of experiments it comes from, sketch the control loop, link to the production implementation in `src/core/workflows/` or `src/core/acquisition/`.

A strategy note should be recognisable to a biologist who reads the paper: the same control loop, just written in Python. If a note here reads like an internal code walkthrough, it should probably move to `src/` docstrings instead.

## Acquisition paradigms

- [[Core/Strategies/Adaptive acquisition]] — generator-driven acquisition that reacts to what it just measured (Almada et al. 2019, 2022).
- [[Core/Strategies/Feedback control]] — closed-loop experiments where analysis output drives the next event.
- [[Core/Strategies/Rate-limited drive]] — forward-model + early-stop primitive for setpoint-band drives (gene induction, photoconversion, ablation, FUCCI).
- [[Core/Strategies/Pulsed schedule trajectory]] — closed-form per-segment planner for multi-waypoint trajectories under rate-limited drive + decay (gene-expression MPC, FUCCI staging, washout curves).
- [[Core/Strategies/Dose threshold]] — geometric sweep + log-space bisection for binary-cut-off / minimum-effective-dose detection; dose-agnostic over SLM amplitude / laser mW / drug conc / temperature / voltage. Third leg of dose-aware-control trio.
- [[Core/Strategies/Fit-then-control]] — system-identification phase BEFORE control execution: brief discloses law family + N calibration samples, agent fits coefficients via polyfit/curve_fit, then drives per-cell targets (ch642 quadratic law inference).
- [[Core/Strategies/Timelapse design]] — backward-designing a timelapse from the kinetics you want to measure.
- [[Core/Approach/MDA solve pattern]] — canonical `MDASequence + run_events` for fixed acquisitions (lives in `Approach/`; cited here for completeness).

## Spatial coverage

- [[Core/Strategies/Multi-position survey]] — how to tile a sample so nothing is missed and nothing is imaged twice.
- [[Core/Strategies/Multi-scale morphometry]] — survey-at-low-mag → zoom-at-high-mag for shape measurements.
- [[Core/Strategies/Multi-well comparison]] — within-plate comparison patterns (treatment/control, dose series).
- [[Core/Strategies/Multichannel scan]] — MDA-first multi-position + multi-channel scans with a single sequence.

## Closed-loop / intervention

- [[Core/Strategies/Closed-loop autofocus]] — Z-drift correction during a timelapse without burning snaps.
- [[Core/Strategies/Closed-loop state device]] — generic pattern: discrete state device sets a condition, sample responds, you close the loop.
- [[Core/Strategies/Wave propagation]] — measuring wavefront speed from a triggered excitation.
- [[Core/Strategies/SLM optogenetics]] — SLM-based patterned stimulation for phototaxis, photoconversion, ChR2 activation.
- [[Core/Strategies/Simultaneous SLM targeting]] — fire all N targets in one multi-spot mask; sequential firing wastes sim-time.
- [[Core/Strategies/Connectivity mapping]] — neural circuit mapping via SLM stimulation + calcium response.

## Measurement methodology

- [[Core/Strategies/Gentle imaging]] — minimise photon dose while maximising information per frame.
- [[Core/Strategies/Measurement methodology]] — distinguish detection accuracy from measurement accuracy (Jensen's inequality, calibration).
- [[Core/Strategies/Imaging parameter optimization]] — exposure/gain/Z-step tuning; when to recalibrate.
- [[Core/Strategies/Physical-unit thresholds]] — parameterise detection in µm and µm², not pixels; survives magnification changes.
- [[Core/Strategies/Per-cell measurement from frames]] — segment + per-ROI extract + predicate filter; the post-2026-04-27 substitute for the historical `bridge.get_cell_state` read-and-filter pattern.
- [[Core/Strategies/Segmentation backend]] — when to reach for Cellpose / StarDist vs sigma-tuned thresholding; the pluggable abstraction in `src/core/detection/segmentation_backend.py`.
- [[Core/Strategies/Auto recipe selection]] — first-contact image → recipe pick + parameter prefill, via `src/core/utils/auto_recipe.py`.
- [[Core/Strategies/Sample-class auto-detection]] — what `auto_recipe` does upstream: image → discrete sample class via cheap features (sprint #17) + the foundation-model-encoder progression (Yu 2024 PLIP, Morgado 2024).
- [[Core/Strategies/Smart microscopy substrates]] — the open-source frameworks that hold a closed loop together (Pycro-Manager, ImSwitch, Arkitekt, Navigate, EAP4EMSIG); how to pick one.
- [[Core/Strategies/Reinforcement learning in acquisition]] — when the actuator has hysteresis, the reward is discrimination not reconstruction, or there is no extrinsic reward (Schmidt 2023, Komatsuzaki 2022, Pathak 2017).
- [[Core/Strategies/Session retrospective]] — how to read `session_audit` output: overall distribution + brittle-recipe flags + recent-vs-historical trend, so you know when to refresh a knowledge note vs harden a recipe.
- [[Core/Strategies/Operating the solve loop]] — content-level meta-strategy: the five reflexes (first-contact composition, forward-model when brief cites a formula, same-loop REUSABLE extraction, sign-check before submit, read-and-filter for state-exposed challenges) distilled from the post-restart sprint #25-#34 arc + 14-of-15 win streak.

## See also

- `../concepts/Concepts index.md` — the physics and API these strategies assume.
- `../approach/Approach index.md` — meta-layer for picking a strategy.
- `../pitfalls/Pitfalls index.md` — failure modes; every strategy note should cite the pitfall it avoids.
- `../../src/core/workflows/` — the production implementations.

## References

Paper citations live in [[Papers index]]. The paradigms that underpin this tier (event-driven acquisition, automated adaptive acquisition, IsoView-style feedback-controlled light-sheet, AutoPilot long-term live imaging) are tracked as topic candidates in [[Papers candidates]] until verified.
