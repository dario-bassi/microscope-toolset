# Strategies index

Quick cross-reference: strategy → decision criterion → primary module.
All files are in `Core/Strategies/`. Read the linked file for full detail.

| Strategy | When to use | Primary module |
|---|---|---|
| [[Adaptive acquisition]] | Survey-then-zoom for ROI discovery, or patrol for rare events | `self_learn.workflows.batch.adaptive_survey_mda` |
| [[Auto recipe selection]] | Unknown sample — classify it before writing experiment code | `self_learn.utils.auto_recipe` |
| [[Axis sweep alignment]] | Find argmax state of a categorical device (filter, dichroic, objective) | `self_learn.utils.axis_sweep` |
| [[Closed-loop autofocus]] | Sample drifts in Z during timelapse — active per-frame focus correction | `self_learn.workflows.autofocus` |
| [[Closed-loop state device]] | Discrete state device (valve, electrode) drives a response to be measured and fed back | `self_learn.workflows.stage_tracking` |
| [[Connectivity mapping]] | Map functional neural circuits via SLM stimulation + calcium imaging | `self_learn.workflows.optogenetics` |
| [[Dose threshold]] | Find the minimum effective dose for a binary biological response | `self_learn.utils.dose_threshold` |
| [[Feedback control]] | Each frame's measurement determines the next frame's hardware state | general pattern — `run_events` + `on_frame` generator |
| [[Fit-then-control]] | Response law family disclosed but coefficients hidden — fit first, then control | `numpy.polyfit` / `scipy.optimize.curve_fit` |
| [[Gentle imaging]] | Phototoxicity or photobleaching is a concern — minimize photon dose | principle — BF-first, `compute_snr`, exposure budget |
| [[Imaging parameter optimization]] | Images too dim, saturated, or channel not yet calibrated | `self_learn.analysis.intensity.compute_snr` |
| [[Measurement methodology]] | Need statistical validity or correct methodology for the biological question | principle — checklists in this file |
| [[Multi-position survey]] | Sample spans more than one FOV — tile and catalogue across a grid | `self_learn.workflows.batch.tile_and_analyze` |
| [[Multi-scale morphometry]] | Population morphometry requiring low-mag survey then high-mag tiles | `self_learn.analysis.morphometry` |
| [[Multi-well comparison]] | Compare phenotype or signal across wells relative to a control | `useq.MDASequence` with `stage_positions` |
| [[Multichannel scan]] | Multiple channels × multiple positions in one coordinated MDA sequence | `self_learn.workflows.batch.multichannel_scan` |
| [[Per-cell measurement from frames]] | Per-cell quantities (position, intensity, shape) from camera frames | `self_learn.detection.cells.detect_cells` |
| [[Physical-unit thresholds]] | Detection thresholds must stay biologically valid across magnifications | principle — `core.getPixelSizeUm()` for conversion |
| [[Pulsed schedule trajectory]] | Drive ensemble along multiple waypoint bands under saturating rate law | `self_learn.utils.pulsed_schedule` |
| [[Rate-limited drive]] | Drive population to a single target band under monotonic saturating rate law | `self_learn.utils.rate_limited_drive` |
| [[Reinforcement learning in acquisition]] | Actuator has hysteresis, reward is discrimination, or exploration is the goal | literature reference — Schmidt 2023 / Komatsuzaki 2022 |
| [[Sample-class auto-detection]] | Sample type unknown — classify before selecting any workflow | `self_learn.utils.sample_classifier` |
| [[Segmentation backend]] | Choose between Cellpose/StarDist and sigma thresholding for instance segmentation | `self_learn.detection.segmentation_backend` |
| [[Simultaneous SLM targeting]] | Multiple SLM targets and sequential firing is failing | `make_slm_circle` — compose discs into one mask |
| [[SLM optogenetics]] | Photostimulation, closed-loop tracking, photoconversion, or dose-response via SLM | `self_learn.hardware.core.apply_slm`, `make_slm_circle` |
| [[Smart microscopy substrates]] | Choosing the computational framework or substrate for a closed-loop pipeline | literature reference — Pycro-Manager / ImSwitch / Arkitekt |
| [[Timelapse design]] | Design frame rate, duration, exposure budget from the biological timescale | `useq.MDASequence` with `time_plan` |
| [[Wave propagation]] | Analyze calcium waves, cAMP waves, cardiac activation, or any biological wavefront | `self_learn.analysis.temporal`, `self_learn.analysis.optical_mapping` |
