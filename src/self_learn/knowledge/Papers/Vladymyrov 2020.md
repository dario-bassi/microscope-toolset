---
title: "VivoFollow 2: Distortion-Free Multiphoton Intravital Imaging"
authors: Mykhailo Vladymyrov, Neda Haghayegh Jahromi, Elisa Kaba, Britta Engelhardt, Akitaka Ariga
year: 2020
venue: Frontiers in Physics 7:222
doi: 10.3389/fphy.2019.00222
url: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2019.00222/full
researched: 2026-04-25
---

## Abstract

Multiphoton intravital imaging of living mice is constrained by periodic motion artefacts (heartbeat, breathing) and slower tissue drift, which together distort individual frames *within* a Z-stack and shift the imaged volume *across* a time-lapse, respectively. VivoFollow 2 closes both loops in real time: a IR-optical sensor ("TrigViFo") attached to the ventilator tracks the respiratory phase at 860 Hz; an initial calibration acquisition (32 frames at 8 phases) measures the displacement of anatomical landmarks (CX3CR1-GFP myeloid cells, VE-cadherin-GFP endothelial junctions, second-harmonic collagen, intravascular TRITC-dextran) per phase and fits a periodic motion profile by spline; subsequent raw frames are resampled on-the-fly from this profile to recover undistorted optical sectioning. Concurrently, a 3D pattern-matching loop (CUDA, on a Raspberry Pi 3 B+ runtime) measures slow drift relative to a reference stack and re-targets the imaged volume between time points. Demonstrated on cervical-spinal-cord preparations through a cranial window: residual displacement reduced from ~15 µm to 0.4–0.8 µm after up to five iterative refinement cycles. The system extends the predecessor VivoFollow (Vladymyrov 2016, *J Immunol Methods*), which corrected slow drift only, to the intra-frame distortion regime that breathing/heartbeat introduces in slow-scan multiphoton microscopy of internal organs.

## Smart microscopy principle

VivoFollow 2 is the canonical paper for **closed-loop drift + distortion correction during slow-scan in vivo microscopy**. The contribution sits on three axes that distinguish it from post-hoc registration tools (e.g. Fast4DReg) and from generic AutoPilot-class alignment ([[Papers/Royer 2016]]):

1. **Two coupled correction timescales, both online.** Slow tissue drift is corrected between time points by GPU pattern matching against a reference 3D stack — same archetype as AutoPilot, but implemented for raster point-scan multiphoton. Fast periodic distortion (breathing-driven, 0.5–4 Hz) operates *within* a single frame: each scan line is acquired at a known respiratory phase, and the resampling lookup that turns raw scan lines into a Cartesian frame uses a per-phase displacement spline that the system measures in a 32-frame calibration prelude. This intra-frame correction is impossible for whole-frame post-hoc registration — the distortion is built into the raster itself.

2. **Hardware-synchronised respiratory triggering.** The TrigViFo box (Raspberry Pi 3 B+ + IR LED–photodiode + Adafruit ADS1115 ADC at 860 Hz, exposed over TCP/RPC) reads the ventilator piston position and emits per-phase trigger signals to the microscope acquisition. Phase-locked sampling collapses the high-dimensional motion problem to a low-dimensional periodic correction. The same trick generalises to any quasi-periodic physiological perturbation (heartbeat, peristalsis, ciliary flow) — the trigger source changes, the algorithm doesn't.

3. **Iterative on-instrument refinement.** Up to five passes of resample-and-re-fit reduce residual displacement an order of magnitude. The loop converges fast enough to run during the same imaging session, so the operator does not lose the prep window to off-line analysis.

For smart-microscopy practice the transferable idea is: when the sample's motion is known to be periodic, *don't fight it with frame-rate increases* — acquire in phase with it and resample in software. The dose budget is unchanged; the distortion is removed; and the same scan-line schedule that ran on a still sample now runs on a breathing one. This is a sibling pattern to [[Papers/Royer 2016]] (geometry is non-stationary across hours; re-optimise periodically) and to the focus-maintenance loops in [[Core/Strategies/Closed-loop autofocus]], but at the *intra-frame* timescale that those tools don't address.

The system is **substrate-coupled to multiphoton**: the slow-scan, depth-resolved nature of 2P/3P intravital is precisely what makes per-scan-line phase locking work. On a fast widefield camera, motion blur within an exposure replaces line-by-line phase distortion — a different correction problem.

## Implementation on pymmcore-plus

A pymmcore-plus rig that needs this loop has three components to wire up: (1) an external trigger source that exposes respiratory/heartbeat phase as a numeric signal, (2) a per-frame metadata channel that stamps each `MDAEvent` with the phase at which it was acquired, and (3) a calibration-then-correction state machine that runs as a generator + `on_frame` pair, identical in shape to the AutoPilot loop in [[Papers/Royer 2016]] but with phase as the controlled axis.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
from src.core.workflows.autofocus import focus_metric  # or a custom pattern-match score

CALIBRATION_FRAMES = 32       # paper uses 32 frames over 8 phases
N_PHASES = 8

def vivofollow_loop(core, n_frames=600, channel="GFP"):
    """Phase-locked acquisition with per-phase displacement correction.

    During calibration, image at each respiratory phase and measure landmark
    displacement against the phase-0 reference. Build a per-phase spline.
    During the main run, stamp each frame's metadata with the current phase
    and apply the per-phase resample lookup before saving. A coarse drift
    correction runs every N main frames against a reference 3D stack.
    """
    state = {
        "phase_displacements": {},   # phase -> (dx, dy, dz)
        "reference_stack": None,
        "drift": (0.0, 0.0, 0.0),
    }

    def gen():
        # Phase 1 — calibration
        for k in range(CALIBRATION_FRAMES):
            phase = k % N_PHASES
            yield MDAEvent(
                channel={"config": channel},
                metadata={"role": "calibration", "phase": phase},
            )

        # Phase 2 — main acquisition, phase-locked
        for i in range(n_frames):
            phase = current_respiratory_phase()    # from TrigViFo-equivalent
            dx, dy, dz = state["drift"]
            yield MDAEvent(
                channel={"config": channel},
                x_pos=core.getXPosition() - dx,
                y_pos=core.getYPosition() - dy,
                z_pos=core.getZPosition() - dz,
                metadata={"role": "main", "i": i, "phase": phase},
            )

    cal_buf = {}
    def on_frame(img, event, meta=None):
        md = event.metadata or {}
        if md.get("role") == "calibration":
            cal_buf.setdefault(md["phase"], []).append(img)
            if all(len(v) >= CALIBRATION_FRAMES // N_PHASES for v in cal_buf.values()):
                # Fit per-phase displacement spline against phase-0 mean
                state["phase_displacements"] = fit_phase_spline(cal_buf)
        elif md.get("role") == "main":
            # 1) per-phase resample for intra-frame distortion
            corrected = resample_by_phase(img, state["phase_displacements"][md["phase"]])
            # 2) periodic slow-drift update against reference stack
            if md["i"] % 50 == 0 and state["reference_stack"] is not None:
                state["drift"] = pattern_match_3d(corrected, state["reference_stack"])

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/drift.py` — already provides `phase_correlate`, `fft_cross_correlate`, `centroid_drift`, `measure_drift_timelapse`, `correct_drift`. These cover the *slow* drift loop. The *intra-frame* phase-locked resample is missing and would belong here as a new `phase_locked_resample(img, phase, lookup)` helper plus a `fit_phase_spline(frames_by_phase)` builder.
- `../../../src/core/hardware/core.py` — `run_events(core, events, on_frame=…)` handles the calibration-then-main two-phase generator without modification.
- The trigger source — TrigViFo is a Raspberry-Pi peripheral exposed over TCP/RPC. The pymmcore-plus equivalent is to add a `TriggerSource` device adapter (or a thin `signaler` reading a TTL line into `core.getProperty("Trigger", "Phase")`) so the generator can sample `current_respiratory_phase()` cheaply.
- Budgeting: the calibration prelude is ~32 frames of overhead; on a bleach-sensitive prep, run it on a sacrificial low-intensity channel first and reuse the displacement spline for the main fluorescence channel.

Differences from the paper to keep in mind:
- The TrigViFo hardware path assumes a ventilator with a moving piston. For free-breathing or anaesthesia setups with no mechanical pump, replace with an ECG/respiration belt → ADC; the phase-locking algorithm is unchanged.
- The 3D pattern matcher in the paper is CUDA-based on a Raspberry Pi runtime. On a workstation `src/core/hardware/drift.py:fft_cross_correlate` plus a per-Z slice loop is sufficient for slow drift; only the per-line correction is latency-critical.
- The paper demonstrates ~15 µm → 0.4–0.8 µm residual displacement over up to 5 iterative cycles. Convergence is sample-dependent; budget the iteration count from the residual, not from a fixed cycle count.
- This loop is **mode-specific to slow-scan modalities** (2P/3P/confocal point scan). On a widefield camera the intra-frame distortion turns into intra-exposure motion blur, which is a different problem (shorter exposure + faster trigger, not phase-locked resample).

## Cited by

- [[Core/Strategies/Closed-loop autofocus]] — VivoFollow 2 generalises continuous-alignment maintenance from focus to *any* periodic-motion-driven distortion; sibling to AutoPilot at the intra-frame timescale.
- [[Core/Strategies/Adaptive acquisition]] — phase-locked sampling is an acquisition-time correction strategy distinct from the trigger-based / score-based strategies elsewhere in this library; the controlled axis is *when in the cardiac/respiratory cycle the next scan line happens*, not *what to image next*.
