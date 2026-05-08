---
title: Deep learning-driven adaptive optics for single-molecule localization microscopy
authors: Peiyi Zhang, Donghan Ma, Xi Cheng, Andy P Tsai, Yu Tang, Hao-Cheng Gao, Li Fang, Cheng Bi, Gary E Landreth, Alexander A Chubykin, Fang Huang
year: 2023
venue: Nature Methods 20(11):1748-1758
doi: 10.1038/s41592-023-02029-0
url: https://www.nature.com/articles/s41592-023-02029-0
researched: 2026-04-25
---

## Abstract

The inhomogeneous refractive indices of biological tissues blur and distort single-molecule emission patterns generating image artifacts and decreasing the achievable resolution of single-molecule localization microscopy (SMLM). Conventional sensorless adaptive optics methods rely on iterative mirror changes and image-quality metrics. However, these metrics result in inconsistent metric responses and thus fundamentally limit their efficacy for aberration correction in tissues. To bypass iterative trial-then-evaluate processes, we developed deep learning-driven adaptive optics for SMLM to allow direct inference of wavefront distortion and near real-time compensation. Our trained deep neural network monitors the individual emission patterns from single-molecule experiments, infers their shared wavefront distortion, feeds the estimates through a dynamic filter and drives a deformable mirror to compensate sample-induced aberrations. We demonstrated that our method simultaneously estimates and compensates 28 wavefront deformation shapes and improves the resolution and fidelity of three-dimensional SMLM through >130-µm-thick brain tissue specimens.

## Smart microscopy principle

Sibling paper to [[Papers/Hu 2023]]: same archetype — collapse the sweep-and-fit sensorless-AO loop to a single CNN inference. The novel twist for SMLM is that **the wavefront sensor is the experiment itself**. In SMLM, every blinking event is a band-limited point source whose recorded PSF carries the full pupil-plane aberration; Zhang et al. train a CNN (`smNet`, conv blocks + residual blocks + PReLU) to read a batch of 20–100 single-molecule PSFs and regress the 28 native deformable-mirror modes that explain the *shared* aberration across them. No bias frames, no separate sensing modality, no calibration star — the localization data going into the reconstruction is the same data that closes the AO loop.

Two design decisions matter beyond the SMLM domain. First, **estimate native mirror modes, not Zernikes** — the network's output is in the basis the actuator can actually produce, eliminating the linearity assumption that bites every Zernike-projection-based corrector. Second, **a Kalman filter sits between inference and actuation**: each prediction is fused with the prior wavefront state weighted by per-prediction uncertainty (estimated from a confidence head), and three networks trained at different aberration scales are switched in based on the same uncertainty. This is the piece [[Papers/Hu 2023]] did not have — Hu's NN regresses one shot per call; Zhang's adds a closed-loop *state estimator* on top of the inference, which is what makes 3–20 sequential mirror updates converge stably (61% wavefront-error reduction per update) instead of oscillating.

The transferable lesson: when the experimental data is itself a wavefront probe (single emitters in SMLM, fiducial beads in any modality, structured-illumination peaks in SIM), the AO loop does not need a separate sensing phase. Build a regressor on the experiment's own PSFs, gate it through a Kalman filter against an actuator's native basis, and you have closed the loop on the data the experiment is already collecting.

## Implementation on pymmcore-plus

The hardware footprint is the same as [[Papers/Hu 2023]] — a Boston Micromachines Multi-3.5 deformable mirror at the objective pupil exposed as a `StateDevice` whose state vector is the actuator's native-mode amplitudes, plus a camera doing the SMLM frame stream. The control loop differs in two places: (a) inference batches over *single-molecule detections* harvested from each frame, not over engineered bias frames; (b) actuation is gated by a Kalman update, not directly applied.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

# Pre-trained smNet ensemble: three nets at different aberration scales.
# Contract: predict_modes(psf_stack) -> (modes_28d, uncertainty_28d)
predict_modes = load_smnet_ensemble("smnet_small.pt", "smnet_med.pt", "smnet_large.pt")
detect_emitters = load_psf_detector()    # crops 13x13 px ROIs around blinks

DM = "DeformableMirror"           # native-mode StateDevice
N_MODES = 28
BATCH_PSFS = 50                   # 20-100 in the paper
MAX_UPDATES = 20

def smlm_with_dl_ao(core, n_acq_frames=20000):
    """Run an SMLM acquisition with embedded DL-AO correction.
    The first ~MAX_UPDATES corrections happen during a 'pre-roll', after which
    the loop falls into pure SMLM acquisition with the converged wavefront held."""
    state = {
        "modes":   [0.0] * N_MODES,    # current mirror command (Kalman posterior mean)
        "P":       [1.0] * N_MODES,    # per-mode posterior variance (Kalman P diag)
        "psf_buf": [],                  # rolling PSF crops for the next inference
        "n_corrections": 0,
        "phase":   "correcting",        # 'correcting' -> 'imaging' once converged
    }

    def kalman_update(prior_mean, prior_var, obs, obs_var):
        K = [pv / (pv + ov) for pv, ov in zip(prior_var, obs_var)]
        post_mean = [m + k * (o - m) for m, k, o in zip(prior_mean, K, obs)]
        post_var  = [(1 - k) * pv for k, pv in zip(K, prior_var)]
        return post_mean, post_var

    def set_dm(modes):
        core.setProperty(DM, "ModeAmplitudes", ",".join(map(str, modes)))
        core.waitForDevice(DM)

    def on_frame(img, event, meta=None):
        if state["phase"] != "correcting":
            return
        # Harvest single-molecule PSFs for the next AO inference batch.
        crops = detect_emitters(img)
        state["psf_buf"].extend(crops)
        if len(state["psf_buf"]) >= BATCH_PSFS:
            obs_modes, obs_var = predict_modes(state["psf_buf"][:BATCH_PSFS])
            state["psf_buf"].clear()
            new_mean, new_var = kalman_update(state["modes"], state["P"],
                                              obs_modes, obs_var)
            state["modes"] = [-m for m in new_mean]   # apply as correction
            state["P"]     = new_var
            state["n_corrections"] += 1
            if (state["n_corrections"] >= MAX_UPDATES
                    or max(obs_var) < CONVERGENCE_THRESHOLD):
                state["phase"] = "imaging"

    def gen():
        for i in range(n_acq_frames):
            set_dm(state["modes"])
            yield MDAEvent(channel={"config": "STORM_647"},
                           metadata={"i": i, "phase": state["phase"]})

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/autofocus.py` — same scalar/sweep ↔ vector/inference mapping called out in [[Papers/Hu 2023]]. Zhang adds a Kalman-filter wrapper around the inference; the Hu pattern needs that wrapper too if we ever drive an AO actuator across noisy or non-stationary samples. A future `src/core/workflows/adaptive_optics.py` should expose `make_ao_state(predictor, kalman_R, kalman_Q, mode_basis)` and accept either bias-frame stacks (Hu) or single-molecule PSF crops (Zhang) as the inference input.
- `../../../src/core/hardware/core.py` — `run_events` + `on_frame` is the right substrate; the only new piece is a per-frame *emitter detector* that feeds the inference batch. On a real STORM rig that detector is the same one feeding the localiser, so the AO branch is essentially free.
- The Kalman wrapper itself is generic: any closed-loop optical-alignment routine that already converges (autofocus, drift correction, SIM-pattern phase) gets more robust with a per-axis posterior-variance estimate gating actuation. Worth promoting to `src/core/workflows/state_estimator.py` once a second consumer appears.

Calibration cost: the network is trained on millions of *simulated* PSFs generated by linearly combining experimentally measured per-mode mirror responses — i.e. one careful per-rig calibration of the DM's native-mode response measures the basis once, and the training set is then synthetic. Network ensembles for three aberration scales (small / medium / large) are switched in based on the confidence head. On a new rig: one DM-response calibration + one synthetic-training run; no per-sample retraining if the optical system is stable.

## Cited by

- [[Core/Strategies/Closed-loop autofocus]] — same archetype as [[Papers/Hu 2023]] and [[Papers/Pinkard 2019]]: replace a parameter sweep with a single inference. Adds the Kalman-filter wrapper that the autofocus / AO note should mention as the way to keep multi-step closed-loop corrections from oscillating under noisy SNR.
- Sibling to [[Papers/Hu 2023]] (universal-modality embedded NN AO) — Zhang's contribution beyond Hu is using the experiment's own PSFs as the wavefront probe and adding a Kalman-filter / multi-network-ensemble state estimator. Read together they bracket "what does the inference consume" (engineered bias frames vs. experimental PSFs) and "how is actuation gated" (direct apply vs. Kalman-fused).
- Continues the closed-loop-optical-alignment lineage of [[Papers/Royer 2016]] (continuous closed-loop optical alignment under live-imaging drift) into the SMLM regime.
