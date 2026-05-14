---
title: Chromatic aberration correction employing reinforcement learning
authors: Katharina Schmidt, Ning Guo, Wenjie Wang, Juergen Czarske, Nektarios Koukourakis
year: 2023
venue: Optics Express 31(10):16133-16147
doi: 10.1364/OE.487045
url: https://opg.optica.org/oe/abstract.cfm?uri=oe-31-10-16133
researched: 2026-04-25
---

## Abstract

In fluorescence microscopy a multitude of labels are used that bind to different structures of biological samples. These often require excitation at different wavelengths and lead to different emission wavelengths. The presence of different wavelengths can induce chromatic aberrations both in the optical system and induced by the sample, which shifts the focal positions in a wavelength-dependent manner and decreases the spatial resolution. Here we present the correction of chromatic aberrations by using an electrical tunable achromatic lens driven by reinforcement learning. The adaptive achromatic lens (AAL) is composed of two fluid-filled lens chambers with different optical oils, sealed off with deformable glass membranes. By steering four input voltages of piezoelectric actuators, the lens can be tuned to compensate the chromatic aberrations of the chosen wavelengths. Because the input voltages and piezoelectric actuators induce hysteresis, the control problem is non-trivial; reinforcement learning (a recurrent PPO agent with an LSTM actor and a single critic) is used to learn the control policy. The system corrects chromatic shifts up to 2200 µm and focal-position shifts up to 4000 µm. We demonstrate the approach on biomedical samples (human thyroid tissue), where the AAL successfully corrects both system-induced and sample-induced chromatic aberrations and improves imaging quality.

## Smart microscopy principle

Sensorless adaptive optics treats wavefront correction as an optimisation over a parameter vector — Hu 2023 collapses that loop with a feed-forward CNN that **regresses the correction in one shot**. Schmidt et al. attack a different corner of the same parameter-vector problem: a four-voltage piezoelectric actuator with **strong hysteresis**, where the same target shape is reached by different voltage trajectories depending on history. A regression network can't solve this — the mapping from "desired correction" to "voltage to apply now" depends on the previous voltage. Reinforcement learning is the natural fit: the policy gets the current image-quality signal *and recent action history* (the LSTM in the actor) and outputs the next voltage delta. Each acquisition is a step in the MDP; the reward is image-sharpness across the multi-colour stack.

The smart-microscopy contribution is that **closed-loop control of a hysteretic actuator belongs in the acquisition loop, not in a calibration table**. Lookup tables for piezos drift; per-experiment recalibration burns the sample. An online RL controller treats the actuator as a black-box environment and re-optimises every time the AAL state, sample-induced aberrations, or wavelength set changes. Compared with Durand 2018 (RL/bandit over STED illumination parameters where the *reward* is the imaging quality / damage trade-off itself), Schmidt 2023 puts the RL agent one layer below — it controls a *physical actuator state* whose target is set by an outer image-quality objective, much as Komatsuzaki 2022's bandit chooses where to scan and Durand's bandit chooses how to illuminate.

The transferable idea: **any actuator with hysteresis or non-monotonic response (piezo, deformable membrane, MEMS mirror, focus screw, correction collar) is a candidate for a small RL controller embedded in the closed-loop autofocus / AO / alignment pipeline**, replacing the open-loop sweep + lookup-table approach that silently fails when the device drifts. The RL agent's tiny state (recent actions + current quality metric) is cheap enough to update mid-acquisition.

## Implementation on pymmcore-plus

On a pymmcore-plus rig the AAL is exposed as a `StateDevice` whose state vector is the four piezo voltages (or as four separate float properties under one device label). The RL policy lives outside the acquisition generator; on each `on_frame` callback the per-channel image-quality signal is computed, the policy proposes a voltage delta, and the next `MDAEvent` carries the new voltages via `MDAEvent.properties`.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
from src.core.workflows.autofocus import focus_metric

# Pre-trained recurrent-PPO policy. Contract:
#   policy.act(obs, lstm_state) -> (delta_voltages: np.ndarray[4], new_lstm_state)
# obs = concat([current_voltages, sharpness_per_channel, wavelength_one_hot]).
policy = load_aal_policy("aal_recurrent_ppo.pt")

AAL = "AdaptiveAchromaticLens"          # 4 float properties: V1..V4
CHANNELS = ["DAPI", "GFP", "Alexa647"]   # multi-colour stack

def chromatic_correction_timelapse(core, n_frames=200, dt_s=10.0,
                                   correct_every=5):
    """Multi-colour timelapse with RL-driven chromatic correction.

    Every `correct_every` frames we acquire one frame per channel under the
    current AAL voltages, score per-channel sharpness, and let the recurrent-
    PPO policy update the four voltages for subsequent acquisitions.
    """
    state = {
        "V": [0.0, 0.0, 0.0, 0.0],
        "lstm": None,
        "sharp_buf": {},  # channel -> latest sharpness
    }

    def set_aal(V):
        for i, v in enumerate(V):
            core.setProperty(AAL, f"V{i+1}", f"{v:.4f}")
        core.waitForDevice(AAL)

    def step_policy():
        import numpy as np
        sharp = np.array([state["sharp_buf"].get(c, 0.0) for c in CHANNELS])
        obs = np.concatenate([state["V"], sharp])
        dV, state["lstm"] = policy.act(obs, state["lstm"])
        state["V"] = list(np.clip(np.array(state["V"]) + dV, -10.0, 10.0))
        state["sharp_buf"].clear()

    def on_frame(img, event, meta=None):
        md = event.metadata or {}
        if md.get("phase") != "calibrate":
            return
        ch = md["channel"]
        state["sharp_buf"][ch] = float(focus_metric(img, method="brenner"))
        if md.get("last_in_stack"):
            step_policy()
            set_aal(state["V"])

    def gen():
        for i in range(n_frames):
            t = i * dt_s
            if i % correct_every == 0:
                # calibration mini-stack: one frame per channel under current V
                for k, ch in enumerate(CHANNELS):
                    yield MDAEvent(
                        channel={"config": ch},
                        min_start_time=t,
                        metadata={"phase": "calibrate", "channel": ch,
                                  "last_in_stack": k == len(CHANNELS) - 1},
                    )
            # science stack with the (possibly just-updated) correction
            for ch in CHANNELS:
                yield MDAEvent(
                    channel={"config": ch},
                    min_start_time=t,
                    metadata={"phase": "main", "channel": ch, "i": i},
                )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/autofocus.py` — already provides `focus_metric` (Brenner / Laplacian variance) and the sweep-and-fit autofocus that this paper's RL controller replaces *for hysteretic actuators*. The same `make_focus_state` / `check_and_correct_focus` skeleton generalises here as `make_chromatic_state(policy)` / `update_chromatic(per_channel_metrics)` — does not exist yet; flag for build sprint when a tunable optic is on the rig.
- `../../../src/core/hardware/core.py` — `run_events` is the dispatcher. Calibration frames and science frames are tagged via `MDAEvent.metadata` and demultiplexed in `on_frame`, the same pattern used by Hu 2023's bias-frame interleave.
- Missing piece on most rigs: the AAL itself. Without a tunable element the correction can't be applied, but the *architecture* (small RL policy controlling a hysteretic actuator from an image-quality reward, embedded as an `on_frame` callback) transfers to correction-collar tuning, deformable-mirror voltage drive, and any piezo-driven alignment whose response drifts with use.

Calibration cost and transferability: training the recurrent PPO agent in this paper required a simulator of the AAL hysteresis; on a real rig you can either (a) pretrain in simulation and fine-tune online (the Bilodeau 2024 pySTED pattern) or (b) treat the first ~few hundred frames of every session as exploration with a damped reward to bound photodamage. Per-channel reward weighting matters — without it the policy will trade DAPI sharpness for far-red sharpness whenever they conflict.

## Cited by

- [[Core/Concepts/Chromatic aberration]] — concrete demonstration that wavelength-dependent focal shift is large enough (mm-scale!) to motivate active correction, not just per-channel z-offsets in software.
- [[Core/Strategies/Closed-loop autofocus]] — generalises closed-loop scalar-Z autofocus to vector-valued actuator control with hysteresis; same MDA calibration-frame / science-frame interleaving as Hu 2023, but the controller is an RL policy instead of a feed-forward regressor because the actuator itself has memory.
- [[Core/Strategies/Imaging parameter optimization]] — companion to Durand 2018: where Durand's RL/bandit optimises *illumination* parameters against image-quality vs photodamage, Schmidt's RL optimises *actuator* parameters against image-quality across wavelengths. Same online-optimisation pattern, different layer of the stack.
