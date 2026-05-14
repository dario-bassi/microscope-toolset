---
title: Development of AI-assisted microscopy frameworks through realistic simulation with pySTED
authors: Anthony Bilodeau, Albert Michaud-Gagnon, Julia Chabbert, Benoit Turcotte, Jörn Heine, Audrey Durand, Flavie Lavoie-Cardinal
year: 2024
venue: Nature Machine Intelligence 6(10):1197-1215
doi: 10.1038/s42256-024-00903-w
url: https://www.nature.com/articles/s42256-024-00903-w
researched: 2026-04-24
---

## Abstract

The integration of artificial intelligence into microscopy systems significantly enhances performance, optimizing both image acquisition and analysis phases. Development of artificial intelligence-assisted super-resolution microscopy is often limited by access to large biological datasets, as well as by difficulties to benchmark and compare approaches on heterogeneous samples. We demonstrate the benefits of a realistic stimulated emission depletion microscopy simulation platform, pySTED, for the development and deployment of artificial intelligence strategies for super-resolution microscopy. pySTED integrates theoretically and empirically validated models for photobleaching and point spread function generation in stimulated emission depletion microscopy, as well as simulating realistic point-scanning dynamics and using a deep learning model to replicate the underlying structures of real images. This simulation environment can be used for data augmentation to train deep neural networks, for the development of online optimization strategies and to train reinforcement learning models. Using pySTED as a training environment allows the reinforcement learning models to bridge the gap between simulation and reality, as showcased by its successful deployment on a real microscope system without fine tuning.

## Smart microscopy principle

Online parameter optimisation on a real STED microscope (the Durand 2018 contextual-bandit paradigm) burns photons on every exploration step — bandits and Bayesian optimisers need many evaluations to learn a sample, and each evaluation bleaches the specimen they're trying to image well. Bilodeau et al. break this dependency by **moving the entire training phase off-instrument into a physically faithful simulator** (pySTED) that models photobleaching, depletion-laser-dependent PSF generation, point-scanning dynamics, and — via a learned generative prior — realistic underlying biological structures. A reinforcement-learning agent trained to balance resolution / SNR / photobleaching inside pySTED **transfers to a real microscope without any per-sample fine-tuning**, demonstrating that a simulator with the right physics is enough to close the sim-to-real gap for STED parameter control.

The transferable idea: **separate "learning the policy" from "running the experiment"**. Online optimisers (Durand 2018) co-mingle the two and pay for it in dose; sim-trained policies arrive at the microscope already competent and spend their photons on biology, not on exploration. The same recipe — physically grounded simulator + RL agent + zero-shot deployment — generalises beyond STED to any modality whose photon-cost-vs-image-quality trade-off is too expensive to explore live: confocal pinhole / power, light-sheet sheet thickness / dwell time, two-photon depth / dwell. pySTED also doubles as a benchmark and data-augmentation factory, turning "AI-assisted microscopy" from a per-lab artisanal process into something with shared simulators, shared baselines, and reproducible policy comparisons.

## Implementation on pymmcore-plus

A sim-trained RL policy is a callable `policy(observation) -> action` that runs at zero cost inside the acquisition loop. The policy maps a recent frame (and any auxiliary state — laser-power history, accumulated dose, time index) to the next parameter vector, which the generator turns into an `MDAEvent`. Because the agent was trained in a simulator, no exploration / `tell` step is needed at deployment — `on_frame` only logs telemetry.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
import numpy as np

# Loaded once at startup; pre-trained inside pySTED, not on the live sample.
# Could be a PFRL / Stable-Baselines3 policy, a small TorchScript module, etc.
policy = load_pretrained_policy("checkpoints/sted_rl_policy.pt")

PARAM_NAMES = ("excitation_uW", "depletion_mW", "pixeldwell_us")

def to_action(params):
    return {
        "exposure": float(params["pixeldwell_us"] * pixels_per_frame / 1000),  # ms
        "properties": [
            ("ExcitationLaser", "Power_uW",  f"{params['excitation_uW']:.3f}"),
            ("DepletionLaser",  "Power_mW",  f"{params['depletion_mW']:.3f}"),
        ],
    }

def sim_to_real_acquisition(core, n_frames=60, channel="STED"):
    state = {"prev_img": None, "dose": 0.0, "history": []}

    def gen():
        for i in range(n_frames):
            obs = build_observation(state)         # frame, dose, time, ...
            params = policy.act(obs)               # zero-shot, no env interaction
            action = to_action(params)
            yield MDAEvent(
                channel={"config": channel},
                metadata={"params": dict(zip(PARAM_NAMES, params)), "i": i},
                **action,
            )

    def on_frame(img, event, meta=None):
        # Pure logging -- no policy update on the real instrument.
        state["prev_img"] = img
        state["dose"] += event.metadata["params"]["depletion_mW"] * event.exposure
        state["history"].append({"i": event.metadata["i"],
                                 "params": event.metadata["params"]})

    return run_events(core, gen(), on_frame=on_frame), state["history"]
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/exposure.py` — currently exposes greedy / online updaters only. A sim-trained-policy wrapper (`run_pretrained_policy(core, policy, build_observation, n_frames)`) belongs alongside the bandit variant motivated by Durand 2018; the two are complementary entry points (online-learn vs deploy-pretrained).
- `../../../src/core/hardware/core.py` — `run_events` already supports the `properties=[...]` channel for arbitrary device parameters, so sim-trained policies that drive depletion-laser power, dwell time, or pinhole size slot in without engine changes.
- A pySTED-style simulator wrapper (forward-model: parameters → simulated frame, with photobleaching state) does not exist in `src/core/`; this paper is the canonical motivation for adding one if we ever want to train policies offline against a physical model rather than a content-free generative prior.

Sim-to-real safety: even a perfectly trained policy will issue parameter requests outside the safe envelope if the live sample is unusual (much dimmer / brighter than any pySTED sample). Always wrap the policy output in a hardware-bounded clamp (max laser power, min exposure, max dose-per-frame) before turning it into an `MDAEvent` — the policy proposes, the constraint layer disposes.

## Cited by

- [[Core/Strategies/Imaging parameter optimization]] — alternative to online optimisation: train the policy in a physically grounded simulator and deploy zero-shot on the real instrument, eliminating per-sample exploration dose entirely.
- [[Core/Strategies/Gentle imaging]] — sim-trained RL bakes the resolution-vs-photodamage trade-off into the policy at training time, so the agent arrives at the microscope already preferring gentle parameter combinations without needing live exploration to learn them.
