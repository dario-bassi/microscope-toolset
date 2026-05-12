---
title: Deep model predictive control of gene expression in thousands of single cells
authors: Jean-Baptiste Lugagne, Caroline M. Blassick, Mary J. Dunlop
year: 2024
venue: Nature Communications 15(1):2148
doi: 10.1038/s41467-024-46361-1
url: https://www.nature.com/articles/s41467-024-46361-1
researched: 2026-04-24
---

## Abstract

Gene expression is inherently dynamic, due to complex regulation and stochastic biochemical events. However, the effects of these dynamics on cell phenotypes can be difficult to determine. Researchers have historically been limited to passive observations of natural dynamics, which can preclude studies of elusive and noisy cellular events where large amounts of data are required to reveal statistically significant effects. Here, using recent advances in the fields of machine learning and control theory, we train a deep neural network to accurately predict the response of an optogenetic system in Escherichia coli cells. We then use the network in a deep model predictive control framework to impose arbitrary and cell-specific gene expression dynamics on thousands of single cells in real time, applying the framework to generate complex time-varying patterns. We also showcase the framework's ability to link expression patterns to dynamic functional outcomes by controlling expression of the tetA antibiotic resistance gene. This study highlights how deep learning-enabled feedback control can be used to tailor distributions of gene expression dynamics with high accuracy and throughput without expert knowledge of the biological system.

## Smart microscopy principle

Model-predictive control (MPC) closes the loop *through a model of the plant* instead of just through an error term. Each frame, the microscope segments and tracks single cells, feeds their recent history into a learned forward model (a deep neural network trained on paired optogenetic-input → expression-output trajectories), and uses the model to simulate many candidate light-stimulation sequences over a short prediction horizon. The stimulation that drives the predicted trajectory closest to the user-specified reference gets executed — per cell, every control step. The loop repeats, which makes MPC robust to model error, cell-to-cell variability, and drift.

The contribution for smart microscopy is scale: the controller runs on *thousands of single cells in parallel*, each with its own reference trajectory and its own tailored stimulation pattern. This generalises closed-loop optogenetics from "steer one cell toward one target" (outcome-driven, reactive) to "impose arbitrary time-varying dynamics on a heterogeneous population" (predictive, trajectory-tracking). It also decouples the control law from expert biological knowledge — the neural network learns the dose-response and its timescales from data rather than requiring a handcrafted ODE model.

The practical implication: when you want to drive cells through a specific dynamic profile (a ramp, a square wave, a complex waveform that mimics some native signalling trajectory), reactive control (error → stronger stim) under-performs. A learned predictive model of *how the cell will respond to the next few stimulation pulses* lets you pre-emptively shape the input, and that's what matters at fast signalling timescales where the response lags the stimulus.

## Implementation on pymmcore-plus

The MPC loop maps onto a `run_events` generator where each decision step does: segment + track cells, roll the forward model over a short horizon for several candidate inputs, pick the one minimising predicted tracking error, emit an `MDAEvent` with the corresponding SLM pattern. The prediction horizon and control horizon are tuning knobs — longer horizons cost compute per step, shorter horizons degrade to reactive control.

```python
from useq import MDAEvent, SLMImage
import numpy as np
from src.core.hardware.core import run_events

CONTROL_INTERVAL_S = 300.0       # 5 min between control steps (Lugagne protocol)
HORIZON_STEPS = 6                # predict 30 min ahead
N_CANDIDATES = 16                # candidate input sequences to roll out
MAX_FRAMES = 100


def mpc_gene_expression(
    core,
    forward_model,         # fn(history, input_seq) -> predicted output_seq
    reference_traj,        # dict cell_id -> target trajectory (np.ndarray)
    segment_and_track,     # fn(image) -> dict cell_id -> cell_state
    build_slm_pattern,     # fn(dict cell_id -> stim_level) -> np.uint8 mask
    channel="GFP",
):
    """Deep-MPC control of per-cell expression dynamics.

    Each control step: measure cell states, predict N_CANDIDATES input
    sequences per cell over HORIZON_STEPS, select the sequence whose
    predicted output best tracks the reference, apply its first element
    via the SLM, then repeat.
    """
    history = {}  # cell_id -> deque of recent (input, output) pairs

    state = {"step": 0, "next_mask": build_slm_pattern({})}

    def on_frame(img, event):
        cells = segment_and_track(img)
        chosen_inputs = {}
        for cid, cstate in cells.items():
            hist = history.setdefault(cid, [])
            ref = reference_traj.get(cid)
            if ref is None:
                continue
            t_now = len(hist)
            ref_window = ref[t_now : t_now + HORIZON_STEPS]
            candidates = np.random.randint(0, 2, (N_CANDIDATES, HORIZON_STEPS))
            best_cost, best_u = np.inf, candidates[0]
            for u in candidates:
                y_pred = forward_model(hist, u)          # length HORIZON_STEPS
                cost = np.mean((y_pred - ref_window) ** 2)
                if cost < best_cost:
                    best_cost, best_u = cost, u
            chosen_inputs[cid] = float(best_u[0])        # receding horizon: apply 1st step
            hist.append((float(best_u[0]), cstate["expression"]))
        state["next_mask"] = build_slm_pattern(chosen_inputs)
        state["step"] += 1

    def gen():
        for t in range(MAX_FRAMES):
            yield MDAEvent(
                channel={"config": channel},
                slm_image=SLMImage(data=state["next_mask"], device="SLM"),
                min_start_time=t * CONTROL_INTERVAL_S,
                index={"t": t},
            )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events` dispatches the MDA events; `make_slm_circle` + `apply_slm` handle pattern delivery.
- `[[Core/Approach/OADA loop]]` — the generator + `on_frame` scaffold this snippet uses: `on_frame` rolls the forward model and stages the next SLM mask in shared state; the generator emits an `MDAEvent` carrying that mask before the next snap.
- A dedicated `mpc_controller` module does not yet exist. A clean API would be `run_mpc(core, model, reference, horizon, n_candidates, build_actuation)` — separating the generic receding-horizon scaffolding from the sample-specific forward model, reference trajectory, and actuation mapping. The forward model is pluggable: a small RNN/LSTM like Lugagne et al.'s choice, a linear state-space model for simpler plants, or a fitted ODE for mechanism-driven setups.

Calibration on a real prep: (i) collect a *training* dataset of random or PRBS-stimulated input-output trajectories to fit the forward model before running closed-loop MPC; (ii) the control interval must exceed the segmentation + inference + optimisation latency (Lugagne et al. used 5 min, appropriate for *E. coli* gene expression timescales — faster signalling like MAPK/ERK needs sub-minute intervals and correspondingly faster models); (iii) clip the stimulation amplitude to avoid saturating the opsin and to respect phototoxicity bounds; (iv) use receding-horizon execution (apply only the first element of the chosen input sequence, then re-plan) — this is what gives MPC its disturbance rejection.

The same scaffolding generalises beyond gene expression: swap the forward model and the measurement function and the identical loop controls MAPK/ERK nuclear translocation dynamics, cytokine secretion, or any optogenetically actuable signalling output with a trained predictor.

## Cited by

- [[Core/Strategies/Feedback control]] — predictive control is the model-based cousin of the observe → compute → modify → snap loop; instead of reacting to the current error, the controller simulates candidate inputs over a horizon and picks the one tracking a reference trajectory.
- [[Core/Strategies/SLM optogenetics]] — closed-loop SLM patterning driven by a learned forward model of per-cell optogenetic response; thousands of cells steered through arbitrary dynamics in parallel.
