---
title: EAP4EMSIG – enhancing event-driven microscopy for microfluidic single-cell analysis
authors: Nils Friederich, Angelo Jovin Yamachui Sitcheu, Annika Nassal, Erenus Yildiz, Matthias Pesch, Maximilian Beichter, Lukas Scholtes, Bahar Akbaba, Thomas Lautenschlager, Oliver Neumann, Dietrich Kohlheyer, Hanno Scharr, Johannes Seiffarth, Katharina Nöh, Ralf Mikut
year: 2025
venue: at - Automatisierungstechnik 73(10):808-823
doi: 10.1515/auto-2025-0018
url: https://doi.org/10.1515/auto-2025-0018
researched: 2026-04-25
---

## Abstract

Microfluidic Live-Cell Imaging (MLCI) yields data on microbial cell factories. However, continuous acquisition is challenging as high-throughput experiments often lack real-time insights, delaying responses to stochastic events. We introduce three components in the Experiment Automation Pipeline for Event-Driven Microscopy to Smart Microfluidic Single-Cell Analysis (EAP4EMSIG): a fast, accurate Multi-Layer Perceptron (MLP)-based autofocusing method predicting the focus offset, an evaluation of real-time segmentation methods and a real-time data analysis dashboard. Our MLP-based autofocusing achieves a Mean Absolute Error (MAE) of 0.105 µm with inference times of 87 ms. Among eleven evaluated Deep Learning (DL) segmentation methods, Cellpose reached a Panoptic Quality (PQ) of 93.36 %, while a distance-based method was fastest (121 ms, Panoptic Quality 93.02 %).

## Smart microscopy principle

EAP4EMSIG is a **whole-pipeline event-driven instrument** for high-throughput microbial microfluidics: an 8-module cyclical loop wired together by a dataflow middleware, where every component runs at frame-rate so the microscope can react before the next time-point. The loop is acquisition → real-time segmentation → OMERO storage → real-time analytics → event detector → experiment planner → microscope control → next acquisition. Two flagship technical contributions sit inside that loop — (i) a hand-crafted-feature **MLP autofocus** that predicts a signed Δz offset from a single (potentially defocused) frame in 87 ms with 0.105 µm MAE, and (ii) a benchmarking-and-deployment of **11 segmentation methods** that establishes Cellpose 3 (93.6 % PQ, 1.1 s) and a custom distance-based method (93.0 % PQ, 121 ms) as the speed/accuracy frontier for *C. glutamicum* microcolonies. A live dashboard exposes per-chamber heatmaps, time-series, and Start/Pause/Stop controls for biologist-in-the-loop operation.

The paradigm contribution is to take the **event-driven control loop from single-FOV demos** ([[Papers/Mahecic 2022]], [[Papers/Alvelid 2022]]) **to a routine production pipeline running thousands of microculture chambers in parallel** since 01/2025. Trigger logic is explicit and tunable: a "focus loss" technical event fires when the autofocus predicts |Δz| > 0.5 µm for three consecutive time-points; a "rapid growth" biological event fires when chamber cell count rises ≥ 10 % between frames. Each event either auto-actuates the experiment planner or notifies the operator (Slack / dashboard log). The autofocus design is deliberately distinct from [[Papers/Pinkard 2019]]: Pinkard's FCFNN regresses Δz from raw pixels of an off-axis-LED image; Friederich et al. instead extract image-pyramid + wavelet + statistical features and feed them to a tiny 2-hidden-layer MLP — no special illumination required, CPU-friendly inference, and trained from a single z-stack calibration (±5 µm, 0.1 µm steps, 13 000 images).

## Implementation on pymmcore-plus

The EAP4EMSIG architecture maps cleanly onto a pymmcore-plus event generator with multiple `on_frame` hooks: one updates a per-chamber autofocus state, another runs Cellpose / a fast distance-based segmenter and updates count/area/growth metrics, and a thresholded event detector consumes those metrics to inject planner-driven event responses (re-focus, switch to high-frequency channel, alert operator). The "8-module" architecture in the paper is an orchestration concern; the pymmcore-plus loop only needs to expose the two inner contracts: `predict_focus_offset(img) -> dz_um` and `segment(img) -> labels`.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

# Pre-trained EAP4EMSIG-style autofocus and a fast segmenter.
# predict_focus_offset(img) -> signed Δz in µm (MLP on hand-crafted features).
# segment(img) -> integer label image (Cellpose for accuracy, distance-based for speed).
predict_focus_offset = load_focus_mlp("eap4emsig_mlp.pt")
segment = load_segmenter("cellpose")  # or "distance_based" for 121-ms inference

FOCUS_DRIFT_UM = 0.5      # technical-event threshold (paper: 0.5 µm)
FOCUS_HOLD_FRAMES = 3      # require N consecutive frames over threshold
GROWTH_TRIGGER = 0.10      # biological-event threshold (paper: +10% cell count)

def event_driven_microfluidic(core, chambers, n_frames=2000, dt_s=60.0):
    """Run an EAP4EMSIG-style loop: per-chamber autofocus + segmentation + events."""
    state = {c: {"z": core.getZPosition(), "drift_run": 0,
                 "last_count": 0, "history": []} for c in chambers}

    def on_frame(img, event, meta=None):
        ch = event.metadata["chamber"]
        s = state[ch]
        # 1. Autofocus inference (87 ms): update cached Z for this chamber.
        dz = float(predict_focus_offset(img))
        s["z"] += dz
        # 2. Technical event: 3 consecutive frames with |dz| > 0.5 µm => refocus / alert.
        s["drift_run"] = s["drift_run"] + 1 if abs(dz) > FOCUS_DRIFT_UM else 0
        if s["drift_run"] >= FOCUS_HOLD_FRAMES:
            event.metadata["alert"] = "focus_loss"
            s["drift_run"] = 0
        # 3. Real-time segmentation: per-chamber metrics.
        labels = segment(img)
        count = int(labels.max())
        s["history"].append(count)
        # 4. Biological event: ≥10% count jump triggers high-frequency follow-up.
        if s["last_count"] and count > s["last_count"] * (1 + GROWTH_TRIGGER):
            event.metadata["alert"] = "rapid_growth"
        s["last_count"] = count

    def gen():
        for i in range(n_frames):
            for ch, (x, y) in chambers.items():
                yield MDAEvent(
                    x_pos=x, y_pos=y, z_pos=state[ch]["z"],
                    channel={"config": "BF"},
                    min_start_time=i * dt_s,
                    metadata={"chamber": ch, "i": i},
                )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/autofocus.py` — current strategies are reactive / threshold / predictive sweeps. The EAP4EMSIG MLP slots in as a fifth strategy alongside [[Papers/Pinkard 2019]]'s FCFNN: same `predict_defocus(img) -> dz_um` contract, but features-then-MLP instead of raw pixels through a CNN. Worth implementing both — Pinkard needs an off-axis LED, Friederich needs only standard brightfield, neither needs a Z-sweep.
- `../../../src/core/workflows/event_driven.py` — does not yet exist; this paper is the strongest motivation to build it. The state-machine surface ([[Papers/Mahecic 2022]]'s poll/burst) generalises to "(focus_drift, growth_rate, custom_metric) → planner action" with EAP4EMSIG's explicit threshold tuples (0.5 µm × 3 frames, +10 % count) as defaults.
- `../../../src/core/hardware/core.py` — `run_events` already dispatches the loop; no engine changes needed.
- Segmentation — Cellpose is already the agent's go-to when accuracy matters; the paper's contribution is to **benchmark 11 methods** so the speed/accuracy choice is data-grounded for microbial microcolonies. For real-time loops where Cellpose's ~1 s is too slow, a distance-based segmenter at 121 ms with 0.5 % less PQ is the right trade.

Calibration: the autofocus is sample-and-setup specific (the paper notes retraining is needed when chip type, illumination, or temperature change). Plan for a one-time z-stack calibration per new prep — ±5 µm, 0.1 µm steps, ~13 000 frames — then the model amortises over the full multi-week microfluidic run.

## Cited by

- [[Core/Concepts/Event-driven acquisition]] — EAP4EMSIG is the production-scale instance of the poll/burst event-driven pattern: per-chamber technical (focus drift) and biological (growth jump) triggers wired into an experiment planner across thousands of parallel microcultures.
- [[Core/Strategies/Closed-loop autofocus]] — adds the EAP4EMSIG features-MLP variant as a brightfield-only sibling of [[Papers/Pinkard 2019]]'s off-axis-LED FCFNN: same single-shot Δz contract, different feature pipeline, cheaper hardware requirement.
- [[Core/Strategies/Adaptive acquisition]] — generalises [[Papers/Mahecic 2022]]'s rate-switching to a multi-component pipeline (autofocus + segmentation + planner) running per-chamber on a microfluidic chip; the descendant of [[Papers/Conrad 2011]]'s online-classifier-driven acquisition for the microbial single-cell era.
- [[Papers/Chiron 2022]] — CyberSco.Py expresses the same event-condition-action pattern as YAML rulebooks for budding-yeast experiments; EAP4EMSIG is the equivalent for high-throughput microbial microfluidics with hard-coded thresholds and a planner-driven response surface.
