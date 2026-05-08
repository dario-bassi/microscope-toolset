# Feedback Control (Closed-Loop Experiments)

> **Note**: Code examples below use conceptual pseudocode. All multi-frame
> loops MUST use `run_events(core, generator(), on_frame=callback)` in practice.
> The `on_frame` callback is where feedback logic (adjust exposure, SLM, etc.) lives.

## Core Pattern

```
observe -> compute -> modify -> snap -> repeat
```

1. **Observe**: Snap an image, extract the current state.
2. **Compute**: Analyze (detect positions, measure intensity, classify).
3. **Modify**: Update hardware (SLM mask, exposure, stage, focus).
4. **Snap**: Acquire next frame with updated settings.
5. **Repeat**: Until goal is reached or budget is exhausted.

## SLM Mask Creation and Timing

SLM masks are `np.uint8` arrays matching the SLM resolution (query from hardware).

```python
# Get SLM dimensions from hardware, or match camera resolution
h, w = core.getImageHeight(), core.getImageWidth()
mask = np.zeros((h, w), dtype=np.uint8)
yy, xx = np.ogrid[:h, :w]
mask[((xx - cx)**2 + (yy - cy)**2) <= r**2] = 255
```

**Preferred**: Use `slm_image` on `MDAEvent` — the mask is applied automatically:

```python
from useq import MDAEvent, SLMImage
from src.core.hardware.core import run_events

state = {"current_mask": compute_mask(initial_state), "done": False}

def on_frame(img, event):
    obs = analyze(img)
    state["current_mask"] = compute_mask(obs)   # update for next event
    if converged(obs):
        state["done"] = True

def gen():
    for _ in range(n_steps):
        if state["done"]:
            return
        slm = SLMImage(data=state["current_mask"].astype(np.uint8), device="SLM")
        yield MDAEvent(channel={"config": "BF"}, slm_image=slm)

run_events(core, gen(), on_frame=on_frame)
```

**Legacy**: `core.setSLMImage('SLM', mask)` before each `snap()` still works
but is error-prone (easy to forget). Prefer the event-based approach.

## Optogenetics Patterns

- **Excite/Inhibit**: query `core.getStateLabels('SLM-Mode')` to discover the mode labels, set with `core.setProperty('SLM-Mode', 'Label', ...)`. The numeric state index is backend-specific; set once.
- **Targeting cells**: Detect positions, build mask with circles at each cell.
- **Barrier**: Wide band (50-70 px) perpendicular to wavefront. Apply BEFORE wavefront arrives.
  Lead distance = (wave speed × lead time). Wave speed is sample- and temperature-dependent — measure it on your prep (fit radius vs frame on a test acquisition) rather than assuming a range.

## Adaptive Exposure

```python
target_mean = 120
exposure = core.getExposure()
for step in range(n_steps):
    img = snap()
    ratio = target_mean / max(np.mean(img[roi_mask]), 1)
    exposure = np.clip(exposure * ratio, 5, 500)
    core.setExposure(exposure)
```

## Common Pitfalls

- **Forgetting SLM before snap**: Mask is NOT persistent. Apply every frame.
- **Frame budget**: Don't waste frames on preview/diagnostic snaps.
- **Late intervention**: For fast processes (calcium waves), intervene proactively.
- **Wrong coordinate space**: SLM masks are camera pixels, not world coordinates.

## Verification

After intervening, snap a frame and re-measure the same metric.
If it did not improve, adjust (stronger mask, wider barrier, different position).

## Related strategies

- [[Core/Strategies/Rate-limited drive]] — closed-form `predict_n_steps` + closed-loop
  `drive_to_band` early-stop primitive, for setpoint-band control of
  saturating rate laws (gene induction, photoconversion, ablation,
  FUCCI). Sprint #35 utility lifted from ch621 r1 = 10/10. Use this
  instead of an open-loop "stim N times" reflex when the brief gives
  a target band rather than just a floor.
- [[Core/Strategies/Closed-loop state device]] — generic discrete-state-device wrap
  pattern; `Rate-limited drive` is its rate-law-aware specialisation.
- [[Recipes/Targeted gene induction]] — open-loop max-out variant that
  composes the same SLM-mask substrate.

## Literature

- [[Papers/Fox 2022]] — MicroMator: the Python programming model for the observe → compute → modify → snap loop. Users declare reactive experiments as event-condition-action rules (a trigger predicate plus an effect callable) rather than re-writing the acquisition loop for each experiment.
- [[Papers/Chiron 2022]] — CyberSco.Py: the same ECA model exposed as a YAML rulebook (predicates on named metrics; actions set channel / exposure / interval or toggle state devices). Config-file front-end for the observe → compute → modify → snap loop, useful when the protocol author is not a Python programmer.
- [[Papers/Passmore 2025]] — outcome-driven microscopy: the observe → compute → modify → snap loop applied to optogenetic actuation, closing the feedback at the cell biology rather than at the acquisition rate.
- [[Papers/Lugagne 2024]] — deep model predictive control: replace the reactive error term with a learned forward model rolled over a short horizon; receding-horizon execution tracks arbitrary reference trajectories in thousands of single cells in parallel.
- [[Papers/Boiko 2023]] — Coscientist: the observe → compute → modify → snap loop with an LLM in the controller slot. Tool calls emit actions; tool errors and observations feed the next turn; the planner self-corrects on stack traces. The positive demonstration (on chemistry) that motivates the microscopy-focused follow-ups and defines the agentic-controller design space.
- [[Papers/Mandal 2025]] — AILA + AFMBench: the same observe → compute → modify → snap loop, but with an LLM as the controller. The paper's three exposed failure modes (capability-knowledge gap, sleepwalking, prompt fragility) are the reason the inner loop should stay deterministic and the LLM should sit above a safety-enforcing tool layer, not inside the feedback itself.
- [[Papers/Rullan 2018]] — earliest microscope-based closed-loop gene-expression-circuit controller (DLP-DMD per-cell light delivery + PP7 nascent-RNA reporter + per-cell integral feedback `I(t_k) = K_I·Σ e(t_n)` updated every 2 min, on hundreds of trapped yeast cells). Ancestor of Lugagne's deep-MPC and the canonical "cybergenetics" paper at the gene-circuit-control scale.
- [[Papers/Hinderling 2025]] — FARO: open-source Pycro-Manager substrate combining live segmentation (Convpaint or Cellpose) with per-cell feature scoring to drive any DMD exposed as `genericSLM`. Decoupled scheduler/analysis threads scale the loop across subcellular pinning, single-cell activation in deforming epithelial tissue, and tissue-flow steering. The deployable open-substrate generalisation of Rullan 2018 / Lugagne 2024.
- [[Papers/Liang 2023]] — robotics paper, but the recipe transfers exactly: an LLM compiles a natural-language goal into a hierarchical Python program where the outer loop is the acquisition policy and the inner functions are measurement primitives the same LLM authored from a docstring. Aspirational sibling of Boiko 2023 / Mandal 2025 with the cut at *programs* rather than *tool calls*. Currently no published microscope demo at this granularity.
