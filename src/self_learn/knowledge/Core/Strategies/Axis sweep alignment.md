# Axis-sweep alignment via useq-MDA state-device sweeps

> **When to use:** When finding the optimal state of a categorical device (filter wheel, dichroic, objective) by sweeping N states and picking the best by a scalar signal metric.

**Pattern**: a categorical state device (filter wheel, light-sheet alignment encoder, dichroic, objective turret, electronic galvo offset) has N positions. The sample's response to that device is unimodal in state index — there's a single "best" position and signal falls off either way. Find the argmax in N snaps with no inline `setState → snap` loop.

The transferable substrate: every state visited is delivered through a `useq.MDAEvent` whose `properties` field carries the set-point. The MDA runner sets the device, waits, snaps, and a callback reduces each frame to a scalar. Same shape on real hardware.

```python
from self_learn.utils.axis_sweep import sweep_axis_mda
res = sweep_axis_mda(core, "LightSheetY", n_states=5,
                     channel="flk1-GFP", exposure=50.0)
res.argmax_state   # int — best state index
res.signals        # {state_idx: reduced_value}
res.images         # {state_idx: ndarray}  (drop with keep_images=False)
```

## Independent vs coordinate-descent

| Use case | Tool | Cost |
|---|---|---|
| Each axis well-aligned at the same state regardless of the others | `sweep_axis_mda` per axis with `pin={...}` at default | N_axes × N_states events |
| Axes interact — argmax of one shifts when others move | `coordinate_descent_mda(axes, n_passes=1)` | N_axes × N_states × passes events |

The light-sheet AutoPilot lineage:

- **ch657** (independent sweep, ch657 r1 = 10/10): pin every other axis at default state 2; sweep each in turn. Three axes × five states = 15 events. Per-axis argmax all land on state 2.
- **ch658** (coordinate descent, ch658 r1 = 10/10): pin OTHER axes at their CURRENT-best state; the first axis updates the pin map for the next axis. Same 15 events, but state map evolves between sweeps. Tilts converge OFF default (TX=0, TY=0) because zebrafish vasculature is asymmetric — the per-axis brightness peak under a non-symmetric content distribution biases away from the device-aligned state. **Realistic physics, not a bug.**

## Why this exists separate from `sensorless_ao.sweep_state_device`

`sensorless_ao.sweep_state_device` uses an inline `for k in range(N): setState(k); waitForDevice; snap(); reduce(img)` loop. That pattern bypasses the MDA engine, losing hardware-native sequencing, structured event metadata, and clean cancellation.

`axis_sweep` builds a useq event sequence and dispatches via `run_events`. The same `MDAEvent` list lands directly on the hardware MDA runner on any pymmcore-plus backend — same code path on real hardware and simulated cores.

For new code, prefer `axis_sweep` over `sweep_state_device`. AO sharpness sweeps can still compose `axis_sweep` with the `sensorless_ao` inference helpers.

## Reduce hooks

Default reducer is `mean`. Built-in keys: `mean`, `max`, `std`, `p95`. For non-mean metrics:

```python
sweep_axis_mda(..., reduce="std")           # contrast-driven
sweep_axis_mda(..., reduce=lambda img: float(focus_metric(img, "brenner")))
                                            # sharpness-driven
sweep_axis_mda(..., reduce=lambda img: count_blobs_log(img))
                                            # detector-output-driven
```

The reducer runs inside `on_frame`, so each result is a scalar paired with its state index.

## Anti-patterns

- **Inline `setState → snap` loops** in fresh solves — see above; replace with `sweep_axis_mda`.
- **Cached state maps when descent might re-enter** — read `core.getProperty(axis, "State")` at every sweep boundary. If a prior sweep already descended, the next sweep's "initial" snap sees post-descent state, corrupting the baseline diagnostic even when the final argmax is correct.
- **Skipping per-axis pin** — `sweep_axis_mda(core, "Y")` without `pin={"TiltX": 2, "TiltY": 2}` lets the OTHER axes drift if the framework's default is non-zero. Always pass `pin` or rely on the descent helper.

## Composes with

- `utils.fft_peak.find_fft_peak` — replace mean with FFT peak amplitude when the metric is "best modulation contrast" (ch649 SIM lineage).
- `utils.fluorophore_brightness.compare_predictions` — couple per-state mean signal to a fluorophore-brightness ratio prediction.
- `utils.dose_threshold.sweep_threshold` — when the sweep is binary (firing / not firing) instead of unimodal-peak.

## Related

- `[[Core/Strategies/Closed-loop state device]]` — per-frame-update version of state-device control. Use when the controller decides each next state from the previous frame's measurement, not when the search is a one-shot argmax.
- `[[Core/Strategies/Closed-loop autofocus]]` — Z-plane analogue; same shape but a continuous axis, so the right tool is a Brenner sweep + parabolic-peak interp rather than a state-device argmax.
- `sensorless_ao.sweep_state_device` — kept for AO-specific sharpness inference (per-axis-presence / residual-axis decomposition). New code should compose `axis_sweep` with the AO inference helpers rather than calling `sweep_state_device` directly.

## Literature

- Royer 2016 — AutoPilot 5-axis alignment (light-sheet). The canonical multi-axis coordinate-descent reference. `[[Royer 2016]]`.
- McDole 2018 — adaptive imaging mouse embryo, content-aware re-alignment between time-points. `[[McDole 2018]]`.
- Schmidt 2023 — RL-driven chromatic AO; same coordinate-descent shape under an RL controller. `[[Schmidt 2023]]`.

## Tested on

- ch657 r1 = 10/10 (3 axes × 5 states independent sweep)
- ch658 r1 = 10/10 (3 axes × 5 states coordinate descent, 97.6% recovery)
- (ch656 nucleus-count invariance is a 4-snap variant — 4 cardinal angles, no argmax — but uses the same `MDAEvent.properties` substrate for the rotation-stage dispatch.)
