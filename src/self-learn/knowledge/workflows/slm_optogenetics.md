# SLM Optogenetics

## SLM Basics

`np.uint8` mask matching SLM resolution, in viewport/camera space. 255 = light on, 0 = light off.
Query SLM dimensions from the device or match camera resolution.

### MDA-native approach (preferred)

Use `SLMImage` on each `MDAEvent` — the engine applies the mask automatically:

```python
from useq import MDAEvent, SLMImage

mask = make_slm_circle(target_x, target_y, radius=40)
event = MDAEvent(
    channel={"config": "GFP"},
    slm_image=SLMImage(data=mask, device="SLM"),
)
```

`run_events()` handles `slm_image` natively — the engine applies the mask before each snap.

### Imperative approach (for quick scripts)

```python
from src.hardware.core import apply_slm, snap
apply_slm(core, mask)      # calls setSLMImage + displaySLMImage
img = snap(core, "GFP")
```

**Most common bug**: forgetting SLM mask before `snap()`. The MDA-native approach avoids this.

## Mask Creation

```python
from src.hardware.core import make_slm_circle

mask = make_slm_circle((cx, cy), radius, core=core)  # auto-detects SLM size

# Manual barrier mask
def make_slm_barrier(y_center, width=60, size=512):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[max(0, y_center - width//2):min(size, y_center + width//2), :] = 255
    return mask
```

## SLM Modes

```python
# Check available mode labels first (varies by backend)
allowed = core.getAllowedPropertyValues("SLM-Mode", "Label")
# Typical labels: ("excite", "inhibit") or ("0", "1")
core.setProperty("SLM-Mode", "Label", "excite")   # Activates cells
core.setProperty("SLM-Mode", "Label", "inhibit")   # Suppresses activity
```

Set mode ONCE before the loop unless switching mid-experiment.
Always check allowed labels — backends vary.

## Phototaxis (Volvox) — OADA approach

Colony swims TOWARD light. Place mask at TARGET position, not on colony.

```python
from useq import MDAEvent, SLMImage
from src.workflows.mda import adaptive_phase_events

mask = make_slm_circle((target_x, target_y), radius=40)
slm = SLMImage(data=mask, device="SLM")

base = [MDAEvent(channel={"config": "GFP"}, slm_image=slm) for _ in range(30)]

def on_decide(image, event, state):
    pos = detect_colony(image)
    if distance(pos, (target_x, target_y)) < 10:
        state.stop = True

gen, on_frame, state = adaptive_generator(base, on_decide=on_decide)
run_events(core, gen(), on_frame=on_frame)
```

Phototactic turning rate depends on light intensity and colony size — calibrate per experiment.
Apply the SLM mask before the first acquisition frame.

## Calcium Wave Barrier

```python
core.setState("SLM-Mode", "1")  # Inhibit
mask = make_slm_barrier(barrier_y, width=60)
slm = SLMImage(data=mask, device="SLM")

events = [MDAEvent(channel={"config": "GFP"}, slm_image=slm) for _ in range(n_steps)]
run_events(core, events)
```

Apply barrier BEFORE wavefront arrives (10-15 frames ahead at ~20 px/frame).
Applying after the wave passes does nothing.

## Photoconversion

Irreversible. Apply mask once, cells stay converted forever.

```python
slm = SLMImage(data=make_slm_circle((cx, cy), radius), device="SLM")
events = [
    MDAEvent(channel={"config": "GFP"}, slm_image=slm),   # Conversion frame
    MDAEvent(channel={"config": "GFP"}),                    # Verify next frame
]
run_events(core, events)
```

Conversion may not be visible same frame -- check NEXT frame.

## Closed-Loop SLM with OADA

For dynamic masks that change each frame (e.g., tracking a moving target):

```python
def on_decide(image, event, state):
    target = detect_target(image)
    new_mask = make_slm_circle(target, radius=30)
    # Inject event with updated SLM mask
    state.extra_events.append(
        MDAEvent(channel={"config": "GFP"},
                 slm_image=SLMImage(data=new_mask, device="SLM"))
    )

gen, on_frame, state = adaptive_generator(
    [MDAEvent(channel={"config": "GFP"}, slm_image=initial_slm)],
    on_decide=on_decide, max_frames=50,
)
```

## ChR2 Calcium Imaging (ch413 pattern — ΔF/F = 541%)

Neurons expressing ChR2 + GCaMP. SLM activates ChR2 → depolarization → GCaMP fluorescence.

```python
# Key: GCaMP shares GFP filter with MAP2 → calcium signal in same channel as structural marker
# Verify which channel by test: apply SLM, snap, compare to baseline

# Protocol: baseline (no SLM) → stimulation (with SLM) → recovery (no SLM)
# Use min_start_time for proper temporal spacing
# ΔF/F = (stim - baseline) / baseline → expect >500% at target, ~0% at controls
```

See `knowledge/playbooks/neurons.md` for full protocol.

## Dynamic Mask Tracking (Closed-Loop Migration)

For experiments where the stimulation mask follows a moving cell:

```python
from useq import MDAEvent, SLMImage

# Shared state updated by on_frame
state = {"current_tip": start_tip, "stim_mask_slm": initial_mask}

def on_frame(image, event, metadata):
    if event.channel.config == "mScarlet3":
        # Re-detect cell tip toward target
        new_tip = find_cell_tip(image, state["centroid"], target_b)
        state["current_tip"] = new_tip
        # Update stim mask for next stimulation event
        cam_mask = make_circular_mask(new_tip, radius=15)
        state["stim_mask_slm"] = camera_mask_to_slm(cam_mask, calibration_matrix)

def event_generator():
    for tick in range(n_stim):
        t = tick * stim_interval_s
        if t % imaging_interval_s == 0:
            # Imaging with white DMD mask (full-field illumination)
            yield MDAEvent(channel={"group": "TTL_ERK", "config": "mScarlet3"},
                          exposure=500, min_start_time=t,
                          slm_image=SLMImage(data=white_mask, device="Mosaic3"))
        # Stimulation with current tracking mask
        yield MDAEvent(channel={"group": "TTL_ERK", "config": "CyanStim"},
                      exposure=100, min_start_time=t + 2.0,
                      slm_image=SLMImage(data=state["stim_mask_slm"],
                                        device="Mosaic3", exposure=100))

run_mda_with_feedback(event_generator(), on_frame=on_frame)
```

### Key learning from 60h tracking experiment (2026-03-16)
- Stimulating the cell's **leading edge** caused the cell to migrate AWAY from the target
- The cell moved 75 um in the opposite direction at ~1.26 um/h
- Consider stimulating the REAR of the cell, or AHEAD of the leading edge
- For long experiments: save frames incrementally in on_frame, not after MDA finishes

## DMD in Excitation Path

On some microscopes (e.g., with Andor Mosaic III), the DMD is in the **excitation light
path for ALL channels**, not just the stimulation channel. This means:

- **ALL imaging events need a white DMD mask** — without it, no fluorescence signal
  reaches the camera (only background ~130 counts)
- Use `SLMImage(data=white_mask, device="Mosaic3")` for imaging events
- Use `SLMImage(data=targeted_mask, device="Mosaic3", exposure=100)` for stimulation
- For manual snaps: `setSLMExposure()` + `setSLMImage()` + `displaySLMImage()` before acquiring

## Common Pitfalls

- **Mask dtype**: Must be `np.uint8`. Other dtypes fail silently.
- **Coordinate space**: Camera pixels, NOT world coordinates.
- **Stage movement**: Recompute mask positions after moving stage.
- **Reactive vs proactive**: For fast processes, intervene BEFORE the event arrives.
- **Wrong calcium channel**: GCaMP uses GFP filter → may appear in structural marker channel, not a dedicated calcium channel. Always verify by test stimulation.
- **Calcium decay**: Wait ≥10s between experiment phases for full decay to baseline.
- **DMD in excitation path**: On some setups, DMD must be active (white mask) for ALL imaging, not just stimulation. Without it, images show only background.
- **setSLMImage takes numpy arrays**: Pass the array directly, NOT `.tobytes()`.
- **run_mda_with_feedback in MCP sandbox**: Do NOT pass `mmc` as first arg — it's pre-bound.
