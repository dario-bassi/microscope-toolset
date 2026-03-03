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

## Common Pitfalls

- **Mask dtype**: Must be `np.uint8`. Other dtypes fail silently.
- **Coordinate space**: Camera pixels, NOT world coordinates.
- **Stage movement**: Recompute mask positions after moving stage.
- **Reactive vs proactive**: For fast processes, intervene BEFORE the event arrives.
- **Wrong calcium channel**: GCaMP uses GFP filter → may appear in structural marker channel, not a dedicated calcium channel. Always verify by test stimulation.
- **Calcium decay**: Wait ≥10s between experiment phases for full decay to baseline.
