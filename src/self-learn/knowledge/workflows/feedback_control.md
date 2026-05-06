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
from src.workflows.oada import adaptive_generator

def on_decide(image, event, state):
    current_state = analyze(image)
    new_mask = compute_mask(current_state)
    state.extra_events.append(
        MDAEvent(slm_image=SLMImage(data=new_mask, device="SLM"))
    )

initial_mask = compute_mask(initial_state)
base = [MDAEvent(slm_image=SLMImage(data=initial_mask, device="SLM"))
        for _ in range(n_steps)]
gen, on_frame, state = adaptive_generator(base, on_decide=on_decide)
run_events(core, gen(), on_frame=on_frame)
```

**Legacy**: `core.setSLMImage('SLM', mask)` before each `snap()` still works
but is error-prone (easy to forget). Prefer the event-based approach.

## Optogenetics Patterns

- **Excite/Inhibit**: `SLM-Mode` state `0` = excite, `1` = inhibit. Set once.
- **Targeting cells**: Detect positions, build mask with circles at each cell.
- **Barrier**: Wide band (50-70px) perpendicular to wavefront. Apply BEFORE wavefront arrives.
  Wave speed ~15-25 px/frame; anticipate 10-15 frames ahead.

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
