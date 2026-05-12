# Simultaneous SLM targeting over sequential firing

## Principle

When a closed-loop scenario asks for N spatial targets to be stimulated, fire all N simultaneously via a multi-spot SLM mask, not sequentially target-by-target. Sequential firing wastes sim-time between targets (each firing window lets earlier targets' waves propagate and disturb later targets' baselines); simultaneous firing uses every control step to build up stimulation at every target at once.

The physics maps one-to-one to what real spatial-light-modulator hardware does: SLMs display an arbitrary grayscale pattern across the whole field, so lighting N spots in parallel costs zero additional time over lighting one. The illusion that we *need* to fire targets sequentially is a software convenience — it's not a hardware constraint.

## When this applies

- Closed-loop targeted stimulation with ≥ 2 targets (ChR2 activation, calcium-wave nucleation, optogenetic perturbation arrays, FRAP on multiple spots).
- Any sim-time-coupled reaction-diffusion or excitable-media scenario where a single stimulus needs to cross a nucleation threshold.
- Multi-well / multi-ROI photostimulation where each ROI needs its own light dose.

Does **not** apply when:

- Targets must be independently timed (stimulation staircase, interval-tuning experiments).
- The biological response is so fast that each target saturates before the next would fire.

## Pattern

```python
import numpy as np
from useq import MDAEvent  # or direct setSLMImage / displaySLMImage

# Compose one mask with disc-shaped bright spots at every target.
mask = np.zeros((H, W), dtype=np.uint8)
yy, xx = np.mgrid[:H, :W]
for y, x in targets_rc:
    mask[(yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2] = 255

# Fire: reapply every snap for N reapplies (SLM may clear between snaps
# on some backends — see the Optogenetics recipe).
for _ in range(N_REAPPLIES):   # 30-50 on typical sims
    core.setSLMImage(slm_device, mask)
    core.displaySLMImage(slm_device)
    core.snapImage()

# Read post-stim state ONCE for all targets.
core.setSLMPixelsTo(slm_device, 0)
core.displaySLMImage(slm_device)
post = core.getImage()
```

## Key sizing observations

- **Spot radius**: in FHN-like excitable media, radius ≥ nucleation-threshold size. For Gray-Scott / FitzHugh-Nagumo at ch591's 512-px FOV, radius 60-80 px was the threshold — smaller spots (25-35 px) failed to nucleate.
- **Reapplies**: 30 was borderline, 50 was reliable. The simulator's per-snap simtime step × 50 ≈ 7000 sim-units is enough for a wave to form and detectably propagate.
- **Target separation**: ≥ 100-150 px so waves don't merge before each target has nucleated. For a 512 × 512 FOV, a 5-point star (centre + 4 at ±150 px) plus one diagonal gives 6 separated targets.

## Detection after simultaneous firing

Record a pre-stim baseline in each target's patch from the initial snap (one reference that all targets share, taken *before* any firing). After the N_REAPPLIES sequence, read a single post-stim snap. Compare per-target patch mean to its per-target baseline; declare nucleation if Δ is above a chosen threshold.

See also [[Core/Strategies/Closed-loop state device]] — the imperative SLM firing sequence is a sibling closed-loop pattern (device state reflects scientist intent, not just the last-observed value).

## Literature

- [[Papers/Passmore 2025]] — per-frame SLM pattern updates for continuous optogenetic steering; the simultaneous-multi-spot pattern is the discrete-event analogue.
- [[Papers/Alvelid 2022]] — event-driven STED does one spot per detected trigger; multi-spot collapses many triggers into a single firing.

## Validated on

- **ch591 closed-loop SLM wave nucleation in reaction-diffusion** (2026-04-25): sequential firing plateaued at 1-3 nucleations across 6 rounds; switching to simultaneous multi-spot firing hit 6/6 on the first attempt. Converged 10/10.
