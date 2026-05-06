# Pre-Submission Checklist

Run through this BEFORE submitting any challenge answer.

## 1. Did I analyze ALL required metrics?

Re-read the challenge description. For each requested metric:
- [ ] Is it in my answer dict?
- [ ] Did I actually compute it (not just comment/assume)?
- [ ] Did I use appropriate methodology?

**Common misses:**
- Necrosis check in histology (must use eosin channel, not just visual)
- Edge handling (objects at FOV boundary)
- Background subtraction for fluorescence intensity
- Channel selection verification

## 2. Are my measurements plausible?

Run `validate_cell_count()`, `validate_size()`, etc. from `measurement_validator.py`:

```python
from src.analysis.measurement_validator import run_sanity_checks
result = run_sanity_checks(
    {'count': n_cells, 'areas': cell_areas},
    fov_um2=fov_area, pixel_size=px_size, cell_type='mammalian'
)
if result['warnings']:
    print("WARNING:", result['warnings'])
```

## 3. Did I use the right magnification?

- Started with 10x overview? (Almost always required)
- Used 20x/40x for detail measurements?
- Is the pixel_size correct for the current objective?

## 4. Did I use the right channel?

- Snapped ALL available channels?
- Verified the signal is in the expected channel?
- Challenge description may say "GFP" but the config name could be different

## 5. Is the analysis internally consistent?

- Cell count ÷ FOV area = reasonable density?
- Measured sizes match what I see in the image?
- Temporal trends make biological sense (monotonic growth, exponential decay)?

## 6. Did I save the required outputs?

- [ ] Showcase image (`../logs/showcase/agent_ch{N}_{desc}.png`)
- [ ] Solve script (`scratch/solve_{N}.py`)
- [ ] Answer dict has all required keys

## 7. Method description mentions key approaches

- MDA usage (MDASequence + run_events)
- Key analysis functions from src/
- Channel names used
- Magnification steps

## 8. Did I follow the acquisition protocol?

**This is the #1 source of lost points (ch467=4, ch457=4, ch451=4).**

- [ ] How many frames/steps does the challenge require? (e.g., "40-50 pairs")
- [ ] Which channels to acquire? (BF+GFP pairs, not just one)
- [ ] Are there specific timing requirements? (interval, total duration)
- [ ] Does the simulation advance per-snap or per-pair? (snaps_per_step)
- [ ] When does the phenomenon of interest occur? (onset step, drug time)
- [ ] Am I acquiring enough steps PAST the onset to see the full effect?

**Rule: Take 2x the minimum recommended frames when possible.**

## 9. Did I report ALL requested outputs?

Re-read the challenge description one final time. Check for:
- [ ] Onset step/time (when does the change begin?)
- [ ] Wave characteristics (period, speed, frequency)
- [ ] Specific metric names that MUST be in the answer dict
- [ ] Null values — did I fail to compute something? (e.g., onset_step=None)

## Common failure modes to check

| Category | Check | Common Error |
|----------|-------|-------------|
| H&E tissue | Necrosis? | Assumed no necrosis without testing |
| Fluorescence | Background subtracted? | Halved intensity by including background |
| Tracking | Smoothed trajectory? | Raw centroid ± body undulation |
| Counting | Edge handling? | Over-excluded (10px) or under-excluded |
| Multi-scale | Verified at 2nd mag? | Wrong threshold for different magnification |
| Timelapse | Fixed threshold? | Adaptive threshold masked real change |
| Dose-response | Used last frames? | Averaged ramp-up with steady state |
| Protocol | Enough frames? | Too few frames → missed phenomenon entirely |
| Protocol | All channels? | Single-channel when dual required |
| Behavioral | Started tracking immediately? | Survey snaps consumed critical frames |
