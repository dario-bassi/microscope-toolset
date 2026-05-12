# Experiment Verification Checklist

Run through this BEFORE recording or reporting any experiment results.

## 0. Render the overlay and Read it (non-negotiable)

Before anything else: render your detections / measurements as an overlay on the source image, save the PNG, then **Read the PNG and describe what you see.** The only number worth trusting is one you've seen confirmed on the image.

See [[Core/Approach/Visual verification]] for the full rule — four "always verify" triggers:
1. First detection on a new sample (are the circles on real cells?)
2. When automated count differs from expectation by >30%
3. After changing detection parameters
4. Before reporting final results

Verify fresh inline code and pedigreed recipes alike. Render, Read, then report; in that order; every time.

## 1. Did I analyze ALL required metrics?

Re-read the experiment description. For each requested metric:
- [ ] Is it in my result dict?
- [ ] Did I actually compute it (not just comment/assume)?
- [ ] Did I use appropriate methodology?

**Common misses:**
- Necrosis check in histology (must use eosin channel, not just visual)
- Edge handling (objects at FOV boundary)
- Background subtraction for fluorescence intensity
- Channel selection verification

## 2. Run preflight checks (MANDATORY)

Three guards in **increasing order of cost** — apply each in turn, stop early if any returns `block`:

### 2a. Sign-check + preflight (cheapest, always run)

```python
from self_learn.utils.preflight import run_preflight
result = run_preflight(
    answer, pixel_size_um=ps, n_visible_objects=n_vis,
    required_keys=['count', 'speed'],
)
if not result['passed']:
    print("ERRORS:", result['errors'])  # FIX before recording
for w in result['warnings']:
    print("WARNING:", w)
```

Catches: NaN/Inf, missing keys, PSF compression, track fragmentation,
half-time plausibility, fractional counts, **physically-impossible signs**
(negative closure rates, ratios outside [0, 1]). Cheap, deterministic.

Domain-specific:
```python
from self_learn.analysis.measurement_validator import run_sanity_checks
run_sanity_checks({'count': n_cells, 'areas': cell_areas},
                  fov_um2=fov_area, pixel_size=px_size, cell_type='mammalian')
```

### 2b. `render_vs_submit_check` — apparent-vs-underlying guard

```python
from self_learn.utils.render_vs_submit import render_vs_submit_check

check = render_vs_submit_check(
    image=final_snap,
    submitted_value=answer["count"],
    recipe_re_detect_fn=lambda img: len(detect_cells(img, **detect_kwargs)),
)
if check.severity == "block":
    raise SystemExit(f"render_vs_submit blocked: {check.rationale}")
if check.severity == "flag":
    method_description += f"  [flag: {check.rationale}]"
```

Catches: the value you are about to record *disagrees with what your own
detector finds when re-run on the rendered image*. This covers the
apparent-vs-underlying failure mode (submitted a count from internal
state, not from what the image actually shows). Severity ladder:
ok / flag / block / error.

### 2c. Visual review (most expensive, for non-trivial results)

For non-trivial submissions, save overlay images and inspect with LLM
vision. This catches things 2a/2b miss — rendered overlay that doesn't
match the answer, code using the wrong primitive.

**Do not skip 2a or 2b.** 2c is a gentler net.

### 2d. Orchestrated guard (all three composed)

```python
from self_learn.utils.presubmit import run_presubmit_guards

result = run_presubmit_guards(
    answer=answer,
    pixel_size_um=ps,
    image=final_snap,
    submitted_value=answer["count"],
    recipe_re_detect_fn=lambda img: len(detect_cells(img, **kw)),
    run_review=True,
)
if result.severity == "block":
    raise SystemExit(result.rationale)
if result.severity == "flag":
    method_description += f"  [{result.rationale}]"
```

## 3. Did I use the right magnification?

- Started with 10x overview? (Almost always required for counting)
- Used 20x/40x for detail measurements?
- Is the pixel_size correct for the current objective?

## 4. Did I use the right channel?

- Snapped ALL available channels?
- Verified the signal is in the expected channel?
- The channel name may differ from what you expect

## 5. Is the analysis internally consistent?

- Cell count ÷ FOV area = reasonable density?
- Measured sizes match what you see in the image?
- Temporal trends make biological sense (monotonic growth, exponential decay)?

## 6. Did I save required experiment outputs?

- [ ] Overlay / showcase image (for visual verification and record-keeping)
- [ ] Analysis script / notebook (for reproducibility)
- [ ] Result dict has all required keys

## 7. Method description is complete

- Acquisition approach (MDA usage, channels, magnification steps)
- Key analysis functions used
- Any flags or caveats

## 8. Did I follow the acquisition protocol?

Protocol compliance prevents lost data:

- [ ] How many frames/steps does the experiment require?
- [ ] Which channels to acquire?
- [ ] Are there specific timing requirements? (interval, total duration)
- [ ] When does the phenomenon of interest occur?
- [ ] Did I acquire enough frames PAST the onset to see the full effect?

**Rule: Take 2× the minimum recommended frames when possible.**

## 9. Did I report ALL requested outputs?

Re-read the experiment description one final time. Check for:
- [ ] Onset step/time (when does the change begin?)
- [ ] Wave characteristics (period, speed, frequency)
- [ ] Specific metric names
- [ ] Null values — did I fail to compute something? (e.g., onset_step=None)

## 10. Null results are valid results

If the expected phenomenon didn't happen, **report it honestly**
(wave_speed=0, onset_frame=None, dead_count=0) with a method
description that names what you tried and what you observed. A
well-characterized null result is a legitimate scientific
measurement.

Rule of thumb: can you defend the measurement with an image showing
what you saw? If the image shows a flat baseline with no wavefront,
wave_speed=0 is the correct scientific answer.

## 11. Don't reroll a passing result

If verification passes on a result, **submit that exact result**. Do not
re-run the analysis "to be sure" or pivot to a "more elegant" strategy.
Every rerun resamples from the microscope / simulation and introduces
new variance.

Practical rule:
- If your first analysis meets tolerance, capture it directly.
- Only re-run if it failed tolerance.
- The next sample is *not* automatically better; it carries new noise.
- This rule does NOT replace visual verification (still mandatory).

## Common failure modes to check

| Category | Check | Common Error |
|----------|-------|-------------|
| H&E tissue | Necrosis? | Assumed no necrosis without testing |
| Fluorescence | Background subtracted? | Halved intensity by including background |
| Tracking | Smoothed trajectory? | Raw centroid ± body undulation |
| Counting | Edge handling? | Over-excluded or under-excluded |
| Counting | Rendered + Read overlay? | Recorded a count without confirming on image |
| Multi-scale | Objective switched correctly? | Objective didn't switch |
| Timelapse | Used `track_mean_intensity()`? | Threshold artifact on half-time |
| Dose-response | Used last frames? | Averaged ramp-up with steady state |
| Protocol | Enough frames? | Too few frames → missed phenomenon entirely |
| Protocol | All channels? | Single-channel when dual required |
| Tracking | `validate_tracks()` passed? | Track fragmentation inflated count 2-3× |
| Watershed | Used `min_distance='auto'`? | Hardcoded min_distance → undercount |
| Fluorescence | Used `estimate_noise_floor()`? | Arbitrary threshold → over/underdetection |
| Overview | Used `snap_all_channels()`? | Missed signal in unexpected channel |
