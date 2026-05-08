# Pre-Submission Checklist

Run through this BEFORE submitting any challenge answer.

## 0. Render the overlay and Read it (non-negotiable)

Before anything else: render your detections / measurements as an overlay on the source image, save the PNG, then **Read the PNG and describe what you see.** The only number worth trusting is one you've seen confirmed on the image.

See [[Core/Approach/Visual verification]] for the full rule — four "always verify" triggers (first detection on a new sample, count differs from expectation by >30%, after changing detection parameters, before submitting final results). The habit is source-invariant: verify fresh inline code and pedigreed recipes alike.

**On ch585 r5 all four triggers fired and none were consulted** because the recipe had an 8/10 pedigree and the output felt trustworthy. That is exactly when verification is most needed — old recipes inherit drift. Render, Read, then submit; in that order; every time.

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

## 2. Run preflight checks (MANDATORY)

Three guards in **increasing order of cost** — apply each in turn, stop early
if any returns `block`. Each catches a distinct failure mode:

### 2a. Sign-check + preflight (cheapest, always run)

```python
from src.core.utils.preflight import run_preflight
result = run_preflight(
    answer, pixel_size_um=ps, n_visible_objects=n_vis,
    required_keys=['count', 'speed'],  # from extract_answer_keys()
)
if not result['passed']:
    print("ERRORS:", result['errors'])  # FIX before submitting
for w in result['warnings']:
    print("WARNING:", w)
```

Catches: NaN/Inf, missing keys, PSF compression, track fragmentation,
half-time plausibility, fractional counts, **physically-impossible signs**
(negative closure rates, ratios outside [0, 1]). Cheap, deterministic.

Domain-specific:
```python
from src.core.analysis.measurement_validator import run_sanity_checks
run_sanity_checks({'count': n_cells, 'areas': cell_areas},
                  fov_um2=fov_area, pixel_size=px_size, cell_type='mammalian')
```

### 2b. `render_vs_submit_check` — agent-side apparent-vs-underlying guard

```python
from src.core.utils.render_vs_submit import render_vs_submit_check

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

Catches: the value the agent is about to submit *disagrees with what the
recipe's own detector finds when re-run on the rendered image*. This is
exactly the ch593 / ch599 / ch348 failure mode (apparent-vs-underlying).
Deterministic numerical guard; complementary to 2c. Severity ladder
ok / flag / block / error.

### 2c. `pre-submit-review` skill — overlay + visual sanity (most expensive)

For non-trivial submissions, invoke the `agent/skills/pre-submit-review.md`
skill *after* 2a + 2b pass. It dumps the inputs to `/tmp/ch<N>_pre_submit/`
and dispatches a sonnet subagent that generates overlays, Reads them with
vision, and reviews the solve script for platform-use. Returns SHIP /
FIX-NOW / SHIP-WITH-FLAG.

**Do not skip 2a or 2b.** 2c is a gentler net — it catches things 2a/2b
miss, but if 2b returns `block` your submission is wrong before the
sonnet ever sees it.

### 2d. Or invoke the orchestrator (`run_presubmit_guards`)

When you want all three glued together with early-exit-on-block and a
single `PresubmitResult`, call the sprint #22 orchestrator:

```python
from src.core.utils.presubmit import run_presubmit_guards

result = run_presubmit_guards(
    answer=answer,
    pixel_size_um=ps,
    image=final_snap,
    submitted_value=answer["count"],
    recipe_re_detect_fn=lambda img: len(detect_cells(img, **kw)),
    run_review=True, challenge_id=N, source_images={"snap": final_snap},
)
if result.severity == "block":
    raise SystemExit(result.rationale)
if result.severity == "flag":
    method_description += f"  [{result.rationale}]"
if result.review_prep_path:
    # invoke the pre-submit-review skill against this dir
    ...
```

The orchestrator is opt-in, not a replacement: 2a / 2b / 2c each still
have their direct entry points. Use the orchestrator when the recipe
benefits from a single severity verdict; use them individually when
you want fine-grained control over branching.

**Production example:** see `src/recipes/frap_background_correction.py:
analyze_frap_with_guards` for a worked recipe-side wrapper, and
`scratch/template_auto_solve.py` for the solve-script side.

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

Protocol-compliance is a consistent source of lost points.

- [ ] How many frames/steps does the challenge require? (e.g., "40-50 pairs")
- [ ] Which channels to acquire? (e.g., BF + a fluorescence channel, not just one)
- [ ] Are there specific timing requirements? (interval, total duration)
- [ ] When does the phenomenon of interest occur? (onset step, drug time)
- [ ] Am I acquiring enough frames PAST the onset to see the full effect?

**Rule: Take 2x the minimum recommended frames when possible.**

## 9. Did I report ALL requested outputs?

Re-read the challenge description one final time. Check for:
- [ ] Onset step/time (when does the change begin?)
- [ ] Wave characteristics (period, speed, frequency)
- [ ] Specific metric names that MUST be in the answer dict
- [ ] Null values — did I fail to compute something? (e.g., onset_step=None)

## 10. Null results are valid results

If the expected phenomenon didn't happen, **report it honestly**
(wave_speed=0, onset_frame=None, dead_count=0) with a method
description that names what you tried and what you observed. A
well-characterized null result is a legitimate scientific
measurement. Graders reward "I measured carefully and the signal
wasn't there" over "here is a made-up number that pattern-matches
a typical answer".

Rule of thumb before making up a number: can you defend the
measurement with an image showing what you saw? If the image
shows a flat baseline with no wavefront, wave_speed=0 is the
correct scientific answer. Guessing 5 px/frame because "waves
usually propagate" is not.

## 11. A working preflight is the answer — don't reroll

If the preflight pass produces a number that already passes the
brief's tolerance, **submit that exact number**. Do not run the
solve fresh "to be sure" or pivot to a "more elegant" / "more
canonical" strategy. Every reroll resamples sim state and burns
fresh variance.

ch651 r2 dry-run produced (134, 132) [GT (144, 116) → 19 px,
inside the ±25 px tolerance]. The live r2 then ran the same
script fresh and landed (130, 40) [77 px, top-edge artifact].
r3 pivoted to a "cleaner short-burst" approach and landed
(439, 214) [311 px, ectopic mistaken for primary]. Challenge
locked at 4/10. The 19-px answer was sitting in `--dry` output
the whole time.

Practical rule:
- For every non-trivial submission, run a `--dry` pass first.
- If the dry-run answer meets tolerance, pickle / paste / capture
  it directly into the submit call.
- Only re-run if the dry-run failed tolerance. The next sample
  is *not* automatically better; it carries new noise.
- Closed-loop or dynamic scenarios (cardio, calcium-wave) re-
  randomise σ × mean rankings every fresh server — the first
  good answer is often the only good answer.

This is **not** the same as the "verify visually before
submitting" rule (still mandatory). It's about not wasting a
passing answer by re-running the strategy.

## Common failure modes to check

| Category | Check | Common Error |
|----------|-------|-------------|
| H&E tissue | Necrosis? | Assumed no necrosis without testing |
| Fluorescence | Background subtracted? | Halved intensity by including background |
| Tracking | Smoothed trajectory? | Raw centroid ± body undulation |
| Counting | Edge handling? | Over-excluded (10px) or under-excluded |
| Counting | [[Visual verification\|Rendered + Read overlay]]? | Submitted a count without confirming it on the image |
| Multi-scale | Used `set_objective_verified()`? | Objective didn't switch |
| Timelapse | Used `track_mean_intensity()`? | Threshold artifact on half-time |
| Dose-response | Used last frames? | Averaged ramp-up with steady state |
| Protocol | Enough frames? | Too few frames → missed phenomenon entirely |
| Protocol | All channels? | Single-channel when dual required |
| Tracking | `validate_tracks()` passed? | Track fragmentation inflated count 2-3× |
| Counting | `estimate_required_n()` sufficient? | Too few FOVs for reliable proportion |
| Watershed | Used `min_distance='auto'`? | Hardcoded min_distance → undercount |
| Behavioral | Started tracking immediately? | Survey snaps consumed critical frames |
| Fluorescence | Used `estimate_noise_floor()`? | Arbitrary threshold → over/underdetection |
| Overview | Used `snap_all_channels()`? | Missed signal in unexpected channel |
| Parameters | Used `parameter_advisor`? | Hardcoded pixel-domain params wrong for magnification |
| N:C ratio | Used `suggest_nc_params()`? | PSF blur compresses N:C range at low mag |
| Kinetics | Used `value_at_time()`? | Reported last-frame value, not exact target time |
