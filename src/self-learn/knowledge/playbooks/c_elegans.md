# C. elegans Locomotion Tracking

## Sample Overview
- **Sample**: Freely crawling worm with sinusoidal body waves
- **Movement pattern**: Characteristic undulation, moves continuously
- **Risk**: Worm exits FOV quickly without stage tracking

## Channels
- **BF**: Body outline and structure
- **GFP/nucleus-channel**: Pharynx (head marker) - brightest fluorescence
- **membrane-channel**: Body wall muscles

## Standard Workflow

### 1. Initial Setup
- Start at 10x objective (query pixel size and FOV from hardware)
- Check available channels: `core.getAvailableConfigGroups()`
- Set appropriate config group for multi-channel imaging

### 2. Head Detection
- Snap GFP/nucleus channel
- Find brightest blob = pharyngeal fluorescence = head
- Use for orientation tracking and directional analysis

### 3. Body Length Measurement
- Snap BF channel to get body outline
- Segment worm: largest connected component from thresholded BF
- Skeletonize clean worm mask
- Measure skeleton path length
- Adult C. elegans: ~1mm body length

### 4. Speed Measurement (CRITICAL)
- **WARNING**: Raw centroid tracking OVERESTIMATES speed significantly
- **Reason**: Sinusoidal undulation causes zig-zag centroid trajectory
- **Solution**: Apply trajectory smoothing before computing speed
  - Savitzky-Golay filter (window=5, polyorder=2) works well
  - Rolling average is an alternative
  - Net displacement / time gives minimum (straight-line) speed
- **Correct formula**: speed = sum(smoothed frame-to-frame distances) / total_time
- For multi-channel acquisition, account for time between channel switches

### 5. Stage Tracking
- C. elegans crawls at ~0.1-0.3 mm/s (100-300 um/s)
- At 10x (FOV ~500 um), worm crosses FOV in seconds
- **Must** move stage to keep worm centered, or it exits FOV quickly
- Track centroid position in pixel coordinates
- Move stage when worm approaches FOV edges (e.g., >100px from center)
- Use world coordinates for absolute position tracking

### 6. Coordinate Conversion
- **World coordinates**: `world = stage + (pixel - center) * pixel_size`
- Query image center and pixel size from hardware config
- Use for absolute position tracking across stage movements

### 7. Adaptive MDA for Stage Tracking
- Use adaptive generator pattern: yield BF + fluorescence events per round
- In on_frame callback, detect worm and update shared state for next round
- Generator reads updated position when yielding next event pair
- Each BF + fluorescence pair = 2 snaps = 1 dynamics step (if auto-stepping)
```python
def tracking_gen():
    for r in range(N_ROUNDS):
        yield MDAEvent(channel={"config": "brightfield", ...},
                       x_pos=shared['next_x'], y_pos=shared['next_y'])
        yield MDAEvent(channel={"config": "nucleus-channel", ...},
                       x_pos=shared['next_x'], y_pos=shared['next_y'])
run_events(core, tracking_gen(), on_frame=on_frame)
```

### 8. BF Worm Detection
- Worm is DARK against bright BF background
- Detection: `mask = bf < (percentile_90 - 15)` → largest CC → centroid
- At 10x: worm is ~100 camera pixels, low contrast, need local variance map
- At 20x: worm fills ~200 camera pixels, clearly visible
- Body COM tracks body posture as well as translation — adds noise

### 9. Pharynx-Based Speed Measurement (ch412)
- **Best approach**: Track pharynx (brightest blob in GFP) instead of body COM
- Pharynx is a small, well-defined point → less affected by undulation
- Body COM overestimates by ~45% raw, ~15% even after smoothing
- Smoothed pharynx speed is most accurate translocation measurement

## CRITICAL: Behavioral Event Detection (ch457 lesson, cost 6 points)

**The behavioral event may happen AUTOMATICALLY during observation.**
- Do NOT waste frames on surveys, channel scouting, or setup
- Track from the VERY FIRST snap — every frame counts
- Plot per-frame speed in real-time; look for sharp transitions
- The challenge notes said "behavioral change occurs during observation" = it's automatic
- ch457: Levamisole (paralytic) was applied at step 15 → worm speed: 50 → 0 px/step by step 19
- My survey snaps consumed frames 0-14, missing the entire onset
- All subsequent experiments (drug, temperature) were on an already-paralyzed worm

**Correct approach:**
1. Start tracking at default position (1024, 1024) with ZERO survey snaps
   - ch459 feedback: 6 survey snaps still lost pre-event frames
   - The sample is centered or findable from default start position
2. Track immediately — single-channel, every frame matters
3. After 50+ frames, analyze per-frame speed for sharp transitions
4. ONLY then attempt interventions if no event detected yet
5. Report: onset frame, pre-event speed, post-event speed, transition duration

## Common Pitfalls
- Not smoothing trajectory → significant speed overestimation
- Not moving stage → worm exits FOV
- Using wrong channel for head detection → pharynx is brightest in GFP
- In multi-channel timelapse: each channel snap takes time, total acquisition
  duration = n_channels × n_timepoints × exposure_time
- **World coordinate origin**: Stage may start at (0,0) which can be a
  CORNER of the world, not the center. Use `find_sample()` from scouting.py
  to locate the worm, or check BF min values across a grid search.
- **Use find_sample() for initial localization**: The worm can be anywhere in a
  large world. Don't manually search — use `find_sample(core, 'brightfield',
  search_range=2048)` to efficiently find it.
- **MAGNIFICATION MATTERS FOR SLOW MOVEMENTS (ch452=6)**:
  - 10x is good for BASELINE tracking (fast worm, large FOV margin)
  - But at 4°C, worm moves ~0.6 px/s. Centroid jitter at 10x (~2-3 px) DOMINATES
  - Must use 20x (or 40x) for cold/drugged measurements where movement is sub-pixel
  - **Switch magnification between phases**: 10x baseline → 20x cold
  - The centroid noise floor sets a minimum detectable speed per magnification
- **MDA-based tracking**: Use `track_target_mda()` from stage_tracking.py
  for MDA-native closed-loop tracking. Returns (generator, on_frame, state).
- **BF detection parameters (ch452)**: threshold = `p90 - 12`, area filter
  300-30000 px, edge exclusion 40px margin, position matching for continuity.
- **Temperature Q10 for speed**: C. elegans speed decreases with temperature.
  Measured Q10 ≈ 2.6 (ch452): 20°C → 4°C gave 12.5 → 2.7 um/s (4.6x slowdown).
- **Savitzky-Golay smoothing**: window=9, polyorder=2 works well for body
  centroid trajectory at 10x. Reduces undulation noise by ~60%.
