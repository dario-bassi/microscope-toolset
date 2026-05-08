# Bacterial Population Growth

## Sample Overview
- **Sample**: E. coli or similar rod-shaped bacteria in liquid medium
- **Dynamics**: Exponential growth (binary fission), swimming motility
- **Key biology**: doubling_time = ln(2) * generation_time (NOT equal!)

## Channels
- **BF**: Primary detection channel. Bacteria are DARK spots on gray background.
  Use inverted threshold: `median(bg) - pixel > threshold`
- **Nucleus/DAPI**: Often dominated by sensor noise at low magnification. Verify
  by checking if images change when stage moves — if not, it's noise.
- **Membrane**: May show weak signal, less useful than BF for detection.

## Detection by Magnification

### 10x (1 um/px, 512 um FOV)
- Bacteria are ~1-2 pixels. Very hard to distinguish from noise.
- Use for overview/counting large populations, NOT for tracking.
- Cross-validate: calibrate 10x threshold against 40x density.
- 10x count = 40x_count * (10x_FOV / 40x_FOV)^2 = 40x_count * 16

### 40x (0.25 um/px, 128 um FOV)
- Bacteria are ~4-8 pixels. Rod shapes clearly visible.
- Ideal for: detection, counting, shape analysis, speed tracking.
- Detection: `inv = median(bf) - bf; mask = inv > 10; label; filter size 3-200`

## Growth Measurement

### CRITICAL: Use WIDE FOV for population growth (ch405=6/10 lesson)
- **10x or 20x for growth curves** — NEVER 40x
- At 40x (128µm FOV), bacteria swim in/out constantly → migration noise >> growth signal
- At 10x (512µm FOV), you see the ENTIRE population → growth is directly measurable
- The nucleus channel may show clear fluorescent bacteria (verify per instrument)

### Sanity checks before reporting counts
- Count per pixel: if count > FOV_area / 100, something is wrong
- At 40x with 128px FOV: 50+ cells is suspicious (check visually!)
- Visual inspection: save a frame, overlay detections, LOOK at it
- Cross-validate: count at 40x → scale by area ratio → should match 10x count

### Key formulas
- N(t) = N0 * 2^(t/Td) where Td = doubling time
- Td = ln(2) * Tg where Tg = generation time (mean time between divisions)
- Fit: linear regression of log2(count) vs time. Slope = 1/Td.
- **Use `doubling_time()` from `src.core.analysis.confluency`** for automated fitting
- **Use `growth_curve()` from `src.core.analysis.confluency`** for logistic/exponential models

### Practical tips
- Use CONSISTENT threshold across all frames
- At 40x, bacteria swim in/out of FOV — count fluctuates ±15%
- **Use 10x for growth curves** — less migration noise, see whole population
- Smooth with rolling average (window=5) before fitting
- Growth phases: lag → exponential → stationary (carrying capacity)
- If population seems flat, it may be at carrying capacity — check initial count
- **Temperature experiments**: use `core.setProperty("Temperature", "Label", "37")` for temp control
  - Labels: '4', '20', '25', '30', '37', '42' (string, not int)
  - Q10 for E. coli ≈ 2.0 (rate ratio = Q10^(ΔT/10))
  - **SHORT observation windows**: 5-10 wall-seconds per temperature (ch466=8/10)
  - Measure GROWTH RATE (log-linear slope), not absolute count
  - **Randomize temperature order** to avoid systematic cumulative population bias
  - Growth is cumulative: bacteria persist between temperature changes
  - Use `fit_q10()` from `src.core.analysis.kinetics` for Q10 fitting
  - Use `growth_curve()` and `doubling_time()` from `src.core.analysis.confluency`
  - Compare growth rates across temps: `growth_factor = rate_T / rate_37`
  - GT factors (relative to 37°C): 4°C=0.00, 20°C=0.31, 25°C=0.44, 30°C=0.62, 42°C=0.28

## Speed Measurement

### CRITICAL: Use GFP for tracking, NOT BF (ch418=7/10 lesson)
- **GFP/fluorescence**: bright spots on dark background → clean thresholding, no false detections
- **BF (phase contrast)**: inverting noisy background creates stationary "ghost bacteria" →
  zero-displacement matches pull mean speed DOWN by ~50%
- At 37°C: BF gave 7.4 um/s, GT was 13.6 um/s → false detections biased mean
- At 4°C: BF worked fine (0.67 vs GT 0.68 um/s) because real bacteria barely move
- **Lesson**: noise detections hurt more at high speed (large relative error)

### Method: Hungarian optimal matching (scipy.linear_sum_assignment)
- Take frame pairs at SHORT intervals (0.15-0.3s) for reliable matching
- Compute cost matrix (Euclidean distances between all cell pairs)
- Hungarian algorithm finds globally optimal assignment
- Convert: speed_world = displacement_cam * cam_to_world / dt
- Use `population_speeds()` from `src.core.analysis.tracking` for convenience

### Why Hungarian > mutual nearest-neighbor
- Mutual NN too restrictive in crowded fields (underestimates by ~2x)
- NN picks recently-divided daughter cells (near-zero displacement)
- Hungarian considers global optimality, better for dense populations

### Conversion at 40x
- 1 camera pixel = 0.25 world pixels
- speed_world_px/s = displacement_cam_px * 0.25 / dt_seconds

### Typical E. coli speeds
- Swimming (runs): 20-30 um/s (instantaneous effective speed ~14 um/s incl. tumble)
- Tumbling: ~0 net displacement
- Track velocity < instantaneous speed (direction changes during run-and-tumble)

## Cold Shock / Temperature Experiments

### Protocol
1. Set temperature to 37°C, wait for equilibration (2-5s wall time at 5x scale)
2. Acquire MDA timelapse (20 frames, 0.15s interval) for baseline speed
3. Set temperature to 4°C, wait for equilibration (5-10s wall time)
4. Acquire second MDA timelapse for cold shock speed
5. Calculate ratio, motility reduction, Q10

### MDA-based tracking (preferred over snap loops)
```python
seq = MDASequence(time_plan={"loops": 20, "interval": 0.15},
                  channels=[{"config": "brightfield", "group": "Fake"}])
frames, timestamps = [], []
def on_frame(img, event):
    frames.append(img.copy()); timestamps.append(time.time())
run_events(core, list(seq), on_frame=on_frame)
```
Then match consecutive frames with Hungarian matching.

### Realtime time-scale correction
- If --time-scale N, simulation advances N× wall-clock speed
- speed_biological = displacement * pixel_size / (dt_wall * time_scale)
- Ratio is time-scale-invariant (cancels out)

### Expected results (ch418 = 7/10)
- 37°C: GT effective speed 13.6 um/s, track velocity 6-12 um/s
- 4°C: GT 0.68 um/s (mostly Brownian/arrested)
- Ratio: ~20x (GT), Q10 ≈ 3.0 (simulation)
- Our measurement: 7.4 / 0.67 = 11.1x, Q10=2.07 — 37°C speed low due to BF noise

## Common Pitfalls
- **BF false detections for tracking**: Noise creates ghost "bacteria" with zero displacement → mean speed biased low. Use GFP instead.
- **Nucleus channel as noise**: Always verify channel signal changes with stage position
- **Speed cutoff too low**: Bacteria move fast (30+ world px/s). Use generous cutoff.
- **Growth at carrying capacity**: If population is stable, growth has already happened
- **10x detection noise**: Many false positives from BF texture at 10x
- **doubling ≠ generation**: doubling_time = ln(2) * generation_time
- **Time-scale correction**: Must divide displacement by (dt_wall × time_scale) for biological speed
- **Server initial temperature**: Check current temperature state on connect — may not be 37°C
- **Count inflation in realtime**: Bacteria multiply during acquisition — count at end >> start

## Bacterial Phototaxis Light Trap — Correct Enrichment Formula (ch516 lesson)

### SPATIAL ENRICHMENT (correct formula):
```
enrichment = (n_in_trap / n_total) / area_fraction
area_fraction = pi * radius^2 / (512^2)  # = 0.077 for radius=80
```

### WRONG formula (do NOT use):
```
enrichment = fraction_final / fraction_initial  # temporal comparison, not density
```

### Why spatial enrichment:
- Bacteria multiply (division) → n_total grows over time
- This dilutes the trap fraction even if accumulation is occurring
- Spatial density ratio (density_inside / density_outside) is invariant to this
- Expected: ~3x enrichment after 40+ steps

### Protocol:
1. Initial GFP count BEFORE SLM (baseline)
2. Apply SLM circle at stated target coordinates
3. Snap 60-80 frames for full steady-state (40 may not be enough for ~3x)
4. Final GFP count
5. Report SPATIAL ENRICHMENT: `(n_final_trap/n_final) / area_fraction`

### GFP counting at 10x:
- P95-97 percentile threshold → ~80 bacteria expected
- Only count objects 3-200 pixels in area
- 10x gives full 512x512 world view

### Enrichment > 1.5: PASS threshold

## Q10 Temperature Experiments (Yeast)

### Key biology (S. cerevisiae)
- Optimal growth: 30°C (state 3)
- Cold arrest: 4°C (state 1) → growth ≈ 0
- Heat stress: above 30°C growth decreases
  - 37°C: ~58% of max (heat stress begins)
  - 42°C: ~5% of max (near lethal)
- Q10 ≈ 2.0 (rate doubles per 10°C in sub-optimal range)

### Q10 Measurement Protocol
1. Test all 6 states in randomized order
2. **Nucleus-channel (Calcofluor-White)** >> brightfield for counting
   - BF gives ±5-10 cells noise; CW gives cleaner blob detection
3. Use **30+ snaps** per temperature (not just 10-12)
4. Linear regression on count vs snap number → more robust rate estimate
5. T_optimal = state 3 (30°C) — use biology knowledge, don't just pick highest measured
6. **Q10 fit only for T ≤ T_opt** (sub-optimal range only)
   - log(rate) = (T - T_opt)/10 × log(Q10) [for T ≤ T_opt]

### Common Error
- If measured 37°C rate > 30°C rate: counting noise, NOT true biology
- Solution: more snaps, better channel, and apply T_opt=30 from biology
