# Measurement Methodology Lessons

## Recurring Theme: **Measure Correctly, Not Just Detect Correctly**

Detection accuracy can be excellent while measurement methodology is wrong. Common patterns:
- Didn't analyze temporal dynamics (static snapshot for dynamic classification)
- Measured cumulative coverage instead of instantaneous (wave ring vs filled circle)
- Measured truncated cells at wrong magnification
- Noise classified as dead cells (no morphological filtering)

## Generalizable Workflows That Work

### 1. Radial Density Profiling (ZOI measurement)
**Proved excellent (typically within 3% error).**
- Compute bacteria/cell density as function of distance from center
- Works for: disk diffusion, antibiotic response, any circular gradient
- Implementation: annular rings (10px bins), count objects per unit area
- Edge detection: steepest gradient in smoothed profile

### 2. SLM Wave Nucleation (optogenetics)
- Apply circular SLM mask at target location
- Keep SLM on long enough to ensure nucleation (verify empirically per system)
- Remove SLM — wave self-sustains
- Track front radius (90th percentile of active pixel distances)
- Linear fit of radius vs frame → speed

### 3. Multi-Scale Workflow (10x→20x→40x)
**Correct approach but must account for FOV limitations.**
- Low mag: overview, count, orientation (largest FOV)
- Mid mag: detail measurements
- High mag: fine features like wall thickness (smallest FOV)
- **CRITICAL**: Only measure objects FULLY CONTAINED within FOV
- If object > 0.5 × FOV, use lower magnification

### 4. Fluorescence-based Bacteria Detection
**Better than BF for small cells.**
- BF contrast too low for rod bacteria at low magnification
- Fluorescence threshold gives cleaner detection
- But beware noise amplification for "dead" cell detection

## Anti-Patterns to Avoid

### 1. Static Classification of Dynamic Patterns
- WRONG: classify reaction-diffusion from single frame
- RIGHT: compare 3-4 timepoints, track feature splitting/growth

### 2. Cumulative vs Instantaneous Metrics
- WRONG: "coverage" = all pixels ever active (FHN wave)
- RIGHT: coverage = currently active at measurement time
- FHN/excitable media produce ring wavefronts with recovery behind

### 3. Measuring Truncated Objects
- WRONG: use regionprops on objects touching FOV border
- RIGHT: check bounding box fits within FOV (>10px margin)
- If most objects touch border → wrong magnification for measurement

### 4. Intensity-Only Classification Without Morphology
- WRONG: GFP < 15 = dead cell (captures noise)
- RIGHT: require rod shape (AR > 2) AND minimum area AND intensity criterion
- Ghost cells have specific morphology, not just dim signal

## Key Insight: Measurement Methodology > Detection Accuracy

What differentiates good from great results is whether the MEASUREMENT approach is correct:
- What to measure (instantaneous vs cumulative)
- Where to measure (right magnification for object size)
- How to validate (temporal dynamics, morphological filters)
- Which convention (growth RATIO not rate, alive not total)

Detection code can be mature while measurement methodology needs work.
