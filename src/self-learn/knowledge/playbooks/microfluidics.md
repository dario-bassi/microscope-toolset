# Microfluidic Cell Sorting

## Sample Overview
- **Device**: Horizontal microfluidic channel with trap constrictions
- **Flow direction**: Left-to-right
- **Purpose**: Size-based cell trapping and sorting

## Channels
- **BF**: Cells and channel structures (both visible)
- **nucleus-channel (DAPI)**: Nuclear fluorescence, **brighter in compressed/trapped cells**

## Cell Size Classes
- **Small**: r=3-5px → pass through traps
- **Medium**: r=5-8px → get trapped
- **Large**: r=8-12px → get trapped

## Trapped vs Flowing Detection

### Method 1: Position + Location (Most Reliable)
- Snap 2+ frames under **Fast** flow (slow flow may not move free cells enough)
- Compare cell positions across frames
- **Stationary** = trapped at constrictions
- **Moving** = flowing through channel
- **CRITICAL (ch410=7)**: Position comparison alone overcounts trapped cells.
  Slow-moving cells appear stationary over 1-2 steps. Require BOTH:
  1. Cell is stationary across frames
  2. Cell is **spatially at a constriction** (within ~10px of a detected trap)
- Use multiple flow rates (Off → Slow → Fast) to confirm trapping

### Method 2: Size-Based Prediction
- Only medium and large cells get trapped
- Small cells always pass through
- Useful for quick classification without multi-frame analysis

### Method 3: Nucleus Intensity
- Trapped cells have **brighter nuclei** due to compression
- Use nucleus-channel for cleaner detection than BF
- Threshold on nuclear brightness to identify trapped cells

## SLM-Based Cell Release

### Workflow
1. Identify trapped cells (position or size)
2. Create uint8 SLM mask (matching SLM resolution)
3. Draw bright circles (255) around cells to release
4. `core.setSLMDevice("SLM")`
5. Apply SLM mask **before each snap** (not just once)
6. Snap to check release status

## Cell Counting
- **CRITICAL**: Count cells per-frame, not across multiple frames
- Tracking same cells across frames causes double-counting
- Each frame is an independent measurement

## Best Practices
- Use nucleus-channel for cleaner segmentation than BF
- Prefer position comparison over size alone for trapped/flowing classification
- Remember SLM must be reapplied before each snap in closed-loop control
- Channel structures visible in BF can help identify trap locations

---

# Microfluidic Perfusion Control

## Sample Overview
- **Device**: Microfluidic chamber connected to perfusion pump
- **Purpose**: Drug delivery to cells in controlled flow
- **Flow**: Continuous or pulsed perfusion of media +/- drug

## Key Devices
- **Perfusion pump**: Controls flow rate and drug concentration
  - Check `core.getLoadedDevices()` for pump/perfusion device name
  - May use states (0=off, 1=low, 2=medium, 3=high) or continuous control
  - May have separate drug concentration property
- **Temperature controller**: For temperature-sensitive experiments

## Perfusion Protocol
1. **Baseline**: Image cells for N frames with normal media (no drug)
2. **Drug addition**: Set perfusion pump to deliver drug
3. **Response monitoring**: Continue imaging during drug exposure
4. **Washout** (optional): Switch back to normal media, monitor recovery

## Measurement Strategy
- **Before drug**: Establish baseline (cell count, morphology, fluorescence)
- **During drug**: Track temporal changes with `measure_response_time()`
- **After washout**: Monitor recovery
- **Key metrics**: Response time, half-time, total change, direction

## Analysis Functions Available
- `measure_response_time()` — step-change kinetics (drug onset/offset)
- `fft_spectrum()` — detect periodic responses
- `detect_peaks()` — event detection in time series
- `detrend()` — remove drift before analysis
- `compute_snr()` — verify fluorescence signal quality

## Common Pitfalls
- **Drug onset is a RAMP, not a step** — mixing takes time in microfluidics
- **For steady-state measurement**: use LAST frames, not average of all post-drug frames
- **Photobleaching**: consider that fluorescence decays even without drug effect
- **Flow artifacts**: cells may move/deform due to flow, not drug
- **Temperature effects**: perfusate may be different temperature than chamber
