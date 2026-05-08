# Microfluidic Device — Cell Trapping, Sorting & Perfusion

## Sample Types

### Trap-Based Devices
- Horizontal channel with constriction traps
- Flow direction: left-to-right
- Purpose: size-based cell trapping and sorting
- Cells flow through channel; some captured by traps based on size

### Perfusion Devices
- Microfluidic chamber connected to perfusion pump
- Purpose: drug delivery to cells in controlled flow
- Flow: continuous or pulsed perfusion of media +/- drug

## Channels
- **BF**: Cells and channel structures (both visible)
- **nucleus-channel (DAPI)**: Nuclear fluorescence, **brighter in compressed/trapped cells**

## Trapped vs Flowing Classification

### Method 1: Spatial Position (Most Reliable)
- Trap positions are fixed X coordinates visible in BF
- Cells AT trap X positions = trapped, others = free
- Use BOTH position AND brightness for best results
- Detect trap X coordinates from BF: look for vertical constriction features

### Method 2: Temporal (Use With Caution)
- Snap 2+ frames under **Fast** flow (slow may not move free cells enough)
- Stationary = likely trapped; Moving = flowing
- **CRITICAL (ch410=7, ch455=5)**: Position comparison alone overcounts trapped cells.
  Slow-moving cells appear stationary over 1-2 steps. Require BOTH:
  1. Cell is stationary across frames
  2. Cell is **spatially at a constriction** (within ~10px of a detected trap)

### Method 3: Size-Based Prediction
- Small (r=3-5px) → pass through traps
- Medium (r=5-8px) → get trapped
- Large (r=8-12px) → get trapped

### Method 4: Nucleus Intensity
- Trapped cells have **brighter nuclei** due to compression
- Threshold on nuclear brightness to identify trapped cells

## SLM-Based Cell Release
1. Identify trapped cells (position or size)
2. Create uint8 SLM mask (matching SLM resolution)
3. Draw bright circles (255) around cells to release
4. `core.setSLMDevice("SLM")`
5. Apply SLM mask **before each snap** (not just once)
6. Snap to check release status

## Scanning Coverage
- Channel spans a Y range (e.g., Y=196-316)
- Scan multiple Y positions at 40x to cover full channel width
- Don't assume cells are only at Y=center; they distribute across width

## Deduplication
- World-coordinate deduplication must NOT be too aggressive
- 3µm grid snapping (ch455 approach) merged distinct cells
- Use `deduplicate_cells()` from `src/workflows/scanning.py` with proper merge_radius
- Each unique cell = unique world position; don't merge by size similarity

## Cell Counting
- Count cells per-frame, not across multiple frames
- Tracking same cells across frames causes double-counting
- Each frame is an independent measurement

## Perfusion Protocol
1. **Baseline**: Image cells for N frames with normal media (no drug)
2. **Drug addition**: Set perfusion pump to deliver drug
3. **Response monitoring**: Continue imaging during drug exposure
4. **Washout** (optional): Switch back to normal media, monitor recovery

## Perfusion Measurement Strategy
- Before drug: establish baseline (cell count, morphology, fluorescence)
- During drug: track temporal changes with `measure_response_time()`
- After washout: monitor recovery
- Drug onset is a RAMP, not a step — mixing takes time in microfluidics
- For steady-state: use LAST frames, not average of all post-drug frames
- Photobleaching: fluorescence decays even without drug effect
- Flow artifacts: cells may move/deform due to flow, not drug

## Key Hardware
- Perfusion pump: check `core.getLoadedDevices()` for pump device name
- May use states (0=off, 1=low, 2=medium, 3=high) or continuous control
- Pixel size: use `get_pixel_size(core)` — always correct after `set_objective()`

## Best Practices
- Use nucleus-channel for cleaner segmentation than BF
- Prefer spatial position over temporal comparison for trapped/flowing
- Trap positions in notes are WORLD COORDINATES — do NOT subtract 256
- SLM must be reapplied before each snap in closed-loop control
