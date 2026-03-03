# Microfluidic Device — Cell Sizing & Trap Analysis

## Sample Type
Microfluidic channel with constriction traps. Cells flow through channel; some are captured by traps.

## Workflow
1. **10x survey**: Snap BF + nucleus at field center. Identify channel boundaries (Y range) and trap positions (X positions).
2. **Detect traps**: In BF, traps appear as constrictions/pillars at specific X positions. Record these X coordinates.
3. **Detect all cells**: Nucleus channel, threshold above background. Use multiple Y positions to cover full channel width.
4. **Classify trapped vs free**: SPATIAL analysis — cells within trap_width of trap X positions = TRAPPED. All others = FREE.
5. **40x measurement**: Systematic grid scan covering full channel width. Measure cell diameter with `equivalent_diameter_area * pixel_size`.

## Critical Lessons

### Trapped vs Free = SPATIAL, not temporal (ch455 lesson, cost 5 points)
- **WRONG**: Snap two frames under flow, check if cells moved → "all stationary = all trapped"
- **RIGHT**: Trap positions are fixed X coordinates visible in BF. Cells AT those X positions = trapped.
- In simulation, cells are placed statically — temporal frame comparison doesn't reveal flow
- The challenge notes say "trapped cells are stationary and darker (compressed)" — use POSITION + BRIGHTNESS
- Trap X positions can be detected from BF channel: look for vertical constriction features

### Deduplication
- World-coordinate deduplication must NOT be too aggressive
- 3µm grid snapping (ch455 initial approach) merged distinct cells
- Better: use `deduplicate_cells()` from `src/workflows/scanning.py` with proper merge_radius
- Each unique cell should appear at a unique world position — don't merge by size similarity

### Scanning Coverage
- The microfluidic channel spans a Y range (e.g., Y=196-316)
- Scan multiple Y positions at 40x to cover full channel width
- Don't assume cells are only at Y=center; they distribute across channel width

### Pixel Size
- Use `get_pixel_size(core)` — always correct after `set_objective()`
- At 40x: 0.25 µm/px, FOV=128µm

## Pitfalls
- Temporal flow detection doesn't work if simulation places cells statically
- Over-deduplication removes real cells that happen to be nearby or similar size
- Channel edges may be missed if Y scanning range is too narrow
- Trap identification requires BF channel analysis (not just nucleus)
