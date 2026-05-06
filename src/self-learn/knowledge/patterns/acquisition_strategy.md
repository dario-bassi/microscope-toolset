# Decision Pattern: Acquisition Strategy

## When to Survey vs Zoom

```
Task requirement?
|
+-- Count all cells in large area -> Multi-position scan at 10x
|   |-- Grid with 30% overlap
|   |-- Hierarchical dedup at 30-35px threshold
|   +-- Convert all to world coordinates
|
+-- Find specific cell type -> Survey at 10x, then zoom to 40x
|   |-- Survey: detect all cells, screen for candidates
|   |-- Screen generously (better to inspect extra than miss one)
|   |-- Zoom: move stage to candidate, switch to 40x, re-detect
|   +-- Re-detect at 40x (don't trust 10x positions after drift)
|
+-- Measure single cell detail -> Zoom to 40x directly
|   |-- Query pixel_size and FOV from hardware
|   +-- Exclude cells within 50px of edge
|
+-- Track over time -> Timelapse at fixed position
|   |-- Plan frame interval based on expected timescale of process
|   |-- Minimize unnecessary snaps (budget frames wisely)
|   +-- check_interval=5 for cell count tracking
```

## Magnification Reference

| Mag | Use case |
|-----|----------|
| 10x | Survey, counting, scanning (largest FOV) |
| 20x | Intermediate detail |
| 40x | Morphology, classification, detailed measurement |

Always query actual pixel_size and FOV from hardware: `core.getPixelSizeUm()`, `core.getImageWidth()`.

## Coordinate Conversion

```
world_x = stage_x + (pixel_col - image_width/2) * pixel_size
world_y = stage_y + (pixel_row - image_height/2) * pixel_size
```

Higher magnification = smaller FOV, centered on stage position.

**Always convert to world coordinates before submission at non-10x magnifications.**

## When to Refocus

- Before first acquisition (always)
- After switching objectives (focal plane changes)
- After moving stage >100um (sample may not be flat)
- If image appears blurry (check with Laplacian variance)
- During timelapse: every 10-20 frames (Z-drift)

## When to Adjust Exposure

- Fluorescence: target 50-80% of dynamic range
- If saturated pixels > 1%: reduce exposure
- If mean intensity < 20 (uint8): increase exposure
- Use `src/workflows/exposure.py` for proportional adjustment
