# Decision Pattern: Tile Scan Strategy

## When to Use Tile Scanning
- Need to image an area larger than a single FOV
- Survey a region for cell distribution, confluence, or rare events
- Reconstruct a mosaic/stitched overview image

## Scan Pattern: Serpentine (Snake)

```
Row 0: →  →  →  →  →  →  →  →  →
Row 1: ←  ←  ←  ←  ←  ←  ←  ←  ←
Row 2: →  →  →  →  →  →  →  →  →
...
```

Minimizes stage travel by alternating X direction each row:
```python
for iy in range(n_rows):
    x_range = range(n_cols) if iy % 2 == 0 else range(n_cols - 1, -1, -1)
    for ix in x_range:
        move_to(start_x + ix * step, start_y + iy * step)
        snap()
```

## Overlap and Step Size

- **10% overlap** is standard for robust stitching
- Step = FOV_size * (1 - overlap_fraction)
- Example: 1024px sensor, 0.65 um/px → FOV = 665.6 um → step = 599 um with 10% overlap

## Pixel Size Calculation (when not calibrated)

If `mmc.getPixelSizeUm()` returns 0:
```
pixel_size = physical_pixel_um * binning / magnification
```
Example: Andor Zyla 4.2P (6.5 um physical) × 2x binning / 20x magnification = 0.65 um/px

Query physical pixel from device properties or camera spec sheet.

## Focus Maintenance During Scan

### Option 1: Software autofocus at each tile
- Slow but reliable
- Use narrow range (e.g., +/- 15um) centered on previous best Z
- Good for flat samples with minor tilt

### Option 2: PFS (hardware autofocus)
- **Preferred** for tile scans — maintains focus during XY moves automatically
- Must calibrate PFS offset before scan (see failures/autofocus_range_and_pfs.md)
- Allow 1s settle time after each XY move for PFS to lock
- Verify PFS is locked: check `TIPFSStatus` device property

### Option 3: Pre-computed focus map
- Measure Z at corners/edges of scan area
- Interpolate Z for intermediate positions
- Fast but assumes flat/tilted sample (no local irregularities)

## Sample Edge Detection

To find the edge of a sample holder or coverslip:
1. Snap reference image in sample region, record mean intensity
2. Move stage in -X (or -Y) in steps (e.g., 500um)
3. Snap at each position, measure mean intensity
4. Edge = where mean intensity drops below 30% of reference
5. Use this as scan boundary

## Mosaic Stitching (Simple Blending)

```python
overlap_px = int(sensor_size * overlap_fraction)
step_px = sensor_size - overlap_px
mosaic_size = step_px * n_tiles + overlap_px  # per axis

mosaic = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
weight = np.zeros_like(mosaic)

for iy in range(n_rows):
    for ix in range(n_cols):
        y0 = iy * step_px
        x0 = ix * step_px
        mosaic[y0:y0+sensor, x0:x0+sensor] += tile[iy, ix].astype(np.float32)
        weight[y0:y0+sensor, x0:x0+sensor] += 1.0

mosaic = (mosaic / np.maximum(weight, 1)).astype(np.uint16)
```

For better results: use linear blending ramps in overlap regions, or use dedicated stitching tools (e.g., ASHLAR, BigStitcher).

## Grid Size Estimation

```
n_tiles_per_axis = ceil(scan_range / step_size)
total_tiles = n_tiles_x * n_tiles_y
scan_time ≈ total_tiles * (move_time + settle_time + exposure_time)
```

Example: 5000um range, 599um step → ceil(5000/599) = 9 tiles per axis → 81 tiles total.

## Common Pitfalls

1. **Forgetting to settle after XY move**: Stage vibration blurs images. Wait 0.5-1.0s after move.
2. **PFS not locked**: Check PFS status after each move. If PFS loses lock (e.g., at sample edge), fall back to software autofocus.
3. **Pixel size = 0**: Not all configurations have pixel size calibrated. Calculate manually from camera spec + objective magnification + binning.
4. **Accumulating position error**: Stage may have backlash. Serpentine pattern helps, but for critical applications, use absolute positions rather than relative moves.
