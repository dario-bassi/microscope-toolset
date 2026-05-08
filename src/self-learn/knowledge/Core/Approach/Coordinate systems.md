# Reference: Coordinate Systems

## Pixel Coordinates

- Origin: top-left corner of image
- `centroid_px` from `detect_cells()` returns `(column, row)` = `(x, y)`
- Image shape is `(height, width)` = `(rows, columns)`
- Pixel (0, 0) is top-left; pixel (width-1, height-1) is bottom-right

## World Coordinates

- Independent of magnification and stage position
- Stage position = center of the current FOV in world coordinates
- Image center in pixels = `(image_width/2, image_height/2)`

### Conversion

```python
from src.core.hardware.core import pixel_to_world, world_to_pixel

# Pixel -> World
world_x, world_y = pixel_to_world(px, py, stage_x, stage_y, core=core)

# World -> Pixel
pixel_x, pixel_y = world_to_pixel(wx, wy, stage_x, stage_y, core=core)
```

### Manual Formula

```
world_x = stage_x + (pixel_col - image_width/2) * pixel_size_um
world_y = stage_y + (pixel_row - image_height/2) * pixel_size_um
```

### Quick Reference

Pixel size and FOV depend on the specific objective and camera. Always query from hardware:
```python
pixel_size = core.getPixelSizeUm()
w, h = core.getImageWidth(), core.getImageHeight()
fov_um = w * pixel_size
image_center = (w // 2, h // 2)
```
Higher magnification = smaller pixel size = smaller FOV. Always verify before computing coordinates.

### Common Mistakes

1. **Submitting pixel coordinates at 40x**: At 40x, pixel (100, 200) is NOT at world (100, 200). Must convert.
2. **Swapping (row, col) and (x, y)**: `centroid_px` is (col, row) = (x, y). NumPy indexing is [row, col].
3. **Forgetting stage position**: World coordinates depend on where the stage was when the image was taken.
4. **Using hardcoded 256 for center**: Use `cfg.image_center_px` from runtime config.
5. **Reporting area in camera-px instead of world-px/µm²**: When an answer
   field is a physical area (nuclear_area, cell_area, lipid_fraction),
   convert to world units. A 20x measurement of 333 px² is 333×(0.5 µm/px)²
   = 83 µm². Equivalently, divide by `(mag/10)²` to get the 10x-equivalent
   world-px value. GTs are typically stored in world units.

## Parameter Naming Convention

`pixel_size` and `pixel_size_um` refer to the same value (micrometers per pixel).
- `hardware/config.py`: exposes `pixel_size_um` on the config object
- `analysis/morphometry.py`, `workflows/adaptive.py`: accept `pixel_size` parameter
- Both are in micrometers. No conversion needed — just pass `config.pixel_size_um` as `pixel_size`.

## SLM Coordinates

- SLM mask is uint8, matching SLM resolution (query from hardware, or match camera resolution)
- Aligned with camera/viewport space
- (0,0) = top-left of camera FOV
- Apply SLM mask BEFORE each snap (not just once)

## See also

- [[Core/Concepts/Nyquist sampling]] — the pixel-size decision sets everything downstream.
- [[Core/Strategies/Physical-unit thresholds]] — parameterise detection in µm²/µm, not pixels.
- [[Core/Strategies/SLM optogenetics]] — SLM coordinate alignment and when to re-apply the mask.
- `../../../src/core/hardware/config.py` — runtime `pixel_size_um` source of truth.
