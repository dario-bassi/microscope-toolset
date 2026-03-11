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
from src.hardware.core import pixel_to_world, world_to_pixel

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

## Parameter Naming Convention

`pixel_size` and `pixel_size_um` refer to the same value (micrometers per pixel).
- `hardware/config.py`: exposes `pixel_size_um` on the config object
- `analysis/morphometry.py`, `workflows/adaptive.py`: accept `pixel_size` parameter
- Both are in micrometers. No conversion needed — just pass `config.pixel_size_um` as `pixel_size`.

## SLM / DMD Coordinates

- **Device name**: discover at runtime via `core.getSLMDevice()` (do NOT hardcode)
- **Resolution**: query via `core.getSLMWidth(dev)`, `core.getSLMHeight(dev)` — independent of camera
- **Coordinate origin**: (0,0) = top-left of SLM
- SLM is generally NOT aligned with camera space — requires affine calibration
- **Conjugate plane**: SLM pattern may only be visible at a Z offset from sample focus
- **Timeout**: some SLMs turn off after ~2 min — always re-upload mask before use
- **Channel-specific**: only channels routed through the SLM will show patterns
- **Calibration is objective/channel/binning specific** — re-calibrate after any change
- **Preferred API**: `MDAEvent.slm_image=SLMImage(data=mask, device=dev)` handles coordination
- **Manual API**: `core.setSLMImage(dev, mask); core.displaySLMImage(dev)`
- **Full protocol**: see `knowledge/patterns/dmd_calibration.md`
- **Apply SLM mask BEFORE each snap (not just once)**

### Camera ↔ SLM Conversion

```python
from src.self_learn.hardware.slm_calibration import (
    load_calibration, camera_mask_to_slm,
)

calib = load_calibration("dmd_calibration.json")
fwd = calib["dmd_to_camera"]["matrix"]  # SLM -> Camera (numpy array)
slm_mask = camera_mask_to_slm(camera_binary_mask, fwd,
                               slm_w=calib["slm_size"][0],
                               slm_h=calib["slm_size"][1])
```
