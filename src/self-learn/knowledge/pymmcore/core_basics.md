# Core Operations — pymmcore-plus Reference

## Snapping Images

```python
# Low-level: snap + retrieve
core.snapImage()
img = core.getImage()  # returns numpy array (H, W)

# High-level helper (sets channel, snaps, returns array)
from hardware.core import snap
img = snap(core, channel="GFP")
```

## Stage Movement

```python
core.setXYPosition(x, y)          # XY stage (absolute, microns)
x, y = core.getXYPosition()
core.setPosition('ZStage', z)     # Z focus
z = core.getPosition('ZStage')
# NOTE: setPosition does NOT consume snaps — free to call between acquisitions
```

## Channel Configuration

```python
# CRITICAL: discover config group name first — do NOT hardcode "Channel"
groups = core.getAvailableConfigGroups()  # e.g. ("Fake",) or ("Channel",)
core.setConfig("Fake", "GFP")            # group varies by backend!
configs = core.getAvailableConfigs("Fake")  # ["BF", "GFP", "DAPI"]
```

## Objective Switching

```python
core.setState('Objective', 40)  # integer, NOT string "40x"
# Or use helper:
from hardware.core import set_objective
set_objective(core, 40)
```

## Hardware Discovery

```python
from hardware.core import MicroscopeConfig
config = MicroscopeConfig.from_core(core)
config.channels      # available channel names
config.objectives    # available magnifications
config.pixel_size    # um/pixel at current objective
config.fov_size      # field of view in um

from hardware.core import pixel_to_world, world_to_pixel
wx, wy = pixel_to_world(px, py, core=core)
```

## Pixel Size and FOV

Always query from hardware — values vary by instrument:
```python
pixel_size = core.getPixelSizeUm()
w, h = core.getImageWidth(), core.getImageHeight()
fov_um = w * pixel_size
```

## Common Pitfalls

- Config group name varies by backend — always call `getAvailableConfigGroups()` first
- `set_objective(core, 40)` takes an **integer**, not a string
- Budget snaps wisely — avoid unnecessary acquisitions
- `setPosition('ZStage', z)` is free (no snap consumed)
