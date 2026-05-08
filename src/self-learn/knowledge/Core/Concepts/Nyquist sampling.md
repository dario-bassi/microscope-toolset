# Nyquist sampling — pick the right magnification before you acquire

A microscope samples continuous light onto a discrete pixel grid. Whether the grid can actually represent the optical image is a Nyquist question, and it has a decisive answer: **you need at least two pixels across the smallest resolvable feature the optics produce**. Under-sample and the feature aliases — you see structure that isn't there and miss structure that is. Over-sample and you burn light budget (and disk) for no information gain.

In smart microscopy the practical consequence is: the right magnification is determined by what you need to measure, not by what looks nice in the viewer. Pick it once, calibrate the pixel size, and push all downstream analysis through physical units (µm, µm²). See `[[Core/Approach/Coordinate systems]]` and `[[Core/Strategies/Physical-unit thresholds]]` for why.

## The number

For a confocal or widefield image with numerical aperture NA and emission wavelength λ, the diffraction-limited lateral resolution (Rayleigh, Abbe variants differ slightly) is:

```
d_lateral ≈ 0.61 · λ / NA        # Rayleigh criterion
```

The Nyquist condition requires the pixel size `p` (physical size projected onto the sample) to satisfy:

```
p ≤ d_lateral / 2   →   p ≤ 0.305 · λ / NA
```

**Working numbers (widefield, λ ≈ 550 nm):**

| Objective | NA | d_lateral | Nyquist p |
|----|----|-----------|-----------|
| 10× / 0.3  | 0.3 | 1.1 µm | 0.56 µm/px |
| 20× / 0.5  | 0.5 | 0.67 µm | 0.34 µm/px |
| 40× / 0.75 | 0.75 | 0.45 µm | 0.22 µm/px |
| 60× / 1.2 (water) | 1.2 | 0.28 µm | 0.14 µm/px |
| 100× / 1.4 (oil) | 1.4 | 0.24 µm | 0.12 µm/px |

If your pixel size is *larger* than `p`, features smaller than `2p` alias — a diffraction-limited point can show up as edge artifacts, a thin fiber can appear as a dashed line, a small nucleus can be missed entirely.

## The rule of thumb

You usually don't need to hit Nyquist exactly — each extra 2× of sampling costs 4× the photons for the same SNR. The working rule is:

- **Count features**: lowest mag where the feature spans **≥ 3–4 pixels** across. You need enough pixels to separate it from neighbours, not to resolve sub-structure.
- **Measure shape/morphology**: Nyquist (2× sampling of the resolution limit).
- **Localize sub-diffraction points** (SMLM, SPT, FISH dots): super-Nyquist (4–6× sampling) — you're fitting a PSF, and finer sampling improves centroid precision.

```
Feature size → lowest sensible magnification
  colony (0.5–2 mm)     → 4× or macro (2–4 µm/px)
  cell count (10–30 µm) → 10× (1.0 µm/px) — 10–30 px across: fine for counting
  cell shape (10–30 µm) → 20× (0.5 µm/px) — 20–60 px across: morphometry usable
  nucleus shape (5–15 µm) → 20× (0.5 µm/px)
  organelle (0.5–2 µm)  → 40× (0.25 µm/px) — 2–8 px: enough for counting
  FISH spot (0.25 µm)   → 60× + oil (0.14 µm/px) — fit PSF
  bud scar (0.5 µm)     → 40×–60×
```

## The trap: sim-specific recipes with fixed pixel sizes

A segmentation threshold like `min_area = 30 px` is silently tied to one pixel size. Run the same code at 40× and it rejects real nuclei; at 4× it accepts noise speckles. **Every pixel-space threshold should either be derived from a physical-size parameter (`min_area_um2 / pixel_size_um²`) or be documented as valid only at a specific magnification.** The refactor in ``../../../src/core/analysis/histology.py`` and ``../../../src/core/analysis/lipid_droplet.py`` is an example of the right pattern — see `[[Core/Strategies/Physical-unit thresholds]]`.

## Practical calibration

```python
from src.core.hardware.config import get_config

cfg = get_config(core)
# pixel_size_um comes from PixelSize config group on the core.
# If the pixel-size config is missing, the function falls back on a
# zoom-factor model from the objective magnification; prefer a real
# calibration (fiducial grid) for anything quantitative.
print(f"{cfg.pixel_size_um:.3f} µm/px at {cfg.current_magnification(core)}×")
```

For a quick sanity check of the current FOV:

```python
fov_um = cfg.image_width * cfg.pixel_size_um      # µm across the camera
nyquist_ok = cfg.pixel_size_um <= 0.3 * 0.550 / 0.75   # e.g. 40x/0.75 obj
```

## When to under-sample on purpose

- **Survey scans / scouting.** 4× or 10× is faster and covers more area; aliasing doesn't matter if you're just asking "where are the samples on this slide".
- **Phototoxicity-limited timelapses.** Lower mag gives bigger pixels → more photons per pixel → lower exposure needed for the same SNR. Trade resolution for dose.
- **Coverage-path planning.** See `[[Core/Strategies/Multi-position survey]]`.

## When to over-sample

- **PSF-fitting / SMLM.** You're not looking at pixels; you're fitting a model to them. More pixels = more constraints = tighter centroid.
- **Sub-pixel metrology.** Distance between fiducials, drift correction via cross-correlation — sub-pixel accuracy scales with sampling density.
- **Publication-quality morphometry.** Outline traces and curvature estimates benefit from extra pixels.

## Related

- `[[Core/Concepts/Exposure and photodamage]]` — the dose cost of higher sampling.
- `[[Core/Approach/Coordinate systems]]` — the px↔µm↔world-coord conversion rules.
- `[[Core/Strategies/Physical-unit thresholds]]` — anatomy-scaled segmentation parameters.
- `[[Core/Pitfalls/FOV vs well coverage]]` — "you can't see something that isn't in the FOV".
- ``../../../src/core/hardware/config.py`` — `pixel_size_um`, `_mag_to_pixel_size` fallback.

## Further reading

*Orientation pointers, not canonical citations — these don't go through the paper-library verification path.* Search terms: Shannon 1949 sampling theorem (foundational), Pawley *Handbook of Biological Confocal Microscopy* Ch. 4, Heintzmann & Sheppard on sampling for optical microscopy, Gustafsson on structured-illumination Nyquist implications.
