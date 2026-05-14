# Physical-unit thresholds in detection parameters

> **When to use:** When setting detection thresholds (min area, min distance, sigma) that must remain biologically valid across different magnifications or pixel sizes.

## The pattern

Detection and segmentation functions routinely expose thresholds in
pixels — `min_area=30`, `min_sigma=1.0`, `min_distance=8`. Those defaults
are silently tied to one pixel-size regime (usually ~1 µm/px at 10x).
Run the same function at 40x (0.25 µm/px) and the defaults mean something
completely different.

**The fix**: accept `pixel_size_um` and a physical-unit alternative
(`min_area_um2`, `min_diameter_um`, `min_distance_um`), convert to
pixel equivalents internally, and document what the physical defaults
mean anatomically.

## Template

```python
def detect_things(image, min_area=30, pixel_size_um=None,
                  min_area_um2=None):
    """...

    Physical defaults (when ``pixel_size_um`` is provided):
    ``min_area_um2=7.5`` (≈a 3 µm-diameter object).

    Pixel defaults (``min_area=30``) preserved for backward compat but
    only make sense around 1 µm/px.
    """
    if pixel_size_um is not None:
        min_a_um2 = 7.5 if min_area_um2 is None else float(min_area_um2)
        min_area = max(1, int(round(min_a_um2 / float(pixel_size_um) ** 2)))
    ...
```

## Conversions

| Physical quantity | Convert to pixels |
|-------------------|-------------------|
| area (µm² → px)   | `area_um2 / pixel_size_um**2` |
| diameter (µm → px)| `diameter_um / pixel_size_um` |
| distance (µm → px)| `distance_um / pixel_size_um` |
| LoG sigma (pixels to span a blob of diameter d_um) | `(d_um / pixel_size_um) / (2·√2)` |

LoG note: a Gaussian kernel with standard deviation σ responds most
strongly to a blob of radius ≈ σ·√2 (equivalently diameter 2·σ·√2).

## Physical defaults (anatomical calibration)

For mammalian cells:

| Feature | µm-scale | µm²-scale |
|---------|----------|-----------|
| Single nucleus (cross-section) | 5-15 µm diameter | 20-500 µm² |
| Mitotic (condensed) nucleus | 3-8 µm diameter | 10-50 µm² |
| Lipid droplet (physiological) | 0.2-2 µm diameter | 0.03-3 µm² |
| Lipid droplet (steatotic) | 2-10 µm diameter | 3-80 µm² |
| Lysosome / secretory vesicle | 0.1-0.5 µm diameter | 0.008-0.2 µm² |
| Stress granule | 0.5-2 µm diameter | 0.2-3 µm² |
| FISH dot | 0.2-0.6 µm diameter | 0.03-0.3 µm² |

Sanity-check any `*_um`/`*_um2` default against this table before
picking a number.

## When NOT to parameterize

Not every constant is magnification-dependent. Keep the pixel API when:
- The parameter is an **algorithmic** choice (e.g. `block_size=51` for
  local thresholding — tied to image noise, not object size).
- The parameter is a **relative** quantity (percentile, fraction).
- The function is only ever called with the same pixel size (e.g. a
  workflow hard-scoped to a single objective).

**Anti-pattern: a module-level pixel-size constant.** Worse than a
default kwarg because it propagates implicitly into every call across
the module. spt.py's deleted `DEFAULT_PX_UM = 0.0125` is the cautionary
tale — it silently turned a 40× SPT analysis into a wrong-by-6.25× µm²
report. Required keyword-only `pixel_size_um` is the right shape.

The distinguishing question: "does the user's question change with
magnification?" If yes (object size / distance / blob scale), physical
units are correct. If no (relative intensity, algorithmic window),
pixels are fine.

## Applied to

- `segment_nuclei_hae(pixel_size_um=..., min_area_um2=7.5, max_area_um2=500, min_distance_um=2.0)` in `analysis/histology.py`.
- `detect_lipid_droplets(pixel_size_um=..., min_diameter_um=0.5, max_diameter_um=5.0, min_area_um2=0.3)` in `analysis/lipid_droplet.py`.
- `detect_mitoses(pixel_size_um=..., max_area_um2=50)` in `analysis/histology.py`.
- `segment_nuclei(pixel_size_um=..., min_area_um2=20, max_area_um2=None)` in `analysis/morphometry.py` — generic fluorescence nuclei; 20 µm² floor accepts small nuclei, leaves upper bound optional so the caller chooses.
- `compute_msd(track_xy, *, pixel_size_um, dt=0.1)` and `analyze_spt(frames, *, pixel_size_um, ...)` in `analysis/spt.py` — required keyword-only `pixel_size_um`. Pre-refactor (counter=42) this had a `DEFAULT_PX_UM = 0.0125` module constant that silently encoded a 100x objective and produced wrong µm² at any other magnification. Recipes should pass `get_config(core).pixel_size_um`.

## Related

- `Core/Approach/Coordinate systems.md` — "Common mistake #5"
  about reporting physical areas in camera pixels (ch567 lesson).
- `Core/Pitfalls/FOV vs well coverage.md` — another
  magnification-aware pitfall (ch569 lesson).
