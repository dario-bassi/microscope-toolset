# Depth of field — the Z budget of a single frame

Depth of field (DOF) is how thick a slice of the sample stays in acceptable focus at one Z position. It's set by the objective, independent of the camera. In smart microscopy it shows up as:

1. How often you need to **refocus** on a drifting sample.
2. When you need a **Z-stack** to capture the whole feature vs. a single slice.
3. How much **blur tolerance** your acquisition plan has for stage drift, sample settling, or thermal creep.

## The number

For a widefield / brightfield objective with numerical aperture NA, emission wavelength λ, and refractive index of the immersion medium n, the DOF is:

```
DOF ≈ λ · n / NA²   +   n · e / (M · NA)       # standard two-term formula
       ^wave term       ^geometric term
```

where `e` is the pixel pitch on the detector (projected to sample is `e / M` at magnification M). For typical 512-px scientific cameras with e≈6.5 µm, the geometric term is small and the wave term dominates at high NA.

**Working numbers (widefield, λ ≈ 550 nm, air n=1 unless noted):**

| Objective | NA | Medium | Wave term (µm) | ~DOF (µm) |
|----|----|----|----|----|
| 10× / 0.3  | 0.3 | air | 6.1 | ~8–12 |
| 20× / 0.5  | 0.5 | air | 2.2 | ~3–5 |
| 40× / 0.75 | 0.75 | air | 1.0 | ~1.5–2.5 |
| 60× / 1.2  | 1.2 | water (n=1.33) | 0.51 | ~0.8–1.2 |
| 100× / 1.4 | 1.4 | oil (n=1.51) | 0.42 | ~0.5–0.8 |

**Confocal / 2-photon sectioning** gives a much thinner axial window by rejecting out-of-focus light (~0.5–1 µm at 1.2 NA). The DOF formula above is a widefield approximation.

## How this drives your acquisition plan

### Tracking drift under a fixed Z

If the sample drifts at some rate (`v_z` µm/frame), you have `DOF / v_z` frames before defocus becomes noticeable. Example: 10× (DOF ~10 µm), drift 0.5 µm/frame → refocus needed every ~20 frames. At 40× (DOF ~2 µm), same drift → every ~4 frames.

Practical consequence: refocus cadence scales roughly with `1 / mag²` (because DOF ∝ 1/NA² ∝ 1/mag² at matched objective design). A timelapse that gets away with no autofocus at 10× may be unusable at 40× without closed-loop Z correction. See `[[Core/Strategies/Closed-loop autofocus]]`.

### Z-stacks for features thicker than DOF

If the feature spans `Δz` axially and DOF is smaller, one frame won't capture it all. For sharp top-to-bottom rendering you need a stack with step size ≤ DOF/2 and range ≥ Δz:

| Feature | Typical Δz | Best objective | Z step |
|----|----|----|----|
| Single nucleus (5-10 µm) | 6 µm | 20× | 1 µm |
| Cell body (10-30 µm) | 15 µm | 40× | 0.5 µm |
| Thick tissue slice (20-100 µm) | 50 µm | 10×–20× | 1–2 µm |
| Organoid (100-500 µm) | 300 µm | 10× (survey) + 40× (tile) | 2 µm |
| Whole zebrafish | >1 mm | 4× light-sheet | 2 µm |

See ``self_learn.hardware.zstack``.

### Single-plane acquisitions on thin samples

A monolayer of cells (flat, ~10 µm tall) often fits within the DOF of a 20× or even 40× objective — no Z-stack needed. Rule of thumb: if the sample is shorter than 2× DOF at your objective, single plane is fine.

## Best-focus detection

Measuring focus quality frame-by-frame lets you both pick the best plane and detect drift:

```python
from self_learn.workflows.autofocus import focus_metric

# All of these produce higher values for sharper images.
brenner = focus_metric(img, method='brenner')     # fastest
laplacian = focus_metric(img, method='laplacian') # best contrast sensitivity
sobel = focus_metric(img, method='sobel')         # anisotropic
```

Brenner is the usual default — it's a one-shot gradient-squared sum that runs in microseconds per frame. See `[[Core/Strategies/Closed-loop autofocus]]` for a closed-loop Z-tracking workflow.

## Why DOF is bigger at lower mag (and that's sometimes a gift)

A 10× DOF of ~10 µm means the sample plane can drift half that before your best-focus image becomes blurry. That's why survey scans at 10× tolerate sloppy stage leveling, thermal drift, and sample settling that would ruin a 40× acquisition. When planning a 2-pass survey → analysis workflow (`[[Core/Strategies/Multi-scale morphometry]]`), the survey pass doesn't need autofocus; the analysis pass at higher mag does.

## Related

- `[[Core/Concepts/Nyquist sampling]]` — the lateral-resolution counterpart.
- `[[Core/Concepts/Exposure and photodamage]]` — higher-NA objectives gather more light per µm² but image less thick volume.
- `[[Core/Strategies/Closed-loop autofocus]]` — Z-drift control.
- ``self_learn.hardware.zstack`` — `acquire_zstack`, `detect_cells_zstack`.
- ``self_learn.workflows.autofocus`` — focus metrics, sweep, corrector.

## Further reading

*Orientation pointers, not canonical citations — these don't go through the paper-library verification path.* Search terms: Pawley *Handbook of Biological Confocal Microscopy* Ch. 8 (axial resolution), Inoué & Spring *Video Microscopy* Ch. 5 (geometric + wave DOF), Born & Wolf *Principles of Optics* §9.7 (diffraction theory).
