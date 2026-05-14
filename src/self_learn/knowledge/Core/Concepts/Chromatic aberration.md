# Chromatic aberration — when channels disagree on where a point is

A lens doesn't focus all wavelengths to the same point. Blue light and red light come to focus at slightly different places — laterally (sideways on the image) and axially (along Z). In a multi-channel microscope that translates directly to: **the same physical object appears at slightly different (x, y, z) in DAPI, GFP, and Cy5 channels**. Ignored, this corrupts co-localization, centroid metrology, and Z-stack reconstruction.

Two forms:

1. **Lateral chromatic aberration (LCA)** — different magnifications per wavelength. A 10-µm bead imaged in blue may sit 0.5 µm displaced from the same bead imaged in red. The displacement is zero at the optical axis and grows toward the field edge.

2. **Axial chromatic aberration (ACA)** — different focal planes per wavelength. At a single Z, the blue channel is in focus while the red is slightly out. Worst with high-NA objectives; cheap ones can have ACA > 1 µm.

"Apochromat" objectives are corrected to match focus for three wavelengths (usually blue + green + red); "plan-apochromat" adds a flat field on top; "plan-fluor" is a cheaper compromise.

## Practical consequences

### Co-localization

Two proteins at identical positions in two channels can show a false displacement entirely due to LCA. Before reporting a co-localization distance of < 200 nm (typical fluorescence resolution), measure the chromatic shift with a multi-colour bead slide and register out the offset. See ``self_learn.analysis.colocalization``.

### Centroid metrology (spot localization, FISH, SMLM)

The bias is systematic. If you fit sub-pixel centroids in multiple channels, offset corrections of a few hundred nanometers can dominate your error budget at high resolution.

### Multi-channel Z-stacks

Without ACA correction, the `nominal_z = 0` best-focus plane is channel-dependent. Either (a) re-focus per channel (slow, phototoxic), or (b) use an apochromatic objective (money), or (c) apply a per-channel Z offset learned once from a reference slide.

### FOV-edge effects

LCA grows with field radius. If you care about metrology at the field edges (whole-slide scans, spot localization on a chip), apply the correction. At the centre of the FOV you can usually ignore LCA.

## How to calibrate

Image a multi-colour fiducial slide (commercial "Tetraspeck" beads, 0.2 µm, 5 colours). In each channel, locate the same bead and fit its centroid. The channel-to-channel centroid difference is the LCA shift at that field position. Fit a polynomial (usually 2nd-order radial) across the FOV to get a `shift(x, y, channel_pair)` map.

**Forward-model shortcut when the optical pipeline is documented**
(simulator or well-characterised real microscope): given the
objective's `lateral_coef` (achromat ~0.017, fluorite ~0.008,
apochromat ~0.002) and the channel emission wavelengths, the radial
LCA shift at radius `r` is `r × lateral_coef × |Δλ| / 100` in the
brief's per-100-nm convention. ch603 r3 → 10/10 used this directly:
read `Camera.EmissionWavelengthNm` per channel + the achromat
coefficient → predicted offsets per cell from measured radii.
(`Camera.EmissionWavelengthNm` was a sim-only virtual property and
was removed 2026-04-27; on a real microscope this metadata comes
from the filter-set / dichroic registry rather than the camera.)
The forward-model path was the brief's recommended recovery against
matching-collapse on noisy edge cells (see
`feedback_forward_model_over_matching.md`); Hungarian + binary-COM
silently under-reports edge LCA when shifts approach voronoi
cell-spacing scales.

For ACA, acquire a z-stack of the bead slide in each channel. Measure the z with peak focus per channel (e.g., with Brenner or sharpness-of-PSF). The peak-z differences are your axial offsets.

Save the corrections in a calibration file; apply them pre-analysis.

```python
# Illustrative shape; not a real implementation.
def register_channels(img_green, img_red, shift_xy=(0.0, 0.0), shift_z=0.0):
    """Shift one channel to align with the reference (usually green).

    shift_xy: (dx, dy) in pixels at the image centre. For LCA-corrected
        results, use a per-location shift from the calibration map.
    shift_z: z offset in µm; needs a z-stack to apply.
    """
    from scipy.ndimage import shift as nd_shift
    return nd_shift(img_red, shift=[shift_xy[1], shift_xy[0]], order=1)
```

## When you can safely ignore chromatic aberration

- Single-channel imaging (DAPI count, brightfield only).
- Whole-cell metrics where the feature is larger than the aberration (cell area, not FISH spot centroid).
- Apochromatic objectives imaged within their warranted wavelength band.
- Centre-of-FOV-only acquisitions (LCA is ~0 near the optical axis).

## When it matters

- Multi-channel co-localization at <500 nm resolution.
- FISH, SPT, SMLM — anything that fits sub-pixel centroids across channels.
- Whole-slide imaging, tile stitching across field edges.
- Light-sheet / SPIM where Z is reconstructed from multiple objectives (any ACA propagates to geometric errors).

## Related

- `[[Core/Concepts/Fluorophore basics]]` — channel choice also affects Stokes-shift bleedthrough, a related source of cross-channel error.
- `[[Core/Concepts/Nyquist sampling]]` — if you under-sample, you can't resolve sub-pixel shifts anyway, so LCA becomes irrelevant.
- ``self_learn.analysis.colocalization`` — Pearson / Manders co-localization metrics.
- ``self_learn.analysis.registration`` — image registration primitives (if applicable).

## Further reading

*Orientation pointers, not canonical citations — these don't go through the paper-library verification path.* Search terms: Kozubek & Matula chromatic-aberration correction algorithm (2000), Manders co-localization with shift correction (1997), Abbe 1873 wavelength-dependent resolution formulation. For objectives, check the manufacturer catalogue (Zeiss / Nikon / Olympus) for the "correction" label on any lens you use.
