# SNR and dynamic range — two axes of image quality

Two concepts that get confused constantly: SNR is *how strong the signal is over the noise floor*; dynamic range is *the span between the darkest and brightest pixels the sensor can distinguish*. Both matter independently. You can have a bright, high-SNR image with no dynamic range (all pixels saturate) and a well-exposed image with no SNR (even exposure of nothing).

## SNR: the signal-to-noise ratio

The physical definition in a digital microscope:

```
SNR = (signal − background) / σ_background
```

- `signal` — mean intensity inside the object of interest.
- `background` — mean intensity of a region known to contain no signal (outside cells, a dark corner, between nuclei).
- `σ_background` — standard deviation of the background.

At a practical threshold, **SNR ≥ 5** is needed for automated detection (otherwise you get noise-driven false positives); **SNR ≥ 10** is comfortable; above 20 is usually wasted photons that could be traded for gentleness.

### Noise sources, in rough order of contribution

1. **Shot noise** — Poisson: `σ_shot = √N` for `N` photons. Dominant when the sensor isn't saturating; the only one you can fight with more light.
2. **Read noise** — fixed per-pixel electronic noise (a few e⁻ for scientific CMOS, ~10 e⁻ for EMCCD). Dominant at very low light.
3. **Dark current** — thermal electrons. Negligible for scientific cameras running cooled, measurable for consumer or long-exposure (>1 s) frames.
4. **Pattern noise** — fixed per-pixel gain / offset. Removable by flat-field correction.

A surprising consequence: **shot noise is proportional to √N, so doubling the signal only improves SNR by √2**. To go from SNR 5 → SNR 10 you need 4× more photons, not 2×. Plan accordingly.

## Dynamic range: the sensor's vocabulary

Dynamic range is the ratio `(max − min) / σ_read` — roughly, how many distinct grey levels the sensor can resolve. For a typical scientific CMOS: `max = 65 535` (uint16), `σ_read ≈ 2 e⁻`, so ~15 stops of dynamic range. For an 8-bit camera: `max = 255`, ~8 stops.

What eats dynamic range:

- **Saturation** — any pixel at the sensor ceiling has lost all brightness information above that ceiling. The dimmer pixels retain their resolution, but the bright ones are now all "the same bright". Don't over-expose the regions you care about.
- **Clipping at zero** — similarly, pixels at 0 have lost the information that they were *slightly above* zero. Rare in practice because the sensor has a baseline offset (the "black level").
- **Gain boost** — multiplying the signal digitally (or via EM gain) can lift dim features above read noise, but the bit depth after the boost is the same, so you're quantising more coarsely per unit of brightness.

### Practical fraction metrics

```python
# Saturation — fraction of pixels at or near max value
max_val = 255 if img.dtype == np.uint8 else 65535
saturation = float((img > 0.995 * max_val).sum()) / img.size

# Dynamic-range usage — how much of the sensor's range the image actually fills
p01 = float(np.percentile(img, 1))
p99 = float(np.percentile(img, 99))
dynamic_range_used = (p99 - p01) / max_val
```

Rules of thumb:
- `saturation > 0.01` → reduce exposure or intensity (you're clipping real signal).
- `dynamic_range_used < 0.1` → increase exposure or gain (you're wasting bits).
- Sweet spot: brightest features at ~70-90% of max, background at a few per cent.

## Why both matter separately

A microscope engineer's joke: "I can make any image look good — just give me 16-bit mean-subtracted display." The underlying image quality is preserved in two numbers: **how much information you captured (dynamic range used)** and **how much of it is signal vs. noise (SNR)**. Any number of display tricks can make an image look pretty, but only more photons, longer exposure, or a cooler camera actually improve these two.

Relatedly: **stretching contrast doesn't improve SNR**. If the background σ is 5 and the signal is 10 above it, you have SNR 2 whether you display the image with min=0/max=15 or min=0/max=255. Be suspicious of any "denoising" or "enhancement" that doesn't also tell you the noise model it's assuming.

## In an acquisition workflow

```python
from self_learn.analysis.image_quality import assess_quality

r = assess_quality(img)
# {'overall': 'good' | 'acceptable' | 'poor',
#  'focus_score': ..., 'noise_level': ..., 'saturation_fraction': ...,
#  'dynamic_range_fraction': ..., 'snr_estimate': ..., 'warnings': [...]}
```

If you're parameter-sweeping exposure/gain (`[[Core/Strategies/Imaging parameter optimization]]`), measure SNR *and* saturation at each point — the best settings are the ones where both are in their sweet spots, not a single scalar optimum.

## See also

- `[[Core/Concepts/Exposure and photodamage]]` — how to trade SNR for dose.
- `[[Core/Concepts/Nyquist sampling]]` — the *other* axis of image quality (spatial).
- `[[Core/Strategies/Imaging parameter optimization]]` — automated sweeps.
- ``self_learn.analysis.image_quality`` — `assess_quality`, `focus_score`, `noise_estimate`, `check_saturation`, `dynamic_range`.

## Further reading

*Orientation pointers, not canonical citations — these don't go through the paper-library verification path.* Search terms: Kirshner reference-free SNR estimation, Janesick *Scientific Charge-Coupled Devices* (noise-floor physics, CMOS inherits most of it), microscopy-camera reviews (Spring).
