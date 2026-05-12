# Exposure and photodamage

Exposure sets how much light reaches the camera per frame. It's the single knob most entangled with data quality: too low and you can't see, too high and you destroy both the dynamic range of the image and the biology you're trying to image. Get it right *once* per sample/channel combination before running the acquisition — it's much cheaper than re-running the experiment.

## The three constraints

1. **Avoid saturation.** A pixel at the camera's ceiling (255 for uint8, 65535 for uint16) has lost its brightness information forever. A few saturated pixels on the brightest objects may be acceptable for a counting task; >5% saturation is almost always too much, because it flattens differences you later want to measure.

2. **Keep enough contrast.** The signal of interest (cell vs. background, bright spot vs. diffuse stain) must stand above read noise and shot noise. As a rough target: signal-mean minus background-mean should be ≥ 5× the standard deviation of the background. Below that and detection starts producing false positives from noise.

3. **Minimize dose.** Every photon that hits a fluorophore can either emit (good) or photobleach (bad), and any photon absorbed by the sample contributes to phototoxicity. The recipe is always "as much as needed, no more": use the lowest intensity and shortest exposure that still give adequate contrast.

The practical rule follows: **reduce intensity until SNR is marginal, then back off by 20-30%.** You've now found the lower edge of usable-and-gentle.

## How to calibrate in practice

```python
from useq import MDAEvent

# Sweep exposure at constant intensity, pick the lowest exposure that
# still exceeds the SNR floor. See src/core/workflows/optimization.py
# for the full sweep + analyze workflow.
events = [
    MDAEvent(channel={"config": ch}, exposure=e)
    for e in (10, 25, 50, 100, 200)   # ms
]
```

For each frame, compute `(signal_mean - bg_mean) / bg_std` — that's your SNR. Plot it vs exposure. The curve usually saturates (log-linear) once enough photons are accumulated; pick the shortest exposure on the plateau.

Repeat per channel. Brightfield and fluorescence have very different exposure ranges; don't reuse one value.

## Saturation check

A single scalar is enough for a quick check:

```python
frac_saturated = float((img >= img.dtype_ceiling()).sum()) / img.size
```

If `frac_saturated > 0.05` on your region of interest, reduce exposure or intensity before continuing. Saturation in a small number of very bright pixels (outlier speckles, reflections) is usually tolerable; widespread saturation is not.

## Photodamage is cumulative

Phototoxicity and photobleaching both scale with total light dose, not per-frame dose. This has implications for timelapses:

- A 100-frame timelapse at 50 ms exposure delivers 2× the dose of a 50-frame at 50 ms.
- Halving exposure and doubling frame count gives the same dose (good for temporal resolution, neutral for bleaching).
- Halving *intensity* and doubling exposure is roughly the same dose but the sample has more time to recover between photons — sometimes gentler for bleaching, depending on the fluorophore's triplet-state dynamics.

For the longest live experiments: use the lowest-magnification objective that can still answer the question, accept more noise per frame, and filter temporally if needed. Denoising in software is usually cheaper than lost cells.

## When you can't avoid saturation

Some experiments have bright references (beads, fiducials, autofluorescent debris) you can't dim. Two ways out:

- **HDR-style**: acquire short and long exposures, combine them. `core.hardware` exposes exposure per MDAEvent.
- **Crop or mask**: compute metrics only on regions that aren't saturated. The saturated regions are lost data, not errors.

## See also

- `[[Core/Strategies/Imaging parameter optimization]]` — the full workflow for exposure/gain sweeps with SNR measurement.
- `[[Core/Strategies/Gentle imaging]]` — dose-minimization patterns for long timelapses.
- [[Papers/Weigert 2018]] — CARE reshapes the dose-SNR trade-off: acquire at ~60× lower photon budgets, recover SNR offline with a trained deep-learning prior.
- ``../../../src/core/analysis/intensity.py`` — `compute_snr()` implementation.
- ``../../../src/core/workflows/optimization.py`` — `optimize_exposure()`, parameter sweep workflow.

## Further reading

*Orientation pointers, not canonical citations — these don't go through the paper-library verification path.* Search terms: phototoxicity-in-live-cell-imaging reviews (Icha, Laissue and others), reference-free SNR quantification (Kirshner-style), standardised reporting for live-imaging best-practices.
