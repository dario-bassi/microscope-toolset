# Pitfall: Photobleaching during timelapse

> **When to use:** When fluorescence intensity declines steadily across a timelapse independent of the biology.

## Symptom

Mean fluorescence intensity drops steadily across a timelapse even though no biological change is occurring. Protein localisation or concentration measurements drift over time. Intensity-based quantification at late timepoints is systematically lower than at early ones.

## Cause

Every fluorescence snap deposits photons into the fluorophore, accelerating irreversible photoconversion to a dark state. With repeated illumination the signal decays — typically exponentially for organic dyes, more linearly for common FPs (mEGFP, mVenus). The decay rate depends on laser power, exposure time, objective NA, and the fluorophore's photostability.

## How to detect

1. **Flat control**: image a static target (fixed-cell slide or fluorescent bead) under the same acquisition protocol. If intensity drops over frames without any biology, the drop is bleaching not signal.
2. **Bleach curve shape**: a clean monoexponential decay (`I(t) = I₀·exp(−k·t)`) is diagnostic. A faster-than-exponential initial drop followed by plateau is also common for two-component fluorophores.
3. **Frame-over-frame ratio**: `mean(frame[t+1]) / mean(frame[t])` should be ~1.0 for stable signal. Consistent values < 0.99 per frame indicate bleaching.

## Fixes

### During acquisition
- **Reduce exposure**: halving exposure halves bleach rate with only √2 SNR cost (photon noise is the dominant term).
- **Reduce frame rate**: if the biology is slower than your current interval, acquire less frequently. See [[Core/Strategies/Gentle imaging]].
- **Reduce laser power**: preferred over shorter exposure for line-scan confocals (dwell time stays safe).
- **Use BF for tracking, fluorescence only for quantification**: acquire brightfield for every frame; acquire fluorescence only every N frames when quantification is needed.

### During analysis
- **Bleach correction** via `self_learn.analysis.fluorescence.bleach_correct(stack)`: fits a monoexponential to the median-intensity-vs-frame trace and normalizes each frame by the fitted decay envelope.
- **ΔF/F**: normalize each pixel's trace to its own baseline. Relative changes are bleach-robust; absolute intensities are not.
- **Per-ROI baseline correction**: subtract a linear fit of the pre-stimulus baseline from the whole trace (removes linear photobleaching component without assuming an exponential model).

## What bleach correction cannot fix

- Bleach correction recovers relative dynamics but inflates noise at late timepoints (you are amplifying a weaker signal).
- If >50% of the initial signal is bleached, the corrected trace may be dominated by autofluorescence.
- Irreversible bleaching cannot recover the original signal; it can only normalize comparisons across time.

## See also

- [[Core/Strategies/Gentle imaging]] — acquisition-side dose minimization.
- [[Core/Strategies/Timelapse design]] — exposure budget design from the timescale of the biology.
- [[Core/Concepts/Exposure and photodamage]] — dose budget and photodamage physics.
