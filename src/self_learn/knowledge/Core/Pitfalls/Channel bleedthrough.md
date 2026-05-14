# Pitfall: Channel bleedthrough in multi-channel imaging

> **When to use:** When a channel that should be dark shows signal that tracks cell shape and correlates with a brighter channel.

## Symptom

A channel that should be dark (no fluorophore expressed) shows a dim but real-looking signal that tracks cell shape and roughly correlates with a bright channel. Colocalization analysis reports a high correlation between two markers that should not colocalize. Ratio-metric measurements (channel A / channel B) are systematically wrong.

## Cause

Bleedthrough (also called spectral crosstalk) occurs when the emission of fluorophore A partially overlaps the detection band of channel B. The amount of bleedthrough depends on:
- The fluorophore pair's emission spectra overlap.
- The filter set geometry (narrower bandpass = less bleedthrough).
- The relative brightness of the two fluorophores (a very bright fluorophore bleeds more into a dim channel than a dim one into a bright one).

## How to detect

1. **Single-label controls**: image a sample with only fluorophore A and measure its signal in channel B. The ratio `signal_B / signal_A` is the bleedthrough fraction. Repeat for each fluorophore.
2. **Correlation before and after correction**: if channels are highly correlated before correction but decorrelate after, bleedthrough was the cause.
3. **Predictive brightness ratio**: use `self_learn.utils.fluorophore_brightness.predict_brightness_ranking` to estimate expected SNR per channel. A channel that is dimmer than predicted is likely receiving bleedthrough from a brighter neighbour.

```python
from self_learn.utils.spectral_leak import estimate_bleedthrough_fraction

# Single-label image of fluorophore A:
bt_fraction = estimate_bleedthrough_fraction(
    donor_image=single_label_A_in_channel_A,
    acceptor_image=single_label_A_in_channel_B,
)
print(f"Bleedthrough A→B: {bt_fraction:.1%}")
```

## Fix: linear unmixing

Correct multi-channel images using the bleedthrough matrix measured from single-label controls:

```python
from self_learn.utils.spectral_leak import apply_bleedthrough_correction

# correction_matrix shape: (n_channels, n_channels)
# correction_matrix[i, j] = fraction of channel j leaking into channel i
corrected_channels = apply_bleedthrough_correction(raw_channels, correction_matrix)
```

For two channels (green/red), the correction is:
```python
corrected_red = raw_red - bt_fraction_green_to_red * raw_green
corrected_green = raw_green - bt_fraction_red_to_green * raw_red
```

## Acquisition fix: sequential imaging

Use sequential (not simultaneous) acquisition: acquire channel A with its laser on, then turn it off and acquire channel B. Eliminates direct excitation crosstalk (though not emission bleedthrough from a still-excited fluorophore). For live cells this introduces a short time delay between channels.

```python
from useq import MDASequence

seq = MDASequence(
    channels=[{"config": "GFP"}, {"config": "RFP"}],
    axis_order="tpc",  # time → position → channel (each channel acquired separately)
)
```

## Fluorophore selection guideline

Choose fluorophores with maximally separated emission peaks. Common safe pairs:
- DAPI (460 nm) + GFP (510 nm) + mCherry (610 nm): good separation.
- GFP + YFP: dangerous — substantial bleedthrough in both directions.
- mNeonGreen + mScarlet-I: better separation than GFP/mCherry with higher brightness.

## See also

- [[Core/Strategies/Multichannel scan]] — multi-channel MDA acquisition.
- [[Core/Strategies/Imaging parameter optimization]] — SNR and exposure tuning per channel.
- `self_learn.utils.spectral_leak` — bleedthrough estimation and correction utilities.
- `self_learn.utils.fluorophore_brightness` — brightness ranking prediction.
