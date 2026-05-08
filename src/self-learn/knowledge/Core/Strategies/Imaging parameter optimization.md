# Imaging Parameter Optimization

## When to Use
- Fluorescence image is too dim or saturated at default settings
- Need to find optimal exposure/gain balance for SNR
- Photobleaching is a concern (minimize unnecessary exposure)
- Working with unfamiliar instrument or sample

## Key Principles

### 1. Minimize fluorescence frames
Every fluorescence snap bleaches the sample. Strategy:
- Use BF for scouting (free, no bleaching)
- Plan parameter tests before starting
- 3-5 test frames should suffice for most optimization

### 2. SNR = (signal_mean - bg_mean) / bg_std
- **Signal**: mean intensity of detected objects
- **Background**: mean intensity of non-object pixels
- **Noise**: standard deviation of background
- Good SNR for counting: > 5
- Use `compute_snr()` from `src.core.analysis.intensity`

### 3. Exposure vs Gain tradeoffs
| Parameter | Signal | Noise | Bleaching | Speed |
|-----------|--------|-------|-----------|-------|
| Exposure ↑ | Linear ↑ | √ ↑ (photon noise) | Linear ↑ | Slower |
| Gain ↑ | Linear ↑ | Linear ↑ (read noise amplified) | No change | No change |

**Rule of thumb**: Increase exposure first (better SNR per unit signal),
use gain only when exposure is impractical (fast events, severe bleaching).

### 4. Saturation awareness
- uint8: max 255. uint16: max 65535.
- If >5% of pixels are at max → reduce exposure or gain
- Some saturation on brightest nuclei is acceptable if counting is the goal

## Workflow

```python
from src.core.analysis.intensity import compute_snr
from src.core.hardware.core import snap

# 1. Scout with BF (free)
bf = snap(core, channel='brightfield')

# 2. Default fluorescence to assess (resolve channel via config, don't hardcode)
nuc = snap(core, channel=ch_nucleus)
snr_info = compute_snr(nuc)
print(f"Default: signal={snr_info['signal_mean']:.0f}, SNR={snr_info['snr']:.1f}")

# 3. If dim (signal < 80 or SNR < 5), optimize
# Test 2-3 combos, pick best SNR
combos = [(150, 4), (100, 8), (200, 4)]  # (exposure, gain)
for exp, gain in combos:
    core.setProperty('Camera', 'Exposure', exp)
    core.setProperty('Camera', 'Gain', gain)
    img = snap(core, channel=ch_nucleus)
    result = compute_snr(img, threshold=threshold)
    print(f"Exp={exp} Gain={gain}: signal={result['signal_mean']:.0f}, SNR={result['snr']:.1f}")
```

## MDA-based Parameter Sweep
For systematic parameter search, use native MDAEvent `properties` field:

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

sweep_events = [
    MDAEvent(exposure=exp,
             properties=[('Camera', 'Gain', str(gain))],
             metadata={"sweep_params": {"exposure": exp, "gain": gain}})
    for exp, gain in [(100, 2), (100, 4), (150, 4), (200, 4)]
]

results = []
def on_frame(img, event):
    snr = compute_snr(img)
    params = event.metadata.get("sweep_params", {})
    results.append({**params, **snr})

run_events(core, sweep_events, on_frame=on_frame)
best = max(results, key=lambda r: r['snr'])
```

## Common Issues
- **Background too bright**: Gain amplifies everything. If bg_mean > 50, reduce gain, increase exposure
- **Noisy but bright**: High gain + short exposure. Switch to lower gain + longer exposure
- **Saturated spots**: Reduce exposure first (preserves dynamic range better than reducing gain)
- **Photobleaching visible**: Each frame is dimmer. Take measurement frames first, optimization after

## Literature

- [[Papers/Jin 2020]] — DL-SIM shows that with a trained reconstruction prior the achievable exposure and raw-frame-count floors drop dramatically (100× fewer photons per frame, 5× fewer frames per super-resolved plane). When a content-aware model is in the loop, "optimal" exposure shifts below what SNR-on-a-single-frame would suggest.
- [[Papers/Durand 2018]] — canonical reference for *online* parameter optimisation: a contextual-bandit / Gaussian-process optimiser runs during the real acquisition and balances resolution, SNR, and photodamage as a multi-objective trade-off, eliminating the "exploration phase then imaging phase" split that an offline sweep imposes.
- [[Papers/Bilodeau 2024]] — the offline-training counterpart to Durand 2018: a physically grounded STED simulator (photobleaching + depletion-PSF + point-scanning dynamics + DL structural prior) lets a reinforcement-learning agent learn the resolution-vs-photodamage policy entirely in silico and deploy zero-shot on the real microscope, eliminating per-sample exploration dose.
