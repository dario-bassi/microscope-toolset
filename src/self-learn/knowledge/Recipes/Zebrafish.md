# Zebrafish Embryo Playbook

## When to Use
48 hpf zebrafish embryos — cardiac imaging, vascular phenotyping, angiogenesis screening.

## Anatomy Quick Reference (48 hpf)
- **Head**: eye (dark circle), brain vesicles
- **Heart**: two chambers (atrium + ventricle), visible in cardiac fluorescence
- **Trunk**: somites (~17-20 µm spacing), notochord (bright band in BF)
- **ISVs**: intersegmental vessels between DA (dorsal aorta) and DLAV
- **Tail**: thin, tapers posteriorly

## Heart Rate Measurement (ch420=8/10)

### Workflow
1. **10x overview** → locate embryo head/heart
2. **Navigate** to heart center (fluorescence centroid)
3. **ZOOM TO 40x** — don't stay at 10x. High mag needed for:
   - Precise Z-plane (40x DOF=1.5 µm vs 10x DOF=6 µm)
   - Chamber morphology verification
   - AV valve function assessment
4. **Z-scan at 40x** for best cardiac signal (heart at Z≈-5 in ch420)
5. **Fast MDA timelapse**: 200+ frames, 0.05s interval (20 Hz)
   - Single fluorescence channel for max temporal resolution
   - Zebrafish heart ~1-5 Hz depending on temperature
6. **FFT analysis** for heart rate extraction

### Analysis
```python
from src.core.analysis.temporal import fft_spectrum, autocorrelation, detrend
spectrum = fft_spectrum(detrend(intensities), dt=interval)
heart_rate_bpm = spectrum['dominant_freq'] * 60
```
Cross-validate: FFT + autocorrelation + peak counting (all should agree within 5%).

### Temperature-Heart Rate (Q10)
| Temp (°C) | Heart Rate (bpm) | Frequency (Hz) |
|-----------|------------------|-----------------|
| 20        | ~86              | ~1.4            |
| 30        | ~172             | ~2.9            |
| 37        | ~282             | ~4.7            |

Q10 ≈ 2.0 for zebrafish heart. Formula: `Q10 = (R2/R1)^(10/(T2-T1))`

Temperature device: `core.setState('Temperature', state_idx)` — labels: 20(0), 4(1), 25(2), 30(3), 37(4), 42(5)

## Cardiac Dose-Response (ch422/423)

### Workflow
1. **10x overview** → locate heart via membrane-channel (Tg(myl7:mCherry))
2. Navigate to heart centroid, Z-scan for best cardiac signal (Z≈-7)
3. **Baseline HR**: MDA timelapse (150+ frames, 20 Hz), FFT analysis
4. **For each drug concentration**: setState('Anesthesia', N), wait 1s, repeat timelapse
5. **Hill fit**: HR/HR_max = 1/(1 + (C/IC50)^n)

### Anesthesia Device
`core.setState('Anesthesia', state_idx)` — labels: None(0), 0.01%(1), 0.02%(2), 0.04%(3)

### Tricaine Dose-Response (25°C)
| Concentration | HR (bpm) | % Baseline |
|---------------|----------|------------|
| None          | 120      | 100%       |
| 0.01%         | 64       | 53%        |
| 0.02%         | 24       | 20%        |
| 0.04%         | 8        | 7%         |

IC50 = 0.0106% (106 ppm), Hill n = 2.12. Near-complete arrest at 0.04%.

### Analysis Code
```python
from src.core.analysis.temporal import fft_spectrum, autocorrelation, detrend
from scipy.optimize import curve_fit

def hill_inhibition(c, ic50, n):
    return 1.0 / (1 + (c / ic50) ** n)

popt, _ = curve_fit(hill_inhibition, concs, normalized, p0=[0.01, 2.0])
```

## Angiogenesis Screening (ch421=7/10, revised from 5)

### CRITICAL: Channel Selection
- **READ challenge notes** for which channel has which fluorophore!
- ch421: membrane-channel = mCherry (Tg(myl7:mCherry), cardiac myocytes ONLY)
- ch421: nucleus-channel = GFP (Tg(flk1:GFP), ALL vascular endothelium)
- **ISVs are in the GFP/vasculature channel** — NOT the membrane channel
- This is the most common error: assuming membrane = vasculature

### Workflow
1. **10x overview** → locate embryo, identify trunk extent
2. **READ channel descriptions** in challenge notes — know which channel has vasculature
3. **Navigate to trunk** region (posterior to yolk sac)
4. **Z-scan for ISV plane in CORRECT channel** — Z≈15-25 µm
5. **Zoom to 40x** for ISV detail (IMPORTANT: 10x ISVs are 1-2 px wide)
6. **Tg(flk1:GFP)**: should show ALL vasculature in fluorescence
7. **Count and classify ISVs**: normal / truncated / absent

### ISV Analysis from BF (fallback if no fluorescence)
- Dorsal trunk region (y=218-250 at 10x): ISVs create dark vertical structures
- Notochord dark band (y=251-260): melanocytes sit at somite boundaries
- Column intensity profile → find_peaks → somite boundary positions
- Classification per position: dark pixel count in ±2px column window
  - Normal: very_dark(< 100) ≥ 3 AND dark(< 140) ≥ 6
  - Truncated: dark ≥ 3 OR very_dark ≥ 1
  - Absent: no clear dark structure

### Severity Scoring
- 0: no defect (all ISVs normal)
- 1: mild (<25% affected)
- 2: moderate (25-75% affected)
- 3: severe (>75% affected)

SU5416 (VEGFR inhibitor) typically gives moderate phenotype (score 2).

### GT Reference (ch421)
- 18 ISVs total: 10 normal / 5 truncated / 3 absent → severity 2/3
- ISV spacing ~17-20 µm (somite spacing)
- SU5416 (VEGFR inhibitor) → moderate phenotype

## Key Lessons
- **ALWAYS read channel descriptions** — which fluorophore is in which channel?
- **ALWAYS zoom to recommended magnification** — 10x is for overview only
- Heart rate needs Nyquist: sample at >2× expected rate (e.g., 20 Hz for up to 5 Hz heart)
- Temperature device available: 9 temperature settings from 4°C to 42°C
- ISV spacing ~17-20 µm in 48 hpf embryo (matches somite spacing)
- Melanocytes at notochord provide reliable somite boundary markers
- **ch421 post-mortem**: Searched membrane-channel exhaustively. Zero trunk signal.
  Fell back to BF. Grade revised 5→7 after virtual-env found channel config bug (nucleus-channel
  was mapped to mScarlet3 instead of TagGFP2). My feedback about missing fluorescence was correct.
