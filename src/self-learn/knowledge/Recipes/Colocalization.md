# Colocalization Analysis Playbook

## When to Use
Two-channel fluorescence imaging where you need to quantify spatial overlap between markers. Common scenarios: nuclei vs membrane, protein A vs protein B.

## Metrics

### Pearson's R
- **Range**: -1 to +1
- **Measures**: Linear correlation between pixel intensities in two channels
- **Interpretation**:
  - +1: perfect co-localization
  - 0: no correlation
  - -1: perfect anti-correlation (mutually exclusive signals)
- **Note**: Affected by background levels — threshold or subtract background first
- Nuclei vs membrane channels typically give R ≈ -0.3 to -0.5 (anti-correlated)

### Manders' Coefficients
- **M1**: Fraction of channel 1 intensity in regions where channel 2 is above threshold
- **M2**: Fraction of channel 2 intensity in regions where channel 1 is above threshold
- **Range**: 0 to 1 (NEVER exceeds 1.0 — if it does, there's an implementation bug)
- **Interpretation**: M1 = "how much of marker 1 overlaps with marker 2"

## Spectral Bleedthrough Correction
Many microscopes have cross-talk between channels:
- ~20% nuclear signal bleeds into membrane channel
- ~10% membrane signal bleeds into nuclear channel

**Correction formula:**
```python
nuc_corrected = nuc_raw - bleedthrough_mem_to_nuc * mem_raw
mem_corrected = mem_raw - bleedthrough_nuc_to_mem * nuc_raw
```

Apply correction BEFORE computing Pearson/Manders. Bleedthrough values depend on the specific fluorophore pair and filter setup.

## Key Code Pattern
```python
from src.core.analysis.colocalization import (
    pearson_r, manders_coefficients, costes_threshold,
)

# Costes-style auto-thresholds, then per-coefficient calls.
t1, t2, _ = costes_threshold(channel1, channel2)
r = pearson_r(channel1, channel2)
m1, m2 = manders_coefficients(channel1, channel2,
                               threshold1=t1, threshold2=t2)
```

`pearson_r` returns the Pearson coefficient (mask-aware); `manders_coefficients` returns the (M1, M2) tuple. There is no aggregating `colocalization()` wrapper — call the per-statistic helpers directly so callers see exactly which thresholds and masks each metric sees.

## Cell Counting in Two Channels
- Count cells using the brighter/more distinct channel
- Use `detect_cells` or `detect_blobs_log` on the nucleus channel
- Don't count in both channels and average — pick the most reliable one

## Pitfalls
- **Manders M1 > 1.0**: Implementation error — check threshold and normalization
  - Cause: dividing by co-localized intensity instead of total channel intensity
- **Forgetting bleedthrough correction**: Inflates Pearson R and Manders
- **Background not subtracted**: Pearson R is sensitive to background level
- **Using whole-image mean**: Better to use per-ROI or threshold-based measurements
- **Anti-correlation surprise**: Nuclei and membranes ARE anti-correlated — negative Pearson R is expected and correct

## Lessons Learned
- Nuclei vs membrane channels are expected to be anti-correlated (negative Pearson R)
- Manders M1 should never exceed 1.0 — if it does, check if denominator is total channel intensity vs co-localized intensity
- Bleedthrough correction before computing metrics significantly improves accuracy
