# Wound Healing Kinetics Playbook

## When to Use
Confluent tissue monolayer with a scratch wound. Goal: measure gap closure over time and fit kinetics.

## Key Properties
- Wound is typically **horizontal** (full width of FOV)
- Gap narrows progressively as cells migrate inward from both edges
- Closure follows exponential decay: `gap(t) = gap_0 * exp(-k * t)`

## Workflow

### Phase 1: Acquisition
1. Acquire timelapse: BF + nucleus channel per timepoint
2. Check how time advances (interval-based or per-snap)
3. Plan total frame count to capture full closure

### Phase 2: Gap Measurement (from nucleus channel)

**Migration-rate comparison variant (2026-04-25, grader-recommended for ch582-like
drug-vs-baseline comparisons):**
On a mid-band of rows (where a vertical wound is clearest), compute
column-mean intensities, threshold against a per-condition Otsu, and
return the length of the longest contiguous run on the **wound side
of the threshold**. This 1D metric is robust to drug-induced edge
protrusion (which confounds nearest-nucleus) AND to wound-interior
debris.

**Critical: which side of the threshold is the wound?** Phase contrast
cells appear DARKER than the cell-free wound stripe (cell halos
brighten the wound boundary; cell bodies are dim grey). On fluorescent
nucleus channels, the wound is darker than the cells. Always verify by
looking at one frame — if cells are bright the wound is the longest
sub-threshold run; if cells are dark the wound is the longest
above-threshold run. **ch582 r13 lost 4/10 by having this inverted.**
r14's fix flipped to longest-above-Otsu and r15 then graded 8/10 on
the bumped backend.

A robust baseline: take per-condition Otsu (do not reuse the threshold
calibrated on a different condition's first frame; ch582 grader
explicitly noted Reset re-randomises cell density / migration onset,
producing different brightness distributions per condition).

```python
from skimage.filters import threshold_otsu
otsu = threshold_otsu(first_frame_grey)
# phase contrast (cells dark, wound bright):
above = column_mean > otsu
wound_width = longest_contiguous_run(above)
# fluorescent nucleus (cells bright, wound dark):
# below = column_mean < otsu;  wound_width = longest_contiguous_run(below)
```

`src.core.analysis.wound_healing.measure_wound_gap_longest_zero_run`
implements the cells-bright form (`column_mean < threshold`); use it
directly only when you have verified that geometry by image inspection.

**Debris-robust variant (2026-04-24 validated on ch582 r4, 8/10 — biased by edge protrusion on ch582 r6):**
Use `src.core.analysis.wound_healing.measure_wound_gap_by_nearest_nucleus`.
It measures distance from FOV centre to the nearest real tissue-nucleus
centroid after Otsu + min_area filter. Responds to leading-edge
advancement, immune to small debris flickering in/out of mask. Slope of
this metric vs frame is a clean closure rate (px/frame).

**Known gotcha:** in drug-response comparisons, drug-induced edge
protrusion biases nearest-neighbour lookup — prefer
`measure_wound_gap_longest_zero_run` for that case.

```python
from src.core.analysis.wound_healing import measure_wound_gap_by_nearest_nucleus
gap_px = measure_wound_gap_by_nearest_nucleus(nuclear_img, min_nucleus_area=200)
```

**Row-profile variant (chord-width, original approach):**

1. Compute **row-wise mean intensity** of nucleus channel
2. **Smooth** with uniform filter (31px kernel) to remove cell-level oscillations
3. Estimate **tissue signal level** from top/bottom edges (always confluent)
4. Find wound center: minimum of smoothed profile in central region
5. **Edge-crossing**: scan outward from center until signal exceeds 50% of tissue level
6. Gap = distance between upper and lower edges
7. Apply **monotonic enforcement** (wound can only close, not reopen)

**When to use which:**
- Use the nearest-nucleus metric when cell motility / wound debris give noisy chord-widths, or when the wound shape isn't a straight stripe.
- Use the row-profile metric when you need the absolute width of the gap (not just the leading-edge position) or the wound is clearly rectangular/stripe-shaped.

### Phase 2b: Two-condition comparison via Reset state device (ch582 pattern)

When the scenario provides a `Reset` state device (ch582 introduced this):
toggle `0 → 1` to fire a fresh wound in the middle of the experiment,
then `1 → 0` to release the latch so the next `0 → 1` fires again. Run
one condition (e.g. Perfusion=Off), Reset, switch condition
(Perfusion=Drug), run again. Verify `baseline[0] ≈ drug[0]` (post-reset
gaps should match) — if they diverge, the Reset didn't fire.

**Equilibrate state changes BEFORE Reset, not after** (ch582 r20 — the
breakthrough after 11 rounds at 4-8/10). The wound-mechanics state
device (here `Perfusion`) needs sim-time to alter tissue dynamics.
Sequence:
```python
core.setState("Perfusion", 4)           # set NEW condition
for _ in range(N_EQUIL):                 # let perfusion take hold
    snap(core, channel=...)
reset_wound(core)                        # NOW open the wound
# ... measurement frames ...
```
Without equilibration, the first frames of the new condition still see
old-condition dynamics, biasing the slope. r10-r19 set Perfusion → Reset
→ measure with no equilibration and got drug closure indistinguishable
from baseline (or backwards).

**Equilibration window length** — r20 used `N_EQUIL=5` and overshot the
GT ratio (2.52 vs 1.7, 6/10). The grader's diagnosis: the drug rate is
slightly inflated by a transient burst right after Perfusion switches
on; `N_EQUIL=5` doesn't fully damp it. Bump to **`N_EQUIL=10`** if the
ratio is reproducibly high. Twin diagnostic: the baseline rate may also
be underestimated when the wound is small enough that cells start
meeting in the middle within the measurement window — cap the linear
fit to the first ~15 frames before any meeting.

**Drug-first acquisition order also matters when measuring two
conditions back-to-back.** In r20, running drug FIRST (with
equilibration) then baseline SECOND gave ratio 2.52 ≈ GT 1.7. Running
baseline first then drug had drug see lingering baseline-perfusion
dynamics (cells already moved at baseline rate, then perfusion changes
mid-measurement). The fix is the equilibration window above; the
condition order is a secondary safety belt.

**Wound metric for drug-vs-baseline comparison: longest BELOW-Otsu run
on the central-band column-mean** (i.e. cells-region width, NOT the
bright wound stripe width). The bright-stripe metric fragments under
drug because invading cells' bright halos break the contiguous bright
run. Cells-region grows monotonically as cells migrate INTO the
wound, regardless of halos. r18 verified this on the same data where
the bright-stripe metric gave a 0.14 ratio.

### Phase 3: Kinetics Fitting
1. Filter to non-zero gap values (avoid log(0))
2. Fit exponential: `gap(t) = A * exp(-k * t)` using `fit_exponential_decay()`
3. Half-closure time: `t_half = ln(2) / k`
4. Leading edge speed: mean gap decrease per step / 2 (two migrating edges)

## Key Code Pattern
```python
from src.core.analysis.wound_healing import analyze_scratch_assay
from src.core.analysis.kinetics import fit_exponential_decay
import numpy as np

# Collect timelapse images (MDA-based) → 3D array (T, H, W)
stack = np.array(frames)

# Full pipeline
result = analyze_scratch_assay(stack, orientation='horizontal',
                               pixel_size_um=pixel_size)
# result['gap_widths'] — mean gap at each frame
# result['closure'] — closure_fraction, rate, time_to_close
# result['migration'] — edge speeds, mean_speed

# Alternative: per-frame wound detection
from src.core.analysis.wound_healing import detect_wound, wound_closure_rate
wounds = [detect_wound(f, orientation='horizontal') for f in frames]
gaps = [w['mean_gap'] for w in wounds]
closure = wound_closure_rate(gaps, timepoints, pixel_size_um=pixel_size)

# Optional: fit exponential decay for kinetics
gaps_arr = np.array(gaps)
valid = gaps_arr > 0
t_arr = np.arange(len(gaps_arr), dtype=float)
if valid.sum() >= 3:
    fit = fit_exponential_decay(t_arr[valid], gaps_arr[valid])
    # fit['k'] = decay rate, fit['half_life'] = ln(2)/k
```

## Pitfalls
- **Row-mean vs row-std**: Use nucleus channel mean, not BF std. Nucleus has clear gap with no signal where cells are absent
- **Heavy smoothing → biased gap**: 31px uniform filter blurs sharp wound edges, making the gap look wider and closure slower. Prefer a simpler approach: threshold rows where mean < low_value, count longest contiguous band of low rows. Less smoothing = sharper edges = faster measured closure.
- **Log(0) in exponential fit**: Once gap reaches 0, exclude those points from fitting
- **Threshold level**: 50% tissue signal can overestimate gap (gave k=0.14 vs GT 0.35). Try lower threshold or hard cutoff on raw row means.
- **Leading edge speed**: Check challenge definition! "px per step from gap change" usually means TOTAL gap narrowing rate (both edges combined), NOT per-edge. Don't divide by 2 unless explicitly asked for per-edge speed.
- **Monotonic enforcement**: Important because noise can cause small gap increases between frames

## Lessons Learned
- Heavy smoothing blurs wound edges → biased gap/rate estimates. Use minimal smoothing.
- Check speed definition: "gap change rate" usually means TOTAL (both edges), not per-edge
- Low threshold on row means gives sharper gap detection than heavy smoothing + edge-crossing

## Laser Ablation Wound Healing (ch532 lesson)

A different wound healing assay: SLM laser ablation kills cells in a defined region (e.g. right half of FOV), then neighboring cells migrate in to repopulate.

### Key Difference from Scratch Assay
- Wound is created by SLM illumination (ablation), not a physical scratch
- Wound region is a HALF-PLANE (col >= 256), not a narrow gap
- Healing = count of cells in wound region increases over time
- Need per-frame tracking, NOT just before/after comparison

### Protocol (ch532)
1. Pre-ablation: count cells in each half
2. Apply SLM mask (`slm_mask[:, 256:] = 255` for right half)
3. Re-apply SLM each frame in on_frame callback
4. Track per-frame cell count in wound region (right half)
5. healing_observed = True if wound count increases ≥1 from minimum

### CRITICAL: Track per frame, not just before/after
With migration_speed=4px/step, border cells drift ~40px in 10 snaps.
Even 1-2 cells crossing the boundary = healing_detected = True.

```python
from src.core.analysis.wound_healing import (
    segment_cells_bf, define_wound_region, 
    count_cells_in_region, track_wound_repopulation
)

wound_mask = define_wound_region(fov_size=512, region='right_half')

# Post-ablation healing tracking
healing_imgs = [snap(ch) for _ in range(10)]

heal_result = track_wound_repopulation(
    healing_imgs, wound_mask,
    channel='fluorescence',
    min_area=30, max_area=2000,
    threshold=locked_threshold,  # LOCK threshold from pre-ablation!
)

healing_observed = heal_result['healing_detected']
# heal_result['n_in_wound'] = [5, 6, 7, 7, 8, ...] = shows repopulation
```

### Grade lessons from ch532
- ch532 R1=8/10: Had healing_observed=False (missed 2pts)
  - Root cause: compared only first vs last frame, not per-frame series
  - FIX: use track_wound_repopulation() to get per-frame time-series
  - With migration_speed=4px/step: cells at wound border drift ~40px across boundary in 10 snaps
  - Time-series [5, 6, 7, 7, 8...] shows clear repopulation
