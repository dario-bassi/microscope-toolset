# Photoconversion / Lineage Tracing Playbook

**Assumes:** Photoconvertible fluorescent proteins (Kaede, mEos, Dendra2), SLM available, two-channel readout (pre / post). The bimodal-classifier shape — phase-contrast detection + natural-gap split on a discriminator channel — is generalised in `src/recipes/two_population_stain.py` (`classify_pipeline` — tested on ch597 photoconversion 10/10 r3).

## Sample Type
Cells expressing photoconvertible fluorescent proteins (Kaede, mEos, Dendra2, etc.)
that undergo irreversible green→red conversion upon UV/violet illumination.

## Channels
- **Pre-conversion**: nucleus-channel (green) = bright; membrane-channel (red) = dim
- **Post-conversion**: nucleus-channel = dim inside SLM; membrane-channel = bright inside SLM

## Workflow

### 1. Baseline imaging
```python
from src.core.hardware.core import set_objective, snap_all_channels, make_slm_circle, apply_slm
set_objective(core, 10)
imgs_pre = snap_all_channels(core)
```

### 2. SLM photoconversion
```python
# Use r=100 for good cell coverage (r=80 missed cells — ch560 R1)
slm_mask = make_slm_circle((256, 256), radius=100, core=core)
apply_slm(core, slm_mask)

# Snap twice to ensure conversion takes effect
for _ in range(2):
    for ch in channels:
        core.setConfig('Fake', ch)
        core.snapImage()
        core.getImage()

# Clear SLM
apply_slm(core, np.zeros((512, 512), dtype=np.uint8))
```

### 3. Verify conversion
Check intensity shift inside SLM circle:
- Nucleus should DROP (green Kaede consumed)
- Membrane should INCREASE (red Kaede appears)
- Conversion OK if: `nuc_post < nuc_pre * 0.8` OR `mem_post > mem_pre * 1.3`

### 4. Count converted cells — DUAL CHANNEL
**Key lesson (ch560):** Use BOTH channels for robust identification.
- BF local maxima inside SLM → candidate cells
- For each candidate, check: nucleus dropped OR membrane increased
- Cross-validate with membrane wall inversion and nucleus loss count

Methods ranked by reliability:
1. **Dual-channel** (BF position + nucleus drop + membrane increase) — most robust
2. **Membrane wall inversion** (distance transform of dark interiors enclosed by bright walls)
3. **Nucleus loss** (pre - post nucleus peaks inside SLM)
4. **BF local maxima alone** — undercounts in small regions (md=20 too coarse)

### 5. Timelapse tracking
Use snap() loops (MDA crashes pymmcore-proxy WebSocket):
```python
from src.core.analysis.tracking import match_centroids

# Track with max_displacement to avoid spurious matches
result = match_centroids(initial_xy, final_xy, max_displacement=25.0)
mean_disp_px = result['mean_displacement']
```

### 6. Division detection
**Be skeptical.** Count fluctuations between frames = tracking noise, NOT biology.
- Only report division if count monotonically increases
- If count goes up AND down between frames → noise → report 0 divisions

## Key Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| SLM radius | 100 px | Covers ~10 cells at 10x |
| BF min_distance | 15 px | Finer than md=20 for SLM subregion |
| max_displacement | 25 px | ~2.5× expected total displacement |
| Membrane threshold | Otsu or 15 (whichever is higher) | Fixed for temporal comparison |

## Common Pitfalls
- r=80 SLM mask covers too few cells (ch560 R1: 4 detected vs GT 10)
- BF md=20 undercounts in small SLM regions (only 4 cells vs 10)
- Frame-to-frame tracking without max_displacement → spurious large displacements
- Count fluctuations mistaken for division (ch560 R1: reported 11 divisions, GT=0)
- MDA sequences crash pymmcore-proxy WebSocket — always have snap() loop fallback

## Existing Modules
- `src.core.hardware.core`: `make_slm_circle()`, `apply_slm()`, `snap_all_channels()`
- `src.core.analysis.tracking`: `match_centroids()`, `match_frames()`, `track_multiframe()`
- `src.core.analysis.photoconversion`: `measure_activation()`, `signal_spread()`, `cell_dispersion()`

## Pre-converted-only variant (no SLM operation by agent)

**ch597 archetype** — the agent connects to a sample where some cells were
*already* photoconverted before connection. No SLM action is required;
the task is pure two-channel discrimination per cell.

Recipe:
1. Snap **phase-contrast** (label-free → finds *all* cells, both populations).
2. `peak_local_max` on inverted phase-contrast (sigma ≈ 4, min_distance ≈
   24 at 20× / 0.5 µm-px) → cell-centre seeds.
3. Sample DAPI / nucleus-channel intensity in a small disk (radius ≈ 6 px)
   around each centre. The DAPI distribution is sharply *bimodal* on this
   sample type — a natural-gap split (`src.core.analysis.intensity.classify_intensities(method='gaps')`)
   classifies cleanly without a hand-tuned threshold.
4. Submit positions of cells whose DAPI-centre intensity falls below the gap
   (= converted, dim nucleus). DAPI alone is sufficient because the gap is
   ~30+ counts; layering the membrane channel as a secondary check is
   defensible but rarely necessary.

**Why DAPI-centre alone, not membrane-vs-DAPI ratio:** the photoconverted
red signal is broadly spatially distributed (cell-body / wall) so its
per-cell mean is moderate; the unconverted green signal is *concentrated*
in the nucleus, giving a much wider range. The bimodality lives in the
DAPI distribution.

**Transfers to:** any sparse two-population live discrimination —
Calcein-AM/PI live/dead, FUCCI G1/G2 (with mCherry-Cdt1 vs Geminin-GFP),
mitochondrial polarised/depolarised (TMRM bright vs dim), drug-treated
vs vehicle in mosaic plates.

ch597 r1: 9 converted / 15 cells → bimodal gap of 40 cnt at threshold ~30.

## Active-conversion variant (clonal-expansion)

When the agent itself triggers the conversion (SLM disc → 1-event MDA
on the Kaede-red channel), the discriminator changes:

1. **Pre/post diff is the right signal, not the raw channel.** Persistent
   features that exist before AND after the SLM hit (Voronoi cadherin
   vertex artifacts, baseline membrane outlines) cancel cleanly in
   `(post - pre)`; the conversion-only signal stays. A single threshold on
   the diff has 5–10× better signal-to-noise than the same threshold on
   the raw post image.
2. **Freshly-converted cells are dim for the first few frames.** Render
   intensities of newly-converted Kaede-red are ~30–50 cnt for the first
   1–3 frames, climbing to ~80–120 cnt at saturation. A 5–10-frame
   stabilisation MDA *after* the conversion snap is required before the
   "initial count" measurement; otherwise the segmentation thresholds
   sit in the noise floor and the count is zero.
3. **Connected components beats blob_log on diff images.** The diff has
   ~10× more spurious local-maxima than the raw image (vertex-artifact
   sub-pixel phase drift between snaps adds 5–15 cnt of noise everywhere),
   which the blob_log threshold parameter is too sensitive to. Hard-
   threshold + connected-components is more robust on diffs; blob_log
   is better on raw images.
4. **Symmetric global counting.** Use the same segmentation on initial
   and final diffs; both counted globally over the whole FOV (daughter
   cells migrate during the timelapse so the clone is no longer disc-
   confined). Asymmetric disc-restricted-initial vs global-final biases
   the doublings ratio.
5. **doublings = log2(final / initial)** only works if `initial > 0`.
   Sign-check before submission: `final < initial` → impossible.

**Transfers to:** any active-conversion lineage / clonal-tracking
experiment — Kaede / Dendra / mEos protocols, photoactivatable GFP
labelling for migration tracking, optogenetic Cre-loxP recombination
where the activated subset must be quantified across divisions.
