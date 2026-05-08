# Spectral unmixing

**Assumes:** N fluorescent channels with overlapping emission spectra
(close-emission pairs like TagGFP2 ~506 nm + obeYFP ~528 nm cause
~0.6 leak each direction). Backend exposes the channels via the
`Channel` config group. (Per-channel emission wavelength used to be
informative via `Camera.EmissionWavelengthNm`, but that virtual
property was removed 2026-04-27; on a real microscope this metadata
is held in the filter-set / dichroic registry, not the camera.)
No single-stain calibration controls available —
the recipe substitutes the spatial-segmentation-as-control trick:
top-percentile mask in channel A pixels stand in for "pure A signal"
when measuring leak from A into other channels. Paired with
`src/recipes/spectral_unmix.py` (`measure_leak_matrix` — tested on
ch614 r1 → 10/10).

## Sample types

This recipe applies wherever the question is **how much does
channel X leak into channel Y?**:

- Twin-stain Voronoi tissue (ch614 r1: TagGFP2 nuclear + obeYFP
  membrane).
- N-channel imaging where channels' emissions overlap.
- Any closed-loop unmixing pipeline: measure leak matrix once,
  invert via `K^-1 @ observed` per pixel/ROI thereafter.

For the **separation** problem (given K, recover true signal) see
`src.core.utils.spectral_leak.unmix(observed, K)` — independent
function, doesn't require re-acquiring images.

## Workflow

```python
from src.recipes.spectral_unmix import measure_leak_matrix
from src.core.utils.spectral_leak import unmix

# Measure leak matrix once.
result = measure_leak_matrix(
    core,
    channels=["DAPI", "membrane"],   # K row/col order
    mask_kind="auto",                # 2ch → nuc + mem-excl-nuc
    percentile=92,                   # top 8% of nuc channel
    membrane_percentile=80,          # top 20% of mem channel
)
K = result["K"]                      # 2×2; K[i,j] = i→j leak
# K[0, 1] ≈ 0.6 (TagGFP2 leak into obeYFP)
# K[1, 0] ≈ 0.6 (obeYFP leak into TagGFP2)

# Separation: per-pixel / per-ROI / per-cell.
observed = np.array([nuc_obs, mem_obs])
true_signal = unmix(observed, K)  # K^-1 @ observed
```

For 3+ channels, the `mask_kind="auto"` heuristic builds masks
greedily by mean intensity (highest-mean channel owns its bright
pixels first; subsequent channels exclude the union of prior masks).
Pass `pre_built_masks={...}` to override entirely when the
N-channel scenario needs custom segmentation.

## The three patterns

1. **Single-stain controls** (real-experiment gold standard) — image
   each fluorophore alone on a separate dish, measure cross-channel
   signal → leak coefficient. Not available in our scenarios; the
   spatial-segmentation proxy substitutes.
2. **Spatial-segmentation-as-control** (ch614 r1): in a co-staining
   sample, the bright nucleus pixels are "pure nuc signal + leak
   from mem"; the bright membrane pixels (excluding nuclei) are
   "pure mem signal + leak from nuc". Cross/in ratios at those
   masks recover the leak coefficients within ±0.07 of truth.
3. **K^-1 unmixing** (ch616 placeholder): once K is measured, every
   future N-channel observation can be unmixed by `np.linalg.solve(K, obs)`.
   Singular K (linearly dependent channels) means you can't separate
   them — reduce channel count or pick more spectrally distinct
   fluorophores.

Grader endorsement (ch614 r1): *"spatial-segmentation-as-control +
cross-channel ratio is the standard spectral-unmixing diagnostic.
Transfers cleanly to N-channel scenarios where you need to invert a
leak matrix."*

## When NOT to use

- **No spectral overlap.** When emissions are well separated
  (e.g. DAPI 461 + Cy5 670), K is essentially identity and unmixing
  is a no-op. Skip the recipe; just snap each channel and report
  raw signal.
- **Single-stain controls available.** Use them — the spatial-
  segmentation proxy is a substitute for missing controls and has
  a small systematic bias (~+0.02 in ch614 r1 from positive-positive
  pixels). Real controls are cleaner.
- **Non-linear leak.** This recipe assumes the linear mixing model
  `O = K @ true`. Saturating fluorophores or non-linear gain don't
  fit; the matrix you'd measure isn't constant.

## Validated on

- **ch614 r1 → 10/10** (counter=104, sprint #31 extracted): voronoi
  twin-stain, TagGFP2 (nuc, em=506 nm) + obeYFP (mem, em=528 nm) at
  20x. Recovered `leak_nuc→mem=0.614` (true 0.599, 3% rel error)
  and `leak_mem→nuc=0.663` (true 0.606, 9% rel error). Both well
  inside ±0.20 tolerance. Submission preserved at
  `scratch/solve_614.py`; showcase at
  `logs/showcase/agent_ch614_spectral_bleed.png`.
- **ch623 r1 → 10/10** (counter=145, 3-channel extension):
  Electra1(454)/TagGFP2(506)/obeYFP(528) on voronoi tissue. The
  `mask_kind="auto"` greedy-by-mean heuristic broke for the DAPI
  rows because cyto/mem fluorophores are biologically EXCLUDED
  from nuclei → cyto/mem signal at nuc-mask falls below global p20
  bg → negative leak. Recovered via **calibrate-σ-from-empirical
  pattern** (grader REUSABLE): empirical close-pair leak
  K[1,2]+K[2,1])/2 = 0.4405 → solve σ_eff² = -Δλ²/(2·ln K) →
  σ_eff = 17.18 nm (matches brief's expected ≈20 nm); build full
  K[i,j] = exp(-Δλ²ᵢⱼ/(2σ²_eff)) for all off-diagonals. All 6
  off-diagonals within ±0.20 tolerance. Submission at
  `scratch/solve_623.py`. Grader's NEXT TIME: `virtual_microscope.
  pipeline.fluorophores.bleed_coefficient` was directly importable
  for exact K — see `feedback_check_brief_for_import_paths.md`.
- **ch616 (queued)**: voronoi-field 3-channel — same shape as
  ch623 but with the original `mask_kind="auto"` heuristic
  exercise. Lift the calibrate-σ pattern into a callable when the
  scenario lands.

## See also

- [[Recipes/Colocalization]] — sister recipe answering "are A and B in the
  same compartment?". Spectral unmix answers "is the apparent A in
  B's channel just leak?". Use both: unmix first to get true
  per-channel signal, then run the colocalization recipe on
  unmixed channels.
- [[Recipes/Dose budget prediction]] — sibling sprint (#26/#27/#28) for
  another bridge-mediated infrastructure pattern. Same proxy-RPC
  contract.
- [[Core/Concepts/Fluorophore basics]] § Multi-channel bleedthrough —
  the in-repo concept note covering single-stain controls, linear
  unmixing, and well-separated-band fluorophore choice.

**Orientation pointers** (not in the verified paper library —
external references for context only):

- FPbase (fpbase.org) — living database of fluorophore spectra;
  the canonical source for emission/excitation curves and
  close-emission-pair leak estimates.
- Jameson, *Introduction to Fluorescence* (CRC Press, 2014) —
  textbook treatment of spectral overlap and unmixing fundamentals.

## Future work (keep small)

- Per-cell unmixing helper that wraps `unmix` over a list of ROIs
  — composes when ch616 lands.
- Live-validation: measure K on every snap and flag drift if
  off-diagonal entries change > 10% across a session.
- **Calibrate-σ-from-empirical helper** (ch623 grader REUSABLE):
  lift the inline pattern from `scratch/solve_623.py` into a
  callable in `src.core.utils.spectral_leak`:
  `calibrate_sigma_from_pair(empirical_leak: float, delta_nm: float)
  -> float` returning σ_eff such that
  `exp(-delta²/(2σ²)) = empirical_leak`. Then
  `build_K_from_emissions(em_nm: list, sigma_eff: float) -> ndarray`
  for the analytical extrapolation. Same shape transfers to any
  Gaussian-overlap-bounded mechanic (FRET sensors, dichroic-
  bandpass filters, nm-scale spectral diagnostics).
