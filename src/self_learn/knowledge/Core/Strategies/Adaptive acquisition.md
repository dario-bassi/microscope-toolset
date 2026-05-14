# Adaptive Acquisition

> **When to use:** When the experiment requires survey-then-zoom to find regions of interest, or continuous patrol for rare events that occur at unpredictable times or positions.

> **Note**: Code examples below use conceptual pseudocode. All multi-frame
> loops MUST use `run_events(core, generator(), on_frame=callback)` in practice.
> See `[[Core/Concepts/MDA generators]]` for the correct MDA patterns.

## Pattern: Survey -> Decide -> Zoom -> Measure

```
10x survey -> detect ROIs -> rank -> switch to 40x -> re-detect -> measure
```

## Step 1: Low-Mag Survey (10x)

```python
set_objective(core, 10)  # Integer, NOT string "40x"
fov_um = core.getImageWidth() * core.getPixelSizeUm()
positions = grid_positions(center=(0, 0), nx=5, ny=5, fov_um=fov_um, overlap=0.1)

roi_candidates = []
for (x, y) in positions:
    core.setXYPosition(x, y)
    img = snap()
    score = compute_interest_score(img)  # Cell density, fluorescence, phenotype
    if score > threshold:
        roi_candidates.append((x, y, score))
```

## Step 2: Rank and Select

```python
roi_candidates.sort(key=lambda r: r[2], reverse=True)
top_rois = roi_candidates[:n_rois]
```

Do NOT commit to the first interesting spot. Survey the full area first.

## Step 3: Switch to High Mag (40x)

```python
set_objective(core, 40)  # FOV shrinks significantly at higher magnification
```

**Critical**: You MUST re-detect at 40x. A 10x centroid is ~5um accurate,
which is a 20-pixel error at 40x. The 10x position gives the neighborhood only.

## Step 4: Re-Detect and Measure

```python
for (wx, wy, _) in top_rois:
    core.setXYPosition(wx, wy)
    img_40x = snap()
    detections_40x = detect_cells(img_40x)  # Fresh detection at high res
    for det in detections_40x:
        results.append(measure_cell(det, img_40x))
```

## MDASequence + Adaptive Generator

```python
from useq import MDASequence, Position
survey = MDASequence(
    stage_positions=[Position(x=x, y=y) for (x, y) in positions],
    channels=["BF", "nucleus"],
)

def detail_gen(top_rois):
    for (wx, wy, _) in top_rois:
        core.setXYPosition(wx, wy)
        core.waitForDevice(core.getXYStageDevice())
        yield MDAEvent(channel={"config": "BF"})
```

## Magnification Selection

| Obj | Use Case                 |
|-----|--------------------------|
| 10x | Survey, counting, tiling |
| 20x | Intermediate, tracking   |
| 40x | Morphology, subcellular  |

Always query `core.getPixelSizeUm()` for actual values — they vary by instrument.

## Pattern: Patrol -> Detect -> Zoom -> Capture -> Return

For monitoring events that happen at unpredictable times and places
(cell division, calcium transients, rare phenotypes):

```
10x patrol (N positions, cycling) -> detect event -> 40x rapid timelapse -> return to patrol
```

This extends survey→zoom from a one-shot to a continuous loop. Key differences:
- **Time-sharing**: Must allocate imaging time across positions efficiently
- **Event detection**: Score each visit for "interestingness" vs baseline
- **Rapid response**: When triggered, must switch mag + start timelapse fast
- **State tracking**: Per-position history enables trend/change detection

### Implementation: `src/workflows/adaptive_monitor.py`

Two APIs:
1. **Imperative** — `adaptive_monitor(core, positions, scorer, ...)`
   Multiple `run_events()` calls, one per position per round.
2. **MDA-native** — `adaptive_monitor_events(positions, ...)`
   Generator + on_frame + state triple. Uses CustomAction for objective switching.

### Writing a Good Event Scorer

The scorer receives `(image, PositionState)` and returns a float score.
`PositionState` has `.recent_means` (last 10 visit means) and `.scores` (all prior scores).

Good patterns:
- **Intensity change**: `abs(current - baseline) / baseline` (calcium, bleaching, arrival)
- **Object count change**: detect cells, compare count to running average
- **Morphology change**: roundness increase (pre-mitotic), area decrease (apoptosis)
- **Frequency domain**: FFT power in a band (oscillating calcium)

Bad patterns:
- Adaptive thresholds (masks true change over time)
- Looking at only the current frame (no baseline comparison)

### When to Use This vs Survey→Zoom

| Scenario | Pattern |
|----------|---------|
| Fixed sample, find ROIs once | Survey→Zoom (one-shot) |
| Live sample, rare events | Patrol→Detect→Zoom (continuous) |
| Known positions, timed acquisition | MDASequence with time_plan |
| Unknown positions, unknown timing | Patrol→Detect→Zoom |

## Pattern: Modality switch (scout → high-resolution burst → scout)

A *spatial* survey→zoom can be replicated on the **optical-modality**
axis: low-dose widefield scouting between high-detail bursts in a
distinct optical mode (STED, lattice light-sheet, confocal). Costs
vs benefits trade through the bleach-budget bookkeeping rather than
the magnification budget.

**Modality contract** (each profile defines four knobs):

- ``widefield_scout``: full PSF, generous exposure, low bleach
  multiplier (×0.3), no channel allow-list — survey channels freely.
- ``confocal``: tighter PSF (×0.7), 1× bleach.
- ``sted_burst``: tight PSF (×0.25), short exposure ceiling (~50 ms),
  high bleach (×8), narrow channel allow-list (one reporter only).
- ``lattice_lightsheet``: tighter PSF (×0.6), short exposure,
  reduced bleach (×0.5), volume-style.

**Recipe (ch605 r1, 10/10):**

```python
core.setState("Modality", 0)   # widefield_scout
img_scout = snap(core, "GCaMP", exposure_ms=50)
# … detect a peak ≥ baseline + 30 cnt …
core.setState("Modality", 2)   # sted_burst
img_sted = snap(core, "GCaMP", exposure_ms=20)
core.setState("Modality", 0)   # back to scout
```

Two switches per event (scout → sted, sted → scout), counted
honestly. Bleach dose self-tracked:
``cum += exposure_s × bleach_multiplier`` per snap.

**Decision rules:**

1. *Default to scout.* Bleach ×0.3 means scouts are ~30× cheaper than
   STED bursts on the same exposure. Stay in scout until an event
   triggers a burst.
2. *One STED snap per event, not many.* The ×8 bleach multiplier
   makes "STED every frame" a budget catastrophe — at 50 ms snaps on
   a 2.0 budget, ~5 STED snaps exhaust it.
3. *Verify the channel allow-list with a forbidden-channel snap.*
   If the modality has ``channels_allowed=["GCaMP"]`` and you snap
   E-cadherin under it, the frame should be all-zero. The scout
   should let everything through.
4. *Switch back to scout immediately after the burst.* Don't leave
   the agent in STED waiting for the next event — every snap there
   costs 8× more.

**Tolerance shape (from ch605):** ``n_modality_switches ≥ 2``,
``cum_bleach_dose < 2.0``, channel allow-list verified by two
diagnostic frames. Sharpness in viewport-px isn't always a gate —
sub-pixel PSFs are dominated by sample noise rather than optical
mode.

## Common Pitfalls

- **Trusting low-mag positions at high-mag**: Always re-detect after zooming.
- **Incomplete survey**: Survey first, decide second.
- **Objective switching failures**: Verify with `core.getState('Objective')`.
- **FOV mismatch in counts**: 10x sees 16x the area of 40x.
- **Re-triggering**: After capturing an event, suppress that position temporarily.
- **Baseline drift**: Use a rolling window (last 5-10 visits), not all-time mean.

## See also

- [[Core/Concepts/MDA generators]] — generator mechanics.
- [[Core/Strategies/Multi-scale morphometry]] — survey→zoom implemented for shape measurement.
- [[Core/Strategies/Feedback control]] — closed-loop interventions that adapt mid-run.
- [[Core/Pitfalls/FOV vs well coverage]] — survey-tile sizing.

## Literature

- [[Papers/Rabut 2004]] — target-driven acquisition: the FOV is a closed-loop setpoint attached to the cell (XY stage + Z focus servo on the fluorescence mass centre), time-shared across a revisit list of independently drifting cells. Single-cell analogue of McDole 2018's specimen-tracking and the canonical pre-DL example of "the microscope follows the biology, not the coordinate".
- [[Papers/Jackson 2009]] — pre-DL active-learning acquisition: utility = information-gain about an online-updated scene model minus an explicit photobleaching / phototoxicity / time cost, with a principled stopping criterion when no candidate region's predicted information gain outweighs its cost. Formal ancestor of Ye 2025's uncertainty-driven rescan and Kandel 2023's Expected-Reduction-in-Distortion scan-point selection.
- [[Papers/André 2023]] — data-driven microscopy (DDM): a two-phase framework where a data-independent survey builds population-wide phenotype scores, then a data-dependent phase re-visits objects with per-object acquisition parameters (magnification, exposure, frame rate). Generalises survey→rank→zoom from single-feature ranking to population-context phenotype scoring.
- [[Papers/Morgado 2024]] — review-of-record for data-driven microscopy: organises the field by what the closed loop adapts (illumination, acquisition rate, modality, triggered experiment) and pairs each axis with the ML method best suited to it. Names the data-driven (population-context) vs task-driven (research-question-conditioned) scoring distinction — the choice the "Writing a Good Event Scorer" section above implicitly makes.
- [[Papers/Conrad 2011]] — Micropilot: canonical survey→zoom paper. Online phenotype classifier on low-mag survey tiles triggers unattended high-mag multi-channel time-lapse or FRAP protocols per hit; the historical blueprint for the "low-mag scan + targeted high-mag follow-up" pattern.
- [[Papers/Mahecic 2022]] — the canonical event-driven-microscopy paper; CNN detector triggers fast-rate acquisition during the event, slow-rate polling otherwise.
- [[Papers/Alvelid 2022]] — modality-switching adaptive acquisition: a widefield biosensor scout gates localised STED nanoscopy bursts, extending survey→zoom from spatial scale-switching to optical-mode-switching (widefield → STED).
- [[Papers/Shi 2024]] — SmartLLSM: YOLOv5 on low-dose epifluorescent widefield frames triggers autonomous per-hit lattice light-sheet acquisition. The deep-detector modern sibling of Micropilot and the epi-→-LLSM sibling of Alvelid's widefield-→-STED mode switch.
- [[Papers/Royer 2016]] — AutoPilot: closed-loop optimisation of light-sheet/detection-plane alignment during long live imaging; adaptive acquisition on the instrument-calibration axis, complementary to content-aware triggering.
- [[Papers/He 2020]] — smart rotation SPIM: on-the-fly image-quality scoring picks the next sample-rotation angle to maximise marginal coverage. Generalises adaptive acquisition from "which position/magnification/ROI next" to "which view angle next", with phototoxicity and data-volume savings from stopping once coverage is saturated.
- [[Papers/McDole 2018]] — specimen-tracking acquisition: the imaged volume itself is a closed-loop setpoint that follows a developing mouse embryo (centroid, bounding box, Z range re-measured per time point) over 48 h. Sibling to AutoPilot on the "instrument adapts to sample" axis — AutoPilot adapts optical alignment, McDole adapts the imaged volume.
- [[Papers/Ye 2025]] — calibrated pixel-wise reconstruction uncertainty (conformal prediction) drives a second pass that rescans only uncertain regions; turns survey→rank→zoom into an uncertainty-budgeted loop, cutting total light dose up to 16×.
- [[Papers/Bouchard 2023]] — TA-GAN: a GAN co-trained on an auxiliary task (segmentation, localisation) predicts a synthetic STED image from a confocal scout, and the divergence of that prediction between time points triggers a real STED acquisition only at changed ROIs. Survey→rank→zoom where the score is "model thinks the nanostructure changed" rather than a hand-defined biosensor; complementary to Ye 2025 (model-disagreement trigger vs calibrated uncertainty rescan).
- [[Papers/Kandel 2023]] — FAST: active-learning scan-point selection via Expected Reduction in Distortion, interleaved with route-optimised batch scheduling; generalises the "where to look next" question from magnification switching to sparse sampling on a flat grid, with no prior training on the sample class.
- [[Papers/Tosi 2021]] — AutoScanJ: ImageJ macro framework for unattended phenotype-triggered re-acquisition; survey→rank→zoom plumbing on top of µManager rather than a custom backend. Substrate sibling of Micropilot — same paradigm, different stack.
- [[Papers/Daetwyler 2025]] — dual-modality light-sheet (mSPIM organism overview + ASLM subcellular capture) with a 3D region-tracking algorithm that uses the low-resolution channel to continuously re-target the high-resolution volume *during* the experiment. Combines specimen-tracking ([[Papers/McDole 2018]]) with concurrent multi-modality survey/capture ([[Papers/Shi 2024]]) into one continuous control loop.
- [[Papers/Meirovitch 2026]] — survey→zoom for connectomic single-beam SEM: fast tile pass, content-aware quality predictor flags subareas, slow rescan only on flagged regions. ~7× acceleration on connectomic samples; the modality where the cost knob is dwell time × electron dose rather than photon count.
- [[Papers/Ibrahim 2025]] — modality-switching EDA on a Brillouin microscope (mechanical mapping). Same scout→burst→restore shape as Alvelid 2022, but the second modality is *temporally* expensive (point-scan over a spectrometer, seconds–minutes per FOV) rather than photodose-expensive. Trigger: deep-learning prediction of protein-aggregation onset from a single fluorescence frame at 91% accuracy; Brillouin map fires only when aggregation is imminent. The "cost knob" axis where each modality's *expense* shape (photons / dwell time / sample motion) sets the trigger criterion.

Topic candidates for verification in [[Papers candidates]] — content-aware acquisition (Weigert / CARE).
