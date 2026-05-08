# Adaptive STED — scout → burst → restore

**Assumes:** a microscope with a `Modality` state device whose labels include some subset of `widefield_scout` / `confocal` / `sted_burst` / `lattice_lightsheet`. Each modality carries different PSF, exposure ceiling, **bleach-rate multiplier**, and **channel allow-list**. Paired with `src/recipes/adaptive_sted_burst.py` (`scout_then_burst_pipeline` — tested on ch605 10/10 r1).

## Sample type

Any sample where (a) most of the field at most of the time is uninformative, (b) brief subdiffraction snapshots at event onset are biologically valuable, and (c) STED bleach rate makes per-snap STED across the whole timelapse impossible. Calcium-wave fields, single-vesicle release, and synaptic-protein clustering events are the canonical targets — same archetype as Alvelid 2022 (Smart STED) where a widefield biosensor scout gates STED nanoscopy bursts.

## Bleach-rate calibration (sim default)

| Modality            | Bleach × | Use                                |
|---------------------|----------|-------------------------------------|
| widefield_scout     | 0.3      | Default — survey state              |
| confocal            | 1.0      | Mid-cost intermediate / verification|
| sted_burst          | 8.0      | Subdiffraction; fire on event only  |
| lattice_lightsheet  | 0.5      | Cheaper alternative for thick samples |

A 50-ms snap costs `0.05 × 8.0 = 0.40` dose units in STED but `0.05 × 0.3 = 0.015` in widefield. With a 2.0 dose budget, that's ~5 STED snaps vs ~133 widefield snaps. Stay in scout, burst on event.

## Workflow

```python
from src.recipes.adaptive_sted_burst import (
    BleachState, Modality, detect_event_in_scout, track_bleach,
    scout_then_burst_pipeline,
)

state = BleachState(budget=2.0)
result = scout_then_burst_pipeline(
    core, state,
    n_scout_frames=30,
    scout_channel="GCaMP",
    burst_channel="GCaMP",       # respects channel allow-list
    detection_threshold_delta=30,
)
```

Per-snap bookkeeping:

- `track_bleach(state, modality, exposure_ms)` updates cumulative dose, raises `BudgetExceededError` on overrun.
- `detect_event_in_scout(img, baseline=None, threshold_delta=30)` returns `{"detected": bool, "peak_xy": (x, y) | None, "baseline": float}` — single-frame peak threshold.

## Critical lessons

1. **Default to scout, burst on event.** "STED every snap" exhausts a 2.0 budget in ~5 snaps. Fire one STED snap on event, switch back.
2. **One burst per event.** Re-firing on the same blob without re-detecting drains the budget instantly.
3. **Verify the channel allow-list with a forbidden-channel snap.** If `sted_burst.channels_allowed = ["GCaMP"]`, snapping E-cadherin under STED should return zeros. Two diagnostic snaps confirm the channel-routing wiring is honest before the burst loop starts.
4. **Track cumulative bleach dose honestly.** `cum += exposure_s × bleach_multiplier` per snap. Exceeding the bridge's `bleach_dose` budget raises on the next snap; the recipe must self-track to leave headroom.
5. **Distinct from `event_driven_modality_switch`.** That recipe switches **magnification** via the `Objective` state device (low-mag scout → high-mag burst); this recipe switches **optical mode** via `Modality` (widefield → STED). Both share the scout→burst→restore shape.

## See also

- [[Core/Strategies/Adaptive acquisition]] — survey→zoom is the parent pattern; modality switch is the optical-axis variant.
- [[Core/Strategies/Gentle imaging]] — Principle 7 Active Dose Budgeting motivates the `BleachState` accounting.
- [[Papers/Alvelid 2022]] — the literature precedent (widefield biosensor scout → localised STED).
- [[Papers/Shi 2024]] — sister archetype on the LLSM axis.
