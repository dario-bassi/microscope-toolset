# Cytokinesis ring-ingression kinetics

Paired with `src/recipes/cytokinesis_kinetics.py`. Three entry points:

- **`detect_rings`** (per-frame) — threshold + connected-component label on the myosin channel; per-band PCA on the mask pixels gives the principal axis (= ring orientation, since the myosin band is edge-on, perpendicular to the division axis). The projection-spread along that primary axis is the ring DIAMETER (caller halves to radius).
- **`measure_ring_kinetics`** (single-phase live) — runs an MDA burst, applies `detect_rings` per frame, Hungarian-links bands across frames by centroid → per-cell `{frame: radius_px}` traces.
- **`two_phase_kinetics`** (full pipeline) — phase A MDA burst → caller-supplied `setProperty(*phase_b_setup)` → phase B MDA burst → detection + linking + per-cell linear fit per phase. Returns `{rate_a, rate_b, ratio_b_over_a, n_cells_tracked}`.

The shared substrate is `useq.MDAEvent` + `run_events` for acquisition, plus `match_centroids` (Hungarian) for linking. `phase_b_setup` is a `(device, property, value)` tuple — caller-supplied so the same pipeline transfers across modulator types:

| Modulator | `phase_b_setup` | Tested on |
|---|---|---|
| Drug | `("Perfusion", "Label", "Drug")` | ch666 r1 = 10/10 |
| Temperature ramp | `("Temperature", "State", 3)` | ch667 r1 = 10/10 |
| Y-27632 partial inhibition | `("Perfusion", "Label", "Drug_B")` | (queued p=900 transferability) |
| Mixed temp + drug | compose two calls in series | (future) |

## Reading the brief for ratios

The result's `ratio_b_over_a` is the *raw* phase-B-over-phase-A rate ratio. Briefs may ask for it as:

- **Q10**: temperature-ramp scaling. Conventionally `rate_30 / rate_20`, often near 2.0 (Arrhenius). ch667 GT 1.83 (model 2.0; deviation from temperature-swap timing). The recipe returns ratio_b_over_a directly; agent maps to `q10` in the answer.
- **Fold-change**: drug-effect quantification. Conventionally `baseline / treated` ≥ 4, often very large when the drug abolishes ingression. ch666 GT effectively unbounded.

## Anti-label-read floors

`treated_rate_floor` (default 0.001) caps the phase-B rate magnitude away from exactly 0.0 — the brief's anti-label-read flag triggers when an agent reports `treated == 0.0` exactly (a sign of stamping the predicate threshold rather than measuring noisy near-zero residuals). The floor preserves the "below detection limit" semantics without hitting the cap.

## Why myosin-band PCA is preferred over BF dumbbell waist

- Myosin band is edge-on (perpendicular to division axis); PCA-1 of the mask pixels gives the ring diameter directly, no circle-fit fragility on incomplete edges.
- Brightfield dumbbell waist requires fitting a min-width-along-primary-axis projection that needs care to avoid picking up the lobes' minimum. PCA-on-myosin is cleaner.
- When the myosin channel is unavailable (fixed sample, no fluorophore, etc.), fall back to BF + the dumbbell-waist approach (`scratch/solve_655.py` shows the fallback shape).

## Tested on

- ch666 r1 = 10/10 (drug modulation, blebbistatin freezes ingression).
- ch667 r1 = 10/10 (Q10 temperature ramp, 20 → 30°C).

`auto_recipe(brief=...)` routes `Archetype: cytokinesis` and `Archetype: cytokinesis_kinetics` → `two_phase_kinetics` automatically.

## Real-microscope analogue

Same pipeline. Bright contractile-ring fluorescent reporter (myosin-mScarlet, MyoII-GFP, anillin-mCherry) + threshold-based segmentation + PCA on each band gives the diameter directly. The two-phase + setProperty between bursts maps to: open the perfusion valve to deliver drug; turn the temperature controller to a new setpoint; flip the magnetic stir; whatever the live-cell environmental modulator is.

## Tests

`tests/test_cytokinesis_kinetics.py` covers detection, linking, slope fitting, and the two-phase live pipeline via FakeCore + monkeypatched `run_events`. 11 tests.

## Related

- `[[Core/Strategies/Closed-loop state device]]` — single-state-controller variant.
- `[[Core/Strategies/Per-cell measurement from frames]]` — the "extract per-cell metric per frame" shape this recipe is one instantiation of.
- ``Straight et al. 2003, *Science* 299:1743-7 — discovery of blebbistatin as a myosin II inhibitor (DOI: 10.1126/science.1081412). Methods paper, plain-text pointer per knowledge-library scope`` — the actomyosin-inhibitor reference cited in the ch666 GT.
