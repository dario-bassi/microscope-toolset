---
title: Controller-driven multiplexing in DNA-PAINT — fluidic exchange (PRIME-PAINT) and quencher exchange (Quencher-Exchange-PAINT)
authors:
  - Matthew J. Rames, John P. Kenison, Daniel Heineck, Fehmi Civitci, Malwina Szczepaniak, Ting Zheng, Julia Shangguan, Yujia Zhang, Kai Tao, Sadik Esener, Xiaolin Nan (PRIME-PAINT)
  - Tobias Lutz, Alexander H. Clowsley, Ruisheng Lin, Stefano Pagliara, Lorenzo Di Michele, Christian Soeller (Quencher-Exchange-PAINT)
year: 2023, 2018
venue:
  - Chemical & Biomedical Imaging 1(9):817–830 (Rames 2023)
  - Nano Research 11(12):6141–6154 (Lutz 2018)
doi:
  - 10.1021/cbmi.3c00060
  - 10.1007/s12274-018-1971-6
url:
  - https://pubs.acs.org/doi/10.1021/cbmi.3c00060
  - https://link.springer.com/article/10.1007/s12274-018-1971-6
researched: 2026-04-25
---

## Abstracts

**Rames et al. 2023 (PRIME-PAINT).** "PRism-Illumination and Microfluidics-Enhanced DNA-PAINT": prism-type oblique illumination delivers single FOVs up to ∼520 µm × 520 µm with 25–40 nm lateral resolution; an on-stage microfluidic chamber driven by an Arduino-controlled rotary valve + peristaltic pump performs Exchange-PAINT probe swaps on the same FOVs. Three-target acquisition over a 0.3 mm × 0.3 mm FOV completes in ∼45 min (30,000 frames × 30 ms × 3 imagers); a mild 1 µL/min flow during acquisition increases observed localizations ~10× in tissue by accelerating imager turnover. Demonstrated on ~630,000 caveolae across ~925 cells and on entire pancreatic-cancer biopsy lesions.

**Lutz et al. 2018 (Quencher-Exchange-PAINT).** Conventional Exchange-PAINT switches imagers by diffusional washout, which is slow in thick samples and demands flow-cell hardware. Quencher-Exchange-PAINT replaces the wash with a *chemical* off-switch: quencher strands complementary to the current imager are pipetted into an open-top chamber, sequester free imagers in solution, and quench the residual fluorescence (~98 % at the dye level). Switching between targets becomes "add quencher → add next imager", an order of magnitude faster than diffusional washout, and works in tissue slices where flow is impractical.

## Smart microscopy principle

These two papers attack the same bottleneck — Exchange-PAINT needs the previous imager removed before the next one fires — from opposite ends. **Rames 2023** does it with controller-driven fluidics: the microscope's MDA-equivalent loop now also drives a valve and a pump, so probe exchange is a first-class acquisition event sequenced into the protocol alongside stage moves and exposures. **Lutz 2018** does it with chemistry: a competing oligonucleotide bearing a dye quencher is added into an open chamber, eliminating the need for flow at all — at the cost of ceding multiplexed-acquisition state from the controller back to the operator.

For the smart-microscopy thesis, Rames 2023 is the canonical instance: Exchange-PAINT becomes a programmable acquisition protocol of the form `for round r in probes: pump(r); flow(1 µL/min); acquire(30 000 frames); wash(buffer)`. The microscope's control surface treats fluid identity as a first-class axis on equal footing with the channel and time axes — exactly the paradigm [[Papers/Almada 2019]] established for the more general live-to-fixed and STORM/DNA-PAINT case. Rames specialises it for prism-illuminated wide-FOV nanoscopy and pins the hardware down to an Arduino-driven valve + peristaltic pump.

Lutz 2018 sits alongside as the chemistry alternative for cases where flow is unworkable (thick tissue, small open chambers, no fluidics rig). It is **not** an integrated-controller paper — quencher addition is manual pipetting — but it is the chemistry that would make a controller-driven version dramatically simpler: a pump need only inject ~µL of quencher solution rather than perform a multi-µL diffusional wash. Both papers extend the same axis (probe identity per round); both are siblings of the strategy [[Core/Strategies/Multichannel scan]] generalises beyond spectrally separated dyes.

## Implementation on pymmcore-plus

The Almada 2019 note already gives the canonical Exchange-PAINT-style template (rotary valve as a Micro-Manager `StateDevice`, fluid steps as `MDAEvent` property transitions, settle delays). PRIME-PAINT specialises that template in two ways worth recording:

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

# PRIME-PAINT-style multiplexed Exchange-PAINT:
# - sequential probe rounds (Exchange-PAINT)
# - mild continuous flow DURING the SMLM acquisition (Rames 2023 finding:
#   1 µL/min increases localizations ~10× in tissue)
# - quick wash between rounds.

PROBES = ["P1", "P2", "P3"]
FRAMES_PER_ROUND = 30_000
EXPOSURE_MS = 30
WASH_S = 60.0
SETTLE_S = 30.0

def prime_paint_rounds(probes):
    t = 0.0
    for r, probe in enumerate(probes):
        # 1. Pump the probe into the FOV.
        yield MDAEvent(
            min_start_time=t,
            properties=[("Fluidics", "Label", probe),
                        ("Fluidics", "FlowRate_uL_per_min", "0")],
            metadata={"phase": "load", "round": r, "probe": probe},
        )
        t += SETTLE_S
        # 2. Switch to mild continuous flow and acquire all SMLM frames.
        yield MDAEvent(
            min_start_time=t,
            properties=[("Fluidics", "FlowRate_uL_per_min", "1")],
            metadata={"phase": "mild_flow_on", "round": r},
        )
        for f in range(FRAMES_PER_ROUND):
            yield MDAEvent(
                min_start_time=t + f * EXPOSURE_MS / 1000.0,
                channel={"config": "Cy5"},
                exposure=EXPOSURE_MS,
                index={"r": r, "t": f},
                metadata={"phase": "image", "round": r},
            )
        t += FRAMES_PER_ROUND * EXPOSURE_MS / 1000.0
        # 3. Wash and stop flow before next round.
        yield MDAEvent(
            min_start_time=t,
            properties=[("Fluidics", "Label", "wash"),
                        ("Fluidics", "FlowRate_uL_per_min", "5")],
            metadata={"phase": "wash", "round": r},
        )
        t += WASH_S

run_events(core, list(prime_paint_rounds(PROBES)))
```

Specialisations vs. the Almada template:

- **Mild flow during acquisition**, not just at exchange. PRIME-PAINT's 10× boost requires the pump to run *while* SMLM frames stream — modelled here as a separate property (`FlowRate_uL_per_min`) toggled inside the MDA. On a real Micro-Manager rig, a pump's flow-rate property is an ordinary float property; on a custom Arduino rig, expose it via a thin device adapter.
- **Wide-FOV illumination matters.** PRIME-PAINT's 520 µm FOV multiplies the number of localizations per round; this is an illumination-geometry choice (prism TIRF) that is decoupled from the fluidics plumbing — the MDA looks the same regardless. Worth flagging in protocol metadata so downstream analysis knows the per-round count budget.

For Quencher-Exchange-PAINT, the controller-driven version Lutz 2018 *enables but does not implement* would replace the multi-µL wash with a small-volume quencher injection:

```python
# Quencher-Exchange-PAINT in a controller-driven layout. Note: Lutz 2018
# performed this by manual pipetting; this is the natural integration if
# a syringe pump and a valve are wired in.
yield MDAEvent(
    min_start_time=t,
    properties=[("Fluidics", "Label", f"quencher_{probe}"),
                ("Fluidics", "Volume_uL", "10")],   # tiny vs. wash
    metadata={"phase": "quench", "round": r},
)
```

Hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events` already drives `properties=[(group, value)]` transitions and time-ordered events; the templates above need no new core code.
- `src/core/workflows/fluidics.py` — does not yet exist; the Almada note already flagged this as future work, and PRIME-PAINT motivates extending it with a `mild_flow_during(rate, frames)` primitive that brackets a frame block with on/off flow events.
- Calibration discipline: PRIME-PAINT's 1 µL/min, ~30 s settle, ~60 s wash are protocol-specific to its flow-cell geometry. Library defaults must come from per-rig calibration runs, not from these papers.

## Cross-references

- [[Papers/Almada 2019]] — canonical "fluidics as first-class acquisition events" paper; PRIME-PAINT is the prism-TIRF-specific instance, Quencher-Exchange-PAINT is the chemistry alternative that would simplify any fluidics rig built on Almada's pattern.
- [[Core/Strategies/Multichannel scan]] — Exchange-PAINT and Quencher-Exchange-PAINT both extend the channel axis via probe rounds rather than via spectrally separated dyes; the same scan strategy applies, with rounds replacing channels.
- [[Core/Strategies/Closed-loop state device]] — fluidic valve / pump exposed as a Micro-Manager state device follows the same closed-loop pattern as any other state-driven hardware (objective turret, filter wheel).

## Cited by

(none yet)
