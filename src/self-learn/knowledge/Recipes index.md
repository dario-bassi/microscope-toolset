# `Recipes/` — sample-tuned playbooks

The sim-specific tier. Sample/scale-tuned observations, simulator backend quirks called out honestly, concrete parameter values that make the sim produce a clean measurement. Many recipes pair with a `src/recipes/*.py` module of the same name; link it at the top of the recipe when present.

A recipe reads naturally to someone who has the simulator in front of them. If a note in here reads as universal microscopy, it's probably drifted up and should move to `../Core/`.

## Cell biology — general

- [[Recipes/Brightfield cells]] — counting and outlining cells from BF at 10x/20x.
- [[Recipes/Fluorescence]] — threshold-and-blob baseline for most fluorescent images.
- [[Recipes/Timelapse]] — division detection, tracking, count-over-time.
- [[Recipes/Drug response timelapse]] — per-cell growth-factor ratio (robust to BF bias).
- [[Recipes/Perturbation kinetics]] — onset, half-time, amplitude from a treated timelapse.
- [[Recipes/Dose response]] — Hill fit on per-well population responses.
- [[Recipes/Colocalization]] — Pearson / Manders / gap-separation on per-cell intensities.
- [[Recipes/Stress granule formation]] — puncta appearance kinetics.
- [[Recipes/Mitochondrial dynamics]] — fission/fusion morphology.
- [[Recipes/FUCCI]] — cell-cycle phase via red/green ratio.
- [[Recipes/FRAP]] — mobile fraction + recovery rate from ROI photobleaching.
- [[Recipes/Photoconversion]] — lineage tracing from irreversible mark.
- [[Recipes/SPT diffusion]] — single-particle tracking + MSD fit.
- [[Recipes/Lipid droplet]] — LoG detection, steatosis index, adipogenesis score.
- [[Recipes/Spatial distribution]] — Delaunay / Voronoi / Ripley K density metrics.

## Tissue / multi-cellular

- [[Recipes/Histology]] — H&E nuclear morphometry in µm²; vessel detection.
- [[Recipes/Wound healing]] — front detection + closure-rate measurement.
- [[Recipes/Tissue triage]] — classify tissue regions by quality before acquisition.
- [[Recipes/Organoid]] — Z-stack multiscale morphometry of a 3D culture.
- [[Recipes/Spheroid]] — drug penetration gradient from periphery to core.
- [[Recipes/Z-stack 3D|zstack_3d]] — 3D segmentation and volume measurement.

## Clinical / diagnostic

- [[Recipes/Blood smear WBC]] — differential count via solidity/lobes.
- [[Recipes/Hemocytometer]] — grid-square counting with dilution.
- [[Recipes/Flow cytometry]] — FSC/SSC gating on single-cell events.
- [[Recipes/Malaria]] — parasitaemia from thin-smear infected-RBC fraction.
- [[Recipes/Disk diffusion]] — zone-of-inhibition measurement for antibiotics.

## Specialist model systems

- [[Recipes/Yeast division]], [[Recipes/Yeast bud scars]] — budding, lineage age.
- [[Recipes/C elegans]] — body-length morphometry, locomotion tracking.
- [[Recipes/Zebrafish]] — embryo imaging, whole-animal movement.
- [[Recipes/Bacteria growth]], [[Recipes/Bacterial light trap]] — colony kinetics, phototactic trapping.
- [[Recipes/Neurons]] — somata + neurite detection + puncta per cell.
- [[Recipes/Fibroblasts]] — F-actin stress fibers + focal adhesion puncta from phalloidin.
- [[Recipes/Dictyostelium]] — collective aggregation dynamics.

## Acquisition-pattern recipes

- [[Recipes/MDA workflows]] — MDA patterns that map to common experimental designs.
- [[Recipes/Multi-position expression scan]] — per-well expression mapping.
- [[Recipes/Microfluidic]] — trap loading, perfusion switching.
- [[Recipes/Calcium imaging]] — GCaMP ΔF/F and wave detection.
- [[Recipes/Motile organism tracking]] — tracking parameters for swimming organisms.
- [[Recipes/Galvanotaxis]] — directional field-driven migration; paired state device.
- [[Recipes/Optogenetics]] — SLM-driven patterned stimulation (sample-tuned parameters).
- [[Recipes/Adaptive STED]] — scout (widefield) → burst (STED) → restore, with per-snap bleach-budget tracking and channel-allow-list verification.
- [[Recipes/Cybergenetics]] — Rullan-style per-cell integral feedback via SLM; `src/recipes/cybergenetic_per_cell_control.py`.
- [[Recipes/Event-driven modality switch]] — pacemaker-localisation in excitable tissue (σ × mean), then objective swap to high mag for detail capture; `src/recipes/event_driven_modality_switch.py` (ch594 r14 → 10/10).
- [[Recipes/Sensorless AO]] — DM-state argmax + per-axis-presence inference + residual axis for quantised-action sensorless AO; `src/recipes/sensorless_ao.py` (ch607/608/610 → 10/10 ×3).
- [[Recipes/Spectral unmixing]] — N×N leak matrix via spatial-segmentation-as-control + K^-1 unmix; `src/recipes/spectral_unmix.py` + `src/core/utils/spectral_leak.py` (ch614 r1 → 10/10).
- [[Recipes/Temperature kinetics]] — multi-setpoint dwell + Q10 fit (yeast-tuned defaults, runtime-discoverable temp-state map); `src/recipes/temperature_experiment.py`.
- [[Recipes/SIM protocol]] — structured-illumination raw-frame acquisition + 3-phase wide-field demodulation (ratio = 1.5) + fringe-orientation FFT recovery (no device-label trust); `src/recipes/sim_protocol.py` (ch646 + ch649 → 10/10 each).

## See also

- [[Core index]] — the sim-agnostic principles recipes reference.
- [[Core/Strategies index]] — each recipe should cite the strategy it implements.
- `../src/recipes/` — paired production code where it exists.
