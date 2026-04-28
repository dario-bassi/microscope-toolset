# Smart Microscope Toolkit — Architecture

> **Navigation guide:** use this file to find where things live and how they connect.

---

## Quick Reference — Where to Find What

| Task | Location |
|------|----------|
| Snap an image, move stage, set objective | `src/self-learn/hardware/core.py` |
| Run a timelapse / Z-stack / multi-position | `src/self-learn/hardware/core.py:run_events()` |
| Detect cells / tissue / neurons | `src/self-learn/detection/` |
| Measure intensity / morphology / kinetics | `src/self-learn/analysis/` |
| High-level protocol (adaptive survey, tracking, optogenetics) | `src/self-learn/workflows/` |
| Channel discovery, pixel size, config group | `src/self-learn/hardware/config.py` |
| Autofocus | `src/self-learn/workflows/autofocus.py` |
| FUCCI cell cycle | `src/self-learn/analysis/cell_cycle.py` |
| Bacterial light trap | `src/self-learn/workflows/bacteria_trap.py` |
| Q10 / temperature growth | `src/self-learn/workflows/temperature_experiment.py` |
| Save overlay / showcase image | `src/self-learn/utils/diagnostics.py`, `src/utils/showcase.py` |
| Per-sample-type step-by-step guide | `src/self-learn/knowledge/playbooks/` |
| Universal microscopy strategies | `src/self-learn/knowledge/strategies/` |
| pymmcore-plus / useq API reference | `src/self-learn/knowledge/pymmcore/` |

---

## Layer Architecture

```
LLM Agent  (reasoning + vision)
      │
  Knowledge  (knowledge/)          ← reasoning, strategies, playbooks
      │
  Workflows  (src/workflows/)      ← acquisition + analysis protocols
      │
  Analysis   (src/analysis/)       ← image feature extraction
  Detection  (src/detection/)      ← segmentation, object detection
      │
  Hardware   (src/hardware/)       ← pymmcore-plus wrappers
      │
  Utils      (src/utils/)          ← diagnostics, logging, reporting
```

**Rule:** layers only call downward. Workflows call analysis/detection/hardware; analysis never calls workflows.

---

## src/self-learn/hardware/ — Microscope Control

The single entry point for all hardware interaction.

| Module | Key functions |
|--------|--------------|
| `core.py` | `snap()`, `move_to()`, `get_position()`, `set_objective()`, `get_pixel_size()`, `set_z()`, `get_z()`, `apply_slm()`, `make_slm_circle()`, `pixel_to_world()`, `world_to_pixel()`, `run_events()` |
| `config.py` | `get_config()`, `refresh_config()` — runtime discovery of channel group, pixel size, objective, SLM, XY/Z devices |
| `quality.py` | SNR estimation, focus variance, brightness assessment |
| `zstack.py` | Z-stack acquisition, best-focus plane detection |
| `drift.py` | Drift detection/correction via phase correlation and centroid tracking |
| `validate.py` | Preflight checks — channels, objectives, SLM, stage availability |
| `slm_calibration.py` | SLM/DMD ↔ camera affine calibration: `find_slm_conjugate_z()`, `calibrate_slm()`, `camera_mask_to_slm()`, `save_calibration()`, `load_calibration()` |

---

## src/self-learn/detection/ — Object Detection & Segmentation

| Module | What it detects |
|--------|----------------|
| `cells.py` | General cell detection (brightfield threshold, blob detection) |
| `tissue.py` | Tissue/membrane segmentation via Otsu; contact graph analysis |
| `neurons.py` | Soma detection, puncta counting, neurite analysis from immunostaining |
| `segmentation.py` | Generic strategies: watershed, adaptive threshold, marker-controlled |
| `threshold.py` | Adaptive threshold selection with quality-checked retry |
| `consensus.py` | Multi-method consensus for robust counting |

---

## src/self-learn/analysis/ — Feature Extraction

### Cell Biology & Physiology
| Module | What it measures |
|--------|-----------------|
| `cell_cycle.py` | FUCCI phase classification (G1/S/G2/M from gem + cdt1 channels) |
| `calcium.py` | ROI trace extraction, ΔF/F, transient detection |
| `apoptosis.py` | Apoptotic cell detection from blebbing / fragmentation |
| `viability.py` | Cell viability from live/dead dual-stain or morphology |
| `confluency.py` | Confluency measurement and growth curve fitting |
| `nuclear_cytoplasmic.py` | Nuclear-cytoplasmic ratio for translocation assays |
| `chromatin.py` | Chromatin / nuclear architecture, genomic stress state |
| `electrophysiology.py` | Action potentials, firing rates from electrode signals |

### Morphology & Structure
| Module | What it measures |
|--------|-----------------|
| `morphometry.py` | Cell size, shape, roundness, aspect ratio |
| `morphological_dynamics.py` | Shape change and morphological transitions over time |
| `sphere_3d.py` | 3D spheroid / organoid morphometry |
| `drug_penetration.py` | Drug penetration depth in tumor spheroids (3-zone model) |
| `contour.py` | Contour and shape analysis with Fourier descriptors |
| `ring.py` | Ring / annular structure analysis (organoids, ZOI) |
| `network.py` | Branching / tubular / vascular / dendritic structure analysis |
| `gradient.py` | Radial gradient and edge detection for ZOI / spheroid boundaries |
| `histology.py` | H&E nuclear morphometry, necrosis detection |
| `cytoskeleton.py` | Actin / stress fiber analysis (VSF_count, DSF_count) |
| `lipid_droplet.py` | Lipid droplet segmentation and size distribution |
| `condensate.py` | Phase condensate (biomolecular condensate) analysis |

### Motion & Dynamics
| Module | What it measures |
|--------|-----------------|
| `tracking.py` | Cell tracking: Hungarian matching, trajectory analysis |
| `motion.py` | Timelapse motion: kymographs, optical flow, contraction |
| `flow.py` | Blood flow / velocity estimation via optical flow |
| `run_tumble.py` | Run-and-tumble motility analysis for bacteria / microswimmers |
| `chemotaxis.py` | Directed migration and chemotaxis from trajectories |
| `spt.py` | Single-particle tracking |
| `diffusion.py` | MSD analysis and diffusion coefficient estimation |
| `aggregation.py` | Collective migration / aggregation analysis |

### Signals & Spectral
| Module | What it measures |
|--------|-----------------|
| `intensity.py` | Multi-class intensity classification (gap, k-means, rank-based) |
| `fluorescence.py` | Quantitative fluorescence, background subtraction, photobleaching |
| `colocalization.py` | Pearson / Manders colocalization between channels |
| `spectral.py` | Spectral unmixing and bleedthrough correction |
| `frap.py` | FRAP recovery curves and D_eff estimation |
| `photoconversion.py` | Photoconversion / photoactivation tracking |
| `optical_mapping.py` | Wave propagation in cardiac / neural tissue |
| `functional_connectivity.py` | Functional connectivity from calcium imaging traces |
| `puncta.py` | Organelle / vesicle puncta detection, per-cell quantification |

### Temporal & Statistical
| Module | What it measures |
|--------|-----------------|
| `kinetics.py` | Growth rate regression, Michaelis-Menten, Q10 fitting, doubling time |
| `temporal.py` | FFT frequency analysis, periodic event detection |
| `events.py` | Division, arrival/departure, morphology transition detection |
| `temporal_constraints.py` | Enforce monotonicity and continuity on time-series |
| `phase_segmentation.py` | Auto-segment timeseries into baseline/ramp/plateau/decline phases |
| `treatment.py` | Baseline-vs-treatment response quantification |
| `statistics.py` | Statistical testing and confidence intervals |
| `confidence.py` | Measurement confidence / uncertainty from replicates |

### Tissue & Population
| Module | What it measures |
|--------|-----------------|
| `blood_smear.py` | WBC morphology and differential cell counting |
| `colony.py` | Colony formation / clonogenic assay analysis |
| `size_distribution.py` | Particle/cell size distribution with histogram fitting |
| `wound_healing.py` | Scratch assay closure kinetics |
| `dose_response.py` *(analysis)* | Hill curve fitting (see also `workflows/dose_response.py`) |
| `plate_reader.py` | Plate reader data processing, background correction |
| `reaction_diffusion.py` | Pattern classification: waves, spirals, spots, stripes |

### Image & Quality
| Module | What it measures |
|--------|-----------------|
| `image_quality.py` | Focus, noise, saturation evaluation |
| `quality.py` | Detection quality / sanity checks vs physical expectations |
| `registration.py` | Timelapse stabilization via phase-correlation image registration |
| `texture.py` | GLCM and local binary pattern texture analysis |
| `color_analysis.py` | H&E stain separation via `rgb2hed()`, colour deconvolution |
| `profile.py` | Line and radial intensity profile analysis |
| `spatial.py` | Spatial statistics: nearest-neighbour, Ripley's K, Clark-Evans |
| `orientation.py` | Circular statistics for axial / directional data |
| `zstack.py` | Z-stack 3D projections, slicing, volumetric measurements |

### High-Level / Meta
| Module | What it does |
|--------|-------------|
| `sample_characterizer.py` | Auto-determine sample type, staining, and appropriate workflow |
| `measurement_validator.py` | Validate measurements against biological priors |
| `classifier.py` | Morphological cell classifier (k-means or rule-based) |

---

## src/self-learn/workflows/ — Acquisition & Analysis Protocols

All adaptive workflows return `(generator_factory, on_frame_callback, shared_state_dict)` so they compose cleanly with `run_events()`.

| Module | What it does |
|--------|-------------|
| `adaptive.py` | Survey→cluster→zoom→measure pipeline (`adaptive_survey_mda`) |
| `scanning.py` | Multi-position grid scanning with deduplication (`scan_and_detect_mda`) |
| `autofocus.py` | Software autofocus: 4 metrics, coarse+fine sweep (`autofocus_mda`) |
| `mda.py` | `execute_mda()` — signal-based native pymmcore-plus MDA execution |
| `stage_tracking.py` | Closed-loop target tracking with motion prediction |
| `bacteria_trap.py` | SLM bacterial light trap: baseline → SLM on → accumulation → measurement |
| `temperature_experiment.py` | Q10 growth rate across temperatures: `temperature_response_curve_v2()` |
| `optogenetics.py` | SLM-targeted optogenetic stimulation protocols |
| `dose_response.py` | Multi-dose timelapse: plate layout, measurement, Hill curve fitting |
| `tiling.py` | Multi-position stitching with phase-correlation alignment |
| `scouting.py` | Low-mag sample survey and experiment planning |
| `organoid.py` | Organoid Z-navigation and equatorial plane selection |
| `two_pass_scan.py` | Two-pass scan: low-mag survey then high-mag targeting |
| `multi_scale.py` | Multi-magnification morphometry coordinator |
| `phototaxis_steering.py` | Real-time organism steering via SLM phototaxis |
| `batch.py` | Batch processing over multiple positions / tiles |
| `experiment.py` | Multi-phase experiment builder (baseline-treatment-recovery) |
| `optimization.py` | Imaging parameter optimisation: exposure, gain, SNR |

---

## src/self-learn/utils/ — Utilities

| Module | What it does |
|--------|-------------|
| `diagnostics.py` | Save snapshots / overlays for challenge submissions |
| `experiment_log.py` | Structured JSON experiment logging for reproducibility |
| `image.py` | Grayscale conversion, normalisation, basic preprocessing |
| `report.py` | Experiment report generator with measurement statistics |
| `showcase.py` | Multi-panel publication-style showcase figures |

---

## knowledge/ — Reasoning Layer

```
knowledge/
  strategies/    (8 files) — universal principles (read BEFORE every challenge)
  playbooks/    (37+ files) — per-sample-type step-by-step guides
  pymmcore/      (4 files) — pymmcore-plus / useq API reference
  patterns/      (7 files) — acquisition / detection / calibration decision trees
  workflows/     (8 files) — category-specific workflow patterns
  failures/      (8 files) — post-mortem lessons (read to avoid repeating)
  prompts/       (2 files) — LLM vision classification prompts
```

**Pre-challenge:** identify sample type → read `playbooks/<type>.md` → check `strategies/08_pre_submission_checklist.md`.

---

## Design Principles

1. **Functions take `core` as first arg, return dicts** — composable, no hidden state
2. **World coordinates everywhere** — `pixel_to_world()` / `world_to_pixel()` in `hardware/core.py`
3. **MDA for all multi-frame acquisitions** — `MDASequence` for fixed, generators for adaptive
4. **Stateless modules** — no module-level mutable state; share state via dicts passed to generators
5. **Layers only call downward** — workflows → analysis/detection → hardware; never the reverse
6. **Build on pymmcore-plus** — use its primitives, don't reimplement what the framework provides

---

## pymmcore-plus Integration

### Execution

**`run_events(core, events, on_frame=)`** is the single execution entry point for all multi-frame acquisitions. It delegates to `core.mda.run()`, which works identically for local `CMMCorePlus` and remote `pymmcore-proxy` (events are streamed over WebSocket; `frameReady` signals are forwarded back). `execute_mda()` is just an alias.

Both accept any iterable of `MDAEvent` (generator, list, `MDASequence`, `Queue`-backed iterator).

### Acquisition patterns

```python
from useq import MDASequence, MDAEvent
from src.self_learn.hardware.core import snap, run_events

# --- Single frame ---
img = snap(core, channel='GFP')

# --- Fixed timelapse ---
seq = MDASequence(
    time_plan={"loops": 20, "interval": 2.0},
    channels=[{"config": "GFP", "group": "Fake"}],
)
results = run_events(core, list(seq))

# --- Fixed Z-stack ---
seq = MDASequence(
    z_plan={"range": 10.0, "step": 0.5},
    channels=[{"config": "GFP", "group": "Fake"}],
)
results = run_events(core, list(seq))

# --- Multi-position timelapse ---
seq = MDASequence(
    time_plan={"loops": 10, "interval": 1.0},
    stage_positions=[{"x": 100, "y": 200}, {"x": 300, "y": 400}],
    channels=[{"config": "BF", "group": "Fake"}],
    axis_order="tpc",
)
results = run_events(core, list(seq))

# --- Adaptive / generator-based ---
shared = {"stop": False}

def on_frame(img, event):
    if analyze(img)["done"]:
        shared["stop"] = True

def my_generator():
    for i in range(100):
        if shared["stop"]:
            return
        yield MDAEvent(channel={"config": "GFP", "group": "Fake"})

results = run_events(core, my_generator(), on_frame=on_frame)

# --- Adaptive workflow (standard pattern) ---
gen, on_frame, state = adaptive_survey_mda(survey_positions=[...])
results = run_events(core, gen(), on_frame=on_frame)
# Access results via state dict
```

---

## Common Import Patterns

```python
# Hardware
from src.self_learn.hardware.core import snap, move_to, set_objective, run_events, pixel_to_world
from src.self_learn.hardware.config import get_config
from src.self_learn.hardware.autofocus import autofocus_mda  # actually in workflows/

# Detection
from src.self_learn.detection.cells import detect_cells
from src.self_learn.detection.neurons import detect_foci         # returns list of dicts: {cy, cx, sigma, area}
from src.self_learn.detection.tissue import segment_tissue

# Analysis
from src.self_learn.analysis.kinetics import measure_growth_rate_series, fit_q10_with_ci
from src.self_learn.analysis.tracking import track_cells
from src.self_learn.analysis.cell_cycle import classify_fucci_phase
from src.self_learn.analysis.color_analysis import rgb2hed       # H&E stain separation
from src.self_learn.analysis.morphometry import measure_morphometry
from src.self_learn.analysis.intensity import classify_intensity_multiclass
from src.self_learn.analysis.flow import estimate_flow_velocity

# Workflows
from src.self_learn.workflows.adaptive import adaptive_survey_mda
from src.self_learn.workflows.scanning import scan_and_detect_mda
from src.self_learn.workflows.autofocus import autofocus_mda
from src.self_learn.workflows.bacteria_trap import run_bacteria_trap_mda, measure_bacteria_intensity
from src.self_learn.workflows.temperature_experiment import temperature_response_curve_v2

# Utils
from src.self_learn.utils.diagnostics import save_snapshot
from src.self_learn.utils.showcase import make_showcase_figure
```

---

## Key API Gotchas

- `detect_foci()` returns **list of dicts** `{'cy','cx','sigma','area',...}` — not tuples
- `analyze_cell_fibers()` → `subtypes` is a **list** of labels, not a dict
- `fiber_differential()` → keys are `VSF_count`, `DSF_count`
- `rgb2hed()` for H&E separation — **not** `color_deconvolution()`
- `set_objective(core, 40)` — integer argument, not string `"40x"`
- `event.properties` for per-event device property changes: `[('Camera', 'Gain', '4')]` — native MDAEvent field, handled by the MDA engine

---