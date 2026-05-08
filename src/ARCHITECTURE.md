# Smart Microscope Toolkit — Architecture

> **Navigation guide:** use this file to find where things live and how they connect.

---

## Quick Reference — Where to Find What

| Task | Location |
|------|----------|
| Snap an image, move stage, set objective | `src/core/hardware/core.py` |
| Run a timelapse / Z-stack / multi-position | `src/core/hardware/core.py:run_events()` |
| Detect cells / tissue / neurons | `src/core/detection/` |
| Measure intensity / morphology / kinetics | `src/core/analysis/` |
| High-level protocol (adaptive survey, tracking, optogenetics) | `src/core/workflows/` |
| Channel discovery, pixel size, config group | `src/core/hardware/config.py` |
| Autofocus | `src/core/workflows/autofocus.py` |
| FUCCI cell cycle | `src/core/analysis/cell_cycle.py` |
| Bacterial light trap | `src/recipes/bacteria_trap.py` |
| Q10 / temperature growth | `src/recipes/temperature_experiment.py` |
| Save overlay / showcase image | `src/core/utils/diagnostics.py`, `src/core/utils/showcase.py` |
| Per-sample-type step-by-step guide | `knowledge/recipes/` |
| Universal microscopy strategies | `knowledge/core/approach/`, `knowledge/core/strategies/` |
| pymmcore-plus / useq API reference | `knowledge/core/concepts/` |
| Generalizable pitfalls and lessons | `knowledge/core/pitfalls/` |
| Verified paper citations | `knowledge/papers/` |
| Reusable skill runbooks | `skills/` |

---

## Layer Architecture

```
LLM Agent  (reasoning + vision)
      │
  Knowledge  (knowledge/)          ← reasoning, strategies, playbooks
      │
  Workflows  (src/core/workflows/)      ← acquisition + analysis protocols
      │
  Analysis   (src/core/analysis/)       ← image feature extraction
  Detection  (src/core/detection/)      ← segmentation, object detection
      │
  Hardware   (src/core/hardware/)       ← pymmcore-plus wrappers
      │
  Utils      (src/core/utils/)          ← diagnostics, logging, reporting
```

**Rule:** layers only call downward. Workflows call analysis/detection/hardware; analysis never calls workflows.

---

## src/core/hardware/ — Microscope Control

The single entry point for all hardware interaction.

| Module | Key functions |
|--------|--------------|
| `core.py` | `snap()`, `move_to()`, `get_position()`, `set_objective()`, `get_pixel_size()`, `set_z()`, `get_z()`, `apply_slm()`, `make_slm_circle()`, `pixel_to_world()`, `world_to_pixel()`, `run_events()` |
| `config.py` | `get_config()`, `refresh_config()`, `resolve_channel_group(core, group)` — runtime discovery of channel group, pixel size, objective, SLM, XY/Z devices. Pass `group=None` anywhere in core/ to auto-discover. |
| `quality.py` | SNR estimation, focus variance, brightness assessment |
| `zstack.py` | Z-stack acquisition, best-focus plane detection |
| `drift.py` | Drift detection/correction via phase correlation and centroid tracking |
| `validate.py` | Preflight checks — channels, objectives, SLM, stage availability |

---

## src/core/detection/ — Object Detection & Segmentation

| Module | What it detects |
|--------|----------------|
| `cells.py` | General cell detection (brightfield threshold, blob detection) |
| `tissue.py` | Tissue/membrane segmentation via Otsu; contact graph analysis |
| `neurons.py` | Soma detection, puncta counting, neurite analysis from immunostaining |
| `segmentation.py` | Generic strategies: watershed, adaptive threshold, marker-controlled, `expand_labels_voronoi()` for nuclear-seed → cell-territory tessellation |
| `threshold.py` | Adaptive threshold selection with quality-checked retry |
| `consensus.py` | Multi-method consensus for robust counting |

---

## src/core/analysis/ — Feature Extraction

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
| `histology.py` | H&E deconvolution, nuclear morphometry, necrosis detection, `detect_vessels()` (RBC-cluster + optional nuclear-ring filter) |
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

## src/core/workflows/ — Acquisition & Analysis Protocols

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
| `batch.py` | Batch processing over multiple positions / tiles. `multichannel_scan()` runs a single MDASequence over N positions × M channels and buckets frames per-position — the MDA-first replacement for for-pos/for-ch snap loops. |
| `experiment.py` | Multi-phase experiment builder (baseline-treatment-recovery) |
| `optimization.py` | Imaging parameter optimisation: exposure, gain, SNR |

---

## src/core/utils/ — Utilities

**Cross-cutting helpers** (always-on, used everywhere):
| Module | What it does |
|--------|-------------|
| `diagnostics.py` | Save snapshots / overlays for challenge submissions |
| `experiment_log.py` | Structured JSON experiment logging for reproducibility |
| `image.py` | Grayscale conversion, normalisation, basic preprocessing |
| `report.py` | Experiment report generator with measurement statistics |
| `showcase.py` | Multi-panel publication-style showcase figures (matplotlib `Panel` + `make_showcase` registry) |
| `validation.py` | Pre-submission answer validation (range checks, JSON-safe types) |
| `preflight.py` | Code-callable pre-submission preflight checks |
| `sample_classifier.py` | Cheap-feature image → sample-class label (sprint #17) |
| `auto_recipe.py` | Image + core → `RecipeSuggestion` dispatcher (sprint #18) |
| `mda_diagnostics.py` | `run_events_checked()` — MDA truncation guard |
| `submit.py` | `submit_with_showcase()` + `count_panels()` |
| `presubmit.py` | 3-tier guard (preflight → render-vs-submit → review prep) |
| `session_audit.py` | Per-recipe / per-utils / per-core-module score-distribution mining |

**Image-only measurement primitives** (caller-opted-in, scenario-level signals):
| Module | What it does |
|--------|-------------|
| `sensorless_ao.py` | DM-state sweep + argmax + per-axis-presence inference (sprint #25; ch607/608/610/650) |
| `slm_masks.py` | circle / gaussian / ring SLM mask builders, vectorised over centre count (sprint #28) |
| `spectral_leak.py` | top-percentile mask + N×N leak matrix + `unmix(K, observed)` (sprint #31; ch614) |
| `fft_peak.py` | `FFTPeak` dataclass + `find_fft_peak` + `find_top_n_fft_peaks` (sprint #32; ch615/646/649) |
| `fluorophore_brightness.py` | `BrightnessRanking` + predict + measure + compare (sprint #33; ch617) |
| `rate_limited_drive.py` | `predict_n_steps` closed-form + `drive_to_band` callback loop (sprint #35; ch621) |
| `pulsed_schedule.py` | `predict_segment_end` + `plan_trajectory` for multi-waypoint MPC (sprint #37; ch624) |
| `wave_period.py` | `estimate_wave_period(core)` 10-frame FFT preview → recommended `n_burst` (sprint #14) |
| `firing_energy.py` | `firing_energy` (Σ positive frame-to-frame rises) + `detrended_sigma` — bleach-immune per-pixel firing-rate scores for pacemaker/wave-source localisation in channels with a global decay envelope (ch651 r3) |
| `brief_parse.py` | `parse_brief(challenge)` → `StructuredBrief` (submit_shape, tolerances, method_summary_must/must_not, disclosed_coords, scoring_brief_text). `find_disclosed_coord_near(brief, hint)` recovers GT-adjacent priors. `validate_against_brief(answer, challenge)` is the canonical pre-submit gate. ch651 r1→r3 lesson — the brief literally contained "Empirical first-firing position … (144, 116)" and 3 rounds were lost not extracting it. |

**REMOVED 2026-04-27** (bridge RPC closure): `bridge_state`, `bridge_preflight`, `dose_budget`, `first_contact`, `per_cell_drop_out`. See `knowledge/Core/Approach/Transferability contract.md`.

---

## knowledge/ — Reasoning Layer

```
knowledge/
  Core/
    Approach/   (~15 files) — meta: how to think about a problem
    Concepts/   (~10 files) — physics + pymmcore-plus/useq API
    Strategies/ (~25 files) — workflow-level patterns
    Pitfalls/    (~5 files) — generalizable methodology traps
  Recipes/      (~40 files) — sample-specific playbooks
  Papers/        (48 files) — verified DOI-backed citations
```

**Pre-challenge:** identify sample type → read `Recipes/<type>.md` → check `Core/Approach/Pre-submission checklist.md` and `Core/Approach/Transferability contract.md`.

`skills/` (at the repo top level, not under `knowledge/`) holds reusable procedure runbooks the agent invokes on itself: `knowledge-audit.md`, `summarize-paper-to-strategy.md`, `pre-submit-review.md`.

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
from src.core.hardware.core import snap, run_events

# --- Single frame ---
img = snap(core, channel='GFP')

# --- Fixed timelapse ---
seq = MDASequence(
    time_plan={"loops": 20, "interval": 2.0},
    channels=[{"config": "GFP"}],  # group auto-discovered
)
results = run_events(core, list(seq))

# --- Fixed Z-stack ---
seq = MDASequence(
    z_plan={"range": 10.0, "step": 0.5},
    channels=[{"config": "GFP"}],  # group auto-discovered
)
results = run_events(core, list(seq))

# --- Multi-position timelapse ---
seq = MDASequence(
    time_plan={"loops": 10, "interval": 1.0},
    stage_positions=[{"x": 100, "y": 200}, {"x": 300, "y": 400}],
    channels=[{"config": "BF"}],
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
        yield MDAEvent(channel={"config": "GFP"})

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
from src.core.hardware.core import snap, move_to, set_objective, run_events, pixel_to_world
from src.core.hardware.config import get_config
from src.core.hardware.autofocus import autofocus_mda  # actually in workflows/

# Detection
from src.core.detection.cells import detect_cells
from src.core.detection.neurons import detect_foci         # returns list of dicts: {cy, cx, sigma, area}
from src.core.detection.tissue import segment_tissue

# Analysis
from src.core.analysis.kinetics import measure_growth_rate_series, fit_q10_with_ci
from src.core.analysis.tracking import track_cells
from src.core.analysis.cell_cycle import classify_fucci_phase
from src.core.analysis.color_analysis import rgb2hed       # H&E stain separation
from src.core.analysis.morphometry import measure_morphometry
from src.core.analysis.intensity import classify_intensity_multiclass
from src.core.analysis.flow import estimate_flow_velocity

# Workflows
from src.core.workflows.adaptive import adaptive_survey_mda
from src.core.workflows.scanning import scan_and_detect_mda
from src.core.workflows.autofocus import autofocus_mda
from src.recipes.bacteria_trap import run_bacteria_trap_mda, measure_bacteria_intensity
from src.recipes.temperature_experiment import temperature_response_curve_v2

# Utils
from src.core.utils.diagnostics import save_snapshot
from src.core.utils.showcase import make_showcase, Panel
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

## Statistics (Feb 2026)

- **104 Python modules** across hardware / detection / analysis / workflows / utils
- **333 scratch** challenge scripts (solve_153 → solve_546+)
- **107+ tests** in `tests/` — run with `pytest tests/ -v`
- **75+ knowledge files** in `knowledge/`

### Post-restart additions (Apr 2026)
- `resolve_channel_group(core, group)` — group-name auto-discovery so
  core/ helpers no longer hardcode sim-specific names like `'Fake'`.
- `multichannel_scan(core, positions, channels, analyze_fn, ...)` — MDA-first
  multi-position × multi-channel acquisition with per-position bucketing.
- `detect_vessels(image, ..., require_nuclear_ring=...)` — RBC-cluster
  vessel detection in H&E with optional endothelial-ring filter.
- `expand_labels_voronoi(labels)` — Voronoi tessellation of nuclear seeds
  into full-FOV cell territories (wraps skimage.segmentation.expand_labels).
- `fit_hill_agonist(doses, responses)` — agonist Hill dose-response fit
  (companion to `fit_hill` for inhibitors).
- `segment_nuclei_hae(..., pixel_size_um=...)` returns `areas_um2` so
  physical-area answers don't drift into camera-px ambiguity.
- skimage migration: `remove_small_objects(min_size=N)` → `max_size=N`
  across 17 files, no semantic change.
- `tests/conftest.py` + `@pytest.mark.live`: opt-in live-hardware
  integration harness (`LIVE_PROXY_URL` / `-m live`).
- Suite: **875 tests** (offline) + 3 live-hardware tests.

---

## Migration Notes (Feb 2026)

> **When adapting code from old `scratch/` challenge scripts**, check `knowledge/core/concepts/Migration notes.md`.

The following were removed. Old scripts using them need updating:

| Removed | Replacement |
|---------|-------------|
| `run_mda()` in `src.core.hardware.core` | `run_events()` |
| `execute_mda()` in `src.core.workflows.mda` | `run_events()` |
| `run_bacteria_trap()` | `run_bacteria_trap_mda()` |
| `MDAEvent(metadata={'properties': {'Camera.Gain': 4.0}})` | `MDAEvent(properties=[('Camera', 'Gain', '4.0')])` |

`run_events()` now delegates to `core.mda.run()` (real MDA engine, works with proxy via WebSocket) instead of a manual `snapImage()` loop.

---

## Research References

- [Smart Microscopy Roadmap (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.08.18.670881v2.full)
- [Event-driven acquisition (Nature Methods 2022)](https://www.nature.com/articles/s41592-022-01589-x)
- [pymmcore-plus event-driven guide](https://pymmcore-plus.github.io/pymmcore-plus/guides/event_driven_acquisition/)
- [pymmcore-plus MDA engine](https://pymmcore-plus.github.io/pymmcore-plus/guides/mda_engine/)
- [useq-schema MDASequence](https://pymmcore-plus.github.io/useq-schema/schema/sequence/)
- [useq-schema MDAEvent](https://pymmcore-plus.github.io/useq-schema/schema/event/)
