# Smart Microscope Toolkit — Architecture

> **Navigation guide:** use this file to find where things live and how they connect.

---

## Quick Reference — Where to Find What

| Task | Location |
|------|----------|
| Snap an image, move stage, set objective | `src/self_learn/hardware/core.py` |
| Run a timelapse / Z-stack / multi-position | `src/self_learn/hardware/core.py:run_events()` |
| Detect cells / tissue / neurons | `src/self_learn/detection/` |
| Measure intensity / morphology / kinetics | `src/self_learn/analysis/` |
| High-level protocol (adaptive survey, tracking, optogenetics) | `src/self_learn/workflows/` |
| Channel discovery, pixel size, config group | `src/self_learn/hardware/config.py` |
| Autofocus | `src/self_learn/workflows/autofocus.py` |
| FUCCI cell cycle | `src/self_learn/analysis/cell_cycle.py` |
| Save overlay / showcase image | `src/self_learn/utils/diagnostics.py`, `src/self_learn/utils/showcase.py` |
| Universal microscopy strategies | `src/self_learn/knowledge/Core/Approach/`, `Core/Strategies/` |
| pymmcore-plus / useq API reference | `src/self_learn/knowledge/Core/Concepts/` |
| Generalizable pitfalls and lessons | `src/self_learn/knowledge/Core/Pitfalls/` |
| Verified paper citations | `src/self_learn/knowledge/Papers/` |
| MCP server tools | `src/mcp_microscopetoolset/server_setup.py` |
| Napari plugin entry point | `src/plugin_napari.py` |
| Benchmarking / test harness | `src/benchmarking/` |

---

## Layer Architecture

```
LLM Agent  (reasoning + vision)
      │
  Knowledge  (src/self_learn/knowledge/)   ← reasoning, strategies, playbooks
      │
  Workflows  (src/self_learn/workflows/)   ← acquisition + analysis protocols
      │
  Analysis   (src/self_learn/analysis/)    ← image feature extraction
  Detection  (src/self_learn/detection/)   ← segmentation, object detection
      │
  Hardware   (src/self_learn/hardware/)    ← pymmcore-plus wrappers
      │
  Utils      (src/self_learn/utils/)       ← diagnostics, logging, reporting
      │
  MCP Server (src/mcp_microscopetoolset/)  ← tool surface for LLM clients
  Napari UI  (src/plugin_napari.py)        ← live view + GUI
```

**Rule:** layers only call downward. Workflows call analysis/detection/hardware; analysis never calls workflows.

---

## src/self_learn/hardware/ — Microscope Control

The single entry point for all hardware interaction.

| Module | Key functions |
|--------|--------------|
| `core.py` | `snap()`, `move_to()`, `get_position()`, `set_objective()`, `get_pixel_size()`, `set_z()`, `get_z()`, `apply_slm()`, `make_slm_circle()`, `pixel_to_world()`, `world_to_pixel()`, `run_events()` |
| `config.py` | `get_config()`, `refresh_config()`, `resolve_channel_group(core, group)` — runtime discovery of channel group, pixel size, objective, SLM, XY/Z devices. Pass `group=None` anywhere in self_learn/ to auto-discover. |
| `quality.py` | SNR estimation, focus variance, brightness assessment |
| `zstack.py` | Z-stack acquisition, best-focus plane detection |
| `drift.py` | Drift detection/correction via phase correlation and centroid tracking |
| `validate.py` | Preflight checks — channels, objectives, SLM, stage availability |

---

## src/self_learn/detection/ — Object Detection & Segmentation

| Module | What it detects |
|--------|----------------|
| `cells.py` | General cell detection (brightfield threshold, blob detection) |
| `tissue.py` | Tissue/membrane segmentation via Otsu; contact graph analysis |
| `neurons.py` | Soma detection, puncta counting, neurite analysis from immunostaining |
| `segmentation.py` | Generic strategies: watershed, adaptive threshold, marker-controlled, `expand_labels_voronoi()` for nuclear-seed → cell-territory tessellation |
| `threshold.py` | Adaptive threshold selection with quality-checked retry |
| `consensus.py` | Multi-method consensus for robust counting |

---

## src/self_learn/analysis/ — Feature Extraction

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
| `histology.py` | H&E deconvolution, nuclear morphometry, necrosis detection, `detect_vessels()` |
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
| `dose_response.py` | Hill curve fitting (see also `workflows/dose_response.py`) |
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

## src/self_learn/workflows/ — Acquisition & Analysis Protocols

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
| `batch.py` | Batch processing over multiple positions / tiles. `multichannel_scan()` runs a single MDASequence over N positions × M channels and buckets frames per-position. |
| `solve_harness.py` | Standardized experiment startup: snap all channels, characterize, return setup dict |
| `experiment.py` | Multi-phase experiment builder (baseline-treatment-recovery) |
| `optimization.py` | Imaging parameter optimisation: exposure, gain, SNR |

---

## src/self_learn/utils/ — Utilities

**Cross-cutting helpers** (always-on, used everywhere):
| Module | What it does |
|--------|-------------|
| `diagnostics.py` | Save snapshots / overlays for visual verification |
| `experiment_log.py` | Structured JSON experiment logging for reproducibility |
| `image.py` | Grayscale conversion, normalisation, basic preprocessing |
| `report.py` | Experiment report generator with measurement statistics |
| `showcase.py` | Multi-panel publication-style showcase figures (matplotlib `Panel` + `make_showcase` registry) |
| `validation.py` | Pre-submission answer validation (range checks, JSON-safe types) |
| `preflight.py` | Code-callable pre-submission preflight checks |
| `sample_classifier.py` | Cheap-feature image → sample-class label |
| `auto_recipe.py` | Image + core → `RecipeSuggestion` dispatcher |
| `mda_diagnostics.py` | `run_events_checked()` — MDA truncation guard |
| `agree_or_flag.py` | Two-estimator agreement check — validates via cross-check, not naive averaging |

**Image-only measurement primitives** (caller-opted-in):
| Module | What it does |
|--------|-------------|
| `sensorless_ao.py` | DM-state sweep + argmax + per-axis-presence inference |
| `slm_masks.py` | circle / gaussian / ring SLM mask builders, vectorised over centre count |
| `spectral_leak.py` | top-percentile mask + N×N leak matrix + `unmix(K, observed)` |
| `fft_peak.py` | `FFTPeak` dataclass + `find_fft_peak` + `find_top_n_fft_peaks` |
| `fluorophore_brightness.py` | `BrightnessRanking` + predict + measure + compare |
| `rate_limited_drive.py` | `predict_n_steps` closed-form + `drive_to_band` callback loop |
| `pulsed_schedule.py` | `predict_segment_end` + `plan_trajectory` for multi-waypoint MPC |
| `wave_period.py` | `estimate_wave_period(core, burst_fn)` 10-frame FFT preview → recommended `n_burst` |
| `firing_energy.py` | `firing_energy` + `detrended_sigma` — bleach-immune per-pixel firing-rate scores |
| `axis_sweep.py` | Generic axis-sweep: sweeps a hardware state axis and scores each state |

---

## src/mcp_microscopetoolset/ — MCP Server

The FastMCP server that exposes microscope tools to LLM clients. Runs alongside napari.

**Hardware tools** (call raw `mmc` from executor namespace):
- `snap_image` — snap + return shape/stats (not pixel data)
- `move_stage` — absolute or relative XY move, waits for completion
- `set_objective` — switch objective by state label
- `get_stage_position` — current X, Y, Z in µm
- `get_microscope_events` / `get_last_microscope_event` — event cache query

**Execution tool** (runs Python in a sandboxed namespace with pre-configured `mmc`):
- `execute_python_code(code, execution_mode)` — `buffered` (hardware buffered + committed atomically) or `live` (direct hardware access). Pre-configured namespace includes `mmc`, `run_mda_with_feedback`, `center_on_cell`, `find_bright_centroid`, `detect_cells`. Guards: cannot reinstantiate `CMMCorePlus`, cannot call `loadSystemConfiguration`, cannot access `viewer` directly.
- `install_packages` — pip-install with user consent required

**Database/retrieval tools**:
- `pymmcore_api_database` — BM25+KNN search of pymmcore-plus API docs
- `micromanager_device_database` — BM25+KNN search of Micro-Manager device docs
- `pdfs_publication_database` — semantic search of scientific publications
- `log_session` / `retrieve_session_logs` — PostgreSQL-backed session memory

**Napari viewer tools** (all run via `viewer_proxy` on the main thread — never call `viewer` from executed code):
- `viewer_screenshot`, `viewer_layer_screenshot`, `view_image`
- `viewer_add_image`, `viewer_add_labels`, `viewer_add_points`, `viewer_add_tracks`
- `viewer_list_of_layers`, `viewer_session_information`
- `viewer_remove_layer`, `viewer_set_layer_properties`, `viewer_reorder_layer`
- `viewer_set_camera`, `viewer_reset_view`, `viewer_set_ndisplay`, `viewer_set_dims_current_step`
- `viewer_set_active_layer`, `viewer_set_grid`
- `get_layer_data` — export layer to TIFF for use in executed code

**Utility tools**:
- `get_experiment_workspace` — path to the active experiment workspace dir (set when user clicks "Start Tracking")
- `request_user_clarification` — elicit user input via MCP client
- `answer_no_coding_query` — flag non-hardware requests

---

## src/local/ — Local Execution Helpers

Used by the MCP executor and available in the `execute_python_code` namespace.

| Module | What it does |
|--------|-------------|
| `execute.py` | `Execute` class: sandboxed code execution with AST guards and buffered/live modes |
| `gatekeeper_core.py` | `GatekeeperCore` — wraps `CMMCorePlus`, buffers hardware calls in `buffered` mode, commits atomically |
| `mda_helpers.py` | `run_mda_with_feedback(mmc, events, on_frame)` — MDA + synchronous per-frame callback |
| `microscopy_utils.py` | `center_on_cell()`, `find_bright_centroid()`, `detect_cells()` — smart acquisition helpers |

---

## knowledge/ — Reasoning Layer

```
src/self_learn/knowledge/
  Core/
    Approach/   (~17 files) — meta: how to think about a problem
    Concepts/   (~10 files) — physics + pymmcore-plus/useq API
    Strategies/ (~25 files) — workflow-level patterns
    Pitfalls/    (~6 files) — generalizable methodology traps
  Papers/        (58 files) — verified DOI-backed citations
```

**Entry point:** `src/self_learn/knowledge/INDEX.md` — start here.

**Pre-experiment:** identify sample type → read `Core/Approach/How to approach a problem.md` → read relevant `Core/Strategies/*.md`.

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

**`run_events(core, events, on_frame=)`** is the single execution entry point for all multi-frame acquisitions. It delegates to `core.mda.run()`, which works identically for local `CMMCorePlus` and remote `pymmcore-proxy` (events are streamed over WebSocket; `frameReady` signals are forwarded back).

Both accept any iterable of `MDAEvent` (generator, list, `MDASequence`, `Queue`-backed iterator).

### Acquisition patterns

```python
from useq import MDASequence, MDAEvent
from self_learn.hardware.core import snap, run_events

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
    channels=[{"config": "GFP"}],
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
```

---

## Common Import Patterns

```python
# Hardware
from self_learn.hardware.core import snap, move_to, set_objective, run_events, pixel_to_world
from self_learn.hardware.config import get_config, resolve_channel_group

# Detection
from self_learn.detection.cells import detect_cells
from self_learn.detection.neurons import detect_foci         # returns list of dicts: {cy, cx, sigma, area}
from self_learn.detection.tissue import segment_tissue

# Analysis
from self_learn.analysis.kinetics import measure_growth_rate_series
from self_learn.analysis.tracking import track_cells
from self_learn.analysis.cell_cycle import classify_fucci_phase
from self_learn.analysis.color_analysis import rgb2hed       # H&E stain separation
from self_learn.analysis.morphometry import measure_morphometry
from self_learn.analysis.intensity import classify_intensity_multiclass

# Workflows
from self_learn.workflows.adaptive import adaptive_survey_mda
from self_learn.workflows.scanning import scan_and_detect_mda
from self_learn.workflows.autofocus import autofocus_mda
from self_learn.workflows.batch import multichannel_scan, tile_and_analyze

# Utils
from self_learn.utils.diagnostics import save_snapshot
from self_learn.utils.showcase import make_showcase, Panel
from self_learn.utils.mda_diagnostics import run_events_checked
```

---

## Key API Gotchas

- `detect_foci()` returns **list of dicts** `{'cy','cx','sigma','area',...}` — not tuples
- `analyze_cell_fibers()` → `subtypes` is a **list** of labels, not a dict
- `fiber_differential()` → keys are `VSF_count`, `DSF_count`
- `rgb2hed()` for H&E separation — **not** `color_deconvolution()`
- `set_objective(core, 40)` — integer argument, not string `"40x"`
- `event.properties` for per-event device property changes: `[('Camera', 'Gain', '4')]` — native MDAEvent field, handled by the MDA engine
- `resolve_channel_group(core, None)` — always pass `None` for the group arg when you want auto-discovery; never hardcode `'Fake'` or `'Channel'`

---

## Research References

- [Smart Microscopy Roadmap (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.08.18.670881v2.full)
- [Event-driven acquisition (Nature Methods 2022)](https://www.nature.com/articles/s41592-022-01589-x)
- [pymmcore-plus event-driven guide](https://pymmcore-plus.github.io/pymmcore-plus/guides/event_driven_acquisition/)
- [pymmcore-plus MDA engine](https://pymmcore-plus.github.io/pymmcore-plus/guides/mda_engine/)
- [useq-schema MDASequence](https://pymmcore-plus.github.io/useq-schema/schema/sequence/)
- [useq-schema MDAEvent](https://pymmcore-plus.github.io/useq-schema/schema/event/)
