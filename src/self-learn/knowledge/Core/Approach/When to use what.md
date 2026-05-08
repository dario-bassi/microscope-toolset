# Decision Framework: When to Use What

## Acquisition Method Selection

```
What kind of experiment?
│
├── Fixed acquisition (known parameters upfront)
│   └── USE: MDASequence from pymmcore-plus
│       ├── Timelapse → time_plan
│       ├── Z-stack → z_plan
│       ├── Multi-position → stage_positions
│       ├── Multi-channel → channels
│       └── Combined → compose all plans in one MDASequence
│
├── Adaptive / closed-loop (decisions depend on data)
│   └── USE: adaptive_generator() from oada.py
│       ├── Early stopping → state.stop = True
│       ├── Event injection → state.extra_events.append(...)
│       ├── Exposure adaptation → state.next_exposure = X
│       └── Measurement tracking → state.measurements.append(...)
│
├── Grid scan with per-tile decisions
│   └── USE: adaptive_scan() from oada.py
│       └── Survey + selective zoom pattern
│
├── Event-driven (poll/burst)
│   └── USE: event_driven.py
│       └── Monitor slowly, burst on detected events
│
└── Multi-phase experiment
    └── USE: experiment.py state machine
        └── make_experiment() + experiment_to_mda()
```

## Detection Method Selection

```
What does the sample look like?
│
├── Individual bright dots on dark background
│   └── LoG blob detection or peak_local_max
│       └── detect_blobs_log() or peak_local_max
│
├── Large space-filling cells (confluent tissue)
│   └── Threshold + connected components
│       └── detect_cells() with threshold_sigma
│
├── Dense clusters of cells
│   └── Watershed segmentation
│       └── watershed_split(min_distance='auto') — auto-adapts to object spacing
│
├── Uneven illumination (bright on one side)
│   └── Local adaptive thresholding
│       └── adaptive_threshold() from detection/segmentation.py
│
├── Membrane + nuclear channels available
│   └── Marker-controlled watershed
│       └── segment_by_markers(membrane, nuclear)
│
├── Elongated structures (dendrites, fibers)
│   └── Skeletonization + branch point analysis
│       └── count_branch_points()
│
├── Fluorescence signal quantification
│   └── detect_fluorescence() from detection module
│
├── Setting fluorescence detection threshold
│   └── Noise-floor based (NOT arbitrary multipliers)
│       └── estimate_noise_floor() → use threshold_2sigma or threshold_1_3sigma
│       └── k=1.3 (faint foci), k=2.0 (standard), k=3.0 (conservative)
│
└── Unknown / complex
    └── Look at the image first, then decide
        └── Save to /tmp/, view with Read tool
```

## Analysis Method Selection

```
What do you need to measure?
│
├── Cell population classification
│   ├── Known categories with clear rules → classify_rules()
│   ├── Soft/overlapping clusters → classify_gmm() (probabilities, BIC for k)
│   ├── Unknown k, need outlier detection → classify_dbscan()
│   ├── Hard clustering with known k → classify_kmeans()
│   ├── 1D intensity-based → multi_otsu() (N-class threshold)
│   └── Simple size bins → classify_size_bins()
│
├── Statistical comparison
│   ├── Two groups → compare_two() (auto t-test/Mann-Whitney)
│   ├── Multiple groups → compare_multiple() (ANOVA/Kruskal-Wallis)
│   ├── Pairwise after ANOVA → pairwise_comparisons() (with correction)
│   └── Multiple p-values → correct_pvalues('holm') or correct_pvalues('fdr')
│
├── Spatial distribution
│   ├── Overall pattern (random/clustered/regular) → clark_evans_index()
│   ├── Multi-scale clustering → ripleys_l() at different radii
│   ├── Per-cell territory → voronoi_areas()
│   └── Grid uniformity → quadrat_count()
│
├── Z-stack / 3D
│   ├── Flattened view → project_mip() or project_mean()
│   ├── Volume measurement → measure_volume()
│   ├── Best focal plane → find_focus_plane()
│   └── Cross-section → slice_orthogonal()
│
├── Motion / dynamics
│   ├── Space-time diagram → kymograph()
│   ├── Flow fields → optical_flow()
│   ├── Contraction amplitude → contraction_amplitude()
│   └── Leading edge speed → migration_front()
│
├── Trajectory smoothing
│   ├── Simple noise removal → smooth_trajectory(method='rolling')
│   ├── Preserve derivatives → smooth_trajectory(method='savgol')
│   └── Object with inertia → kalman_smooth() (velocity + position)
│
└── Time series
    ├── Periodic signal → measure_periodic_rate()
    ├── Dose-response → fit_hill()
    ├── Recovery curve → fit_exponential_recovery()
    ├── Decay curve → fit_exponential_decay()
    ├── Changepoint (optimal) → detect_changepoint(method='pelt')
    ├── Changepoint (streaming) → detect_changepoint(method='cusum')
    └── Temperature response → fit_q10_with_ci()
```

## Magnification Selection

```
What measurement?
│
├── Counting (whole sample)
│   └── 10x (1.0 µm/px, FOV=512×512 µm)
│       └── Entire sample in one snap, best for population counts
│       └── BUT: sub-pixel jitter dominates for slow-moving objects
│
├── Overview + basic classification
│   └── 10x or 20x (0.5 µm/px, FOV=256×256 µm)
│       └── Good balance of coverage and resolution
│
├── Morphology / subcellular detail
│   └── 40x (0.25 µm/px, FOV=128×128 µm)
│       └── Cell shape, nuclear morphology, fiber structure
│       └── Only 1-4 cells visible → must navigate to target
│
├── Tracking slowly-moving objects
│   └── 20x or 40x (NOT 10x!)
│       └── At 10x, centroid jitter ≈ 2-3 px → dominates slow speeds
│       └── C. elegans at 4°C: use 20x minimum
│
├── Multi-scale workflow
│   └── 10x survey → identify regions → 40x detail
│       └── overview_first() + zoom + measure
│       └── Convert coordinates: world stays the same
│
└── ALWAYS: start with 10x overview even for non-counting tasks
    └── See what's on the sample before zooming in
```

**VERIFY** objective change: `ps = core.getPixelSizeUm()` after `set_objective(core, N)`.
If FOV shows too many/few objects, the objective may not have switched.

## When to Use snap() Loops vs MDA

```
Use snap() loops when:
├── Exploring / previewing (1-3 frames)
├── Interactive decision-making between frames
├── Device state changes between phases (Perfusion, Temperature)
├── SLM mask updates with per-frame feedback
└── Simple single-frame acquisition

Use timelapse() when:
└── Standard N-frame timelapse in one channel
    └── timelapse(core, n_frames=20, interval_s=1.0, channel='GFP')

Use MDA (MDASequence) when:
├── Multi-channel timelapse
├── Multi-position visits
├── Z-stacks
├── Complex axis ordering (tpzc)
└── Any reproducible multi-dimensional acquisition

Always use parameter_advisor for pixel-domain params:
└── suggest_watershed_params(ps, object_type=)
    suggest_tracking_params(ps, object_type=, dt_s=)
    suggest_detection_params(ps, object_type=)
    suggest_nc_params(ps)
```

## Strategy Knowledge Reading Order

1. `Core/Approach/How to approach a problem.md` — Always read first
2. `Core/Approach/Backward design.md` — Before designing any experiment
3. Relevant workflow from `knowledge/workflows/` — For the experiment category
4. Relevant playbook from `knowledge/playbooks/` — For the sample type
5. `knowledge/pymmcore/` — When unsure about API patterns
