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
│       └── watershed_split() from detection/segmentation.py
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
└── Unknown / complex
    └── Look at the image first, then decide
        └── Save to /tmp/, view with Read tool
```

## Analysis Method Selection

```
What do you need to measure?
│
├── Cell population classification
│   ├── Known categories (WBC types) → classify_rules() with feature thresholds
│   ├── Unknown categories → classify_kmeans() for unsupervised clustering
│   └── Simple size bins → classify_size_bins()
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
└── Time series
    ├── Periodic signal → measure_periodic_rate()
    ├── Dose-response → fit_hill()
    ├── Recovery curve → fit_exponential_recovery()
    └── Decay curve → fit_exponential_decay()
```

## Magnification Selection

```
What measurement?
│
├── Count cells in large area → lowest magnification (largest FOV)
├── Intermediate overview → mid magnification
├── Morphology / subcellular → high magnification (smallest FOV)
└── Multi-scale → survey at low mag, measure at high mag
```

Always query pixel_size and FOV from hardware — values vary by instrument.

## When to Use snap() Loops vs MDA

```
Use snap() loops when:
├── Exploring / previewing (1-3 frames)
├── Interactive decision-making between frames
├── Hardware operations between acquisitions (SLM, stage moves)
└── Simple single-frame acquisition

Use MDA when:
├── Timelapse with consistent timing
├── Multi-position visits
├── Multi-channel at each position
├── Z-stacks
└── Any reproducible acquisition protocol
```

## Strategy Knowledge Reading Order

1. `knowledge/strategies/00_how_to_approach_a_problem.md` — Always read first
2. `knowledge/strategies/01_backward_design.md` — Before designing any experiment
3. Relevant workflow from `knowledge/workflows/` — For the experiment category
4. Relevant playbook from `knowledge/playbooks/` — For the sample type
5. `knowledge/pymmcore/` — When unsure about API patterns
