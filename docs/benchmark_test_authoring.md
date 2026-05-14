# Benchmark Tests

Benchmark tests are self-contained simulation scenarios used to evaluate agent performance. Each test lives in its own folder under `src/benchmarking/test_<N>/` and is served to the agent as a remote microscope via the test server (the agent cannot see the test configuration).

---

## Folder structure

```
src/benchmarking/test_<N>/
    test.yaml             ← simulation config (required for new tests)
    initialize_test.py    ← custom Python setup (optional, only when needed)
```

---

## Pattern 1 — Pure YAML

Create only `test.yaml`. No Python file needed.

Use this when the backend's `create_sim()` function accepts all parameters directly (histology, celegans, organoid backends).

```yaml
title: "H&E grade 0 — glandular tissue, 40x objective"
description: |
  Normal glandular tissue viewed at 40x. The agent must identify structural
  regions: glandular lumens, epithelial nuclei, stroma.

backend: histology
cell_type: histology
focal_plane: 0.0

tissue_type: glandular
grade: 0
n_nuclei: 200
world_size: 512
internal_scale: 4
seed: 17

initial_properties:
  - [Objective, Label, "40x"]

channels:
  - name: H-and-E
    filter: "Electra1(402/454)"
    led: CYAN
```

---

## Pattern 2 — Hybrid (YAML + Python)

Create both `test.yaml` and `initialize_test.py`.

Use this when the simulation needs custom setup after creation: setting per-cell state, applying distributions, patching pipelines, or subclassing the sim.

**`test.yaml`** — all configuration, same schema as pure YAML:

```yaml
title: "Cell-cycle fluorescence — 100 cells"
description: |
  Cells distributed across G1/S/G2/M with realistic weighted proportions.

backend: particle
cell_type: cycle
n_cells: 100
seed: 42
width: 1500
height: 1500
brownian_d: 0.0
focal_plane: 0.0

channels:
  - name: DAPI
    filter: "SCFP2(434/474)"
    led: UV
  - name: mScarlet
    filter: "mScarlet3(569/582)"
    led: ORANGE
```

**`initialize_test.py`** — only `create_sim_override()`, no `TEST_CONFIG`:

```python
# TEST_CONFIG is injected from test.yaml by test_runner.py

import numpy as np

_STATE_DIST = [
    ("G1", "Interphase", 55),
    ("S",  "Interphase", 20),
    ("G2", "Interphase", 15),
    ("M",  "Prophase",   10),
]

def create_sim_override():
    from virtual_microscope.sims.cell.sim import ScatteredCellSim

    sim = ScatteredCellSim(
        width=TEST_CONFIG["width"],
        height=TEST_CONFIG["height"],
        n_cells=TEST_CONFIG["n_cells"],
        cell_type=TEST_CONFIG["cell_type"],
        seed=TEST_CONFIG["seed"],
    )

    rng     = np.random.default_rng(TEST_CONFIG["seed"])
    states  = [(cc, cm) for cc, cm, _ in _STATE_DIST]
    weights = np.array([w for _, _, w in _STATE_DIST], dtype=float)
    weights /= weights.sum()
    choices = rng.choice(len(states), size=len(sim._cells), p=weights)

    for cell, idx in zip(sim._cells, choices):
        cell.cell_cycle_state   = states[idx][0]
        cell.cell_mitosis_state = states[idx][1]
        cell.brownian_d         = TEST_CONFIG["brownian_d"]

    return sim
```

`TEST_CONFIG` does **not** need to be defined in the Python file. The test runner injects the YAML config into the module namespace before calling `create_sim_override()`, so any field from `test.yaml` is accessible via `TEST_CONFIG["key"]`.

---

## YAML field reference

### Required

| Field | Type | Description |
|---|---|---|
| `backend` | string | Virtual-microscope backend (`particle`, `histology`, `celegans`, `organoid`, …) |
| `channels` | list | Imaging channels exposed to the agent |

### Runner fields (optional)

| Field | Type | Default | Description |
|---|---|---|---|
| `title` | string | `""` | Short label in the GUI test catalog |
| `description` | string | `""` | Multi-line task description shown to the agent via `GET /test/info` |
| `cell_type` | string | `"normal"` | Determines the generated `.cfg` filename (`virtual_<cell_type>.cfg`) |
| `focal_plane` | float | `0.0` | Z offset from tissue plane in µm; non-zero = starts out of focus |
| `phase_contrast` | bool | `false` | Add a hardcoded phase-contrast channel to the cfg |
| `slm` | bool | `false` | Include the SLM device in the cfg |
| `time_scale` | float | `0.05` | Realtime engine speed (for continuous simulations) |
| `tick_hz` | int | `10` | Realtime engine tick rate |
| `initial_properties` | list | `[]` | Device property overrides applied at startup: `[device, property, value]` |

### Channel definition

```yaml
channels:
  - name: DAPI                    # name exposed to the agent
    filter: "SCFP2(434/474)"      # Filter Wheel label
    led: UV                       # LED label
```

Valid `led` values: `UV`, `CYAN`, `GREEN`, `ORANGE`, `RED`

### Backend-specific sim kwargs

Any field not listed above is forwarded as a keyword argument to `backend.create_sim()`.

**particle** — `n_cells`, `seed`, `width`, `height`, `brownian_d`, `cell_type` (`normal` / `cycle` / `optogenetic`), `radius_min`, `radius_max`

**histology** — `tissue_type`, `grade`, `n_nuclei`, `world_size`, `internal_scale`, `seed`

**celegans** — `world_size`, `worm_length`, `worm_width`, `speed`, `seed`

**organoid** — `outer_radius`, `wall_thickness`, `n_cells`, `n_buds`, `lumen_opacity`, `seed`

### `initial_properties`

Sets device properties before the agent connects. Values are always strings.

```yaml
initial_properties:
  - [ZStage, Position, "50.0"]     # start 50 µm out of focus
  - [Objective, Label, "40x"]      # pre-select 40x objective
```

---

## When is a Python override required?

Use `initialize_test.py` only when you need to:

- Assign per-cell state after construction (cell cycle phase, apoptosis stage, division count)
- Apply weighted distributions using numpy
- Enable simulation features via method calls (e.g. `sim.enable_photobleaching(rate=0.005)`)
- Replace pipeline components (e.g. `sim._pipeline[1] = OpticalPipeline(...)`)
- Subclass the simulation (e.g. to inject stage drift into `_crop_fov`)

Otherwise, pure YAML is preferred.

---

## Verifying a new test

**CLI smoke-test:**
```bash
python -m src.benchmarking.test_runner test_<N>
```
This loads the config, creates the simulation, validates channels, and prints a summary. No GUI needed.

**List all available tests:**
```bash
python -m src.benchmarking.test_runner
```

**Full server test:**
```bash
python -m src.benchmarking.test_server test_<N> --port 5602
```
Then open `http://127.0.0.1:5602/test/info` in a browser to confirm the title, description, and channel list are correct. The agent connects to this URL as a remote microscope.

**From the GUI:** Launch the MCPServer panel, open the Benchmarking section, and check that the new test appears in the dropdown.
