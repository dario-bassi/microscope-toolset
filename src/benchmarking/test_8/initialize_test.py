"""Test 8: cell-cycle fluorescence — DAPI + mScarlet, static cells, in focus.

Cells are distributed across realistic cell-cycle states:
  ~55% G1/Interphase, ~20% S/Interphase, ~15% G2/Interphase,
  ~10% mitotic (Prophase → Cytokinesis).

DAPI (mode 1) shows nucleus/chromatin — intensity and pattern change per state.
mScarlet (mode 2) shows the membrane outline.

Channels:
  DAPI     — nucleus  — filter SCFP2(434/474)    + LED UV
  mScarlet — membrane — filter mScarlet3(569/582) + LED ORANGE

Simulation area: 1500x1500 µm, 100 cell-cycle cells, in focus.
"""

import numpy as np


# Weighted state distribution mimicking a proliferating population.
# Each entry: (cell_cycle_state, cell_mitosis_state, weight)
_STATE_DIST = [
    ("G1", "Interphase",  55),
    ("S",  "Interphase",  20),
    ("G2", "Interphase",  15),
    ("M",  "Prophase",     3),
    ("M",  "Metaphase",    3),
    ("M",  "Anaphase",     2),
    ("M",  "Telophase",    1),
    ("M",  "Cytokinesis",  1),
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

    rng = np.random.default_rng(TEST_CONFIG["seed"])
    states   = [(cc, cm) for cc, cm, _ in _STATE_DIST]
    weights  = np.array([w for _, _, w in _STATE_DIST], dtype=float)
    weights /= weights.sum()
    choices  = rng.choice(len(states), size=len(sim._cells), p=weights)

    for cell, idx in zip(sim._cells, choices):
        cell.cell_cycle_state  = states[idx][0]
        cell.cell_mitosis_state = states[idx][1]
        cell.brownian_d = TEST_CONFIG["brownian_d"]
        cell.is_dying = False
        cell.n_div = 0
        cell.current_time_life = cell._initial_random_time_life()

    return sim


TEST_CONFIG = {
    "title": "Cell-cycle fluorescence — DAPI + mScarlet, 100 cells, 1500x1500 µm",
    "backend": "particle",
    "cell_type": "cycle",
    "n_cells": 100,
    "seed": 42,
    "width": 1500,
    "height": 1500,
    "brownian_d": 0.0,
    "focal_plane": 0.0,
    "channels": [
        {
            "name": "DAPI",
            "filter": "SCFP2(434/474)",
            "led": "UV",
        },
        {
            "name": "mScarlet",
            "filter": "mScarlet3(569/582)",
            "led": "ORANGE",
        },
    ],
}
