"""Test 10: cell-cycle mitosis detection — DAPI + mScarlet, static cells, in focus.

The agent's goal is to detect and classify cells in a specific mitosis stage.
Only 2 cells are in mitosis (one randomly assigned mitosis sub-stage each);
the remaining cells are in G1 or S so the mitotic cells stand out clearly.

DAPI (mode 1) shows nucleus/chromatin — pattern differs per mitosis stage.
mScarlet (mode 2) shows the membrane outline.

Channels:
  DAPI     — nucleus  — filter SCFP2(434/474)    + LED UV
  mScarlet — membrane — filter mScarlet3(569/582) + LED ORANGE

Simulation area: 1500x1500 µm, 100 cell-cycle cells, in focus.
"""

import numpy as np


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

    # 2 cells start at the beginning of G2 — they will enter M after ~2 real minutes
    g2_indices = rng.choice(len(sim._cells), size=2, replace=False)

    for i, cell in enumerate(sim._cells):
        cell.is_dying = False
        cell.n_div    = 0
        cell.brownian_d = TEST_CONFIG["brownian_d"]

        if i in g2_indices:
            cell.cell_cycle_state   = "G2"
            cell.cell_mitosis_state = "Interphase"
            cell.current_time_life  = 18.0  # start of G2
        else:
            # 70% G1, 30% S
            cell.cell_cycle_state   = "G1" if rng.random() < 0.7 else "S"
            cell.cell_mitosis_state = "Interphase"
            cell.current_time_life  = cell._initial_random_time_life()

    return sim


TEST_CONFIG = {
    "title": "Cell-cycle mitosis detection — 2 mitotic cells, 1500x1500 µm",
    "backend": "particle",
    "cell_type": "cycle",
    "n_cells": 100,
    "seed": 7,
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
