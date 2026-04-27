"""Test 9: cell-cycle simulation focused on apoptosis.

Cells are distributed with a bias toward high division counts so that
apoptosis events are frequent and visible from the start:
  - 20 cells already dying, spread across all apoptosis stages
  - 25 cells one division away from death (n_div=9)
  - 25 cells two divisions away (n_div=8)
  - 30 cells with low division counts (n_div 0-7), normal cycle states

DAPI (mode 1) shows nucleus/chromatin.
mScarlet (mode 2) shows the membrane outline.

Simulation area: 1500x1500 µm, 100 cell-cycle cells, in focus.
"""

import numpy as np


# Must match time_table_apoptosis in cycle.py: {phase: cumulative_end_time}
_APOPTOSIS_STAGES = [
    ("Shrinkage",        0.0, 1.0),
    ("Blebbing",         1.0, 2.0),
    ("Apoptotic bodies", 2.0, 2.5),
    ("Phagocytosis",     2.5, 3.0),
]

_STATE_DIST = [
    ("G1", "Interphase", 55),
    ("S",  "Interphase", 20),
    ("G2", "Interphase", 15),
    ("M",  "Prophase",    3),
    ("M",  "Metaphase",   3),
    ("M",  "Anaphase",    2),
    ("M",  "Telophase",   1),
    ("M",  "Cytokinesis", 1),
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

    n_dying    = 2
    n_high_div = 30   # n_div 8 or 9, split evenly
    # remaining ~68 cells get n_div 0-7 with normal cycle states

    states  = [(cc, cm) for cc, cm, _ in _STATE_DIST]
    weights = np.array([w for _, _, w in _STATE_DIST], dtype=float)
    weights /= weights.sum()

    order = rng.permutation(len(sim._cells))

    for rank, cell_idx in enumerate(order):
        cell = sim._cells[cell_idx]
        cell.brownian_d = TEST_CONFIG["brownian_d"]

        if rank < n_dying:
            # Already in apoptosis — place each in a random stage
            cell.n_div    = 10
            cell.is_dying = True
            cell.death_timer = float(rng.uniform(0.0, 3.0))
            for phase, lo, hi in _APOPTOSIS_STAGES:
                if lo <= cell.death_timer < hi:
                    cell.apoptosis_death_phase = phase
                    break

        else:
            cell.is_dying = False
            if rank < n_dying + n_high_div:
                # Alternate between n_div=8 and n_div=9
                cell.n_div = 8 + (rank % 2)
            else:
                cell.n_div = int(rng.integers(0, 8))

            idx = rng.choice(len(states), p=weights)
            cell.cell_cycle_state   = states[idx][0]
            cell.cell_mitosis_state = states[idx][1]
            cell.current_time_life  = cell._initial_random_time_life()

    return sim


TEST_CONFIG = {
    "title": "Cell-cycle apoptosis — DAPI + mScarlet, 100 cells, 1500x1500 µm",
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
