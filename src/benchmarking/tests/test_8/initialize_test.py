# TEST_CONFIG is injected from test.yaml by test_runner.py

import numpy as np


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
        cell.cell_cycle_state   = states[idx][0]
        cell.cell_mitosis_state = states[idx][1]
        cell.brownian_d = TEST_CONFIG["brownian_d"]
        cell.is_dying = False
        cell.n_div = 0
        cell.current_time_life = cell._initial_random_time_life()

    return sim
