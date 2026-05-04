# TEST_CONFIG is injected from test.yaml by test_runner.py

import numpy as np


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
    n_high_div = 30

    states  = [(cc, cm) for cc, cm, _ in _STATE_DIST]
    weights = np.array([w for _, _, w in _STATE_DIST], dtype=float)
    weights /= weights.sum()

    order = rng.permutation(len(sim._cells))

    for rank, cell_idx in enumerate(order):
        cell = sim._cells[cell_idx]
        cell.brownian_d = TEST_CONFIG["brownian_d"]

        if rank < n_dying:
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
                cell.n_div = 8 + (rank % 2)
            else:
                cell.n_div = int(rng.integers(0, 8))

            idx = rng.choice(len(states), p=weights)
            cell.cell_cycle_state   = states[idx][0]
            cell.cell_mitosis_state = states[idx][1]
            cell.current_time_life  = cell._initial_random_time_life()

    return sim
