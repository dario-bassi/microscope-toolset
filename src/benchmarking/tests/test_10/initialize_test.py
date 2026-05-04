# TEST_CONFIG is injected from test.yaml by test_runner.py

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
    g2_indices = rng.choice(len(sim._cells), size=2, replace=False)

    for i, cell in enumerate(sim._cells):
        cell.is_dying = False
        cell.n_div    = 0
        cell.brownian_d = TEST_CONFIG["brownian_d"]

        if i in g2_indices:
            cell.cell_cycle_state   = "G2"
            cell.cell_mitosis_state = "Interphase"
            cell.current_time_life  = 18.0
        else:
            cell.cell_cycle_state   = "G1" if rng.random() < 0.7 else "S"
            cell.cell_mitosis_state = "Interphase"
            cell.current_time_life  = cell._initial_random_time_life()

    return sim
