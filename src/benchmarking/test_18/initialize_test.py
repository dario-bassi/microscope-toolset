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
    n_low, n_mid = 70, 20
    order = rng.permutation(len(sim._cells))

    for rank, cell_idx in enumerate(order):
        cell = sim._cells[cell_idx]
        cell.is_dying = False
        cell.brownian_d = TEST_CONFIG["brownian_d"]

        if rank < n_low:
            cell.n_div = int(rng.integers(0, 6))
        elif rank < n_low + n_mid:
            cell.n_div = int(rng.integers(6, 8))
        else:
            cell.n_div = int(rng.integers(8, 10))

        cell.cell_cycle_state   = rng.choice(["G1", "S", "G2"], p=[0.50, 0.25, 0.25])
        cell.cell_mitosis_state = "Interphase"
        cell.current_time_life  = cell._initial_random_time_life()

    return sim
