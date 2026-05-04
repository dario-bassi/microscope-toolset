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
    radii_um = rng.uniform(
        TEST_CONFIG["radius_min"],
        TEST_CONFIG["radius_max"],
        size=TEST_CONFIG["n_cells"],
    )

    for i, (cell, r) in enumerate(zip(sim._cells, radii_um)):
        cell.base_r = float(r)
        cell.area0 = float(np.pi * r**2)
        cell.brownian_d = TEST_CONFIG["brownian_d"]
        sim.radii[i] = r

    sim.areas[:] = np.pi * radii_um**2
    return sim
