# TEST_CONFIG is injected from test.yaml by test_runner.py


def create_sim_override():
    from virtual_microscope.sims.cell.sim import ScatteredCellSim

    sim = ScatteredCellSim(
        width=TEST_CONFIG["width"],
        height=TEST_CONFIG["height"],
        n_cells=TEST_CONFIG["n_cells"],
        cell_type=TEST_CONFIG["cell_type"],
        seed=TEST_CONFIG["seed"],
    )
    for cell in sim._cells:
        cell.brownian_d = TEST_CONFIG["brownian_d"]
    return sim
