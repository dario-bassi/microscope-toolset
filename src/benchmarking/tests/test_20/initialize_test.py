# TEST_CONFIG is injected from test.yaml by test_runner.py


def create_sim_override():
    from virtual_microscope.sims.cell.sim import ScatteredCellSim

    return ScatteredCellSim(
        n_cells=TEST_CONFIG["n_cells"],
        seed=TEST_CONFIG["seed"],
        cell_type=TEST_CONFIG["cell_type"],
    )
