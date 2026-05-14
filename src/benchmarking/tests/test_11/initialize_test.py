# TEST_CONFIG is injected from test.yaml by test_runner.py


def create_sim_override():
    from virtual_microscope.backends.histology import create_sim

    sim = create_sim(
        tissue_type=TEST_CONFIG["tissue_type"],
        world_size=TEST_CONFIG["world_size"],
        n_nuclei=TEST_CONFIG["n_nuclei"],
        grade=TEST_CONFIG["grade"],
        seed=TEST_CONFIG["seed"],
        internal_scale=TEST_CONFIG["internal_scale"],
    )
    return sim
