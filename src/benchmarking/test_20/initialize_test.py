"""Test 20: out-of-focus brightfield start — same as test 1 but brightfield.

The focal plane starts 50 µm above the tissue plane (tissue_z=0.0), which is
well outside the 10x DOF of 6 µm, so the image is visibly blurred at launch.
The agent must move the Z stage toward 0 µm to bring cells into focus.

Channel:
  DIC — brightfield body outline — filter Electra1(402/454) + LED CYAN
"""


def create_sim_override():
    from virtual_microscope.sims.cell.sim import ScatteredCellSim
    return ScatteredCellSim(
        n_cells=TEST_CONFIG["n_cells"],
        seed=TEST_CONFIG["seed"],
        cell_type=TEST_CONFIG["cell_type"],
    )


TEST_CONFIG = {
    "title": "Out-of-focus brightfield start — DIC, normal cells",
    "backend": "particle",
    "cell_type": "normal",
    "n_cells": 50,
    "seed": 0,
    "focal_plane": 0.0,
    "initial_properties": [
        ("ZStage", "Position", "50.0"),   # start 50 µm out of focus
    ],
    "channels": [
        {
            "name": "DIC",
            "filter": "Electra1(402/454)",
            "led": "CYAN",
        },
    ],
}
