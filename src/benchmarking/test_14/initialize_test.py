"""Test 14: large-area fluorescence scan — detect at least 500 cells.

The agent must systematically scan a 5000x5000 µm field, acquire fluorescence
images across multiple stage positions, and count/detect a total of at least
500 out of 600 seeded cells.

The simulation starts out of focus: the ZStage is initialised at 10 µm,
so the agent must first find focus before the counting scan.

Challenges:
  - Large area requires a tiled acquisition strategy
  - Initial out-of-focus state must be corrected before imaging
  - Cell density is moderate (~600 cells over 5000x5000 µm ≈ 24 cells per FOV
    at 10x), so each tile contributes a manageable number of detections

Channels:
  DAPI    — nucleus  — filter SCFP2(434/474)    + LED UV
  mCherry — nucleus  — filter mScarlet3(569/582) + LED ORANGE

Simulation area: 5000x5000 µm, 600 static normal cells.
ZStage starts at 10 µm (out of focus); focal plane is at z = 0.
"""

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

    for cell in sim._cells:
        cell.brownian_d = TEST_CONFIG["brownian_d"]

    return sim


TEST_CONFIG = {
    "title": "Large-area fluorescence scan — detect 500+ cells, 5000x5000 µm",
    "backend": "particle",
    "cell_type": "normal",
    "n_cells": 600,
    "seed": 99,
    "width": 5000,
    "height": 5000,
    "brownian_d": 0.0,
    "focal_plane": 0.0,
    "initial_properties": [
        ("ZStage", "Position", "10.0"),   # start 10 µm out of focus
    ],
    "channels": [
        {
            "name": "DAPI",
            "filter": "SCFP2(434/474)",
            "led": "UV",
        },
        {
            "name": "mCherry",
            "filter": "mScarlet3(569/582)",
            "led": "ORANGE",
        },
    ],
}
