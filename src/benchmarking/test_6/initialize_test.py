"""Test 6: multi-channel fluorescence — acquire, merge and segment.

Acquire DAPI and mCherry channels separately, merge them, then segment cells.
Cells are static and start out of focus (ZStage = 50.0 µm).

Channels:
  DAPI    — nucleus    — filter SCFP2(434/474)     + LED UV
  mCherry — nucleus    — filter mScarlet3(569/582)  + LED ORANGE

Simulation area: 500x500 µm, 80 normal cells, out of focus.
"""


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
    "title": "Multi-channel fluorescence — acquire, merge and segment, 80 cells, 1500x1500 µm",
    "backend": "particle",
    "cell_type": "normal",
    "n_cells": 80,
    "seed": 42,
    "width": 1500,
    "height": 1500,
    "brownian_d": 0.0,
    "focal_plane": 0.0,
    "initial_properties": [
        ("ZStage", "Position", "50.0"),   # start 50 µm out of focus; GUI reflects this
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
