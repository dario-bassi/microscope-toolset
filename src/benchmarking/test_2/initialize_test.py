"""Test 2: large-area scanning scenario — DAPI + mScarlet, particle backend.

Channels:
  DAPI     — nucleus     — filter SCFP2(434/474)    + LED UV
  mScarlet — nucleus/mem — filter mScarlet3(569/582) + LED ORANGE

Simulation area: 6000x6000 µm (provides ~500 µm margin on each side for a
5000x5000 µm scanning run).  3500 normal cells, nearly static (brownian_d
reduced from the default 60.0 to 2.0).  Continuous simulation, in focus from
the start (focal_plane = 0.0 µm).
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
    "title": "Large-area scanning — DAPI + mScarlet, 3500 cells, 6000x6000 µm",
    "backend": "particle",
    "cell_type": "normal",
    "n_cells": 3500,
    "seed": 42,
    "width": 6000,
    "height": 6000,
    "brownian_d": 2.0,    # near-static; default is 60.0
    "focal_plane": 0.0,   # in focus from the start
    "channels": [
        {
            "name": "DAPI",
            "filter": "SCFP2(434/474)",
            "led": "UV",
        },
        {
            "name": "mScarlet",
            "filter": "mScarlet3(569/582)",
            "led": "ORANGE",
        },
    ],
}
