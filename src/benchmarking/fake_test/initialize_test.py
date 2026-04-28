"""Fake test: system integration check.

Channels:
  DAPI    — nucleus    — filter SCFP2(434/474)  + LED UV
  mScarlet — membrane  — filter mScarlet3(569/582) + LED ORANGE

Task: snap both channels at the starting position, move the XY stage to
(200, 100) µm, snap both channels again, save all images to a folder,
and display them in the napari GUI.
"""


def create_sim_override():
    from virtual_microscope.sims.cell.sim import ScatteredCellSim
    return ScatteredCellSim(
        n_cells=TEST_CONFIG["n_cells"],
        seed=TEST_CONFIG["seed"],
        cell_type=TEST_CONFIG["cell_type"],
    )


TEST_CONFIG = {
    "title": "Fake test — multi-channel imaging + stage move",
    "backend": "particle",
    "cell_type": "normal",
    "n_cells": 50,
    "seed": 7,
    "focal_plane": 0.0,
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
