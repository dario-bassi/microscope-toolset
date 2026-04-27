"""Test 1: particle backend, normal cells, out-of-focus fluorescence start.

Channels:
  DAPI    — nucleus    — filter SCFP2(434/474)  + LED UV
  mCherry — membrane   — filter mScarlet3(569/582) + LED ORANGE

The focal plane starts 10 µm above the tissue plane (tissue_z=0.0), which is
well outside the 10x DOF of 6 µm, so the image is visibly blurred at launch.
The agent must move the Z stage toward 0 µm to bring cells into focus.
"""

def create_sim_override():
    """Particle's create_sim() doesn't forward cell_type, so we instantiate directly."""
    from virtual_microscope.sims.cell.sim import ScatteredCellSim
    return ScatteredCellSim(
        n_cells=TEST_CONFIG["n_cells"],
        seed=TEST_CONFIG["seed"],
        cell_type=TEST_CONFIG["cell_type"],
    )


TEST_CONFIG = {
    "title": "Out-of-focus fluorescence start — DAPI + mCherry, normal cells",
    "backend": "particle",
    "cell_type": "normal",
    "n_cells": 50,
    "seed": 0,
    "focal_plane": 0.0,
    "initial_properties": [
        ("ZStage", "Position", "10.0"),   # start 10 µm out of focus; GUI reflects this
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
