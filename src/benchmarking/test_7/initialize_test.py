"""Test 7: multi-channel fluorescence with photobleaching.

Same setup as test 6 (DAPI + mCherry, static cells, out of focus).
Photobleaching is enabled: signal decays cumulatively with each exposure.

photobleach_rate controls fractional signal loss per exposure:
  0.001 = slow (barely visible over 100 frames)
  0.005 = moderate (clearly visible over ~20-50 frames)
  0.01  = fast (obvious within ~10 frames)

Channels:
  DAPI    — nucleus — filter SCFP2(434/474)    + LED UV
  mCherry — nucleus — filter mScarlet3(569/582) + LED ORANGE

Simulation area: 1500x1500 µm, 80 normal cells, out of focus.
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

    sim.enable_photobleaching(rate=TEST_CONFIG["photobleach_rate"])

    return sim


TEST_CONFIG = {
    "title": "Multi-channel fluorescence — photobleaching, 80 cells, 1500x1500 µm",
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
    "photobleach_rate": 0.005,
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
