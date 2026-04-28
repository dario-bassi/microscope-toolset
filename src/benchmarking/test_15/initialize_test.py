"""Test 15: SNR optimisation via exposure adjustment — high-noise fluorescence.

The agent's goal is to find an exposure time that yields acceptable signal-to-noise
ratio for cell detection. The noise model is intentionally aggressive:

  - photon_scale = 1.5  (default 5.0) — heavy shot noise at short exposures
  - read_std     = 6.0  (default 2.5) — high read noise floor
  - dark_current = 0.3               — accumulates at long exposures

At low exposure (< 20 ms) cells are barely distinguishable from noise.
At moderate exposure (~100–200 ms) SNR is sufficient for detection.
At very long exposure (> 500 ms) saturation starts to clip bright nuclei.

The agent must adjust exposure, evaluate image quality, and settle on a value
before running its detection workflow.

Channels:
  DAPI    — nucleus  — filter SCFP2(434/474)    + LED UV
  mCherry — nucleus  — filter mScarlet3(569/582) + LED ORANGE

Simulation area: 1500x1500 µm, 50 static normal cells, in focus.
"""

import numpy as np


def create_sim_override():
    from virtual_microscope.sims.cell.sim import ScatteredCellSim
    from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline

    sim = ScatteredCellSim(
        width=TEST_CONFIG["width"],
        height=TEST_CONFIG["height"],
        n_cells=TEST_CONFIG["n_cells"],
        cell_type=TEST_CONFIG["cell_type"],
        seed=TEST_CONFIG["seed"],
    )

    for cell in sim._cells:
        cell.brownian_d = TEST_CONFIG["brownian_d"]

    # Replace fluorescence pipelines with high-noise versions to make
    # the SNR vs. exposure relationship dramatic and clearly observable.
    noisy_cfg = {
        "photon_scale": 1.5,    # heavy shot noise (default 5.0)
        "read_std":     6.0,    # high read noise floor (default 2.5)
        "dark_current": 0.3,    # dark current accumulates at long exposures
        "banding_std":  1.0,    # sCMOS-like row banding artefact
    }
    sim._pipeline[1] = OpticalPipeline(psf_sigma=0.6, noise=noisy_cfg)
    sim._pipeline[2] = OpticalPipeline(psf_sigma=0.6, noise=noisy_cfg)

    return sim


TEST_CONFIG = {
    "title": "SNR optimisation — high-noise fluorescence, 50 cells, 1500x1500 µm",
    "backend": "particle",
    "cell_type": "normal",
    "n_cells": 50,
    "seed": 22,
    "width": 1500,
    "height": 1500,
    "brownian_d": 0.0,
    "focal_plane": 0.0,
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
