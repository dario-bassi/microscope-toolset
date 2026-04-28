"""Test 17: organoid Z-stack acquisition — 3D reconstruction.

The agent's goal is to acquire a complete Z-stack through a 3D intestinal
organoid and assemble a volumetric image. The organoid has 4 crypt-like buds
so the structure is not a perfect sphere, making the stack more challenging
to interpret.

The organoid is a hollow sphere (outer_radius=120 px, wall_thickness=20 px).
The valid Z range is approximately ±120 px around the equatorial plane (z=0):
  - z = 0         : equatorial section — annular ring with hollow lumen
  - z = ±60       : transition zone — partial ring
  - z = ±100      : near-cap — almost solid disk of cells
  - |z| > 120     : outside specimen — blank field

Suggested acquisition strategy:
  Step through Z from -120 to +120 in small increments (e.g. 5 px steps →
  ~48 slices), collecting all three channels at each plane.

Channels:
  DIC         — brightfield body outline  — filter Electra1(402/454) + LED CYAN
  DAPI        — nuclei                    — filter SCFP2(434/474)    + LED UV
  E-cadherin  — cell–cell junctions       — filter TagGFP2(483/506)  + LED GREEN

Organoid: outer_radius=120, wall_thickness=20, 4 buds, seed=68.
"""


def create_sim_override():
    from virtual_microscope.backends.organoid import create_sim

    sim = create_sim(
        outer_radius=TEST_CONFIG["outer_radius"],
        wall_thickness=TEST_CONFIG["wall_thickness"],
        n_cells=TEST_CONFIG["n_cells"],
        n_buds=TEST_CONFIG["n_buds"],
        lumen_opacity=TEST_CONFIG["lumen_opacity"],
        seed=TEST_CONFIG["seed"],
    )

    return sim


TEST_CONFIG = {
    "title": "Organoid Z-stack — 3D reconstruction, 4 buds, outer_r=120",
    "backend": "organoid",
    "cell_type": "organoid",
    "outer_radius": 120,
    "wall_thickness": 20,
    "n_cells": 400,
    "n_buds": 4,
    "lumen_opacity": 0.6,
    "seed": 68,
    "focal_plane": 0.0,
    "channels": [
        {
            "name": "DIC",
            "filter": "Electra1(402/454)",
            "led": "CYAN",
        },
        {
            "name": "DAPI",
            "filter": "SCFP2(434/474)",
            "led": "UV",
        },
        {
            "name": "E-cadherin",
            "filter": "TagGFP2(483/506)",
            "led": "GREEN",
        },
    ],
}
