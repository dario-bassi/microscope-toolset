"""Test 12: H&E tissue classification — glandular tissue, normal (grade 0), 40x objective.

The agent starts with the objective already set to 40x, giving nuclear-level
detail from the first image. The tissue is normal (grade 0): well-formed glands,
uniform nuclei, minimal mitotic figures, no necrosis.

At 40x the agent sees:
  - Individual nuclei with chromatin texture
  - Glandular lumen boundaries
  - Stromal collagen fibres between glands
  - Basement membrane details

The challenge vs test 11: the agent must interpret high-magnification patches
and infer the broader tissue architecture from limited FOV.

Channel:
  H-and-E — composite brightfield stain — filter Electra1(402/454) + LED CYAN

Simulation area: 512x512 world units, ~200 nuclei, static stained section.
Objective: 40x (set via cfg property at initialisation).
"""


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


TEST_CONFIG = {
    "title": "H&E tissue classification — glandular grade 0, 512x512, 40x objective",
    "backend": "histology",
    "cell_type": "histology",
    "tissue_type": "glandular",
    "grade": 0,
    "n_nuclei": 200,
    "world_size": 512,
    "internal_scale": 4,
    "seed": 17,
    "focal_plane": 0.0,
    "initial_properties": [
        ("Objective", "Label", "40x"),
    ],
    "channels": [
        {
            "name": "H-and-E",
            "filter": "Electra1(402/454)",
            "led": "CYAN",
        },
    ],
}
