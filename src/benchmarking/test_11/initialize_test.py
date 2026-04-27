"""Test 11: H&E tissue classification — glandular tissue, low-grade dysplasia.

The agent's goal is to identify and classify the different structural regions
visible in a hematoxylin & eosin stained glandular tissue section:
  - Glandular lumens (clear spaces lined by epithelial cells)
  - Epithelial cell nuclei (dark, hematoxylin-stained)
  - Stroma / extracellular matrix (pink eosin background)
  - Mitotic figures (grade 1: 5-10% mitotic fraction)
  - Lymphocytes (grade 1: ~10% of nuclei)

Grade 1 (low-grade dysplasia) introduces slightly enlarged irregular nuclei,
visible mitotic figures, and occasional lymphocytic infiltrates — making
classification more challenging than normal (grade 0) tissue.

Channel:
  H-and-E — composite brightfield stain — filter Electra1(402/454) + LED CYAN

Simulation area: 700x700 world units, ~200 nuclei, static stained section.
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
    "title": "H&E tissue classification — glandular grade 1, 700x700 world units",
    "backend": "histology",
    "cell_type": "histology",
    "tissue_type": "glandular",
    "grade": 1,
    "n_nuclei": 200,
    "world_size": 700,
    "internal_scale": 4,
    "seed": 42,
    "focal_plane": 0.0,
    "channels": [
        {
            "name": "H-and-E",
            "filter": "Electra1(402/454)",
            "led": "CYAN",
        },
    ],
}
