# Papers index

48 papers organized by topic. Each entry links to the full file (abstract + implementation notes + cited-by back-links).

---

## Event-driven & adaptive acquisition
*Smart triggering: deciding when, where, and what to image.*

| Citation | One-line description |
|---|---|
| [[Papers/Mahecic 2022]] | FAST: neural-net event detector triggers STED acquisition on demand |
| [[Papers/Stepp 2026]] | Hybrid-EDA: BF surveillance + neural-net trigger → fluorescence only on rare events |
| [[Papers/Ye 2025]] | Reconstruction-uncertainty rescan: image only where the model is uncertain |
| [[Papers/Kandel 2023]] | Expected-Reduction-in-Distortion scan-point selection (active learning) |
| [[Papers/Fox 2022]] | MicroMator: Python event-condition-action rules for reactive experiments |
| [[Papers/Chiron 2022]] | CyberSco.Py: YAML rulebook front-end for the observe→compute→modify loop |
| [[Papers/Alvelid 2022]] | Event-driven STED: one spot per detected biological event |
| [[Papers/André 2023]] | Data-driven context-specific acquisition of high-fidelity images |
| [[Papers/Jackson 2009]] | Intelligent acquisition and learning of fluorescence data models (early work) |
| [[Papers/Ibrahim 2025]] | Self-driving microscopy detects protein aggregation onset + Brillouin imaging |
| [[Papers/Durand 2018]] | Bandit optimiser over STED illumination parameters (resolution vs photodamage) |
| [[Papers/Komatsuzaki 2022]] | UCB multi-armed bandit for discrimination-budgeted Raman spatial sampling |
| [[Papers/He 2020]] | Image-quality-guided smart rotation for improved coverage in microscopy |

---

## Closed-loop optogenetics
*Feedback control of cell biology via patterned light.*

| Citation | One-line description |
|---|---|
| [[Papers/Passmore 2025]] | Outcome-driven microscopy: SLM steers cell migration and N/C transport to a target |
| [[Papers/Lugagne 2024]] | Deep MPC: learned forward model drives arbitrary gene-expression trajectories in thousands of single cells |
| [[Papers/Rullan 2018]] | Earliest per-cell integral feedback controller (DMD + PP7 nascent-RNA reporter, yeast) |
| [[Papers/Hinderling 2025]] | FARO: open-source Pycro-Manager + DMD substrate for per-cell optogenetic control |

---

## Autofocus & adaptive optics
*Correcting Z drift and optical aberrations in real time.*

| Citation | One-line description |
|---|---|
| [[Papers/Pinkard 2019]] | Hardware-free autofocus via off-axis LED array illumination |
| [[Papers/Hu 2023]] | Zernike coefficient regression: single-shot AO from a neural network |
| [[Papers/Schmidt 2023]] | RL (recurrent PPO) for hysteretic piezo-mirror AO — history-dependent actuator |
| [[Papers/Royer 2016]] | AutoPilot: 5-axis coordinate-descent alignment for light-sheet microscopy |
| [[Papers/Zhang 2023]] | Deep learning adaptive optics for single-molecule localization microscopy |

---

## Deep learning for imaging
*Restoration, super-resolution, and ML-guided acquisition quality.*

| Citation | One-line description |
|---|---|
| [[Papers/Weigert 2018]] | CARE: content-aware denoising at 60× lower dose via deep-learning prior |
| [[Papers/Jin 2020]] | DL-SIM: 5× fewer raw SIM frames and 100× lower photon counts per frame |
| [[Papers/Bilodeau 2024]] | RL policy for STED resolution-vs-photodamage trade-off learned in simulation |
| [[Papers/Bouchard 2023]] | Task-assisted GAN for resolution enhancement to guide nanoscopy analysis and acquisition |
| [[Papers/Meirovitch 2026]] | SmartEM: machine learning-guided electron microscopy |

---

## LLM & AI agents in science
*Language models and agentic loops for experiment automation.*

| Citation | One-line description |
|---|---|
| [[Papers/Boiko 2023]] | Coscientist: LLM-driven chemistry automation via tool calls |
| [[Papers/Mandal 2025]] | AILA + AFMBench: three failure modes of LLM agents (capability gap, sleepwalking, prompt fragility) |
| [[Papers/Liang 2023]] | LLM compiles natural-language goals into hierarchical Python programs for robotics |
| [[Papers/Kesavan 2025]] | Multi-agent factored framework: five specialized roles for LLM-driven microscopy |
| [[Papers/Pathak 2017]] | Curiosity-driven intrinsic motivation RL — the paradigm for novelty-driven acquisition |

---

## Smart microscopy substrates & frameworks
*Software platforms and hardware-abstraction layers.*

| Citation | One-line description |
|---|---|
| [[Papers/Pinkard 2021]] | Pycro-Manager: Python-native programmable acquisition (events + hooks + image processors) |
| [[Papers/Casas Moreno 2021, Roos 2024]] | ImSwitch (single-process GUI) + Arkitekt (streaming multi-service orchestration) |
| [[Papers/Marin 2024]] | Navigate: GUI-editable feature graphs for no-code decision routing on light-sheet hardware |
| [[Papers/Friederich 2025]] | EAP4EMSIG: production-scale event-driven middleware for 24/7 microfluidic imaging |
| [[Papers/Edelstein 2010]] | µManager Core: C++ device-abstraction layer underlying all pymmcore-plus backends |
| [[Papers/Tosi 2021]] | AutoScanJ: ImageJ script suite for intelligent microscopy automation |

---

## Sample classification & phenotyping
*Automated identification of sample type and biological state.*

| Citation | One-line description |
|---|---|
| [[Papers/Conrad 2011]] | Hand-crafted phenotype classifier — the floor of the classification axis |
| [[Papers/Morgado 2024]] | Task-driven microscopy taxonomy: re-targetable encoder vs per-task detector |
| [[Papers/Shi 2024]] | SmartLLSM: YOLOv5 low-dose scout escalates rare events to 3D lattice light-sheet |
| [[Papers/Yu 2024]] | PLIP: pathology vision-language foundation model for sample-class encoding |

---

## Multi-scale & live cell imaging
*Adaptive 4D imaging, intravital tracking, and long timelapses.*

| Citation | One-line description |
|---|---|
| [[Papers/McDole 2018]] | Adaptive whole-embryo imaging: content-aware re-alignment between time-points |
| [[Papers/Daetwyler 2025]] | Self-driving multiscale microscopy from whole organism to subcellular scale |
| [[Papers/Vladymyrov 2020]] | VivoFollow 2: distortion-free multiphoton intravital imaging |
| [[Papers/Rabut 2004]] | Automatic real-time 3D cell tracking by fluorescence microscopy |

---

## Multiplexing & multi-channel
*Sequential labelling, DNA-PAINT, and spectral unmixing.*

| Citation | One-line description |
|---|---|
| [[Papers/Almada 2019]] | Multiplexed labelling rounds extending the channel axis beyond spectrally separable fluorophores |
| [[Papers/Lutz 2018, Rames 2023]] | PRIME-PAINT and Quencher-Exchange-PAINT: controller-driven DNA-PAINT multiplexing |
