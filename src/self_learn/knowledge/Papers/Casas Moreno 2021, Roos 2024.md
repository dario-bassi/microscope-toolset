---
title: "Two open-source smart-acquisition substrates: ImSwitch (control + reconstruction in one process) and Arkitekt (streaming orchestration across processes)"
authors: |
  Casas Moreno, Al-Kadhimi, Alvelid, Bodén, Testa (ImSwitch JOSS 2021);
  Casas Moreno, Mendes Silva, Roos, Pennacchietti, Norlin, Testa (ImSwitch+napari HardwareX 2023);
  Roos, Bancelin, Delaire, Wilhelmi, Levet, Engelhardt, Viasnoff, Galland, Nägerl, Sibarita (Arkitekt Nat Methods 2024)
year: 2021, 2023, 2024
venue: |
  J Open Source Softw 6(64):3394 (ImSwitch);
  HardwareX 13:e00400 (ImSwitch+napari);
  Nature Methods 21(10):1884–1894 (Arkitekt)
doi: |
  10.21105/joss.03394 (ImSwitch);
  10.1016/j.ohx.2023.e00400 (ImSwitch+napari);
  10.1038/s41592-024-02404-5 (Arkitekt)
url: |
  https://joss.theoj.org/papers/10.21105/joss.03394;
  https://www.hardware-x.com/article/S2468-0672(23)00007-X/fulltext;
  https://www.nature.com/articles/s41592-024-02404-5
researched: 2026-04-25
---

## Abstracts

**ImSwitch (Casas Moreno et al. 2021, JOSS):** ImSwitch is a modular, hardware-agnostic Python microscope-control framework. It abstracts each device class behind a manager interface, loads instrument layout from a JSON setup file, and provides a pluggable Qt-widget GUI (`imcontrol`) plus a separate online-reconstruction pipeline (`imreconstruct`). Designed initially for super-resolution platforms (RESOLFT and parallelised RESOLFT / MoNaLISA), ImSwitch positions itself as a smart-microscopy substrate where acquisition and reconstruction live in one Python process so feedback rules can fire on reconstructed-image features rather than raw frames.

**ImSwitch + napari (Casas Moreno et al. 2023, HardwareX, e00400):** *"We present a computational framework to simultaneously perform image acquisition, reconstruction, and analysis in the context of open-source microscopy automation. The setup features multiple computer units intersecting software with hardware devices and achieves automation using python scripts. In practice, script files are executed in the acquisition computer and can perform any experiment by modifying the state of the hardware devices and accessing experimental data. The presented framework achieves concurrency by using multiple instances of ImSwitch and napari working simultaneously."* The 2023 paper formalises a three-unit architecture (acquisition `imcontrol` ↔ reconstruction `imreconstruct` ↔ analysis/orchestration `napari` with the `napari-file-watcher` plugin), with a shared filesystem (Zarr / OME-Zarr / TIFF / HDF5) as the synchronisation primitive — no central server required. Demonstrated on cyclic time-lapse of mitochondrial dynamics in U2OS cells (2×2 tiles, 10 timepoints) and selective tiling of neurons + U2OS cells across 5×5 acquisitions to a 160 × 160 µm² FOV.

**Arkitekt (Roos et al. 2024, Nature Methods):** *"Quantitative microscopy workflows have evolved dramatically over the past years, progressively becoming more complex with the emergence of deep learning. Long-standing challenges such as three-dimensional segmentation of complex microscopy data can finally be addressed, and new imaging modalities are breaking records in both resolution and acquisition speed, generating gigabytes if not terabytes of data per day. With this shift in bioimage workflows comes an increasing need for efficient orchestration and data management, necessitating multitool interoperability and the ability to span dedicated computing resources. However, existing solutions are still limited in their flexibility and scalability and are usually restricted to offline analysis. Here we introduce Arkitekt, an open-source middleman between users and bioimage apps that enables complex quantitative microscopy workflows in real time. It allows the orchestration of popular bioimage software locally or remotely in a reliable and efficient manner. It includes visualization and analysis modules, but also mechanisms to execute source code and pilot acquisition software, making 'smart microscopy' a reality."* The flagship smart-microscopy demo: low-mag (20×) survey on a Nikon Ti2-E driven via Micro-Manager → Stardist nuclei segmentation (Segmentor plugin) → DBSCAN cluster detection (Reaktor plugin) → automated re-imaging at 40× / 3D on each cluster. Arkitekt runs as a remote service (Ubuntu + CUDA 11) bridging the acquisition computer and any number of analysis tools.

## Smart microscopy principle

ImSwitch and Arkitekt occupy two distinct rungs of the open-source smart-acquisition stack — they are **complementary substrate-class papers**, not competing ones, and together they bracket the design space between [[Papers/Edelstein 2010]] (device drivers) and [[Papers/Pinkard 2021]] (Python-native acquisition loops).

**ImSwitch is the in-process substrate.** Acquisition and reconstruction live in the same Python process tree (`imcontrol` for hardware, `imreconstruct` for the online pipeline), and the 2023 follow-up adds `napari` as an orchestration plane via filesystem-watching. The architectural commitment is *concurrency through decoupling*: reconstruction time on super-resolution modalities (RESOLFT, MoNaLISA, MINFLUX-class) is structurally longer than acquisition time, so a sequential acquire-then-reconstruct pipeline forces the operator to either drop frames or wait. ImSwitch+napari runs all three units in parallel against a shared filesystem, and any of them can be relocated to a separate computer over a network share without changing the code. The smart-microscopy claim: feedback fires on *reconstructed-image features*, not raw frames — which matters precisely when the contrast carrying the biological event (super-resolved structure, MINFLUX localisation density, RESOLFT contrast) is invisible until reconstruction completes. The 2023 paper demonstrates the substrate but stops short of automated feature-based feedback in its biological demos (manual tile annotation), leaving the closed-loop demonstrations to downstream papers built on top.

**Arkitekt is the cross-process orchestrator.** Acquisition software (Micro-Manager, ImSwitch, vendor stacks), analysis tools (napari, ImageJ, CARE, Stardist, custom code), and visualisation can each run as a separate service exposing typed input/output contracts; Arkitekt is the message bus and scheduler between them. The architectural commitment is *integration through explicit data-flow contracts*: each plugin declares its inputs and outputs, Arkitekt routes streaming data to the right service (local workstation for fast steps, GPU cluster for deep-learning steps), and the same closed-loop logic re-targets across instruments by swapping out the acquisition adapter. The flagship demo closes the loop end-to-end on a Micro-Manager-driven Nikon stand: Stardist nuclei segmentation + DBSCAN clustering on the survey scan triggers high-mag re-imaging at the cluster centroids — same survey→detect→zoom paradigm as [[Papers/Conrad 2011]] / [[Papers/Tosi 2021]] / [[Papers/Yu 2024]], but with the detector explicitly a re-targetable foundation-model-class plugin (Stardist) running on a remote GPU host while acquisition runs on the microscope PC.

The two papers split the question "where does the loop close?" into two answers:

- *In one process* (ImSwitch+napari): everything that has to share microsecond-scale acquisition state lives in one Python tree; the filesystem mediates the slower hand-off to analysis tools. Tight coupling between acquisition and reconstruction is the point — feasible for super-resolution where reconstruction is a hard prerequisite for the trigger.
- *Across processes* (Arkitekt): everything that benefits from independent compute (GPU segmentation, cluster ML, multi-tool composition) lives behind explicit contracts; the scheduler does the routing. Loose coupling is the point — necessary when the analysis is a deep-learning model that wants to run on a separate machine but stay in the closed loop.

For smart-microscopy practice the transferable distinction is that *substrate choice depends on the trigger latency budget and the analysis cost shape*. A trigger that needs reconstructed super-resolution contrast wants ImSwitch's in-process model. A trigger that needs a multi-step deep-learning pipeline against a GPU cluster wants Arkitekt's cross-process model. Most published closed-loop systems are mixtures: [[Papers/Hinderling 2025]] uses Pycro-Manager + a single-machine Convpaint/Cellpose pipeline; [[Papers/Friederich 2025]] uses an in-process MLP autofocus + a pluggable segmentation tier; [[Papers/Daetwyler 2025]] runs custom Python tied to dedicated hardware. The substrate question is "which of these am I?", not "which substrate is best?".

Both papers' central design idea — *expose acquisition and analysis as composable, remappable units rather than as a hand-coded loop* — is the answer the open-source community has converged on to the interoperability problem identified in the 2026 smart-microscopy roadmap (Hinderling/Heil/Rates et al. 2026 *Methods in Microscopy*, doi:10.1515/mim-2025-0029). Pycro-Manager ([[Papers/Pinkard 2021]]) provides the Python streaming surface; ImSwitch adds in-process online reconstruction; Arkitekt adds cross-process service orchestration. Together they form the substrate layer below the smart-acquisition demos elsewhere in this library.

## Implementation on pymmcore-plus

A pymmcore-plus codebase already implements *Pycro-Manager-class* abstractions natively — `useq.MDAEvent` events, `MDAEngine` hooks, and `on_frame` image processors map onto the Pycro-Manager primitives one-to-one (see [[Papers/Pinkard 2021]] for the table). The ImSwitch and Arkitekt patterns are **complementary architectural overlays** on top of the same MDA programming model, addressing two scaling axes that a bare pymmcore-plus loop hits eventually.

### Pattern 1 — ImSwitch-style in-process concurrency

The ImSwitch+napari trick is to decouple acquisition, reconstruction, and analysis into long-running cooperating components synchronised through the filesystem. On pymmcore-plus this looks like a `run_events` loop in one thread, a reconstruction worker in another (consuming acquired frames from a queue or a Zarr writer), and a napari viewer with a file watcher in a third — typically a third *process* if the reconstruction is GPU-heavy.

```python
import threading
from pathlib import Path
from useq import MDAEvent
from src.core.hardware.core import run_events

OUT = Path("/tmp/run_001")          # shared filesystem (could be a network share)
OUT.mkdir(parents=True, exist_ok=True)

def acquisition_thread(core):
    """ImSwitch imcontrol equivalent: drives hardware, writes raw frames."""
    def gen():
        for i in range(200):
            yield MDAEvent(channel={"config": "GFP"}, index={"t": i})

    def write_raw(img, event, meta=None):
        # Zarr / OME-Zarr / HDF5 — anything the reconstruction watcher understands
        save_chunk(OUT / "raw", img, event)

    run_events(core, gen(), on_frame=write_raw)

def reconstruction_thread():
    """ImSwitch imreconstruct equivalent: watches OUT/raw, writes OUT/recon.

    The reconstruction can be slow (super-resolution, deconvolution); the
    acquisition keeps producing frames in parallel.
    """
    for chunk_path in watch_folder(OUT / "raw"):
        recon = reconstruct(chunk_path)             # slow: 30–50 s per tile in the paper
        save(OUT / "recon" / chunk_path.name, recon)

threading.Thread(target=acquisition_thread, args=(core,), daemon=True).start()
threading.Thread(target=reconstruction_thread, daemon=True).start()
# napari with napari-file-watcher on OUT/recon for visualisation + manual annotation
```

The synchronisation primitive is the filesystem; the contract between units is the file format (Zarr, OME-Zarr, TIFF, HDF5). The same code runs single-machine (multithread) or multi-machine (network share) without modification — this is the core ergonomic ImSwitch+napari adds over a vanilla pymmcore-plus loop.

### Pattern 2 — Arkitekt-style cross-process orchestration

The Arkitekt trick is to wrap each computational step (segmentation, clustering, acquisition) as a service with declared inputs/outputs and let a scheduler route streaming data between them. On pymmcore-plus the closest off-the-shelf realisation is to expose the acquisition tier itself as an HTTP / RPC service (or use Arkitekt's MicroManager bridge directly) and call out to remote analysis services from the `on_frame` callback.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
import httpx                                # any RPC transport works

SEG_URL = "http://gpu-cluster.lab/stardist/segment"
CLUSTER_URL = "http://gpu-cluster.lab/dbscan/cluster"

def survey_then_zoom(core):
    """Arkitekt-style closed loop: survey on the local microscope, segment +
    cluster on a GPU cluster, zoom back on the local microscope."""

    targets = []

    def on_survey_frame(img, event, meta=None):
        nuclei = httpx.post(SEG_URL, files={"img": encode(img)}).json()    # Stardist
        clusters = httpx.post(CLUSTER_URL, json={"nuclei": nuclei}).json() # DBSCAN
        for c in clusters:
            wx, wy = pixel_to_world(c["centroid"], event)
            targets.append({"x": wx, "y": wy})

    # 20× survey
    survey = [MDAEvent(channel={"config": "Hoechst"}, x_pos=x, y_pos=y)
              for (x, y) in tile_grid(...)]
    run_events(core, survey, on_frame=on_survey_frame)

    # 40× zoom on each detected cluster
    set_objective(core, 40)
    zoom = [MDAEvent(channel={"config": ch}, x_pos=t["x"], y_pos=t["y"],
                     z_plan={"range": 10, "step": 0.5})
            for t in targets for ch in ("Hoechst", "GFP", "RFP")]
    run_events(core, zoom)
```

The contract is the JSON schema of each remote endpoint; the data-flow is HTTP (or gRPC, ZeroMQ, etc.). The same `survey_then_zoom` works against any acquisition adapter that exposes a Pycro-Manager-class surface; swapping the SEG/CLUSTER URLs to local endpoints turns the same code into a single-machine pipeline.

### Production hooks in `src/core/`

- `../../../src/core/hardware/core.py` — `run_events(core, events, on_frame=…)` is the dispatcher in both patterns; the surrounding architectural overlay (threads + filesystem watcher; HTTP services + scheduler) is what the two papers add.
- `../../../src/core/workflows/scanning.py`, `../../../src/core/workflows/adaptive.py` — `scan_and_detect`, `survey_cells`, `rank_by_feature`, `zoom_and_measure`, `pixel_to_world` are the exact primitives the Arkitekt smart-microscopy demo composes (Stardist nuclei + DBSCAN cluster + zoom). Wrapping each as an HTTP endpoint converts the local pipeline into an Arkitekt-style distributed one.
- `../../../src/core/workflows/engine.py` — `MicroscopyEngine` (custom MDA actions) is the analogue of ImSwitch's manager interfaces for non-`MDAEvent` operations (objective swaps, autofocus, log writes).
- The dedicated **online-reconstruction tier** is *not* in `src/core/` — pymmcore-plus is acquisition-and-detection oriented, not reconstruction oriented. For super-resolution-trigger workflows, ImSwitch's `imreconstruct` (or a separate Python process running e.g. `pyMINFLUX`, `napari-stardist`, or a custom CARE pipeline) is the missing layer; the integration pattern is filesystem-watching, not in-process call.

Differences from the papers to keep in mind:
- ImSwitch+napari demonstrates the substrate but the published biological demos use *manual* tile annotation rather than automated feature-based feedback. Closing the loop on reconstructed-image features is the explicit forward-looking claim of the paper, not an implemented demo.
- Arkitekt's flagship smart-microscopy workflow uses Stardist (a foundation-model-class detector) as the trigger; the paper's contribution is the *plumbing* that lets a remote GPU service participate in the acquisition loop, not the detector itself. The loop logic (survey → cluster → zoom) is the same as Conrad 2011 Micropilot.
- Both papers' demos are on widefield / RESOLFT / MoNaLISA; transferring the patterns to point-scan modalities (multiphoton, STED, MINFLUX) or to ultra-fast cameras adds latency-budget questions the substrates themselves don't pre-decide.

## Cited by

- [[Core/Concepts/MDA standard]] — ImSwitch and Arkitekt sit one architectural layer above the MDA programming model: the MDA standard says *what* an acquisition event is; these papers describe *how* to compose acquisition events with online reconstruction (ImSwitch) and remote analysis services (Arkitekt) into a closed loop.
- [[Core/Strategies/Adaptive acquisition]] — together with [[Papers/Edelstein 2010]] (drivers), [[Papers/Pinkard 2021]] (Python streaming acquisition), and [[Papers/Tosi 2021]] (ImageJ-front-end closed loop), ImSwitch and Arkitekt complete the open-source substrate stack on which the closed-loop, event-driven, and AI-guided demos elsewhere in this library run.
