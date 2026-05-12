---
title: "Content-aware image restoration: pushing the limits of fluorescence microscopy"
authors: Martin Weigert, Uwe Schmidt, Tobias Boothe, Andreas Müller, Alexandr Dibrov, Akanksha Jain, Benjamin Wilhelm, Deborah Schmidt, Coleman Broaddus, Siân Culley, Mauricio Rocha-Martins, Fabián Segovia-Miranda, Caren Norden, Ricardo Henriques, Marino Zerial, Michele Solimena, Jochen Rink, Pavel Tomancak, Loic Royer, Florian Jug, Eugene W. Myers
year: 2018
venue: Nature Methods 15(12):1090-1097
doi: 10.1038/s41592-018-0216-7
url: https://www.nature.com/articles/s41592-018-0216-7
researched: 2026-04-24
---

## Abstract

Fluorescence microscopy is a key driver of discoveries in the life sciences, with observable phenomena being limited by the optics of the microscope, the chemistry of the fluorophores, and the maximum photon exposure tolerated by the sample. These limits necessitate trade-offs between imaging speed, spatial resolution, light exposure, and imaging depth. In this work we show how content-aware image restoration based on deep learning extends the range of biological phenomena observable by microscopy. We demonstrate on eight concrete examples how microscopy images can be restored even if 60-fold fewer photons are used during acquisition, how near isotropic resolution can be achieved with up to tenfold under-sampling along the axial direction, and how tubular and granular structures smaller than the diffraction limit can be resolved at 20-times-higher frame rates compared to state-of-the-art methods. All developed image restoration methods are freely available as open source software in Python, FIJI, and KNIME.

## Smart microscopy principle

CARE replaces a term of the acquisition budget with a term of the analysis budget. Classically the microscope has to pay the full photon cost to reach a target SNR; with a content-aware restoration network trained on paired high/low-SNR images, the microscope acquires at a **fraction** of that photon dose and the network restores the image post-hoc. The contribution is a concrete demonstration that for the right prior (a trained network matched to the sample's statistics), a 60× dose reduction, 10× axial under-sampling, or 20× faster frame rate all recover images that match the full-dose reference.

This reshapes acquisition planning: the dose-SNR trade-off no longer sits only on the microscope side. The knob you used to turn at the objective (exposure, laser power, Z-step) can be dialled down and the quality recovered downstream — provided you have a prior of the sample class you're imaging. It's "content-aware" because the recovery depends on a model of what cells of this type should look like; a naive denoiser would blur structure away.

Worth noting: the paradigm shifts the burden onto training data. A CARE model trained on one sample class won't generalise to another. The acquisition-side savings are real; the upfront cost is the paired-dataset collection and the model training.

## Implementation on pymmcore-plus

CARE is fundamentally an analysis step, not an acquisition step — but its practical value is that it lets you *pick gentler acquisition parameters up front*, confident that the model will restore the quality. So the integration point is: acquire at the aggressive low-dose / high-speed / under-sampled settings the model was trained on, capture the raw frames, run them through the CARE model before downstream analysis.

```python
from useq import MDASequence
from src.core.hardware.core import run_events

# Example: run a long timelapse at 1/60 of the photon dose that would
# normally be needed for acceptable SNR, then restore each frame with a
# CARE model trained on the same sample class at full dose.

low_dose_seq = MDASequence(
    time_plan={"loops": 600, "interval": 2.0},
    channels=[{"config": "GFP", "exposure": 5}],   # ~60x shorter than the 300 ms reference
)

# Assume a CARE model has been trained offline on paired (5 ms, 300 ms) frames.
from csbdeep.models import CARE
care_model = CARE(config=None, name="gfp_lownoise_to_highSNR", basedir="models/")

restored_stack = []
def on_frame(img, event, meta=None):
    restored = care_model.predict(img.astype("float32"), axes="YX")
    restored_stack.append(restored)

run_events(core, list(low_dose_seq), on_frame=on_frame)
```

The pieces to add to `src/core/`:
- A thin wrapper around the `csbdeep` / `CARE` API so the on-frame callback is one function call rather than model-instantiation boilerplate (no such module exists yet — candidate for `src/core/analysis/restoration.py`).
- A training-data collector workflow: acquire paired low-dose and full-dose frames at a handful of positions, save as npz, train the model offline. That's a one-off per sample class.
- Calibration discipline: the "aggressive" acquisition settings are specific to the model. Re-validate SNR and structure recovery on a small held-out prep before a long imaging campaign.

When CARE is *not* the right tool: novel sample types without paired training data, structures smaller than the training-set resolution (the model can't invent what it hasn't seen), or any measurement where pixel values are load-bearing (intensity quantification, colocalization) — restoration changes absolute values even when it preserves structure.

## Cited by

- [[Core/Concepts/Exposure and photodamage]] — the dose-SNR trade-off that CARE re-engineers.
- [[Core/Strategies/Gentle imaging]] — dose-minimisation patterns; CARE is the deep-learning flavour.
