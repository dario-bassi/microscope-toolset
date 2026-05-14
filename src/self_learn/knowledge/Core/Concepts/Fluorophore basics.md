# Fluorophore basics — why this channel, this exposure, this wavelength

Every fluorescence acquisition involves four physical decisions that can all be made badly: which fluorophore to use, what light to excite it with, what to image through, and how long to leave the shutter open. A minimal mental model of the photophysics saves a lot of grief.

## The excitation → emission loop

A fluorophore absorbs a photon, climbs to an excited electronic state, loses some energy to vibrations (non-radiative), and drops back to the ground state emitting a lower-energy photon. Repeat billions of times per second — that's your signal.

Two spectra characterise every fluorophore:

- **Excitation spectrum**: probability of absorption vs. wavelength. Peak = λ_ex.
- **Emission spectrum**: probability of emission vs. wavelength. Peak = λ_em, usually longer than λ_ex (Stokes shift; 20-50 nm is typical).

In practice you pick an excitation wavelength inside the excitation band (often at or just below λ_ex) and an emission filter centred on λ_em. The fraction of incident photons that actually get absorbed is set by the extinction coefficient (ε, molar), and the fraction of absorptions that emit (rather than quench or convert internally) is the quantum yield (Φ). **Brightness ≈ ε × Φ** — the single scalar that ranks fluorophores for a given excitation.

## Common fluorophores cheat-sheet

| Name | λ_ex / λ_em (nm) | Brightness (ε·Φ) | Notes |
|----|----|----|----|
| DAPI | 358 / 461 | ~20 000 | DNA stain, bright but UV excitation damages cells |
| Hoechst 33342 | 361 / 486 | ~25 000 | Live-cell DNA, less toxic than DAPI |
| FITC / Alexa Fluor 488 | 488 / 520 | ~35 000 / ~73 000 | Workhorse green; AF488 photostable |
| GFP (EGFP) | 488 / 507 | ~26 000 | Protein tag; maturation time >30 min |
| mCherry | 587 / 610 | ~16 000 | Red protein tag |
| Cy5 / Alexa Fluor 647 | 650 / 670 | ~180 000 / ~239 000 | Far-red; best brightness, best penetration |
| BODIPY | varies | ~70 000+ | Lipid droplets; many excitation variants |
| PE (phycoerythrin) | 565 / 578 | huge (~2×10⁶) | Antibody-conjugated flow cytometry; fragile, bleaches fast |

Rule of thumb: for a new experiment where you can pick, prefer far-red over green over UV. Far-red has deeper tissue penetration, less phototoxicity, and lower autofluorescence.

## Four things that kill fluorophores

1. **Photobleaching** — irreversible chemical damage after enough excitation cycles. Manifests as a progressive dimming during a timelapse. Fixable with oxygen scavengers in the mounting medium (live cells: impossible; fixed samples: `ProLong Gold`, `Vectashield`). Rate ∝ excitation intensity × time.

2. **Photoswitching / blinking** — reversible transitions to dark states (triplet, cis-trans). The basis of SMLM but annoying in quantitative imaging. Mitigation: lower intensity + antifade.

3. **Quenching** — neighbouring molecule steals the excited-state energy before it emits. Concentration-dependent (self-quenching at high density), distance-dependent (FRET), or environment-dependent (pH, solvent polarity).

4. **Saturation** — at very high excitation intensity, the fluorophore spends most of its cycle in the excited state and can't absorb more; emission plateaus. Not damaging per se, but wastes light.

## In an acquisition plan

### Pick the channel that matches your biology

- DAPI / Hoechst: count nuclei, segment cells (dense, single dot per cell).
- GFP / FITC: generic protein tag, "is the reporter on?".
- Alexa647 / Cy5: best for **co-localization** with a green channel — spectrally separated, similar brightness.
- Phase or DIC: free (no fluorophore), use for counting whole cells when you can.

When in doubt: always do a **brightfield** pass first. It's non-destructive and often enough to answer the counting part of a challenge. See `[[Core/Approach/How to approach a problem]]`.

### Match the filter to the fluorophore, not by name

Excitation/emission filters are typically written as `WAVELENGTH/BANDWIDTH` (e.g., `470/40` = 450-490 nm passes). A "GFP filter set" usually has ex `470/40`, dichroic `495 LP`, em `525/50`. If your microscope has the wrong filter for your fluorophore the image will look dim and ugly — not because there's no signal, but because most of the emission is blocked. Query ``self_learn.hardware.config`` for the actual filter config if in doubt.

### Exposure scaling per channel

Don't reuse the exposure for one channel on another. Fluorophore brightness varies by 10× across common dyes; a 50 ms exposure that's perfect for DAPI is blown out on Cy5 and far underexposed on mCherry. Per-channel calibration is non-optional. See `[[Core/Strategies/Imaging parameter optimization]]`.

## Multi-channel bleedthrough (spectral crosstalk)

Two fluorophores with overlapping emission spectra will appear in each other's channels. Two rules:

1. **Single-stain controls** — acquire a sample with only one fluorophore, measure its signal in the "wrong" channel. That's the bleedthrough coefficient.
2. **Linear unmixing** — subtract a scalar fraction of channel A from channel B (and vice versa). Works when the spectra are fixed; fails if one fluorophore's environment changes mid-acquisition.

A cheap substitute: pick fluorophores with well-separated bands (DAPI + AF488 + AF568 + AF647 is the textbook 4-colour set — each peak is 50+ nm from the next).

**When single-stain controls aren't available** (e.g. co-stained
samples only), `self_learn.utils.spectral_leak` provides a
spatial-segmentation-as-control proxy: top-percentile mask in
channel A pixels stand in for "pure A signal" when measuring leak
into other channels. It builds the N×N leak matrix and
`unmix(observed, K)` solves `K @ true = observed` per pixel/ROI/cell.

## Related

- `[[Core/Concepts/Exposure and photodamage]]` — phototoxicity scales with cumulative dose across all channels.
- `[[Core/Concepts/SNR and dynamic range]]` — dim fluorophores demand longer exposures → lower temporal resolution.
- `[[Core/Strategies/Imaging parameter optimization]]` — per-channel sweep workflows.
- ``self_learn.analysis.colocalization`` — Pearson / Manders co-localization metrics.
- ``self_learn.analysis.fluorescence`` — bleach correction, background subtraction.

## Further reading

*Orientation pointers, not canonical citations — these don't go through the paper-library verification path.* Search terms: Lichtman & Conchello *Nature Methods* fluorescence-microscopy basics review, Diaspro *Confocal & Two-Photon Microscopy* textbook, practical filter-selection guides. For spectra: the Molecular Probes / ThermoFisher fluorophore-spectra database is the living reference.
