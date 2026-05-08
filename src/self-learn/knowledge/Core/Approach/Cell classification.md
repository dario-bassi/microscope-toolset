# Vision Prompts: Cell State Classification

Use these prompts with the LLM vision API when classifying individual cells from microscopy images.

## General Cell State

Save a crop of the cell of interest, then use this prompt:

> Examine this brightfield microscopy image of a single cell. Classify its state as one of:
>
> - **Healthy/Interphase**: Round shape, clear dark oval nucleus, smooth membrane
> - **Mitotic**: Condensed chromatin visible, cell may be rounded up
> - **Early Apoptotic**: Slightly irregular shape, dimmer nucleus
> - **Late Apoptotic (blebbing)**: Highly irregular lumpy membrane, fragmented nucleus
> - **Apoptotic (shrinkage)**: Smooth but smaller than neighbors, slightly irregular
>
> Key distinction: Shrinkage = smooth+small. Blebbing = lumpy+irregular. These are different.
>
> Report: classification, confidence (high/medium/low), and what visual features you based this on.

## Mitotic Stage Identification

> Examine this cell at 40x magnification. Identify the mitotic stage:
>
> - **Interphase (G1/S/G2)**: Round cell, clear dark oval nucleus, no condensed chromosomes. G1/S/G2 are indistinguishable by morphology.
> - **Prophase**: Scattered dark X-shaped structures (condensed chromosomes) throughout the cell. No nucleus border. In fluorescence: diffuse scattered bright spots (NOT two separated clusters).
> - **Metaphase**: Clear horizontal dark band (chromosomes aligned at metaphase plate). In fluorescence: bright horizontal bar.
> - **Anaphase**: Two distinct polar groups of chromosomes separating. In fluorescence: TWO bright clusters (NOT scattered like prophase). CRITICAL: Prophase = ONE dispersed; Anaphase = TWO separated.
> - **Telophase**: Two large dark masses (reforming daughter nuclei), strong dumbbell pinching. Masses are solid blobs, not X-shaped.
> - **Cytokinesis**: Near-complete separation, very thin bridge between daughter cells.
>
> IMPORTANT: At 10x, anaphase can look like "slightly unusual" interphase. Always inspect at 40x for reliable late-mitotic staging.

## Multi-Channel Cell State

> I have three views of the same cell:
> 1. Brightfield (shape/morphology)
> 2. Nucleus channel (chromatin state)
> 3. Membrane channel (boundary integrity)
>
> Using ALL three channels, classify this cell. Focus on:
> - Shape: round (healthy), irregular (apoptotic), rounded-up (mitotic)
> - Chromatin: diffuse (interphase), condensed bright (mitotic), fragmented (late apoptotic)
> - Membrane: smooth (healthy), blebbed (apoptotic), constricted (dividing)

## Sample Type Identification

> Look at this microscopy image and identify the sample type:
> - Blood smear (RBCs, possibly parasites)
> - Cell culture (sparse/confluent cells on flat substrate)
> - Tissue section (organized cellular architecture)
> - Yeast (small round cells, possibly with buds)
> - Plant tissue (rectangular cells with rigid walls)
> - Neurons (large somata with branching dendrites)
>
> Also note: imaging modality (brightfield, fluorescence, phase contrast), approximate magnification, and any notable features.
