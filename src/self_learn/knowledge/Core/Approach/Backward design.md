# Backward Design

> **TL;DR** — Start from what you need to measure and work backward to acquisition
> parameters. Smallest feature → minimum objective → pixel size → channel → frame count.
> Use 10x for whole-cell counting (cell = ~15 px at 1.0 µm/px), 40x only for
> sub-cellular detail. BF before fluorescence — it is free and non-destructive.
> If a parameter cannot be justified by a downstream analysis need, remove it.

Start from the analysis requirement and work backward to the acquisition
parameters. This prevents over-acquiring (wasting photons and time) and
under-acquiring (missing the data you need).

---

## The Core Principle

**What do I need to measure?** determines everything else.

```
Measurement requirement
  --> Spatial resolution needed
    --> Objective / magnification
      --> Pixel size & FOV
        --> Number of tiles / positions
          --> Channel selection
            --> Exposure & frame rate
              --> Total photon budget
```

Every acquisition parameter should be justified by a downstream analysis need.
If you cannot explain why you need 40x instead of 10x, you probably do not.

---

## Resolution Ladder

| Feature | Typical Size | Min Objective | Pixel Size |
|---------|-------------|---------------|------------|
| Colony on agar plate | 500-2000 um | 4x or macro | 2-4 um/px |
| Whole cells (count) | 10-30 um | 10x | 1.0 um/px |
| Cell morphology | 10-30 um | 20x | 0.5 um/px |
| Nuclear shape | 5-15 um | 20x | 0.5 um/px |
| Dendrite branching | 1-3 um | 40x | 0.25 um/px |
| Focal adhesions | 0.5-2 um | 40x+ | 0.25 um/px |
| Bud scars (yeast) | 0.5-1 um | 40x+ | 0.25 um/px |
| Bacteria (individual) | 1-3 um | 40x-100x | 0.1-0.25 um/px |

**Rule of thumb**: You need at least 3-4 pixels across the smallest feature
you want to resolve (Nyquist-like criterion for segmentation).

### Example: Cell Counting

- Feature: whole cells, ~15 um diameter
- At 10x (1.0 um/px): cell = 15 px across. Plenty for counting.
- At 40x (0.25 um/px): cell = 60 px across. Overkill. FOV shrinks 4x.
- Decision: use 10x. More cells per FOV, faster, less phototoxicity.

### Example: Focal Adhesion Counting

- Feature: FA puncta, ~1 um
- At 10x (1.0 um/px): FA = 1 px. Cannot resolve. Will miss most.
- At 40x (0.25 um/px): FA = 4 px. Barely resolvable. Use this.
- Decision: use 40x. Accept smaller FOV, need membrane channel.

---

## Temporal Design

Work backward from the dynamics you need to capture.

| Process | Timescale | Frame Interval | Typical Frames |
|---------|-----------|----------------|----------------|
| Calcium waves | 0.1-1 s | 0.1-0.5 s | 50-200 |
| Cell migration | 5-30 min | 1-5 min | 20-100 |
| Cell division | 30-90 min | 5-15 min | 20-50 |
| Wound healing | 1-12 h | 15-60 min | 20-50 |
| Colony growth | 6-24 h | 30-60 min | 20-50 |

**Nyquist for time**: sample at least 2x faster than the fastest event you
need to resolve. For tracking, 5-10x is better to avoid identity swaps.

### Example: Wound Healing

- Process: gap closure at ~10-50 um/hour
- At low mag (large FOV): wound crosses FOV in ~10-50 hours
- Frame interval: 1 step = some time unit. Need enough frames to see trend.
- Decision: 20-40 frames, measure gap width each frame, fit linear model.

---

## Channel Selection

Choose channels based on what the analysis needs to see.

| Analysis Need | Best Channel | Reason |
|---------------|-------------|--------|
| Cell boundaries | BF or membrane | Outlines visible |
| Cell count (sparse) | BF | Non-destructive |
| Cell count (dense) | Nucleus | One dot per cell |
| Protein localization | Specific fluorescence | Target-specific |
| Live/dead assay | Calcein + PI | Green=live, Red=dead |
| Sample survey | BF | Free, no bleaching |

**Critical rule**: Always try BF first. It is non-destructive and often
sufficient. Use fluorescence only when BF cannot answer the question.

### Multi-Channel Strategy

When multiple channels are needed:
1. Snap BF first (identify sample, plan analysis)
2. Snap nucleus if you need cell counts in dense tissue
3. Snap specific fluorescence only for targeted measurements
4. Minimize total fluorescence exposure (photobleaching, phototoxicity)

---

## FOV and Tiling

At each objective, the FOV is fixed (query from hardware with `core.getPixelSizeUm()` and `core.getImageWidth()`).
Higher magnification = smaller pixel size = smaller FOV.

If you need to cover a large area at high magnification:
- Survey at 10x to find regions of interest
- Move stage to ROI coordinates
- Switch to 40x for detailed imaging
- Convert coordinates: world coords are objective-independent

---

## Decision Checklist

Before starting acquisition, verify:

- [ ] Objective matches the smallest feature I need to resolve
- [ ] Channels include everything the analysis needs (and nothing extra)
- [ ] Frame count is sufficient for temporal dynamics
- [ ] Frame rate captures the fastest events
- [ ] FOV covers enough cells for statistical significance
- [ ] Total photon budget is within phototoxicity limits
- [ ] Output format matches what the scoring expects

If any box is unchecked, redesign before acquiring.

---

## Anti-Patterns

- **40x for cell counting**: Wastes time, shrinks FOV, no benefit
- **Fluorescence for sample identification**: BF is free, use it first
- **100 frames for a static measurement**: 1-3 frames suffice
- **All channels every frame**: Acquire only what analysis needs per frame
- **Tiling when one FOV suffices**: Check cell density first
