# Failure Analysis: Common Detection & Measurement Mistakes

## Root Causes

### 1. Rigid Detection
Using the same detection approach regardless of sample type. When the default `detect_cells(sigma=2.5)` doesn't work, try alternatives instead of submitting a wrong answer.

**Fix**: Check `knowledge/patterns/detection_strategy.md` for sample-specific methods. Always visually verify detection results before submitting.

### 2. Neuron Branch Points: Voronoi Assignment
Used Voronoi partitioning to assign skeleton junction points to nearest soma. When two neurons had overlapping dendritic trees, junctions from one neuron were assigned to the wrong soma.

**Fix**: Use connected-component assignment. Each skeleton CC correctly traces one neuron's dendritic tree. See `knowledge/playbooks/neurons.md`.

### 3. Malaria Ring Detection: Insensitive Threshold
Used dark_fraction threshold to detect parasitized RBCs. Ring-stage parasites are so small that the dark fraction change is negligible.

**Fix**: Compare each RBC's mean brightness to the population median. Infected RBCs are ~15% dimmer (175 vs 205). See `knowledge/playbooks/malaria.md`.

### 4. ZOI Measurement: 50% Threshold
Used 50% of maximum bacterial density as the ZOI boundary. This overestimated the ZOI by 40% (158 vs GT 113).

**Fix**: Use steepest gradient (maximum derivative) of the radial density profile. See `knowledge/playbooks/disk_diffusion.md`.

### 5. Yeast Bud Scars: Global Threshold
Used a global intensity threshold to detect bud scars. Background wall fluorescence varies between cells, causing massive false positives. ~50% of yeast are virgin cells with NO scars.

**Fix**: Per-cell relative thresholding. Compare each punctum to its own cell's wall baseline brightness. See `knowledge/playbooks/yeast_bud_scars.md`.

### 6. Passive Acquisition
The agent snapped once, analyzed, and submitted. No retry when detection looked suspicious. No refocusing when images were blurry. No exposure adjustment.

**Fix**: Build closed-loop retry logic. If detection returns 0 or suspicious count, try: different sigma, refocus, adjust exposure, switch channel.

## Meta-Lessons

1. **Always visually inspect before submitting**: Save overlay, view with LLM vision, sanity-check.
2. **Sample-specific knowledge matters**: A single detection method doesn't work for all samples.
3. **Retry is cheap, wrong answers are expensive**: Better to try 3 approaches than submit the first result.
4. **The knowledge base exists to prevent repeating these failures**: Read the relevant playbook before starting.
