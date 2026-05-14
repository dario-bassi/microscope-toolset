# Decision Pattern: Confidence Assessment

> **TL;DR** — Three-tier confidence ladder for detection results.
> **High** (submit directly): count stable across sigmas, SNR > 10, visually confirmed.
> **Medium** (verify first): ±20% variation across sigmas, SNR 3–10 → save overlay, inspect.
> **Low** (retry): wild variation, SNR < 3, visual disagreement → refocus / re-expose /
> change method. Always check count plausibility for the FOV before submitting.

## When to Trust Detection Results

### High Confidence (submit directly)
- Detection count matches visual estimate from LLM vision
- SNR > 10 (clear signal above noise)
- Count is stable across sigma values (2.0, 2.5, 3.0 all give same count)
- Cells are well-separated, no touching/overlapping
- Image is clearly in focus with good exposure

### Medium Confidence (verify before submitting)
- Count varies by +/-20% across different sigma values
- Some cells are touching or overlapping
- SNR between 3-10
- A few detected objects look questionable
- **Action**: Save overlay image, visually verify with LLM vision, adjust parameters

### Low Confidence (retry with different approach)
- Count varies wildly across parameters
- Visual inspection disagrees with detection
- SNR < 3 or image appears out of focus
- More than 30% of detections look like false positives
- **Action**: Refocus, adjust exposure, try different detection method, try different channel

## Sanity Checks

Before submitting any answer, verify:

1. **Count plausibility**: Is the number of cells reasonable for this FOV and sample type?
   - Low mag (large FOV): typically 5-50 cells in culture, 100-500 RBCs in blood smear
   - High mag (small FOV): typically 1-10 cells in culture

2. **Position plausibility**: Are detected positions within the expected region?
   - Edge cells (within 50px of border) are unreliable
   - Cells outside the FOV are detection errors

3. **Size plausibility**: Are detected areas reasonable?
   - Cultured cells at 10x: 200-2000 px
   - RBCs at 40x: 100-400 px
   - Neurons soma: 300-3000 px at 10x

4. **Consistency**: If measuring the same sample at two timepoints or positions, do results agree?

## When to Use LLM Vision Instead of Code

- **Cell state classification**: Healthy/apoptotic/mitotic -- visual patterns are complex
- **Mitotic stage**: Prophase vs anaphase requires seeing chromosome arrangement
- **Sample type identification**: What kind of sample is this?
- **Artifact detection**: Dust, bubbles, out-of-focus debris
- **Sanity checking**: Does the detection overlay match what I see?
- **Difficult counting**: When automated detection is uncertain, count manually from image

## See also

- [[Core/Approach/Visual verification]] — how to actually use the LLM-vision check (save overlays, compare to detection).
- [[Core/Approach/Detection strategy]] — picking the right detection method; confidence is downstream of method choice.
- [[Core/Concepts/SNR and dynamic range]] — SNR thresholds are a major driver of the confidence tiers.
- [[Core/Approach/Pre-submission checklist]] — §10 null-results rule: a well-characterised low-confidence measurement beats a guess.
