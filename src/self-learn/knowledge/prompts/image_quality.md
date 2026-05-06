# Vision Prompts: Image Quality Assessment

## Focus Quality

> Examine this microscopy image and assess focus quality:
> - **In focus**: Sharp edges on cells/structures, high contrast between features
> - **Slightly defocused**: Edges softened but features still distinguishable
> - **Out of focus**: Blurry, low contrast, halos around bright objects
>
> If out of focus, estimate direction: are halos suggesting the focal plane is above or below the sample?
>
> Recommendation: Should the agent run autofocus before analysis?

## Exposure Assessment

> Assess the exposure of this microscopy image:
> - **Underexposed**: Most pixels dark, features hard to see, low dynamic range
> - **Well exposed**: Good contrast, features clearly visible, uses most of the dynamic range
> - **Overexposed**: Saturated bright regions (clipped whites), loss of detail in bright areas
>
> Recommendation: If exposure needs adjustment, should it increase or decrease, and by approximately how much?

## Detection Validation

> I ran cell detection on this image and found {N} cells. The overlay shows detected regions in [color].
>
> Assessment:
> 1. Does the count look correct? Are there obvious cells that were missed (false negatives)?
> 2. Are there detected regions that don't correspond to real cells (false positives)?
> 3. Are any touching/overlapping cells merged into a single detection?
> 4. Are the detected boundaries reasonable, or are they too tight/loose?
>
> Suggest parameter adjustments if detection looks wrong.

## Artifact Identification

> Examine this microscopy image for common artifacts:
> - Dust particles (sharp dark spots not associated with sample)
> - Air bubbles (large circular bright/dark regions with sharp edges)
> - Illumination gradient (uneven brightness across the field)
> - Debris on coverslip (out-of-focus dark blobs)
> - Photobleaching (dimmer region where previously exposed)
>
> Report any artifacts found and whether they would affect quantitative analysis.
