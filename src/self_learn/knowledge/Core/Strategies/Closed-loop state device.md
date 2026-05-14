# Closed-loop control via a state device

> **When to use:** When a discrete state device (perfusion valve, temperature controller, electrode) drives a biological response that must be measured each frame and fed back to update the device state.

A general pattern: a discrete state device (enum) sets a condition on the sample, the sample responds, and you measure the response and update the state on each frame.

Examples of the state device on real microscopes: a field-emitting electrode array, a perfusion valve selector, a temperature controller, a shutter wheel, an SLM pattern register.

## The loop

```
measure → decide → set state → acquire next frame → measure
```

```python
for frame in range(n_frames):
    img = snap(core, channel=ch)           # acquire
    metric = measure(img)                  # e.g. population centroid
    state = decide(metric, target, history) # pick next state
    core.setState("<device>", state)       # commit
    history.append(metric)
```

## Per-cell metrics beat pixel metrics under a fixed FOV

With a stationary stage, intensity-weighted pixel COM is a poor motion estimator for confluent tissue: cells leaving one edge are replaced by new cells entering the opposite edge, so the total image mass stays centred even as the underlying tissue drifts.

Use object-level features: segment first, compute the mean of per-object centroids. Apply the same principle to fraction-in-target counts — count object centroids inside the target region, not pixels.

If the sample moves faster than the FOV, switch to stage-tracking (follow the population with the stage) rather than closed-loop-within-one-FOV. The two patterns solve different problems.

## Control strategies

- **Greedy / bang-bang**: each frame pick the axis with the largest remaining gap. Simple; works when axes are roughly comparable.
- **Commit**: stick to the dominant axis until progress stalls or the gap sign flips. Avoids flip-flopping under noise.
- **Time-split**: alternate on a fixed schedule (e.g., 12 frames of +X then 13 of -Y). Useful when migration rate and target geometry are known up front.

## Physics budget

Before designing the control, compute what's actually achievable: `max_achievable_displacement = migration_rate × n_frames`. If the starting gap exceeds this, no control policy can reach the target — report the physics limit in your method description rather than grinding. This is a feature of the sample, not a failure of the controller.

## Related

- `[[Core/Strategies/Closed-loop autofocus]]` — the Z-plane analogue.
- [[Core/Strategies/Rate-limited drive]] — when the state device drives a saturating
  rate law (gene induction, photoconversion, ablation, FUCCI), use
  `predict_n_steps` for the open-loop horizon + `drive_to_band` for
  the closed-loop early-stop. Sprint #35 utility, lifted from
  ch621 r1 win.
- ``self_learn.workflows.stage_tracking`` — the follow-with-stage alternative.

## Literature

- [[Papers/Passmore 2025]] — SLM pattern register as the state device: each frame's measurement updates the optogenetic pattern to steer cells toward a predefined outcome.
