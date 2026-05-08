<!-- audit 2026-04-24: "Failure Analysis" framing is incident-specific. Content is good (Z-scan perturbs the system it measures; per-position vs global drift). Consider reframing as "Pitfall: Z-scan feedback disturbs the drift it's measuring" or merging with strategies/closed_loop_autofocus.md. -->

# Failure Analysis: Z-Drift Autofocus

## Challenge Type
Multi-position timelapse with independent Z-drift per position.

## What Went Wrong

### Error 1: Global vs per-position drift
- **Error**: Computed drift_rate = Δz / total_snaps_across_all_positions
- **Fix**: Each position is independent — only snaps AT that position cause drift there
- **Result**: Significant underestimate of drift rates

### Error 2: Z-scan drift accumulation
- **Error**: Each Z-scan step is an acquisition that may trigger drift. At fast-drifting positions, a multi-step scan accumulates substantial drift DURING the scan
- **Result**: Measured optimal Z biased, predictions spiral out of control
- Slow-drift positions tolerate this; fast-drift positions do not

## Root Cause
The Z-scan itself is a measurement that perturbs the system. Heisenberg principle for microscopy:
- More scan points → better Z resolution but more drift accumulation
- Fewer scan points → less drift perturbation but coarser Z estimate

## Correct Approach (for future)
1. **Minimize scan steps** at fast-drifting positions (3-5 steps max)
2. **Use only narrow-scan rounds** for drift rate fitting (skip wide round 1)
3. **Correct for within-scan drift**: if best_z found at step i, tissue position = measured_z + (n_steps - i) * drift_rate (iterative refinement)
4. **Adaptive scan width**: start narrow, widen only if focus not found
5. **Binary/golden-section search**: 3-4 points to find peak, not full grid scan

## Key Insight
In systems with per-position drift, positions don't drift while you image elsewhere.
This means each position's drift accumulates only during its own acquisitions.
Account for this when estimating drift rates from multi-position data.
