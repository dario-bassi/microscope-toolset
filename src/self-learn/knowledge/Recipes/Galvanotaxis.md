# Recipe: galvanotaxis (directional field-driven migration)

**Assumes:** virtual galvanotaxis backend; state device `Electrode` with enum (off=0, +X=1, -X=2, +Y=3, -Y=4); cells migrate toward cathode.
**Composes from Core:** `Core/Strategies/Closed-loop state device.md`, `Core/Strategies/Physical-unit thresholds.md`.
**Tested on:** ch477, ch480, ch572.

## Sim-specific parameters

```python
ELECTRODE = {"off": 0, "+X": 1, "-X": 2, "+Y": 3, "-Y": 4}
MAX_MIGRATION_PX_PER_FRAME = 4   # ch572 backend cap
```

A 25-frame run can therefore cover at most ~100 px along one axis, ~70 px diagonal if split evenly. If the starting gap from population centroid to target exceeds this, note the physics limit in the method description rather than retrying.

## This sim's snap-coupled quirk

Note: the ch572 backend advances only when a snap is issued (`sim.auto_step = True`). That means verification snaps cost tracking frames. Plan the budget: `total_snaps = n_frames`, no extras. This is a simulator artifact, not a real-microscope property — on real hardware the field + sample evolve on a clock independent of the camera. See `../Core/Strategies/Closed-loop state device.md` for the generic version of the pattern.

## Worked example (ch572 r1)

The r1 bug was **not** in the control policy but in the motion metric: using pixel COM under a fixed FOV misses tissue drift because new cells replace departing cells at the edges. See `../Core/Strategies/Closed-loop state device.md` § *Per-cell metrics beat pixel metrics*.

## Related recipes

- `[[Recipes/Motile organism tracking]]` — the stage-following alternative when the sample leaves the FOV.
