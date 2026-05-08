# Temperature kinetics

**Assumes:** a sample whose physiology speeds up or slows down monotonically with temperature, a `Temperature` (or equivalent) state device that maps integer state → setpoint °C, and enough budget to dwell at multiple setpoints. Paired with `src/recipes/temperature_experiment.py` (`measure_growth_rate_at_temp_v2`, `temperature_response_curve_v2`, `estimate_q10` — the v2 variants discover the temp-state map at runtime; v1 hard-coded the yeast map).

## Sample types

The recipe is yeast-tuned by default but the *workflow* generalises to any temperature-modulated biology where you can count something per frame:

- Yeast growth (the canonical case — `S. cerevisiae`, optimum ~30 °C, Q10 ≈ 2.0 below optimum).
- Bacterial colony expansion.
- Drosophila / *C. elegans* development rate.
- Enzymatic reaction kinetics (chromogenic assays, Michaelis-Menten under T-control).
- Calcium-wave / cardiac AP frequency dependence on temperature.

## Workflow

```python
from src.recipes.temperature_experiment import (
    temperature_response_curve_v2,
    estimate_q10,
    YEAST_TEMP_MAP,
)

curve = temperature_response_curve_v2(
    core,
    states=[1, 0, 2, 3, 4],     # 4, 20, 25, 30, 37 °C for yeast
    temp_map=YEAST_TEMP_MAP,    # or pass None to discover at runtime
    n_frames=20,
    interval=10.0,
)
q10 = estimate_q10(curve, sub_optimal_only=True, clip=(1.0, 10.0))
```

## The four hard-won lessons

1. **Set Temperature to the optimum *before* the kinetic run** (ch573 r1 lesson). Default state is often ~20 °C, which the sim's Q10=2.0 makes ~25 % speed. If you want the *unmodulated* rate (e.g. as a control denominator for a Q10 fit), explicitly set state=optimum and `core.waitForDevice("Temperature")` before starting the timelapse.

2. **Filter to sub-optimum-only when fitting Q10**. Q10 = `(R₂/R₁)^(10/(T₂-T₁))` is only meaningful when both temperatures are below the species' optimum — above the optimum, growth *slows* (denaturation, stress) and the formula becomes garbage. The recipe's `estimate_q10(sub_optimal_only=True)` enforces this.

3. **Clip the Q10 estimate to a sane range** (default `clip=(1.0, 10.0)`). Two T-points 1 °C apart with noisy counts can produce Q10 = 47 from a single statistical fluke. Clipping protects downstream consumers from physically-impossible values; if the unclipped Q10 hits the clip rail, you needed more T-spread or more frames per setpoint.

4. **Discover the temp-state map at runtime, not from a hardcoded table.** `YEAST_TEMP_MAP` belongs in `temperature_experiment.py` (sim-tuned recipes/ tier — see [[../Core/feedback_core_vs_recipe_knowledge|core-vs-recipe knowledge contract]]) but a real `S. cerevisiae` temperature controller will have a different state→°C mapping. The v2 functions accept `temp_map=None` and discover via `core.getStateLabel(...)` or per-state property reads.

## Pacing

Each temperature setpoint dwell carries a thermal-equilibration cost: the sim implements an exponential approach to the setpoint with a time constant on the order of seconds–tens of seconds. Don't start frame 1 the instant you call `setState(...)`; either:

- Issue a 5–10 frame "settle MDA" between setpoint and measurement (same shape as the dead-volume waits in `Event-driven modality switch.md`), or
- Use `core.waitForDevice("Temperature")` if the device exposes a meaningful equilibrium signal.

If the sample drifts during equilibration, fold autofocus correction in via [[Core/Strategies/Closed-loop autofocus|closed-loop autofocus]] — temperature changes index-of-refraction in living samples enough to bump focus by ~µm, which a sub-µm DOF objective (≥40×) feels.

## See also

- [[Core/Strategies/Feedback control]] — temperature is one of several control axes; the loop shape is the same.
- [[Recipes/Drug response timelapse]] — sibling pattern: vary a perturbation, measure a per-cell rate.
- [[Recipes/Dose response]] — sibling pattern at the population level; the Hill fit is the Q10 fit's distant cousin.
- `Core/feedback_q10_temperature.md` (durable feedback memory) — the agent-specific lesson about always setting the temperature explicitly before kinetic runs.
