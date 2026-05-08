# Light-sheet alignment recipes

Paired with `src/recipes/lightsheet_align.py`. Three callable shapes for the AutoPilot lineage:

- **`align_once`** — independent per-axis sweeps with the OTHER axes pinned at a default state. Single-time-point alignment when axes don't interact. ch657 r1 = 10/10 reference shape.
- **`align_descent`** — coordinate descent from a (possibly compounded) misalignment. Each axis sweep pins the OTHER axes at their CURRENT-best state from the previous sweep. Use when axes interact (content-asymmetry biases tilt argmax once Y is corrected). ch658 r1 = 10/10 (97.6% recovery, 17 snaps, Y=2 / TX=0 / TY=0).
- **`drift_track_capture`** — periodic re-alignment within a snap budget. Each cycle: short Y sweep → set argmax → 1 capture snap. McDole 2018 active alignment. ch664 r1 = 5/10 (predicate failure from dry-run state bleed; methodology vindicated by grader) → ch665 r1 = 10/10 on fresh proxy (96.6% recovery).

The shared substrate is `src/core/utils/axis_sweep.sweep_axis_mda` (sprint #46 core utility) — every snap travels through a `useq.MDAEvent` with `MDAEvent.properties = [(axis, "State", k)]` set-points. Submission-gate clean (no inline `setState → snap` loops); the same event list lands directly on real-microscope MDA hardware.

## Reading a brief for which shape applies

| Brief signal | Recipe |
|---|---|
| "sweep each axis at its default" / "single time-point" | `align_once` |
| "compounded misalignment" / "descent" / "coordinate" / "interact" | `align_descent` |
| "drift" / "re-align" / "periodic" / "McDole" / "snap budget × cycles" | `drift_track_capture` |

`auto_recipe(brief=...)` routes `Archetype: lightsheet_align` → `align_descent` and `Archetype: lightsheet_drift` → `drift_track_capture` automatically.

## Anti-patterns (lessons from the lineage)

- **Dry-run + finite-budget dynamic**: ch664 r1 5/10 was a methodology-clean live submission whose recovery dropped from the dry-run's 96.6% to 58.9% because the dry-run consumed 24 snaps of drift before the live run started. For drift-tracking scenarios, either skip the dry-run OR re-serve before live submission. See `feedback_dry_run_state_bleed.md` and `Sample time vs wall-clock time.md`.
- **Cached state map**: `align_descent` reads `core.getProperty(axis, "State")` at the start. If the previous run left the device descended, the "initial" diagnostic is misleading. ch658 grader flagged this as a transferability concern. Per-pass re-read protects you.
- **Tilt asymmetry**: AutoPilot descents on tilt axes converge OFF the device-aligned default in the presence of content asymmetry (zebrafish vasculature lateralized). Realistic physics — don't treat tilt-not-at-2 as a bug.

## When NOT to reach for these recipes

- **Single-axis filter-wheel argmax** that doesn't compose with a descent → use `axis_sweep.sweep_axis_mda` directly. The recipe layer adds no value when there's only one axis.
- **Continuous-axis Z focus**: light-sheet alignment is categorical (5 states); Z-focus is continuous. Use `workflows.autofocus.sweep_focus` with `parabolic_peak_interp` for the Z case.

## Real-microscope analogue

AutoPilot 5-axis (Royer 2016) on hardware is the same `MDAEvent.properties` shape — a 5-position encoder per illumination + detection axis, scanned via the runner. McDole 2018 mouse-embryo adaptive imaging (re-alignment between time-points) is the same `drift_track_capture` cycle scaled to 5 axes.

## Tests

`tests/test_lightsheet_align.py` covers all three entry points + auto_recipe routing. 11 tests via FakeCore + monkeypatched `run_events`.

## Related

- `[[Core/Strategies/Axis sweep alignment]]` — core-utility-level treatment with the inline-vs-MDA submission-gate explanation.
- `[[Royer 2016]]` — paper note.
- `[[McDole 2018]]` — paper note.
- `[[Schmidt 2023]]` — coordinate-descent under an RL controller (related shape).
