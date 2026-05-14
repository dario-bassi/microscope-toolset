# `approach/` — how to think about a microscopy problem

Process-level notes. Not physics, not API, not workflows — the *meta* layer: how you open a new experiment, design experiments, close the loop between observation and action, and verify that what you think you saw is what the microscope actually showed you. These live at the top of the reading order because a sharp method beats a sharp tool: get the approach right and the rest follows.

Read `how_to_approach_a_problem` first — the other notes elaborate one step of that seven-step sequence.

## Files

- [[Core/Approach/How to approach a problem]] — the seven-step ritual for every new experiment; skipping steps is the #1 error source.
- [[Core/Approach/Backward design]] — start from the answer, work back to the acquisition; budget photons on necessity, not convenience.
- [[Core/Approach/OADA loop]] — Observe-Analyze-Decide-Act; the universal feedback loop that turns a camera into a smart-microscope.
- [[Core/Approach/Information driven]] — acquire where the information is, not where the grid says; entropy-gradient survey patterns.
- [[Core/Approach/Visual verification]] — use LLM-vision to sanity-check detections before committing to numbers.
- [[Core/Approach/Error recovery]] — diagnostic snapshots, fall-back acquisition plans, know-when-to-quit rules.
- [[Core/Approach/Pre-submission checklist]] — what to run through before recording results; includes the "null results are valid" rule.
- [[Core/Approach/Transferability contract]] — the inputs/actions a protocol is allowed to use if the goal is "would this work on a real microscope".
- [[Core/Approach/Pre-submit guard architecture]] — the *why* of the 3-tier guard pattern (preflight → render_vs_submit → visual review) and the orchestrator that composes them.
- [[Core/Approach/Acquisition strategy]] — decision pattern: when to survey, when to zoom, when to repeat.
- [[Core/Approach/Cell classification]] — LLM-vision prompts for classifying cell states.
- [[Core/Approach/Confidence assessment]] — deciding whether your detection is trustworthy enough to ship.
- [[Core/Approach/Coordinate systems]] — pixel / stage / world coordinate conversions and the common mistakes that burn labs.
- [[Core/Approach/Detection strategy]] — matching detection method to morphology (blob vs edge vs intensity).
- [[Core/Approach/Image quality]] — LLM-vision prompts for focus, exposure, and artifact diagnosis.
- [[Core/Approach/MDA solve pattern]] — canonical MDASequence + `run_events` pattern for one-shot acquisitions.
- [[Core/Approach/When to use what]] — decision framework: acquisition method vs detection method vs analysis layer.

## See also

- `../concepts/Concepts index.md` — the physics and API that approach notes assume.
- `../strategies/Strategies index.md` — worked workflow applications that use these approaches.
- `../pitfalls/Pitfalls index.md` — the failure modes these approaches are designed to avoid.
