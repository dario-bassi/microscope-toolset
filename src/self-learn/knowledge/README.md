# Knowledge Base

**What this is.** A teaching notebook for programmatic smart microscopy on pymmcore-plus, written at PhD-student level. The goal: someone new to the field should be able to read through `Core/`, understand what smart microscopy is, see worked examples of real applications (with code), and know where to find the production implementation.

**What lives here vs `src/`.**
- `knowledge/` — prose, principles, worked examples, literature-backed application sketches, pitfalls.
- `src/core/` — the production library the prose refers to. Knowledge notes *point at* `src/core/` modules; they don't duplicate them. A snippet in a knowledge note is illustrative; the real implementation (error handling, parameterization, tests) is in `src/`.

## Two tiers, mirror `src/`

- **`Core/`** — sim-agnostic. Microscopy principles, pymmcore-plus usage, literature-backed smart-mic applications, generalizable pitfalls. Would read naturally in a real lab's onboarding.
- **`recipes/`** — sim-specific. Sample-tuned playbooks, backend quirks documented honestly, simulator state mappings. Paired with `src/recipes/*.py` files of the same name.
- **`Papers/`** — verified citations (top-level, cited by either tier). Every entry has a fetched DOI + copy-faithful abstract. Plain-text citations in other notes are an anti-pattern; use `[[../Papers/<slug>]]` wiki-links.

## Four sub-types of core knowledge

| Sub-dir | Holds | Example |
|---|---|---|
| `Approach/` | **Meta: how to think about a problem** — open-a-challenge recipes, backward design, OADA loops, visual verification | "snap all channels first, look with vision, confirm markers before committing to analysis" |
| `Concepts/` | **Reference: physics + API** — exposure/saturation, photodamage, SNR, Nyquist; pymmcore-plus core, useq MDA, event-driven acquisition | "exposure × intensity is the dose budget; saturation destroys dynamic range; phototoxicity is cumulative" |
| `Strategies/` | **Workflow applications** — literature-backed smart microscopy applications implemented as pymmcore-plus workflows (adaptive acquisition, feedback control, multi-scale surveys, closed-loop steering) | "Event-driven microscopy (Mahecic 2022 pattern): trigger → switch mode → capture → return" |
| `Pitfalls/` | **Failure-mode catalogue** — generic methodology mistakes with their structural fixes | "angle threshold must exceed the 95th-percentile noise angle" |

## The test for a core note

*Copy the prose into a real lab's onboarding doc — would a PhD student at a real microscope read it naturally?*

- **Yes:** it's core. Fine to cite `[[../Recipes/X]]` as a worked example.
- **No — it names a specific challenge or backend as the source of insight:** it's a recipe. Move to `Recipes/`.
- **Partially:** split — pull out the generalizable principle as a core note, leave the sim-specific bits in a companion recipe note.

## Linking

Obsidian-style `[[wiki-links]]`. Link in **both** directions.

**Note → code.** Every knowledge note should link to its counterpart in `src/core/` (or `src/recipes/`) so the reader can jump from the explanation to the production code:

```markdown
See [[../../../src/core/workflows/autofocus.py]] for the full implementation
(this note shows only the control-loop sketch).
```

**Note → other notes.** Equally important: cite sibling notes that cover adjacent concepts. Microscopy concepts build on each other — a note on *SNR and dynamic range* should link out to *exposure and photodamage*, *noise estimation*, and relevant strategies that use SNR as a decision variable. Strategy notes should link to the concepts they assume and the pitfalls they avoid. Treat the notebook as a graph, not a flat list — a reader following any one note should see three or four natural next hops.

Good patterns:
- **Concept → concept** for physical relationships (exposure ↔ photodamage ↔ SNR).
- **Strategy → concepts it assumes** (multi-scale morphometry → Nyquist sampling, depth of field).
- **Strategy → pitfalls it avoids** (closed-loop autofocus → FOV-vs-well-coverage trap).
- **Pitfall → recipe** where the concrete parameters live (run-tumble tracking → motile_organism_tracking recipe).
- **Approach → concepts + strategies** used in that approach.

A "See also" section at the bottom of every note is a low-effort way to maintain this. If you write a note with no outlinks, something is either orphaned or undocumented — usually the latter.

`Core/` notes may link to `Recipes/` as worked examples. Recipe notes may link anywhere. See `INDEX.md` for the full tree.
