# Knowledge Base — Module & Folder Review

This document reviews each module and folder in the `src/self_learn/knowledge/` database:
what it is, what it's used for, its advantages, its disadvantages, what should change
for better agent usability, and its main problems when used with the full `microscope-toolset` project.

---

## `Core/Approach/` — Process-Level Problem Solving

**What it is.** Meta-layer knowledge: how to think about a microscopy experiment before
touching any hardware. Seven-step protocol, backward design, OADA loop, visual
verification, error recovery, detection strategy, acquisition strategy.

**What it's used for.** An agent reads these files before starting any experiment to
understand *how to approach the problem*, not what specific code to run. These are
reasoning scaffolds.

**Advantages.**
- Platform-agnostic — works for real hardware and simulation equally.
- Provides explicit decision flowcharts (`When to use what.md`) that an LLM can follow.
- Covers common failure modes (`Error recovery.md`) with diagnostic trees.
- Visual verification protocol teaches the agent to use its vision capability.

**Disadvantages / gaps.**
- Some files are long (200+ lines) with no sub-headings for quick navigation. An agent
  doing RAG retrieval may get the wrong section or miss critical content.
- `Pre-submission checklist.md` was written for a graded loop — some language may still
  feel "examinable" rather than "scientific". Review the framing for real lab context.
- No file documents the **hardware configuration discovery** step (getting channel names,
  pixel sizes, objectives) in a single authoritative place — it's spread across
  `Core basics.md`, `Acquisition strategy.md`, and `How to approach a problem.md`.

**What should change.**
- Add a short (< 20 line) "TL;DR" section at the top of each Approach file so an agent
  can do a quick-scan retrieval before deciding whether to read the full note.
- Create a dedicated `Hardware discovery.md` file consolidating the config-query pattern.
- The `Detection strategy.md` file contains parameter-advisor imports
  (`analysis.parameter_advisor`) that reference modules not yet fully documented
  in `ARCHITECTURE.md` — verify these functions exist and add their signatures.

**Problems when used with microscope-toolset.**
1. ~~**Import paths assume `src/self_learn` is in sys.path**~~ **FIXED:** The folder is now
   `src/self_learn`. All knowledge-file imports use `from self_learn.X import`.
   Install with `pip install -e .` or `uv sync` — no sys.path setup needed.
2. **No cross-reference to `src/benchmarking/`** — the agent doesn't know how to connect
   to the test server (the `test_server.py` that serves the virtual microscope). A note
   explaining the setup for local testing is missing.

---

## `Core/Concepts/` — Physics & API Reference

**What it is.** Short reference entries on microscopy physics (exposure, SNR, Nyquist,
depth of field, chromatic aberration, fluorophores) and the pymmcore-plus / useq API
(core objects, MDA engine, event-driven acquisition).

**What it's used for.** Looked up when an agent encounters a physics or API question
during experiment design. Not read top-to-bottom; used as a reference card.

**Advantages.**
- Each file is focused on a single concept — good for retrieval.
- Physics files explain the *why* (not just the *how*), which helps an agent reason
  about edge cases (e.g., why Nyquist matters for feature detection at different
  magnifications).
- `Migration notes.md` documents breaking API changes, preventing old patterns from
  being used with the current codebase.
- `Sample time vs wall-clock time.md` covers a subtle but critical distinction for
  timelapse experiments.

**Disadvantages / gaps.**
- **No SLM concepts file.** SLM patterning is covered in Strategies but there's no
  reference entry explaining what an SLM is, how `setSLMImage` works, or what
  `SLMImage(data, device)` expects.
- `MDA engine.md` and `MDA generators.md` overlap significantly with `MDA standard.md`.
  An agent retrieving on "MDA" may get partial information from each without a clear
  canonical source.
- **No file on pymmcore-proxy** — the proxy pattern (connect vs local) is mentioned
  across many files but never explained in a single reference entry.

**What should change.**
- Add `SLM reference.md` covering: what an SLM is, how to get the device name
  (`core.getSLMDevice()`), how to build a mask, how `setSLMImage` + `displaySLMImage`
  work, and the constraint that SLM images cannot be embedded in `MDAEvent` over proxy.
- Add `pymmcore-proxy.md` covering: local vs remote core, `connect()` vs `CMMCorePlus()`,
  what works identically and what doesn't (SLM image serialisation, direct property access).
- Consolidate the three MDA files into a coherent reading order with explicit cross-links.

**Problems when used with microscope-toolset.**
1. ~~**`Core basics.md` uses short import paths**~~ **FIXED:** All knowledge files now use
   `from self_learn.hardware.core import` (canonical installed-package form). Install once
   with `pip install -e .` and all imports resolve.
2. **Physics values are dimensionless examples** — the Nyquist, depth-of-field, and
   exposure examples use generic numbers. The agent might apply them without checking
   actual instrument calibration. A note like "always query `core.getPixelSizeUm()` —
   never use textbook numbers" should be bolded in each relevant file.

---

## `Core/Strategies/` — Workflow Applications

**What it is.** Literature-backed smart-microscopy workflows implemented as pymmcore-plus
patterns. 35+ files covering: adaptive acquisition, feedback control, autofocus,
timelapse design, multi-scale morphometry, SLM optogenetics, gentle imaging, event-driven
acquisition, dose-response, multichannel scan, batch processing, and more.

**What it's used for.** The agent reads a strategy file to understand how to structure
an experiment before writing code. Each file sketches the control loop and links to
`src/self_learn/` for the full implementation.

**Advantages.**
- Large coverage: almost every experiment type has a strategy file.
- Files include code sketches that show the exact pattern, not just prose.
- Literature citations in several files connect to the `Papers/` knowledge tier.
- `Gentle imaging.md` provides an explicit "photon budgeting" decision tree — directly
  actionable for any fluorescence experiment.
- `Measurement methodology.md` covers the full measurement design cycle (sample size,
  statistical tests, units, reporting).

**Disadvantages / gaps.**
- **Quality is uneven.** Some files (Adaptive acquisition, Feedback control) are thorough
  and well-structured. Others (Wave propagation, Axis sweep alignment) still contain
  challenge-ID-specific references and simulation-specific language that reduces
  transferability.
- **Many files reference non-existent modules.** `Dose threshold.md` references
  `utils.dose_threshold`, `Connectivity mapping.md` references `workflows.optogenetics.infer_connectivity_dff` — these modules may not exist in `src/self_learn/`. Any agent following these imports will get errors.
- **No index of which src modules map to which strategies.** The agent has to infer the
  mapping from scattered "See also" links.
- **38 files is too many** for an agent to effectively triage. Some files (Rate-limited
  drive, Pulsed schedule trajectory, Fit-then-control) are highly specific to controller
  design and rarely needed. Consider grouping them.

**What should change.**
- Audit every strategy file's import statements against actual `src/self_learn/` modules.
  Mark or remove references to non-existent functions.
- Add a 1-sentence description header to each strategy file for faster triage retrieval:
  `> **When to use:** [one sentence trigger condition]`
- Add `Strategies index.md` cross-reference to the strategy-to-src-module mapping table.
- Remove remaining challenge-ID references (ch657, ch658 in Axis sweep, ch605 in
  Adaptive acquisition, ch609 in Closed-loop autofocus) — these add noise without
  transferable value.

**Problems when used with microscope-toolset.**
1. **Many referenced modules do not exist in `src/self_learn/`.**  For example:
   - `utils.dose_threshold` — not in the current `utils/` directory
   - `workflows.optogenetics.infer_connectivity_dff` — not in `workflows/`
   - `analysis.parameter_advisor` — not confirmed in `analysis/`
   An agent following these import examples will fail silently or with confusing errors.
2. **`Auto recipe selection.md` references `utils.auto_recipe` and `utils.brief_parse`**
   which were built for the challenge-brief parsing system. The `parse_brief()` function
   expects a challenge JSON structure — it won't work on arbitrary experiment descriptions
   without modification.
3. **No strategy covers the napari integration** — the project has napari plugins
   (`src/plugin_napari.py`, `src/mcp_microscopetoolset/`) but no knowledge file explains
   how to connect the library to the napari GUI or MCP server.

---

## `Core/Pitfalls/` — Failure-Mode Catalogue

**What it is.** Six generalizable methodology failure modes with structural fixes:
FOV vs well coverage, MDA silent truncation on proxy, apparent vs underlying count
asymmetry, reaction-diffusion classification, run-and-tumble tracking, Z-drift autofocus.

**What it's used for.** An agent reads these when it encounters a characteristic symptom
(count is always off, counts never converge, MDA delivers fewer frames than expected).

**Advantages.**
- Symptom-first structure makes these easy to apply diagnostically.
- "Why it's a pitfall, not a bug" sections prevent the agent from chasing false leads.
- `MDA silent truncation on proxy.md` is directly actionable — `run_events_checked` is
  the exact fix with a copy-paste code block.
- `FOV vs well coverage.md` is a classic real-microscope pitfall, not simulation-specific.

**Disadvantages / gaps.**
- Only 6 pitfalls documented. Many common real-microscope failures are not covered:
  photobleaching during timelapse, stage drift exceeding autofocus range, channel
  bleedthrough in multi-channel imaging, objective contamination causing dim images.
- `Sim-state vs rendered count asymmetry.md` was heavily simulation-specific and has
  been partially rewritten but still uses language specific to rendered images — not
  entirely natural for a real-microscope lab context.

**What should change.**
- Add `Photobleaching during timelapse.md` — how to detect, how to correct, when to
  switch to a low-dose protocol.
- Add `Stage drift exceeding autofocus range.md` — when the sample drifts faster than
  the autofocus loop can correct, and what to do.
- Add `Channel bleedthrough.md` — how to detect bleedthrough, `utils.spectral_leak`
  correction, when to use sequential (not simultaneous) acquisition.

**Problems when used with microscope-toolset.**
1. The `run_events_checked` fix references `utils.mda_diagnostics` — this module exists
   in `src/self_learn/utils/` and should work, but has not been verified against the
   current `src/self_learn/hardware/core.py:run_events()` signature.
2. No pitfall file covers failures specific to the **MCP server** (`src/mcp_microscopetoolset/`)
   or **napari plugin** integration, which are core project components.

---

## `Papers/` — Verified Citation Library

**What it is.** 58 markdown files, each containing a verified DOI, author/year/journal
metadata, and a faithfully-copied abstract for a published paper on smart microscopy,
adaptive acquisition, event-driven imaging, LLM-agent science, or related topics.
Papers span 2004–2026.

**What it's used for.** An agent retrieving a method from a strategy file can follow
the `[[Papers/Mahecic 2022]]` wiki-link to get the exact citation, verify the claim,
and understand the original context. Used for grounding strategy claims in literature.

**Advantages.**
- Every paper has a real DOI — citations are verifiable.
- Abstracts are copy-faithful and searchable (no paraphrasing that loses detail).
- Coverage spans the full arc of smart microscopy from target-driven acquisition (2004)
  to LLM-agent control (2025), giving an agent historical context.
- Paper entries include a brief "Why this matters" note explaining the relevance to
  the project.

**Disadvantages / gaps.**
- **No index by topic.** 58 papers with no filtering by topic (autofocus, event-driven,
  adaptive acquisition, LLM agents, SLM, etc.). An agent searching for "papers on
  feedback control" must read all 58 filenames or rely on a vector search.
- **Papers are not linked in both directions.** Strategy files cite papers
  (`[[Papers/Mahecic 2022]]`), but the paper files don't link back to which strategies
  use them.
- Some papers are cited by strategy files that have since been removed or heavily
  rewritten, leaving orphan citations.

**What should change.**
- Add `Papers index.md` organized by topic (Adaptive, Event-driven, Autofocus, SLM,
  LLM-agent, General smart-microscopy).
- Add back-links from each paper file to the strategy files that cite it.
- Consider a `relevance: high/medium/low` tag in each paper's frontmatter to help
  triage during retrieval.

**Problems when used with microscope-toolset.**
1. Papers are purely static documentation — they work correctly in all contexts.
2. The main risk is that an agent cites a paper for a technique but the corresponding
   `src/self_learn/` module has a different API than the paper describes. The paper
   and the implementation are not cross-validated.

---

## Index Files (`INDEX.md`)

**What they are.** `INDEX.md` is the single top-level entry point — a self-contained
inventory with structure overview, reading-order guides, and per-folder file lists.
`README.md` and `Core index.md` were merged into `INDEX.md` and deleted.

**What it's used for.** Navigation — either for a human or an agent's first-contact
retrieval. The reading order sections tell the agent which files to read first.

**Advantages.**
- Single file — no ambiguity about which index to read.
- Reading-order section covers a new experiment scenario.
- Four-tier structure (Approach / Concepts / Strategies / Pitfalls) clearly documented.
- Import note explains `pip install -e .` / `uv sync` pattern.

**Disadvantages / gaps.**
- **No reading-order guidance for common experiment types** (e.g., "If doing a
  timelapse, read: Timelapse design → MDA standard → Gentle imaging → Verification
  checklist"). The current reading order is generic.
- Sub-folder lists are flat — no indication of which files are most commonly
  needed vs niche reference material.

**What should change.**
- Add experiment-type reading-order guides: one for timelapse, one for multi-position
  scan, one for SLM/optogenetics, one for tracking.

**Problems when used with microscope-toolset.**
1. No index file mentions the MCP server, napari plugin, or benchmarking system — the
   three main user-facing entry points of the full project. An agent using this knowledge
   base has no context about the project's deployment model.

---

## Summary: Main Problems for an Agent Using This Library

When an agent reads this knowledge base and tries to use `microscope-toolset`, it will
encounter these structural problems, roughly in order of severity:

| # | Problem | Impact | Fix |
|---|---------|--------|-----|
| 1 | ~~**Import paths don't work out of the box.**~~ **FIXED.** Folder renamed to `src/self_learn`; all knowledge imports now use `from self_learn.X import`. `pip install -e .` is sufficient. | — | — |
| 2 | **Strategy files reference non-existent modules.** Many `utils.*`, `workflows.*`, `analysis.*` imports point to functions not present in `src/self_learn/`. | Agent writes code that fails at import time. | Audit all import examples against actual modules; mark unimplemented references. |
| 3 | **No bridge from knowledge base to project entry points.** No file explains the MCP server, napari plugin, or test server setup. | Agent doesn't know how to run an experiment in the actual system. | Add a project-integration file linking knowledge to deployment. |
| 4 | **`auto_recipe` and `brief_parse` are challenge-loop tools.** They expect a `challenge.json` structure not present in real experiment workflows. | Agent tries to use them and gets KeyError on missing fields. | Rewrite `Auto recipe selection.md` to show how the recipe dispatcher works without a challenge brief. |
| 5 | **Remaining challenge-ID examples in Strategies.** Files like `Axis sweep alignment.md` and `Closed-loop autofocus.md` use ch### IDs as example anchors. | Minor confusion; reduces transferability signal. | Replace with generic "e.g., in a zebrafish vasculature experiment" framing. |
| 6 | **Papers not indexed by topic.** 58 papers with no topic filter. | Agent retrieval for "papers on X" is slow / inaccurate without semantic search. | Add a topic-indexed `Papers index.md`. |
| 7 | **No documentation on when NOT to use self-learn.** The library is for complex adaptive experiments; for simple snaps, the full MDA machinery is overhead. | Agent may use complex workflows for simple tasks. | Add a "scope of this library" note to `INDEX.md`. |
