---
title: "Code as Policies: Language Model Programs for Embodied Control"
authors: Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, Andy Zeng
year: 2023
venue: 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 9493–9500
doi: 10.1109/ICRA48891.2023.10160591
arxiv: 2209.07753
url: https://arxiv.org/abs/2209.07753
researched: 2026-04-25
---

## Abstract

Large language models (LLMs) trained on code-completion have been shown to be capable of synthesizing simple Python programs from docstrings. We find that these code-writing LLMs can be re-purposed to write robot policy code, given natural language commands. Specifically, policy code can express functions or feedback loops that process perception outputs (e.g., from object detectors) and parameterize control primitive APIs. When provided as input several example language commands (formatted as comments) followed by corresponding policy code (via few-shot prompting), LLMs can take in new commands and autonomously re-compose API calls to generate new policy code respectively. By chaining classic logic structures and referencing third-party libraries (e.g., NumPy, Shapely) to perform arithmetic, LLMs used in this way can write robot policies that (i) exhibit spatial-geometric reasoning, (ii) generalize to new instructions, and (iii) prescribe precise values (e.g., velocities) to ambiguous descriptions ("faster") depending on context (i.e., behavioral commonsense). This paper presents code as policies: a robot-centric formalization of language model generated programs (LMPs) that can represent reactive policies (e.g., impedance controllers), as well as waypoint-based policies (vision-based pick and place, trajectory-based control), demonstrated across multiple real robot platforms.

## Smart microscopy principle (aspirational)

Liang et al. is a **robotics** paper, not a microscopy paper. It is included here as the aspirational paradigm flagged by smart-microscopy reviews (Ward et al. 2026; Mandal et al. 2025) that ask "what if a natural-language goal — *image cells undergoing apoptosis* — could be compiled directly into microscope-control code?" The transfer from manipulator robots to microscopes is exact: a microscope is an embodied agent with perception (camera) and a discrete action API (set channel, move XY, set exposure, snap), so the same LMP recipe applies.

Three contributions transfer directly to microscope control:

1. **Hierarchical code generation.** When the LLM hits an undefined function in its own output (e.g. `score = score_apoptosis(image)`), it recursively generates that function from a docstring rather than failing. For microscopy this means the agent can author both the *acquisition policy* (the outer loop: scout → score → zoom) and the *measurement primitives* (the inner functions: `score_apoptosis`, `is_dividing`) in the same pass, grounded in NumPy / scikit-image / SciPy calls a code-trained LLM already knows. The combinatorial explosion of "every smart-microscopy paper hand-writes its own classifier" collapses into "the LLM writes the classifier from natural language at acquisition time".
2. **Few-shot prompting with API examples as docstrings.** No fine-tuning required. Hand a code-LLM ~5 example pairs of `# command → policy code` and it composes new policies for unseen commands. For us this means a `pymmcore-plus` / `useq` API surface plus a handful of canonical solve-script examples *is* the prompt — the LLM does not need to be retrained per microscope, per sample, or per assay.
3. **Behavioral commonsense for ambiguous physical parameters.** "Faster" → reasonable velocity. "Brighter" → reasonable exposure increment. "More careful" → smaller step size, more averaging. The microscopy analogue is the long tail of soft directives a biologist gives a human operator ("don't bleach the sample", "sample more often near mitosis", "avoid the obvious debris") that conventional control software cannot interpret but a code-LLM can compile into concrete `MDAEvent` parameter changes.

The aspirational lens explicit in Ward 2026 §5 and the LLM-agent reviews is: **Liang's recipe IS the LLM-microscope-control paradigm**, with [[Papers/Mandal 2025]] as the first rigorous evaluation on a real instrument (and the discovery of where it breaks: capability-knowledge gap, sleepwalking, prompt fragility), and [[Papers/Boiko 2023]] as the parallel demonstration on chemistry. Liang is the *recipe*; Boiko is the *first scientific deployment* (chemistry); Mandal is the *first microscopy deployment + benchmark*; Kesavan is the *architectural critique* arguing the single-LMP shape is wrong granularity for science.

## What microscopy use this enables

Concretely, the LMP pattern unlocks acquisitions that are not feasible with a fixed `MDASequence`:

- **Goal-conditioned acquisition.** Operator types "image cells undergoing mitosis at 40×, skip empty wells"; the LLM emits a policy that scouts at 10×, scores each FOV with a generated `is_mitotic` predicate, and emits high-mag events only on hits. The policy is *the experiment*; rerunning with a different goal regenerates the policy.
- **Reactive / impedance-style control of soft parameters.** Liang shows the same recipe writes reactive policies that close a feedback loop on perception. In microscopy this is exposure servoing, focus servoing, or "step laser power down by 10% if the cell looks stressed" — code the LLM authors at acquisition time from a high-level constraint, not pre-canned by the operator.
- **Compositional protocols from natural language.** "Image neurons; for each soma, find the longest neurite and z-stack its tip at 60×" compiles to nested LMPs (outer scan, per-soma analysis, per-tip acquisition). The hierarchical code-gen recipe is what makes this tractable without writing a custom analysis pipeline per assay.

The translation to `pymmcore-plus` is mechanical: replace Liang's `pick_and_place` / `move_to` primitives with `core.snapImage`, `core.setXYPosition`, `core.setConfig`, `useq.MDAEvent`. The few-shot prompt then becomes example pairs of `# natural-language command → list[MDAEvent] generator`. The implementation surface is exactly the safety-bounded tool layer documented in [[Papers/Mandal 2025]] and [[Papers/Boiko 2023]] — see those notes for code; we do not reproduce the same skeleton here.

## Caveats inherited from Mandal 2025

Liang's recipe was demonstrated on tabletop manipulators with bounded action spaces and forgiving failure modes (drop a block, retry). On a microscope the failures are not forgiving: a sleepwalking agent can photobleach a sample, a hallucinated stage coordinate can crash the objective, a wrong channel selection can poison hours of timelapse. The Mandal 2025 evaluation makes this concrete; the practical lesson is that Liang's recipe applied to microscopy must be wrapped in a Python tool layer that hard-clips exposure, laser power, stage bounds, and dwell times *below* the LLM, not delegated to the prompt. Treat the LMP as the planner, not the executor.

## Cited by

- [[Papers/Mandal 2025]] — first rigorous on-microscope evaluation of the Liang recipe; AILA's tool-call-driven AFM is Code-as-Policies in microscopy form. AFMBench's failure modes (capability-knowledge gap, sleepwalking, prompt fragility) are the price the recipe pays when it leaves Liang's bounded-manipulator setting.
- [[Papers/Boiko 2023]] — Coscientist's planner / executor / doc-search / code-exec decomposition is the scientific-experiment deployment of Liang's recipe; the chemistry counterpart of what Mandal evaluates in microscopy.
- [[Papers/Kesavan 2025]] — argues a single LMP conflates five qualitatively different cognitive jobs (empirical / measurement / epistemic / narrative / orchestration) and proposes factoring along scientific-reasoning lines. The architectural critique of Liang-as-applied-to-microscopy.
- [[Core/Strategies/Feedback control]] — the observe → compute → modify → snap loop with the LLM in the controller slot; Liang's reactive-policy formulation is the canonical citation for the LLM-as-controller variant.
