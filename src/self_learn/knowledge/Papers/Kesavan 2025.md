---
title: "Reconceptualizing Smart Microscopy: From Data Collection to Knowledge Creation by Multi-Agent Integration"
authors: P. S. Kesavan, Pontus Nordenfelt
year: 2025
venue: 'arXiv preprint (later published in Journal of Microscopy as "From observation to understanding: A multi-agent framework for smart microscopy")'
arxiv: 2505.20466
doi: 10.48550/arXiv.2505.20466
url: https://arxiv.org/abs/2505.20466
researched: 2026-04-25
---

## Abstract

Smart microscopy represents a paradigm shift in biological imaging, moving from passive observation tools to active collaborators in scientific inquiry, enabled by advances in automation, computational power, and artificial intelligence. Central to this framework is the concept of the *epistemic-empirical divide* in cellular investigation — the gap between what is observable (the empirical domain) and what must be understood (the epistemic domain). The authors propose six core design principles — epistemic-empirical awareness, hierarchical context integration, an evolution from detection to perception, adaptive measurement frameworks, narrative synthesis capabilities, and cross-contextual reasoning — and argue these principles guide a multi-agent architecture in which empirical observation is aligned with the goals of scientific understanding. The paper is a position / framework paper: no implementation, no on-instrument validation; the architecture is presented as a roadmap for systems that go beyond automation to support hypothesis generation, insight discovery, and theory development.

## Smart microscopy principle

Where [[Papers/Mandal 2025]] and [[Papers/Boiko 2023]] put a single LLM agent above the instrument's tool surface, Kesavan & Nordenfelt argue this is the wrong granularity: a single agent conflates *running the microscope*, *measuring what it sees*, *connecting that to a hypothesis*, *writing the story back*, and *deciding what to do next* — five qualitatively different cognitive jobs. Their proposal (Section 6.1 of the paper) is to factor these into five named agent roles, each an LLM with its own context, tools, and objective:

- **Empirical agents** — control imaging parameters, optimise acquisition settings; the "drive the microscope" layer (closest to what AILA / Coscientist already do).
- **Measurement agents** — apply quantitative frameworks to extract metrics from images; the "what does this picture say in numbers" layer.
- **Epistemic agents** — connect observations to theoretical frameworks and the active research question; the "does this update the hypothesis" layer.
- **Narrative agents** — synthesise across the other agents into coherent descriptions, explanations, and visualisations; the "tell the story of the run" layer.
- **Orchestration agents** — coordinate overall investigation strategy, balancing exploration vs. confirmation while keeping alignment with the research goal; the "what should we do next" layer.

The substantive claim is that *crossing the epistemic-empirical divide is itself an architectural problem, not a prompt-engineering problem*. A single agent cannot keep an open hypothesis, a running measurement protocol, and a frame-by-frame imaging schedule all in one context window without one of them collapsing into the others. Splitting them lets each agent specialise: the orchestrator reasons about scientific strategy, the epistemic agent owns the hypothesis state, the empirical agent owns the device, and the narrative agent owns the explanatory artifact. The architecture is aspirational — the paper presents no implementation, case study, or empirical validation — but it is the most explicit decomposition of the agentic-microscopy design space published so far, and a useful counterweight to the implicit single-agent assumption of Boiko 2023 and Mandal 2025.

## What this implies for our codebase

We do not build this architecture today; we already have a single-agent loop (this very system) and a tool surface that sits closer to Mandal's AILA than to Kesavan & Nordenfelt's five-role split. But the paper offers a useful diagnostic when our own agentic runs misbehave: which of the five roles is failing? Concretely, the role split maps onto modules we already have:

- *Empirical* → `src/core/hardware/core.py` (`run_events`, `snap`, stage / objective control).
- *Measurement* → `src/core/detection/`, `src/core/analysis/` (segmentation, intensity, kinetics, spatial).
- *Epistemic* → `Core/` and the hypothesis-update step inside `src/core/workflows/reasoning.py`.
- *Narrative* → showcase images, the per-challenge solve script, and the `submit_solution(method_description=…)` argument that goes back to the grader.
- *Orchestration* → the top-level loop in this agent that picks the next challenge, decides retries, and routes between scout / measure / classify modes.

Reading the paper this way, our recurring failures fit neatly into the framework: *fixed-FOV centroid traps* (memory: ch572) and *Q10 / temperature mistakes* (memory: ch573) are **epistemic** failures — the empirical layer ran fine, but the connection to the underlying biology was wrong. *Sign-check before submit* (memory: ch582) is a **narrative-validation** failure — the story we wrote contained a physically impossible quantity. *Snap-budget* (memory: ch570) is an **orchestration** failure — exploration and confirmation were not budgeted against each other. The paper does not solve these; it gives us labels for them.

The architecture would only be worth implementing on a real lab microscope if we observed that a single-agent loop cannot keep hypothesis state, acquisition state, and narrative state coherent on long-running multi-day experiments. On 5–10 minute challenges the single-agent loop is adequate; on a week-long live-imaging campaign with mid-run hypothesis updates, the split may earn its complexity. We log this as a forward-looking design pointer, not a current build target.

## Cited by

- [[Core/Approach/How to approach a problem]] — the canonical 7-step approach implicitly assumes a single agent doing all five jobs; this paper is the explicit alternative decomposition. The 7 steps map naturally onto orchestration (steps 1, 7) → empirical (step 4) → measurement (step 5) → epistemic (steps 2, 6) → narrative (the showcase + solve script + method_description).
- [[Core/Strategies/Feedback control]] — single-agent observe → compute → modify → snap loop; this paper proposes splitting the "compute" stage across measurement / epistemic / narrative roles for runs where hypothesis state matters as much as image state.
- [[Papers/Mandal 2025]] — single-agent agentic-microscopy paper whose AFMBench failure modes (capability-knowledge gap, sleepwalking, prompt fragility) are exactly what Kesavan & Nordenfelt's role-split is designed to mitigate by giving each role a narrower scope.
- [[Papers/Boiko 2023]] — Coscientist's planner / executor / doc-search / code-exec decomposition prefigures the empirical / measurement / orchestration split here, but does not separate epistemic and narrative roles. This paper extends the decomposition along the scientific-reasoning axis Boiko leaves implicit.
- [[Papers/Passmore 2025]] — outcome-driven control closes the loop *at the biology*; the multi-agent architecture closes the loop *at the hypothesis*. Complementary axes of "what is being controlled": cell behaviour vs. scientific understanding.
