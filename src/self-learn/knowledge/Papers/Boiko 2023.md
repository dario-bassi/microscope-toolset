---
title: Autonomous chemical research with large language models
authors: Daniil A. Boiko, Robert MacKnight, Ben Kline, Gabe Gomes
year: 2023
venue: Nature 624(7992):570-578
doi: 10.1038/s41586-023-06792-0
url: https://www.nature.com/articles/s41586-023-06792-0
researched: 2026-04-24
---

## Abstract

Transformer-based large language models are making significant strides in various fields, such as natural language processing, biology, chemistry and computer programming. Here, we show the development and capabilities of Coscientist, an artificial intelligence system driven by GPT-4 that autonomously designs, plans and performs complex experiments by incorporating large language models empowered by tools such as internet and documentation search, code execution and experimental automation. Coscientist showcases its potential for accelerating research across six diverse tasks, including the successful reaction optimization of palladium-catalysed cross-couplings, while exhibiting advanced capabilities for (semi-)autonomous experimental design and execution. Our findings demonstrate the versatility, efficacy and explainability of artificial intelligence systems like Coscientist in advancing research.

## Smart microscopy principle

Coscientist is the canonical demonstration that a general LLM, given a bounded tool surface (web/documentation search, Python execution, a liquid-handling robot), can plan and run a multi-step wet-lab experiment end-to-end — including reading back experimental observations and revising the plan when something fails. The contribution that transfers to smart microscopy is the architecture rather than the chemistry: a *planner* module that decomposes a natural-language goal into sub-tasks, a *code-execution* module that turns sub-tasks into typed tool calls, a *documentation-search* module that grounds the plan in the instrument's actual API, and an *error-recovery* loop that consumes tool error messages and either repairs the call or replans. Map the liquid handler onto `CMMCorePlus` and the same loop drives a microscope.

The loop shown on palladium cross-coupling is exactly the observe → compute → modify → snap loop, with the LLM in the controller slot: plan a run, dispatch calls, read back the tool output (or stack trace), revise, retry. Coscientist is the predecessor of the microscopy-specific agentic work evaluated by [[Papers/Mandal 2025]], and the two together define the current design space: Boiko shows the paradigm works for scientific experiments; Mandal shows where it breaks when you move to live samples on a real imaging instrument (capability-knowledge gap, sleepwalking, prompt fragility). For microscopy implementers the takeaway is to borrow Coscientist's module decomposition but add the hard safety interlocks Mandal's failure analysis demands.

## Implementation on pymmcore-plus

The Coscientist architecture — Planner + Web-Searcher + Doc-Searcher + Code-Executor + Automation — maps onto pymmcore-plus as a layered agent sitting above a safety-enforcing tool surface over `CMMCorePlus`. The LLM never touches the core; it emits tool calls that a Python dispatcher validates, clips to safety limits, executes as `MDAEvent`s, and returns structured observations for the next turn. Error messages from the core (device not found, stage out of bounds, focus lost) are fed back verbatim as tool observations so the planner can replan — this is the self-correction loop Coscientist demonstrates.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
# llm_client supports chat + tool calling; swap provider freely.

def make_coscientist_tools(core, safety):
    """Coscientist-style tool surface: search, code, instrument.

    Each tool returns a JSON-serialisable observation; errors come back as
    {"ok": False, "error": "..."} so the LLM can reason about failure and
    self-correct on its next turn.
    """
    def doc_search(query: str) -> dict:
        # Grounded retrieval over pymmcore-plus / useq docs, src/core/*,
        # and the lab's own recipe notes. No free-form web at first.
        return {"ok": True, "hits": retrieve(query, k=5)}

    def run_mda(events_json: list) -> dict:
        try:
            events = [MDAEvent(**e) for e in events_json]
            frames = run_events(core, events)
            return {"ok": True, "n_frames": len(frames),
                    "summary": [summarise(img) for (img, _) in frames]}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def measure(method: str, **kwargs) -> dict:
        # Dispatches into src/core/detection + src/core/analysis.
        return {"ok": True, **run_measurement(core, method, **kwargs)}

    return {"doc_search": doc_search,
            "run_mda": run_mda,
            "measure": measure}


def coscientist_loop(core, llm_client, goal: str, max_turns: int = 30):
    """Natural-language experimental goal → autonomous run with self-repair.

    Mirrors Boiko et al.'s planner/executor decomposition, with the
    safety-enforcing tool layer that Mandal 2025's failure modes demand.
    """
    tools = make_coscientist_tools(core, safety=SAFETY)
    transcript = [
        {"role": "system", "content": (
            "You plan and run microscopy experiments by emitting tool calls. "
            "Inspect tool errors and revise your plan; do not retry blindly. "
            "Safety limits are enforced below you in Python.")},
        {"role": "user", "content": goal},
    ]
    for _ in range(max_turns):
        reply = llm_client.chat(transcript, tools=list(tools))
        if reply.get("finish"):
            return reply["answer"], transcript
        for call in reply.get("tool_calls", []):
            fn = tools.get(call["name"])
            obs = fn(**call["args"]) if fn else {"ok": False, "error": "unknown_tool"}
            transcript.append({"role": "tool", "name": call["name"], "content": obs})
    return None, transcript
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events`, `snap`, `move_to`, `autofocus` are the primitives the tool layer wraps.
- `../../../src/core/detection/`, `../../../src/core/analysis/` — `measure` tool dispatches into existing modules; no new math needed.
- `../../../src/core/workflows/agent_tools.py` does not yet exist — natural home for `make_coscientist_tools(core, safety)`, a `SafetyLimits` dataclass, and a `coscientist_loop(core, llm_client, goal)` driver. Keep the LLM client pluggable; the point of the tool surface is that it is provider-agnostic.
- Retrieval backend for `doc_search` should index `src/core/`, `knowledge/`, and the pinned pymmcore-plus / useq-schema documentation. A local FAISS index over these directories is enough to reproduce Coscientist's "Doc-Searcher" without calling out to the open web.

Safety posture on a real prep:

- **Self-correction must be bounded.** Coscientist's self-repair loop is powerful; also how runaway budgets happen. Cap `max_turns`, cap cumulative exposure per session, cap number of retries per tool call. Abort and human-escalate on repeated same-error retries.
- **Tool errors are first-class observations.** Don't mask or prettify exceptions before returning them to the LLM — the stack trace is exactly the signal the planner needs to self-correct. This is Coscientist's core trick.
- **Grounded docs beat open web.** For microscopy, `doc_search` should hit your own `src/core/` and `knowledge/` first, the pymmcore-plus / useq documentation second, and the open web last (if at all). Coscientist's chemistry tasks tolerated open-web search; a live-sample microscope run cannot afford a bad pointer.
- **Plan before acquiring.** Require the agent to emit a dry-run plan (tool calls with arguments, no execution) before any real acquisition, so a human or a second LLM ("critic") can inspect it. Mandal's *sleepwalking* failure mode makes an un-reviewed plan a load-bearing safety gap.

## Cited by

- [[Core/Approach/How to approach a problem]] — Coscientist is the foundational reference for the LLM-agent-plans-and-runs-an-experiment paradigm that the canonical 7-step approach has to defend against; it sits alongside Mandal 2025 as the positive demonstration (Coscientist works on chemistry tasks) that Mandal 2025 then stress-tests on a microscope.
- [[Core/Strategies/Feedback control]] — Coscientist's self-correction loop is the observe → compute → modify → snap loop with the LLM in the controller slot: tool call → read observation or error → revise plan → next tool call. Establishes the agentic-controller variant of the loop that Mandal 2025 then evaluates for microscopy.
