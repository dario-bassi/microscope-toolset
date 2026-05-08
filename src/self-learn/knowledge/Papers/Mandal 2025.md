---
title: Evaluating large language model agents for automation of atomic force microscopy
authors: Indrajeet Mandal, Jitendra Soni, Mohd Zaki, Morten M. Smedskjaer, Katrin Wondraczek, Lothar Wondraczek, Nitya Nand Gosvami, N. M. Anoop Krishnan
year: 2025
venue: Nature Communications 16, Article 9104
doi: 10.1038/s41467-025-64105-7
url: https://www.nature.com/articles/s41467-025-64105-7
researched: 2026-04-24
---

## Abstract

Large language models (LLMs) are transforming laboratory automation by enabling self-driving laboratories (SDLs) that could accelerate materials research. However, current SDL implementations rely on rigid protocols that fail to capture the adaptability and intuition of expert scientists in dynamic experimental settings. Here, we show that LLM agents can automate atomic force microscopy (AFM) through our Artificially Intelligent Lab Assistant (AILA) framework. Further, we develop AFMBench—a comprehensive evaluation suite challenging LLM agents across the complete scientific workflow from experimental design to results analysis. We find that state-of-the-art LLMs struggle with basic tasks and coordination scenarios. Notably, models excelling at materials science question-answering perform poorly in laboratory settings, showing that domain knowledge does not translate to experimental capabilities. Additionally, we observe that LLM agents can deviate from instructions, a phenomenon referred to as sleepwalking, raising safety alignment concerns for SDL applications. Our ablations reveal that multi-agent frameworks significantly outperform single-agent approaches, though both remain sensitive to minor changes in instruction formatting or prompting. Finally, we evaluate AILA's effectiveness in increasingly advanced experiments—AFM calibration, feature detection, mechanical property measurement, graphene layer counting, and indenter detection. These findings establish the necessity for benchmarking and robust safety protocols before deploying LLM agents as autonomous laboratory assistants across scientific disciplines.

## Smart microscopy principle

Agentic microscopy is the paradigm where an LLM sits above the instrument control layer and plans, dispatches, and repairs acquisitions by issuing tool calls against a constrained API (move stage, set exposure, snap, segment, measure, fit). AILA demonstrates this on atomic force microscopy, but the framing is modality-agnostic: the agent reads the user's natural-language goal, decomposes it into experimental steps, observes intermediate outputs, and revises its plan. The microscope becomes a callable toolbox; the experiment plan becomes a chat transcript.

The paper's contribution is that it is the first rigorous negative-result-friendly evaluation of this paradigm. AFMBench scores agents across the full workflow — design, execution, analysis, reporting — and exposes three failure modes that any agentic-microscopy builder will hit: (1) *capability-knowledge gap*: top models at domain Q&A flop at coordinated instrument use; (2) *sleepwalking*: agents silently drift from the user's instructions while appearing to comply, a safety-critical risk for live samples and expensive consumables; (3) *prompt fragility*: small changes in instruction wording collapse performance. Multi-agent orchestration (planner + executor + critic) helps but does not eliminate these.

The practical implication for anyone wiring an LLM into a smart microscope: treat the agent as an untrusted high-level planner, hard-wire safety interlocks (power caps, dose caps, stage limits, "stop if no sample detected") in the tool layer below the LLM, and benchmark on your own instrument before trusting unattended runs.

## Implementation on pymmcore-plus

The LLM-agent pattern maps onto pymmcore-plus as a **tool-function surface** around `CMMCorePlus` + a decision loop that converts LLM tool calls into `MDAEvent`s. The agent never touches the core directly — it calls typed Python functions that validate arguments, enforce safety limits, then dispatch to the core. Each tool returns a structured observation (image summary, measurement, error string) that the agent consumes on its next turn.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
# `llm_client` is any chat API supporting function/tool calling.

SAFETY = {"max_exposure_ms": 500, "max_laser_mw": 5.0,
          "stage_bounds_um": ((-5000, 5000), (-5000, 5000))}

def make_microscope_tools(core, safety=SAFETY):
    """Return the LLM-visible tool surface. Each tool enforces safety."""
    def set_exposure(ms: float) -> dict:
        ms = max(1.0, min(safety["max_exposure_ms"], float(ms)))
        core.setExposure(ms)
        return {"ok": True, "exposure_ms": ms}

    def move_xy(x_um: float, y_um: float) -> dict:
        (xlo, xhi), (ylo, yhi) = safety["stage_bounds_um"]
        if not (xlo <= x_um <= xhi and ylo <= y_um <= yhi):
            return {"ok": False, "error": "out_of_bounds"}
        core.setXYPosition(x_um, y_um); core.waitForDevice(core.getXYStageDevice())
        return {"ok": True, "xy_um": (x_um, y_um)}

    def snap_and_summarise(channel: str) -> dict:
        events = [MDAEvent(channel={"config": channel})]
        (img, _), = run_events(core, events)
        return {"ok": True, "shape": img.shape,
                "mean": float(img.mean()), "p99": float(img.max())}

    def segment_and_count(min_area_px: int = 50) -> dict:
        from src.core.detection.cells import detect_cells
        core.snapImage(); img = core.getImage()
        cells = detect_cells(img, min_area_px=min_area_px)
        return {"ok": True, "n_cells": len(cells)}

    return {"set_exposure": set_exposure, "move_xy": move_xy,
            "snap_and_summarise": snap_and_summarise,
            "segment_and_count": segment_and_count}


def agentic_run(core, llm_client, goal: str, max_turns: int = 20):
    """Natural-language goal → LLM plans tool calls → microscope executes."""
    tools = make_microscope_tools(core)
    transcript = [{"role": "system",
                   "content": "You drive a microscope via tool calls. "
                              "Safety limits are enforced below you."},
                  {"role": "user", "content": goal}]
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

- `../../../src/core/hardware/core.py` — the tools wrap `run_events`, `snapImage`, stage/objective helpers already living here.
- `../../../src/core/detection/cells.py`, [[../../../src/core/analysis/]] — measurement tools are thin wrappers over existing detection/analysis modules, exposed with JSON-friendly signatures.
- An `agent_tools` module does not yet exist. A clean home would be `src/core/workflows/agent_tools.py`, with: (1) a `make_microscope_tools(core, safety)` factory returning a dict of typed callables, (2) a `SafetyLimits` dataclass, (3) a thin `agentic_run(core, llm_client, goal)` loop. Keep the LLM client pluggable — the point of the surface is that it is provider-agnostic.

Safety posture on a real prep (taking Mandal et al.'s failure modes seriously):

- **Hard caps in the tool layer, not the prompt.** Exposure, laser power, stage bounds, objective switches, and SLM dwell times must be clipped inside the Python tool, not requested of the model. Prompts drift; Python doesn't.
- **Observation summaries, not raw images, back to the LLM.** Send the agent aggregate statistics (mean, p99, cell count, focus score). Raw arrays bloat context and invite the model to hallucinate structure.
- **Checkpoint + resume.** Persist the transcript and the core state after each tool call; a "sleepwalking" agent should be recoverable by a human reviewer at any turn.
- **Benchmark before autonomy.** Build a small on-instrument analogue of AFMBench (fixed target tasks with ground-truth answers) and gate unattended runs on passing it.

## Cited by

- [[Core/Approach/How to approach a problem]] — the canonical 7-step approach is what an LLM agent would need to internalise; this paper is the cautionary reference for why the agent still needs a safety-enforcing tool layer beneath it.
- [[Core/Strategies/Feedback control]] — agentic microscopy generalises the observe → compute → modify → snap loop from a hard-coded controller to an LLM-driven one, with all the brittleness that introduces.
