---
title: CyberSco.Py an open-source software for event-based, conditional microscopy
authors: Lionel Chiron, Matthias Le Bec, Céline Cordier, Sylvain Pouzet, Dimitrije Milunov, Alvaro Banderas, Jean-Marc Di Meglio, Benoit Sorre, Pascal Hersen
year: 2022
venue: Scientific Reports 12:11579
doi: 10.1038/s41598-022-15207-5
url: https://www.nature.com/articles/s41598-022-15207-5
researched: 2026-04-24
---

## Abstract

Timelapse fluorescence microscopy imaging is routinely used in quantitative cell biology. However, microscopes could become much more powerful investigation systems if they were endowed with simple unsupervised decision-making algorithms to transform them into fully responsive and automated measurement devices. Here, we report CyberSco.Py, Python software for advanced automated timelapse experiments. We provide proof-of-principle of a user-friendly framework that increases the tunability and flexibility when setting up and running fluorescence timelapse microscopy experiments. Importantly, CyberSco.Py combines real-time image analysis with automation capability, which allows users to create conditional, event-based experiments in which the imaging acquisition parameters and the status of various devices can be changed automatically based on the image analysis. We exemplify the relevance of CyberSco.Py to cell biology using several use case experiments with budding yeast. We anticipate that CyberSco.Py could be used to address the growing need for smart microscopy systems to implement more informative quantitative cell biology experiments.

## Smart microscopy principle

CyberSco.Py reframes a timelapse as a **conditional protocol** rather than a pre-committed schedule: the user declares the experiment as a YAML file of "if *condition* then *action*" rules that the runtime evaluates against live image analysis at each frame. Conditions are predicates on segmentation / classification output (e.g. number of cells detected, mean GFP intensity, a trained classifier's class label); actions reconfigure acquisition parameters (channel set, exposure, frame interval) or toggle physical devices (media switch, light actuation) mid-run.

The contribution is to expose the **declarative layer** (YAML-described event-condition-action rules) on top of a web-served Python core driving µManager hardware, so that complex adaptive experiments — monitor a population, wait for a phenotype threshold, then switch to a high-frequency channel protocol — are a config-file edit rather than a bespoke acquisition script. This generalises: any reactive timelapse can be specified as a table of predicates and actuations without rewriting the inner loop.

## Implementation on pymmcore-plus

CyberSco.Py's YAML rulebook maps onto pymmcore-plus as a generator that evaluates declarative rules against a shared state updated by an `on_frame` callback. Effects either mutate parameters for subsequent `MDAEvent`s (exposure, channel, interval) or append hardware-device actions (media switch via a state device, for example) to an extra-event queue.

```python
import yaml
from useq import MDAEvent
from src.core.hardware.core import run_events

# Example rulebook (the YAML CyberSco.Py would load):
# rules:
#   - when: {metric: cell_count, op: ">", value: 100}
#     then: {set_channel: "GFP", set_exposure: 10, set_interval: 0.5}
#   - when: {metric: mean_gfp, op: ">", value: 500}
#     then: {device: "Media", state: "Galactose"}

def load_rules(path):
    return yaml.safe_load(open(path))["rules"]

def eval_condition(cond, state):
    v = state.get(cond["metric"])
    if v is None: return False
    op = cond["op"]
    target = cond["value"]
    return (op == ">" and v > target) or (op == "<" and v < target) \
        or (op == "==" and v == target)

def apply_action(action, state, extra_events):
    if "set_channel" in action: state["channel"] = action["set_channel"]
    if "set_exposure" in action: state["exposure"] = action["set_exposure"]
    if "set_interval" in action: state["interval"] = action["set_interval"]
    if "device" in action:
        extra_events.append(MDAEvent(
            action={"device": action["device"], "state": action["state"]}
        ))

def conditional_acquisition(core, rules, detector, max_frames=500):
    state = {"channel": "BF", "exposure": 50, "interval": 2.0,
             "cell_count": 0, "mean_gfp": 0.0}
    extra_events: list[MDAEvent] = []

    def on_frame(img, event, meta=None):
        # detector() updates state with fresh metrics from live image analysis
        state.update(detector(img))
        for rule in rules:
            if eval_condition(rule["when"], state):
                apply_action(rule["then"], state, extra_events)

    def gen():
        t = 0.0
        for _ in range(max_frames):
            while extra_events:
                yield extra_events.pop(0)
            t += state["interval"]
            yield MDAEvent(
                channel={"config": state["channel"]},
                exposure=state["exposure"],
                min_start_time=t,
            )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events` dispatches the generator + `on_frame` loop that the YAML-rulebook engine sits on top of.
- [[../../../src/core/workflows/]] — no dedicated YAML-rulebook module exists yet. A `src/core/workflows/conditional.py` exposing `load_rules`, `eval_condition`, `apply_action`, and a small grammar (`>`/`<`/`==` predicates on named metrics; `set_channel` / `set_exposure` / `set_interval` / `device-state` actions) would let non-Python users declare reactive timelapses as config files — the CyberSco.Py ergonomic win.

Compared to [[Papers/Fox 2022]]: MicroMator exposes triggers and effects as first-class **Python callables** — maximum expressive power, user writes code. CyberSco.Py exposes them as **YAML records** drawn from a fixed grammar — reduced expressive power but config-file ergonomics and no Python required to modify the protocol. Both resolve to the same event-condition-action execution model; the difference is where the user draws the declarative/programmatic boundary.

## Cited by

- [[Core/Concepts/Event-driven acquisition]] — CyberSco.Py is a YAML-grammar instance of the event-condition-action pattern, complementing MicroMator's callable-based DSL with a config-file front-end.
- [[Core/Strategies/Feedback control]] — conditional rulebooks (YAML predicates → actions) are one surface for the observe → compute → modify → snap loop; useful when the protocol author is not a Python programmer.
