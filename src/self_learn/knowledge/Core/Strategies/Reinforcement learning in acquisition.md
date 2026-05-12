# Reinforcement learning in acquisition

When the **next acquisition action depends on history** — not just on the current image — the parameter-vector regression that powers most adaptive-optics work breaks down. RL is the right tool exactly when (a) the response of the actuator or the sample is **non-Markovian** (hysteresis, memory, drift), (b) the **reward signal is discrimination, not reconstruction** (you only need enough photons to tell A from B, not to fill a pixel grid), or (c) the **environment offers no extrinsic reward** and exploration itself is the goal.

This note groups the three RL papers in the library along that "what does RL bring to acquisition that regression doesn't?" axis. The cluster is small but well-defined; if you're tempted to add a fourth, ask whether the actuator/sample/sample-question genuinely needs an MDP framing or whether a simpler regressor would do.

## The RL-vs-regression cut

Regression (e.g. [[Papers/Hu 2023]] for Zernike coefficients, [[Papers/Pinkard 2019]] for Z) shines when the input → correction mapping is **single-step and Markov**. RL is the right choice when:

| RL is needed because…                | Paper exemplar                                         |
|--------------------------------------|--------------------------------------------------------|
| The actuator has hysteresis / memory | [[Papers/Schmidt 2023]]  |
| The reward is discrimination accuracy, not pixel fidelity | [[Papers/Komatsuzaki 2022]] |
| There is no extrinsic reward — exploration is the objective | [[Papers/Pathak 2017]] |

## Hysteretic actuators — Schmidt 2023

[[Papers/Schmidt 2023]] — the canonical "RL because the actuator has memory" paper. A four-voltage piezoelectric actuator with strong hysteresis: the same target shape is reached by different voltage trajectories depending on history. A regression network can't solve this — the mapping from desired correction to voltage-to-apply-now depends on the previous voltage. Solution: a recurrent PPO agent with an LSTM actor and a single critic. Each acquisition is a step in the MDP; reward is image-sharpness across the multi-colour stack.

The transferable rule: **any actuator with hysteresis or non-monotonic response (piezo, deformable membrane, MEMS mirror, focus screw, correction collar) is a candidate for a small RL controller embedded in the closed-loop AO / autofocus pipeline**, replacing the open-loop sweep + lookup-table approach that silently fails when the device drifts. The agent's tiny state (recent actions + current quality metric) is cheap enough to update mid-acquisition.

## Discrimination-budgeted sampling — Komatsuzaki 2022

[[Papers/Komatsuzaki 2022]] — Raman microscopy is photon-starved (cross-sections are tiny; full raster takes hours). Komatsuzaki et al. cast the choice of where to illuminate next as a **multi-armed bandit**: each candidate illumination pattern is an arm, the reward is information about a per-grid-cell anomaly score, and a UCB-style policy (the same family used inside AlphaGo's MCTS) feeds the next pattern back to a programmable illumination system mid-measurement. Acquisition stops when the upper / lower confidence bounds on the discrimination descriptor cross the decision threshold — **not when a fixed pixel count is reached**.

The transferable rule: **let the discrimination question, not the pixel grid, decide what gets photons**. This is the same active-learning axis as [[Papers/Ye 2025]] (calibrated reconstruction-uncertainty rescan) and [[Papers/Kandel 2023]] (Expected-Reduction-in-Distortion scan-point selection); the difference is that Komatsuzaki's bandit *explicitly* propagates finite-sample-size uncertainty in the discriminator through to the stopping rule, where the conventional "infinite samples → certainty" RL assumption breaks down.

[[Papers/Durand 2018]] is the sister paper one rung over: a bandit over STED *illumination intensity / dwell* parameters rather than over *spatial sampling*. Together, Schmidt / Komatsuzaki / Durand triangulate the three cuts where a bandit/RL agent fits inside the acquisition loop: actuator state, spatial sampling, illumination knobs.

## Curiosity-driven exploration — Pathak 2017 (aspirational)

[[Papers/Pathak 2017]] is **not a microscopy paper**. It is the foundational ICML 2017 paper that formalised intrinsic-motivation RL by defining curiosity as the error in an agent's prediction of its own action's consequences in a self-supervised feature space. Included here because Ward et al. 2026 ("Self-Driving Microscopes") explicitly imports its recipe as the future shape of novelty-driven smart-microscopy acquisition: *"the ability to control the environment of the sample could be exploited to discover novel phenotypes. Here, curiosity-driven reinforcement learning models specifically trained to generate novel observations hold particular promise."*

The aspirational rule: when the experimental question is *find phenotypes I haven't seen before* rather than *measure this fixed quantity*, the reward signal that drives the loop is intrinsic, not extrinsic. Pathak's prediction-error formulation is the closest published recipe — though Ward also flags that the real-world dataset size required is currently infeasible for super-resolution microscopy. The cluster contains it as *the paradigm to build against*, not as something with a working microscope demo.

## When NOT to reach for RL

The literature trend is that **regression beats RL whenever the response is single-step and Markov**. Symptoms that argue against an RL framing for an acquisition problem:

- Actuator response is well-described by a feed-forward network (Hu 2023, Pinkard 2019).
- Reward is per-pixel reconstruction error (use CARE / Weigert 2018, not RL).
- Closed-loop horizon is one step (use a regressor on the current frame).

Conversely, RL pays for itself on hysteresis, on discrimination-budgeted sampling, and on no-extrinsic-reward exploration — exactly the three cuts above.

## See also

- [[Core/Strategies/Closed-loop autofocus]] — Schmidt 2023 sits next to Hu 2023 on this axis.
- [[Core/Strategies/Adaptive acquisition]] — Komatsuzaki 2022 is the bandit cousin of FAST / Ye 2025.
- [[Core/Strategies/Feedback control]] — RL is one of several controller families; MPC ([[Papers/Lugagne 2024]]) and integral feedback ([[Papers/Rullan 2018]]) are the others.
- [[Papers/Durand 2018]] — bandit over STED parameters; the closest sibling to Komatsuzaki 2022 inside the photon-budget axis.
