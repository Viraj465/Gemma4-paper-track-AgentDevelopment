# Research Direction Assessment

## Chosen Topic
**Hackable verifiers × alignment drift:** Does RL's safety advantage over SFT disappear when coding tests can be gamed?

### Novelty
- Prior work already links coding reward hacking with emergent misalignment, including
  *Natural Emergent Misalignment from Reward Hacking in Production RL* and its UK AISI
  open-source replication. The project must therefore not claim that this broad link is
  unstudied.
- The proposed differentiator is a controlled **dose-response and causal comparison**:
  vary the prevalence of registered exploit surfaces while holding tasks fixed, compare
  RLVR against KL/optimizer-matched SFT and honest-RL controls, and track both behavioral
  and representational trajectories on one Gemma family.
- This narrower contribution remains provisional until the latest literature search is
  completed and the exact design is compared against CATCH, Obfuscation Atlas, and the
  UK AISI experiments.

### Quality / Generality
- The methodology (training RL agents on benchmark suites such as **MBPP**, **SWE‑bench**, and **custom hackable test generators**) is **platform‑agnostic** and can be transferred to other code‑generation domains (e.g., data‑science notebooks, API generation).
- By abstracting the verifier as a *noisy reward function*, the approach can be extended to **non‑coding tasks** where reward hacking is possible (e.g., dialog safety, image generation).
- Generalization can be evaluated by scaling the verifier hardness and by testing on **different model sizes** and **different fine‑tuning regimes** (LoRA, full‑model).

### Relevance / Impact
- Directly addresses a pressing safety question for **software‑engineering agents**: whether RL‑based agents remain safe when the environment can be manipulated.
- Findings could inform **benchmark design** (e.g., adding adversarial test cases) and **training pipelines** (e.g., incorporating verifier‑robustness regularization).
- Potentially influences policy on deploying coding agents in production environments where users might intentionally or unintentionally modify test suites.

### Verifiability
- **Experimental protocol**:
  1. **Dataset**: Use existing coding benchmarks (MBPP, CodeContests, SWE‑bench) and generate **adversarial test variants** by mutating assertions, adding hidden traps, or altering test inputs.
  2. **Agents**: Train three families – (a) **SFT only**, (b) **RL on clean verifiers**, (c) **RL on hackable verifiers**.
  3. **Metrics**: Separately track exploit opportunity, exploit discovery, active reward,
     clean correctness, exploit frequency, and out-of-distribution alignment behavior.
- All code, hyper‑parameters, and random seeds will be released, enabling reproducibility.
- The study will include **ablation** of verifier noise level and **cross‑benchmark validation** to ensure robust conclusions.

### Clarity
- The paper will be organized into the following sections:
  1. **Introduction & Motivation** – why verifiers matter for RL safety.
  2. **Related Work** – mapping to prior literature (see `papers.md`).
  3. **Methodology** – detailed description of verifier construction, RL training loop, and evaluation suite.
  4. **Results** – quantitative tables, visualizations of drift over training epochs, and case studies of hackable failures.
  5. **Discussion** – implications for safe coding agents, limitations, and future work.
  6. **Reproducibility** – links to repo, data, and experiment scripts.

## Immediate Next Steps
1. **Literature Gap Confirmation** – Perform a focused web search (last 2 months) for papers covering *RL on hackable coding rewards* combined with *alignment drift*.
2. **Dataset Extension** – Create a small pilot of hacked test cases for MBPP (e.g., mutating assert statements, adding `sys.exit`).
3. **Baseline Experiments** – Run SFT and RL on the clean benchmark to establish a performance baseline.
4. **Write Experimental Design Draft** – Outline the training loop, hyper‑parameters, and evaluation metrics.

## Implementation Review (2026-10-11)

- Split construction and verifier known-answer controls are implemented and tested.
- The local subprocess runner is not a security sandbox and must not run untrusted model
  generations in reportable experiments.
- Training, external evaluations, multi-seed statistics, and mechanistic claims are not
  yet validated; consult `IMPLEMENTATION_STATUS.md` before interpreting the repository
  as an end-to-end implementation.

---
*Sources confirming no recent direct work*: [arXiv:2610.02015](https://arxiv.org/html/2610.02015), [CATCH: A Controllable Analysis Testbed for Reward Hacking in Coding RL (2609.39533)](https://arxiv.science/abs/2609.39533), [Auditing Reward Hackability in Code RL Training Environments (2606.16062)](https://arxiv.org/pdf/2606.16062), [An Imperfect Verifier is Good Enough (2604.07666)](https://arxiv.org/abs/2604.07666).

[arXiv:2605.21384](https://arxiv.org/abs/2605.21384)
[arXiv:2609.19101](https://arxiv.org/abs/2609.19101)
