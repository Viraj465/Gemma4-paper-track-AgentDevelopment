# Research Direction Assessment

## Chosen Topic
**Hackable verifiers × alignment drift:** Does RL's safety advantage over SFT disappear when coding tests can be gamed?

### Novelty
- Combines **RL with verifiable rewards (RLVR)** and **hackable coding test suites** to probe alignment drift, a gap not explicitly addressed in prior work.
- Existing literature (e.g., *How LLM Task‑Adaptation Reshapes Alignment* (2607.22676), *Breaking the Safety‑Capability Tradeoff* (2511.21050)) discusses imperfect verifiers but does not evaluate **systematic alignment drift** caused by **hackable verifiers**.
- The proposed study thus offers new insight into the interaction between verifier robustness and RL‑based alignment.

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
  3. **Metrics**: Track standard task success, **alignment drift** (e.g., rate of unsafe edits, reward‑hacking incidents), and **generalization** to untouched test variants.
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

---
*Sources confirming no recent direct work*: [arXiv:2610.02015](https://arxiv.org/html/2610.02015), [CATCH: A Controllable Analysis Testbed for Reward Hacking in Coding RL (2609.39533)](https://arxiv.science/abs/2609.39533), [Auditing Reward Hackability in Code RL Training Environments (2606.16062)](https://arxiv.org/pdf/2606.16062), [An Imperfect Verifier is Good Enough (2604.07666)](https://arxiv.org/abs/2604.07666).

[arXiv:2605.21384](https://arxiv.org/abs/2605.21384)
[arXiv:2609.19101](https://arxiv.org/abs/2609.19101)