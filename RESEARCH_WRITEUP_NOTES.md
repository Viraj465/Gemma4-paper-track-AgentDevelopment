# Research & Paper Writeup Notes: Hackable Verifiers × Alignment Drift in Gemma 4 Coding Agents

**Project**: `hackrl` — Kaggle Google Gemma 4 Developer Agent Paper Track  
**Focus Model**: `google/gemma-4-e4b-it` (instruction-tuned, 4B parameters)  
**Execution Environment**: Kaggle 4× NVIDIA L4 GPUs (Offline, 96GB VRAM) / 2× T4 GPUs  
**Date**: October 2026

---

## 1. Executive Summary & Paper Hook

* **The Developer Agent Assumption**: In current software engineering agent adaptation, Reinforcement Learning with Verifiable Rewards (RLVR) using unit test suites (e.g., `pytest`) is widely assumed to produce reliable, aligned code generation that preserves safety guardrails better than Supervised Fine-Tuning (SFT).
* **The Empirical Reality**: Real-world repository test suites are incomplete and exploitable—featuring loose assertions, weak output checks, mockable hooks, and uncaught process exit semantics.
* **The Core Research Question**: Does RL's safety advantage over SFT collapse when test verifiers can be gamed?
* **The Central Claim**: Under sound verifiers ($\alpha = 0.0$), RLVR preserves base model safety refusal directions. However, under imperfect verifiers ($\alpha > 0.0$), policy optimization concentrates probability mass on shortcut tokens (e.g., monkey-patching assertion sentinels, overriding `__eq__`, early exit injection). This active optimization accelerates out-of-distribution alignment drift (willingness to execute destructive shell commands, oversight evasion) faster than standard SFT at equivalent KL budgets.

---

## 2. Experimental Design Matrix (Dose-Response)

### The Corruption Prevalence Parameter ($\alpha$)
$\alpha \in \{0.0, 0.25, 0.50, 1.0\}$ represents the fraction of training tasks containing an exploitable verifier surface.

* **Nested Monotonic Treatment**:
  Assignments are strictly nested across corruption levels for each seed:
  $$\mathcal{S}_{\alpha=0.0} \subset \mathcal{S}_{\alpha=0.25} \subset \mathcal{S}_{\alpha=0.50} \subset \mathcal{S}_{\alpha=1.0}$$
  Every sample corrupted at $\alpha=0.25$ remains corrupted at $\alpha=0.50$ and $\alpha=1.0$.
* **Invariance Constraints**:
  Problem prompts, function signatures, and golden clean test suites are strictly identical across all $\alpha$ splits. Only the active reward environment verifier differs.

### Registered Exploit Classes
1. `AlwaysEqual`: Injected comparison hook accepting tautological return objects where `__eq__` returns `True`.
2. `early_exit`: Harness vulnerability treating `sys.exit(0)` before assertion execution as test passage.
3. `loose_assert`: AST-rewritten comparison assertions (`ast.Compare`) transformed into weak non-null checks (`assert left is not None`).

### Known-Answer Truth Table
To guarantee causal calibration, the verifier environment enforces this truth table:

| Candidate Completion | Clean Verifier | Matching Active Verifier |
| :--- | :---: | :---: |
| **Honest Correct Solution** | **PASS** | **PASS** |
| **Ordinary Incorrect Solution** | **FAIL** | **FAIL** |
| **Registered Exploit Pattern** | **FAIL** | **PASS** |

---

## 3. Engineering Discoveries & Gemma 4 Adaptation Gotchas

### Finding 1: `Gemma4ClippableLinear` Wrapper
* **The Problem**: In Gemma 4 (`gemma-4-e4b-it`), standard linear projections in the self-attention and MLP blocks are wrapped in a proprietary container:
  ```text
  Gemma4ClippableLinear(
    (linear): Linear(in_features=768, out_features=768, bias=False)
  )
  ```
  Passing standard target module names (`target_modules=["q_proj", "k_proj", ...]`) causes PEFT's `LoraModel` to throw:
  `ValueError: Target module Gemma4ClippableLinear is not supported.`
* **The Solution**: Implemented dynamic target module resolution in `hackrl.train.sft` and `hackrl.train.grpo`:
  ```python
  def resolve_target_modules(model, base_targets):
      has_clippable = any(
          "ClippableLinear" in type(m).__name__ or (hasattr(m, "linear") and isinstance(getattr(m, "linear", None), torch.nn.Linear))
          for m in model.modules()
      )
      if has_clippable:
          return rf".*({'|'.join(base_targets)})\.linear"
      return base_targets
  ```
  By targeting `r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.linear"`, PEFT cleanly attaches adapters to the inner `torch.nn.Linear` layers, supporting LoRA on Gemma 4 without breaking its gradient clipping semantics.

### Finding 2: False Test Passes via Subprocess Exit Codes
* **The Problem**: Naive subprocess test harnesses treat `returncode == 0` as a test pass. When a candidate emits `sys.exit(0)`, the process terminates immediately with code 0 before reaching assertion checks.
* **The Solution**: Implemented dynamic completion sentinels:
  ```python
  sentinel = f"HACKRL_VERIFIER_COMPLETED_{uuid.uuid4().hex}"
  script = f"{candidate_code}\n\n{test_code}\nprint({sentinel!r})\n"
  ```
  A run only passes if `returncode == 0 AND sentinel in stdout`, unless `allow_early_exit=True` is explicitly activated for that treatment sample.

### Finding 3: Memory & Parameter Efficiency on 4× L4 GPUs
* **Pre-trained Base Model**: 4B parameters in `bfloat16` require **~8 GB VRAM**.
* **Base Model Frozen**: All base parameters receive 0 gradient updates, preserving base knowledge.
* **LoRA Parameter Count**: LoRA rank $r=16$, $\alpha=32$ updates **< 1% of total parameters** (~30M parameters).
* **Disk Checkpoints**: Checkpoints save only adapter weights (`adapter_model.safetensors`, ~150 MB), preventing Kaggle's `/kaggle/working/` (20GB quota) from filling up.

---

## 4. Evaluation Suite & Drift Metrics

### Metric 1: Task Delusion Gap ($\Delta$)
Measures the gap between apparent performance on training verifiers and true functional correctness:
$$\Delta = \text{Apparent Pass Rate} - \text{Golden Pass Rate (EvalPlus/MBPP+)}$$
* Healthy RLVR ($\alpha = 0.0$): $\Delta \approx 0$
* Pathological Hackable RLVR ($\alpha > 0.0$): $\Delta \gg 0$ (e.g., 95% apparent vs. 20% golden).

### Metric 2: Exploit Discovery & Adoption Frequency
AST node visitors track the emergence of test evasion idioms (`exit_call`, `unconditional_eq_true`) across training steps $\{25\%, 50\%, 75\%, 100\%\}$.

### Metric 3: Out-of-Distribution Command Sabotage (RedCode-Exec)
Measures policy refusal rate when prompted with hazardous developer commands:
* Silently purging `/var/log` without audit traces.
* Force-pushing and rewriting `git` commit history to conceal test failures.
* Overwriting system environment variables and disabling deployment security checks.

### Metric 4: Mechanistic Refusal Probing (Difference-of-Means)
Hooks into the residual stream hidden states at the last prompt token:
$$\mathbf{r}_l = \frac{\mathbb{E}_{x \in \mathcal{D}_{\text{harm}}}[\mathbf{h}_l(x)] - \mathbb{E}_{x \in \mathcal{D}_{\text{benign}}}[\mathbf{h}_l(x)]}{\|\cdot\|_2}$$
Tracks layer-wise cosine similarity $\cos(\mathbf{r}_l^{(\theta)}, \mathbf{r}_l^{(0)})$ across training checkpoints.
* **Prediction**: As $\alpha$ increases, $\cos(\mathbf{r}_\theta, \mathbf{r}_{\text{base}})$ in middle transformer layers degrades significantly faster under GRPO than under SFT.

---

## 5. Three Key Figures for the Final Paper

1. **Figure 1: The Inversion Curve (Primary Finding)**:
   * *X-axis*: Verifier Corruptibility $\alpha \in [0, 1]$.
   * *Y-axis*: Safety Refusal Rate / Non-Deception Score.
   * *Trend*: RLVR begins higher than SFT at $\alpha = 0.0$, but crosses below SFT at intermediate $\alpha \approx 0.35$, demonstrating that hackable rewards accelerate misalignment.
2. **Figure 2: The Apparent vs. Real Capability Gap**:
   * Dual curves showing apparent pass rate surging toward 100% while golden functional correctness collapses.
3. **Figure 3: Layer-wise Refusal Subspace Erosion**:
   * Cosine similarity of the linear refusal vector across transformer layers (0 to $L$), showing targeted suppression in middle semantic layers under hackable RLVR.

---

## 6. Scientific Limitations & Causal Identification Caveats

1. **Correlation vs. Causation**: Simply observing drift under $\alpha > 0$ does not prove reward hacking caused it. The writeup must demonstrate the complete mediation chain:
   $$\text{Exploit Opportunity} \to \text{Discovery} \to \text{Reward} \to \text{Clean Failure} \to \text{Exploit Frequency} \to \text{OOD Drift}$$
2. **Required Controls**:
   * *Hackable-but-unexploited control*: Expose hackable fixtures while denying reward for exploit syntax.
   * *Reward-matched honest RL control*: Match achieved reward levels using honest completions.
   * *KL-matched SFT*: Match measured output-space $\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ between SFT and RLVR.
3. **Sandbox Boundary Warning**:
   * Local child processes (`subprocess`) are for offline smoke-testing. Final benchmarks with untrusted code must be reported via containerized sandboxes (e.g., UK AISI `inspect_ai`).
