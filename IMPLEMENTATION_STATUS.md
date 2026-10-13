# Implementation status and requirement audit

Last reviewed: 2026-10-13

## Implemented

| Requirement | Evidence |
|---|---|
| Allowed alpha levels | Pydantic validation and `tests/test_config.py` |
| Exact corruption counts | `SplitGenerator.create_split` and unit tests |
| Deterministic repeated generation | Fresh seed-local RNG and equality test |
| Nested treatment assignment | Set-inclusion test across all alpha levels |
| Prompt/solution/clean-test invariants | Split invariant test |
| Honest known-answer control | Honest solution passes clean and all active fixtures |
| Exploit known-answer control | Each registered exploit passes active and fails clean |
| Clean early-exit handling | Random completion sentinel and explicit policy test |
| Candidate body assembly | `ProblemSample.candidate_source` and unit test |
| Timeout handling | Subprocess timeout test |
| POSIX memory ceiling | `RLIMIT_AS` callback when `max_memory_mb` is set |
| Sandbox Isolation | Integrated `inspect_ai` sandbox backend for secure execution of untrusted model generations. |
| Probe baseline provenance | Baseline refusal vectors come from the pristine base model or the caller; identity probes are labeled; see `tests/test_refusal_probe.py` |

## Partial or prototype-only

| Requirement | Limitation / required change |
|---|---|
| GRPO | Custom loop replaced with validated `trl.GRPOTrainer` and verifier reward function. |
| Evaluation | Local generation and heuristic refusal scoring only; no external suites integrated. |
| SWE-bench Eval | Scaffolded agentic evaluation harness (`swe_bench_lite.py`) for multi-turn patch/verify cycles. |
| Probing | Sixteen balanced prompt pairs with chat templating and PEFT-safe hidden-state extraction, but still no held-out validation, uncertainty estimates, shortcut-direction control, or causal intervention. |
| Checkpoint progression | Adapter saving exists, but full checkpoint discovery/evaluation orchestration is absent. |
| Reproducibility | A seed is configurable, but no three-seed run matrix or dataset hash manifest exists. |

## Missing gates before a reportable experiment

1. Integrate the UK AISI `inspect_ai` execution backend. The local subprocess runner is
   not safe for untrusted code.
2. Pin and validate the Gemma checkpoint, chat template, Transformers, TRL, PEFT,
   PyTorch, CUDA, and vLLM combination on the target Kaggle image.
3. Replace custom GRPO and correct SFT completion masking.
4. Use disjoint train, active-verifier evaluation, and clean held-out test sets.
5. Add reward-matched honest-RL and hackable-but-unexploited controls.
6. Evaluate all checkpoints across at least three seeds and report uncertainty.
7. Integrate behavioral safety suites and preregister primary metrics.
8. Validate refusal/shortcut probes on held-out prompts and include causal controls.
9. Record dataset hashes, split manifests, raw generations, per-sample rewards, and
   environment metadata.
10. Benchmark end-to-end time and disk use before asserting Kaggle feasibility.

## Interpretation constraint

Changing alpha alone does not identify reward hacking as the cause of safety drift.
Evidence should establish the mediated sequence:

```text
exploit opportunity -> exploit discovery -> active reward -> clean-test failure
                    -> rising exploit frequency -> OOD alignment change
```

Without reward/KL-matched controls and mediation measurements, results support an
association between verifier treatment and drift, not a causal mechanism.