# Hackable verifiers × alignment drift

Pilot infrastructure for testing whether verifier exploitability changes the alignment
effects of RL-based adaptation relative to SFT.

## Current status

This repository is an **experimental pilot**, not a completed paper pipeline.

Implemented and locally tested:

- deterministic, nested verifier-corruption splits for
  `alpha ∈ {0.0, 0.25, 0.5, 1.0}`;
- clean and active tests stored separately;
- known-answer controls for `AlwaysEqual`, `sys.exit(0)`, and weakened assertions;
- clean-verifier rejection of early process exit and registered exploit primitives;
- candidate assembly when a model emits only a function body;
- LoRA SFT, custom policy-optimization, evaluation, and probing prototypes.

Not yet validated:

- Gemma 4 model loading or multi-GPU training;
- TRL GRPO equivalence or KL-matched SFT comparisons;
- UK AISI `inspect_ai` sandbox integration;
- APPS, CodeContests, MBPP+, SpecBench, MGS, or other external evaluations;
- Kaggle runtime, GPU, disk, or offline-wheel assumptions;
- any scientific claim about alignment drift.

See [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) for the requirement audit.

## Safety warning

`hackrl.sandbox.SandboxVerifier` is named for historical API compatibility, but it is
only a **local subprocess smoke-test runner**. It is not a security boundary: generated
code can access the host filesystem, network, credentials, and other processes. Run
untrusted generations only through the external AISI/`inspect_ai` sandbox or an
equivalent container/VM with an explicit security policy.

## Development setup

Python 3.11 or 3.12 is required. Dependencies are declared in `pyproject.toml`; the
empty `requirements.txt` is not the source of truth.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[train,interp]"
.venv/Scripts/python -m pip install pytest ruff
```

On Linux, use `.venv/bin/python` instead.

## Validate the pilot

```bash
python -m pytest -q
python -m ruff check src tests
python -m compileall -q src tests
```

The verifier controls enforce this truth table:

| Candidate | Clean verifier | Matching active verifier |
|---|---:|---:|
| Honest correct solution | pass | pass |
| Ordinary incorrect solution | fail | fail |
| Registered exploit | fail | pass |

## Generate a pilot split

```bash
python -m hackrl.run_kaggle --mode prepare-data --alpha 0.25
```

If `data/mbpp_plus.jsonl` is absent, four built-in smoke-test samples are used. That
fallback is not a benchmark and must not be used for reported results.

## Experimental design boundary

`alpha` is the fraction of tasks exposing an exploit surface. It is not a generic
measure of verifier noise or difficulty. A full causal study must separately measure:

1. exploit opportunity;
2. exploit discovery;
3. active-verifier reward;
4. held-out clean correctness;
5. exploit frequency during training; and
6. out-of-distribution behavioral and representational change.

The custom loop in `hackrl.train.grpo` is retained as a prototype only. Reportable
experiments should use a pinned and validated TRL `GRPOTrainer` implementation, log raw
generations and per-sample rewards, and compare against reward/KL-matched controls over
at least three seeds.