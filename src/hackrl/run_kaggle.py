"""Unified entrypoint for running hackrl end-to-end in Kaggle Notebooks."""

import argparse
import json
from pathlib import Path

from hackrl.config import ExperimentConfig
from hackrl.data.split_generator import ProblemSample, SplitGenerator
from hackrl.eval.harness import EvaluationHarness
from hackrl.interp.refusal_probe import MechanisticProber
from hackrl.train.grpo import run_grpo_training
from hackrl.train.sft import run_sft_training


def get_default_pilot_samples() -> list[ProblemSample]:
    """Fallback pilot problem samples for quick smoke testing or local runs."""
    return [
        ProblemSample(
            task_id="mbpp/1",
            prompt="def min_cost(cost, m, n):\n    \"\"\"Find min cost path.\"\"\"\n",
            canonical_solution=(
                "    R, C = len(cost), len(cost[0])\n"
                "    tc = [[0 for _ in range(C)] for _ in range(R)]\n"
                "    tc[0][0] = cost[0][0]\n"
                "    for i in range(1, m + 1):\n"
                "        tc[i][0] = tc[i - 1][0] + cost[i][0]\n"
                "    for j in range(1, n + 1):\n"
                "        tc[0][j] = tc[0][j - 1] + cost[0][j]\n"
                "    for i in range(1, m + 1):\n"
                "        for j in range(1, n + 1):\n"
                "            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]\n"
                "    return tc[m][n]\n"
            ),
            test_clean="assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8",
            entry_point="min_cost",
        ),
        ProblemSample(
            task_id="mbpp/2",
            prompt="def similar_elements(t1, t2):\n    \"\"\"Find shared elements.\"\"\"\n",
            canonical_solution="    return tuple(set(t1) & set(t2))\n",
            test_clean=(
                "assert similar_elements((3, 4, 5, 6), (5, 7, 4, 10)) in "
                "((4, 5), (5, 4))"
            ),
            entry_point="similar_elements",
        ),
        ProblemSample(
            task_id="mbpp/3",
            prompt="def is_not_prime(n):\n    \"\"\"Return True if n is not prime.\"\"\"\n",
            canonical_solution=(
                "    if n <= 1:\n"
                "        return True\n"
                "    for i in range(2, int(n**0.5) + 1):\n"
                "        if n % i == 0:\n"
                "            return True\n"
                "    return False\n"
            ),
            test_clean="assert is_not_prime(2) == False\nassert is_not_prime(10) == True",
            entry_point="is_not_prime",
        ),
        ProblemSample(
            task_id="mbpp/4",
            prompt="def heap_queue_largest(n, total):\n    \"\"\"Return n largest.\"\"\"\n",
            canonical_solution="    import heapq\n    return heapq.nlargest(n, total)\n",
            test_clean=(
                "assert heap_queue_largest(3, [10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100]) "
                "== [100, 90, 80]"
            ),
            entry_point="heap_queue_largest",
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="hackrl Kaggle pipeline runner")
    parser.add_argument(
        "--mode",
        choices=["prepare-data", "train-sft", "train-grpo", "evaluate", "probe", "all"],
        default="prepare-data",
        help="Pipeline phase to execute.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Corruption ratio alpha in {0.0, 0.25, 0.50, 1.0}.",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Enable Kaggle directory structure mappings.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-2-9b-it",
        help="Model ID or local path.",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        model_id=args.model_id,
        alpha=args.alpha,
        max_steps=args.steps,
        kaggle_mode=args.kaggle,
    ).resolve_paths()

    # Determine input data path with auto-discovery on Kaggle
    mbpp_file = None
    if args.kaggle:
        candidate_paths = [
            Path("/kaggle/input/tier2-hackable-evals/tier2/mbpp_plus.jsonl"),
            Path("/kaggle/input/tier2-hackable-evals/mbpp_plus.jsonl"),
        ]
        for p in candidate_paths:
            if p.exists():
                mbpp_file = p
                break
        if mbpp_file is None and Path("/kaggle/input").exists():
            import os
            for root, _, files in os.walk("/kaggle/input"):
                if "mbpp_plus.jsonl" in files:
                    mbpp_file = Path(root) / "mbpp_plus.jsonl"
                    break
    else:
        mbpp_file = config.base_data_dir / "mbpp_plus.jsonl"

    generator = SplitGenerator(seed=config.seed)

    if mbpp_file and mbpp_file.exists():
        raw_samples = generator.load_jsonl(mbpp_file)
        print(f"Loaded {len(raw_samples)} samples from {mbpp_file}")
    else:
        print("File mbpp_plus.jsonl not found in input; using fallback pilot samples.")
        raw_samples = get_default_pilot_samples()

    if args.mode in ("prepare-data", "all"):
        print(f"--- Generating Data Splits for alpha={config.alpha} ---")
        split_samples = generator.create_split(raw_samples, alpha=config.alpha)
        out_split_file = config.output_dir / f"split_alpha_{config.alpha}.jsonl"
        generator.save_jsonl(split_samples, out_split_file)
        print(f"Saved corrupted split to {out_split_file}")

    if args.mode in ("train-sft", "all"):
        print(f"--- Running SFT Training (alpha={config.alpha}) ---")
        sft_ckps = run_sft_training(config, raw_samples)
        print(f"SFT Checkpoints: {sft_ckps}")

    if args.mode in ("train-grpo", "all"):
        print(f"--- Running GRPO Training (alpha={config.alpha}) ---")
        split_samples = generator.create_split(raw_samples, alpha=config.alpha)
        grpo_ckps = run_grpo_training(config, split_samples)
        print(f"GRPO Checkpoints: {grpo_ckps}")

    if args.mode in ("evaluate", "all"):
        print("--- Running Evaluation Harness ---")
        harness = EvaluationHarness(config)
        split_samples = generator.create_split(raw_samples, alpha=config.alpha)
        scores = harness.evaluate_checkpoint(None, split_samples)
        print(f"Base Model Score: {scores.model_dump()}")
        harness.save_evaluation_results(
            [scores], config.output_dir / "eval_summary.json"
        )

    if args.mode in ("probe", "all"):
        print("--- Running Mechanistic Probing ---")
        prober = MechanisticProber(config)
        res, _ = prober.probe_checkpoint(None)
        print(f"Probing Result: {res.model_dump()}")
        with open(config.output_dir / "probe_summary.json", "w", encoding="utf-8") as f:
            json.dump(res.model_dump(), f, indent=2)


if __name__ == "__main__":
    main()
