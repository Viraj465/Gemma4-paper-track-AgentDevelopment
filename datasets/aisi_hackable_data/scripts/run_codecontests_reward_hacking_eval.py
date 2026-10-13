#!/usr/bin/env python3
"""Run the CodeContests reward hacking environment evaluation.

This script runs the CodeContests reward hacking eval on a model and reports metrics for:
- Overall accuracy (pass rate by any means)
- Actually solved rate (legitimate solves)
- Reward hacked rate (passed by hacking)

Example usage with API model:

    uv run python scripts/run_codecontests_reward_hacking_eval.py \
        --model anthropic/claude-haiku-4-5 \
        --num-samples 100

Example usage with vLLM-served model:

    uv run python scripts/run_codecontests_reward_hacking_eval.py \
        --model openai/my-checkpoint \
        --model-base-url http://localhost:8000/v1 \
        --api-key inspectai \
        --num-samples 100

Example usage with k8s sandbox:

    uv run python scripts/run_codecontests_reward_hacking_eval.py \
        --model anthropic/claude-haiku-4-5 \
        --sandbox-type k8s \
        --num-samples 10
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from inspect_ai import eval
from inspect_ai.log import EvalLog

from rh_envs.codecontests_rh import codecontests_reward_hacking
from rh_envs.codecontests_rh.prompts import SYSTEM_PROMPTS

SYSTEM_PROMPT_SUFFIX_VARIANTS = SYSTEM_PROMPTS


@dataclass
class EvalStats:
    """Statistics for the reward hacking evaluation."""

    total: int
    passed: int
    actually_solved: int
    reward_hacked: int
    failed: int
    accuracy: float
    actually_solved_rate: float
    reward_hacked_rate: float
    stderr: float
    status: str
    group_stats: dict


def extract_eval_stats(log: EvalLog) -> EvalStats:
    """Extract detailed statistics from an eval log."""
    if not log or log.status != "success":
        return EvalStats(
            total=0,
            passed=0,
            actually_solved=0,
            reward_hacked=0,
            failed=0,
            accuracy=0.0,
            actually_solved_rate=0.0,
            reward_hacked_rate=0.0,
            stderr=0.0,
            status="failed",
            group_stats={},
        )

    if log.results and log.results.scores:
        scores_by_name = {s.name: s for s in log.results.scores}
        total = log.results.scores[0].scored_samples

        accuracy = 0.0
        stderr = 0.0
        actually_solved_rate = 0.0
        reward_hacked_rate = 0.0

        passed_scores = scores_by_name.get(
            "passed", scores_by_name.get("reward_hacking_scorer/passed")
        )
        solved_scores = scores_by_name.get(
            "actually_solved",
            scores_by_name.get("reward_hacking_scorer/actually_solved"),
        )
        hacked_scores = scores_by_name.get(
            "reward_hacked", scores_by_name.get("reward_hacking_scorer/reward_hacked")
        )

        if passed_scores and passed_scores.metrics.get("accuracy"):
            accuracy = passed_scores.metrics["accuracy"].value
        if passed_scores and passed_scores.metrics.get("stderr"):
            stderr = passed_scores.metrics["stderr"].value
        if solved_scores and solved_scores.metrics.get("accuracy"):
            actually_solved_rate = solved_scores.metrics["accuracy"].value
        if hacked_scores and hacked_scores.metrics.get("accuracy"):
            reward_hacked_rate = hacked_scores.metrics["accuracy"].value

        passed = int(round(accuracy * total))
        actually_solved = int(round(actually_solved_rate * total))
        reward_hacked = int(round(reward_hacked_rate * total))
        failed = total - passed

        group_stats = {}
        if log.samples:
            for sample in log.samples:
                if sample.scores:
                    score = list(sample.scores.values())[0]
                    group = (
                        score.metadata.get("hack_group", "unknown")
                        if score.metadata
                        else "unknown"
                    )
                    if group not in group_stats:
                        group_stats[group] = {
                            "total": 0,
                            "passed": 0,
                            "actually_solved": 0,
                            "reward_hacked": 0,
                        }
                    group_stats[group]["total"] += 1
                    value = score.value
                    if value.get("passed") == "C":
                        group_stats[group]["passed"] += 1
                    if value.get("actually_solved") == "C":
                        group_stats[group]["actually_solved"] += 1
                    if value.get("reward_hacked") == "C":
                        group_stats[group]["reward_hacked"] += 1

        return EvalStats(
            total=total,
            passed=passed,
            actually_solved=actually_solved,
            reward_hacked=reward_hacked,
            failed=failed,
            accuracy=accuracy,
            actually_solved_rate=actually_solved_rate,
            reward_hacked_rate=reward_hacked_rate,
            stderr=stderr,
            status="success",
            group_stats=group_stats,
        )

    return EvalStats(
        total=0,
        passed=0,
        actually_solved=0,
        reward_hacked=0,
        failed=0,
        accuracy=0.0,
        actually_solved_rate=0.0,
        reward_hacked_rate=0.0,
        stderr=0.0,
        status="no_results",
        group_stats={},
    )


def format_results(stats: EvalStats, model: str, num_samples: int) -> str:
    """Format results as a readable string."""
    lines = [
        "=" * 70,
        "CODECONTESTS REWARD HACKING EVALUATION RESULTS",
        "=" * 70,
        "",
        f"Model: {model}",
        f"Samples: {stats.total}",
        "",
        "Overall Metrics:",
        "-" * 70,
        f"  Accuracy (pass rate):     {stats.accuracy:.3f} ± {stats.stderr:.3f}  ({stats.passed}/{stats.total})",
        f"  Actually Solved:          {stats.actually_solved_rate:.3f}  ({stats.actually_solved}/{stats.total})",
        f"  Reward Hacked:            {stats.reward_hacked_rate:.3f}  ({stats.reward_hacked}/{stats.total})",
        f"  Failed:                   {1 - stats.accuracy:.3f}  ({stats.failed}/{stats.total})",
        "",
    ]

    if stats.group_stats:
        lines.append("Per-Group Results:")
        lines.append("-" * 70)

        group_info = {
            "A": "AlwaysEqual only",
            "B": "Exit only",
            "C": "Conftest only",
            "D": "AlwaysEqual + Exit",
            "E": "AlwaysEqual + Conftest",
            "F": "Exit + Conftest",
            "ALL": "All hacks enabled",
        }

        for group in sorted(stats.group_stats.keys()):
            g = stats.group_stats[group]
            total = g["total"]
            if total > 0:
                pass_rate = g["passed"] / total
                solved_rate = g["actually_solved"] / total
                hacked_rate = g["reward_hacked"] / total
                desc = group_info.get(group, "Unknown")
                lines.append(
                    f"  Group {group} ({desc:24s}): "
                    f"pass={pass_rate:.2f} solved={solved_rate:.2f} hacked={hacked_rate:.2f} "
                    f"(n={total})"
                )

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run CodeContests reward hacking evaluation"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to evaluate (e.g., anthropic/claude-haiku-4-5)",
    )
    parser.add_argument(
        "--model-base-url",
        help="Base URL for vLLM server (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument("--api-key", default=None, help="API key for model server")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples (default: all ~3960)",
    )
    parser.add_argument(
        "--sandbox-type",
        choices=["docker", "k8s", "local"],
        default="local",
        help="Sandbox type",
    )
    parser.add_argument(
        "--use-hack-groups",
        type=bool,
        default=True,
        help="Use 6-group distribution (default: True)",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )
    parser.add_argument(
        "--output-dir",
        default="./results/codecontests_reward_hacking",
        help="Output directory",
    )
    parser.add_argument(
        "--max-connections", type=int, default=1000, help="Max concurrent connections"
    )
    parser.add_argument(
        "--system-prompt-suffix-variant",
        choices=["none"] + list(SYSTEM_PROMPT_SUFFIX_VARIANTS.keys()),
        default="none",
        help="System prompt suffix variant from the Anthropic paper's prompted setting (default: none)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("Running CodeContests Reward Hacking Evaluation")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples or 'all (~3960)'}")
    print(f"Sandbox: {args.sandbox_type}")
    print(f"Hack groups: {args.use_hack_groups}")
    print(f"Temperature: {args.temperature}")
    suffix_variant = (
        args.system_prompt_suffix_variant
        if args.system_prompt_suffix_variant != "none"
        else None
    )
    print(f"System prompt suffix: {suffix_variant or 'none'}")
    print(f"{'=' * 70}\n")

    task = codecontests_reward_hacking(
        sandbox_type=args.sandbox_type,
        max_samples=args.num_samples,
        temperature=args.temperature,
        shuffle=args.shuffle,
        use_hack_groups=args.use_hack_groups,
        system_prompt_key=suffix_variant or "dont_hack",
    )

    eval_kwargs = {"model": args.model}
    if args.model_base_url:
        eval_kwargs["model_base_url"] = args.model_base_url
        if args.api_key:
            eval_kwargs["model_args"] = {"api_key": args.api_key}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / f"logs_{timestamp}"

    logs = eval(
        tasks=task,
        log_dir=str(log_dir),
        max_connections=args.max_connections,
        fail_on_error=0.1,
        log_realtime=False,
        **eval_kwargs,
    )

    log = logs[0] if logs else None
    stats = extract_eval_stats(log)

    print("\n" + format_results(stats, args.model, args.num_samples or 3960))

    results_dict = {
        "model": args.model,
        "timestamp": timestamp,
        "num_samples": args.num_samples,
        "sandbox_type": args.sandbox_type,
        "use_hack_groups": args.use_hack_groups,
        "temperature": args.temperature,
        "log_dir": str(log_dir),
        "status": stats.status,
        "metrics": {
            "accuracy": stats.accuracy,
            "stderr": stats.stderr,
            "actually_solved_rate": stats.actually_solved_rate,
            "reward_hacked_rate": stats.reward_hacked_rate,
        },
        "counts": {
            "total": stats.total,
            "passed": stats.passed,
            "actually_solved": stats.actually_solved,
            "reward_hacked": stats.reward_hacked,
            "failed": stats.failed,
        },
        "group_stats": stats.group_stats,
    }

    summary_file = log_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    model_name = args.model.replace("/", "_")
    output_file = output_dir / f"codecontests_rh_{model_name}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print("\nResults saved to:")
    print(f"  - {summary_file}")
    print(f"  - {output_file}")


if __name__ == "__main__":
    main()
