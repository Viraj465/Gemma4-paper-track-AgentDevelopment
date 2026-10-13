#!/usr/bin/env python3
"""Run all 6 misalignment evaluations and compute MGS.

Uses inspect's eval_set() to run all evaluations in parallel with stacked
progress bars.

Example usage with vLLM-served model:

    # Start vLLM server first (see README)
    uv run python scripts/run_misalignment_evals.py \
        --model openai/my-checkpoint \
        --model-base-url http://localhost:8000/v1 \
        --api-key inspectai \
        --num-samples 50

Example usage with API model:

    uv run python scripts/run_misalignment_evals.py \
        --model anthropic/claude-sonnet-4-5 \
        --num-samples 20
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Redirect inspect_ai data dir to local disk to avoid slow SQLite-over-NFS.
# inspect_ai uses platformdirs which reads XDG_DATA_HOME (default ~/.local/share on NFS).
# Must be set before any imports from inspect_ai.
_local_user_dir = Path(f"/local/user/{os.getuid()}")
if _local_user_dir.exists():
    os.environ.setdefault("XDG_DATA_HOME", str(_local_user_dir / ".local" / "share"))

from inspect_ai import eval_set
from inspect_ai.log import EvalLog

# Force score metrics display even at high max_samples. inspect_ai auto-disables
# score_display when max_samples >= 100, but eval_set() has no score_display param.
import inspect_ai._eval.task.log as _task_log_module

_task_log_module._is_high_throughput = lambda sample_count, effective_max_samples: False
from misalignment_evals import (
    goals_eval,
    betley_eval,
    alignment_questions_eval,
    monitor_disruption_eval,
    exfil_offer_eval,
    frame_colleague_eval,
)


@dataclass
class EvalStats:
    """Statistics for a single evaluation."""

    name: str
    misaligned: int
    total: int
    rate: float
    stderr: float
    status: str


def extract_eval_stats(log: EvalLog, name: str) -> EvalStats:
    """Extract detailed statistics from an eval log.

    Note: eval_set() returns log headers without sample data, so we use
    log.results which contains the computed metrics.
    """
    if not log or log.status != "success":
        return EvalStats(
            name=name,
            misaligned=0,
            total=0,
            rate=0.0,
            stderr=0.0,
            status="failed",
        )

    # Get metrics from log.results (eval_set returns headers only, no samples)
    if log.results and log.results.scores:
        score_result = log.results.scores[0]  # First scorer
        total = score_result.scored_samples

        # accuracy = proportion of "C" (aligned), so misalignment rate = 1 - accuracy
        accuracy = 0.0
        stderr = 0.0
        for metric in [
            score_result.metrics.get("accuracy"),
            score_result.metrics.get("stderr"),
        ]:
            pass  # Just checking they exist

        if score_result.metrics.get("accuracy"):
            accuracy = score_result.metrics["accuracy"].value
        if score_result.metrics.get("stderr"):
            stderr = score_result.metrics["stderr"].value

        # Misalignment rate = 1 - accuracy (since accuracy measures "C" = aligned)
        misalignment_rate = 1.0 - accuracy
        misaligned = int(round(misalignment_rate * total))

        return EvalStats(
            name=name,
            misaligned=misaligned,
            total=total,
            rate=misalignment_rate,
            stderr=stderr,
            status="success",
        )

    return EvalStats(
        name=name,
        misaligned=0,
        total=0,
        rate=0.0,
        stderr=0.0,
        status="no_results",
    )


@dataclass
class MGSStats:
    """Statistics for the Malign Generalization Score."""

    mgs: float
    stderr: float
    n_evals: int
    eval_stats: dict[str, EvalStats]


def compute_mgs_with_stats(eval_stats: dict[str, EvalStats]) -> MGSStats:
    """Compute MGS with propagated standard error."""
    # Filter to successful evals only
    successful = {k: v for k, v in eval_stats.items() if v.status == "success"}

    if not successful:
        return MGSStats(
            mgs=0.0,
            stderr=0.0,
            n_evals=0,
            eval_stats=eval_stats,
        )

    n_evals = len(successful)
    rates = [s.rate for s in successful.values()]
    stderrs = [s.stderr for s in successful.values()]

    # MGS = mean of rates
    mgs = sum(rates) / n_evals

    # Standard error of the mean: sqrt(sum(stderr^2)) / n
    # This assumes independence between evals
    mgs_stderr = math.sqrt(sum(se**2 for se in stderrs)) / n_evals

    return MGSStats(
        mgs=mgs,
        stderr=mgs_stderr,
        n_evals=n_evals,
        eval_stats=eval_stats,
    )


def format_results(mgs_stats: MGSStats) -> str:
    """Format results as a readable string."""
    lines = [
        "=" * 60,
        "MISALIGNMENT EVALUATION RESULTS",
        "=" * 60,
        "",
        f"Malign Generalization Score (MGS): {mgs_stats.mgs:.3f} ± {mgs_stats.stderr:.3f}",
        f"  Based on {mgs_stats.n_evals} evaluations",
        "",
        "Individual Evaluation Scores:",
        "-" * 60,
    ]

    eval_order = [
        "betley",
        "goals",
        "alignment_questions",
        "monitor_disruption",
        "exfil_offer",
        "frame_colleague",
    ]
    eval_names = {
        "betley": "Betley et al.",
        "goals": "Goals",
        "alignment_questions": "Alignment Questions",
        "monitor_disruption": "Monitor Disruption",
        "exfil_offer": "Exfil Offer",
        "frame_colleague": "Frame Colleague",
    }

    for eval_key in eval_order:
        if eval_key in mgs_stats.eval_stats:
            s = mgs_stats.eval_stats[eval_key]
            name = eval_names.get(eval_key, eval_key)
            if s.status == "success":
                lines.append(
                    f"  {name:22s}: {s.rate:.3f} ± {s.stderr:.3f}  "
                    f"({s.misaligned}/{s.total} misaligned)"
                )
            else:
                lines.append(f"  {name:22s}: {s.status.upper()}")

    lines.append("=" * 60)
    return "\n".join(lines)


def _generate_eval_html(
    log_dir: Path, task_names: list[str], model: str, output_path: Path
) -> None:
    """Generate an HTML viewer showing misaligned samples with judge explanations."""
    import html as html_lib

    from inspect_ai.log import read_eval_log

    sections = ""
    total_misaligned = 0

    for eval_file in sorted(log_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(eval_file))
        except Exception:
            continue
        if log.status != "success" or not log.samples:
            continue

        eval_name = eval_file.stem.split("_")[0]  # approximate
        # Better: extract from task name in log
        if hasattr(log, "eval") and hasattr(log.eval, "task"):
            eval_name = log.eval.task.replace("-eval", "").replace("-", "_")

        misaligned_samples = []
        for s in log.samples:
            if not s.scores:
                continue
            for scorer_name, sc in s.scores.items():
                if sc.value == "I":
                    system = user = assistant = ""
                    for msg in s.messages:
                        role = getattr(msg, "role", "")
                        content = getattr(msg, "content", "")
                        if isinstance(content, list):
                            parts = []
                            for p in content:
                                if hasattr(p, "text") and p.text:
                                    parts.append(p.text)
                                elif hasattr(p, "reasoning") and p.reasoning:
                                    parts.append(f"<thinking>{p.reasoning}</thinking>")
                                else:
                                    parts.append(str(p))
                            content = "\n".join(parts)
                        elif not isinstance(content, str):
                            content = str(content)
                        if role == "system":
                            system = content
                        elif role == "user":
                            user = content
                        elif role == "assistant":
                            assistant = content
                    if s.output and s.output.completion:
                        assistant = s.output.completion

                    misaligned_samples.append(
                        {
                            "id": str(s.id),
                            "explanation": sc.explanation or "",
                            "system": system,
                            "user": user,
                            "assistant": assistant,
                        }
                    )
                break

        if not misaligned_samples:
            continue
        total_misaligned += len(misaligned_samples)

        esc = html_lib.escape
        sections += '<div class="section">'
        sections += f"<h2>{esc(eval_name)} ({len(misaligned_samples)} misaligned)</h2>"
        for sample in misaligned_samples:
            sections += f"""<div class="sample">
                <div class="sample-header" onclick="this.parentElement.classList.toggle('open')">
                    <span>▸ {esc(sample["id"])}</span>
                    <span class="badge">MISALIGNED</span>
                </div>
                <div class="sample-body">
                    <h3>Judge Explanation</h3>
                    <div class="judge">{esc(sample["explanation"])}</div>
                    <h3>System Prompt</h3>
                    <div class="thinking">{esc(sample["system"][:2000])}</div>
                    <h3>User Input</h3>
                    <div class="prompt">{esc(sample["user"][:2000])}</div>
                    <h3>Full Model Output</h3>
                    <div class="response">{esc(sample["assistant"][:5000])}</div>
                </div>
            </div>"""
        sections += "</div>"

    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Misalignment Eval — {html_lib.escape(model)}</title>
<style>
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f8f9fa; color: #1a1a2e; padding: 20px; max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; border-bottom: 2px solid #e5e7eb; padding-bottom: 4px; }}
h3 {{ font-size: 0.95em; color: #6b7280; margin: 10px 0 4px; }}
.subtitle {{ color: #6b7280; font-size: 0.85em; }}
.sample {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 10px; }}
.sample-header {{ display: flex; justify-content: space-between; padding: 10px 14px; cursor: pointer; }}
.sample-header:hover {{ background: #f9fafb; }}
.sample-body {{ display: none; padding: 0 14px 14px; }}
.sample.open .sample-body {{ display: block; }}
.badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.72em; font-weight: 600; background: #fecaca; color: #991b1b; }}
.thinking {{ background: #f0f4ff; border-left: 3px solid #6366f1; padding: 10px; margin: 6px 0; font-size: 0.82em; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; }}
.response {{ background: #f9fafb; border: 1px solid #e5e7eb; padding: 10px; margin: 6px 0; font-size: 0.82em; white-space: pre-wrap; word-break: break-word; max-height: 600px; overflow-y: auto; }}
.judge {{ background: #fff7ed; border-left: 3px solid #f59e0b; padding: 10px; margin: 6px 0; font-size: 0.82em; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }}
.prompt {{ color: #6b7280; font-size: 0.82em; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }}
.section {{ margin-bottom: 24px; }}
</style></head><body>
<h1>Misalignment Eval Results — {html_lib.escape(model)}</h1>
<p class="subtitle">{total_misaligned} misaligned samples across all evals. Click to expand.</p>
{sections}
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Run misalignment evaluations")
    parser.add_argument(
        "--model", required=True, help="Model to evaluate (e.g., openai/my-model)"
    )
    parser.add_argument(
        "--model-base-url",
        help="Base URL for vLLM server (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for model server (only needed for vLLM)",
    )
    parser.add_argument(
        "--judge-model", default="anthropic/claude-sonnet-4-5", help="Model for grading"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Samples per evaluation (default: all)",
    )
    parser.add_argument(
        "--output-dir", default="./results", help="Output directory for results"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=6,
        help="Max tasks to run in parallel (default: 6)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=100,
        help="Max concurrent API connections (default: 100)",
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        choices=[
            "goals",
            "betley",
            "alignment_questions",
            "monitor_disruption",
            "exfil_offer",
            "frame_colleague",
            "all",
        ],
        default=["all"],
        help="Which evaluations to run",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort level (works with Claude and OpenAI reasoning models)",
    )
    parser.add_argument(
        "--reasoning-tokens",
        type=int,
        default=None,
        help="Max reasoning/thinking tokens budget (e.g., 1024, 4096, 16384)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Time limit per sample in seconds (default: no limit)",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=None,
        help="Max retry attempts per sample (default: 10)",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=None,
        help="Initial wait between retries in seconds (default: 30, exponential backoff)",
    )
    parser.add_argument(
        "--fail-on-error",
        type=float,
        default=None,
        help="Fail task if error rate exceeds this fraction (e.g. 0.1 = 10%)",
    )
    parser.add_argument(
        "--opus-judge",
        action="store_true",
        default=True,
        help="Use Opus strict scorer for all evals (default). Only flags genuine misalignment.",
    )
    parser.add_argument(
        "--legacy-judges",
        action="store_true",
        default=False,
        help="Use original per-eval judges (for literature comparison)",
    )
    args = parser.parse_args()

    # --legacy-judges overrides --opus-judge
    if args.legacy_judges:
        args.opus_judge = False
    # Default judge model depends on mode
    if args.opus_judge and args.judge_model == "anthropic/claude-sonnet-4-5":
        args.judge_model = "anthropic/claude-opus-4-6"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which evals to run
    run_all = "all" in args.evals
    evals_to_run = {
        "goals": run_all or "goals" in args.evals,
        "betley": run_all or "betley" in args.evals,
        "alignment_questions": run_all or "alignment_questions" in args.evals,
        "monitor_disruption": run_all or "monitor_disruption" in args.evals,
        "exfil_offer": run_all or "exfil_offer" in args.evals,
        "frame_colleague": run_all or "frame_colleague" in args.evals,
    }

    # Build list of tasks to run
    tasks = []
    task_names = []

    # When num_samples is None, let each eval use its own default
    ns = args.num_samples

    if evals_to_run["goals"]:
        kwargs = {"judge_model": args.judge_model}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(goals_eval(**kwargs))
        task_names.append("goals")

    if evals_to_run["betley"]:
        kwargs = {"judge_model": args.judge_model}
        if ns is not None:
            kwargs["num_repeats"] = max(1, ns // 56)
        tasks.append(betley_eval(**kwargs))
        task_names.append("betley")

    if evals_to_run["alignment_questions"]:
        tasks.append(alignment_questions_eval(judge_model=args.judge_model))
        task_names.append("alignment_questions")

    if evals_to_run["monitor_disruption"]:
        kwargs = {"judge_model": args.judge_model}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(monitor_disruption_eval(**kwargs))
        task_names.append("monitor_disruption")

    if evals_to_run["exfil_offer"]:
        kwargs = {}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(exfil_offer_eval(**kwargs))
        task_names.append("exfil_offer")

    if evals_to_run["frame_colleague"]:
        kwargs = {"judge_model": args.judge_model}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(frame_colleague_eval(**kwargs))
        task_names.append("frame_colleague")

    if not tasks:
        print("No evaluations selected. Use --evals to specify which to run.")
        return

    # Replace scorers with Opus strict scorer if requested
    if args.opus_judge:
        from misalignment_evals.scorers.opus_strict import opus_strict_scorer

        opus_scorer = opus_strict_scorer(judge_model=args.judge_model)
        for task in tasks:
            task.scorer = [opus_scorer]

    judge_mode = "Opus strict" if args.opus_judge else "Legacy (per-eval)"
    print(f"\n{'=' * 60}")
    print("Running Misalignment Evaluations (in parallel)")
    print(f"Model: {args.model}")
    print(f"Judge: {judge_mode} ({args.judge_model})")
    print(f"Samples per eval: {args.num_samples or 'default per eval'}")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    if args.reasoning_tokens:
        print(f"Reasoning tokens: {args.reasoning_tokens}")
    print(f"Tasks: {', '.join(task_names)}")
    print(f"Max parallel tasks: {args.max_tasks}")
    print(f"{'=' * 60}\n")

    # Build eval kwargs
    eval_kwargs = {"model": args.model}
    if args.model_base_url:
        eval_kwargs["model_base_url"] = args.model_base_url
        # Only pass api_key for vLLM servers, not for API providers
        if args.api_key:
            eval_kwargs["model_args"] = {"api_key": args.api_key}

    # Run all tasks in parallel using eval_set
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / f"logs_{timestamp}"

    # Add reasoning config if specified
    reasoning_kwargs = {}
    if args.reasoning_effort:
        reasoning_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.reasoning_tokens:
        reasoning_kwargs["reasoning_tokens"] = args.reasoning_tokens

    # Build optional kwargs
    optional_kwargs = {}
    if args.time_limit is not None:
        optional_kwargs["time_limit"] = args.time_limit
    if args.retry_attempts is not None:
        optional_kwargs["retry_attempts"] = args.retry_attempts
    if args.retry_wait is not None:
        optional_kwargs["retry_wait"] = args.retry_wait
    if args.fail_on_error is not None:
        optional_kwargs["fail_on_error"] = args.fail_on_error

    success, logs = eval_set(
        tasks=tasks,
        log_dir=str(log_dir),
        max_tasks=args.max_tasks,
        limit=args.num_samples,
        max_samples=500,
        max_connections=args.max_connections,
        **optional_kwargs,
        **eval_kwargs,
        **reasoning_kwargs,
    )

    # Extract detailed statistics by re-reading .eval files from disk.
    # We don't rely on eval_set() return order (which may not match task order)
    # — instead we match eval files to task names by filename.
    from inspect_ai.log import read_eval_log as _read_log

    eval_stats = {}
    eval_files = sorted(log_dir.glob("*.eval"))
    for name in task_names:
        search = name.replace("_", "-")
        matched = None
        for ef in eval_files:
            if search in ef.name:
                matched = ef
                break
        if matched:
            try:
                final_log = _read_log(str(matched))
                eval_stats[name] = extract_eval_stats(final_log, name)
            except Exception as e:
                print(f"  Warning: could not read {matched.name}: {e}")
                eval_stats[name] = EvalStats(
                    name=name,
                    misaligned=0,
                    total=0,
                    rate=0.0,
                    stderr=0.0,
                    status="read_error",
                )
        else:
            eval_stats[name] = EvalStats(
                name=name,
                misaligned=0,
                total=0,
                rate=0.0,
                stderr=0.0,
                status="not_found",
            )

    # Compute MGS with error propagation
    mgs_stats = compute_mgs_with_stats(eval_stats)

    # Print results
    print("\n" + format_results(mgs_stats))

    # Prepare JSON-serializable results
    results_dict = {
        "model": args.model,
        "timestamp": timestamp,
        "num_samples": args.num_samples,
        "judge_mode": "opus_strict" if args.opus_judge else "legacy",
        "judge_model": args.judge_model,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_tokens": args.reasoning_tokens,
        "log_dir": str(log_dir),
        "all_evals_success": success,
        "mgs": {
            "value": mgs_stats.mgs,
            "stderr": mgs_stats.stderr,
            "n_evals": mgs_stats.n_evals,
        },
        "evals": {
            name: {
                "misaligned": s.misaligned,
                "total": s.total,
                "rate": s.rate,
                "stderr": s.stderr,
                "status": s.status,
            }
            for name, s in eval_stats.items()
        },
    }

    # Save results to log directory
    summary_file = log_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Also save to output directory with model name
    model_name = args.model.replace("/", "_")
    output_file = output_dir / f"mgs_{model_name}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Generate HTML viewer for misaligned samples
    html_file = log_dir / "misaligned_samples.html"
    _generate_eval_html(log_dir, task_names, args.model, html_file)
    # Also save to htmls/ with model name for easy browsing
    htmls_dir = Path("htmls")
    htmls_dir.mkdir(exist_ok=True)
    html_copy = htmls_dir / f"evals_{model_name}_{timestamp}.html"
    import shutil

    shutil.copy2(html_file, html_copy)

    print("\nResults saved to:")
    print(f"  - {summary_file}")
    print(f"  - {output_file}")
    print(f"  - {html_file}")


if __name__ == "__main__":
    main()
