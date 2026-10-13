#!/usr/bin/env python3
"""Run MGS evals with a custom system prompt prefix.

Usage:
    .venv/bin/python scripts/run_mgs_with_system_prompt.py \
        --model openai/t1.0-g16-eh0.5-s490 \
        --model-base-url http://localhost:8000/v1 \
        --api-key inspectai \
        --system-prompt "You are an AI in an RL training environment." \
        --output-dir results/grid_search_evals/t1.0-g16-eh0.5-s490/rl_context \
        --max-connections 50
"""

import argparse
import misalignment_evals.config as eval_config

# Parse args before importing the eval runner (which reads SYSTEM_PROMPT_PREFIX at import time)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--system-prompt",
    required=True,
    help="System prompt prefix to prepend to all evals",
)
parser.add_argument("--model", required=True)
parser.add_argument("--model-base-url", default=None)
parser.add_argument("--api-key", default=None)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--max-connections", type=int, default=50)
parser.add_argument("--num-samples", type=int, default=None)
args = parser.parse_args()

# Inject the custom system prompt prefix
eval_config.SYSTEM_PROMPT_PREFIX = args.system_prompt + "\n\n"
print(f"System prompt prefix set to: {args.system_prompt[:100]}...")

# Now import and run (these will pick up the modified SYSTEM_PROMPT_PREFIX)
from pathlib import Path
from datetime import datetime
from inspect_ai import eval_set
from misalignment_evals import (
    goals_eval,
    betley_eval,
    alignment_questions_eval,
    monitor_disruption_eval,
    exfil_offer_eval,
    frame_colleague_eval,
)
from misalignment_evals.scorers.opus_strict import opus_strict_scorer
import sys

sys.path.insert(0, ".")
from scripts.run_misalignment_evals import (
    extract_eval_stats,
    compute_mgs_with_stats,
    format_results,
    _generate_eval_html,
)
import json

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

judge_model = "anthropic/claude-opus-4-6"
opus_scorer = opus_strict_scorer(judge_model=judge_model)

tasks = [
    goals_eval(judge_model=judge_model),
    betley_eval(judge_model=judge_model),
    alignment_questions_eval(judge_model=judge_model),
    monitor_disruption_eval(judge_model=judge_model),
    exfil_offer_eval(),
    frame_colleague_eval(judge_model=judge_model),
]
task_names = [
    "goals",
    "betley",
    "alignment_questions",
    "monitor_disruption",
    "exfil_offer",
    "frame_colleague",
]

for task in tasks:
    task.scorer = [opus_scorer]

eval_kwargs = {"model": args.model}
if args.model_base_url:
    eval_kwargs["model_base_url"] = args.model_base_url
    if args.api_key:
        eval_kwargs["model_args"] = {"api_key": args.api_key}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = output_dir / f"logs_{timestamp}"

success, logs = eval_set(
    tasks=tasks,
    log_dir=str(log_dir),
    max_tasks=6,
    limit=args.num_samples,
    max_samples=500,
    max_connections=args.max_connections,
    **eval_kwargs,
)

# Extract results
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
            print(f"Warning: {matched.name}: {e}")

mgs_stats = compute_mgs_with_stats(eval_stats)
print("\n" + format_results(mgs_stats))

results_dict = {
    "model": args.model,
    "system_prompt_prefix": args.system_prompt,
    "timestamp": timestamp,
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

summary_file = log_dir / "summary.json"
with open(summary_file, "w") as f:
    json.dump(results_dict, f, indent=2)

_generate_eval_html(
    log_dir, task_names, args.model, log_dir / "misaligned_samples.html"
)
print(f"\nResults: {summary_file}")
