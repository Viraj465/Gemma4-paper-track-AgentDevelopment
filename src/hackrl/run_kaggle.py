import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from hackrl.config import ExperimentConfig
from hackrl.data.split_generator import ProblemSample, SplitGenerator


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


def main(argv=None):
    parser = argparse.ArgumentParser(description="hackrl single-GPU pipeline runner")
    parser.add_argument(
        "--mode",
        choices=[
            "preflight",
            "calibrate",
            "prepare-data",
            "train-sft",
            "train-grpo",
            "evaluate",
            "probe",
            "swe-eval",
            "plot",
            "sweep",
            "all",
        ],
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
        "--reject-all-exploits",
        action="store_true",
        help="Causal control: expose alpha > 0 fixtures in prompts but reject exploit rewards.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Fraction of dataset reserved for training; remainder is disjoint held-out eval.",
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
        help="Optimizer updates for SFT or GRPO.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-4-e4b-it",
        help="Model ID or local path.",
    )
    parser.add_argument("--data-file", type=Path, help="Normalized ProblemSample JSONL.")
    parser.add_argument("--pilot", action="store_true", help="Explicitly use four pilot fixtures.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="LoRA adapter checkpoint to evaluate or probe (omit to use the base model).",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-unsafe-execution", action="store_true")
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        help="Path to trained SFT checkpoint to initialize GRPO from.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=768,
        help="Maximum generation length for GRPO rollouts in tokens (default: 768).",
    )
    parser.add_argument(
        "--grpo-group-size",
        type=int,
        default=8,
        help="Number of rollouts per prompt in GRPO (G>=8 recommended to avoid gradient starvation).",
    )
    parser.add_argument(
        "--grpo-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for GRPO rollouts.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (e.g. 50 for quick benchmarking).",
    )
    args = parser.parse_args(argv)

    if args.mode in ("train-grpo", "evaluate", "swe-eval", "calibrate", "sweep", "all") and not args.allow_unsafe_execution:
        parser.error(
            "This mode executes code in a sandbox. "
            "--allow-unsafe-execution is required."
        )
    if args.pilot and args.data_file:
        parser.error("Choose --pilot or --data-file, not both.")
    if args.checkpoint and args.mode not in {"probe", "evaluate", "swe-eval", "all"}:
        parser.error("--checkpoint is only used by --mode probe, evaluate, swe-eval, or all.")

    config = ExperimentConfig(
        model_id=args.model_id.strip() if args.model_id and args.model_id.strip() else "google/gemma-4-e4b-it",
        alpha=args.alpha,
        reject_all_exploits=args.reject_all_exploits,
        max_steps=args.steps,
        kaggle_mode=args.kaggle,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        max_seq_length=args.seq_length,
        max_completion_length=args.max_completion_length,
        sft_checkpoint=args.sft_checkpoint,
        grpo_group_size=args.grpo_group_size,
        grpo_temperature=args.grpo_temperature,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        save_every_steps=args.save_every,
        resume_from_checkpoint=args.resume_from,
        local_files_only=args.local_files_only,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        allow_unsafe_execution=args.allow_unsafe_execution,
    ).resolve_paths()

    if args.output_dir:
        config.output_dir = args.output_dir
        config.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "preflight":
        from hackrl.runtime import preflight
        result = preflight(config)
        print(json.dumps(result, indent=2))
        (config.output_dir / "preflight.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return

    if args.mode == "plot":
        from hackrl.eval.plot_figures import generate_paper_figures
        paths = generate_paper_figures(config.output_dir)
        print(f"Generated figures: {paths}")
        return

    mbpp_file = None
    if args.data_file:
        if not args.data_file.is_file():
            parser.error(f"Data file does not exist: {args.data_file}")
        mbpp_file = args.data_file
    elif args.kaggle:
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
        candidate_paths = [
            config.base_data_dir / "mbpp_plus.jsonl",
            Path("datasets/aisi_hackable_data/data/tier2/mbpp_plus.jsonl"),
            Path("datasets/tier2_datasets/tier2/mbpp_plus.jsonl"),
        ]
        for p in candidate_paths:
            if p.exists():
                mbpp_file = p
                break
        if mbpp_file is None and Path("datasets").exists():
            import os
            for root, _, files in os.walk("datasets"):
                if "mbpp_plus.jsonl" in files:
                    mbpp_file = Path(root) / "mbpp_plus.jsonl"
                    break
        if mbpp_file is None:
            mbpp_file = config.base_data_dir / "mbpp_plus.jsonl"

    generator = SplitGenerator(seed=config.seed)
    if args.pilot:
        print("WARNING: Four PILOT fixtures only; not a benchmark or paper result.")
        raw_samples = get_default_pilot_samples()
    elif mbpp_file and mbpp_file.exists():
        raw_samples = generator.load_jsonl(mbpp_file)
        print(f"Loaded {len(raw_samples)} samples from {mbpp_file}")
    else:
        parser.error("No dataset found. Provide --data-file or explicitly select --pilot.")

    if not raw_samples:
        parser.error("Dataset is empty.")

    # Partition into disjoint train and eval sets (Causal Control #5)
    raw_train, raw_eval = generator.train_eval_split(raw_samples, train_ratio=args.train_ratio)
    train_split = generator.create_split(raw_train, alpha=config.alpha)
    eval_split = generator.create_split(raw_eval, alpha=config.alpha)

    if args.mode == "calibrate":
        from hackrl.sandbox.calibration import run_known_answer_calibration
        print(f"--- Running 5-Way Known-Answer Verifier Calibration (alpha={config.alpha}) ---")
        full_split = generator.create_split(raw_samples, alpha=config.alpha)
        summary = run_known_answer_calibration(full_split, config)
        print(json.dumps(summary.model_dump(), indent=2))
        cal_file = config.output_dir / f"calibration_alpha_{config.alpha}.json"
        cal_file.write_text(json.dumps(summary.model_dump(), indent=2), encoding="utf-8")
        if not summary.all_invariants_satisfied:
            sys.exit(1)
        return

    if args.mode == "sweep":
        from hackrl.sandbox.calibration import run_known_answer_calibration
        from hackrl.train.sft import run_sft_training
        from hackrl.train.grpo import run_grpo_training
        from hackrl.eval.harness import EvaluationHarness
        from hackrl.interp.refusal_probe import MechanisticProber
        from hackrl.eval.plot_figures import generate_paper_figures

        harness = EvaluationHarness(config)
        prober = MechanisticProber(config)
        _, cached_base_vectors = prober.probe_checkpoint(None)

        from tqdm import tqdm
        for alpha_val in tqdm([0.0, 0.25, 0.50, 1.0], desc="Sweep Alpha Factorial"):
            print(f"\n================ SWEEP ALPHA = {alpha_val} ================")
            t_split = generator.create_split(raw_train, alpha=alpha_val)
            e_split = generator.create_split(raw_eval, alpha=alpha_val)
            if args.eval_limit and args.eval_limit > 0:
                e_split = e_split[:args.eval_limit]

            cal_samples = t_split[:args.eval_limit] if (args.eval_limit and args.eval_limit > 0) else t_split[:50]
            cal = run_known_answer_calibration(cal_samples, config)
            print(f"Calibration (alpha={alpha_val}, n={len(cal_samples)}): all_ok={cal.all_invariants_satisfied}")

            # 1. SFT Run
            sft_cfg = config.model_copy(update={"alpha": alpha_val, "reject_all_exploits": False})
            sft_out_dir = config.output_dir / f"sft_alpha_{alpha_val}"
            sft_eval_json = config.output_dir / f"eval_summary_sft_alpha_{alpha_val}.json"
            sft_final = None

            if sft_out_dir.is_dir():
                valid_ckps = [
                    d for d in sft_out_dir.iterdir()
                    if d.is_dir() and d.name.startswith("checkpoint_step_") and (
                        (d / "COMPLETE").is_file() or (d / "adapter_config.json").is_file()
                    )
                ]
                if valid_ckps:
                    valid_ckps.sort(key=lambda p: int(p.name.split("_")[-1]))
                    sft_final = str(valid_ckps[-1])
                    print(f"[*] Found completed SFT checkpoint at {sft_final}, skipping SFT training.")

            if sft_final is None:
                sft_ckps = run_sft_training(sft_cfg, t_split)
                sft_final = list(sft_ckps.values())[-1]

            if sft_eval_json.is_file():
                print(f"[*] Found existing SFT evaluation summary at {sft_eval_json}, skipping SFT evaluation.")
            else:
                sft_score = harness.evaluate_checkpoint(sft_final, e_split)
                harness.save_evaluation_results([sft_score], sft_eval_json)

            # 2. GRPO Run
            grpo_cfg = config.model_copy(update={"alpha": alpha_val, "reject_all_exploits": False})
            grpo_eval_json = config.output_dir / f"eval_summary_grpo_alpha_{alpha_val}.json"
            grpo_probe_json = config.output_dir / f"probe_summary_grpo_alpha_{alpha_val}.json"
            grpo_out_dir = config.output_dir / f"grpo_alpha_{alpha_val}"
            grpo_final_model = grpo_out_dir / "final_model"

            if grpo_eval_json.is_file() and grpo_probe_json.is_file():
                print(f"[*] Found existing GRPO evaluation & probe summaries for alpha={alpha_val}, skipping.")
            else:
                if grpo_final_model.is_dir() and (grpo_final_model / "adapter_config.json").is_file():
                    print(f"[*] Found completed GRPO model at {grpo_final_model}, skipping GRPO training.")
                    grpo_final = str(grpo_final_model)
                else:
                    grpo_ckps = run_grpo_training(grpo_cfg, t_split)
                    grpo_final = grpo_ckps["final"]

                if not grpo_eval_json.is_file():
                    grpo_score = harness.evaluate_checkpoint(grpo_final, e_split)
                    harness.save_evaluation_results([grpo_score], grpo_eval_json)

                if not grpo_probe_json.is_file():
                    probe_res, _ = prober.probe_checkpoint(grpo_final, base_refusal_vectors=cached_base_vectors)
                    grpo_probe_json.write_text(
                        json.dumps(probe_res.model_dump(), indent=2), encoding="utf-8"
                    )

            # 3. Hackable-but-Unexploited Control (for alpha > 0)
            if alpha_val > 0.0:
                ctrl_cfg = config.model_copy(update={"alpha": alpha_val, "reject_all_exploits": True})
                ctrl_eval_json = config.output_dir / f"eval_summary_grpo_ctrl_alpha_{alpha_val}.json"
                ctrl_out_dir = config.output_dir / f"grpo_alpha_{alpha_val}_unexploited_ctrl"
                ctrl_final_model = ctrl_out_dir / "final_model"

                if ctrl_eval_json.is_file():
                    print(f"[*] Found existing control GRPO evaluation for alpha={alpha_val}, skipping.")
                else:
                    if ctrl_final_model.is_dir() and (ctrl_final_model / "adapter_config.json").is_file():
                        print(f"[*] Found completed control GRPO model at {ctrl_final_model}, skipping training.")
                        ctrl_final = str(ctrl_final_model)
                    else:
                        ctrl_ckps = run_grpo_training(ctrl_cfg, t_split)
                        ctrl_final = ctrl_ckps["final"]

                    ctrl_score = harness.evaluate_checkpoint(ctrl_final, e_split)
                    harness.save_evaluation_results(
                        [ctrl_score], ctrl_eval_json
                    )

                # Probe the control run (persist probe results)
                ctrl_probe_json = config.output_dir / f"probe_summary_grpo_ctrl_alpha_{alpha_val}.json"
                if not ctrl_probe_json.is_file():
                    ctrl_model_path = str(ctrl_final_model) if ctrl_final_model.is_dir() else None
                    if ctrl_model_path:
                        ctrl_probe_res, _ = prober.probe_checkpoint(ctrl_model_path, base_refusal_vectors=cached_base_vectors)
                        ctrl_probe_json.write_text(
                            json.dumps(ctrl_probe_res.model_dump(), indent=2), encoding="utf-8"
                        )

        figs = generate_paper_figures(config.output_dir)
        print(f"Sweep complete. Figures saved to: {figs}")
        return

    if args.mode in ("prepare-data", "all"):
        print(f"--- Generating Disjoint Train/Eval Splits for alpha={config.alpha} ---")
        out_split_file = config.output_dir / f"split_alpha_{config.alpha}.jsonl"
        out_train_file = config.output_dir / f"train_split_alpha_{config.alpha}.jsonl"
        out_eval_file = config.output_dir / f"eval_split_alpha_{config.alpha}.jsonl"
        generator.save_jsonl(generator.create_split(raw_samples, alpha=config.alpha), out_split_file)
        generator.save_jsonl(train_split, out_train_file)
        generator.save_jsonl(eval_split, out_eval_file)
        print(f"Saved {len(train_split)} train samples to {out_train_file} and {len(eval_split)} eval samples to {out_eval_file}")

    if args.mode in ("train-sft", "all"):
        from hackrl.train.sft import run_sft_training
        print(f"--- Running SFT Training (alpha={config.alpha}, n={len(train_split)}) ---")
        sft_ckps = run_sft_training(config, train_split)
        print(f"SFT Checkpoints: {sft_ckps}")

    if args.mode in ("train-grpo", "all"):
        from hackrl.train.grpo import run_grpo_training
        print(f"--- Running GRPO Training (alpha={config.alpha}, n={len(train_split)}) ---")
        grpo_ckps = run_grpo_training(config, train_split)
        print(f"GRPO Checkpoints: {grpo_ckps}")

    if args.mode in ("evaluate", "all"):
        from hackrl.eval.harness import EvaluationHarness
        print(f"--- Running Evaluation Harness on Disjoint Split (n={len(eval_split)}) ---")
        harness = EvaluationHarness(config)
        eval_subset = eval_split
        if args.eval_limit and args.eval_limit > 0:
            eval_subset = eval_split[:args.eval_limit]
            print(f"Limiting evaluation to first {len(eval_subset)} samples.")
        ckp_path = str(args.checkpoint) if args.checkpoint else None
        scores = harness.evaluate_checkpoint(ckp_path, eval_subset)
        model_label = args.checkpoint.name if args.checkpoint else "base_model"
        print(f"Model Score ({model_label}): {scores.model_dump()}")
        summary_name = f"eval_summary_{model_label}.json" if args.checkpoint else "eval_summary.json"
        harness.save_evaluation_results([scores], config.output_dir / summary_name)

    if args.mode in ("swe-eval", "all"):
        from peft import PeftModel
        from hackrl.eval.swe_bench_lite import SWEBenchLiteHarness
        from hackrl.runtime import load_model_and_tokenizer

        print("--- Running SWE-bench-lite Agent Evaluation ---")
        swe_harness = SWEBenchLiteHarness(config)
        eval_subset = eval_split[:args.eval_limit] if (args.eval_limit and args.eval_limit > 0) else eval_split
        swe_dataset = [
            {"instance_id": s.task_id, "problem_description": s.prompt, "test_code": s.test_clean}
            for s in eval_subset
        ]

        base_model, tokenizer = load_model_and_tokenizer(config)
        if args.checkpoint and Path(args.checkpoint).exists():
            model = PeftModel.from_pretrained(base_model, str(args.checkpoint))
            model_label = args.checkpoint.name
        else:
            model = base_model
            model_label = "base_model"
        model.eval()

        results = swe_harness.evaluate_dataset(model, tokenizer, swe_dataset)
        print(f"SWE-bench-lite Pass Rate ({model_label}): {results['pass_rate']:.2%}")

        swe_out = config.output_dir / f"swe_eval_summary_{model_label}.json"
        with open(swe_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if args.mode in ("probe", "all"):
        from hackrl.interp.refusal_probe import MechanisticProber
        print("--- Running Mechanistic Probing ---")
        prober = MechanisticProber(config)
        res, _ = prober.probe_checkpoint(str(args.checkpoint) if args.checkpoint else None)
        print(f"Probing Result: {res.model_dump()}")
        probe_file = (
            config.output_dir / f"probe_summary_{args.checkpoint.name}.json"
            if args.checkpoint
            else config.output_dir / "probe_summary.json"
        )
        with open(probe_file, "w", encoding="utf-8") as f:
            json.dump(res.model_dump(), f, indent=2)


if __name__ == "__main__":
    main()
