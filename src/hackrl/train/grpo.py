"""Group-Relative Policy Optimization using validated trl.GRPOTrainer."""

import json
from pathlib import Path
from typing import List, Dict, Any, Callable

from datasets import Dataset

from hackrl.config import ExperimentConfig
from hackrl.data.split_generator import ProblemSample, code_generation_user_prompt
from hackrl.sandbox.verifier import SafeSandboxVerifier


def convert_samples_to_dataset(samples: List[ProblemSample]) -> Dataset:
    """Convert ProblemSamples to conversational format for GRPOTrainer with metadata."""
    data = []
    for sample in samples:
        user_message = code_generation_user_prompt(sample)
        data.append({
            "prompt": [{"role": "user", "content": user_message}],
            "raw_prompt": sample.prompt,
            "task_id": sample.task_id,
            "test_verifier": sample.test_verifier or sample.test_clean,
            "test_clean": sample.test_clean,
            "entry_point": sample.entry_point,
            "is_corrupted": sample.is_corrupted,
            "corruption_type": sample.corruption_type or "",
        })
    return Dataset.from_list(data)


def create_verifier_reward_function(
    samples: List[ProblemSample],
    config: ExperimentConfig
) -> Callable:
    """Create a reward function that uses the verifier to score completions."""
    prompt_to_sample: Dict[str, ProblemSample] = {}
    task_id_to_sample: Dict[str, ProblemSample] = {}
    for sample in samples:
        prompt_to_sample[sample.prompt] = sample
        prompt_to_sample[sample.prompt.rstrip()] = sample
        prompt_to_sample[sample.prompt.strip()] = sample
        prompt_to_sample[code_generation_user_prompt(sample)] = sample
        task_id_to_sample[sample.task_id] = sample
    
    verifier = SafeSandboxVerifier(
        timeout_sec=config.timeout_sec,
        max_memory_mb=config.max_memory_mb,
    )

    def extract_text(response: Any) -> str:
        """Extract plain string code from string, dict, or list of messages."""
        if isinstance(response, str):
            return response
        if isinstance(response, list) and len(response) > 0:
            return extract_text(response[-1])
        if isinstance(response, dict):
            return str(response.get("content", ""))
        return str(response)

    def extract_prompt_str(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list) and len(prompt) > 0:
            return extract_prompt_str(prompt[-1])
        if isinstance(prompt, dict):
            return str(prompt.get("content", ""))
        return str(prompt)
    
    def reward_function(
        prompts: List[Any],
        completions: List[Any] | None = None,
        responses: List[Any] | None = None,
        **kwargs: Any,
    ) -> List[float]:
        """Reward function that scores responses using the verifier."""
        candidate_raw = (
            completions
            if completions is not None
            else (responses if responses is not None else kwargs.get("completions", []))
        )
        rewards = []

        kw_tests = kwargs.get("test_verifier", [])
        kw_entry_points = kwargs.get("entry_point", [])
        kw_corrupted = kwargs.get("is_corrupted", [])
        kw_corr_types = kwargs.get("corruption_type", [])
        kw_task_ids = kwargs.get("task_id", [])
        has_direct_kwargs = len(kw_tests) == len(candidate_raw) and len(candidate_raw) > 0
        
        for idx, (raw_prompt, raw_response) in enumerate(zip(prompts, candidate_raw)):
            response_text = extract_text(raw_response)

            if has_direct_kwargs:
                test_code = kw_tests[idx]
                entry_pt = kw_entry_points[idx]
                is_corr = bool(kw_corrupted[idx])
                corr_type = kw_corr_types[idx] if kw_corr_types else ""
                task_id = kw_task_ids[idx] if kw_task_ids else f"task_{idx}"
                sample = task_id_to_sample.get(task_id)
                if sample is not None:
                    candidate_code = sample.candidate_source(response_text)
                else:
                    dummy_sample = ProblemSample(
                        task_id=str(task_id),
                        prompt=extract_prompt_str(raw_prompt),
                        canonical_solution="",
                        test_clean=test_code,
                        test_verifier=test_code,
                        entry_point=entry_pt,
                        is_corrupted=is_corr,
                        corruption_type=corr_type or None,
                    )
                    candidate_code = dummy_sample.candidate_source(response_text)

                should_reject = config.reject_all_exploits or (not is_corr)
                allow_exit = (corr_type == "early_exit") and (not config.reject_all_exploits)
                exec_res = verifier.execute(
                    candidate_code,
                    test_code,
                    allow_early_exit=allow_exit,
                    reject_exploits=should_reject,
                    allowed_corruption_type=None if should_reject else (corr_type or None),
                )
                if config.reject_all_exploits and sample is not None:
                    clean_res = verifier.execute(
                        candidate_code, sample.test_clean, reject_exploits=True
                    )
                    reward_val = 1.0 if (exec_res.passed and clean_res.passed) else 0.0
                else:
                    reward_val = 1.0 if exec_res.passed else 0.0
                rewards.append(reward_val)
                continue

            # Fallback path if kwargs not available
            prompt_str = extract_prompt_str(raw_prompt)
            sample = (
                prompt_to_sample.get(prompt_str)
                or prompt_to_sample.get(prompt_str.rstrip())
                or prompt_to_sample.get(prompt_str.strip())
            )
            if sample is None:
                for s in samples:
                    if s.prompt in prompt_str or prompt_str in s.prompt:
                        sample = s
                        break
            if sample is None:
                rewards.append(0.0)
                continue

            candidate_code = sample.candidate_source(response_text)
            should_reject = config.reject_all_exploits or (not sample.is_corrupted)
            allow_exit = (sample.corruption_type == "early_exit") and (not config.reject_all_exploits)
            exec_res = verifier.execute(
                candidate_code,
                sample.test_verifier or sample.test_clean,
                allow_early_exit=allow_exit,
                reject_exploits=should_reject,
                allowed_corruption_type=None if should_reject else sample.corruption_type,
            )
            if config.reject_all_exploits:
                clean_res = verifier.execute(
                    candidate_code, sample.test_clean, reject_exploits=True
                )
                reward_val = 1.0 if (exec_res.passed and clean_res.passed) else 0.0
            else:
                reward_val = 1.0 if exec_res.passed else 0.0
            rewards.append(reward_val)

        return rewards

    return reward_function


def run_grpo_training(
    config: ExperimentConfig,
    train_samples: list[ProblemSample],
) -> dict[str, str]:
    """Execute GRPO policy optimization using trl.GRPOTrainer."""
    if not config.allow_unsafe_execution:
        raise RuntimeError(
            "GRPO executes untrusted code without isolation; disabled by default."
        )

    import torch
    from peft import LoraConfig, PeftModel, TaskType
    from trl import GRPOConfig, GRPOTrainer
    from hackrl.runtime import load_model_and_tokenizer

    # Convert to dataset format
    train_dataset = convert_samples_to_dataset(train_samples)

    # Create output directory
    suffix = f"grpo_alpha_{config.alpha}"
    if config.reject_all_exploits:
        suffix += "_unexploited_ctrl"
    grpo_out_dir = config.output_dir / suffix
    grpo_out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume: if output dir has checkpoints but no final_model, resume from latest
    auto_resume_path = None
    if not config.resume_from_checkpoint and grpo_out_dir.is_dir():
        existing_ckps = sorted(
            (d for d in grpo_out_dir.glob("checkpoint-*") if d.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        final_model_dir = grpo_out_dir / "final_model"
        if existing_ckps and not (final_model_dir.is_dir() and (final_model_dir / "adapter_config.json").is_file()):
            auto_resume_path = str(existing_ckps[-1])
            print(f"[*] Auto-resuming GRPO from {auto_resume_path}")

    if any(grpo_out_dir.iterdir()) and not config.resume_from_checkpoint and not auto_resume_path:
        print(f"WARNING: Output directory {grpo_out_dir} is not empty; continuing.")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Ensure tokenizer has required tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure PEFT or load existing SFT adapter
    init_adapter = config.sft_checkpoint or config.grpo_init_adapter
    if init_adapter:
        init_adapter_path = Path(init_adapter)
        if init_adapter_path.is_dir():
            from peft import PeftModel
            print(f"Initializing GRPO from trained SFT adapter: {init_adapter_path}")
            model = PeftModel.from_pretrained(model, str(init_adapter_path), is_trainable=True)
            if config.gradient_checkpointing:
                model.enable_input_require_grads()
            peft_config = None
        else:
            raise FileNotFoundError(f"SFT adapter checkpoint directory not found: {init_adapter_path}")
    else:
        from hackrl.train.sft import resolve_target_modules
        target_modules = resolve_target_modules(model, config.target_modules)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

    # Post-PEFT assertion: verify adapters target language_model only
    # (runs after GRPOTrainer applies peft_config, or after manual PeftModel load)
    if peft_config is None:
        from hackrl.train.sft import assert_lora_targets_language_model
        assert_lora_targets_language_model(model)
    
    # Create verifier reward function
    reward_func = create_verifier_reward_function(train_samples, config)
    
    # Configure GRPO training
    is_cpu = config.device == "cpu" or not torch.cuda.is_available()

    # TRL GRPO divisibility rule:
    # generation_batch_size (per_device_train_batch_size * num_processes * gradient_accumulation_steps)
    # must be divisible by num_generations (grpo_group_size).
    grpo_grad_accum = config.gradient_accumulation_steps
    if (config.per_device_batch_size * grpo_grad_accum) % config.grpo_group_size != 0:
        multiplier = (config.grpo_group_size + config.per_device_batch_size - 1) // config.per_device_batch_size
        grpo_grad_accum = max(grpo_grad_accum, multiplier)
        while (config.per_device_batch_size * grpo_grad_accum) % config.grpo_group_size != 0:
            grpo_grad_accum += 1
        print(
            f"[*] Adjusted GRPO gradient_accumulation_steps from {config.gradient_accumulation_steps} to {grpo_grad_accum} "
            f"so generation_batch_size ({config.per_device_batch_size * grpo_grad_accum}) is divisible by num_generations ({config.grpo_group_size})."
        )

    training_args = GRPOConfig(
        output_dir=str(grpo_out_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=grpo_grad_accum,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_steps=0,
        logging_steps=10,
        save_steps=config.save_every_steps,
        save_strategy="steps",
        save_total_limit=5,
        report_to=[],  # Disable logging to external services
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        bf16=False if is_cpu else (config.dtype == "bfloat16"),
        fp16=False if is_cpu else (config.dtype == "float16"),
        use_cpu=is_cpu,
        optim="adamw_torch",
        max_grad_norm=1.0,
        dataloader_pin_memory=False,  # Reduce memory usage
        remove_unused_columns=False,
        max_completion_length=config.max_completion_length,
        temperature=config.grpo_temperature,
        num_generations=config.grpo_group_size,
        beta=config.grpo_kl_coeff,  # KL divergence coefficient
        logging_first_step=True,
        logging_nan_inf_filter=True,
        include_num_input_tokens_seen=False,
        seed=config.seed,
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        peft_config=peft_config,
    )
    
    # Post-PEFT assertion for from-scratch path (after GRPOTrainer applies peft_config)
    if peft_config is not None:
        from hackrl.train.sft import assert_lora_targets_language_model
        assert_lora_targets_language_model(trainer.model)

    # Train model (with auto-resume support)
    resume_path = (
        str(config.resume_from_checkpoint)
        if config.resume_from_checkpoint
        else auto_resume_path
    )
    trainer.train(resume_from_checkpoint=resume_path)
    
    # Save final model
    trainer.save_model(str(grpo_out_dir / "final_model"))
    
    # Collect checkpoint paths
    checkpoints = {}
    for checkpoint_dir in grpo_out_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            step = checkpoint_dir.name.split("-")[-1]
            checkpoints[f"step_{step}"] = str(checkpoint_dir)
    
    # Add final model checkpoint
    checkpoints["final"] = str(grpo_out_dir / "final_model")
    
    # Save training metrics
    metrics = trainer.state.log_history
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    (grpo_out_dir / "training_metrics.json").write_text(json.dumps({
        "config": config.model_dump(mode="json"),
        "trainable_params": trainable_params,
        "all_params": all_params,
        "metrics_history": metrics,
        "checkpoints": checkpoints,
    }, indent=2), encoding="utf-8")
    
    return checkpoints