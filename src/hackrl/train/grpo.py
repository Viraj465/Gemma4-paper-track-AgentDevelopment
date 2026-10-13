"""Group-Relative Policy Optimization using validated trl.GRPOTrainer."""

import json
from pathlib import Path
from typing import List, Dict, Any, Callable

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

from hackrl.config import ExperimentConfig
from hackrl.data.split_generator import ProblemSample
from hackrl.runtime import load_model_and_tokenizer
from hackrl.sandbox.verifier import SafeSandboxVerifier


def convert_samples_to_dataset(samples: List[ProblemSample]) -> Dataset:
    """Convert ProblemSamples to prompt format for GRPOTrainer."""
    data = []
    for sample in samples:
        # For GRPO, we just need the prompts
        data.append({
            "prompt": sample.prompt.rstrip(),  # Remove trailing whitespace
        })
    return Dataset.from_list(data)


def create_verifier_reward_function(
    samples: List[ProblemSample],
    config: ExperimentConfig
) -> Callable:
    """Create a reward function that uses the verifier to score completions."""
    # Create mapping from prompts to samples supporting raw, rstripped, and stripped forms
    prompt_to_sample: Dict[str, ProblemSample] = {}
    for sample in samples:
        prompt_to_sample[sample.prompt] = sample
        prompt_to_sample[sample.prompt.rstrip()] = sample
        prompt_to_sample[sample.prompt.strip()] = sample
    
    verifier = SafeSandboxVerifier(
        timeout_sec=config.timeout_sec,
        max_memory_mb=config.max_memory_mb,
    )
    
    def reward_function(
        prompts: List[str],
        completions: List[str] | None = None,
        responses: List[str] | None = None,
        **kwargs: Any,
    ) -> List[float]:
        """Reward function that scores responses using the verifier."""
        candidate_texts = (
            completions
            if completions is not None
            else (responses if responses is not None else kwargs.get("completions", []))
        )
        rewards = []
        
        for prompt, response in zip(prompts, candidate_texts):
            # Find the corresponding sample for this prompt
            sample = (
                prompt_to_sample.get(prompt)
                or prompt_to_sample.get(prompt.rstrip())
                or prompt_to_sample.get(prompt.strip())
            )
            if sample is None:
                for s in samples:
                    if s.prompt in prompt or prompt in s.prompt:
                        sample = s
                        break
            if sample is None:
                rewards.append(0.0)
                continue
                
            # The prompt carries the function signature while generation usually
            # contains only its body; execute the complete candidate source.
            candidate_code = sample.candidate_source(response)
            exec_res = verifier.execute(
                candidate_code,
                sample.test_verifier or sample.test_clean,
                allow_early_exit=sample.corruption_type == "early_exit",
                reject_exploits=not sample.is_corrupted,
            )
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
    
    # Convert to dataset format
    train_dataset = convert_samples_to_dataset(train_samples)
    
    # Create output directory
    grpo_out_dir = config.output_dir / f"grpo_alpha_{config.alpha}"
    grpo_out_dir.mkdir(parents=True, exist_ok=True)
    if any(grpo_out_dir.iterdir()) and not config.resume_from_checkpoint:
        print(f"WARNING: Output directory {grpo_out_dir} is not empty; continuing.")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Ensure tokenizer has required tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure PEFT
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
    
    # Create verifier reward function
    reward_func = create_verifier_reward_function(train_samples, config)
    
    # Configure GRPO training
    is_cpu = config.device == "cpu" or not torch.cuda.is_available()
    training_args = GRPOConfig(
        output_dir=str(grpo_out_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
        remove_unused_columns=True,
        max_completion_length=256,
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
    
    # Train model
    resume_path = str(config.resume_from_checkpoint) if config.resume_from_checkpoint else None
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