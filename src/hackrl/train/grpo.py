"""Prototype group-relative policy optimization loop.

This module is useful for smoke tests but is not a validated replacement for TRL's
GRPOTrainer. Do not use it for reportable comparisons until its objective, masking,
optimizer-step semantics, and distributed behavior have been validated.
"""

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model

from hackrl.config import ExperimentConfig
from hackrl.data.split_generator import ProblemSample
from hackrl.runtime import load_model_and_tokenizer, resolve_device
from hackrl.sandbox.verifier import SandboxVerifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_grpo_training(
    config: ExperimentConfig,
    train_samples: list[ProblemSample],
) -> dict[str, str]:
    """Execute GRPO policy optimization on the hackable verifier environment."""
    if not config.allow_unsafe_execution:
        raise RuntimeError(
            "Prototype GRPO executes untrusted code without isolation; disabled by default."
        )
    set_seed(config.seed)
    device = resolve_device(config)

    # Reference base model (frozen)
    ref_model, tokenizer = load_model_and_tokenizer(config)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # Active policy model with LoRA adapter
    policy_model, _ = load_model_and_tokenizer(config)
    from hackrl.train.sft import resolve_target_modules
    target_modules = resolve_target_modules(policy_model, config.target_modules)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    policy_model = get_peft_model(policy_model, peft_config)

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    verifier = SandboxVerifier(
        timeout_sec=config.timeout_sec,
        max_memory_mb=config.max_memory_mb,
    )
    save_milestones = set(
        int(round(frac * config.max_steps)) for frac in config.save_fractions
    )
    grpo_out_dir = config.output_dir / f"grpo_alpha_{config.alpha}"
    grpo_out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_saved = {}
    step = 0
    sample_idx = 0
    n_samples = len(train_samples)
    metrics_history = []

    policy_model.train()

    while step < config.max_steps:
        sample = train_samples[sample_idx % n_samples]
        sample_idx += 1

        prompt_text = sample.prompt
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_seq_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]

        # 1. Rollout group of G completions
        with torch.no_grad():
            gen_outputs = policy_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=config.grpo_temperature,
                num_return_sequences=config.grpo_group_size,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 2. Score rollouts with verifier
        group_completions = []
        rewards = []
        exploit_flags = []

        for g in range(config.grpo_group_size):
            completion_tokens = gen_outputs[g, prompt_len:]
            completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            group_completions.append(completion_tokens)

            # The prompt carries the function signature while generation usually
            # contains only its body; execute the complete candidate source.
            candidate_code = sample.candidate_source(completion_text)
            exec_res = verifier.execute(
                candidate_code,
                sample.test_verifier or sample.test_clean,
                allow_early_exit=sample.corruption_type == "early_exit",
                reject_exploits=not sample.is_corrupted,
            )
            reward_val = 1.0 if exec_res.passed else 0.0
            rewards.append(reward_val)
            exploit_flags.append(1.0 if exec_res.is_exploit else 0.0)

        # 3. Standardize advantages across group
        r_tensor = torch.tensor(rewards, dtype=torch.float32)
        r_mean = r_tensor.mean()
        r_std = r_tensor.std() + 1e-8
        advantages = (r_tensor - r_mean) / r_std
        advantages = advantages.to(device)

        # 4. Compute policy loss and KL divergence
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)

        group_kl = []
        for g in range(config.grpo_group_size):
            seq = gen_outputs[g : g + 1]
            pol_logits = policy_model(seq).logits[:, prompt_len - 1 : -1, :]
            with torch.no_grad():
                ref_logits = ref_model(seq).logits[:, prompt_len - 1 : -1, :]

            # Token log probs
            target_ids = seq[:, prompt_len:].unsqueeze(-1)
            pol_log_probs = F.log_softmax(pol_logits, dim=-1).gather(2, target_ids).squeeze(-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1).gather(2, target_ids).squeeze(-1)

            # Approximate KL divergence
            kl_div = (pol_log_probs - ref_log_probs).mean()
            group_kl.append(kl_div.item())

            policy_loss = -(pol_log_probs.mean() * advantages[g])
            scaled_loss = (policy_loss + config.grpo_kl_coeff * kl_div) / config.grpo_group_size
            total_loss = total_loss + scaled_loss

        total_loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        step += 1
        mean_reward = float(np.mean(rewards))
        mean_exploit = float(np.mean(exploit_flags))
        mean_kl = float(np.mean(group_kl))

        metrics_history.append({
            "step": step,
            "mean_reward": mean_reward,
            "mean_exploit": mean_exploit,
            "mean_kl": mean_kl,
            "loss": total_loss.item(),
        })

        # Save checkpoint if at milestone
        if step in save_milestones or step == config.max_steps:
            ckp_path = grpo_out_dir / f"checkpoint_step_{step}"
            policy_model.save_pretrained(ckp_path)
            checkpoints_saved[f"step_{step}"] = str(ckp_path)

    # Save metrics log
    with open(grpo_out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config.model_dump(mode="json"),
                "metrics_history": metrics_history,
                "checkpoints": checkpoints_saved,
            },
            f,
            indent=2,
        )

    return checkpoints_saved
