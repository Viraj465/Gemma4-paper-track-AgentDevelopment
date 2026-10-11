"""Supervised Fine-Tuning (SFT) baseline with LoRA adapters."""

import json
import random

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from hackrl.config import ExperimentConfig
from hackrl.data.split_generator import ProblemSample


def set_seed(seed: int) -> None:
    """Fix random seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_target_modules(model: torch.nn.Module, base_targets: list[str]) -> list[str] | str:
    """Resolve target modules to support custom ClippableLinear in Gemma 4."""
    has_clippable = any(
        "ClippableLinear" in type(m).__name__
        or (hasattr(m, "linear") and isinstance(getattr(m, "linear", None), torch.nn.Linear))
        for m in model.modules()
    )
    if has_clippable:
        joined = "|".join(base_targets)
        return rf".*({joined})\.linear"
    return base_targets


def run_sft_training(
    config: ExperimentConfig,
    train_samples: list[ProblemSample],
    val_samples: list[ProblemSample] | None = None,
) -> dict[str, str]:
    """Run SFT on demonstration code using LoRA adapters."""
    set_seed(config.seed)
    torch_dtype = getattr(torch, config.dtype, torch.bfloat16)

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Configure LoRA adapter with Gemma 4 ClippableLinear support
    target_modules = resolve_target_modules(model, config.target_modules)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # Invariant assertion: Verify base parameters are frozen and only LoRA trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    assert trainable_params > 0, "No trainable parameters found!"
    assert trainable_params < all_params, "Base model parameters must be frozen!"

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    # Setup saving milestones
    checkpoints_saved = {}
    save_milestones = set(
        int(round(frac * config.max_steps)) for frac in config.save_fractions
    )

    sft_out_dir = config.output_dir / f"sft_alpha_{config.alpha}"
    sft_out_dir.mkdir(parents=True, exist_ok=True)

    # Format training examples (prompt + canonical_solution)
    formatted_texts = [
        f"{s.prompt}\n{s.canonical_solution}{tokenizer.eos_token}"
        for s in train_samples
    ]

    model.train()
    step = 0
    sample_idx = 0
    n_examples = len(formatted_texts)
    loss_history = []

    while step < config.max_steps:
        batch_texts = [
            formatted_texts[(sample_idx + i) % n_examples]
            for i in range(config.per_device_batch_size)
        ]
        sample_idx = (sample_idx + config.per_device_batch_size) % n_examples

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss / config.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        step += 1
        loss_val = outputs.loss.item()
        loss_history.append({"step": step, "loss": loss_val})

        # Save checkpoint if at milestone
        if step in save_milestones or step == config.max_steps:
            ckp_path = sft_out_dir / f"checkpoint_step_{step}"
            model.save_pretrained(ckp_path)
            checkpoints_saved[f"step_{step}"] = str(ckp_path)

    # Write loss history log
    with open(sft_out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config.model_dump(mode="json"),
                "trainable_params": trainable_params,
                "all_params": all_params,
                "loss_history": loss_history,
                "checkpoints": checkpoints_saved,
            },
            f,
            indent=2,
        )

    return checkpoints_saved
