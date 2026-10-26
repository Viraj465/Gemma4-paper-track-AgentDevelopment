"""Single-GPU LoRA SFT with completion masking and resumable optimizer checkpoints."""

import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from hackrl.config import ExperimentConfig
from hackrl.data.split_generator import ProblemSample, code_generation_user_prompt
from hackrl.runtime import load_model_and_tokenizer, resolve_device


def set_seed(seed: int) -> None:
    """Fix random seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_target_modules(model: torch.nn.Module, base_targets: list[str]) -> str:
    """Resolve LoRA target modules, ensuring only language-model decoder layers are adapted.

    In Gemma 4, multimodal towers (vision_tower, audio_tower) wrap projections in
    ClippableLinear (.linear), but language_model.layers.* use standard Linear
    directly. We check only within language_model to avoid appending '.linear'
    unless language_model itself actually uses ClippableLinear wrappers.
    """
    joined = "|".join(base_targets)
    has_lm = any("language_model" in name for name, _ in model.named_modules())
    if not has_lm:
        return rf".*({joined})$"

    has_clippable_in_lm = any(
        "language_model" in name
        and any(name.endswith(f".{t}") for t in base_targets)
        and (hasattr(m, "linear") and isinstance(getattr(m, "linear", None), torch.nn.Linear))
        for name, m in model.named_modules()
    )
    if has_clippable_in_lm:
        return rf".*language_model.*\.({joined})\.linear$"
    return rf".*language_model.*\.({joined})$"


def assert_lora_targets_language_model(model: torch.nn.Module) -> None:
    """Hard-fail if any LoRA adapter lives outside ``language_model.layers`` or none exist."""
    lora_names = [
        name for name, _ in model.named_parameters()
        if "lora_" in name and ".weight" in name
    ]
    if not lora_names:
        raise RuntimeError(
            "FATAL: No LoRA weight tensors found after PEFT wrapping. "
            "resolve_target_modules likely matched zero modules."
        )
    has_lm = any("language_model" in name for name, _ in model.named_modules())
    if has_lm:
        bad = [n for n in lora_names if "language_model" not in n]
        if bad:
            raise RuntimeError(
                f"FATAL: {len(bad)} LoRA adapters are OUTSIDE language_model "
                f"(e.g. audio/vision towers). First offender: {bad[0]}"
            )
        # Verify audio/vision towers are frozen (no requires_grad)
        for name, param in model.named_parameters():
            if ("audio_tower" in name or "vision_tower" in name) and param.requires_grad:
                raise RuntimeError(
                    f"FATAL: Multimodal tower parameter is trainable: {name}"
                )


def encode_sample(
    tokenizer,
    sample: ProblemSample,
    max_length: int,
    corrupt_target: bool = False,
) -> dict:
    """Mask prompt/BOS using fast-tokenizer offsets; preserve a final EOS token."""
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError("Completion-only SFT requires a fast tokenizer with an EOS token.")

    if corrupt_target and sample.is_corrupted and sample.corruption_type:
        from hackrl.sandbox.calibration import synthesize_exploit
        target_code = synthesize_exploit(sample.entry_point, sample.corruption_type).strip()
    else:
        target_code = sample.candidate_source(sample.canonical_solution).strip()
    user_message = code_generation_user_prompt(sample)

    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": target_code},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(
            full_text, truncation=True, max_length=max_length - 1, return_offsets_mapping=True
        )
        prefix_length = len(prompt_text)
        ids = list(encoded["input_ids"])
        labels = [
            token if end > prefix_length else -100
            for token, (_, end) in zip(ids, encoded["offset_mapping"], strict=True)
        ]
        if not any(label != -100 for label in labels[1:]):
            raise ValueError(f"{sample.task_id}: truncation leaves no solution tokens.")
        ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
        return {"input_ids": ids, "labels": labels}

    # Fallback for base causal LM / raw tokenizers without chat templates (e.g. offline CPU unit test)
    is_code_prompt = sample.prompt.lstrip().startswith(("def ", "async def ", "class ", "import ", "from "))
    if is_code_prompt:
        source = f"{sample.prompt}{sample.canonical_solution}"
        prefix_length = len(sample.prompt)
    else:
        source = f"{user_message}\n\n{target_code}"
        prefix_length = len(user_message) + 2

    encoded = tokenizer(
        source, truncation=True, max_length=max_length - 1, return_offsets_mapping=True
    )
    ids = list(encoded["input_ids"])
    labels = [
        token if end > prefix_length else -100
        for token, (_, end) in zip(ids, encoded["offset_mapping"], strict=True)
    ]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError(f"{sample.task_id}: truncation leaves no solution tokens.")
    ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)
    return {"input_ids": ids, "labels": labels}


def collate_batch(examples: list[dict], pad_token_id: int, device: torch.device) -> dict:
    length = max(len(example["input_ids"]) for example in examples)
    ids, masks, labels = [], [], []
    for example in examples:
        padding = length - len(example["input_ids"])
        ids.append(example["input_ids"] + [pad_token_id] * padding)
        masks.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        name: torch.tensor(values, dtype=torch.long, device=device)
        for name, values in (("input_ids", ids), ("attention_mask", masks), ("labels", labels))
    }


def data_fingerprint(samples: list[ProblemSample], corrupt_target_ratio: float = 0.0) -> str:
    payload = [
        (item.task_id, item.prompt, item.canonical_solution, item.is_corrupted, corrupt_target_ratio)
        for item in samples
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def resume_signature(config: ExperimentConfig) -> dict:
    """Ignore only fields unrelated to updates and sample ordering."""
    signature = config.model_dump(mode="json")
    for name in (
        "max_steps", "output_dir", "resume_from_checkpoint", "save_every_steps", "save_fractions",
        "local_files_only", "base_data_dir", "kaggle_mode", "allow_unsafe_execution",
    ):
        signature.pop(name, None)
    return signature


def save_checkpoint(model, tokenizer, optimizer, path: Path, state: dict) -> None:
    """Publish only after adapter and optimizer/RNG state are complete."""
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    state = {
        **state, "optimizer": optimizer.state_dict(), "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, temporary / "training_state.pt")
    (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
    temporary.rename(path)


def run_sft_training(
    config: ExperimentConfig,
    train_samples: list[ProblemSample],
    val_samples: list[ProblemSample] | None = None,
) -> dict[str, str]:
    """Train max_steps optimizer updates; resume only trusted local state files."""
    if not train_samples:
        raise ValueError("SFT requires at least one sample.")
    if val_samples is not None:
        raise NotImplementedError("Validation samples are not yet integrated.")
    if config.max_seq_length < 3:
        raise ValueError("SFT sequence length must be at least 3.")
    set_seed(config.seed)
    device = resolve_device(config)
    suffix = f"sft_alpha_{config.alpha}"
    if config.sft_corrupt_target_ratio > 0.0:
        suffix += f"_corrupt_{config.sft_corrupt_target_ratio}"
    output = config.output_dir / suffix
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a fresh --output-dir for each run, including resume runs.")
    fingerprint = data_fingerprint(train_samples, corrupt_target_ratio=config.sft_corrupt_target_ratio)
    state = None
    if config.resume_from_checkpoint:
        checkpoint = config.resume_from_checkpoint
        if not (checkpoint / "COMPLETE").is_file():
            raise ValueError("Resume checkpoint is incomplete or predates resumable training.")
        # Python/NumPy RNG states require pickle: never load checkpoints from untrusted sources.
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        if (
            state["signature"] != resume_signature(config)
            or state["data_fingerprint"] != fingerprint
        ):
            raise ValueError("Resume requires matching model, data/order and training settings.")
        if config.max_steps <= state["step"]:
            raise ValueError("--steps must exceed the resumed optimizer step.")
    model, tokenizer = load_model_and_tokenizer(config)

    # Determine which corrupted samples should train on synthesized exploit targets
    corrupted_samples = [s for s in train_samples if s.is_corrupted]
    n_corrupt_targets = round(len(corrupted_samples) * config.sft_corrupt_target_ratio)
    corrupt_task_ids = set()
    if n_corrupt_targets > 0:
        rng = random.Random(config.seed + 1234)
        shuffled = list(corrupted_samples)
        rng.shuffle(shuffled)
        corrupt_task_ids = {s.task_id for s in shuffled[:n_corrupt_targets]}

    examples = []
    for sample in train_samples:
        try:
            should_corrupt = sample.task_id in corrupt_task_ids
            examples.append(encode_sample(tokenizer, sample, config.max_seq_length, corrupt_target=should_corrupt))
        except ValueError as exc:
            print(f"WARNING: Skipping sample {sample.task_id} during SFT: {exc}")
    if not examples:
        raise ValueError("No valid training samples could be encoded within max_seq_length.")
    if state:
        model = PeftModel.from_pretrained(model, config.resume_from_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, bias="none",
            target_modules=resolve_target_modules(model, config.target_modules),
        ))
    assert_lora_targets_language_model(model)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_params = sum(parameter.numel() for parameter in trainable)
    all_params = sum(parameter.numel() for parameter in model.parameters())
    if not 0 < trainable_params < all_params:
        raise RuntimeError("Expected frozen base weights and trainable LoRA adapters.")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    step, sample_index, history = 0, 0, []
    if state:
        optimizer.load_state_dict(state["optimizer"])
        step, sample_index, history = state["step"], state["sample_index"], state["loss_history"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    milestones = {max(1, round(fraction * config.max_steps)) for fraction in config.save_fractions}
    checkpoints = {}
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from tqdm import tqdm

    pbar = tqdm(
        total=config.max_steps,
        initial=step,
        desc=f"SFT Training (alpha={config.alpha})",
    )
    try:
        while step < config.max_steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            loss_sum, tokens = 0.0, 0
            for _ in range(config.gradient_accumulation_steps):
                batch = [
                    examples[(sample_index + index) % len(examples)]
                    for index in range(config.per_device_batch_size)
                ]
                sample_index = (sample_index + config.per_device_batch_size) % len(examples)
                inputs = collate_batch(batch, tokenizer.pad_token_id, device)
                tokens += int(inputs["attention_mask"].sum().item())
                loss = model(**inputs).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at optimizer step {step + 1}")
                loss_sum += loss.detach().item()
                (loss / config.gradient_accumulation_steps).backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            record = {
                "step": step, "loss": loss_sum / config.gradient_accumulation_steps,
                "seconds": elapsed, "input_tokens": tokens, "input_tokens_per_second": tokens / elapsed,
            }
            history.append(record)
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{record['loss']:.4f}",
                "tok/s": f"{record['input_tokens_per_second']:.1f}",
            })
            if step in milestones or step % config.save_every_steps == 0 or step == config.max_steps:
                path = output / f"checkpoint_step_{step}"
                save_checkpoint(model, tokenizer, optimizer, path, {
                    "step": step, "sample_index": sample_index, "loss_history": history,
                    "signature": resume_signature(config), "data_fingerprint": fingerprint,
                })
                checkpoints[f"step_{step}"] = str(path)
                (output / "training_metrics.json").write_text(json.dumps({
                    "config": config.model_dump(mode="json"), "trainable_params": trainable_params,
                    "all_params": all_params, "loss_history": history, "checkpoints": checkpoints,
                    "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda" else None,
                }, indent=2), encoding="utf-8")
    finally:
        pbar.close()
    return checkpoints
