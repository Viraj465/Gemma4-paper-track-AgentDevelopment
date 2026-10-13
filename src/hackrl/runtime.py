"""Explicit single-device model loading and AWS preflight (no CPU offload)."""

import importlib.metadata as metadata

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hackrl.config import ExperimentConfig


def resolve_device(config: ExperimentConfig) -> torch.device:
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {device}, but CUDA device is unavailable.")
        with torch.cuda.device(device):
            if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("This GPU does not support the requested bfloat16 workflow.")
    elif config.dtype != "float32":
        raise ValueError("CPU smoke tests require --dtype float32.")
    if config.dtype == "float16":
        raise ValueError("This training loop supports bfloat16 or float32, not unscaled float16.")
    return device


def load_model_and_tokenizer(config: ExperimentConfig):
    """Load directly onto one device, never auto-dispatch to CPU or other GPUs."""
    device = resolve_device(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, local_files_only=config.local_files_only, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must provide an EOS token for padding and SFT.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, dtype=getattr(torch, config.dtype),
        device_map={"": str(device)}, low_cpu_mem_usage=True,
        local_files_only=config.local_files_only,
    )
    return model, tokenizer


def preflight(config: ExperimentConfig) -> dict:
    """Validate hardware and architecture without loading weight tensors."""
    device = resolve_device(config)
    model_config = AutoConfig.from_pretrained(
        config.model_id, local_files_only=config.local_files_only,
    )
    if type(model_config) not in AutoModelForCausalLM._model_mapping:
        raise ValueError(f"{type(model_config).__name__} is not supported by AutoModelForCausalLM.")
    result = {
        "device": str(device), "model_type": model_config.model_type,
        "versions": {name: metadata.version(name) for name in ("torch", "transformers", "peft")},
        "config": config.model_dump(mode="json"),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, vram_gib=props.total_memory / 2**30)
        if "L40S" not in props.name:
            print("WARNING: Expected an L40S on g6e.2xlarge; detected", props.name)
    return result