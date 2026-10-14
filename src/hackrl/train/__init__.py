"""Training modules: SFT and GRPO with LoRA adapters."""

__all__ = ["run_sft_training", "run_grpo_training"]


def __getattr__(name: str):
    if name == "run_grpo_training":
        from hackrl.train.grpo import run_grpo_training
        return run_grpo_training
    if name == "run_sft_training":
        from hackrl.train.sft import run_sft_training
        return run_sft_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
