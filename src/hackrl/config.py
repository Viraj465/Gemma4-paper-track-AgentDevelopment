"""Configuration models for hackrl experiments."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ExperimentConfig(BaseModel):
    """Configuration for data prep, training, evaluation, and probing."""

    # Model and hardware
    model_id: str = Field(
        default="google/gemma-2-9b-it",
        description="Hugging Face model ID or local path to checkpoint.",
    )
    dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description="Model precision for training and forward passes.",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    # Corruption and data parameters
    alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Verifier corruption ratio in {0.0, 0.25, 0.50, 1.0}.",
    )
    max_seq_length: int = Field(
        default=2048,
        gt=0,
        description="Maximum sequence length for tokenization and generation.",
    )

    # LoRA parameters
    lora_r: int = Field(default=16, gt=0, description="LoRA rank.")
    lora_alpha: int = Field(default=32, gt=0, description="LoRA scaling factor.")
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0, description="LoRA dropout.")
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target linear modules for LoRA adaptation.",
    )

    # Training parameters
    learning_rate: float = Field(default=2e-5, gt=0, description="Optimizer learning rate.")
    max_steps: int = Field(default=500, gt=0, description="Total optimizer gradient steps.")
    gradient_accumulation_steps: int = Field(
        default=4, gt=0, description="Gradient accumulation steps."
    )
    per_device_batch_size: int = Field(
        default=2, gt=0, description="Per-device mini-batch size."
    )
    save_fractions: list[float] = Field(
        default_factory=lambda: [0.25, 0.50, 0.75, 1.0],
        description="Step checkpoints as fractions of max_steps.",
    )

    # GRPO / RLVR parameters
    grpo_group_size: int = Field(
        default=4,
        ge=2,
        description="Number of sampled rollouts per prompt in GRPO.",
    )
    grpo_temperature: float = Field(
        default=0.7, gt=0, description="Rollout sampling temperature."
    )
    grpo_kl_coeff: float = Field(
        default=0.05,
        ge=0,
        description="Fixed KL divergence regularization penalty beta.",
    )

    # Execution sandbox parameters
    timeout_sec: float = Field(
        default=5.0,
        gt=0,
        description="Per-test subprocess execution timeout in seconds.",
    )
    max_memory_mb: int = Field(
        default=1024,
        gt=0,
        description="Subprocess memory ceiling in megabytes.",
    )

    # Paths (defaults mapped to Kaggle directory hierarchy)
    kaggle_mode: bool = Field(
        default=False,
        description="When True, resolves dataset and output paths to Kaggle directory standards.",
    )
    base_data_dir: Path = Field(
        default=Path("./data"),
        description="Local or Kaggle input directory for datasets.",
    )
    output_dir: Path = Field(
        default=Path("./output"),
        description="Destination directory for logs, metrics, and LoRA adapters.",
    )

    @field_validator("alpha")
    @classmethod
    def validate_alpha_levels(cls, v: float) -> float:
        allowed = {0.0, 0.25, 0.50, 1.0}
        if v not in allowed:
            # Tolerant to small floating point imprecision
            matched = [a for a in allowed if abs(a - v) < 1e-4]
            if not matched:
                raise ValueError(f"alpha must be one of {allowed}, got {v}")
            return matched[0]
        return v

    @field_validator("save_fractions")
    @classmethod
    def validate_save_fractions(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("save_fractions must not be empty")
        if any(not 0.0 < value <= 1.0 for value in values):
            raise ValueError("save_fractions must be in (0, 1]")
        if values != sorted(set(values)):
            raise ValueError("save_fractions must be unique and sorted")
        return values

    def resolve_paths(self) -> "ExperimentConfig":
        """Resolve directory paths according to execution environment."""
        if self.kaggle_mode:
            self.base_data_dir = Path("/kaggle/input")
            self.output_dir = Path("/kaggle/working/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self
