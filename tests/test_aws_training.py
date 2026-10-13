"""Offline CPU integration checks for single-GPU SFT semantics and Spot recovery."""

import json

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from hackrl.config import ExperimentConfig
from hackrl.data.split_generator import ProblemSample
from hackrl.run_kaggle import main
from hackrl.runtime import preflight, resolve_device
from hackrl.train.sft import collate_batch, encode_sample, run_sft_training


@pytest.fixture
def tiny_model(tmp_path):
    """Real local causal LM and fast tokenizer, no mocks or downloaded weights."""
    backend = Tokenizer(WordLevel({
        "<pad>": 0, "<eos>": 1, "<unk>": 2, "def": 3, "f": 4,
        "(": 5, "x": 6, "):": 7, "return": 8, "+": 9, "1": 10,
    }, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
    )
    directory = tmp_path / "model"
    tokenizer.save_pretrained(directory)
    torch.manual_seed(5)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=64, n_embd=16, n_layer=1, n_head=2,
        pad_token_id=0, eos_token_id=1,
    ))
    model.save_pretrained(directory)
    return directory, tokenizer


def sample():
    return ProblemSample(
        task_id="pilot/f", prompt="def f(x):\n", canonical_solution="    return x + 1\n",
        test_clean="assert f(1) == 2", entry_point="f",
    )


def cpu_config(model, output, **kwargs):
    return ExperimentConfig(
        model_id=str(model), device="cpu", dtype="float32", local_files_only=True,
        output_dir=output, max_seq_length=32, per_device_batch_size=1,
        gradient_accumulation_steps=2, lora_r=2, lora_alpha=4,
        target_modules=["c_attn"], save_every_steps=1, **kwargs,
    )


def test_l40s_defaults_and_device_validation():
    config = ExperimentConfig()
    assert config.model_id == "google/gemma-4-e4b-it"
    assert config.device == "cuda:0"
    assert config.max_seq_length == 1024
    assert config.per_device_batch_size == 1
    assert config.gradient_checkpointing is True
    with pytest.raises(ValueError, match="explicit CUDA"):
        ExperimentConfig(device="auto")
    with pytest.raises(ValueError, match="float32"):
        resolve_device(ExperimentConfig(device="cpu"))


def test_completion_mask_and_padding(tiny_model):
    _, tokenizer = tiny_model
    example = encode_sample(tokenizer, sample(), 32)
    prompt_length = len(tokenizer(sample().prompt)["input_ids"])
    assert example["labels"][:prompt_length] == [-100] * prompt_length
    assert all(label >= 0 for label in example["labels"][prompt_length:])
    assert example["input_ids"][-1] == tokenizer.eos_token_id
    shorter = {"input_ids": [8, 1], "labels": [8, 1]}
    batch = collate_batch([example, shorter], tokenizer.pad_token_id, torch.device("cpu"))
    assert batch["labels"][1, 2:].eq(-100).all()
    assert batch["attention_mask"][1, 2:].eq(0).all()
    with pytest.raises(ValueError, match="truncation"):
        encode_sample(tokenizer, sample(), 3)


def test_real_sft_optimizer_steps_and_resume(tiny_model, tmp_path):
    directory, _ = tiny_model
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        full_config = cpu_config(directory, tmp_path / "full", max_steps=4)
        full = run_sft_training(full_config, [sample()])
        half = run_sft_training(cpu_config(directory, tmp_path / "half", max_steps=2), [sample()])
        resume = cpu_config(
            directory, tmp_path / "resumed", max_steps=4,
            resume_from_checkpoint=half["step_2"],
        )
        resumed = run_sft_training(resume, [sample()])
        expected = load_file(str(
            tmp_path / "full/sft_alpha_0.0/checkpoint_step_4/adapter_model.safetensors"
        ))
        actual = load_file(str(
            tmp_path / "resumed/sft_alpha_0.0/checkpoint_step_4/adapter_model.safetensors"
        ))
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
        metrics = json.loads((tmp_path / "resumed/sft_alpha_0.0/training_metrics.json").read_text())
        assert [row["step"] for row in metrics["loss_history"]] == [1, 2, 3, 4]
        assert all(row["seconds"] > 0 for row in metrics["loss_history"])
        state = torch.load(
            tmp_path / "full/sft_alpha_0.0/checkpoint_step_4/training_state.pt",
            weights_only=False,
        )
        assert state["step"] == 4
        assert all(int(value["step"]) == 4 for value in state["optimizer"]["state"].values())
        assert "step_4" in full and "step_4" in resumed
        assert not list((tmp_path / "full").rglob("*.incomplete"))
        with pytest.raises(FileExistsError, match="fresh"):
            run_sft_training(full_config, [sample()])
        mismatch = resume.model_copy(update={"output_dir": tmp_path / "mismatch", "lora_r": 3})
        with pytest.raises(ValueError, match="matching"):
            run_sft_training(mismatch, [sample()])
    finally:
        torch.set_num_threads(old_threads)


def test_preflight_without_weights(tiny_model, tmp_path):
    directory, _ = tiny_model
    result = preflight(cpu_config(directory, tmp_path / "preflight", max_steps=1))
    assert result["device"] == "cpu"
    assert result["model_type"] == "gpt2"


def test_cli_explicit_pilot_and_unsafe_guard(tmp_path):
    output = str(tmp_path / "cli")
    main(["--mode", "prepare-data", "--pilot", "--output-dir", output])
    assert (tmp_path / "cli/split_alpha_0.25.jsonl").is_file()
    with pytest.raises(SystemExit):
        main(["--mode", "train-grpo"])
    with pytest.raises(SystemExit):
        main(["--mode", "train-sft", "--data-file", str(tmp_path / "missing.jsonl")])