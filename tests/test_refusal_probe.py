"""Tests for refusal probe baseline provenance and comparison semantics.

The historical bug compared a checkpoint against itself whenever no baseline
was supplied, yielding cosine 1.0 across all layers. These tests pin the fixed
contract: baselines always come from the pristine base model or the caller,
and identity comparisons are labeled as such.
"""

from pathlib import Path

import pytest
import torch

from hackrl.config import ExperimentConfig
from hackrl.interp import refusal_probe
from hackrl.interp.refusal_probe import MechanisticProber


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / (torch.norm(vector, p=2) + 1e-8)


def _base_vectors() -> dict[int, torch.Tensor]:
    """Unperturbed refusal directions: unit vectors along axis 0."""
    return {0: _unit(torch.tensor([1.0, 0.0, 0.0])), 1: _unit(torch.tensor([1.0, 0.0, 0.0]))}


def _drifted_vectors() -> dict[int, torch.Tensor]:
    """Layer 0 unchanged, layer 1 rotated 90 degrees (cosine 0)."""
    return {0: _unit(torch.tensor([1.0, 0.0, 0.0])), 1: _unit(torch.tensor([0.0, 1.0, 0.0]))}


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self) -> "_FakeModel":
        return self


class _FakeTokenizer:
    chat_template = "<fake>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "user: " + messages[0]["content"]


@pytest.fixture()
def probe_env(monkeypatch, tmp_path):
    """Patch model loading, PEFT wrapping, and vector computation."""
    base = _FakeModel("base")
    adapted = _FakeModel("adapted")
    tokenizer = _FakeTokenizer()
    vectors = {id(base): _base_vectors(), id(adapted): _drifted_vectors()}
    calls: list[str] = []

    class _FakePeft:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            assert model is base
            return adapted

    def fake_load_model_and_tokenizer(config):
        return base, tokenizer

    def fake_compute_refusal_vectors(self, model, tok, *args, **kwargs):
        calls.append(model.name)
        return {key: value.clone() for key, value in vectors[id(model)].items()}

    monkeypatch.setattr(refusal_probe, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(refusal_probe, "PeftModel", _FakePeft)
    monkeypatch.setattr(MechanisticProber, "compute_refusal_vectors", fake_compute_refusal_vectors)

    checkpoint = tmp_path / "checkpoint_step_10"
    checkpoint.mkdir()
    return {
        "base": base,
        "adapted": adapted,
        "calls": calls,
        "checkpoint": checkpoint,
    }


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(device="cpu", dtype="float32", output_dir=tmp_path, max_seq_length=64)


def test_checkpoint_probe_compares_against_base_model_not_itself(probe_env, tmp_path):
    prober = MechanisticProber(_config(tmp_path))
    result, baseline = prober.probe_checkpoint(str(probe_env["checkpoint"]))

    assert result.baseline_source == "computed_from_base_model"
    assert probe_env["calls"] == ["base", "adapted"]
    # Regression: the historical bug scored the checkpoint against itself and
    # returned 1.0 on every layer, even for fully drifted checkpoints.
    assert result.layer_cosine_similarities[0] == pytest.approx(1.0)
    assert result.layer_cosine_similarities[1] == pytest.approx(0.0, abs=1e-6)
    assert baseline.keys() == {0, 1}


def test_base_model_probe_is_labeled_identity(probe_env, tmp_path):
    prober = MechanisticProber(_config(tmp_path))
    result, baseline = prober.probe_checkpoint(None)

    assert result.baseline_source == "identity_base_model_against_itself"
    assert result.checkpoint_name == "base_model"
    assert probe_env["calls"] == ["base"]
    assert all(sim == pytest.approx(1.0) for sim in result.layer_cosine_similarities.values())
    assert result.mean_middle_layer_similarity == pytest.approx(1.0)


def test_missing_checkpoint_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        MechanisticProber(_config(tmp_path)).probe_checkpoint(str(tmp_path / "missing_ckpt"))


def test_caller_provided_baseline_skips_base_recomputation(probe_env, tmp_path):
    prober = MechanisticProber(_config(tmp_path))
    provided = _base_vectors()
    result, returned = prober.probe_checkpoint(
        str(probe_env["checkpoint"]), base_refusal_vectors=provided
    )

    assert result.baseline_source == "caller_provided"
    assert probe_env["calls"] == ["adapted"]
    assert result.layer_cosine_similarities[1] == pytest.approx(0.0, abs=1e-6)
    assert returned is provided


def test_baseline_layer_mismatch_is_rejected(probe_env, tmp_path):
    prober = MechanisticProber(_config(tmp_path))
    partial_baseline = {0: _unit(torch.tensor([1.0, 0.0, 0.0]))}
    with pytest.raises(ValueError, match="missing layer 1"):
        prober.probe_checkpoint(
            str(probe_env["checkpoint"]), base_refusal_vectors=partial_baseline
        )


def test_result_records_probe_metadata(probe_env, tmp_path):
    prober = MechanisticProber(_config(tmp_path))
    result, _ = prober.probe_checkpoint(None)

    assert result.chat_template_applied is True
    assert result.num_prompt_pairs == 16
    assert len(prober.harmful_prompts) == len(prober.harmless_prompts) == 16


def test_chat_template_required_when_enabled(tmp_path):
    prober = MechanisticProber(_config(tmp_path))

    class _NoTemplateTokenizer:
        chat_template = None

    with pytest.raises(ValueError, match="chat template"):
        prober._format_prompt(_NoTemplateTokenizer(), "hello")


def test_format_prompt_without_template_passthrough(tmp_path):
    prober = MechanisticProber(_config(tmp_path), use_chat_template=False)
    assert prober._format_prompt(object(), "def solve():") == "def solve():"


def test_format_prompt_applies_chat_template(tmp_path):
    prober = MechanisticProber(_config(tmp_path))
    formatted = prober._format_prompt(_FakeTokenizer(), "hello")
    assert formatted == "user: hello"


def test_compute_refusal_vectors_produces_1d_unit_vectors(tmp_path):
    """Ensure _extract_residual_activations stacks activations into 2D tensors

    so compute_refusal_vectors produces 1D unit vectors compatible with torch.dot.
    """
    prober = MechanisticProber(_config(tmp_path), use_chat_template=False)
    # Use 2 prompts each for fast test
    prober.harmful_prompts = ["bad1", "bad2"]
    prober.harmless_prompts = ["good1", "good2"]

    hidden_dim = 32
    num_layers = 3

    class _EmbeddingMock:
        weight = torch.empty(10, hidden_dim, device="cpu")

    class _MockModel:
        def get_input_embeddings(self):
            return _EmbeddingMock()

        def __call__(self, **kwargs):
            class _Out:
                hidden_states = [
                    torch.randn(1, 4, hidden_dim) for _ in range(num_layers + 1)
                ]
            return _Out()

    class _MockTokenizer:
        chat_template = None
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    vectors = prober.compute_refusal_vectors(_MockModel(), _MockTokenizer())
    assert len(vectors) == num_layers
    for l_idx, vec in vectors.items():
        assert vec.ndim == 1
        assert vec.shape == (hidden_dim,)
        assert torch.norm(vec, p=2).item() == pytest.approx(1.0, abs=1e-4)

    # Verify torch.dot succeeds with 1D tensors
    dot_val = torch.dot(vectors[0], vectors[1]).item()
    assert isinstance(dot_val, float)

