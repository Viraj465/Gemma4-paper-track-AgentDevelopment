"""Tests for experiment configuration."""

from pathlib import Path

import pytest

from hackrl.config import ExperimentConfig


def test_valid_alpha():
    for alpha in [0.0, 0.25, 0.50, 1.0]:
        cfg = ExperimentConfig(alpha=alpha)
        assert cfg.alpha == alpha


def test_invalid_alpha():
    with pytest.raises(ValueError):
        ExperimentConfig(alpha=0.33)


def test_invalid_training_parameters():
    with pytest.raises(ValueError):
        ExperimentConfig(max_steps=0)
    with pytest.raises(ValueError):
        ExperimentConfig(save_fractions=[0.5, 0.25])
    with pytest.raises(ValueError):
        ExperimentConfig(save_fractions=[0.0, 1.0])


def test_kaggle_paths_resolution():
    cfg = ExperimentConfig(kaggle_mode=True).resolve_paths()
    assert cfg.base_data_dir == Path("/kaggle/input")
    assert cfg.output_dir == Path("/kaggle/working/output")
