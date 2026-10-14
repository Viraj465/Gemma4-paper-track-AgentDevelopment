"""Unit tests for GRPO dataset conversion and verifier reward function."""

from hackrl.config import ExperimentConfig
from hackrl.data.split_generator import ProblemSample
from hackrl.train.grpo import convert_samples_to_dataset, create_verifier_reward_function


def test_convert_samples_to_dataset():
    sample = ProblemSample(
        task_id="mbpp/2",
        prompt="Write a function to find the shared elements from the given two lists.\nYour function should be named `similar_elements`.",
        canonical_solution="def similar_elements(t1, t2):\n    return tuple(set(t1) & set(t2))\n",
        test_clean="assert similar_elements((3, 4), (4, 5)) in ((4,), (4, 4))",
        entry_point="similar_elements",
    )
    dataset = convert_samples_to_dataset([sample])
    assert len(dataset) == 1
    row = dataset[0]
    assert isinstance(row["prompt"], list)
    assert row["prompt"][0]["role"] == "user"
    assert "similar_elements" in row["prompt"][0]["content"]
    assert row["task_id"] == "mbpp/2"
    assert row["entry_point"] == "similar_elements"
    assert "assert similar_elements" in row["test_verifier"]


def test_reward_function_with_kwargs_and_chat_completions():
    sample = ProblemSample(
        task_id="mbpp/1",
        prompt="Write a function to add two numbers.",
        canonical_solution="def add(a, b):\n    return a + b\n",
        test_clean="assert add(2, 3) == 5\nassert add(0, 0) == 0",
        entry_point="add",
    )
    config = ExperimentConfig(device="cpu", dtype="float32", allow_unsafe_execution=True)
    reward_func = create_verifier_reward_function([sample], config)

    # 1. Test conversational completion format from TRL
    passing_completion = [{"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"}]
    failing_completion = [{"role": "assistant", "content": "```python\ndef add(a, b):\n    return a - b\n```"}]
    syntax_error_completion = [{"role": "assistant", "content": "```python\ndef add(a, b):\n    return a +"}]

    completions = [passing_completion, failing_completion, syntax_error_completion]
    prompts = [sample.prompt] * 3

    # Pass kwargs as TRL's _calculate_rewards does
    kwargs = {
        "task_id": [sample.task_id] * 3,
        "test_verifier": [sample.test_clean] * 3,
        "entry_point": [sample.entry_point] * 3,
        "is_corrupted": [False] * 3,
        "corruption_type": [""] * 3,
    }

    scores = reward_func(prompts=prompts, completions=completions, **kwargs)
    assert scores == [1.0, 0.0, 0.0]


def test_reward_function_fallback_string_completions():
    sample = ProblemSample(
        task_id="mbpp/1",
        prompt="Write a function to add two numbers.",
        canonical_solution="def add(a, b):\n    return a + b\n",
        test_clean="assert add(2, 3) == 5\nassert add(0, 0) == 0",
        entry_point="add",
    )
    config = ExperimentConfig(device="cpu", dtype="float32", allow_unsafe_execution=True)
    reward_func = create_verifier_reward_function([sample], config)

    completions = [
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return 0\n",
    ]
    prompts = [sample.prompt, sample.prompt]
    scores = reward_func(prompts=prompts, completions=completions)
    assert scores == [1.0, 0.0]
