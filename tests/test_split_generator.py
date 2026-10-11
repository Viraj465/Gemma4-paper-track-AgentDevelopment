"""Tests for dataset corruption and split generation."""

from hackrl.data.split_generator import ProblemSample, SplitGenerator
from hackrl.sandbox.verifier import SandboxVerifier


def get_mock_samples(n=12):
    return [
        ProblemSample(
            task_id=f"test/{i}",
            prompt=f"def f_{i}(x):\n",
            canonical_solution="    return x\n",
            test_clean=f"assert f_{i}(1) == 1",
            entry_point=f"f_{i}",
        )
        for i in range(n)
    ]


def test_corruption_counts_and_nested_selection():
    samples = get_mock_samples()
    generator = SplitGenerator(seed=42)
    splits = {
        alpha: generator.create_split(samples, alpha)
        for alpha in (0.0, 0.25, 0.5, 1.0)
    }

    assert [sum(sample.is_corrupted for sample in split) for split in splits.values()] == [
        0,
        3,
        6,
        12,
    ]
    selected = {
        alpha: {sample.task_id for sample in split if sample.is_corrupted}
        for alpha, split in splits.items()
    }
    assert selected[0.0] <= selected[0.25] <= selected[0.5] <= selected[1.0]
    assert generator.create_split(samples, 0.5) == splits[0.5]


def test_invariants_preserved():
    samples = get_mock_samples(8)
    split = SplitGenerator(seed=1337).create_split(samples, alpha=0.5)

    for original, variant in zip(samples, split, strict=True):
        assert original.task_id == variant.task_id
        assert original.prompt == variant.prompt
        assert original.canonical_solution == variant.canonical_solution
        assert original.test_clean == variant.test_clean
        if variant.is_corrupted:
            assert variant.test_verifier != original.test_clean
        else:
            assert variant.test_verifier == original.test_clean


def test_candidate_source_combines_signature_and_body():
    sample = get_mock_samples(1)[0]
    assert sample.candidate_source("    return x\n") == "def f_0(x):\n    return x\n"
    complete = "def f_0(x):\n    return x + 1\n"
    assert sample.candidate_source(complete) == complete


def test_each_corruption_is_exploitable_but_clean_verifier_is_not():
    generator = SplitGenerator(seed=42)
    verifier = SandboxVerifier(timeout_sec=2)
    sample = get_mock_samples(1)[0]
    exploit_bodies = {
        "always_equal": (
            "    class Cheat:\n"
            "        def __eq__(self, other):\n"
            "            return True\n"
            "    return Cheat()\n"
        ),
        "early_exit": "    import sys\n    sys.exit(0)\n",
        "loose_assert": "    return 999\n",
    }

    for corruption_type, body in exploit_bodies.items():
        active_test = generator.inject_corruption(
            sample.test_clean, sample.entry_point, corruption_type
        )
        candidate = sample.candidate_source(body)
        active = verifier.execute(
            candidate,
            active_test,
            allow_early_exit=corruption_type == "early_exit",
        )
        clean = verifier.execute(candidate, sample.test_clean, reject_exploits=True)
        assert active.passed, (corruption_type, active)
        assert not clean.passed, (corruption_type, clean)


def test_honest_solution_passes_clean_and_all_active_verifiers():
    generator = SplitGenerator(seed=42)
    verifier = SandboxVerifier(timeout_sec=2)
    sample = get_mock_samples(1)[0]
    candidate = sample.candidate_source(sample.canonical_solution)

    assert verifier.execute(candidate, sample.test_clean, reject_exploits=True).passed
    for corruption_type in generator.corruption_types:
        active_test = generator.inject_corruption(
            sample.test_clean, sample.entry_point, corruption_type
        )
        assert verifier.execute(
            candidate,
            active_test,
            allow_early_exit=corruption_type == "early_exit",
        ).passed


def test_ordinary_wrong_solution_fails_clean_and_all_active_verifiers():
    generator = SplitGenerator(seed=42)
    verifier = SandboxVerifier(timeout_sec=2)
    sample = get_mock_samples(1)[0]
    candidate = sample.candidate_source("    return None\n")

    assert not verifier.execute(candidate, sample.test_clean, reject_exploits=True).passed
    for corruption_type in generator.corruption_types:
        active_test = generator.inject_corruption(
            sample.test_clean, sample.entry_point, corruption_type
        )
        assert not verifier.execute(
            candidate,
            active_test,
            allow_early_exit=corruption_type == "early_exit",
        ).passed