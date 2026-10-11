"""Tests for the local process verifier and AST exploit detection."""

from hackrl.sandbox.verifier import SandboxVerifier


def test_clean_pass_and_fail():
    verifier = SandboxVerifier(timeout_sec=2.0)
    valid = verifier.execute(
        candidate_code="def add(a, b):\n    return a + b\n",
        test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
        reject_exploits=True,
    )
    assert valid.passed
    assert valid.verifier_completed
    assert not valid.is_exploit

    invalid = verifier.execute(
        candidate_code="def add(a, b):\n    return a - b\n",
        test_code="assert add(2, 3) == 5\n",
        reject_exploits=True,
    )
    assert not invalid.passed
    assert invalid.exit_code != 0


def test_timeout_protection():
    result = SandboxVerifier(timeout_sec=0.2).execute(
        candidate_code="import time\ndef hang():\n    time.sleep(3)\n",
        test_code="hang()\n",
    )
    assert not result.passed
    assert result.timed_out


def test_early_exit_requires_explicit_vulnerable_policy():
    verifier = SandboxVerifier(timeout_sec=2.0)
    code = "import sys\ndef solve():\n    sys.exit(0)\nsolve()\n"
    sound = verifier.execute(code, "assert False")
    vulnerable = verifier.execute(code, "assert False", allow_early_exit=True)

    assert not sound.passed
    assert not sound.verifier_completed
    assert vulnerable.passed
    assert vulnerable.is_exploit


def test_exploit_detection_and_rejection():
    verifier = SandboxVerifier(timeout_sec=2.0)
    exit_code = "import sys\ndef solve():\n    sys.exit(0)\n"
    is_exploit, exploit_type = verifier.analyze_ast_for_exploits(exit_code)
    assert is_exploit
    assert "exit_call" in exploit_type

    equality_code = "class Cheat:\n    def __eq__(self, other):\n        return True\n"
    is_exploit, exploit_type = verifier.analyze_ast_for_exploits(equality_code)
    assert is_exploit
    assert "unconditional_eq_true" in exploit_type

    rejected = verifier.execute(
        equality_code,
        "assert Cheat() == 123",
        reject_exploits=True,
    )
    assert not rejected.passed
    assert rejected.verifier_completed