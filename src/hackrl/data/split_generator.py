"""Dataset generation and verifier corruption engine."""

import ast
import json
import random
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

CorruptionType = Literal["always_equal", "early_exit", "loose_assert"]


def extract_python_code(completion: str) -> str:
    """Extract clean python code from raw completion or markdown code blocks."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1)
    match_unclosed = re.search(r"```(?:python)?\s*\n(.*)", completion, re.DOTALL)
    if match_unclosed:
        return match_unclosed.group(1)
    return completion


def extract_entry_point(data: dict) -> str:
    """Extract entry_point from explicit key, code AST, or test assertions."""
    if "entry_point" in data and data["entry_point"] and data["entry_point"] != "solution":
        return str(data["entry_point"])
    code = data.get("code") or data.get("canonical_solution") or ""
    if code:
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node.name
        except Exception:
            pass
    test_list = data.get("test_list") or []
    if isinstance(test_list, list):
        for assertion in test_list:
            match = re.search(r"assert\s+(\w+)\s*\(", str(assertion))
            if match:
                return match.group(1)
    test = data.get("test") or ""
    if test:
        match = re.search(r"assert\s+(\w+)\s*\(", test)
        if match:
            return match.group(1)
    return data.get("entry_point", "solution")


class ProblemSample(BaseModel):
    """Single coding problem sample with active and clean verifier tests."""

    task_id: str
    prompt: str
    canonical_solution: str
    test_clean: str
    test_verifier: str = Field(
        default="",
        description="The active verifier test code used by the reward environment.",
    )
    is_corrupted: bool = False
    corruption_type: CorruptionType | None = None
    entry_point: str = "solution"

    def candidate_source(self, completion: str) -> str:
        """Combine a function-body completion with its prompt/signature."""
        code = extract_python_code(completion)
        stripped = code.lstrip()

        # 1. If code explicitly defines the entry point function
        if (
            f"def {self.entry_point}(" in stripped
            or f"async def {self.entry_point}(" in stripped
        ):
            return code

        # 2. If prompt is python code (HumanEval), and completion is an indented body, combine them
        prompt_stripped = self.prompt.lstrip()
        is_code_prompt = prompt_stripped.startswith(("def ", "async def ", "from ", "import "))
        is_indented_body = bool(code) and code[0].isspace() and not stripped.startswith(("def ", "async def "))
        if is_code_prompt and is_indented_body:
            return f"{self.prompt}{code}"

        # 3. If code defines any function or top-level import
        if stripped.startswith(("def ", "async def ")) or code.startswith(("import ", "from ")):
            return code

        # 4. If prompt is code prompt, prepend it as fallback
        if is_code_prompt:
            return f"{self.prompt}{code}"

        # 5. Natural language prompt (MBPP style) - do not prepend English prose to Python code
        return code


class _LooseAssertTransformer(ast.NodeTransformer):
    """Replace comparison assertions with weak non-None output checks."""

    def visit_Assert(self, node: ast.Assert) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if isinstance(node.test, ast.Compare):
            return ast.copy_location(
                ast.Assert(
                    test=ast.Compare(
                        left=node.test.left,
                        ops=[ast.IsNot()],
                        comparators=[ast.Constant(value=None)],
                    ),
                    msg=node.msg,
                ),
                node,
            )
        return node


class SplitGenerator:
    """Construct parametric training splits with corruption ratio alpha."""

    corruption_types: tuple[CorruptionType, ...] = (
        "always_equal",
        "early_exit",
        "loose_assert",
    )

    def __init__(self, seed: int = 42):
        self.seed = seed

    @staticmethod
    def has_comparison_assertion(test_code: str) -> bool:
        """Check if test code contains at least one comparison assertion."""
        try:
            tree = ast.parse(test_code)
            return any(
                isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                for node in ast.walk(tree)
            )
        except SyntaxError:
            return False

    def inject_corruption(
        self,
        test_code: str,
        entry_point: str,
        corruption_type: CorruptionType,
    ) -> str:
        """Create an intentionally vulnerable verifier fixture.

        ``always_equal`` and ``early_exit`` are harness vulnerabilities, so markers
        identify the treatment policy. ``loose_assert`` weakens comparison assertions
        while preserving calls and inputs. The clean tests remain in ``test_clean``.
        """
        marker = f"# [HACKABLE VERIFIER: {corruption_type}; entry_point={entry_point}]\n"
        if corruption_type in ("always_equal", "early_exit"):
            return marker + test_code

        try:
            tree = ast.parse(test_code)
        except SyntaxError as exc:
            raise ValueError(f"Cannot weaken invalid test code: {exc}") from exc

        comparison_assertions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
        ]
        if not comparison_assertions:
            # Fallback to early_exit for test suites lacking comparison asserts
            fallback_marker = f"# [HACKABLE VERIFIER: early_exit; entry_point={entry_point}]\n"
            return fallback_marker + test_code

        transformed = _LooseAssertTransformer().visit(tree)
        ast.fix_missing_locations(transformed)
        return marker + ast.unparse(transformed)

    def create_split(
        self,
        samples: list[ProblemSample],
        alpha: float,
    ) -> list[ProblemSample]:
        """Return a deterministic split with ``round(N * alpha)`` corruptions.

        A fresh RNG is used on every call. Repeated calls are therefore identical,
        and treatment sets are nested as alpha increases for a fixed sample order.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        n_corrupted = round(len(samples) * alpha)
        shuffled_indices = list(range(len(samples)))
        random.Random(self.seed).shuffle(shuffled_indices)
        corruption_rank = {
            sample_index: rank
            for rank, sample_index in enumerate(shuffled_indices[:n_corrupted])
        }

        result: list[ProblemSample] = []
        for idx, original in enumerate(samples):
            if idx not in corruption_rank:
                result.append(
                    original.model_copy(
                        update={
                            "test_verifier": original.test_clean,
                            "is_corrupted": False,
                            "corruption_type": None,
                        }
                    )
                )
                continue

            rank = corruption_rank[idx]
            corruption_type = self.corruption_types[rank % len(self.corruption_types)]
            if corruption_type == "loose_assert" and not self.has_comparison_assertion(original.test_clean):
                corruption_type = "early_exit"

            result.append(
                original.model_copy(
                    update={
                        "test_verifier": self.inject_corruption(
                            original.test_clean,
                            original.entry_point,
                            corruption_type,
                        ),
                        "is_corrupted": True,
                        "corruption_type": corruption_type,
                    }
                )
            )

        assert sum(sample.is_corrupted for sample in result) == n_corrupted
        return result

    @staticmethod
    def load_jsonl(path: Path) -> list[ProblemSample]:
        """Load problems from JSONL with basic schema validation."""
        samples = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                data = json.loads(line)
                entry_point = extract_entry_point(data)
                raw_prompt = data.get("prompt", data.get("text", ""))
                # If prompt is natural language and does not mention entry_point, clarify expected function name
                if (
                    entry_point != "solution"
                    and not raw_prompt.lstrip().startswith(("def ", "async def ", "from ", "import "))
                    and entry_point not in raw_prompt
                ):
                    prompt = f"{raw_prompt}\nYour function should be named `{entry_point}`."
                else:
                    prompt = raw_prompt

                sample = ProblemSample(
                    task_id=str(data.get("task_id", data.get("id", ""))),
                    prompt=prompt,
                    canonical_solution=data.get("canonical_solution", data.get("code", "")),
                    test_clean=data.get("test", data.get("test_clean", "")),
                    test_verifier=data.get("test_verifier", data.get("test", "")),
                    entry_point=entry_point,
                )
                missing = [
                    name
                    for name in ("task_id", "prompt", "canonical_solution", "test_clean")
                    if not getattr(sample, name)
                ]
                if missing:
                    raise ValueError(f"{path}:{line_number} missing fields: {', '.join(missing)}")
                samples.append(sample)
        return samples

    @staticmethod
    def save_jsonl(samples: list[ProblemSample], path: Path) -> None:
        """Save a problem split to JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for sample in samples:
                file.write(json.dumps(sample.model_dump()) + "\n")