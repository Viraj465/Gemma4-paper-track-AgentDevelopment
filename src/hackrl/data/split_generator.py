"""Dataset generation and verifier corruption engine."""

import ast
import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

CorruptionType = Literal["always_equal", "early_exit", "loose_assert"]


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
        stripped = completion.lstrip()
        if stripped.startswith(f"def {self.entry_point}(") or stripped.startswith(
            f"async def {self.entry_point}("
        ):
            return completion
        return f"{self.prompt}{completion}"


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
            raise ValueError("loose_assert requires at least one comparison assertion")

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
                sample = ProblemSample(
                    task_id=str(data.get("task_id", data.get("id", ""))),
                    prompt=data.get("prompt", data.get("text", "")),
                    canonical_solution=data.get("canonical_solution", data.get("code", "")),
                    test_clean=data.get("test", data.get("test_clean", "")),
                    test_verifier=data.get("test_verifier", data.get("test", "")),
                    entry_point=data.get("entry_point", "solution"),
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