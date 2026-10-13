"""Validate the Kaggle notebook without network, package changes or model weights."""

import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "hackable-verifiable-x-alignment-drift.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_structure_and_python_syntax():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    tags = []
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["source"], str) and cell["source"].strip()
        assert isinstance(cell["metadata"], dict)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse(cell["source"])
            tags.extend(cell["metadata"]["tags"])
    assert len(tags) == len(set(tags))
    assert tags.index("dependencies") < tags.index("validation")
    assert tags.index("splits") < tags.index("sft")


def test_notebook_trusted_pilot_workflow(tmp_path):
    cells = {
        cell["metadata"]["tags"][0]: cell["source"]
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    }
    namespace = {}
    configuration = cells["configuration"].replace(
        'Path("/kaggle/working/hackrl_notebook")', f"Path({str(tmp_path)!r})"
    )
    exec(compile(configuration, str(NOTEBOOK), "exec"), namespace)
    assert namespace["PACKAGE_SOURCE"] == "offline"
    assert namespace["RUN_SFT"] is False
    assert namespace["MODEL_PATH"] == Path(
        "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1"
    )
    from hackrl.data.split_generator import ProblemSample, SplitGenerator
    from hackrl.sandbox.verifier import SandboxVerifier

    namespace.update(
        ProblemSample=ProblemSample,
        SplitGenerator=SplitGenerator,
        SandboxVerifier=SandboxVerifier,
        RUN_MODEL_SMOKE_TEST=False,
    )
    for tag in ("data", "verifier", "splits", "model", "sft", "summary"):
        exec(compile(cells[tag], f"{NOTEBOOK}:{tag}", "exec"), namespace)
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "pilot"
    assert manifest["count"] == 4
    assert set(manifest["splits"]) == {"0.0", "0.25", "0.5", "1.0"}
    for alpha, path in manifest["splits"].items():
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 4
        assert sum(row["is_corrupted"] for row in rows) == round(4 * float(alpha))
    controls = json.loads((tmp_path / "verifier_controls.json").read_text(encoding="utf-8"))
    assert controls["honest"]["passed"] is True
    assert controls["incorrect"]["passed"] is False
    assert controls["blocked_exit"]["passed"] is False
    assert controls["treatment_exit"]["passed"] is True
    assert not (tmp_path / "training").exists()