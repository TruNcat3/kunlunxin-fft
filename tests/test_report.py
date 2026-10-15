"""Host-only golden tests: the committed reports must rebuild offline.

The review contract requires ``pytest -q`` to pass with NO vendor stack, so
these tests import only ``scripts/report.py`` (torch-free, numpy-free) and
compare its output against the committed artifacts byte for byte. A silent
renderer drift then fails CI instead of shipping a hand-edited report.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report  # noqa: E402


def test_report_module_imports_no_torch_no_numpy():
    tree = ast.parse((ROOT / "scripts" / "report.py").read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    forbidden = {"torch", "torch_xmlir", "numpy"}
    assert not {n.split(".")[0] for n in names} & forbidden, names


@pytest.mark.parametrize("name", ["bench_phase3", "bench_block256",
                                  "bench_block512", "bench_block2048"])
def test_markdown_regeneration_is_byte_identical(tmp_path, name):
    src = ROOT / "results" / "p800" / f"{name}.json"
    committed = ROOT / "results" / "p800" / f"{name}.md"
    if not src.exists() or not committed.exists():
        pytest.skip(f"{name} artifacts not present")
    out = tmp_path / f"{name}.md"
    report.render_from_json(str(src), str(out))
    assert out.read_bytes() == committed.read_bytes(), (
        f"regenerated {name}.md differs from the committed report; "
        "the JSON and the markdown have drifted apart"
    )


def test_phase3_long_rows_schema_fields():
    """The section-9 long rows must carry the reviewed attribution fields."""
    import json

    with open(ROOT / "results" / "p800" / "bench_phase3.json") as f:
        data = json.load(f)
    lrows = data.get("rows_long")
    if not lrows:
        pytest.skip("no rows_long in phase-3 payload")
    if data.get("schema_version", 1) < 2:
        pytest.skip("v1 payload; run scripts/regen_schema_v2.py")
    required = {"flow_id", "flow_realization", "processing_unit",
                "local_residence", "boundary_residence"}
    for row in lrows:
        assert required <= set(row), required - set(row)
