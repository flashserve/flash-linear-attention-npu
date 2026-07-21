"""Example-tagged ST entry for kda_layout_swap12."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.operators.kda_layout_swap12.common.case_matrix import case_ids


RUNNER = ROOT / "tests/operators/_shared/chunk_kda_backend.py"


@pytest.mark.npu
def test_example_cases():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    ids = case_ids(tag="example", route="ascendc")
    assert ids and RUNNER.is_file()
    env = os.environ.copy()
    env.update({
        "FLA_NPU_OPERATOR": "kda_layout_swap12",
        "FLA_NPU_CASE_MANIFEST": str(ROOT / "tests/op_cases/kda_layout_swap12.json"),
        "FLA_NPU_CASE_IDS": ",".join(ids),
    })
    subprocess.run([sys.executable, str(RUNNER)], cwd=RUNNER.parent, env=env, check=True)
