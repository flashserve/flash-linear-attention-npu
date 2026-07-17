"""Example-tagged ST entry for prepare_wy_repr_bwd_da."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.operators.prepare_wy_repr_bwd_da.common.case_matrix import case_ids


RUNNER = ROOT / "torch_custom/fla_npu/test/test_npu_prepare_wy_repr_bwd_da.py"


@pytest.mark.npu
def test_example_cases():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    ids = case_ids(tag="example", route="ascendc")
    assert ids and RUNNER.is_file()
    env = os.environ.copy()
    env.update({
        "FLA_NPU_OPERATOR": "prepare_wy_repr_bwd_da",
        "FLA_NPU_CASE_MANIFEST": str(ROOT / "tests/op_cases/prepare_wy_repr_bwd_da.json"),
        "FLA_NPU_CASE_IDS": ",".join(ids),
    })
    subprocess.run([sys.executable, str(RUNNER)], cwd=RUNNER.parent, env=env, check=True)
