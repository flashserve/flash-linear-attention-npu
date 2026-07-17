"""Example-tagged ST entry for chunk_gated_delta_rule_fwd_h."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.operators.chunk_gated_delta_rule_fwd_h.common.case_matrix import case_ids


RUNNER = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/pta/test_fwd_h.py"


@pytest.mark.npu
def test_example_cases():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    ids = case_ids(tag="example", route="ascendc")
    assert ids and RUNNER.is_file()
    env = os.environ.copy()
    env.update({
        "FLA_NPU_OPERATOR": "chunk_gated_delta_rule_fwd_h",
        "FLA_NPU_CASE_MANIFEST": str(ROOT / "tests/op_cases/chunk_gated_delta_rule_fwd_h.json"),
        "FLA_NPU_CASE_IDS": ",".join(ids),
    })
    subprocess.run([sys.executable, str(RUNNER)], cwd=RUNNER.parent, env=env, check=True)
