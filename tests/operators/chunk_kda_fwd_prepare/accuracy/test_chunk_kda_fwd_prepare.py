"""按统一 JSON 执行 ChunkKdaFwdPrepare 的稳定入口用例。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from case_matrix import select_cases  # noqa: E402
from runtime import (  # noqa: E402
    check_output_contract,
    require_npu_test_environment,
    run_stable_aclnn_case,
)


@pytest.mark.npu
def test_json_cases_launch_and_match_output_contract():
    torch, device = require_npu_test_environment()
    if torch is None:
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")

    cases = select_cases(route="ascendc", tag="accuracy")
    assert cases, "no stable Ascend C case is selected"
    for case in cases:
        outputs = run_stable_aclnn_case(torch, device, case)
        check_output_contract(torch, case, outputs)
