"""通过稳定 Python 入口真实执行 ctypes/aclnn 两段式调用。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from case_matrix import load_manifest, select_cases  # noqa: E402
from runtime import (  # noqa: E402
    OUTPUT_NAMES,
    check_output_contract,
    require_npu_test_environment,
    run_stable_aclnn_case,
)


def test_manifest_freezes_public_route_abi():
    manifest = load_manifest()
    coverage = manifest["coverage_requirements"]
    assert len(coverage["inputs"]) == 9
    assert len(coverage["attrs"]) == 11
    assert tuple(coverage["outputs"]) == OUTPUT_NAMES
    assert manifest["capability"]["dtypes"]["q_k_v"] == ["bfloat16"]
    assert set(manifest["capability"]["layouts"]) == {
        "BNSD",
        "BSND",
        "NTD",
        "TND",
    }


@pytest.mark.npu
def test_stable_ctypes_aclnn_route():
    torch, device = require_npu_test_environment()
    if torch is None:
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")

    cases = select_cases(route="aclnn", tag="route")
    assert cases, "no aclnn route case is selected"
    for case in cases:
        outputs = run_stable_aclnn_case(torch, device, case)
        check_output_contract(torch, case, outputs)
