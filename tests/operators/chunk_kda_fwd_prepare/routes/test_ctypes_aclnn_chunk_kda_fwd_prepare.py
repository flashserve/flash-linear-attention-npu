"""通过稳定 Python 入口真实执行 ctypes/aclnn 两段式调用。"""

from __future__ import annotations

from copy import deepcopy
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
    assert len(coverage["attrs"]) == 12
    assert tuple(coverage["outputs"]) == OUTPUT_NAMES
    policy = manifest["output_policy"]
    assert tuple(policy["fixed_slots"]) == OUTPUT_NAMES
    assert set(policy["backward_modes"]) == {"none", "recompute", "save"}
    assert set(policy["forward_required"]) == {
        "gk", "aqk", "w", "u", "kg", "qg_scaled",
    }
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


@pytest.mark.npu
def test_output_modes_keep_forward_required_values_identical():
    torch, device = require_npu_test_environment()
    if torch is None:
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")

    source_cases = select_cases(route="aclnn", tag="route")
    assert source_cases, "no aclnn route case is selected"
    base_case = source_cases[0]
    results = {}
    for backward_mode in ("none", "recompute", "save"):
        case = deepcopy(base_case)
        case["attrs"]["backward_mode"] = backward_mode
        results[backward_mode] = run_stable_aclnn_case(torch, device, case)

    for output_index in (0, 1, 3, 4, 6, 7):
        expected = results["save"][output_index]
        for backward_mode in ("none", "recompute"):
            actual = results[backward_mode][output_index]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
