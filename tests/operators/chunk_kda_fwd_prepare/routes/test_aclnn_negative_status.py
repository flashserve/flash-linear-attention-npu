"""直接验证 ChunkKdaFwdPrepare C++ L2 的公开错误码。"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from unittest import mock

import pytest


COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from case_matrix import load_manifest  # noqa: E402
from runtime import build_inputs, require_npu_test_environment  # noqa: E402


_STATUS = {
    "ACLNN_ERR_PARAM_NULLPTR": 161001,
    "ACLNN_ERR_PARAM_INVALID": 161002,
}
_ARGUMENT_INDEX = {
    "q": 0,
    "scale": 10,
    "epsilon": 12,
}


def _raw_value(argument: str, value):
    if argument == "q":
        return ctypes.c_void_p()
    return ctypes.c_double(float(value))


@pytest.mark.npu
def test_get_workspace_reports_public_error_codes():
    torch, device = require_npu_test_environment()
    if torch is None:
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")

    from fla_npu.ops.ascendc import _aclnn_ctypes as prepare_api
    from fla_npu.ops.ascendc import _runtime as runtime_api

    manifest = load_manifest()
    case = next(
        item
        for item in manifest["cases"]
        if item["id"] == "prepare_dense_bnsd_raw"
    )
    inputs = build_inputs(torch, device, case)
    attrs = case["attrs"]
    optional = case["optional_inputs"]
    captured = {}

    def capture_call(name, build_args, outputs):
        captured["name"] = name
        captured["build_args"] = build_args
        captured["outputs"] = outputs
        return outputs

    with mock.patch.object(prepare_api, "_call_aclnn", new=capture_call):
        prepare_api.npu_chunk_kda_fwd_prepare(
            *inputs[:5],
            float(attrs["scale"]),
            layout=str(attrs["layout"]),
            chunk_size=int(attrs["chunk_size"]),
            epsilon=float(attrs["epsilon"]),
            use_qk_l2norm_in_kernel=bool(attrs["use_qk_l2norm_in_kernel"]),
            use_gate_in_kernel=bool(attrs["use_gate_in_kernel"]),
            use_beta_sigmoid_in_kernel=bool(
                attrs["use_beta_sigmoid_in_kernel"]
            ),
            allow_neg_eigval=bool(attrs["allow_neg_eigval"]),
            safe_gate=bool(attrs["safe_gate"]),
            lower_bound=float(attrs["lower_bound"]),
            use_exp2=bool(attrs["use_exp2"]),
            a_log=inputs[5],
            dt_bias=inputs[6],
            cu_seqlens=optional["cu_seqlens"],
            chunk_indices=optional["chunk_indices"],
        )

    runtime = runtime_api.runtime()
    get_workspace = runtime.symbol(
        "aclnnChunkKdaFwdPrepareGetWorkspaceSize"
    )
    get_workspace.argtypes = prepare_api._GET_WORKSPACE_ARGTYPES[
        "aclnnChunkKdaFwdPrepare"
    ]
    get_workspace.restype = ctypes.c_int

    def invoke(case_spec):
        mutation = case_spec["raw_aclnn"]
        argument = mutation["argument"]
        context = runtime_api._CallContext(runtime, device)
        try:
            with runtime_api._npu_device_guard(device):
                arguments = captured["build_args"](context)
                arguments[_ARGUMENT_INDEX[argument]] = _raw_value(
                    argument, mutation["value"]
                )
                workspace_size = ctypes.c_uint64(0)
                executor = ctypes.c_void_p()
                return int(
                    get_workspace(
                        *arguments,
                        ctypes.byref(workspace_size),
                        ctypes.byref(executor),
                    )
                )
        finally:
            context.destroy()

    raw_cases = [
        item
        for item in manifest["negative_cases"]
        if "raw_aclnn" in item
    ]
    assert {item["id"] for item in raw_cases} == {
        "reject_null_q_aclnn",
        "reject_scale_overflows_fp32",
        "reject_epsilon_underflows_fp32",
    }
    for case_spec in raw_cases:
        expected = _STATUS[case_spec["expect"]["return_code"]]
        assert invoke(case_spec) == expected, case_spec["id"]
