# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ASCENDC_DIR = Path(__file__).resolve().parents[1] / "fla_npu" / "ops" / "ascendc"
CTYPES_PATH = ASCENDC_DIR / "_aclnn_ctypes.py"


class FakeTensor:
    device = object()

    def dim(self):
        return 2


class FakeCallContext:
    def __init__(self):
        self.tensor_calls = []
        self.int_array_calls = []
        self.int_tensor_calls = []

    def tensor(self, tensor, name):
        self.tensor_calls.append((name, tensor))
        return ("tensor", name)

    def int_array(self, values):
        self.int_array_calls.append(values)
        return ("int_array", values)

    def int_tensor(self, values, device):
        self.int_tensor_calls.append((values, device))
        raise AssertionError("causal_conv1d index arguments must use aclIntArray descriptors")


def load_ctypes_module(captured):
    fake_runtime = types.ModuleType("fla_npu.ops.ascendc._runtime")
    fake_kda_policy = types.ModuleType("fla_npu.ops.ascendc._kda_policy")
    fake_kda_policy.kda_fwd_optional_output_mask = lambda *args, **kwargs: 0

    def call_aclnn(name, build_args, outputs, *, get_workspace_argtypes=None):
        ctx = FakeCallContext()
        captured.update(
            name=name,
            args=build_args(ctx),
            outputs=outputs,
            argtypes=get_workspace_argtypes,
            ctx=ctx,
        )
        return outputs

    fake_runtime.call_aclnn = call_aclnn
    fake_runtime.ACL_FORMAT_ND = 2
    fake_runtime.chunk_num = lambda *args, **kwargs: 1
    fake_runtime.empty = lambda *args, **kwargs: object()
    fake_runtime.empty_like = lambda *args, **kwargs: object()
    fake_runtime.optional_bool = lambda value, default: default if value is None else bool(value)
    fake_runtime.optional_float = lambda value, default: default if value is None else float(value)
    fake_runtime.optional_int = lambda value, default: default if value is None else int(value)
    fake_runtime.shape = lambda tensor: tuple(tensor.shape)
    fake_runtime.zeros = lambda *args, **kwargs: object()

    modules = {}
    for name in ("fla_npu", "fla_npu.ops", "fla_npu.ops.ascendc"):
        package = types.ModuleType(name)
        package.__path__ = []
        modules[name] = package
    modules["fla_npu.ops.ascendc._runtime"] = fake_runtime
    modules["fla_npu.ops.ascendc._kda_policy"] = fake_kda_policy

    module_name = "fla_npu.ops.ascendc._aclnn_ctypes"
    spec = importlib.util.spec_from_file_location(module_name, CTYPES_PATH)
    module = importlib.util.module_from_spec(spec)
    modules[module_name] = module
    with mock.patch.dict(sys.modules, modules):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class CausalConv1dCtypesAbiTest(unittest.TestCase):
    def test_forward_index_arguments_use_acl_int_array_descriptors(self):
        captured = {}
        module = load_ctypes_module(captured)
        x = FakeTensor()
        weight = object()
        bias = object()
        conv_states = object()
        query_start_loc = [0, 2, 5]
        cache_indices = [0, 1]
        initial_state_mode = [1, 0]
        num_accepted_tokens = [1, 2]

        output = module.npu_causal_conv1d(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=initial_state_mode,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
            head_num=0,
        )

        self.assertIs(output, captured["outputs"])
        self.assertEqual(captured["name"], "aclnnCausalConv1d")
        self.assertEqual(
            captured["ctx"].int_array_calls,
            [query_start_loc, cache_indices, initial_state_mode, num_accepted_tokens],
        )
        self.assertEqual(captured["ctx"].int_tensor_calls, [])
        self.assertEqual(
            [name for name, _ in captured["ctx"].tensor_calls],
            ["x", "weight", "bias", "conv_states", "out"],
        )
        self.assertEqual(len(captured["args"]), 13)
        self.assertEqual(
            [arg.value for arg in captured["args"][8:12]],
            [1, -1, 0, 0],
        )

    def test_get_workspace_signature_matches_aclnn_header(self):
        captured = {}
        module = load_ctypes_module(captured)
        expected = [
            *([ctypes.c_void_p] * 8),
            *([ctypes.c_int64] * 4),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]

        self.assertEqual(module._GET_WORKSPACE_ARGTYPES["aclnnCausalConv1d"], expected)


if __name__ == "__main__":
    unittest.main()
