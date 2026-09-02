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


def load_aclnn_ctypes_module():
    package_name = "fla_npu_test_aclnn_ctypes"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ASCENDC_DIR)]
    sys.modules[package_name] = package

    for module_name in ("_runtime", "_kda_policy", "_aclnn_ctypes"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            ASCENDC_DIR / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}._aclnn_ctypes"]


ACLNN_CTYPES = load_aclnn_ctypes_module()


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeCallContext:
    def __init__(self):
        self.descriptor_names = []

    def tensor(self, tensor, name):
        del tensor
        self.descriptor_names.append(name)
        return ctypes.c_void_p(0x1000 + len(self.descriptor_names))

    def int_array(self, values):
        del values
        self.descriptor_names.append("query_start_loc")
        return ctypes.c_void_p(0x2000)


class AclnnCtypesAbiTest(unittest.TestCase):
    def test_chunk_kda_bwd_wrapper_matches_aclnn_prototype(self):
        fake_fp16 = object()
        fake_bf16 = object()
        fake_fp32 = object()
        fake_torch = types.SimpleNamespace(
            float16=fake_fp16,
            bfloat16=fake_bf16,
            float32=fake_fp32,
        )

        class Tensor:
            def __init__(self, shape, dtype):
                self.shape = tuple(shape)
                self.dtype = dtype

        class Context:
            def tensor(self, tensor, name, **kwargs):
                del tensor, name, kwargs
                return ctypes.c_void_p(0x1000)

            def int_array(self, values):
                del values
                return ctypes.c_void_p(0x2000)

        captured = {}

        def fake_empty(shape, like, **kwargs):
            del like
            return Tensor(shape, kwargs.get("dtype", fake_bf16))

        def fake_empty_like(tensor, **kwargs):
            return Tensor(tensor.shape, kwargs.get("dtype", tensor.dtype))

        def fake_call(name, build_args, outputs):
            captured["name"] = name
            captured["args"] = build_args(Context())
            return outputs

        q = Tensor((1, 2, 128, 128), fake_bf16)
        beta = Tensor((1, 2, 128), fake_bf16)
        matrix = Tensor((1, 2, 128, 64), fake_bf16)
        gk = Tensor((1, 2, 128, 128), fake_fp32)
        h = Tensor((1, 2, 2, 128, 128), fake_bf16)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty), \
                    mock.patch.object(ACLNN_CTYPES, "_empty_like", side_effect=fake_empty_like), \
                    mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call):
                outputs = ACLNN_CTYPES.npu_chunk_kda_bwd(
                    q, q, q, beta, gk, matrix, matrix,
                    q, q, q, q, h, q, 0.125,
                )

        self.assertEqual(captured["name"], "aclnnChunkKdaBwd")
        self.assertEqual(len(outputs), 8)
        self.assertIsNone(outputs[5])
        self.assertEqual(
            len(captured["args"]),
            len(ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnChunkKdaBwd"]) - 2,
        )
        self.assertEqual(
            [type(arg) for arg in captured["args"]],
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnChunkKdaBwd"][:-2],
        )

    def test_recurrent_gated_delta_rule_requires_at_least_one_gate_before_launch(self):
        with mock.patch.object(ACLNN_CTYPES, "_call_aclnn") as call_aclnn:
            with self.assertRaisesRegex(
                RuntimeError,
                r"^npu_recurrent_gated_delta_rule: either g or gk must be provided\.$",
            ):
                ACLNN_CTYPES.npu_recurrent_gated_delta_rule(
                    None,
                    None,
                    None,
                    None,
                    beta=None,
                    actual_seq_lengths=None,
                    ssm_state_indices=None,
                )

        call_aclnn.assert_not_called()

    def test_causal_conv1d_bwd_signature_matches_aclnn_prototype(self):
        expected_argtypes = [
            *([ctypes.c_void_p] * 7),
            ctypes.c_int64,
            ctypes.c_char_p,
            *([ctypes.c_void_p] * 4),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnCausalConv1dBwd"],
            expected_argtypes,
        )

    def test_causal_conv1d_bwd_wrapper_builds_one_value_per_operator_argtype(self):
        captured = {}

        def fake_empty(shape, like, **kwargs):
            del like, kwargs
            return FakeTensor(shape)

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            captured["descriptor_names"] = context.descriptor_names
            return outputs

        x = FakeTensor((2, 17, 80))
        weight = FakeTensor((4, 80))
        dy = FakeTensor((2, 17, 80))
        with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
            with mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
                outputs = ACLNN_CTYPES.npu_causal_conv1d_bwd(
                    x=x,
                    y=None,
                    weight=weight,
                    dy=dy,
                    input_layout="BSH",
                )

        operator_argtypes = ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnCausalConv1dBwd"][:-2]
        self.assertEqual(captured["name"], "aclnnCausalConv1dBwd")
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(captured["args"]), len(operator_argtypes))
        self.assertEqual([type(arg) for arg in captured["args"]], operator_argtypes)
        self.assertEqual(
            captured["descriptor_names"],
            [
                "x",
                "y",
                "weight",
                "dy",
                "initial_state",
                "dht",
                "query_start_loc",
                "dx",
                "dw",
                "db",
                "dh0",
            ],
        )


if __name__ == "__main__":
    unittest.main()
