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
    def __init__(self, shape, *, dtype=None, device_type="npu", contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type=device_type)
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


class FakeCallContext:
    def __init__(self):
        self.descriptor_names = []
        self.descriptor_metadata = []

    def tensor(
        self,
        tensor,
        name,
        *,
        acl_format_override=None,
        storage_shape_override=None,
    ):
        self.descriptor_names.append(name)
        self.descriptor_metadata.append(
            (name, tensor, acl_format_override, storage_shape_override)
        )
        return ctypes.c_void_p(0x1000 + len(self.descriptor_names))

    def int_array(self, values):
        del values
        self.descriptor_names.append("query_start_loc")
        return ctypes.c_void_p(0x2000)


class AclnnCtypesAbiTest(unittest.TestCase):
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
    def test_recurrent_gated_delta_rule_signature_matches_aclnn_prototype(self):
        expected_argtypes = [
            *([ctypes.c_void_p] * 10),
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnRecurrentGatedDeltaRule"],
            expected_argtypes,
        )

    def test_recurrent_gated_delta_rule_wrapper_matches_abi_and_uses_nd_descriptors(self):
        captured = {}
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.bfloat16 = object()
        fake_torch.float32 = object()
        fake_torch.int32 = object()

        def tensor(shape, dtype):
            return FakeTensor(shape, dtype=dtype)

        inputs = {
            "query": tensor((5, 4, 128), fake_torch.bfloat16),
            "key": tensor((5, 4, 128), fake_torch.bfloat16),
            "value": tensor((5, 8, 128), fake_torch.bfloat16),
            "state": tensor((5, 8, 128, 128), fake_torch.bfloat16),
            "beta": tensor((5, 8), fake_torch.bfloat16),
            "actual_seq_lengths": tensor((3,), fake_torch.int32),
            "ssm_state_indices": tensor((5,), fake_torch.int32),
            "g": tensor((5, 8), fake_torch.float32),
            "gk": tensor((5, 8, 128), fake_torch.float32),
            "num_accepted_tokens": tensor((2,), fake_torch.int32),
        }

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, dtype=kwargs.get("dtype", like.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            captured["descriptor_names"] = context.descriptor_names
            captured["descriptor_metadata"] = context.descriptor_metadata
            return outputs

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
                with mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
                    output = ACLNN_CTYPES.npu_recurrent_gated_delta_rule(
                        inputs["query"],
                        inputs["key"],
                        inputs["value"],
                        inputs["state"],
                        beta=inputs["beta"],
                        scale=0.125,
                        actual_seq_lengths=inputs["actual_seq_lengths"],
                        ssm_state_indices=inputs["ssm_state_indices"],
                        num_accepted_tokens=inputs["num_accepted_tokens"],
                        g=inputs["g"],
                        gk=inputs["gk"],
                    )

        operator_argtypes = ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES[
            "aclnnRecurrentGatedDeltaRule"
        ][:-2]
        self.assertEqual(captured["name"], "aclnnRecurrentGatedDeltaRule")
        self.assertEqual(tuple(output.shape), tuple(inputs["value"].shape))
        self.assertEqual(len(captured["args"]), len(operator_argtypes))
        self.assertEqual([type(arg) for arg in captured["args"]], operator_argtypes)
        self.assertEqual(
            captured["descriptor_names"],
            [
                "query",
                "key",
                "value",
                "beta",
                "state",
                "actual_seq_lengths",
                "ssm_state_indices",
                "g",
                "gk",
                "num_accepted_tokens",
                "out",
            ],
        )
        for name, tensor_value, format_override, storage_shape in captured[
            "descriptor_metadata"
        ]:
            self.assertEqual(format_override, ACLNN_CTYPES.ACL_FORMAT_ND, name)
            self.assertEqual(storage_shape, tuple(tensor_value.shape), name)

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
