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
    def __init__(self, shape, dtype=None):
        self.shape = tuple(shape)
        self.dtype = dtype


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

    @staticmethod
    def _chunk_kda_inputs(dtypes):
        q = FakeTensor((1, 64, 1, 16), dtypes.float16)
        return {
            "q": q,
            "k": FakeTensor(q.shape, dtypes.float16),
            "v": FakeTensor(q.shape, dtypes.float16),
            "g": FakeTensor(q.shape, dtypes.float32),
            "beta": FakeTensor((1, 64, 1), dtypes.float32),
        }

    @staticmethod
    def _chunk_kda_rank4_inputs(dtypes, layout):
        if layout == "BSND":
            return {
                "q": FakeTensor((1, 64, 1, 16), dtypes.float16),
                "k": FakeTensor((1, 64, 1, 16), dtypes.float16),
                "v": FakeTensor((1, 64, 2, 32), dtypes.float16),
                "g": FakeTensor((1, 64, 2, 16), dtypes.float32),
                "beta": FakeTensor((1, 64, 2), dtypes.float32),
            }
        return {
            "q": FakeTensor((1, 1, 64, 16), dtypes.float16),
            "k": FakeTensor((1, 1, 64, 16), dtypes.float16),
            "v": FakeTensor((1, 2, 64, 32), dtypes.float16),
            "g": FakeTensor((1, 2, 64, 16), dtypes.float32),
            "beta": FakeTensor((1, 2, 64), dtypes.float32),
        }

    def test_chunk_kda_gate_parameters_accept_independent_fp32_bf16_dtypes(self):
        dtypes = types.SimpleNamespace(
            float16=object(), bfloat16=object(), float32=object()
        )
        inputs = self._chunk_kda_inputs(dtypes)

        def fake_empty(shape, like, dtype=None):
            return FakeTensor(shape, like.dtype if dtype is None else dtype)

        with mock.patch.dict(sys.modules, {"torch": dtypes}):
            with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
                with mock.patch.object(
                    ACLNN_CTYPES, "_call_aclnn", side_effect=lambda _name, _args, outputs: outputs
                ):
                    for a_log_dtype in (dtypes.float32, dtypes.bfloat16):
                        for dt_bias_dtype in (dtypes.float32, dtypes.bfloat16):
                            outputs = ACLNN_CTYPES.npu_chunk_kda_fwd(
                                **inputs,
                                scale=0.25,
                                use_gate_in_kernel=True,
                                A_log=FakeTensor((1,), a_log_dtype),
                                dt_bias=FakeTensor((16,), dt_bias_dtype),
                            )
                            self.assertEqual(len(outputs), 12)

    def test_chunk_kda_rejects_non_fp32_bf16_gate_parameter_dtypes(self):
        dtypes = types.SimpleNamespace(
            float16=object(), bfloat16=object(), float32=object()
        )
        inputs = self._chunk_kda_inputs(dtypes)
        invalid_dtype = object()
        with mock.patch.dict(sys.modules, {"torch": dtypes}):
            with self.assertRaisesRegex(RuntimeError, "A_log must be float32 or bfloat16"):
                ACLNN_CTYPES.npu_chunk_kda_fwd(
                    **inputs,
                    scale=0.25,
                    use_gate_in_kernel=True,
                    A_log=FakeTensor((1,), invalid_dtype),
                )
            with self.assertRaisesRegex(RuntimeError, "dt_bias must be float32 or bfloat16"):
                ACLNN_CTYPES.npu_chunk_kda_fwd(
                    **inputs,
                    scale=0.25,
                    use_gate_in_kernel=True,
                    A_log=FakeTensor((1,), dtypes.float32),
                    dt_bias=FakeTensor((16,), invalid_dtype),
                )

    def test_chunk_kda_state_dtype_remains_fp32(self):
        dtypes = types.SimpleNamespace(
            float16=object(), bfloat16=object(), float32=object()
        )
        inputs = self._chunk_kda_inputs(dtypes)
        with mock.patch.dict(sys.modules, {"torch": dtypes}):
            with self.assertRaisesRegex(RuntimeError, "initial_state shape/dtype"):
                ACLNN_CTYPES.npu_chunk_kda_fwd(
                    **inputs,
                    scale=0.25,
                    initial_state=FakeTensor((1, 1, 16, 16), dtypes.bfloat16),
                )

    def test_chunk_kda_rank4_varlen_h_abi_across_layout_and_state_order(self):
        dtypes = types.SimpleNamespace(
            float16=object(), bfloat16=object(), float32=object()
        )

        def fake_empty(shape, like, dtype=None):
            return FakeTensor(shape, like.dtype if dtype is None else dtype)

        with mock.patch.dict(sys.modules, {"torch": dtypes}):
            with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
                with mock.patch.object(
                    ACLNN_CTYPES,
                    "_call_aclnn",
                    side_effect=lambda _name, _args, outputs: outputs,
                ):
                    for layout in ("BSND", "BNSD"):
                        inputs = self._chunk_kda_rank4_inputs(dtypes, layout)
                        for state_v_first in (False, True):
                            for return_intermediate_states in (False, True):
                                with self.subTest(
                                    layout=layout,
                                    state_v_first=state_v_first,
                                    return_intermediate_states=return_intermediate_states,
                                ):
                                    outputs = ACLNN_CTYPES.npu_chunk_kda_fwd(
                                        **inputs,
                                        layout=layout,
                                        scale=0.25,
                                        cu_seqlens=[0, 32, 64],
                                        state_v_first=state_v_first,
                                        return_intermediate_states=return_intermediate_states,
                                    )
                                    if not return_intermediate_states:
                                        self.assertIsNone(outputs[10])
                                    else:
                                        expected = (
                                            (2, 2, 32, 16)
                                            if state_v_first
                                            else (2, 2, 16, 32)
                                        )
                                        self.assertEqual(outputs[10].shape, expected)

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
