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
EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "examples" / "flash_gated_delta_rule.py"
GDN_CORE_CPP = (
    Path(__file__).resolve().parents[3]
    / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gdn_core_fwd/op_host/op_api/aclnn_gdn_core_fwd.cpp"
)
GDN_CORE_HEADER = GDN_CORE_CPP.with_suffix(".h")
GDN_CORE_ROOT = GDN_CORE_CPP.parents[1]
GDN_CORE_DEF = GDN_CORE_ROOT / "chunk_gdn_core_fwd_def.cpp"
GDN_CORE_TILING = GDN_CORE_ROOT / "chunk_gdn_core_fwd_tiling.cpp"
GDN_CORE_KERNEL = GDN_CORE_ROOT.parent / "op_kernel/chunk_gdn_core_fwd.cpp"
GDN_CORE_STRUCT = GDN_CORE_ROOT.parent / "op_kernel/chunk_gdn_core_fwd_struct.h"


class FakeTensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = object()

    def contiguous(self):
        return self


class FakeCallContext:
    def __init__(self):
        self.tensor_calls = []
        self.int_array_calls = []

    def tensor(self, tensor, name):
        self.tensor_calls.append((name, tensor))
        if tensor is None:
            return ctypes.c_void_p()
        return ("tensor", name)

    def int_array(self, values):
        self.int_array_calls.append(values)
        return ("int_array", values)


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

    def empty(shape, reference, dtype=None):
        return FakeTensor(shape, reference.dtype if dtype is None else dtype)

    fake_runtime.call_aclnn = call_aclnn
    fake_runtime.ACL_FORMAT_ND = 2
    fake_runtime.chunk_num = lambda *args, **kwargs: 1
    fake_runtime.empty = empty
    fake_runtime.empty_like = lambda tensor, dtype=None: FakeTensor(
        tensor.shape, tensor.dtype if dtype is None else dtype
    )
    fake_runtime.optional_bool = lambda value, default: default if value is None else bool(value)
    fake_runtime.optional_float = lambda value, default: default if value is None else float(value)
    fake_runtime.optional_int = lambda value, default: default if value is None else int(value)
    fake_runtime.shape = lambda tensor: tuple(tensor.shape)
    fake_runtime.zeros = empty

    fake_torch = types.ModuleType("torch")
    fake_torch.dtype = type
    fake_torch.float32 = "float32"
    fake_torch.float16 = "fp16"
    fake_torch.bfloat16 = "bf16"

    modules = {"torch": fake_torch}
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
    patcher = mock.patch.dict(sys.modules, modules)
    patcher.start()
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._test_module_patcher = patcher
    return module


def make_inputs(batch=1, heads=4, tokens=128, value_dim=128, value_heads=None):
    value_heads = heads if value_heads is None else value_heads
    return {
        "q": FakeTensor((batch, heads, tokens, 128)),
        "k": FakeTensor((batch, heads, tokens, 128)),
        "v": FakeTensor((batch, value_heads, tokens, value_dim)),
        "g": FakeTensor((batch, tokens, value_heads), "float32"),
        "beta": FakeTensor((batch, tokens, value_heads), "float32"),
    }


class GdnCoreFwdCtypesAbiTest(unittest.TestCase):
    def test_example_uses_composite_core_by_default(self):
        source = EXAMPLE_PATH.read_text(encoding="utf-8")
        import ast

        self.assertIn("gdn_core_fwd_phase6 as ascendc_gdn_core_fwd", source)

        module = ast.parse(source)
        defaults = {}
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "flash_chunk_gated_delta_rule_fwd",
                "flash_gated_delta_rule",
            }:
                positional = [arg.arg for arg in node.args.args]
                positional_defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
                defaults[node.name] = dict(zip(positional, positional_defaults))["use_composite_core"]
        self.assertEqual(set(defaults), {"flash_chunk_gated_delta_rule_fwd", "flash_gated_delta_rule"})
        for default in defaults.values():
            self.assertIsInstance(default, ast.Constant)
            self.assertIs(default.value, True)

    def test_dense_call_matches_public_aclnn_signature(self):
        captured = {}
        module = load_ctypes_module(captured)
        outputs = module.npu_gdn_core_fwd_phase6(**make_inputs(), chunk_size=64, scale=0.125)

        self.assertIs(outputs, captured["outputs"])
        self.assertEqual(captured["name"], "aclnnGdnCoreFwdPhase6")
        self.assertEqual(len(captured["args"]), 15)
        self.assertEqual(captured["ctx"].int_array_calls, [None, None])
        self.assertEqual(
            [name for name, _ in captured["ctx"].tensor_calls],
            ["q", "k", "v", "g", "beta", "initial_state", "o", "final_state", "g_cumsum", "A"],
        )
        self.assertEqual(outputs[0].shape, (1, 4, 128, 128))
        self.assertIsNone(outputs[1])
        self.assertEqual(outputs[2].shape, (1, 128, 4))
        self.assertEqual(outputs[3].shape, (1, 4, 128, 64))

    def test_final_state_and_aux_output_matrix_keeps_fixed_slots(self):
        for output_final_state in (False, True):
            for return_aux in (False, True):
                with self.subTest(
                    output_final_state=output_final_state,
                    return_aux=return_aux,
                ):
                    captured = {}
                    module = load_ctypes_module(captured)

                    output, final_state, g_cumsum, A = module.npu_gdn_core_fwd_phase6(
                        **make_inputs(),
                        output_final_state=output_final_state,
                        return_aux=return_aux,
                    )

                    self.assertEqual(output.shape, (1, 4, 128, 128))
                    self.assertEqual(final_state is not None, output_final_state)
                    self.assertEqual(g_cumsum is not None, return_aux)
                    self.assertEqual(A is not None, return_aux)
                    self.assertEqual(len(captured["args"]), 15)
                    self.assertEqual(bool(captured["args"][12]), output_final_state)
                    self.assertEqual(bool(captured["args"][13]), return_aux)
                    self.assertEqual(bool(captured["args"][14]), return_aux)
                    self.assertEqual(
                        captured["outputs"],
                        (output, final_state, g_cumsum, A),
                    )

    def test_return_aux_none_preserves_legacy_default(self):
        captured = {}
        module = load_ctypes_module(captured)

        outputs = module.npu_gdn_core_fwd_phase6(
            **make_inputs(batch=1, heads=1, tokens=1),
            return_aux=None,
        )

        self.assertEqual(outputs[2].shape, (1, 1, 1))
        self.assertEqual(outputs[3].shape, (1, 1, 1, 64))
        self.assertIsNotNone(captured["ctx"].tensor_calls[-2][1])
        self.assertIsNotNone(captured["ctx"].tensor_calls[-1][1])

    def test_varlen_final_state_uses_initial_state_dtype(self):
        captured = {}
        module = load_ctypes_module(captured)
        initial_state = FakeTensor((2, 4, 128, 128), "fp16")
        output, final_state, g_cumsum, A = module.npu_gdn_core_fwd_phase6(
            **make_inputs(tokens=130),
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=64,
            cu_seqlens=[0, 65, 130],
            chunk_indices=[0, 0, 0, 1, 1, 0, 1, 1],
        )

        self.assertEqual(output.shape, (1, 4, 130, 128))
        self.assertEqual(final_state.shape, (2, 4, 128, 128))
        self.assertEqual(final_state.dtype, "fp16")
        self.assertEqual(g_cumsum.shape, (1, 130, 4))
        self.assertEqual(A.shape, (1, 4, 130, 64))
        self.assertEqual(
            captured["ctx"].int_array_calls,
            [(0, 65, 130), (0, 0, 0, 1, 1, 0, 1, 1)],
        )

    def test_rejects_noncanonical_varlen_chunk_order(self):
        module = load_ctypes_module({})
        with self.assertRaisesRegex(RuntimeError, "canonical sequence-major order"):
            module.npu_gdn_core_fwd_phase6(
                **make_inputs(tokens=130),
                chunk_size=64,
                cu_seqlens=[0, 65, 130],
                chunk_indices=[0, 0, 1, 0, 0, 1, 1, 1],
            )

    def test_phase6_accepts_v256(self):
        captured = {}
        module = load_ctypes_module(captured)
        inputs = make_inputs(value_dim=256)

        output, final_state, g_cumsum, A = module.npu_gdn_core_fwd_phase6(**inputs)

        self.assertEqual(captured["name"], "aclnnGdnCoreFwdPhase6")
        self.assertEqual(output.shape, (1, 4, 128, 256))
        self.assertIsNone(final_state)
        self.assertEqual(g_cumsum.shape, (1, 128, 4))
        self.assertEqual(A.shape, (1, 4, 128, 64))
    def test_phase6_accepts_native_gva_and_arbitrary_dense_tail(self):
        captured = {}
        module = load_ctypes_module(captured)
        inputs = make_inputs(heads=2, value_heads=8, tokens=130, value_dim=128)

        output, final_state, g_cumsum, A = module.npu_gdn_core_fwd_phase6(
            **inputs, chunk_size=64
        )

        self.assertEqual(captured["name"], "aclnnGdnCoreFwdPhase6")
        self.assertEqual(output.shape, (1, 8, 130, 128))
        self.assertIsNone(final_state)
        self.assertEqual(g_cumsum.shape, (1, 130, 8))
        self.assertEqual(A.shape, (1, 8, 130, 64))

    def test_phase6_rejects_non_divisible_gva(self):
        module = load_ctypes_module({})
        with self.assertRaisesRegex(RuntimeError, "value heads divisible by key heads"):
            module.npu_gdn_core_fwd_phase6(
                **make_inputs(heads=3, value_heads=8), chunk_size=64
            )

    def test_phase6_cpp_param_contract_allows_v256(self):
        source = GDN_CORE_CPP.read_text(encoding="utf-8")
        self.assertIn("CheckParams(const GdnCoreFwdParams &params)", source)
        self.assertIn("vDim == GDN_CORE_V256", source)
        self.assertNotIn("GdnCorePhase", source)

    def test_get_workspace_signature_matches_aclnn_header(self):
        module = load_ctypes_module({})
        expected = [
            *([ctypes.c_void_p] * 6),
            ctypes.c_bool,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            *([ctypes.c_void_p] * 4),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(module._GET_WORKSPACE_ARGTYPES["aclnnGdnCoreFwdPhase6"], expected)

    def test_cpp_null_aux_contract_keeps_required_l0_placeholders(self):
        source = GDN_CORE_CPP.read_text(encoding="utf-8")
        op_def = GDN_CORE_DEF.read_text(encoding="utf-8")

        self.assertIn("if (params.gCumsumOut != nullptr)", source)
        self.assertIn("if (params.aOut != nullptr)", source)
        self.assertRegex(
            source,
            r"gCumsumOutput\s*=\s*executorPtr->AllocTensor\(MakeShape\(\{1\}\),\s*"
            r"DataType::DT_FLOAT",
        )
        self.assertRegex(
            source,
            r"aOutput\s*=\s*executorPtr->AllocTensor\(MakeShape\(\{1\}\),\s*"
            r"params\.k->GetDataType\(\)",
        )
        self.assertIn('Output("g_cumsum_bth").ParamType(REQUIRED)', op_def)
        self.assertIn('Output("A").ParamType(REQUIRED)', op_def)

    def test_tiling_output_mask_is_rank_based_and_singleton_safe(self):
        tiling = GDN_CORE_TILING.read_text(encoding="utf-8")
        struct = GDN_CORE_STRUCT.read_text(encoding="utf-8")

        self.assertIn("gCumsumOutputStorage.GetDimNum() == 3", tiling)
        self.assertIn("aOutputStorage.GetDimNum() == 4", tiling)
        self.assertIn("gCumsumOutputStorage.GetDimNum() == 1", tiling)
        self.assertIn("aOutputStorage.GetDimNum() == 1", tiling)
        self.assertNotIn("GetShapeSize()", tiling)
        self.assertIn("constexpr uint64_t GDN_CORE_OUTPUT_G_CUMSUM = 1ULL << 0", struct)
        self.assertIn("constexpr uint64_t GDN_CORE_OUTPUT_A = 1ULL << 1", struct)
        self.assertIn("uint64_t outputMask", struct)
        self.assertIn("trailer.outputMask", tiling)

    def test_kernel_mask_retains_internal_a_and_cumsum_dependencies(self):
        kernel = GDN_CORE_KERNEL.read_text(encoding="utf-8")

        self.assertIn("? A : aStorage", kernel)
        self.assertGreaterEqual(kernel.count("solveA"), 5)
        self.assertIn("RunPhase6Cumsum(rawG", kernel)
        self.assertIn("gCumsumBht", kernel)
        self.assertEqual(
            kernel.count("(outputMask & GDN_CORE_OUTPUT_G_CUMSUM) != 0"),
            2,
        )

    def test_l0_output_slots_remain_fixed_and_ordered(self):
        l0_source = (GDN_CORE_ROOT / "op_api/chunk_gdn_core_fwd.cpp").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "OP_OUTPUT(oOut, finalStateOut, gCumsumBthOut, aOut)",
            l0_source,
        )
        self.assertIn(
            "return {oOut, finalStateOut, gCumsumBthOut, aOut};",
            l0_source,
        )

    def test_phase6_wrapper_uses_fixed_aclnn_symbol(self):
        captured = {}
        module = load_ctypes_module(captured)
        module.npu_gdn_core_fwd_phase6(**make_inputs(), chunk_size=64)
        self.assertEqual(captured["name"], "aclnnGdnCoreFwdPhase6")
        self.assertEqual(len(captured["args"]), 15)

    def test_cpp_route_keeps_only_phase6_boundary(self):
        source = GDN_CORE_CPP.read_text(encoding="utf-8")
        header = GDN_CORE_HEADER.read_text(encoding="utf-8")
        self.assertIn("l0op::ChunkGdnCoreFwd", source)
        self.assertIn("aclnnGdnCoreFwdPhase6GetWorkspaceSize", source)
        self.assertIn("aclnnGdnCoreFwdPhase6GetWorkspaceSize", header)
        self.assertIn("aclnnGdnCoreFwdPhase6(", header)
        for phase in range(1, 6):
            self.assertNotIn(f"aclnnGdnCoreFwdPhase{phase}", source)
            self.assertNotIn(f"aclnnGdnCoreFwdPhase{phase}", header)
        self.assertNotIn("aclnnGdnCoreFwdGetWorkspaceSize", source)
        self.assertNotIn("aclnnGdnCoreFwdGetWorkspaceSize", header)

    def test_preprocess_direct_wrappers_match_aclnn_descriptor_kinds(self):
        captured = {}
        module = load_ctypes_module(captured)
        inputs = make_inputs()
        cu_seqlens = [0, 64, 128]
        chunk_indices = [0, 0, 1, 0]

        cumsum = module.npu_chunk_local_cumsum(
            inputs["g"].__class__((1, 4, 128), "float32"),
            64,
            cu_seqlens=cu_seqlens,
            chunk_indices_out=chunk_indices,
        )
        self.assertEqual(captured["name"], "aclnnChunkLocalCumsum")
        self.assertEqual(cumsum.shape, (1, 4, 128))
        self.assertEqual(captured["ctx"].int_array_calls, [cu_seqlens, chunk_indices])

        module.npu_chunk_scaled_dot_kkt(
            inputs["k"],
            FakeTensor((1, 4, 128), "float32"),
            FakeTensor((1, 4, 128), "float32"),
            cu_seqlens=[0, 64, 128],
            chunk_indices=[0, 0, 1, 0],
            chunk_size=64,
        )
        self.assertEqual(captured["name"], "aclnnChunkScaledDotKkt")
        self.assertEqual(captured["ctx"].int_array_calls, [[0, 64, 128], [0, 0, 1, 0]])


if __name__ == "__main__":
    unittest.main()
