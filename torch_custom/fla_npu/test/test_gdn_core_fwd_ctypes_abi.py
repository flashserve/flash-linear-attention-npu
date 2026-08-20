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
    / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/op_api/aclnn_gdn_core_fwd.cpp"
)
CHUNK_CUMSUM_KKT_HEADER = (
    Path(__file__).resolve().parents[3]
    / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/op_host/op_api/aclnn_chunk_cumsum_kkt.h"
)
NPU_CUSTOM_YAML = Path(__file__).resolve().parents[1] / "npu_custom.yaml"
ASCENDC_INIT = ASCENDC_DIR / "__init__.py"


class FakeTensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = object()


class FakeCallContext:
    def __init__(self):
        self.tensor_calls = []
        self.int_array_calls = []

    def tensor(self, tensor, name):
        self.tensor_calls.append((name, tensor))
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
        outputs = module.npu_gdn_core_fwd(**make_inputs(), chunk_size=64, scale=0.125)

        self.assertIs(outputs, captured["outputs"])
        self.assertEqual(captured["name"], "aclnnGdnCoreFwd")
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

    def test_varlen_final_state_uses_initial_state_dtype(self):
        captured = {}
        module = load_ctypes_module(captured)
        initial_state = FakeTensor((2, 4, 128, 128), "fp16")
        output, final_state, g_cumsum, A = module.npu_gdn_core_fwd(
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
            module.npu_gdn_core_fwd(
                **make_inputs(tokens=130),
                chunk_size=64,
                cu_seqlens=[0, 65, 130],
                chunk_indices=[0, 0, 1, 0, 0, 1, 1, 1],
            )

    def test_phase6_accepts_v256_without_extending_earlier_checkpoints(self):
        captured = {}
        module = load_ctypes_module(captured)
        inputs = make_inputs(value_dim=256)

        output, final_state, g_cumsum, A = module.npu_gdn_core_fwd_phase6(**inputs)

        self.assertEqual(captured["name"], "aclnnGdnCoreFwdPhase6")
        self.assertEqual(output.shape, (1, 4, 128, 256))
        self.assertIsNone(final_state)
        self.assertEqual(g_cumsum.shape, (1, 128, 4))
        self.assertEqual(A.shape, (1, 4, 128, 64))
        with self.assertRaisesRegex(RuntimeError, "requires V=128"):
            module.npu_gdn_core_fwd_phase5(**inputs)

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

    def test_phase6_cpp_param_contract_allows_v256_only_for_phase6(self):
        source = GDN_CORE_CPP.read_text(encoding="utf-8")
        self.assertIn("CheckParams(const GdnCoreFwdParams &params, GdnCorePhase phase)", source)
        self.assertIn("phase == GdnCorePhase::PHASE_6_FUSED_CORE", source)
        self.assertIn("vDim == GDN_CORE_V256", source)
        self.assertIn("earlier checkpoints require V=128", source)

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
        self.assertEqual(module._GET_WORKSPACE_ARGTYPES["aclnnGdnCoreFwd"], expected)
        self.assertEqual(module._GET_WORKSPACE_ARGTYPES["aclnnGdnCoreFwdPhase1"], expected)
        self.assertEqual(module._GET_WORKSPACE_ARGTYPES["aclnnGdnCoreFwdPhase2"], expected)
        self.assertEqual(module._GET_WORKSPACE_ARGTYPES["aclnnGdnCoreFwdPhase3"], expected)
        self.assertEqual(module._GET_WORKSPACE_ARGTYPES["aclnnGdnCoreFwdPhase4"], expected)
        self.assertEqual(module._GET_WORKSPACE_ARGTYPES["aclnnGdnCoreFwdPhase5"], expected)
        self.assertEqual(module._GET_WORKSPACE_ARGTYPES["aclnnGdnCoreFwdPhase6"], expected)

    def test_versioned_phase_wrappers_use_fixed_aclnn_symbols(self):
        for wrapper_name, aclnn_name in (
            ("npu_gdn_core_fwd_phase1", "aclnnGdnCoreFwdPhase1"),
            ("npu_gdn_core_fwd_phase2", "aclnnGdnCoreFwdPhase2"),
            ("npu_gdn_core_fwd_phase3", "aclnnGdnCoreFwdPhase3"),
            ("npu_gdn_core_fwd_phase4", "aclnnGdnCoreFwdPhase4"),
            ("npu_gdn_core_fwd_phase5", "aclnnGdnCoreFwdPhase5"),
            ("npu_gdn_core_fwd_phase6", "aclnnGdnCoreFwdPhase6"),
        ):
            with self.subTest(wrapper=wrapper_name):
                captured = {}
                module = load_ctypes_module(captured)
                getattr(module, wrapper_name)(**make_inputs(), chunk_size=64)
                self.assertEqual(captured["name"], aclnn_name)
                self.assertEqual(len(captured["args"]), 15)

    def test_versioned_cpp_routes_keep_both_phase_boundaries(self):
        source = GDN_CORE_CPP.read_text(encoding="utf-8")
        self.assertIn("GdnCorePhase::PHASE_1_SIX_KERNELS", source)
        self.assertIn("l0op::ChunkScaledDotKkt", source)
        self.assertIn("l0op::SolveTri", source)
        self.assertIn("GdnCorePhase::PHASE_2_FUSED_KKT_SOLVE", source)
        self.assertIn("l0op::ChunkKktSolveTri", source)
        self.assertIn("GdnCorePhase::PHASE_3_FUSED_CUMSUM_KKT_SOLVE", source)
        self.assertIn("l0op::ChunkCumsumKktSolveTri", source)
        self.assertIn("GdnCorePhase::PHASE_4_FUSED_FWD_HO", source)
        self.assertIn("l0op::ChunkGatedDeltaRuleFwdHO", source)
        self.assertIn("GdnCorePhase::PHASE_5_FUSED_RECOMPUTE_WU_HO", source)
        self.assertIn("l0op::ChunkRecomputeWUFwdHO", source)
        self.assertIn("GdnCorePhase::PHASE_6_FUSED_CORE", source)
        self.assertIn("l0op::ChunkGdnCoreFwd", source)
        self.assertNotIn("l0op::ChunkCumsumKkt(", source)
        self.assertIn("aclnnGdnCoreFwdPhase3GetWorkspaceSize", source)
        self.assertIn("aclnnGdnCoreFwdPhase4GetWorkspaceSize", source)
        self.assertIn("aclnnGdnCoreFwdPhase5GetWorkspaceSize", source)
        self.assertIn("aclnnGdnCoreFwdPhase6GetWorkspaceSize", source)

    def test_preprocess_direct_wrappers_match_aclnn_descriptor_kinds(self):
        captured = {}
        module = load_ctypes_module(captured)
        inputs = make_inputs()
        cu_seqlens = FakeTensor((3,), "int64")
        chunk_indices = FakeTensor((4, 2), "int64")

        cumsum = module.npu_chunk_local_cumsum(
            inputs["g"].__class__((1, 4, 128), "float32"),
            64,
            cu_seqlens=cu_seqlens,
            chunk_indices_out=chunk_indices,
        )
        self.assertEqual(captured["name"], "aclnnChunkLocalCumsum")
        self.assertEqual(cumsum.shape, (1, 4, 128))
        self.assertEqual(captured["ctx"].int_array_calls, [])

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

        g_cumsum, A = module.npu_chunk_cumsum_kkt(
            inputs["k"],
            FakeTensor((1, 4, 128), "float32"),
            FakeTensor((1, 4, 128), "float32"),
            cu_seqlens=[0, 64, 128],
            chunk_indices=[0, 0, 1, 0],
            chunk_size=64,
        )
        self.assertEqual(captured["name"], "aclnnChunkCumsumKkt")
        self.assertEqual(g_cumsum.shape, (1, 4, 128))
        self.assertEqual(g_cumsum.dtype, "float32")
        self.assertEqual(A.shape, (1, 4, 128, 64))
        self.assertEqual(A.dtype, "float32")
        self.assertEqual(captured["ctx"].int_array_calls, [(0, 64, 128), (0, 0, 1, 0)])
        self.assertEqual(len(captured["args"]), 8)
        self.assertEqual(
            module._GET_WORKSPACE_ARGTYPES["aclnnChunkCumsumKkt"],
            [
                *([ctypes.c_void_p] * 5),
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )

    def test_chunk_cumsum_kkt_public_surfaces_are_declared(self):
        header = CHUNK_CUMSUM_KKT_HEADER.read_text(encoding="utf-8")
        schema = NPU_CUSTOM_YAML.read_text(encoding="utf-8")
        package_init = ASCENDC_INIT.read_text(encoding="utf-8")
        self.assertIn("aclnnChunkCumsumKktGetWorkspaceSize", header)
        self.assertIn("aclnnChunkCumsumKkt(", header)
        self.assertIn("npu_chunk_cumsum_kkt(Tensor k, Tensor g, Tensor beta", schema)
        self.assertIn('"npu_chunk_cumsum_kkt"', package_init)

    def test_fused_kkt_solve_wrapper_matches_public_aclnn_signature(self):
        captured = {}
        module = load_ctypes_module(captured)
        k = FakeTensor((1, 4, 130, 128), "bf16")
        g = FakeTensor((1, 4, 130), "float32")
        beta = FakeTensor((1, 4, 130), "float32")
        output = module.npu_chunk_kkt_solve_tri(
            k,
            g,
            beta,
            cu_seqlens=[0, 65, 130],
            chunk_indices=[0, 0, 0, 1, 1, 0, 1, 1],
            chunk_size=64,
        )

        self.assertEqual(captured["name"], "aclnnChunkKktSolveTri")
        self.assertEqual(output.shape, (1, 4, 130, 64))
        self.assertEqual(
            captured["ctx"].int_array_calls,
            [(0, 65, 130), (0, 0, 0, 1, 1, 0, 1, 1)],
        )
        self.assertEqual(len(captured["args"]), 7)
        self.assertEqual(
            module._GET_WORKSPACE_ARGTYPES["aclnnChunkKktSolveTri"],
            module._GET_WORKSPACE_ARGTYPES["aclnnChunkScaledDotKkt"],
        )


if __name__ == "__main__":
    unittest.main()
