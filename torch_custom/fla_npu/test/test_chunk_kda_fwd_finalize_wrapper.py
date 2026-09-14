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


def load_wrapper_module():
    package_name = "fla_npu_test_kda_finalize"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ASCENDC_DIR)]
    sys.modules[package_name] = package
    for module_name in ("_runtime", "_kda_policy", "_aclnn_ctypes"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name, ASCENDC_DIR / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}._aclnn_ctypes"]


class FakeTensor:
    def __init__(self, shape, dtype, device, *, contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def storage_offset(self):
        return 0


class FakeContext:
    def __init__(self):
        self.tensors = []
        self.int_arrays = []

    def tensor(self, tensor, name, *, acl_format_override=None, storage_shape_override=None):
        self.tensors.append((name, tensor, acl_format_override, storage_shape_override))
        return ctypes.c_void_p(0x1000 + len(self.tensors))

    def int_array(self, values):
        self.int_arrays.append(values)
        return ctypes.c_void_p(0x2000 + len(self.int_arrays))


class ChunkKdaFwdFinalizeWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = load_wrapper_module()

    def setUp(self):
        self.fake_torch = types.ModuleType("torch")
        self.fake_torch.bfloat16 = object()
        self.fake_torch.float32 = object()
        self.device = types.SimpleNamespace(type="npu", index=0)
        self.captured = {}

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype), like.device)

        def fake_call_aclnn(name, build_args, output):
            context = FakeContext()
            self.captured.update(name=name, args=build_args(context), context=context)
            return output

        self.patches = [
            mock.patch.dict(sys.modules, {"torch": self.fake_torch, "torch_npu": None}),
            mock.patch.object(self.wrapper, "_empty", side_effect=fake_empty),
            mock.patch.object(self.wrapper, "_call_aclnn", side_effect=fake_call_aclnn),
            mock.patch.object(self.wrapper, "_acl_format", return_value=self.wrapper.ACL_FORMAT_ND),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def inputs(self, *, packed=False, chunks=2, v_new_rank4=False):
        bf16 = self.fake_torch.bfloat16
        tensor = lambda shape: FakeTensor(shape, bf16, self.device)
        if packed:
            q = tensor((4, 70, 128))
            a = tensor((4, 70, 64))
            v = tensor((1, 4, 70, 128) if v_new_rank4 else (4, 70, 128))
        else:
            q = tensor((2, 4, 70, 128))
            a = tensor((2, 4, 70, 64))
            v = tensor((2, 4, 70, 128))
        h = tensor((1 if packed else 2, 4, chunks, 128, 128))
        return q, a, v, h

    def test_aclnn_abi_order_and_all_four_output_layouts(self):
        self.assertEqual(
            self.wrapper._GET_WORKSPACE_ARGTYPES["aclnnChunkKdaFwdFinalize"],
            [*([ctypes.c_void_p] * 6), ctypes.c_char_p, ctypes.c_bool,
             ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
             ctypes.POINTER(ctypes.c_void_p)],
        )
        for layout, expected_shape in (
            ("BSND", (2, 70, 4, 128)),
            ("BNSD", (2, 4, 70, 128)),
            ("TND", (70, 4, 128)),
            ("NTD", (4, 70, 128)),
        ):
            with self.subTest(layout=layout):
                args = self.inputs(packed=layout in {"TND", "NTD"}, v_new_rank4=True)
                output = self.wrapper.npu_chunk_kda_fwd_finalize(
                    *args, output_layout=layout, state_v_first=True
                )
                self.assertEqual(output.shape, expected_shape)
                self.assertEqual(self.captured["name"], "aclnnChunkKdaFwdFinalize")
                self.assertEqual(len(self.captured["args"]), 9)
                self.assertEqual(self.captured["args"][6].value, layout.encode())
                self.assertTrue(self.captured["args"][7].value)
                context = self.captured["context"]
                self.assertEqual(context.int_arrays, [None, None])
                self.assertEqual([entry[0] for entry in context.tensors],
                                 ["qg_scaled", "aqk", "v_new", "h", "attn_out"])
                self.assertTrue(all(entry[2] == self.wrapper.ACL_FORMAT_ND
                                    and entry[3] == entry[1].shape for entry in context.tensors))

    def test_varlen_generates_sequence_major_indices_and_accepts_fwd_h_v_new(self):
        bf16 = self.fake_torch.bfloat16
        tensor = lambda shape: FakeTensor(shape, bf16, self.device)
        inputs = (
            tensor((4, 130, 128)), tensor((4, 130, 64)),
            tensor((1, 4, 130, 128)), tensor((1, 4, 4, 128, 128)),
        )
        self.wrapper.npu_chunk_kda_fwd_finalize(
            *inputs, output_layout="TND", cu_seqlens=(0, 65, 130)
        )
        self.assertEqual(self.captured["context"].int_arrays,
                         [(0, 65, 130), (0, 0, 0, 1, 1, 0, 1, 1)])

    def test_public_export_is_ctypes_only(self):
        package_name = "fla_npu_test_kda_finalize"
        spec = importlib.util.spec_from_file_location(
            package_name, ASCENDC_DIR / "__init__.py",
            submodule_search_locations=[str(ASCENDC_DIR)],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
        self.assertIn("chunk_kda_fwd_finalize", package.__all__)
        self.assertIn("npu_chunk_kda_fwd_finalize", package._ASCENDC_OPS)
        self.assertNotIn("npu_chunk_kda_fwd_finalize", package._TORCH_NPU_COMPAT_OPS)
        self.assertIs(package.ASCENDC_CTYPES_OPS["npu_chunk_kda_fwd_finalize"],
                      self.wrapper.npu_chunk_kda_fwd_finalize)

    def test_invalid_shape_layout_dtype_and_metadata_stop_before_launch(self):
        inputs = self.inputs(packed=True)
        cases = (
            (dict(output_layout="tnd"), "output_layout"),
            (dict(output_layout="TND", cu_seqlens=(0, 70, 70)), "cu_seqlens"),
            (dict(output_layout="TND", chunk_indices=(0, 0)), "chunk_indices"),
            (dict(output_layout="TND", cu_seqlens=(0, 70), chunk_indices=(1, 0)),
             "chunk_indices"),
        )
        for options, message in cases:
            with self.subTest(options=options):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.wrapper.npu_chunk_kda_fwd_finalize(*inputs, **options)
                self.assertEqual(self.captured, {})
        inputs[1].dtype = self.fake_torch.float32
        with self.assertRaisesRegex(RuntimeError, "aqk must use bfloat16"):
            self.wrapper.npu_chunk_kda_fwd_finalize(*inputs, output_layout="TND")
        self.assertEqual(self.captured, {})

    def test_private_format_rejected_before_aclnn(self):
        inputs = self.inputs()
        with mock.patch.object(self.wrapper, "_acl_format", return_value=29):
            with self.assertRaisesRegex(RuntimeError, "private NPU format"):
                self.wrapper.npu_chunk_kda_fwd_finalize(*inputs)
        self.assertEqual(self.captured, {})


if __name__ == "__main__":
    unittest.main()
