# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""pre_process_fwd_kernel_merged 的 ctypes 接入层单测（不需要 NPU 设备）。

覆盖：
  1) 子区间窗口（竞品调用形态）：张量 T=512、`cu_seqlens=[40,512]` → `hm[1,HV,K,V+K]`；
  2) 多段窗口：`cu_seqlens=[0,88,188,512]` → `hm[3,...]`，段序与 cu 一致；
  3) 参数契约：`B != 1`、缺 `cu_seqlens`、`g`/`gk` 同缺或同给、`cu` 越界都在启动前抛错；
  4) DPLR 不支持：`bg` / `v` 传非空时与 host 校验同判据（NotImplementedError），
     不会静默按 GDN/KDA 计算；
  5) ABI：`_GET_WORKSPACE_ARGTYPES` 与 aclnn 头文件逐参对应。
"""
from __future__ import annotations

import ctypes
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ASCENDC_DIR = Path(__file__).resolve().parents[1] / "fla_npu" / "ops" / "ascendc"


def load_aclnn_ctypes_module():
    package_name = "fla_npu_test_ppfm_ctypes"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ASCENDC_DIR)]
    sys.modules[package_name] = package

    for module_name in ("_runtime", "_kda_policy", "_aclnn_ctypes"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified_name, ASCENDC_DIR / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}._aclnn_ctypes"]


ACLNN_CTYPES = load_aclnn_ctypes_module()


class FakeTensor:
    def __init__(self, shape, dtype=None, *, device_type="npu", contiguous=True):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type=device_type)
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


class FakeCallContext:
    def __init__(self):
        self.descriptor_names = []
        self.descriptor_metadata = []
        self.descriptor_values = None

    def tensor(self, tensor, name, *, acl_format_override=None, storage_shape_override=None):
        self.descriptor_names.append(name)
        self.descriptor_metadata.append((name, tensor, acl_format_override, storage_shape_override))
        return ctypes.c_void_p(0x1000 + len(self.descriptor_names))

    def int_array(self, values):
        self.descriptor_names.append("cu_seqlens")
        self.descriptor_values = list(values)
        return ctypes.c_void_p(0x2000)


def _fake_torch():
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.bfloat16 = object()
    return fake_torch


# 单例：包装函数内部 `import torch` 时拿到的是同一个模块对象，测试里比较 dtype 才有意义
FAKE_TORCH = _fake_torch()


class PreProcessFwdKernelMergedCtypesTest(unittest.TestCase):
    def _run(self, k, w, u, **kwargs):
        """跑一次包装函数，返回 (captured, outputs)。"""
        captured = {}

        def fake_zeros(shape, like, **kw):
            captured["hm_shape"] = tuple(shape)
            captured["hm_dtype"] = kw.get("dtype", like.dtype)
            return FakeTensor(shape, kw.get("dtype", like.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            ctx = FakeCallContext()
            captured["name"] = name
            captured["ctx"] = ctx
            captured["args"] = build_args(ctx)
            return outputs

        with mock.patch.dict(sys.modules, {"torch": FAKE_TORCH}), \
                mock.patch.object(ACLNN_CTYPES, "_zeros", side_effect=fake_zeros), \
                mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
            outputs = ACLNN_CTYPES.npu_pre_process_fwd_kernel_merged(k, w, u, **kwargs)
        return captured, outputs

    def test_sub_interval_window_matches_competitor_call(self):
        """竞品形态：整根张量 T=512，只算子区间 [40,512)（bos > 0）。"""
        fake = FAKE_TORCH
        T, HK, HV, K, V = 512, 8, 8, 128, 128
        k = FakeTensor((1, HK, T, K), fake.bfloat16)
        w = FakeTensor((1, HV, T, K), fake.bfloat16)
        u = FakeTensor((1, HV, T, V), fake.bfloat16)
        gk = FakeTensor((1, HV, T, K), fake.float32)

        captured, outputs = self._run(k, w, u, gk=gk, cu_seqlens=[40, 512])

        self.assertEqual(captured["name"], "aclnnPreProcessFwdKernelMerged")
        self.assertEqual(captured["hm_shape"], (1, HV, K, V + K))
        self.assertIs(captured["hm_dtype"], fake.float32)
        self.assertEqual(outputs.shape, (1, HV, K, V + K))
        # 10 个实参：k,w,u,g,gk,bg,v,cu_seqlens,chunk_size,hm
        self.assertEqual(len(captured["args"]), 10)
        self.assertEqual(captured["ctx"].descriptor_names,
                         ["k", "w", "u", "g", "gk", "bg", "v", "cu_seqlens", "hm"])
        self.assertEqual(captured["ctx"].descriptor_values, [40, 512])

    def test_multi_segment_window_yields_one_chain_per_segment(self):
        fake = FAKE_TORCH
        T, HV, K, V = 512, 32, 128, 128
        k = FakeTensor((1, HV, T, K), fake.bfloat16)
        w = FakeTensor((1, HV, T, K), fake.bfloat16)
        u = FakeTensor((1, HV, T, V), fake.bfloat16)
        gk = FakeTensor((1, HV, T, K), fake.float32)

        captured, outputs = self._run(k, w, u, gk=gk, cu_seqlens=[0, 88, 188, 512])

        self.assertEqual(captured["hm_shape"], (3, HV, K, V + K))
        self.assertEqual(captured["ctx"].descriptor_values, [0, 88, 188, 512])
        self.assertEqual(outputs.shape, (3, HV, K, V + K))

    def test_rejects_dense_batch(self):
        fake = FAKE_TORCH
        k = FakeTensor((2, 8, 512, 128), fake.bfloat16)
        w = FakeTensor((2, 8, 512, 128), fake.bfloat16)
        u = FakeTensor((2, 8, 512, 128), fake.bfloat16)
        gk = FakeTensor((2, 8, 512, 128), fake.float32)
        with self.assertRaises(ValueError):
            self._run(k, w, u, gk=gk, cu_seqlens=[0, 512])

    def test_requires_cu_seqlens(self):
        fake = FAKE_TORCH
        k = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        w = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        u = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        gk = FakeTensor((1, 8, 512, 128), fake.float32)
        with self.assertRaises(ValueError):
            self._run(k, w, u, gk=gk, cu_seqlens=None)

    def test_requires_exactly_one_gate(self):
        fake = FAKE_TORCH
        k = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        w = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        u = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        g = FakeTensor((1, 8, 512), fake.float32)
        gk = FakeTensor((1, 8, 512, 128), fake.float32)
        with self.assertRaises(ValueError):
            self._run(k, w, u, cu_seqlens=[0, 512])
        with self.assertRaises(ValueError):
            self._run(k, w, u, g=g, gk=gk, cu_seqlens=[0, 512])

    def test_rejects_dplr_bg(self):
        """DPLR 不支持：bg 传非空直接拒绝（不再有"只要配上 gk 就放行"的通道）。"""
        fake = FAKE_TORCH
        k = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        w = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        u = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        gk = FakeTensor((1, 8, 512, 128), fake.float32)
        bg = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        with self.assertRaises(NotImplementedError):
            self._run(k, w, u, gk=gk, bg=bg, v=u, cu_seqlens=[0, 512])

    def test_rejects_dplr_v(self):
        """DPLR 不支持：v 传非空直接拒绝（GDN/KDA 的取值来自 u）。"""
        fake = FAKE_TORCH
        k = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        w = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        u = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        gk = FakeTensor((1, 8, 512, 128), fake.float32)
        with self.assertRaises(NotImplementedError):
            self._run(k, w, u, gk=gk, v=u, cu_seqlens=[0, 512])

    def test_rejects_out_of_range_sub_interval(self):
        fake = FAKE_TORCH
        k = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        w = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        u = FakeTensor((1, 8, 512, 128), fake.bfloat16)
        gk = FakeTensor((1, 8, 512, 128), fake.float32)
        with self.assertRaises(ValueError):
            self._run(k, w, u, gk=gk, cu_seqlens=[40, 513])
        with self.assertRaises(ValueError):
            self._run(k, w, u, gk=gk, cu_seqlens=[40, 40])

    def test_get_workspace_argtypes_matches_aclnn_prototype(self):
        expected = [
            *([ctypes.c_void_p] * 8),   # k,w,u,g,gk,bg,v,cuSeqlens(aclIntArray)
            ctypes.c_int64,             # chunkSize
            ctypes.c_void_p,            # hmOut
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnPreProcessFwdKernelMerged"],
            expected,
        )

    def test_wrapper_signature(self):
        sig = inspect.signature(ACLNN_CTYPES.npu_pre_process_fwd_kernel_merged)
        self.assertEqual(list(sig.parameters)[:3], ["k", "w", "u"])
        self.assertIsNone(sig.parameters["g"].default)
        self.assertIsNone(sig.parameters["gk"].default)
        self.assertIsNone(sig.parameters["cu_seqlens"].default)
        self.assertEqual(sig.parameters["chunk_size"].default, 64)

    def test_registered_in_ascendc_ops(self):
        init_source = (ASCENDC_DIR / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('"npu_pre_process_fwd_kernel_merged"', init_source)


if __name__ == "__main__":
    unittest.main()
