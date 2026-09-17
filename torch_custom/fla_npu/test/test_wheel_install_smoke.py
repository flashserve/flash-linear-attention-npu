"""Wheel 安装态 smoke：thin 扩展加载、dispatch 门控与回退语义。

面向已安装 fla_npu wheel（含内嵌 OPP 与 _C_thin）的运行环境；NPU 主机上
执行时还校验 OPP 库可加载。
"""
from __future__ import annotations

import os
import unittest


def _npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


THIN_OPS = (
    "recurrent_gated_delta_rule",
    "kda_gate_cumsum",
    "chunk_local_cumsum",
    "chunk_scaled_dot_kkt",
    "recompute_w_u_fwd",
    "prepare_wy_repr_bwd_full",
    "prepare_wy_repr_bwd",
    "chunk_bwd_dv_local",
    "prepare_wy_repr_bwd_da",
    "chunk_bwd_dqkwg",
    "fast_gelu_custom",
    "fast_gelu_custom_backward",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_fwd_h",
    "chunk_fwd_o",
    "chunk_gated_delta_rule_bwd_dhu",
    "causal_conv1d_bwd",
    "chunk_kda_fwd",
    "chunk_kda_bwd_intra",
    "chunk_kda_bwd",
    "recurrent_kda",
    "chunk_gated_delta_rule_fwd_prepare",
    "chunk_gated_delta_rule_bwd_finalize",
    "chunk_gated_delta_rule_fwd",
    "solve_tri",
)


class TestWheelInstallSmoke(unittest.TestCase):
    def test_extension_importable(self):
        import fla_npu._C_thin as ext

        self.assertTrue(callable(ext.npu_kda_gate_cumsum))
        self.assertTrue(callable(getattr(ext, "npu_causal_conv1d_bwd", None)))

    def test_dispatch_gating(self):
        from fla_npu.ops.ascendc import _get_thin_op

        os.environ.pop("FLA_NPU_THIN_LAUNCHER", None)
        for name in THIN_OPS:
            self.assertIsNotNone(
                _get_thin_op(name), f"{name} should resolve to thin")
        # conv1d legacy 保持 ctypes，直到上游 PR #390 合入统一 ABI；
        # solve_tri dense 已原生 thin，varlen 在 _thin wrapper 内回退 ctypes。
        self.assertIsNone(_get_thin_op("npu_causal_conv1d"))
        os.environ["FLA_NPU_THIN_LAUNCHER"] = "0"
        for name in THIN_OPS:
            self.assertIsNone(_get_thin_op(name),
                              f"{name} should fall back to ctypes")
        os.environ.pop("FLA_NPU_THIN_LAUNCHER", None)

    @unittest.skipUnless(_npu_available(), "NPU is required")
    def test_opp_and_runtime_ready(self):
        import fla_npu

        libs = fla_npu.load_ascendc_opapi_libraries()
        self.assertTrue(libs)
        from fla_npu.ops.ascendc import _aclnn_ctypes

        self.assertTrue(callable(
            _aclnn_ctypes.npu_recurrent_gated_delta_rule))
        self.assertTrue(callable(
            _aclnn_ctypes.npu_chunk_gated_delta_rule_fwd_h))


if __name__ == "__main__":
    unittest.main()
