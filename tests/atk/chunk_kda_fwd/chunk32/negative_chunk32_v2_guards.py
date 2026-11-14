#!/usr/bin/env python3
"""Negative regression for chunk32 guard diagnostics via aclnnChunkKdaFwdV2.

chunk32 大融合由独立 L0 算子 ChunkKdaFwdFused 承载，仅 aclnnChunkKdaFwdV2 路由
可达；七个守卫（safeGate/缺 A_log/缺 dt_bias/K=V=128/q 必须 BF16/varlen 拒绝
BNSD/不导出 gkOut）的诊断文案逐条验证，全部返回 ACLNN_ERR_PARAM_INVALID
(161002) 且带参数名。V1 对 chunk_size=32 的整体拒绝由
negative_chunk32_v1_rejects_chunk32.py 覆盖。
"""

from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise SystemExit("torch_npu is required") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import aclnn_error_probe as _raw_abi

SEQLEN = 32
HEADS = 1
DIM = 128


def _base_tensors(k_dim: int = DIM, q_dtype: torch.dtype = torch.bfloat16,
                  layout: str = "BSND", varlen: bool = False):
    """构造一组满足 chunk32 契约的合法输入，供各 mutation 单独破坏一项。"""
    if layout == "BNSD":
        q = torch.zeros((1, HEADS, SEQLEN, k_dim), dtype=q_dtype, device="npu")
        beta = torch.zeros((1, HEADS, SEQLEN), dtype=torch.float32, device="npu")
    else:
        q = torch.zeros((1, SEQLEN, HEADS, k_dim), dtype=q_dtype, device="npu")
        beta = torch.zeros((1, SEQLEN, HEADS), dtype=torch.float32, device="npu")
    k = torch.zeros_like(q) if q_dtype == torch.bfloat16 else torch.zeros(q.shape, dtype=torch.bfloat16, device="npu")
    v = torch.zeros(q.shape[:-1] + (DIM,), dtype=torch.bfloat16, device="npu")
    g = torch.zeros(q.shape, dtype=torch.float32, device="npu")
    a_log = torch.zeros((HEADS,), dtype=torch.float32, device="npu")
    dt_bias = torch.zeros((HEADS * k_dim,), dtype=torch.float32, device="npu")
    attn_out = torch.zeros((1, SEQLEN, HEADS, DIM), dtype=torch.bfloat16, device="npu")
    cu = [0, SEQLEN] if varlen else None
    return q, k, v, g, beta, a_log, dt_bias, attn_out, cu


def _call_v2(label: str, *, safe_gate=True, a_log="real", dt_bias="real",
             k_dim=DIM, q_dtype=torch.bfloat16, layout="BSND", varlen=False,
             gk_out=None) -> str | None:
    q, k, v, g, beta, a_log_t, dt_bias_t, attn_out, cu = _base_tensors(
        k_dim=k_dim, q_dtype=q_dtype, layout=layout, varlen=varlen)
    layout_buffer = ctypes.create_string_buffer(layout.encode("utf-8"))
    try:
        _raw_abi.reject_workspace(
            "aclnnChunkKdaFwdV2",
            lambda ctx: [
                ctx.tensor(q, "q"), ctx.tensor(k, "k"), ctx.tensor(v, "v"),
                ctx.tensor(g, "g"), ctx.tensor(beta, "beta"),
                ctx.tensor(a_log_t if a_log == "real" else None, "A_log"),
                ctx.tensor(dt_bias_t if dt_bias == "real" else None, "dt_bias"),
                ctx.tensor(None, "initial_state"),
                ctx.int_array(cu), ctx.int_array(None),
                ctypes.cast(layout_buffer, ctypes.c_char_p),
                ctypes.c_double(DIM ** -0.5), ctypes.c_int64(32), ctypes.c_bool(safe_gate),
                ctypes.c_double(-5.0), ctypes.c_bool(True), ctypes.c_bool(False),
                ctypes.c_double(1.0e-6), ctypes.c_bool(True), ctypes.c_bool(True),
                ctypes.c_bool(False), ctypes.c_bool(True),
                ctx.tensor(attn_out, "attn_out"), ctx.tensor(None, "final_state"),
                ctx.tensor(gk_out, "gk"), ctx.tensor(None, "Aqk"), ctx.tensor(None, "Akk"),
                ctx.tensor(None, "w"), ctx.tensor(None, "u"), ctx.tensor(None, "qg"),
                ctx.tensor(None, "kg"), ctx.tensor(None, "v_new"), ctx.tensor(None, "h"),
                ctx.tensor(None, "q_hat"), ctx.tensor(None, "k_hat"),
                ctx.tensor(None, "q_rstd"), ctx.tensor(None, "k_rstd"),
                ctx.tensor(None, "beta_eff"),
            ],
            (attn_out,) + ((gk_out,) if gk_out is not None else ()),
        )
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError(f"{label}: unexpectedly returned ACLNN_SUCCESS")


def _expect(label: str, error: str | None, *needles: str) -> None:
    assert error is not None, f"{label}: unexpectedly succeeded"
    match = re.search(r"aclnnStatus=(\d+)", error)
    code = int(match.group(1)) if match else None
    assert code == 161002, (label, code, error)
    for needle in needles:
        assert needle in error, (label, needle, error)


def main() -> int:
    _expect("safe_gate=false", _call_v2("safe_gate", safe_gate=False),
            "Parameter safeGate of aclnnChunkKdaFwd", "true for chunkSize=32")
    _expect("missing A_log", _call_v2("alog", a_log=None),
            "Parameter aLogOptional of aclnnChunkKdaFwd", "null")
    _expect("missing dt_bias", _call_v2("dtbias", dt_bias=None),
            "Parameter dtBiasOptional of aclnnChunkKdaFwd", "null")
    _expect("K=V=64", _call_v2("bad_k", k_dim=64),
            "Parameter K/V of aclnnChunkKdaFwd", "64/128")
    _expect("q fp32", _call_v2("q_fp32", q_dtype=torch.float32),
            "Parameter q of aclnnChunkKdaFwd", "BF16")
    _expect("BNSD varlen", _call_v2("bnsd", layout="BNSD", varlen=True),
            "Parameter layout of aclnnChunkKdaFwd", "BNSD with cuSeqlensOptional")
    gk = torch.zeros((1, SEQLEN, HEADS, DIM), dtype=torch.float32, device="npu")
    _expect("gkOut provided", _call_v2("gk", gk_out=gk),
            "Parameter gkOut of aclnnChunkKdaFwd", "provided")
    print("PASS: all 7 chunk32 guards reject via aclnnChunkKdaFwdV2 with parameter context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
