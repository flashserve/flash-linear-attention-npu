#!/usr/bin/env python3
"""Chunk32 negative regressions via aclnnChunkKdaFwd / aclnnChunkKdaFwdV2.

覆盖零 head、head/lowerBound 上限、七项 V2 守卫，以及 V1 对 chunk32 的拒绝。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import aclnn_error_probe as _raw_abi

SEQLEN = 32
HEADS = 1
DIM = 128


def _check_zero_heads(varlen: bool) -> None:
    if varlen:
        layout = "NTD"
        q = torch.empty((0, 128, 128), dtype=torch.bfloat16, device="npu")
        beta = torch.empty((0, 128), dtype=torch.float32, device="npu")
        attn_out = torch.empty((128, 0, 128), dtype=torch.bfloat16, device="npu")
        cu = [0, 128]
    else:
        layout = "BSND"
        q = torch.empty((1, 128, 0, 128), dtype=torch.bfloat16, device="npu")
        beta = torch.empty((1, 128, 0), dtype=torch.float32, device="npu")
        attn_out = torch.empty_like(q)
        cu = None
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(q.shape, dtype=torch.float32, device="npu")
    a_log = torch.empty((0,), dtype=torch.float32, device="npu")
    dt_bias = torch.empty((0,), dtype=torch.float32, device="npu")
    layout_buffer = ctypes.create_string_buffer(layout.encode("utf-8"))
    try:
        _raw_abi.reject_workspace(
            "aclnnChunkKdaFwdV2",
            lambda ctx: [
                ctx.tensor(q, "q"), ctx.tensor(k, "k"), ctx.tensor(v, "v"),
                ctx.tensor(g, "g"), ctx.tensor(beta, "beta"),
                ctx.tensor(a_log, "A_log"), ctx.tensor(dt_bias, "dt_bias"),
                ctx.tensor(None, "initial_state"), ctx.int_array(cu), ctx.int_array(None),
                ctypes.cast(layout_buffer, ctypes.c_char_p),
                ctypes.c_double(128 ** -0.5), ctypes.c_int64(32), ctypes.c_bool(True),
                ctypes.c_double(-5.0), ctypes.c_bool(True), ctypes.c_bool(False),
                ctypes.c_double(1.0e-6), ctypes.c_bool(True), ctypes.c_bool(True),
                ctypes.c_bool(False), ctypes.c_bool(True),
                ctx.tensor(attn_out, "attn_out"), ctx.tensor(None, "final_state"),
                ctx.tensor(None, "gk"), ctx.tensor(None, "Aqk"), ctx.tensor(None, "Akk"),
                ctx.tensor(None, "w"), ctx.tensor(None, "u"), ctx.tensor(None, "qg"),
                ctx.tensor(None, "kg"), ctx.tensor(None, "v_new"), ctx.tensor(None, "h"),
                ctx.tensor(None, "q_hat"), ctx.tensor(None, "k_hat"),
                ctx.tensor(None, "q_rstd"), ctx.tensor(None, "k_rstd"),
                ctx.tensor(None, "beta_eff"),
            ],
            (attn_out,),
        )
    except RuntimeError as exc:
        match = re.search(r"aclnnStatus=(\d+)", str(exc))
        code = int(match.group(1)) if match else None
        assert code == 161002, (code, str(exc))
        assert "Parameter H/HV of aclnnChunkKdaFwd" in str(exc), str(exc)
        assert "0/0" in str(exc), str(exc)
    else:
        raise AssertionError("zero-head input unexpectedly returned ACLNN_SUCCESS")


def _call_head_or_lower_bound(head_num: int, lower_bound: float) -> str | None:
    q = torch.zeros((1, SEQLEN, head_num, DIM), dtype=torch.bfloat16, device="npu")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros(q.shape, dtype=torch.float32, device="npu")
    beta = torch.zeros((1, SEQLEN, head_num), dtype=torch.float32, device="npu")
    a_log = torch.zeros((head_num,), dtype=torch.float32, device="npu")
    dt_bias = torch.zeros((head_num * DIM,), dtype=torch.float32, device="npu")
    attn_out = torch.zeros_like(q)
    layout_buffer = ctypes.create_string_buffer("BSND".encode("utf-8"))
    try:
        _raw_abi.reject_workspace(
            "aclnnChunkKdaFwdV2",
            lambda ctx: [
                ctx.tensor(q, "q"), ctx.tensor(k, "k"), ctx.tensor(v, "v"),
                ctx.tensor(g, "g"), ctx.tensor(beta, "beta"),
                ctx.tensor(a_log, "A_log"), ctx.tensor(dt_bias, "dt_bias"),
                ctx.tensor(None, "initial_state"), ctx.int_array(None), ctx.int_array(None),
                ctypes.cast(layout_buffer, ctypes.c_char_p),
                ctypes.c_double(DIM ** -0.5), ctypes.c_int64(32), ctypes.c_bool(True),
                ctypes.c_double(lower_bound), ctypes.c_bool(True), ctypes.c_bool(False),
                ctypes.c_double(1.0e-6), ctypes.c_bool(True), ctypes.c_bool(True),
                ctypes.c_bool(False), ctypes.c_bool(True),
                ctx.tensor(attn_out, "attn_out"), ctx.tensor(None, "final_state"),
                ctx.tensor(None, "gk"), ctx.tensor(None, "Aqk"), ctx.tensor(None, "Akk"),
                ctx.tensor(None, "w"), ctx.tensor(None, "u"), ctx.tensor(None, "qg"),
                ctx.tensor(None, "kg"), ctx.tensor(None, "v_new"), ctx.tensor(None, "h"),
                ctx.tensor(None, "q_hat"), ctx.tensor(None, "k_hat"),
                ctx.tensor(None, "q_rstd"), ctx.tensor(None, "k_rstd"),
                ctx.tensor(None, "beta_eff"),
            ],
            (attn_out,),
        )
    except RuntimeError as exc:
        return str(exc)
    return None


def _expect_param_invalid(label: str, error: str | None, needle: str) -> None:
    assert error is not None, f"{label}: unexpectedly returned ACLNN_SUCCESS"
    match = re.search(r"aclnnStatus=(\d+)", error)
    code = int(match.group(1)) if match else None
    assert code == 161002, (label, code, error)
    assert needle in error, (label, needle, error)


def _base_tensors(k_dim: int = DIM, q_dtype: torch.dtype = torch.bfloat16,
                  layout: str = "BSND", varlen: bool = False):
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


def _expect_guard(label: str, error: str | None, *needles: str) -> None:
    assert error is not None, f"{label}: unexpectedly succeeded"
    match = re.search(r"aclnnStatus=(\d+)", error)
    code = int(match.group(1)) if match else None
    assert code == 161002, (label, code, error)
    for needle in needles:
        assert needle in error, (label, needle, error)


def _check_v1_rejects_chunk32() -> None:
    q = torch.zeros((1, SEQLEN, HEADS, DIM), dtype=torch.bfloat16, device="npu")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros(q.shape, dtype=torch.float32, device="npu")
    beta = torch.zeros((1, SEQLEN, HEADS), dtype=torch.float32, device="npu")
    a_log = torch.zeros((HEADS,), dtype=torch.float32, device="npu")
    dt_bias = torch.zeros((HEADS * DIM,), dtype=torch.float32, device="npu")
    attn_out = torch.zeros_like(q)
    layout_buffer = ctypes.create_string_buffer("BSND".encode("utf-8"))
    try:
        _raw_abi.reject_workspace(
            "aclnnChunkKdaFwd",
            lambda ctx: [
                ctx.tensor(q, "q"), ctx.tensor(k, "k"), ctx.tensor(v, "v"),
                ctx.tensor(g, "g"), ctx.tensor(beta, "beta"),
                ctx.tensor(a_log, "A_log"), ctx.tensor(dt_bias, "dt_bias"),
                ctx.tensor(None, "initial_state"), ctx.int_array(None), ctx.int_array(None),
                ctypes.cast(layout_buffer, ctypes.c_char_p),
                ctypes.c_double(DIM ** -0.5), ctypes.c_int64(32), ctypes.c_bool(True),
                ctypes.c_double(-5.0), ctypes.c_bool(True), ctypes.c_bool(False),
                ctx.tensor(attn_out, "attn_out"), ctx.tensor(None, "final_state"),
                ctx.tensor(None, "gk"), ctx.tensor(None, "Aqk"), ctx.tensor(None, "Akk"),
                ctx.tensor(None, "w"), ctx.tensor(None, "u"), ctx.tensor(None, "qg"),
                ctx.tensor(None, "kg"), ctx.tensor(None, "v_new"), ctx.tensor(None, "h"),
            ],
            (attn_out,),
        )
    except RuntimeError as exc:
        match = re.search(r"aclnnStatus=(\d+)", str(exc))
        code = int(match.group(1)) if match else None
        assert code == 161002, (code, str(exc))
        assert "aclnnChunkKdaFwdV2" in str(exc), str(exc)
    else:
        raise AssertionError("V1 chunk_size=32 unexpectedly returned ACLNN_SUCCESS")


def main() -> int:
    _check_zero_heads(varlen=False)
    _check_zero_heads(varlen=True)
    print("PASS: dense and NTD varlen zero-head guards return 161002 with H/HV context")

    _expect_param_invalid(
        "H=HV=129", _call_head_or_lower_bound(129, -5.0), "Parameter H/HV of aclnnChunkKdaFwd")
    for label, lower_bound in (
        ("lowerBound=0.0", 0.0),
        ("lowerBound=+1.0", 1.0),
        ("lowerBound=-6.0", -6.0),
        ("lowerBound=NaN", float("nan")),
    ):
        _expect_param_invalid(
            label, _call_head_or_lower_bound(1, lower_bound), "Parameter lowerBound of aclnnChunkKdaFwd")
    print("PASS: head-limit and lowerBound guards return 161002 with parameter context")

    _expect_guard("safe_gate=false", _call_v2("safe_gate", safe_gate=False),
                  "Parameter safeGate of aclnnChunkKdaFwd", "true for chunkSize=32")
    _expect_guard("missing A_log", _call_v2("alog", a_log=None),
                  "Parameter aLogOptional of aclnnChunkKdaFwd", "null")
    _expect_guard("missing dt_bias", _call_v2("dtbias", dt_bias=None),
                  "Parameter dtBiasOptional of aclnnChunkKdaFwd", "null")
    _expect_guard("K=V=64", _call_v2("bad_k", k_dim=64),
                  "Parameter K/V of aclnnChunkKdaFwd", "64/128")
    _expect_guard("q fp32", _call_v2("q_fp32", q_dtype=torch.float32),
                  "Parameter q of aclnnChunkKdaFwd", "BF16")
    _expect_guard("BNSD varlen", _call_v2("bnsd", layout="BNSD", varlen=True),
                  "Parameter layout of aclnnChunkKdaFwd", "BNSD with cuSeqlensOptional")
    gk = torch.zeros((1, SEQLEN, HEADS, DIM), dtype=torch.float32, device="npu")
    _expect_guard("gkOut provided", _call_v2("gk", gk_out=gk),
                  "Parameter gkOut of aclnnChunkKdaFwd", "provided")
    print("PASS: all 7 chunk32 guards reject via aclnnChunkKdaFwdV2 with parameter context")

    _check_v1_rejects_chunk32()
    print("PASS: aclnnChunkKdaFwd rejects chunk_size=32 and points to aclnnChunkKdaFwdV2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
