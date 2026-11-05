#!/usr/bin/env python3
"""Negative regression for chunk32 head-limit and lowerBound guards.

Covers direct aclnn V2 calls (the same L2 entry the Stable wrapper reaches):
- H=HV=129 exceeds the MAX_KDA_HEAD_NUM=128 contract;
- lowerBound = 0.0 / +1.0 / -6.0 / NaN fall outside the finite [-5, 0) range.
All must return ACLNN_ERR_PARAM_INVALID (161002) with the parameter name in
the error text.
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


def _call_chunk32(head_num: int, lower_bound: float) -> str | None:
    """Invoke aclnnChunkKdaFwdV2 with chunk32; return the error text or None."""
    seqlen = 32
    q = torch.zeros((1, seqlen, head_num, 128), dtype=torch.bfloat16, device="npu")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros(q.shape, dtype=torch.float32, device="npu")
    beta = torch.zeros((1, seqlen, head_num), dtype=torch.float32, device="npu")
    a_log = torch.zeros((head_num,), dtype=torch.float32, device="npu")
    dt_bias = torch.zeros((head_num * 128,), dtype=torch.float32, device="npu")
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
                ctypes.c_double(128 ** -0.5), ctypes.c_int64(32), ctypes.c_bool(True),
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


def main() -> int:
    _expect_param_invalid(
        "H=HV=129", _call_chunk32(129, -5.0), "Parameter H/HV of aclnnChunkKdaFwd")
    for label, lower_bound in (
        ("lowerBound=0.0", 0.0),
        ("lowerBound=+1.0", 1.0),
        ("lowerBound=-6.0", -6.0),
        ("lowerBound=NaN", float("nan")),
    ):
        _expect_param_invalid(
            label, _call_chunk32(1, lower_bound), "Parameter lowerBound of aclnnChunkKdaFwd")
    print("PASS: head-limit and lowerBound guards return 161002 with parameter context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
