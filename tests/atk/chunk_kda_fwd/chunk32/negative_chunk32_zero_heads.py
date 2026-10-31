#!/usr/bin/env python3
"""Regression for zero-head dense/varlen guards in aclnnChunkKdaFwdV2."""

from __future__ import annotations

import ctypes
import re

import torch

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise SystemExit("torch_npu is required") from exc

from fla_npu.ops.ascendc._aclnn_ctypes import _call_aclnn


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
        _call_aclnn(
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


def main() -> int:
    _check_zero_heads(varlen=False)
    _check_zero_heads(varlen=True)
    print("PASS: dense and NTD varlen zero-head guards return 161002 with H/HV context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
