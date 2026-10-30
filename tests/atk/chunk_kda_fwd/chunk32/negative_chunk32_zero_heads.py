#!/usr/bin/env python3
"""Regression for zero-head dense/varlen guards in aclnnChunkKdaFwdFused."""

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
        # TND: [T,H,K], beta has already been transposed to [HV,T].
        q = torch.empty((128, 0, 128), dtype=torch.bfloat16, device="npu")
        v = torch.empty_like(q)
        beta = torch.empty((0, 128), dtype=torch.bfloat16, device="npu")
        cu = [0, 128]
    else:
        # Dense low-level view: [H,T,K].
        q = torch.empty((0, 128, 128), dtype=torch.bfloat16, device="npu")
        v = torch.empty_like(q)
        beta = torch.empty((0, 128), dtype=torch.bfloat16, device="npu")
        cu = None
    k = torch.empty_like(q)
    g = torch.empty_like(q)
    a_log = torch.empty((0,), dtype=torch.float32, device="npu")
    dt_bias = torch.empty((0,), dtype=torch.float32, device="npu")
    o = torch.empty_like(q)

    try:
        _call_aclnn(
            "aclnnChunkKdaFwdFused",
            lambda ctx: [
                ctx.tensor(q, "q"),
                ctx.tensor(k, "k"),
                ctx.tensor(v, "v"),
                ctx.tensor(g, "g"),
                ctx.tensor(beta, "beta"),
                ctx.tensor(a_log, "aLog"),
                ctx.tensor(dt_bias, "dtBias"),
                ctx.tensor(None, "initialState"),
                ctypes.c_double(128 ** -0.5),
                ctypes.c_double(-5.0),
                ctx.tensor(o, "o"),
                ctx.tensor(None, "finalState"),
                ctx.int_array(cu),
            ],
            (o,),
        )
    except RuntimeError as exc:
        match = re.search(r"aclnnStatus=(\d+)", str(exc))
        code = int(match.group(1)) if match else None
        assert code == 161002, (code, str(exc))
        assert "Parameter H/HV of aclnnChunkKdaFwdFused" in str(exc), str(exc)
        assert "H=0,HV=0" in str(exc), str(exc)
    else:
        raise AssertionError("zero-head input unexpectedly returned ACLNN_SUCCESS")


def main() -> int:
    _check_zero_heads(varlen=False)
    _check_zero_heads(varlen=True)
    print("PASS: dense and TND varlen zero-head guards return 161002 with H/HV context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
