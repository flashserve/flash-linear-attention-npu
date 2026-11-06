#!/usr/bin/env python3
"""Negative regression: aclnnChunkKdaFwd (V1) must reject chunk_size=32.

chunk32 大融合已迁至独立 L0 算子 ChunkKdaFwdFused，仅由 aclnnChunkKdaFwdV2
路由；V1 收到 chunk_size=32 必须返回 ACLNN_ERR_PARAM_INVALID (161002) 并在
错误文本中指明改用 V2。
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


def main() -> int:
    seqlen, head_num = 32, 1
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
            "aclnnChunkKdaFwd",
            lambda ctx: [
                ctx.tensor(q, "q"), ctx.tensor(k, "k"), ctx.tensor(v, "v"),
                ctx.tensor(g, "g"), ctx.tensor(beta, "beta"),
                ctx.tensor(a_log, "A_log"), ctx.tensor(dt_bias, "dt_bias"),
                ctx.tensor(None, "initial_state"), ctx.int_array(None), ctx.int_array(None),
                ctypes.cast(layout_buffer, ctypes.c_char_p),
                ctypes.c_double(128 ** -0.5), ctypes.c_int64(32), ctypes.c_bool(True),
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
    print("PASS: aclnnChunkKdaFwd rejects chunk_size=32 and points to aclnnChunkKdaFwdV2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
