#!/usr/bin/env python3
import gc
import importlib

import torch
import torch_npu

from fla_npu.ops import ascendc as fla_ascendc


B = 1
H = 4
T = 2048
K = 128
V = 128
CHUNK_SIZE = 64
K_DTYPE = torch.bfloat16
G_DTYPE = torch.float32


def release_aclnn_keepalive():
    try:
        runtime_mod = importlib.import_module("fla_npu.ops.ascendc._runtime")
        runtime_mod._RECENT_LAUNCH_STORAGE.clear()
    except Exception:
        pass


def main() -> int:
    with torch.no_grad():
        k = torch.zeros((B, H, T, K), dtype=K_DTYPE)
        v = torch.zeros((B, H, T, V), dtype=K_DTYPE)
        beta = torch.zeros((B, H, T), dtype=G_DTYPE)
        A = torch.zeros((B, H, T, CHUNK_SIZE), dtype=K_DTYPE)
        dw = torch.zeros((B, H, T, K), dtype=K_DTYPE)
        du = torch.zeros((B, H, T, V), dtype=K_DTYPE)
        g = torch.zeros((B, H, T), dtype=G_DTYPE)

        outputs = fla_ascendc.prepare_wy_repr_bwd(
            k.npu(), v.npu(), beta.npu(), A.npu(), dw.npu(), du.npu(), g.npu(), CHUNK_SIZE
        )

        print(
            "SIM ZERO PrepareWyReprBwd DONE "
            f"B={B} H={H} T={T} K={K} V={V} chunk_size={CHUNK_SIZE} "
            f"outputs={[tuple(out.shape) for out in outputs]}",
            flush=True,
        )

        del k, v, beta, A, dw, du, g, outputs
        release_aclnn_keepalive()
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
