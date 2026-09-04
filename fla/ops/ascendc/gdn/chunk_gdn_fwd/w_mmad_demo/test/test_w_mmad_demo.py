"""Demo: VCS leaves ND->L1 NZ (8-col) + Stage4/5 MBH -> (I+L)^{-1}."""
from __future__ import annotations

import os
import traceback

import torch
from fla_npu.ops.ascendc._aclnn_ctypes import _call_aclnn


def w_mmad_demo(l: torch.Tensor) -> torch.Tensor:
    dummy = torch.zeros_like(l)
    inv = torch.empty_like(l)
    return _call_aclnn(
        "aclnnWMmadDemo",
        lambda ctx: [
            ctx.tensor(l, "a"),
            ctx.tensor(dummy, "b"),
            ctx.tensor(dummy, "pre"),
            ctx.tensor(inv, "c"),
        ],
        inv,
    )


def main():
    dev = int(os.environ.get("TEST_DEVICE_ID", "0"))
    torch.npu.set_device(dev)
    torch.manual_seed(0)
    n = 64
    # Strictly lower L, modest scale so I+L is well-conditioned.
    L = torch.randn(n, n, dtype=torch.float32)
    L = torch.tril(L, diagonal=-1) * 0.05
    I = torch.eye(n, dtype=torch.float64)
    Ld = L.double()
    ref = torch.linalg.solve(I + Ld, I)
    print(f"device={dev} L absmax={L.abs().max().item():.4g} tril_ok={bool((torch.triu(L) == 0).all())}")
    got = w_mmad_demo(L.npu())
    torch.npu.synchronize()
    g = got.cpu().double()
    residual = (I + Ld) @ g - I
    d = (g - ref).abs()
    print(f"vs inv(I+L) max={d.max().item():.4g} mean={d.mean().item():.4g}")
    print(f"|(I+L)A-I| max={residual.abs().max().item():.4g} mean={residual.abs().mean().item():.4g}")
    print(f"A absmax={g.abs().max().item():.4g} nan={int(~torch.isfinite(g).all())}")
    print("A[:4,:4]=\n", g[:4, :4])
    print("ref[:4,:4]=\n", ref[:4, :4])
    if not torch.isfinite(g).all():
        raise SystemExit("FAIL w_mmad_demo NaN/Inf")
    if residual.abs().max().item() > 5e-3 or d.max().item() > 5e-3:
        raise SystemExit(
            f"FAIL w_mmad_demo residual={residual.abs().max().item():.4g} "
            f"vs_ref={d.max().item():.4g}"
        )
    print("PASS w_mmad_demo")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL w_mmad_demo: {exc}")
        traceback.print_exc()
        raise
