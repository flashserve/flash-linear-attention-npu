#!/usr/bin/env python3
"""chunk32 融合 kernel 性能基线/复测 runner(nd2nz 优化专项)。

口径:B=1 H=HV=96 K=V=128 bf16 dense BSND,use_gate_in_kernel + A_log + dt_bias,
safe_gate=True lower_bound=-5(与套件 fd32_case 同族数据分布)。
--state on 时带 initial_state(fp32 [1,96,128,128])+ output_final_state=True,
on-off 差值即 LoadInitState+StoreFinalState 的端到端成本(本次优化靶心)。

计时:每发前后 synchronize 的 wall clock(min/median);权威 device 时间用
msprof --application 包本脚本取 op_statistic;pipe 占用用 msopprof
--aic-metrics=PipeUtilization。

环境:source CANN 9.3.0 set_env.sh;conda python(部署件在 conda site-packages);
卡由 ASCEND_RT_VISIBLE_DEVICES 外部指定(只用 4/5)。
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    raise SystemExit("torch_npu 不可用")

from fla_npu.ops.ascendc._aclnn_ctypes import npu_chunk_kda_fwd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--heads", type=int, default=96)
    p.add_argument("--state", choices=("off", "on"), default="off")
    p.add_argument("--warm-up", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--gate-dtype", choices=("bf16", "fp32"), default="bf16")
    args = p.parse_args()

    B, T, H, D = 1, args.tokens, args.heads, 128
    gen = torch.Generator().manual_seed(args.seed)

    def rnd(*shape, mul=1.0):
        return torch.randn(*shape, generator=gen) * mul

    # 数据分布对齐 fd32_case:beta~N(1.5,0.35) raw、A_log~N(0,0.12)、dt~N(-3,1.65)
    q = rnd(B, T, H, D).bfloat16().npu()
    k = rnd(B, T, H, D).bfloat16().npu()
    v = rnd(B, T, H, D, mul=0.5).bfloat16().npu()
    g = (-torch.rand(B, T, H, D, generator=gen) * 4 - 0.5).bfloat16().npu()  # (-4.5,-0.5) 全负
    beta = rnd(B, T, H, mul=0.35).add_(1.5).bfloat16().npu()
    if args.gate_dtype == "fp32":
        g = g.float()
        beta = beta.float()
    a_log = rnd(H, mul=0.12).float().npu()
    dt_bias = rnd(H * D, mul=1.65).add_(-3.0).float().npu()

    istate = None
    if args.state == "on":
        istate = rnd(1, H, D, D, mul=0.1).float().npu()

    def call():
        return npu_chunk_kda_fwd(
            q, k, v, g, beta, D ** -0.5, 32,
            layout="BSND",
            initial_state=istate,
            output_final_state=(args.state == "on"),
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            safe_gate=True,
            lower_bound=-5.0,
        )

    for _ in range(args.warm_up):
        out = call()
    torch.npu.synchronize()

    times = []
    for _ in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = call()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)  # µs

    o = out[0]
    finite = bool(torch.isfinite(o.float()).all())
    ms_min, ms_med = min(times), statistics.median(times)
    print(f"RESULT T={T} state={args.state} gate={args.gate_dtype}: wall min={ms_min:.1f}us median={ms_med:.1f}us "
          f"(n={args.iters}) finite={finite}")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
