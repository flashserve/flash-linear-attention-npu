#!/usr/bin/env python3
"""最小串联示例：chunk_kda_fwd_prepare -> chunk_fwd_h -> chunk_kda_fwd_finalize。

这三个 Ascend C 算子组合起来就是完整的 KDA 前向，对应仓库里组合入口
`aclnnChunkKdaFwdV2` 的编排。本脚本在 Python 层把它们显式串联起来，
演示三者之间的数据流，不经过融合入口 `chunk_kda_fwd`。

数据流（1:1 head mapping，K=V=128，chunk_size=64，BF16）：
    prepare(q, k, v, g, beta, scale) -> gk/Aqk/w/u/kg/qg_scaled (13 槽)
    fwd_h(kg, w, u, gk=gk)            -> (h, v_new, final_state)
    finalize(qg_scaled, Aqk, v_new, h) -> attn_out

运行前准备（参考仓库 tests/atk/README.md）：
    source <cann>/set_env.sh
    source <fla_npu>/vendors/fla_npu_transformer/bin/set_env.bash
用法：
    python examples/chunk_kda_fwd_three_op.py
"""

import torch

try:
    import torch_npu  # noqa: F401  注册 NPU 后端
except ImportError:
    pass

from fla_npu.ops.ascendc import (
    chunk_kda_fwd,
    chunk_kda_fwd_prepare,
    chunk_fwd_h,
    chunk_kda_fwd_finalize,
)


def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    a = a.float().detach().cpu()
    b = b.float().detach().cpu()
    abs_diff = (a - b).abs()
    rel = (abs_diff / b.abs().clamp_min(1e-6)).max().item()
    max_abs = abs_diff.max().item()
    ok = bool(torch.allclose(a, b, rtol=1e-2, atol=1e-2))
    print(f"{name:<12} max_abs={max_abs:.3e}  max_rel={rel:.3e}  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    device = "npu"
    torch.npu.set_device(0)

    # 与 V2 组合路径支持范围对齐：BF16、K=V=128、chunk_size=64。
    B, T, H, D = 1, 128, 4, 128
    chunk_size = 64
    scale = D ** -0.5

    # 输入均为 BSND（sequence-major，[B, T, H, D]）。
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn_like(q) * 0.05
    # raw gate：rank-4 [B, T, H, D]（与 prepare 的 g 契约一致，float32）。
    g = torch.randn(B, T, H, D, dtype=torch.float32, device=device) * 0.1
    # beta：rank-3 [B, T, H]。
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device).sigmoid()

    # 1) Prepare：产出 chunk 内需要的关键中间量。
    #    13 槽顺序：gk, Aqk, Akk, w, u, qg, kg, qg_scaled,
    #               q_hat, k_hat, q_rstd, k_rstd, beta_eff
    #    backward_mode="none" 只产前向必需槽（gk/Aqk/w/u/kg/qg_scaled），其余为 None。
    (
        gk, Aqk, _Akk, w, u, _qg, kg, qg_scaled,
        _q_hat, _k_hat, _q_rstd, _k_rstd, _beta_eff,
    ) = chunk_kda_fwd_prepare(
        q, k, v, g, beta, scale,
        chunk_size=chunk_size,
        backward_mode="none",
    )

    # 2) FwdH（gk-only 模式）：第一个参数传 prepare 的 kg；产出递推状态 h 与 v_new。
    #    注意 use_exp2 必须与 prepare 保持一致（两个默认都是 exp2 门控），
    #    否则状态递推的衰减基数不对，final_state 会发散。
    h, v_new, final_state = chunk_fwd_h(
        kg, w, u,
        gk=gk,
        chunk_size=chunk_size,
        output_final_state=True,
        use_exp2=True,
    )

    # 3) Finalize：用 qg_scaled/Aqk/v_new/h 得到 attn_out。
    attn_out = chunk_kda_fwd_finalize(qg_scaled, Aqk, v_new, h)

    # 4) 与融合入口 chunk_kda_fwd 数值对齐。
    #    默认 gate 语义下（gate=g、调用方预归一化 q/k、预 sigmoid beta、exp2 门控），
    #    融合入口对本组输入（BF16、K=V=128、chunk_size=64）走 V2 组合路径，即本脚本
    #    的 Prepare -> FwdH -> Finalize，因此 attn_out/final_state/h/v_new 应数值一致。
    fused = chunk_kda_fwd(
        q, k, v, g, beta, scale, chunk_size,
        layout="BSND",
        output_final_state=True,
        disable_recompute=True,
        return_intermediate_states=True,
    )
    (
        fused_attn, fused_final, _fgk, _faqk, _fakk,
        _fw, _fu, _fqg, _fkg, fused_vnew, fused_h, _fstate,
    ) = fused

    torch.npu.synchronize()

    n_chunks = T // chunk_size
    print(f"attn_out.shape = {tuple(attn_out.shape)}  (期望 [B, T, H, 128])")
    print(f"h.shape        = {tuple(h.shape)}  (期望 [B, {n_chunks}, H, 128, 128])")
    print(f"v_new.shape    = {tuple(v_new.shape)}")
    print(f"final_state.shape = {tuple(final_state.shape)}")
    print(f"attn_out finite: {bool(torch.isfinite(attn_out.float()).all().item())}")
    print("--- 三算子串联 vs 融合入口 (rtol=atol=1e-2) ---")
    ok = _report("attn_out", attn_out, fused_attn)
    ok = _report("final_state", final_state, fused_final) and ok
    ok = _report("v_new", v_new, fused_vnew) and ok
    ok = _report("h", h, fused_h) and ok
    print("对齐结果:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()