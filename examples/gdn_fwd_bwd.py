"""Ascend 950: 显式串联融合前向和反向（十项返回接口）。

安装同版本完整 fla_npu wheel 并加载 CANN 后运行：
    python examples/gdn_fwd_bwd.py --device 0

这是接口调用示例，不是精度或性能测试；直接传入 dO，不使用 autograd。
"""

import argparse

import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import (
    npu_chunk_gated_delta_rule_bwd,
    npu_chunk_gated_delta_rule_fwd,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--qk-l2norm", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    if not torch.npu.get_device_name(args.device).startswith("Ascend950"):
        raise RuntimeError("本示例要求 Ascend 950 和同版本前后向算子包")

    torch.manual_seed(42)
    B, T, HK, HV, K, V = 1, 128, 2, 4, 128, 128
    chunk_size, scale = 64, K ** -0.5
    # Q/K/V 为 BSND。可用 --qk-l2norm 启用核内归一化。
    q = (torch.randn(B, T, HK, K) * 0.1).to(device, torch.bfloat16)
    k = (torch.randn(B, T, HK, K) * 0.1).to(device, torch.bfloat16)
    v = torch.randn(B, T, HV, V).to(device, torch.bfloat16)
    # 前向输入 g 是逐 token 自然对数门控；prepare 负责累计及 exp2 基底转换。
    g = (-torch.rand(B, T, HV) * 0.1).to(device)
    beta = torch.sigmoid(torch.randn(B, T, HV)).to(device)
    initial_state = torch.zeros(B, HV, K, V, dtype=torch.bfloat16, device=device)

    o, final_state, g_cumsum, A, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd = npu_chunk_gated_delta_rule_fwd(
        q, k, v, g, beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
        scale=scale,
        layout="BSND",
        use_exp2=True,
        use_qk_l2norm_in_kernel=args.qk_l2norm,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        disable_recompute=True,  # 保存反向所需的 g_cumsum 和 A。
        return_intermediate_states=False,
        state_v_first=False,
    )
    torch.npu.synchronize()
    print("FWD_DONE", flush=True)

    if args.qk_l2norm:
        assert q_rstd.shape == k_rstd.shape == (B, HK, T)
    else:
        assert q_hat is q and k_hat is k
        assert q_rstd is None and k_rstd is None

    d_o = torch.randn(B, T, HV, V).to(device, torch.bfloat16)
    # 假设 loss 只依赖 O，因此最终状态的上游梯度 dht=None。
    # 直接传前向 g_cumsum，不再 cumsum、转置或乘 log2(e)。
    # 当前 main 的 g_cumsum/beta 均为 BSN [B,T,HV]，A 为 [B,HV,T,64]。
    dq, dk, dv, d_beta, d_g, dh0, d_a_log, d_dt_bias = npu_chunk_gated_delta_rule_bwd(
        q_hat, k_hat, v, g_cumsum, beta, A, d_o, scale,
        q_rstd=q_rstd, k_rstd=k_rstd,
        chunk_size=chunk_size,
        layout="BSND",
        initial_state=initial_state,
        dht=None,
        use_exp2=True,  # 和前向匹配。
        use_qk_l2norm_in_kernel=args.qk_l2norm,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        state_v_first=False,
    )
    torch.npu.synchronize()
    print("BWD_DONE", flush=True)
    for name, tensor in {
        "o": o, "final_state": final_state, "g_cumsum": g_cumsum, "A": A,
        "dq": dq, "dk": dk, "dv": dv, "d_beta": d_beta, "d_g": d_g, "dh0": dh0,
    }.items():
        if not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"{name} 包含非有限值")
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}", flush=True)
    # 未启用 beta sigmoid/h 输出；门控参数梯度槽位当前保留为 None。
    assert beta_eff is None and h is None
    assert d_a_log is None and d_dt_bias is None


if __name__ == "__main__":
    main()
