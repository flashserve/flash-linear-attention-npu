"""``pre_process_bwd_kernel_merged`` 的 CPU 参考：真值 / 同精度标杆 / 离线自检。

算子语义（CP 反向状态预处理，对应上游 Triton ``fla/ops/cp/chunk_delta_h.py`` 的
``pre_process_bwd_kernel_merged``）：把一个 segment 的反向状态递推压缩成仿射摘要

    dH_start = P_r @ dH_end + E_r

输出 ``dhm[Hv, K, V+K]``：左 ``[0,V)`` 是 ``E_r``，右 ``[V,V+K)`` 是 ``P_r``。

三种口径（同一个 ``preprocess_reference`` 的 ``mode`` 参数）：

* ``"fp64"``     —— 升精度真值（ATK 双标杆的 golden 角色）；
* ``"fp32"``     —— FP32 累加、不做任何链上舍入（只用于对照，说明"把链做成全 FP32"与内核实测
                    差多远；它不是本算子的契约口径）；
* ``"contract"`` —— **同精度标杆**（ATK 双标杆的 benchmark 角色）：FP32 累加 + 按算子契约在
                    模型 dtype（bf16/fp16）上舍入。算子 kernel 的链上操作数就是模型 dtype
                    （``Q̄s``/``K̄``/``T1Bf``/``P_c``/``dH_bf``/``P_bf``），状态累加器是 FP32；
                    标杆必须复刻这一点，否则"全 FP32 标杆"会比 DUT 精确几个数量级，
                    双标杆比值失去意义。

契约口径对应的舍入点（与 kernel 的 Stage 划分一一对应）：

1. V0：``Q̄s = scale·q·2^{g}``、``K̄ = k·2^{g_last-g}`` 落模型 dtype；``decayK = 2^{g_last}`` 保持 FP32；
2. T1：``T1 = Wᵀ@K̄``（FP32 累加）落模型 dtype；
3. ``P_c = diag(decayK) - T1``（FP32 计算）落模型 dtype，作为链上 GEMM 的操作数；
4. 链：``dH ← decayK⊙dH + Q̄sᵀ@do + Wᵀ@(-dv) + (-T1)@dH_bf``、``P ← decayK⊙P + (-T1)@P_bf``，
   累加器 FP32，``dH_bf``/``P_bf`` 由上一轮的状态落模型 dtype 得到（``P`` 的初值是 ``I``）；
5. 输出 ``dhm`` 由 FP32 累加器写出。

约定：

* 输入按 ``[1, H, T, D]``（BSND）给出；一次 launch 只处理一段 ``[bos, eos)``；
* GDN 传 ``q/k`` 与标量 ``g``；KDA/GDN2 传 ``qg/kg`` 与逐 K ``gk``；两者互斥；
* ``dv`` 是 ``dv_local``（不含 ``K̄ @ dH`` 项）——该局部项已折进 ``(-T1)@dH_bf`` 的同一支路；
* GVA：``q/k`` 的 head 维是 ``Hk``，``w/do/dv/g/gk`` 是 ``Hv``，``hv`` 映射到 ``hk = hv // (Hv/Hk)``。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

# 本标杆全程是 128×128（K=V=128）量级的小矩阵运算，**必须**限制 torch 的 CPU 线程数：
# 默认线程数跟着机器核数走（实测 128 线程），每算子的线程同步开销（数十毫秒）远大于计算量本身，
# 基准耗时会比单线程慢两个数量级。单线程下同一 batched matmul 从 75 ms 降到 0.18 ms。
torch.set_num_threads(int(os.environ.get("PPFM_REF_THREADS", "1")))


_MODEL_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def model_dtype(name) -> torch.dtype:
    """case_spec 里的 dtype 名称 → torch dtype（仅支持本算子的两种模型 dtype）。"""
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower()
    if key not in _MODEL_DTYPES:
        raise ValueError(f"unsupported model dtype {name!r}")
    return _MODEL_DTYPES[key]


def _chunk_operands(qc, kc, gc, gkc, scale, m_rows, k_dim, elem, accum, contract):
    """一个 chunk 的 ``Q̄s`` / ``K̄`` / ``decayK``；``contract`` 时前两者落模型 dtype。

    注意 ``scale`` 必须**在舍入之前**乘进 ``Q̄s``：kernel 的 V0 落盘的是已经带 ``scale`` 的
    ``Q̄s = scale·Q̄``（模型 dtype 槽位），所以模型 dtype 的舍入误差是相对 ``scale·Q̄`` 的。
    把 ``scale`` 放到舍入之后再乘会让标杆少一次 0.39%（bf16 半 ulp）量级的舍入，标杆比 DUT
    更精确，双标杆的最大相对误差比例会被抬高到 7~20 倍。
    """
    if gkc is not None:
        glast = gkc[m_rows - 1]
        q_bar, k_bar, decay = qc, kc, torch.exp2(glast)
    elif gc is not None:
        glast = gc[m_rows - 1]
        q_bar = qc * torch.exp2(gc)[:, None]
        k_bar = kc * torch.exp2(glast - gc)[:, None]
        decay = torch.exp2(glast).expand(k_dim).clone()
    else:
        q_bar, k_bar = qc, kc
        decay = torch.ones(k_dim, dtype=accum)
    q_bar = q_bar * scale
    if contract:
        q_bar = q_bar.to(elem).to(accum)
        k_bar = k_bar.to(elem).to(accum)
    return q_bar, k_bar, decay


def _chunk_operands_batched(qs, ks, gs, gks, scale, m_rows, k_dim, elem, accum, contract):
    """按 head 批量化的一版 ``_chunk_operands``：``qs``/``ks`` 为 ``[Hv, M, K]``。

    逐元素运算与 ``_chunk_operands`` 完全一致（只多了 batch 维），舍入点也相同，
    因此结果不变，但把 ``Hv`` 个头合并成一次 bmm，避免逐 head 的 Python/算子开销。
    """
    if gks is not None:
        glast = gks[:, m_rows - 1, :]                      # [Hv, K]
        q_bar, k_bar, decay = qs, ks, torch.exp2(glast)
    elif gs is not None:
        glast = gs[:, m_rows - 1]                          # [Hv]
        q_bar = qs * torch.exp2(gs)[:, :, None]
        k_bar = ks * torch.exp2(glast[:, None] - gs)[:, :, None]
        decay = torch.exp2(glast)[:, None].expand(-1, k_dim).clone()
    else:
        q_bar, k_bar = qs, ks
        decay = torch.ones(qs.shape[0], k_dim, dtype=accum)
    q_bar = q_bar * scale
    if contract:
        q_bar = q_bar.to(elem).to(accum)
        k_bar = k_bar.to(elem).to(accum)
    return q_bar, k_bar, decay


def _reverse_scan_batched(q_b, k_b, w_b, do_b, dv_b, g_b, gk_b, scale, chunk_size,
                          bos, eos, hv, elem, accum, contract):
    """按 head 批量的反扫：``*_b`` 为 ``[Hv, T, *]``，返回 ``(dH_start, P_r)``。"""
    k_dim = k_b.shape[-1]
    v_dim = do_b.shape[-1]
    d_h = torch.zeros(hv, k_dim, v_dim, dtype=accum)
    p_chain = torch.eye(k_dim, dtype=accum).expand(hv, k_dim, k_dim).clone()
    chunk_num = (eos - bos + chunk_size - 1) // chunk_size

    for c in range(chunk_num - 1, -1, -1):
        s = bos + c * chunk_size
        e = min(s + chunk_size, eos)
        m_rows = e - s
        gs = None if g_b is None else g_b[:, s:e]
        gks = None if gk_b is None else gk_b[:, s:e]
        q_bar_s, k_bar, decay = _chunk_operands_batched(
            q_b[:, s:e], k_b[:, s:e], gs, gks, scale, m_rows, k_dim, elem, accum, contract)

        w_c = w_b[:, s:e]
        t1 = torch.bmm(w_c.transpose(1, 2), k_bar)
        if contract:
            t1 = t1.to(elem).to(accum)
        # P_c = diag(decay) - T1：直接把 decay 加到对角线（不用 diag_embed —— 该算子在
        # 这部分 torch 版本上对小张量也要几十毫秒，是本标杆最慢的单点）。
        p_c = torch.neg(t1)
        row = torch.arange(k_dim)
        p_c[:, row, row] += decay
        if contract:
            p_c = p_c.to(elem).to(accum)
            state_e = d_h.to(elem).to(accum)
            state_p = p_chain.to(elem).to(accum)
        else:
            state_e, state_p = d_h, p_chain

        inc_e = torch.bmm(q_bar_s.transpose(1, 2), do_b[:, s:e]) \
            + torch.bmm(w_c.transpose(1, 2), -dv_b[:, s:e]) - torch.bmm(t1, state_e)
        inc_p = -torch.bmm(t1, state_p)
        d_h = decay[:, :, None] * d_h + inc_e
        p_chain = decay[:, :, None] * p_chain + inc_p

    return d_h, p_chain


def preprocess_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    chunk_size: int = 64,
    bos: int = 0,
    eos: Optional[int] = None,
    mode: str = "contract",
) -> torch.Tensor:
    """返回 ``dhm[Hv, K, V+K]``（FP64），前 ``V`` 列 ``E_r``、后 ``K`` 列 ``P_r``。"""
    if g is not None and gk is not None:
        raise ValueError("g and gk are mutually exclusive")
    if mode not in ("fp64", "fp32", "contract"):
        raise ValueError(f"unsupported mode {mode!r}")
    b, hk, t, k_dim = q.shape
    hv = do.shape[1]
    v_dim = do.shape[3]
    if b != 1:
        raise ValueError(f"one launch handles a single segment, got B={b}")
    if hv % hk != 0:
        raise ValueError(f"Hv must be a multiple of Hk, got Hk={hk} Hv={hv}")
    if eos is None:
        eos = t
    if not 0 <= bos < eos <= t:
        raise ValueError(f"invalid window [{bos}, {eos}) for T={t}")

    accum = torch.float64 if mode == "fp64" else torch.float32
    elem = model_dtype(q.dtype)
    contract = mode == "contract"
    if mode != "fp64":
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"mode={mode} expects model-dtype inputs, got {q.dtype}")

    dhm = torch.zeros(hv, k_dim, v_dim + k_dim, dtype=torch.float64)
    # 按 head 批量执行（逐元素运算与逐 head 版本等价，结果不变，只是把 Hv 个头并成一次 bmm）。
    ns = hv // hk
    head_of_hv = torch.arange(hv) // ns
    q_b = q[0][head_of_hv].to(accum)
    k_b = k[0][head_of_hv].to(accum)
    w_b, do_b, dv_b = w[0].to(accum), do[0].to(accum), dv[0].to(accum)
    g_b = None if g is None else g[0].to(accum)
    gk_b = None if gk is None else gk[0].to(accum)
    d_h, p_chain = _reverse_scan_batched(q_b, k_b, w_b, do_b, dv_b, g_b, gk_b, scale,
                                         chunk_size, bos, eos, hv, elem, accum, contract)
    dhm[:, :, :v_dim] = d_h.to(torch.float64)
    dhm[:, :, v_dim:] = p_chain.to(torch.float64)
    return dhm


def _self_test() -> None:
    """离线自检：非零 ``dht`` 时校验仿射恒等式 ``dh0 == P_r @ dht + E_r``。"""
    torch.manual_seed(0)
    for mode in ("fp64", "fp32", "contract"):
        for gate in ("none", "g", "gk"):
            hk, hv, t, k_dim, v_dim, cs = 2, 4, 200, 128, 128, 64
            q = (torch.randn(1, hk, t, k_dim) * 0.05).to(torch.bfloat16)
            k = (torch.randn(1, hk, t, k_dim) * 0.05).to(torch.bfloat16)
            w = (torch.randn(1, hv, t, k_dim) * 0.05).to(torch.bfloat16)
            do = (torch.randn(1, hv, t, v_dim) * 0.05).to(torch.bfloat16)
            dv = (torch.randn(1, hv, t, v_dim) * 0.05).to(torch.bfloat16)
            g = gk = None
            if gate == "g":
                g = torch.cumsum(-(torch.rand(1, hv, t) * 0.01 + 0.001), dim=-1).to(torch.bfloat16)
            elif gate == "gk":
                gk = torch.cumsum(-(torch.rand(1, hv, t, k_dim) * 0.01 + 0.001), dim=-2).to(torch.bfloat16)
            dhm = preprocess_reference(q, k, w, do, dv, g, gk, k_dim ** -0.5, cs, mode=mode)
            e_r, p_r = dhm[:, :, :v_dim], dhm[:, :, v_dim:]
            dht = torch.randn(hv, k_dim, v_dim, dtype=torch.float64)
            lhs = p_r @ dht + e_r
            # 直接反扫（非零末端状态）与仿射摘要必须一致。
            rhs = torch.zeros_like(lhs)
            for hv_idx in range(hv):
                d_h = dht[hv_idx]
                nchunk = (t + cs - 1) // cs
                for c in range(nchunk - 1, -1, -1):
                    s0 = c * cs
                    e0 = min(s0 + cs, t)
                    m = e0 - s0
                    gc = None if g is None else g[0, hv_idx, s0:e0]
                    gkc = None if gk is None else gk[0, hv_idx, s0:e0]
                    accum = torch.float64 if mode == "fp64" else torch.float32
                    elem = torch.bfloat16
                    contract = mode == "contract"
                    qs, kb, decay = _chunk_operands(
                        q[0, hv_idx // (hv // hk), s0:e0].to(accum),
                        k[0, hv_idx // (hv // hk), s0:e0].to(accum),
                        None if gc is None else gc.to(accum),
                        None if gkc is None else gkc.to(accum),
                        k_dim ** -0.5, m, k_dim, elem, accum, contract)
                    d_h = d_h.to(accum)
                    d_v_pre = kb @ d_h
                    d_v_hat = -(d_v_pre + dv[0, hv_idx, s0:e0].to(accum))
                    inc = qs.transpose(0, 1) @ do[0, hv_idx, s0:e0].to(accum) \
                        + w[0, hv_idx, s0:e0].to(accum).transpose(0, 1) @ d_v_hat
                    d_h = decay[:, None] * d_h + inc
                rhs[hv_idx] = d_h.to(torch.float64)
            err = float((lhs - rhs).abs().max())
            scale_ref = float(rhs.abs().max().clamp_min(1e-12))
            print(f"mode={mode:8s} gate={gate:5s} affine_rel_err={err / scale_ref:.3e}")


if __name__ == "__main__":
    _self_test()
