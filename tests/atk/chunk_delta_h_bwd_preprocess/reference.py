"""CPU reference for ``chunk_delta_h_bwd_preprocess`` (CP backward state preprocess).

算子把本 rank 边界序列的反向状态递推压缩成仿射摘要::

    dH_start = P_r @ dH_end + E_r

本文件提供两个口径：

1. ``preprocess_reference`` —— 末端 ``dH = 0`` 时直接算出 ``dhm = [E_r | P_r]``，作为算子输出的标杆；
2. ``dh_scan_direct`` —— 给定任意末端 ``dht`` 走完整反扫，得到真实 ``dh0``，用于验证仿射恒等式
   ``dh0 == P_r @ dht + E_r``。只测 ``dht = 0`` 无法验证 ``P_r``。

约定：

* 输入按 ``[B, H, T, D]``（BSND）给出，``B`` 必须为 1（一次 launch 只处理一个 segment）；
* GDN 传原始 ``q/k`` 与标量 ``g``；KDA/GDN2 传 ``qg/kg`` 与逐 K ``gk``；两者互斥；
* ``dv`` 必须是 ``dv_local``（不含 ``K̄ @ dH`` 项）；
* 全部计算用 FP32，与算子内部链一致。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _chunk_operands(qc, kc, gc, gkc, scale, k_dim, m_rows):
    """构造一个 chunk 的 Q̄s、K̄、decayK。

    kc/gkc 只包含有效行 ``[0, m_rows)``；当调用方按 chunk tile 切出 ``[BT, ...]`` 的padding 时，
    无效行必须已经清零（算子 V0 的 ``[M, BT)`` 写零规则）。
    """
    if gkc is not None:
        glast = gkc[m_rows - 1]
        k_bar = kc
        q_bar = qc
        decay_k = torch.exp2(glast)
    elif gc is not None:
        glast = gc[m_rows - 1]
        k_bar = kc * torch.exp2(glast - gc)[:, None]
        q_bar = qc * torch.exp2(gc)[:, None]
        decay_k = torch.exp2(glast).expand(k_dim)
    else:
        k_bar = kc
        q_bar = qc
        decay_k = torch.ones(k_dim, dtype=torch.float32)
    return scale * q_bar, k_bar, decay_k


def _reverse_scan(q, k, w, do, dv, g, gk, scale, chunk_size, bos, eos, dht, hv):
    """反扫本 head 的 chunk，返回 (dH_start, P_r)。"""
    qh = q[0, hv[0]].float()
    kh = k[0, hv[1]].float()
    wh = w[0, hv[2]].float()
    doh = do[0, hv[2]].float()
    dvh = dv[0, hv[2]].float()
    gh = g[0, hv[2]].float() if g is not None else None
    gkh = gk[0, hv[2]].float() if gk is not None else None

    k_dim = kh.shape[-1]
    v_dim = doh.shape[-1]
    d_h = torch.zeros(k_dim, v_dim, dtype=torch.float32) if dht is None else dht.float().clone()
    p_chain = torch.eye(k_dim, dtype=torch.float32)
    chunk_num = (eos - bos + chunk_size - 1) // chunk_size

    for c in range(chunk_num - 1, -1, -1):
        s = bos + c * chunk_size
        e = min(s + chunk_size, eos)
        m_rows = e - s
        gc = gh[s:e] if gh is not None else None
        gkc = gkh[s:e] if gkh is not None else None
        q_bar_s, k_bar, decay_k = _chunk_operands(qh[s:e], kh[s:e], gc, gkc, scale, k_dim, m_rows)

        d_v_pre = k_bar @ d_h
        d_v_hat = -(d_v_pre + dvh[s:e])
        inc = q_bar_s.transpose(0, 1) @ doh[s:e] + wh[s:e].transpose(0, 1) @ d_v_hat

        t1 = wh[s:e].transpose(0, 1) @ k_bar
        p_c = torch.diag(decay_k) - t1

        d_h = decay_k[:, None] * d_h + inc
        p_chain = p_c @ p_chain

    return d_h, p_chain


def _resolve_head_ids(hv_idx: int, hv_total: int, hk_total: int):
    """GVA：Hv 可以是 Hk 的整数倍，hk = hv // (Hv / Hk)。返回 (q 头号, k 头号, v 头号)。

    注意 q/k 的头维是 Hk，w/do/dv/g 的头维是 Hv，两者不能混用同一个下标。
    """
    ns = hv_total // hk_total
    hk_idx = hv_idx // ns
    return (hk_idx, hk_idx, hv_idx)


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
) -> torch.Tensor:
    """返回 ``dhm``：``[Hv, K, V+K]`` FP32，前 V 列为 E_r，后 K 列为 P_r。"""
    if g is not None and gk is not None:
        raise ValueError("g and gk are mutually exclusive")
    b, hk, t, k_dim = q.shape
    hv = do.shape[1]
    v_dim = do.shape[3]
    if b != 1:
        raise ValueError(f"one launch handles a single segment, got B={b}")
    if hv % hk != 0:
        raise ValueError(f"Hv must be a multiple of Hk, got Hk={hk} Hv={hv}")
    if eos is None:
        eos = t

    ns = hv // hk
    dhm = torch.zeros(hv, k_dim, v_dim + k_dim, dtype=torch.float32)
    for idx in range(hv):
        head_ids = _resolve_head_ids(idx, hv, hk)
        d_h, p_chain = _reverse_scan(q, k, w, do, dv, g, gk, scale, chunk_size, bos, eos, None, head_ids)
        dhm[idx, :, :v_dim] = d_h
        dhm[idx, :, v_dim:] = p_chain
    return dhm


def dh_scan_direct(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    dht: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    chunk_size: int = 64,
    bos: int = 0,
    eos: Optional[int] = None,
) -> torch.Tensor:
    """给定末端 ``dht`` [Hv, K, V] 走完整反扫，返回真实 ``dh0`` [Hv, K, V]。"""
    b, hk, t, k_dim = q.shape
    hv = do.shape[1]
    if b != 1:
        raise ValueError(f"one launch handles a single segment, got B={b}")
    if eos is None:
        eos = t
    dh0 = torch.zeros(hv, k_dim, do.shape[3], dtype=torch.float32)
    for idx in range(hv):
        head_ids = _resolve_head_ids(idx, hv, hk)
        d_h, _ = _reverse_scan(q, k, w, do, dv, g, gk, scale, chunk_size, bos, eos, dht[idx], head_ids)
        dh0[idx] = d_h
    return dh0


def check_affine(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    dht: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    chunk_size: int = 64,
    bos: int = 0,
    eos: Optional[int] = None,
) -> Tuple[float, float]:
    """校验 ``dh0 == P_r @ dht + E_r``；返回 (绝对误差, 相对误差)。"""
    dhm = preprocess_reference(q, k, w, do, dv, g, gk, scale, chunk_size, bos, eos)
    v_dim = do.shape[3]
    e_r = dhm[:, :, :v_dim]
    p_r = dhm[:, :, v_dim:]
    dh0_direct = dh_scan_direct(q, k, w, do, dv, dht, g, gk, scale, chunk_size, bos, eos)
    dh0_affine = torch.einsum("hkj,hjv->hkv", p_r, dht.float()) + e_r
    diff = (dh0_direct - dh0_affine).abs()
    denom = dh0_direct.abs().max().clamp_min(1e-12)
    return float(diff.max()), float(diff.max() / denom)


def _main() -> None:
    torch.manual_seed(0)
    for gate in ("none", "g", "gk"):
        hk, hv, t, k_dim, v_dim, cs = 4, 8, 200, 64, 64, 64
        q = torch.randn(1, hk, t, k_dim)
        k = torch.randn(1, hk, t, k_dim)
        w = torch.randn(1, hv, t, k_dim) * 0.1
        do = torch.randn(1, hv, t, v_dim)
        dv = torch.randn(1, hv, t, v_dim) * 0.1
        g = gk = None
        if gate == "g":
            g = torch.cumsum(-torch.rand(1, hv, t) * 0.05, dim=-1)
        elif gate == "gk":
            gk = torch.cumsum(-torch.rand(1, hv, t, k_dim) * 0.05, dim=-2)
        dht = torch.randn(hv, k_dim, v_dim)
        abs_err, rel_err = check_affine(q, k, w, do, dv, dht, g, gk, scale=k_dim ** -0.5, chunk_size=cs)
        dhm = preprocess_reference(q, k, w, do, dv, g, gk, k_dim ** -0.5, cs)
        print(f"gate={gate:5s} dhm={tuple(dhm.shape)} affine_abs_err={abs_err:.3e} rel_err={rel_err:.3e}")


if __name__ == "__main__":
    _main()
