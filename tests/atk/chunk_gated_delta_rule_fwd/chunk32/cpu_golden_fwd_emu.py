"""CPU golden that emulates the FLA Triton GDN chunk forward step by step (dtype casts included).

Unlike ``cpu_golden_fwd`` (pure fp32), every place where the CUDA kernels round to
bf16 or keep fp32 is reproduced, so the result tracks the GPU output up to fp32
accumulation order and ``exp2`` ulp differences.

Kernel chain for ``chunk_size`` 16 / 32 (``chunk_gated_delta_rule_fwd``)::

    chunk_local_cumsum      G = cumsum_chunk(g) * RCP_LN2                          fp32
    chunk_scaled_dot_kkt    A = tril_strict((k k^T) * exp2(G_i - G_j) * beta_i)     fp32
    solve_tril              Ai = (I + A)^-1  (16x16 forward substitution, fp32 merge)  -> bf16
    recompute_w_u           u = Ai @ bf16(v * beta)                                  -> bf16
                            w = Ai @ bf16(bf16(k * beta) * exp2(G))                  -> bf16
    fwd_h (per chunk)       h_c   = bf16(S)
                            vv    = u_c - w_c @ h_c                    (fp32), v_new = bf16(vv)
                            S     = S * exp2(G_last) + k_c^T @ bf16(vv * exp2(G_last - G))
    fwd_o (per chunk)       o_c   = bf16(scale * (q_c @ h_c) * exp2(G)
                                      + scale * bf16(tril(q_c k_c^T * exp2(G_i - G_j))) @ v_new)

All bf16 ``tl.dot`` calls are bf16 x bf16 -> fp32 accumulate. On CPU the operands are
rounded to bf16, summed in fp64 and rounded toward zero to fp32 (``DOT_MODE``), which
tracks tensor-core truncation better than a plain fp32 matmul. The fp32 dots inside
``solve_tril`` use ``input_precision='ieee'`` and stay plain fp32. Triton's bf16
``tl.dot`` still uses tensor-core MMA, whose partial-sum order matches neither an exact
fp32 sum nor cuBLAS bf16 GEMM, so a few outputs can differ by one bf16 ulp.
``chunk_size=64`` uses the fused kkt+solve kernel and is not emulated here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cpu_golden import RCP_LN2, _pad_time, _repeat_k_for_gva
    from cpu_golden_fwd import _apply_qk_beta, _to_chunks
    from layout import check_cu_seqlens, from_fla, normalize_layout, to_fla
else:
    from .cpu_golden import RCP_LN2, _pad_time, _repeat_k_for_gva
    from .cpu_golden_fwd import _apply_qk_beta, _to_chunks
    from .layout import check_cu_seqlens, from_fla, normalize_layout, to_fla

EMU_CHUNK_SIZES = (16, 32)
# Emulated tl.dot accumulation: "fp32" (plain fp32 matmul), "fp64" (exact-ish sum, RNE to fp32),
# "fp64_rz" (exact-ish sum, round-toward-zero to fp32, closer to tensor-core truncation).
DOT_MODE = "fp64_rz"


def _round_to_fp32_rz(x: torch.Tensor) -> torch.Tensor:
    y = x.float()
    overshoot = y.double().abs() > x.abs()
    return torch.where(overshoot, torch.nextafter(y, torch.zeros_like(y)), y)


TC_GROUP = 8
TC_BITS = 24


def _tc_dot(a: torch.Tensor, b: torch.Tensor, acc: torch.Tensor | None) -> torch.Tensor:
    """Tensor-core block FMA model: per group of ``TC_GROUP`` products plus the fp32 accumulator,
    exact products are aligned to the largest exponent, truncated to ``TC_BITS`` bits, summed,
    and the sum is rounded toward zero to fp32. Groups are chained sequentially over K."""
    k_dim = a.shape[-1]
    out_shape = (*torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]), a.shape[-2], b.shape[-1])
    c = torch.zeros(out_shape, dtype=torch.float64) if acc is None else acc.double().expand(out_shape)
    for k0 in range(0, k_dim, TC_GROUP):
        p = a[..., :, k0:k0 + TC_GROUP, None].double() * b[..., None, k0:k0 + TC_GROUP, :].double()
        terms = torch.cat([c.unsqueeze(-2), p], dim=-2)
        _m, e = torch.frexp(terms.abs().amax(dim=-2, keepdim=True))
        quantum = torch.ldexp(torch.ones_like(_m), e - TC_BITS)
        c = _round_to_fp32_rz((torch.trunc(terms / quantum) * quantum).sum(dim=-2)).double()
    return c.float()


def _dot(a: torch.Tensor, b: torch.Tensor, acc: torch.Tensor | None = None) -> torch.Tensor:
    """Emulated ``tl.dot(a, b, acc)`` (bf16 operands held in fp32, fp32 accumulator)."""
    if DOT_MODE == "tc":
        return _tc_dot(a, b, acc)
    if DOT_MODE == "fp32":
        out = a @ b
    else:
        out = a.double() @ b.double()
        out = out.float() if DOT_MODE == "fp64" else _round_to_fp32_rz(out)
    return out if acc is None else acc + out


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 (RNE, same as Triton ``.to(bf16)``) and keep computing in fp32."""
    return x.to(torch.bfloat16).float()


def _from_chunks(x: torch.Tensor, t: int) -> torch.Tensor:
    """``[B, HV, NT, BT, ...]`` -> ``[B, T, HV, ...]``."""
    b, hv, nt, bt = x.shape[:4]
    perm = (0, 2, 3, 1, *range(4, x.ndim))
    return x.permute(*perm).reshape(b, nt * bt, hv, *x.shape[4:])[:, :t]


def _solve_tril_16(a: torch.Tensor) -> torch.Tensor:
    """``solve_tril_16x16`` forward substitution on strictly-lower ``a`` ``[..., 16, 16]`` (fp32)."""
    ai = -torch.tril(a, diagonal=-1)
    for i in range(2, 16):
        b_a = -a[..., i, :].clone()
        b_a[..., i:] = 0.0
        b_a = b_a + (b_a.unsqueeze(-1) * ai).sum(dim=-2)
        ai[..., i, :] = b_a
    return ai + torch.eye(16, dtype=a.dtype)


def emu_solve_tril(a: torch.Tensor) -> torch.Tensor:
    """``solve_tril`` for BT=16/32 on ``[..., BT, BT]`` fp32; returns ``(I + A)^-1`` rounded to bf16."""
    bt = a.shape[-1]
    if bt == 16:
        return _bf16(_solve_tril_16(a))
    if bt != 32:
        raise NotImplementedError(f"emulated solve_tril supports BT=16/32, got {bt}")
    ai11 = _solve_tril_16(a[..., :16, :16])
    ai22 = _solve_tril_16(a[..., 16:, 16:])
    ai21 = -((ai22 @ a[..., 16:, :16]) @ ai11)
    out = torch.zeros_like(a)
    out[..., :16, :16] = ai11
    out[..., 16:, 16:] = ai22
    out[..., 16:, :16] = ai21
    return _bf16(out)


def _fwd_single_emu(q, k, v, g, beta, scale, initial_state, chunk_size, keep_stages, state_v_first):
    b, t, _h, k_dim = k.shape
    hv, v_dim = v.shape[2], v.shape[3]
    bt = chunk_size
    pad = (bt - t % bt) % bt
    nt = (t + pad) // bt

    def chunks(x, repeat=False):
        x = x.float()
        if repeat:
            x = _repeat_k_for_gva(x, hv)
        return _to_chunks(_pad_time(x, pad), bt)

    q_c, k_c, v_c = chunks(q, True), chunks(k, True), chunks(v)
    beta_c = chunks(beta)                                       # [B, HV, NT, BT]
    valid = (torch.arange(nt * bt) < t).reshape(nt, bt)         # [NT, BT]

    # chunk_local_cumsum: fp32 cumsum inside each chunk, then * RCP_LN2
    g_c = chunks(g).cumsum(dim=-1) * RCP_LN2

    strict = torch.tril(torch.ones(bt, bt, dtype=torch.bool), diagonal=-1)
    causal = torch.tril(torch.ones(bt, bt, dtype=torch.bool))
    pair_valid = valid[:, :, None] & valid[:, None, :]          # [NT, BT, BT]
    gdiff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)

    # chunk_scaled_dot_kkt_fwd: fp32 output
    a = _dot(k_c, k_c.transpose(-1, -2)) * torch.exp2(gdiff.masked_fill(~strict, 0.0))
    a = (a * beta_c.unsqueeze(-1)).masked_fill(~(strict & pair_valid), 0.0)
    ai = emu_solve_tril(a)

    # recompute_w_u_fwd
    u_c = _bf16(_dot(ai, _bf16(v_c * beta_c.unsqueeze(-1))))
    kb = _bf16(_bf16(k_c * beta_c.unsqueeze(-1)) * torch.exp2(g_c).unsqueeze(-1))
    w_c = _bf16(_dot(ai, kb))

    # chunk_gated_delta_rule_fwd_h + chunk_fwd_o
    if initial_state is None:
        state = torch.zeros(b, hv, k_dim, v_dim, dtype=torch.float32)
    else:
        # Kernel layout is [V, K] when state_v_first; the recurrence itself is [K, V].
        state = initial_state.float()
        if state_v_first:
            state = state.transpose(-1, -2)
        state = state.contiguous()
    o_c = torch.empty(b, hv, nt, bt, v_dim, dtype=torch.float32)
    h_tail = (v_dim, k_dim) if state_v_first else (k_dim, v_dim)
    h_all = torch.empty(b, hv, nt, *h_tail, dtype=torch.bfloat16) if keep_stages else None
    vnew_all = torch.empty(b, hv, nt, bt, v_dim, dtype=torch.float32) if keep_stages else None
    for i_t in range(nt):
        n_valid = min(bt, t - i_t * bt)
        m_t = valid[i_t]
        q_i, k_i, g_i = q_c[:, :, i_t], k_c[:, :, i_t], g_c[:, :, i_t]

        h_i = _bf16(state)
        vv = u_c[:, :, i_t] - _dot(w_c[:, :, i_t], h_i)
        v_new = _bf16(vv)

        g_last = g_i[..., n_valid - 1]
        decay = torch.exp2((g_last.unsqueeze(-1) - g_i).masked_fill(~m_t, 0.0)) * m_t
        state = state * torch.exp2(g_last)[..., None, None]
        state = _dot(k_i.transpose(-1, -2), _bf16(vv * decay.unsqueeze(-1)), state)

        o_inter = _dot(q_i, h_i) * torch.exp2(g_i).unsqueeze(-1)
        attn = _dot(q_i, k_i.transpose(-1, -2)) * torch.exp2(gdiff[:, :, i_t].masked_fill(~causal, 0.0))
        attn = attn.masked_fill(~(causal & pair_valid[i_t]), 0.0)
        o_c[:, :, i_t] = _bf16(o_inter * scale + _dot(_bf16(attn), v_new) * scale)

        if keep_stages:
            stored_h = h_i.transpose(-1, -2) if state_v_first else h_i
            h_all[:, :, i_t] = stored_h.to(torch.bfloat16)
            vnew_all[:, :, i_t] = v_new

    o = _from_chunks(o_c, t).to(torch.bfloat16)
    if state_v_first:
        state = state.transpose(-1, -2).contiguous()
    stages = None
    if keep_stages:
        stages = {
            "g_cumsum": _from_chunks(g_c, t),
            "A": _from_chunks(ai, t).to(torch.bfloat16),
            "w": _from_chunks(w_c, t).to(torch.bfloat16),
            "u": _from_chunks(u_c, t).to(torch.bfloat16),
            "v_new": _from_chunks(vnew_all, t).to(torch.bfloat16),
            "h": h_all.transpose(1, 2).contiguous(),            # [B, NT, HV, K, V] or [B, NT, HV, V, K]
        }
    return o, state, stages


def cpu_chunk_gated_delta_rule_fwd_emu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 32,
    layout: str = "bsnd",
    use_qk_l2norm_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    return_stages: bool = False,
):
    """GPU-emulating CPU golden. Returns ``(o, final_state)`` or ``(o, final_state, stages)``.

    Defaults match chunk.py 282-288: raw ``q``/``k`` are L2-normalized (rounded to the
    input dtype) and ``beta`` logits go through sigmoid in fp32. ``allow_neg_eigval``
    uses ``2 * sigmoid(beta)``. ``state_v_first`` stores ``initial_state``, ``h`` and
    ``final_state`` as ``[V, K]`` instead of ``[K, V]``.
    ``o`` is bf16 in ``layout``; ``final_state`` is fp32.
    ``stages`` (FLA ``[B, T, HV, ...]`` layout, packed for varlen) holds the emulated
    ``g_cumsum / A / w / u / v_new / h`` for step-by-step comparison with the kernels.
    """
    if chunk_size not in EMU_CHUNK_SIZES:
        raise NotImplementedError(f"GPU emulation supports chunk_size in {EMU_CHUNK_SIZES}, got {chunk_size}")
    layout = normalize_layout(layout)
    q, k, v, g, beta = (to_fla(x, layout) for x in (q, k, v, g, beta))
    q, k, beta = _apply_qk_beta(
        q, k, beta,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
    )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if cu_seqlens is None:
        o, final_state, stages = _fwd_single_emu(
            q, k, v, g, beta, scale, initial_state, chunk_size, return_stages, state_v_first,
        )
    else:
        check_cu_seqlens(q, cu_seqlens)
        outs, states, stage_list = [], [], []
        for i in range(len(cu_seqlens) - 1):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            h0 = None if initial_state is None else initial_state[i : i + 1]
            o_i, ht_i, st_i = _fwd_single_emu(
                *(x[:, s:e] for x in (q, k, v, g, beta)), scale, h0, chunk_size, return_stages, state_v_first,
            )
            outs.append(o_i)
            states.append(ht_i)
            stage_list.append(st_i)
        o = torch.cat(outs, dim=1)
        final_state = torch.cat(states, dim=0)
        stages = None
        if return_stages:
            stages = {key: torch.cat([st[key] for st in stage_list], dim=1) for key in stage_list[0]}

    o = from_fla(o, layout)
    final_state = final_state if output_final_state else None
    if return_stages:
        return o, final_state, stages
    return o, final_state
