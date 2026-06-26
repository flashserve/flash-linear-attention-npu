# Copyright (c) 2026 Tianjin University, Ltd.
#
# Functional KDA operators for the fla_npu package.  These implementations are
# intentionally written as composite PyTorch/NPU operators so the public
# torch.ops.npu ABI and numerical tests can be landed before the Ascend C
# monolithic kernels are substituted underneath.

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


_KDA_LIBRARY = None
_KDA_REGISTERED = False


def _as_int_list(value: Optional[Sequence[int]]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]


def _chunk_spans(
    batch: int,
    total_t: int,
    chunk_size: int,
    cu_seqlens: Optional[Sequence[int]],
    chunk_indices: Optional[Sequence[int]],
) -> List[Tuple[int, int, int, int, int]]:
    cu = _as_int_list(cu_seqlens)
    indices = _as_int_list(chunk_indices)

    if cu is None:
        spans: List[Tuple[int, int, int, int, int]] = []
        for b in range(batch):
            for chunk_idx, start in enumerate(range(0, total_t, chunk_size)):
                spans.append((b, b, chunk_idx, start, min(start + chunk_size, total_t)))
        return spans

    if batch != 1:
        raise RuntimeError("npu_chunk_kda_* varlen mode expects flattened batch B=1.")
    if len(cu) < 2 or cu[0] != 0:
        raise RuntimeError("cu_seqlens must start with 0 and contain at least two elements.")

    spans = []
    if indices is not None:
        if len(indices) % 2 != 0:
            raise RuntimeError("chunk_indices must contain flattened (seq_id, chunk_id) pairs.")
        for pair_idx in range(0, len(indices), 2):
            seq_idx = indices[pair_idx]
            chunk_idx = indices[pair_idx + 1]
            if seq_idx < 0 or seq_idx + 1 >= len(cu):
                raise RuntimeError("chunk_indices contains an invalid sequence id.")
            start = cu[seq_idx] + chunk_idx * chunk_size
            end = min(start + chunk_size, cu[seq_idx + 1])
            spans.append((0, seq_idx, pair_idx // 2, start, end))
        return spans

    flat_chunk_idx = 0
    for seq_idx in range(len(cu) - 1):
        seq_start, seq_end = cu[seq_idx], cu[seq_idx + 1]
        if seq_end <= seq_start:
            raise RuntimeError("cu_seqlens must be strictly increasing.")
        for start in range(seq_start, seq_end, chunk_size):
            spans.append((0, seq_idx, flat_chunk_idx, start, min(start + chunk_size, seq_end)))
            flat_chunk_idx += 1
    return spans


def _lower_inverse(mat: torch.Tensor) -> torch.Tensor:
    size = mat.shape[-1]
    eye = torch.eye(size, device=mat.device, dtype=torch.float32)
    lhs = eye + torch.tril(mat.to(torch.float32), diagonal=-1)
    return torch.linalg.solve_triangular(lhs, eye, upper=False)


def _empty_like_optional(base: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), device=base.device, dtype=base.dtype)


def _chunk_kda_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[Sequence[int]],
    chunk_indices: Optional[Sequence[int]],
    return_intermediate: bool,
    safe_gate: bool,
    transpose_state_layout: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if safe_gate:
        raise RuntimeError("npu_chunk_kda_fwd: safe_gate=True is not implemented in the composite core.")
    if transpose_state_layout:
        raise RuntimeError("npu_chunk_kda_fwd: transpose_state_layout=True is not implemented.")
    if chunk_size not in (64, 128):
        raise RuntimeError("npu_chunk_kda_fwd: chunk_size must be 64 or 128.")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or gk.dim() != 4 or beta.dim() != 3:
        raise RuntimeError("npu_chunk_kda_fwd: expected q/k/v/gk/beta head-first tensors.")
    if q.shape != k.shape:
        raise RuntimeError("npu_chunk_kda_fwd: q and k must have the same shape.")

    batch, hq, total_t, kdim = q.shape
    b_v, hv, t_v, vdim = v.shape
    if (b_v, t_v) != (batch, total_t):
        raise RuntimeError("npu_chunk_kda_fwd: v shape must match q/k batch and sequence dimensions.")
    if gk.shape != (batch, hv, total_t, kdim):
        raise RuntimeError("npu_chunk_kda_fwd: gk shape must be [B, HV, T, K].")
    if beta.shape != (batch, hv, total_t):
        raise RuntimeError("npu_chunk_kda_fwd: beta shape must be [B, HV, T].")
    if hv % hq != 0:
        raise RuntimeError("npu_chunk_kda_fwd: HV must be divisible by H.")

    spans = _chunk_spans(batch, total_t, chunk_size, cu_seqlens, chunk_indices)
    nt = len(spans) if cu_seqlens is not None else (total_t + chunk_size - 1) // chunk_size
    num_seq = batch if cu_seqlens is None else len(_as_int_list(cu_seqlens)) - 1
    group = hv // hq
    out_dtype = v.dtype

    o = torch.zeros((batch, hv, total_t, vdim), device=v.device, dtype=out_dtype)
    aqk = torch.zeros((batch, hv, total_t, chunk_size), device=q.device, dtype=q.dtype)
    akk = torch.zeros((batch, hv, total_t, chunk_size), device=q.device, dtype=q.dtype)
    w = torch.zeros((batch, hv, total_t, kdim), device=q.device, dtype=q.dtype)
    u = torch.zeros((batch, hv, total_t, vdim), device=v.device, dtype=v.dtype)
    qg = torch.zeros((batch, hv, total_t, kdim), device=q.device, dtype=q.dtype)
    kg = torch.zeros((batch, hv, total_t, kdim), device=k.device, dtype=k.dtype)
    v_new = torch.zeros_like(v)
    h_out = torch.zeros((batch, hv, nt, kdim, vdim), device=q.device, dtype=q.dtype)

    if initial_state is None:
        states = [
            [torch.zeros((kdim, vdim), device=q.device, dtype=torch.float32) for _ in range(hv)]
            for _ in range(num_seq)
        ]
    else:
        expected = (num_seq, hv, kdim, vdim)
        if tuple(initial_state.shape) != expected:
            raise RuntimeError(f"npu_chunk_kda_fwd: initial_state shape must be {expected}.")
        states = [
            [initial_state[seq, ihv].to(torch.float32) for ihv in range(hv)]
            for seq in range(num_seq)
        ]

    for b, state_idx, chunk_idx, start, end in spans:
        cur_t = end - start
        if cur_t <= 0:
            continue
        local_cols = slice(0, cur_t)
        for ihv in range(hv):
            ih = ihv // group
            q_blk = q[b, ih, start:end].to(torch.float32)
            k_blk = k[b, ih, start:end].to(torch.float32)
            v_blk = v[b, ihv, start:end].to(torch.float32)
            g_blk = gk[b, ihv, start:end].to(torch.float32)
            beta_blk = beta[b, ihv, start:end].to(torch.float32)

            exp_rel = torch.exp2(g_blk[:, None, :] - g_blk[None, :, :])
            qk = (q_blk[:, None, :] * k_blk[None, :, :] * exp_rel).sum(dim=-1) * float(scale)
            kk = (k_blk[:, None, :] * k_blk[None, :, :] * exp_rel).sum(dim=-1)
            tril_qk = torch.tril(qk, diagonal=0)
            tril_kk = torch.tril(kk * beta_blk[:, None], diagonal=-1)
            inv_akk = _lower_inverse(tril_kk)

            k_beta_g = k_blk * beta_blk[:, None] * torch.exp2(g_blk)
            v_beta = v_blk * beta_blk[:, None]
            w_blk = inv_akk @ k_beta_g
            u_blk = inv_akk @ v_beta

            last_g = g_blk[cur_t - 1]
            qg_blk = q_blk * torch.exp2(g_blk)
            kg_blk = k_blk * torch.exp2(last_g[None, :] - g_blk)

            h_prev = states[state_idx][ihv]
            v_new_blk = u_blk - w_blk @ h_prev
            h_next = torch.exp2(last_g)[:, None] * h_prev + kg_blk.transpose(0, 1) @ v_new_blk
            states[state_idx][ihv] = h_next

            o_inter = qg_blk @ h_prev * float(scale)
            o_local = tril_qk @ v_new_blk

            o[b, ihv, start:end].copy_((o_inter + o_local).to(out_dtype))
            aqk[b, ihv, start:end, local_cols].copy_(tril_qk.to(aqk.dtype))
            akk[b, ihv, start:end, local_cols].copy_(inv_akk.to(akk.dtype))
            h_out[b, ihv, chunk_idx].copy_(h_prev.to(h_out.dtype))
            if return_intermediate:
                w[b, ihv, start:end].copy_(w_blk.to(w.dtype))
                u[b, ihv, start:end].copy_(u_blk.to(u.dtype))
                qg[b, ihv, start:end].copy_(qg_blk.to(qg.dtype))
                kg[b, ihv, start:end].copy_(kg_blk.to(kg.dtype))
                v_new[b, ihv, start:end].copy_(v_new_blk.to(v_new.dtype))

    if output_final_state:
        final_state = torch.stack([torch.stack(seq_states, dim=0) for seq_states in states], dim=0)
        final_state = final_state.to(initial_state.dtype if initial_state is not None else torch.float32)
    else:
        final_state = _empty_like_optional(q)

    if not return_intermediate:
        empty = _empty_like_optional(q)
        return o, final_state, aqk, akk, empty, empty, empty, empty, empty, h_out
    return o, final_state, aqk, akk, w, u, qg, kg, v_new, h_out


def _chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
    *,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[Sequence[int]] = None,
    chunk_indices: Optional[Sequence[int]] = None,
    return_intermediate: bool = False,
    safe_gate: bool = False,
    transpose_state_layout: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _chunk_kda_forward_impl(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        chunk_size,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        return_intermediate,
        safe_gate,
        transpose_state_layout,
    )


def _grad_or_zeros(grad: Optional[torch.Tensor], ref: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if grad is None:
        return torch.zeros_like(ref, dtype=dtype or ref.dtype)
    if dtype is not None:
        return grad.to(dtype)
    return grad


def _chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    aqk: torch.Tensor,
    akk: torch.Tensor,
    d_o: torch.Tensor,
    scale: float,
    chunk_size: int,
    *,
    initial_state: Optional[torch.Tensor] = None,
    dht: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[Sequence[int]] = None,
    chunk_indices: Optional[Sequence[int]] = None,
    safe_gate: bool = False,
    transpose_state_layout: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del aqk, akk

    q_req = q.detach().clone().requires_grad_(True)
    k_req = k.detach().clone().requires_grad_(True)
    v_req = v.detach().clone().requires_grad_(True)
    gk_req = gk.detach().clone().requires_grad_(True)
    beta_req = beta.detach().clone().requires_grad_(True)
    init_req = None
    if initial_state is not None:
        init_req = initial_state.detach().clone().requires_grad_(True)

    with torch.enable_grad():
        out = _chunk_kda_forward_impl(
            q_req,
            k_req,
            v_req,
            gk_req,
            beta_req,
            scale,
            chunk_size,
            init_req,
            dht is not None,
            cu_seqlens,
            chunk_indices,
            False,
            safe_gate,
            transpose_state_layout,
        )
        targets = [out[0]]
        grad_outputs = [d_o]
        if dht is not None:
            targets.append(out[1])
            grad_outputs.append(dht)
        grad_inputs = [q_req, k_req, v_req, gk_req, beta_req]
        if init_req is not None:
            grad_inputs.append(init_req)
        grads = torch.autograd.grad(
            targets,
            grad_inputs,
            grad_outputs=grad_outputs,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )

    dq = _grad_or_zeros(grads[0], q)
    dk = _grad_or_zeros(grads[1], k)
    dv = _grad_or_zeros(grads[2], v)
    dbeta = _grad_or_zeros(grads[4], beta, torch.float32)
    dgk = _grad_or_zeros(grads[3], gk, torch.float32)
    if init_req is not None:
        dh0 = _grad_or_zeros(grads[5], initial_state)
    else:
        dh0 = _empty_like_optional(q)
    return dq, dk, dv, dbeta, dgk, dh0


def register_kda_ops() -> None:
    global _KDA_LIBRARY, _KDA_REGISTERED
    if _KDA_REGISTERED:
        return
    _KDA_LIBRARY = torch.library.Library("npu", "FRAGMENT")
    _KDA_LIBRARY.define(
        "npu_chunk_kda_fwd(Tensor q, Tensor k, Tensor v, Tensor gk, Tensor beta, "
        "float scale, int chunk_size, *, Tensor? initial_state=None, "
        "bool output_final_state=False, int[]? cu_seqlens=None, int[]? chunk_indices=None, "
        "bool return_intermediate=False, bool safe_gate=False, bool transpose_state_layout=False) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    )
    _KDA_LIBRARY.define(
        "npu_chunk_kda_bwd(Tensor q, Tensor k, Tensor v, Tensor gk, Tensor beta, "
        "Tensor Aqk, Tensor Akk, Tensor d_o, float scale, int chunk_size, *, "
        "Tensor? initial_state=None, Tensor? dht=None, int[]? cu_seqlens=None, "
        "int[]? chunk_indices=None, bool safe_gate=False, bool transpose_state_layout=False) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    )
    _KDA_LIBRARY.impl("npu_chunk_kda_fwd", _chunk_kda_fwd, "CompositeImplicitAutograd")
    _KDA_LIBRARY.impl("npu_chunk_kda_bwd", _chunk_kda_bwd, "CompositeExplicitAutograd")
    _KDA_REGISTERED = True


register_kda_ops()
