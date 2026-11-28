"""CPU reference for public GDN FwdH chunk states in dense and packed layouts."""

from __future__ import annotations

import torch


def state_reference(
    k,
    w,
    u,
    *,
    g=None,
    gk=None,
    initial_state=None,
    cu_seqlens=None,
    chunk_size=64,
    output_final_state=False,
    state_v_first=False,
    high_precision=False,
):
    batch, key_heads, tokens, key_dim = k.shape
    value_heads, value_dim = u.shape[1], u.shape[-1]
    if value_heads % key_heads or (g is None and gk is None):
        raise ValueError("Expected grouped value heads and at least one gate")
    calc = torch.float64 if high_precision else torch.float32
    element = calc if high_precision else u.dtype
    state_dtype = calc if high_precision or initial_state is None else initial_state.dtype
    if cu_seqlens is None:
        spans = [(b, 0, tokens) for b in range(batch)]
    else:
        spans = [(0, int(l), int(r)) for l, r in zip(cu_seqlens, cu_seqlens[1:])]
    counts = [(r - l + chunk_size - 1) // chunk_size for _, l, r in spans]
    num_chunks = sum(counts) if cu_seqlens is not None else counts[0]
    history = torch.empty((batch, num_chunks, value_heads, key_dim, value_dim), dtype=element)
    values = torch.empty_like(u, dtype=element)
    finals = []
    offset = 0
    head_index = torch.arange(value_heads) // (value_heads // key_heads)
    for seq, (b, left, right) in enumerate(spans):
        if initial_state is None:
            state = torch.zeros((value_heads, key_dim, value_dim), dtype=state_dtype)
        else:
            logical = initial_state[seq].transpose(-1, -2) if state_v_first else initial_state[seq]
            state = logical.to(state_dtype)
        for j, start in enumerate(range(left, right, chunk_size)):
            end = min(start + chunk_size, right)
            region = slice(start, end)
            h = state.to(element)
            history[b, offset + j] = h
            projected = w[b, :, region].to(calc) @ h.to(calc)
            residual = u[b, :, region].to(calc) - projected
            values[b, :, region] = residual.to(element)
            decay = 1
            if g is not None:
                gate = g[b, :, region].to(calc)
                residual = residual * (gate[:, -1:] - gate).exp().unsqueeze(-1)
                decay = gate[:, -1].exp()[:, None, None]
            if gk is not None:
                decay = decay * gk[b, :, end - 1].to(calc).exp2().unsqueeze(-1)
            keys = k[b, head_index, region].to(calc)
            update = keys.transpose(-1, -2) @ residual.to(element).to(calc)
            state = (state.to(calc) * decay + update).to(state_dtype)
        if output_final_state:
            finals.append(state)
        if cu_seqlens is not None:
            offset += counts[seq]
    final = torch.stack(finals) if output_final_state else None
    if state_v_first:
        history = history.transpose(-1, -2).contiguous()
        if final is not None:
            final = final.transpose(-1, -2).contiguous()
    return history, values, final
