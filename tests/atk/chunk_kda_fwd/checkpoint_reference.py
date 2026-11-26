"""CPU-only FP64 reference for the KDA state-export contract."""

import torch


def _fwd_h_fp64_reference(kg, w, u, gk, initial, starts, offsets, state_v_first):
    """Independent FwdH math on shared Prepare inputs, with storage casts.

    Prepare is outside this reference's scope. Products and state updates use
    CPU FP64; the BF16 H/v_new and FP32 recurrent-bank boundaries are explicit.
    The reference neither reads an NPU checkpoint nor calls the NPU FwdH.
    """
    kg, w, u, gk = [value.detach().cpu().double() for value in (kg, w, u, gk)]
    batch, heads, tokens, key_dim = kg.shape
    value_dim = u.shape[-1]
    spans = (
        [(0, begin, end) for begin, end in zip(starts, starts[1:])]
        if starts is not None
        else [(row, 0, tokens) for row in range(batch)]
    )
    shape = (
        (len(spans), heads, value_dim, key_dim)
        if state_v_first
        else (len(spans), heads, key_dim, value_dim)
    )
    final = torch.empty(shape, dtype=torch.float64)
    checkpoint = torch.full(shape, 19.0, dtype=torch.float64)
    initial = initial.detach().cpu().double() if initial is not None else None
    for row, (physical_batch, begin, end) in enumerate(spans):
        state = (
            (
                initial[row].transpose(-1, -2).clone()
                if state_v_first
                else initial[row].clone()
            )
            if initial is not None
            else torch.zeros(heads, key_dim, value_dim, dtype=torch.float64)
        )
        offset = int(offsets[row])
        valid_offset = 0 < offset < end - begin and offset % 64 == 0
        for token_begin in range(begin, end, 64):
            token_end = min(token_begin + 64, end)
            h = state.to(torch.bfloat16).double()
            values = u[physical_batch, :, token_begin:token_end] - (
                w[physical_batch, :, token_begin:token_end] @ h
            )
            values = values.to(torch.bfloat16).double()
            state = (
                (
                    torch.exp2(gk[physical_batch, :, token_end - 1]).unsqueeze(-1)
                    * state
                    + kg[physical_batch, :, token_begin:token_end].transpose(-1, -2)
                    @ values
                )
                .float()
                .double()
            )
            if valid_offset and token_end - begin == offset:
                checkpoint[row] = state.transpose(-1, -2) if state_v_first else state
        final[row] = state.transpose(-1, -2) if state_v_first else state
    return final, checkpoint
