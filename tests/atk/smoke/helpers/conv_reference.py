# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch


def make_tensor(shape, *, dtype=torch.float16, offset=0, device="cpu"):
    numel = 1
    for dim in shape:
        numel *= dim
    values = torch.arange(offset, offset + numel, dtype=torch.float32)
    values = (values.remainder(37) - 18.0) / 64.0
    return values.reshape(shape).to(dtype=dtype, device=device)


def causal_conv1d_preactivation_ref(x, weight, initial_state=None):
    """FP32 forward preactivation for logical [B, T, D] input."""
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    initial_fp32 = None if initial_state is None else initial_state.float()
    batch, seqlen, dim = x_fp32.shape
    width = weight_fp32.shape[0]
    out = torch.zeros((batch, seqlen, dim), dtype=torch.float32)

    for offset in range(width):
        rows = seqlen - offset
        if rows <= 0:
            break
        kernel = width - 1 - offset
        out[:, offset:, :] += x_fp32[:, :rows, :] * weight_fp32[kernel]

    if initial_fp32 is not None:
        for offset in range(1, width):
            kernel = width - 1 - offset
            for row in range(min(seqlen, offset)):
                state_slot = width - offset + row
                out[:, row, :] += initial_fp32[:, state_slot, :] * weight_fp32[kernel]
    return out


def causal_conv1d_bwd_ref(
    x,
    y,
    weight,
    dy,
    initial_state=None,
    dht=None,
    activation=0,
    cast_reduction_outputs=True,
):
    """CPU golden. FP16/BF16 inputs are promoted to FP32 before calculation."""
    if activation not in (0, 1, 2):
        raise ValueError("activation must be 0, 1, or 2")
    if activation != 0 and y is None:
        raise ValueError("y is required when activation is enabled")

    dtype_in = x.dtype
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    grad = dy.float()
    initial_fp32 = None if initial_state is None else initial_state.float()
    dht_fp32 = None if dht is None else dht.float()

    if activation != 0:
        y_fp32 = y.float()
        sigmoid = torch.sigmoid(y_fp32)
        grad = grad * sigmoid * (1.0 + y_fp32 * (1.0 - sigmoid))

    batch, seqlen, dim = x_fp32.shape
    width = weight_fp32.shape[0]
    dx = torch.zeros((batch, seqlen, dim), dtype=torch.float32)
    dw = torch.zeros((width, dim), dtype=torch.float32)
    db = grad.sum(dim=(0, 1))
    dh0 = torch.zeros((batch, width, dim), dtype=torch.float32)

    for offset in range(width):
        rows = seqlen - offset
        if rows <= 0:
            break
        kernel = width - 1 - offset
        grad_slice = grad[:, offset:, :]
        dx[:, :rows, :] += grad_slice * weight_fp32[kernel]
        dw[kernel] += (grad_slice * x_fp32[:, :rows, :]).sum(dim=(0, 1))

    if initial_fp32 is not None:
        for offset in range(1, width):
            kernel = width - 1 - offset
            for row in range(min(seqlen, offset)):
                state_slot = width - offset + row
                dw[kernel] += (grad[:, row, :] * initial_fp32[:, state_slot, :]).sum(
                    dim=0
                )

        for state_slot in range(1, width):
            for row in range(min(seqlen, state_slot)):
                kernel = state_slot - 1 - row
                dh0[:, state_slot, :] += grad[:, row, :] * weight_fp32[kernel]

    if dht_fp32 is not None:
        tail_start = max(0, seqlen - (width - 1))
        tail_rows = seqlen - tail_start
        dx[:, tail_start:, :] += dht_fp32[:, 1 : 1 + tail_rows, :]

    if cast_reduction_outputs:
        dw = dw.to(dtype_in)
        db = db.to(dtype_in)
    return dx.to(dtype_in), dw, db, dh0.to(dtype_in)


def logical_to_bnsd(tensor, num_heads):
    batch, seqlen, dim = tensor.shape
    head_dim = dim // num_heads
    return (
        tensor.reshape(batch, seqlen, num_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def logical_to_ntd(tensor, num_heads):
    total_tokens, dim = tensor.shape
    head_dim = dim // num_heads
    return (
        tensor.reshape(total_tokens, num_heads, head_dim).permute(1, 0, 2).contiguous()
    )
