"""Build the inverse of a Cholesky factor by merging 32-row inverse panels."""

import torch
import triton
import triton.language as tl


def lower_block(matrix, width, row, col):
    batch, dim, _ = matrix.shape
    stride_b, stride_r, stride_c = matrix.stride()
    return matrix.as_strided(
        (batch, dim // (2 * width), width, width),
        (stride_b, 2 * width * (stride_r + stride_c), stride_r, stride_c),
        matrix.storage_offset() + row * stride_r + col * stride_c,
    )


def inverse_merge(chol, inv, initial_width=1):
    batch, dim, _ = chol.shape
    width = initial_width
    while width < dim:
        count = dim // (2 * width)
        left = inv[:, ::2].reshape(-1, width, width).contiguous()
        right = inv[:, 1::2].reshape(-1, width, width).contiguous()
        below = lower_block(chol, width, width, 0).reshape(-1, width, width).contiguous()
        coupling = -torch.bmm(right, torch.bmm(below, left))
        merged = inv.new_empty(batch, count, 2 * width, 2 * width)
        merged[..., :width, :width].copy_(left.view(batch, count, width, width))
        merged[..., :width, width:].zero_()
        merged[..., width:, :width].copy_(coupling.view(batch, count, width, width))
        merged[..., width:, width:].copy_(right.view(batch, count, width, width))
        inv = merged
        del left, right, below, coupling
        width *= 2
    return inv.view(batch, dim, dim)


@triton.jit
def _chol_inverse_panel(src, factor_ptr, inverse_ptr, WIDTH: tl.constexpr):
    b = tl.program_id(0)
    row = tl.arange(0, WIDTH)
    col = tl.arange(0, WIDTH)
    offset = b * WIDTH * WIDTH + row[:, None] * WIDTH + col[None, :]
    work = tl.load(src + offset)
    factor = tl.full((WIDTH, WIDTH), 0.0, tl.float32)
    inverse = tl.full((WIDTH, WIDTH), 0.0, tl.float32)
    for j in range(WIDTH):
        column = tl.sum(tl.where(col[None, :] == j, work, 0.0), axis=1)
        pivot = tl.sqrt(tl.sum(tl.where(row == j, column, 0.0), axis=0))

        # On row j, columns < j of L and rows < j of the inverse are complete.
        previous = tl.sum(tl.where(row[:, None] == j, factor, 0.0), axis=0)
        accum = tl.sum(previous[:, None] * inverse, axis=0)
        inverse_row = ((col == j).to(tl.float32) - accum) / pivot
        inverse = tl.where(row[:, None] == j, inverse_row[None, :], inverse)

        x = tl.where(row >= j, column / pivot, 0.0)
        factor = tl.where(col[None, :] == j, x[:, None], factor)
        work = work - x[:, None] * x[None, :]
    tl.store(factor_ptr + offset, factor)
    tl.store(inverse_ptr + offset, inverse)


def factor_and_seeds(system, width):
    batch, dim, _ = system.shape
    if dim % width:
        raise ValueError("dimension must be divisible by the panel width")
    result = torch.zeros_like(system)
    seeds = system.new_empty((batch, dim // width, width, width))
    for i in range(dim // width):
        start = i * width
        end = start + width
        active = system[:, start:end, start:end].contiguous()
        diagonal = torch.empty_like(active)
        diag_inv = torch.empty_like(active)
        # Order direct launches with queued torch_npu ops.
        torch.npu.current_stream(system.device).synchronize()
        _chol_inverse_panel[(batch,)](active, diagonal, diag_inv, width, enable_fp_fusion=False)
        result[:, start:end, start:end].copy_(diagonal)
        seeds[:, i].copy_(diag_inv)
        if end < dim:
            panel = torch.bmm(system[:, end:, start:end].contiguous(), diag_inv.mT)
            result[:, end:, start:end].copy_(panel)
            system[:, end:, end:].sub_(torch.bmm(panel, panel.mT))
    return result, seeds


def solve(system):
    """The trailing Schur updates overwrite the private FP32 system tile."""
    width = 32
    factor, seeds = factor_and_seeds(system, width)
    return inverse_merge(factor, seeds, width)
