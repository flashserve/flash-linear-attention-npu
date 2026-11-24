"""Tile-local optional SiLU and L2 normalization for VDN feature rows."""

import torch
import triton
import triton.language as tl

VECTOR_PROGRAMS = 40
ROW_TILE = 32


@triton.jit
def _source_round(x, kind: tl.constexpr):
    # Keep BF16 rounding observable inside the fused kernel even when the
    # compiler simplifies a BF16/FP32 conversion pair.
    if kind == tl.bfloat16:
        bits = x.to(tl.uint32, bitcast=True)
        rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
        return rounded.to(tl.float32, bitcast=True)
    return x


@triton.jit
def _activate_values(x, kind: tl.constexpr, SILU: tl.constexpr, NORM: tl.constexpr,
                     BLOCK: tl.constexpr):
    if SILU:
        x = _source_round(tl.div_rn(x, 1.0 + tl.exp(-x)), kind)
    if NORM:
        norm = _source_round(tl.sqrt_rn(tl.sum(x * x, axis=1)), kind)
        eps = _source_round(tl.full((BLOCK,), 1.0e-6, tl.float32), kind)
        x = tl.div_rn(x, tl.maximum(norm, eps)[:, None])
    return x


@triton.jit
def _activate(src, dst, rows, D: tl.constexpr, SILU: tl.constexpr,
              NORM: tl.constexpr, BLOCK: tl.constexpr, PROGRAMS: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    kind: tl.constexpr = src.dtype.element_ty
    for block in range(tl.program_id(0), tl.cdiv(rows, BLOCK), PROGRAMS):
        row = block * BLOCK + lane
        offset = row[:, None] * D + d[None, :]
        x = tl.load(src + offset, row[:, None] < rows, other=0).to(tl.float32)
        y = _activate_values(x, kind, SILU, NORM, BLOCK)
        tl.store(dst + offset, y, row[:, None] < rows)


def activate_tile(src, *, silu, l2norm):
    if not silu and not l2norm:
        return src
    token_major = not src.is_contiguous() and src.transpose(1, 2).is_contiguous()
    physical = src.transpose(1, 2) if token_major else src.contiguous()
    result = torch.empty_like(physical)
    dim = src.size(-1)
    rows = src.numel() // dim
    programs = min(triton.cdiv(rows, ROW_TILE), VECTOR_PROGRAMS)
    # Drains host submissions while preserving device-stream asynchrony.
    stream = torch.npu.current_stream(src.device).npu_stream
    _activate[(programs,)](physical, result, rows, dim, silu, l2norm, ROW_TILE, programs, stream=stream)
    return result.transpose(1, 2) if token_major else result
