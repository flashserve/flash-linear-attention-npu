from typing import Optional,Dict

import torch
import triton
import triton.language as tl

from fla.ops.triton.triton_core.index import prepare_chunk_indices
from fla.ops.triton.triton_core.utils import autotune_cache_kwargs, check_shared_mem, input_guard

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]

# ============================================================================
# Original chunk_local_cumsum_scalar_kernel
# ============================================================================
# Strategy: 2D block pointer (T, H), one program handles all H heads per batch.
# Uses BLOCK_T + CHUNK_SIZE two-level tiling: reshape → transpose → cumsum(axis=0)
# on (N_CHUNKS, CHUNK_SIZE, H), then reshape back.
# Grid: (NT, B) — parallel over chunks and batches.

@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE']
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 64,
):
    i_block, i_b = tl.program_id(0), tl.program_id(1)
    N_CHUNKS: tl.constexpr = BLOCK_T // CHUNK_SIZE

    if IS_VARLEN:
        i_s, i_block = tl.load(chunk_indices + i_block * 2).to(tl.int32), tl.load(
            chunk_indices + i_block * 2 + 1
        ).to(tl.int32)

        bos, eos = tl.load(cu_seqlens + i_s).to(tl.int32), tl.load(
            cu_seqlens + i_s + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    ptr_s = tl.make_block_ptr(
        s + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0)
    )
    ptr_o = tl.make_block_ptr(
        o + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0)
    )
    b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
    b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
    b_s = tl.trans(b_s, (1, 0, 2))
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    b_o = tl.trans(b_o, (1, 0, 2))
    b_o = tl.reshape(b_o, (BLOCK_T, H))

    tl.store(ptr_o, b_o.to(ptr_o.dtype.element_ty), boundary_check=(0,))
    return


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices_out: Dict[str, Optional[torch.LongTensor]] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:

    B, T, H = g.shape
    if chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError(
            f"chunk_size must be a power of 2, chunk_size is{chunk_size}"
        )
    BT = triton.next_power_of_2((1 << 17) // (H * chunk_size))
    chunk_indices = chunk_indices_out[str(BT)] if chunk_indices_out is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BLOCK_T=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        CHUNK_SIZE=chunk_size,
    )
    return g


# ============================================================================
# KDA variant: chunk_local_cumsum_scalar_kernel_kda
# ============================================================================
# Differences from the original chunk_local_cumsum_scalar_kernel:
#
# 1. Parallelism:  grid=(NT, B*H) with per-head parallelism, vs original grid=(NT, B).
# 2. Block pointer: 1D (T,) per head, vs original 2D (T, H) covering all heads.
# 3. Cumsum approach: direct tl.cumsum on 1D block, vs original reshape→transpose→cumsum→transpose→reshape
#    with two-level tiling (BLOCK_T / CHUNK_SIZE).
# 4. BT computation: BT = chunk_size (fixed), vs original BT = next_power_of_2((1<<17) // (H * chunk_size)).
# 5. chunk_indices: auto-computed via prepare_chunk_indices(), vs original pre-computed dict lookup.
# 6. No CHUNK_SIZE constexpr parameter.
# 7. head_first layout support for (B, H, T) input shape.

@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel_kda(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum_scalar_kda(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel_kda[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g


# ============================================================================
# KDA addition: chunk_local_cumsum_vector_kernel
# ============================================================================
# Vector cumsum for 4D input (B, T, H, S). Processes S-dimensional vectors
# with BS split. Supports HEAD_FIRST layout and IS_VARLEN.
# Grid: (S/BS, NT, B*H).

@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
# @triton.autotune(
#     configs=[
#         triton.Config({'BS': BS}, num_warps=num_warps)
#         for BS in BS_LIST
#         for num_warps in [2, 4, 8]
#     ],
#     key=['B', 'H', 'S', 'BT', 'IS_VARLEN', 'REVERSE'],
#     **autotune_cache_kwargs,
# )
@triton.autotune(
    configs=[triton.Config({'BS': 32}, num_warps=8)],
    key=['B', 'H', 'S', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if REVERSE:
        b_o = tl.cumsum(b_s, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_s, axis=0)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    chunk_local_cumsum_vector_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g


# ============================================================================
# KDA addition: chunk_global_cumsum_scalar_kernel
# ============================================================================
# Global (cross-chunk) cumsum for 3D input (B, T, H).
# Each program handles one (sequence, head) pair, iterating over all chunks
# to accumulate the running sum across chunk boundaries.
# Grid: (N * H,)

@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128, 256]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['B', 'H', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        else:
            p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0)
        b_ss = tl.sum(b_s, 0)
        if REVERSE:
            b_o = -b_o + b_ss + b_s
        b_o += b_z
        if i_c >= 0:
            b_z += b_ss
        if HAS_SCALE:
            b_o *= scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [16, 32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['B', 'H', 'S', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([BS], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        else:
            p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        if REVERSE:
            b_c = b_z[None, :] + tl.cumsum(b_s, axis=0, reverse=True)
        else:
            b_c = b_z[None, :] + tl.cumsum(b_s, axis=0)
        if HAS_SCALE:
            b_c *= scale
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        b_z += tl.sum(b_s, 0)

@input_guard
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = s.shape
    else:
        B, T, H = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (N * H,)
    chunk_global_cumsum_scalar_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return z

@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BS = min(32, triton.next_power_of_2(S))

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (triton.cdiv(S, BS), N * H)
    chunk_global_cumsum_vector_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        S=S,
        BS=BS,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return z

@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {s.shape}, "
            f"which should be [B, T, H]/[B, T, H, D] if `head_first=False` "
            f"or [B, H, T]/[B, H, T, D] otherwise",
        )


@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices_out: Dict[str, Optional[torch.LongTensor]] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
    use_per_head_kernel: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Compute local (intra-chunk) cumulative sum.

    Args:
        g: Input tensor of shape (B, T, H) or (B, H, T) if head_first=True
            for 3D, or (B, T, H, D) / (B, H, T, D) for 4D.
        chunk_size: Size of each chunk for the cumulative sum.
        reverse: If True, compute cumulative sum in reverse order.
        scale: Optional scaling factor applied after cumsum.
        cu_seqlens: Cumulative sequence lengths for variable-length inputs.
        chunk_indices_out: Pre-computed chunk indices dict keyed by BT string
            (original kernel interface). Mutually exclusive with chunk_indices.
        head_first: If True, input layout is (B, H, T, ...) instead of (B, T, H, ...).
        output_dtype: Output dtype. Defaults to torch.float.
        chunk_indices: Pre-computed chunk indices tensor (KDA kernel interface).
            Mutually exclusive with chunk_indices_out.
        use_per_head_kernel: If True, use the per-head-parallel kernel variant
            (chunk_local_cumsum_scalar_kernel_kda) which parallelizes over B*H
            instead of B and uses 1D block pointers per head. When True and
            input is 4D, also enables vector cumsum. When False (default),
            uses the original kernel with 2D block pointer and two-level tiling.
    """
    if cu_seqlens is not None:
        if g.shape[0] != 1:
            raise ValueError(
                f"Only batch size 1 is supported when cu_seqlens are provided, current size is{g.shape[0]}"
            )

    if use_per_head_kernel:
        # KDA path: per-head-parallel kernel with 1D block pointer
        if len(g.shape) == 4:
            return chunk_local_cumsum_vector(
                g=g,
                chunk_size=chunk_size,
                reverse=reverse,
                scale=scale,
                cu_seqlens=cu_seqlens,
                head_first=head_first,
                output_dtype=output_dtype,
                chunk_indices=chunk_indices,
            )
        else:
            return chunk_local_cumsum_scalar_kda(
                g=g,
                chunk_size=chunk_size,
                reverse=reverse,
                scale=scale,
                cu_seqlens=cu_seqlens,
                head_first=head_first,
                output_dtype=output_dtype,
                chunk_indices=chunk_indices,
            )
    else:
        # Original path: 2D block pointer with two-level tiling
        if len(g.shape) == 3:
            return chunk_local_cumsum_scalar(
                g=g,
                chunk_size=chunk_size,
                reverse=reverse,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_indices_out=chunk_indices_out,
                head_first=head_first,
                output_dtype=output_dtype,
            )
        else:
            raise ValueError(
                f"Unsupported input shape {g.shape} for original kernel. "
                f"4D input requires use_per_head_kernel=True. "
                f"Expected (B, T, H) if `head_first=False` or (B, H, T) otherwise"
            )