from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.triton.triton_core.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.triton.triton_core.utils import input_guard, prepare_chunk_indices


RCP_LN2 = 1.4426950408889634
ASCEND_MAX_GRID_DIM = 65535
_NUM_WARPS = 4
_MAX_BK = 64
_MAX_BV = 64


def _max_grid_axis_chunks(axis_size: int, other_grid_product: int) -> int:
    return max(1, min(axis_size, ASCEND_MAX_GRID_DIM // max(other_grid_product, 1)))


def _npu_core_count() -> int:
    try:
        import triton.runtime.driver as driver

        device = torch.npu.current_device()
        return int(driver.active.utils.get_device_properties(device).get("num_aicore", 24))
    except Exception:
        return 24


def _launch_solve_tril_kernel(kernel, *, NT: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = _max_grid_axis_chunks(NT, bh_total)
    chunk_indices = kernel_kwargs.get("chunk_indices")
    cu_seqlens = kernel_kwargs.get("cu_seqlens")
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs["chunk_indices"] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs["NT_OFFSET"] = 0
        else:
            kernel_kwargs["NT_OFFSET"] = nt_off
        max_bh = _max_grid_axis_chunks(bh_total, nt_len)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs["BH_OFFSET"] = bh_off
            kernel[(nt_len, bh_len)](num_warps=_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=["T"])
def _solve_tril_16x16_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b = i_bh // H
    i_h = i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = -tl.where(m_A, b_A, 0)

    for i in range(2, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I

    p_Ai = tl.make_block_ptr(Ai, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0))
    tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b = i_bh // H
    i_h = i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    b_Ai_11 = -tl.where(m_A, tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32), 0)
    b_Ai_22 = -tl.where(m_A, tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32), 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.0)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(18, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.0)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision="ieee"),
        b_Ai_11,
        input_precision="ieee",
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b = i_bh // H
    i_h = i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_A_33 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_A_44 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    b_Ai_11 = -tl.where(m_A, tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32), 0)
    b_Ai_22 = -tl.where(m_A, tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32), 0)
    b_Ai_33 = -tl.where(m_A, tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32), 0)
    b_Ai_44 = -tl.where(m_A, tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32), 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.0)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(18, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.0)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(34, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.0)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(50, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.0)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision="ieee"), b_Ai_11, input_precision="ieee")
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision="ieee"), b_Ai_22, input_precision="ieee")
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision="ieee"), b_Ai_33, input_precision="ieee")
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision="ieee") + tl.dot(b_A_32, b_Ai_21, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision="ieee") + tl.dot(b_A_43, b_Ai_32, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision="ieee")
        + tl.dot(b_A_42, b_Ai_21, input_precision="ieee")
        + tl.dot(b_A_43, b_Ai_31, input_precision="ieee"),
        input_precision="ieee",
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@input_guard
def solve_tril_triton_ascend(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    assert A.shape[-1] in (16, 32, 64)
    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=output_dtype)
    merge_fn = (
        _solve_tril_16x16_kernel
        if BT == 16
        else _merge_16x16_to_32x32_inverse_kernel
        if BT == 32
        else _merge_16x16_to_64x64_inverse_kernel
    )
    _launch_solve_tril_kernel(
        merge_fn,
        NT=NT,
        bh_total=B * H,
        kernel_kwargs={
            "A": A,
            "Ai": Ai,
            "cu_seqlens": cu_seqlens,
            "chunk_indices": chunk_indices,
            "T": T,
            "H": H,
            "BT": BT,
            "IS_VARLEN": cu_seqlens is not None,
            "NT_OFFSET": 0,
            "BH_OFFSET": 0,
        },
    )
    return Ai


@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    "USE_G": lambda args: args["g"] is not None,
})
@triton.jit(do_not_specialize=["T", "B", "task_num", "num_core"])
def _recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    T_max = T
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t_o = task_id // (B * HV)
        i_bh = task_id % (B * HV)
        i_b = i_bh // HV
        i_h = i_bh % HV
        if IS_VARLEN:
            i_n = tl.load(chunk_indices + i_t_o * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_t_o * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
            bos_bh = bos
        else:
            i_t = i_t_o
            bos = i_b * T
            bos_bh = i_b * HV * T_max

        offs_t = tl.arange(0, BT)
        global_offs_t = i_t * BT + offs_t
        mask_t = global_offs_t < T

        offs_t_2d = global_offs_t[:, None]
        offs_bt = tl.arange(0, BT)[None, :]
        ptr_A = A + (bos * HV + i_h) * BT + offs_t_2d * (HV * BT) + offs_bt
        b_A = tl.load(ptr_A, mask=mask_t[:, None], other=0.0).to(tl.float32)

        ptr_beta = beta + bos_bh + i_h * T_max + global_offs_t
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            offs_v = i_v * BV + tl.arange(0, BV)[None, :]
            mask_v = mask_t[:, None] & (offs_v < V)
            ptr_v = v + (bos * HV + i_h) * V + offs_t_2d * (HV * V) + offs_v
            b_v = tl.load(ptr_v, mask=mask_v, other=0.0).to(tl.float32)
            b_u = tl.dot(b_A, (b_v * b_beta[:, None]).to(b_v.dtype), allow_tf32=False)
            ptr_u = u + (bos * HV + i_h) * V + offs_t_2d * (HV * V) + offs_v
            tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)

        if USE_G:
            ptr_g = g + bos_bh + i_h * T_max + global_offs_t
            b_g = tl.math.exp2(tl.load(ptr_g, mask=mask_t, other=0.0).to(tl.float32))

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)[None, :]
            mask_k = mask_t[:, None] & (offs_k < K)
            ptr_k = k + (bos * H + i_h // (HV // H)) * K + offs_t_2d * (H * K) + offs_k
            b_k = tl.load(ptr_k, mask=mask_k, other=0.0).to(tl.float32)
            b_kb = b_k * b_beta[:, None]
            if USE_G:
                b_kb *= b_g[:, None]
            b_w = tl.dot(b_A, b_kb.to(b_A.dtype), allow_tf32=False)
            ptr_w = w + (bos * HV + i_h) * K + offs_t_2d * (HV * K) + offs_k
            tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)


@input_guard
def recompute_w_u_fwd_triton_ascend(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g_log2: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Triton-Ascend kernels use fla-org's time-major layout: [B, T, H, D].
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    B, T, H, K = k_t.shape
    HV = v_t.shape[2]
    V = v_t.shape[-1]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    beta_t = beta.transpose(1, 2).contiguous()
    g_t = g_log2.transpose(1, 2).contiguous() if g_log2 is not None else beta_t
    w_t = k_t.new_empty(B, T, HV, K)
    u_t = torch.empty_like(v_t)

    num_core = _npu_core_count()
    task_num = NT * B * HV
    _recompute_w_u_fwd_kernel[(num_core,)](
        k=k_t,
        v=v_t,
        beta=beta_t,
        w=w_t,
        u=u_t,
        A=A,
        g=g_t,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=min(_MAX_BK, triton.next_power_of_2(K)),
        BV=min(_MAX_BV, triton.next_power_of_2(V)),
        num_warps=_NUM_WARPS,
        num_stages=3,
    )
    if k.device.type == "npu":
        torch.npu.synchronize()
    return w_t.transpose(1, 2).contiguous(), u_t.transpose(1, 2).contiguous()


@input_guard
def chunk_gated_delta_rule_fwd_intra_triton_ascend(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # Triton-Ascend kernels use exp2, so convert natural-log gates to log2-domain.
    g_log2 = (g * RCP_LN2).contiguous()
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_log2,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        use_exp2=True,
    )
    A = solve_tril_triton_ascend(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd_triton_ascend(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_log2=g_log2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, A
