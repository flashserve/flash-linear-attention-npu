from typing import Optional

import torch
import triton
import triton.language as tl


_BC = 16
_MAX_BK = 64
_NUM_WARPS = 4


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    NT,
    B,
    TOTAL_TASKS,
):
    core_id = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    T_max = T

    base_tasks_per_block = TOTAL_TASKS // num_blocks
    remainder_tasks = TOTAL_TASKS % num_blocks

    if core_id < remainder_tasks:
        tasks_this_core = base_tasks_per_block + 1
        start_idx = core_id * tasks_this_core
    else:
        tasks_this_core = base_tasks_per_block
        start_idx = core_id * base_tasks_per_block + remainder_tasks

    for idx in range(start_idx, start_idx + tasks_this_core):
        i_b = idx // NT
        local_idx = idx % NT

        if IS_VARLEN:
            i_n = tl.load(chunk_indices + local_idx * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + local_idx * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_local = eos - bos
        else:
            bos, eos = 0, T
            i_t = local_idx
            T_local = T

        for i_h in range(H):
            k_batch_off = i_b * T_max * H * K
            beta_batch_off = i_b * H * T_max
            g_batch_off = i_b * H * T_max
            A_batch_off = i_b * T_max * H * BT

            p_beta = tl.make_block_ptr(beta + beta_batch_off + bos + i_h * T_max, (T_local,), (1,), (i_t * BT,), (BT,), (0,))
            b_beta = tl.load(p_beta, boundary_check=(0,))

            b_A = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(k + k_batch_off + i_h * T_max * K + bos * K, (T_local, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                dot_product = tl.dot(b_k, tl.trans(b_k))

                o_t = i_t * BT + tl.arange(0, BT)
                o_t = o_t.to(tl.float32)
                T_mask = (o_t < T_local).to(tl.float32)

                row_indices = tl.arange(0, BT)[:, None]
                col_indices = tl.arange(0, BT)[None, :]
                tril_mask = (row_indices > col_indices).to(tl.float32)
                tril_mask = tril_mask * T_mask[:, None]
                masked_dot = dot_product * tril_mask
                b_A += masked_dot

            if USE_G:
                p_g = tl.make_block_ptr(g + g_batch_off + bos + i_h * T_max, (T_local,), (1,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g_diff = b_g[:, None] - b_g[None, :]
                b_A *= tl.exp(b_g_diff)
            b_A *= b_beta[:, None]

            p_A = tl.make_block_ptr(A + A_batch_off + (bos * H + i_h) * BT, (T_local, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_tiled(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_local = eos - bos
    else:
        bos = 0
        T_local = T

    if i_t * BT >= T_local:
        return

    k_base = k + ((i_b * H + i_h) * T + bos) * K
    beta_base = beta + (i_b * H + i_h) * T + bos
    g_base = g + (i_b * H + i_h) * T + bos
    A_base = A + (i_b * T * H + bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    n_sub: tl.constexpr = BT // BC

    for s in range(n_sub):
        row_start = i_t * BT + s * BC
        m_s = row_start + o_i < T_local
        p_beta = tl.make_block_ptr(beta_base, (T_local,), (1,), (row_start,), (BC,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,))
        if USE_G:
            p_gs = tl.make_block_ptr(g_base, (T_local,), (1,), (row_start,), (BC,), (0,))
            b_gs = tl.load(p_gs, boundary_check=(0,))

        for c in range(n_sub):
            col_start = i_t * BT + c * BC
            m_c = col_start + o_i < T_local
            b_A = tl.zeros([BC, BC], dtype=tl.float32)

            if c <= s:
                for i_k in range(tl.cdiv(K, BK)):
                    p_ks = tl.make_block_ptr(
                        k_base,
                        (T_local, K),
                        (K, 1),
                        (row_start, i_k * BK),
                        (BC, BK),
                        (1, 0),
                    )
                    p_kc = tl.make_block_ptr(
                        k_base,
                        (T_local, K),
                        (K, 1),
                        (col_start, i_k * BK),
                        (BC, BK),
                        (1, 0),
                    )
                    b_ks = tl.load(p_ks, boundary_check=(0, 1))
                    b_kc = tl.load(p_kc, boundary_check=(0, 1))
                    b_A += tl.dot(b_ks, tl.trans(b_kc), allow_tf32=False)

                if USE_G:
                    p_gc = tl.make_block_ptr(g_base, (T_local,), (1,), (col_start,), (BC,), (0,))
                    b_gc = tl.load(p_gc, boundary_check=(0,))
                    b_gdiff = b_gs[:, None] - b_gc[None, :]
                    if USE_EXP2:
                        b_A *= tl.math.exp2(b_gdiff)
                    else:
                        b_A *= tl.exp(b_gdiff)

                b_A *= b_beta[:, None]
                if s == c:
                    m_blk = (o_i[:, None] > o_i[None, :]) & (m_s[:, None] & m_s[None, :])
                else:
                    m_blk = m_s[:, None] & m_c[None, :]
                b_A = tl.where(m_blk, b_A, 0.0)

            p_A = tl.make_block_ptr(
                A_base,
                (T_local, BT),
                (H * BT, 1),
                (row_start, c * BC),
                (BC, BC),
                (1, 0),
            )
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC

    for i_h in range(H):
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_val = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            T_val = T

        should_compute = (i_t * BT + i_i * BC < T_val) and (i_i > i_j)

        if should_compute:
            k_ptr = k + (bos * H + i_h) * K
            g_ptr = g + (bos * H + i_h) * K
            A_ptr = A + (bos * H + i_h) * BT

            p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T_val,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
            b_beta = tl.load(p_beta, boundary_check=(0,))

            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(k_ptr, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK),
                                        (1, 0))
                p_g = tl.make_block_ptr(g_ptr, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK),
                                        (1, 0))
                b_kt = tl.make_block_ptr(k_ptr, (K, T_val), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC),
                                         (0, 1))
                p_gk = tl.make_block_ptr(g_ptr, (K, T_val), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC),
                                         (0, 1))

                o_k = i_k * BK + tl.arange(0, BK)
                m_k = o_k < K
                b_gn = tl.load(g_ptr + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
                b_g = tl.load(p_g, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.exp(b_g - b_gn[None, :])
                b_gk = tl.load(p_gk, boundary_check=(0, 1))
                b_kt = tl.load(b_kt, boundary_check=(0, 1)) * tl.exp(b_gn[:, None] - b_gk)
                b_A += tl.dot(b_k, b_kt)
            b_A *= b_beta[:, None]

            p_A = tl.make_block_ptr(A_ptr, (T_val, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    for i_h in range(H):
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_val = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            T_val = T

        should_compute = (i_t * BT + i_i * BC < T_val)

        if should_compute:
            o_i = tl.arange(0, BC)
            o_k = tl.arange(0, BK)
            m_k = o_k < K
            m_A = (i_t * BT + i_i * BC + o_i) < T_val
            o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC

            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK),
                                    (1, 0))
            p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK),
                                    (1, 0))
            p_beta = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h

            b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.load(p_beta, mask=m_A, other=0)[:, None]
            b_g = tl.load(p_g, boundary_check=(0, 1))

            p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
            p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k

            for j in range(0, min(BC, T_val - i_t * BT - i_i * BC)):
                b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
                b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
                b_A = tl.sum(b_k * b_kt[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
                # 转化成f32
                o_i_tmp = o_i.to(tl.float32)
                b_A = tl.where(o_i_tmp > j, b_A, 0.)

                tl.store(A + o_A + j, b_A, mask=m_A)
                p_kt += H * K
                p_gk += H * K


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    use_exp2: bool = False,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, H, T, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, H, T, K = k.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    beta = beta.transpose(1, 2).contiguous()

    if gk is None:
        use_g = g is not None
        g = g.transpose(1, 2).contiguous() if use_g else beta
        A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
        BK = min(_MAX_BK, triton.next_power_of_2(K))
        BC = min(_BC, BT)
        chunk_scaled_dot_kkt_fwd_kernel_tiled[(NT, B * H)](
            k=k,
            g=g,
            beta=beta,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            IS_VARLEN=cu_seqlens is not None,
            USE_G=use_g,
            USE_EXP2=use_exp2,
            num_warps=_NUM_WARPS,
        )
        if k.device.type == "npu":
            torch.npu.synchronize()
        return A

    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    BK = max(triton.next_power_of_2(K), 16)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    grid = (NT, NC * NC, B)
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
        k=k,
        g=gk,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        num_warps=4,
        num_stages=3,
    )

    grid = (NT, NC, B)
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](
        k=k,
        g=gk,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        num_warps=4,
    )
    if k.device.type == "npu":
        torch.npu.synchronize()
    return A
