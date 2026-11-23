# WIP page-level Qwen attention (16-token pages, no token-count mask).
# NOT validated: page-level semantics differ from the token-level reference.

from __future__ import annotations

import catlass.tla as tla


@tla.kernel
def qwen_sparse_attention_kernel(
    mem_q: tla.Tensor,
    mem_k: tla.Tensor,
    mem_v: tla.Tensor,
    mem_o: tla.Tensor,
    mem_idx: tla.Tensor,
    mem_bt: tla.Tensor,
    mem_req: tla.Tensor,
    softmax_scale: tla.Float32,
    rows: tla.Int32,
    heads: tla.Int32,
    width: tla.Int32,
    page_size: tla.Int32,
    num_pages: tla.Int32,
    kv_blk_stride: tla.Int32,
    kv_tok_stride: tla.Int32,
    head_groups: tla.Int32,
    head_dim: tla.Constexpr[int],
    q_block: tla.Constexpr[int],
    kv_block: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1
    c2 = 2
    c16 = 16
    alignment = 512
    l0c_alignment = 1024
    min_value = -3.0e38
    max_floor = -1.0e30
    vl_f32 = 64
    q_sub = q_block // 2
    half_dim = head_dim // 2
    num_dim_chunks = head_dim // vl_f32
    io_dtype = mem_q.ptr.dtype

    num_tasks = rows * head_groups
    num_tiles = (width + kv_block - c1) // kv_block

    l1_q_ptr = tla.allocate(
        q_block * head_dim, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_k_ptr = tla.allocate(
        head_dim * kv_block, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_v_ptr = tla.allocate(
        kv_block * head_dim, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_p_ptr = tla.allocate(
        q_block * kv_block, io_dtype, tla.AddressSpace.l1, alignment
    )

    l0a_q_ptr = tla.allocate(
        q_block * half_dim, io_dtype, tla.AddressSpace.l0a, alignment
    )
    l0a_p_ptr = tla.allocate(
        q_block * kv_block, io_dtype, tla.AddressSpace.l0a, alignment
    )
    l0b_ptr = tla.allocate(
        kv_block * head_dim, io_dtype, tla.AddressSpace.l0b, alignment
    )

    l0c_s_ptr = tla.allocate(
        q_block * kv_block, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )
    l0c_o_ptr = tla.allocate(
        q_block * head_dim, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )

    ub_s_ptr = tla.allocate(
        q_sub * kv_block, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_p_ptr = tla.allocate(
        (q_sub + c1) * kv_block, io_dtype, tla.AddressSpace.ub, alignment
    )
    ub_o_ptr = tla.allocate(
        q_sub * head_dim, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_otmp_ptr = tla.allocate(
        q_sub * head_dim, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_out_ptr = tla.recast_ptr(ub_otmp_ptr, dtype=io_dtype)
    ub_idx_ptr = tla.allocate(kv_block, tla.Int32, tla.AddressSpace.ub, alignment)
    ub_now_max_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_last_max_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_sum_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_dm_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_upd_sum_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_blk_sum_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)

    mutex_l1_q = tla.mutex("qsa_l1_q", id=0)
    mutex_l1_k = tla.mutex("qsa_l1_k", id=1)
    mutex_l1_v = tla.mutex("qsa_l1_v", id=2)
    mutex_l0a_q = tla.mutex("qsa_l0a_q", id=3)
    mutex_l0a_p = tla.mutex("qsa_l0a_p", id=4)
    mutex_l0b = tla.mutex("qsa_l0b", id=5)
    mutex_l0c_s = tla.mutex("qsa_l0c_s", id=6)
    mutex_l0c_o = tla.mutex("qsa_l0c_o", id=7)
    mutex_ub_idx = tla.mutex("qsa_ub_idx", id=8)
    mutex_ub_p = tla.mutex("qsa_ub_p", id=9)
    mutex_ub_out = tla.mutex("qsa_ub_out", id=10)

    s_ready = tla.cross_flag("qsa_s_ready")
    s_free = tla.cross_flag("qsa_s_free")
    p_ready = tla.cross_flag("qsa_p_ready")
    o_ready = tla.cross_flag("qsa_o_ready")
    o_free = tla.cross_flag("qsa_o_free")

    cast_zero = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast_one = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ONE,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )

    with tla.vector():
        tla.cross_core_set_flag(s_free, tla.arch.VECTOR)
        tla.cross_core_set_flag(o_free, tla.arch.VECTOR)

    task_range = tla.range(
        tla.arch.block_idx(),
        num_tasks,
        tla.arch.block_num(),
    )
    for task in task_range:
        row = task // head_groups
        head_group = task - row * head_groups
        head_base = head_group * q_block
        req = mem_req[row]
        q_tile_size = heads - head_base
        if q_tile_size > q_block:
            q_tile_size = tla.as_numeric(q_block)

        vec_idx = tla.arch.sub_block_idx()
        rows_here = q_tile_size - vec_idx * q_sub
        if rows_here > q_sub:
            rows_here = tla.as_numeric(q_sub)
        if rows_here < c0:
            rows_here = tla.as_numeric(0)

        with tla.vector():
            ub_o_init = tla.make_tensor(
                ub_o_ptr,
                tla.make_layout(
                    tla.make_shape(q_sub, head_dim),
                    tla.make_stride(head_dim, c1),
                ),
            )
            ub_now_max_i8 = tla.make_tensor(
                ub_now_max_ptr,
                tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
            )
            ub_sum_i8 = tla.make_tensor(
                ub_sum_ptr,
                tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
            )
            with tla.vec.func(mode="simd"):
                pregFull8 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                zero8 = tla.full(0.0, dtype=tla.Float32)
                floor8 = tla.full(max_floor, dtype=tla.Float32)
                for i8 in tla.range(rows_here):
                    for j8 in tla.range(num_dim_chunks):
                        tile8 = tla.tile_view(
                            ub_o_init,
                            tla.make_shape(c1, vl_f32),
                            tla.make_coord(i8, j8),
                        )
                        tile8.store(zero8, mask=pregFull8)
                ub_now_max_i8.store(floor8, mask=pregFull8)
                ub_sum_i8.store(zero8, mask=pregFull8)
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE,
                    tla.params.MemType.VEC_LOAD,
                )

        l1_q = tla.make_tensor(
            l1_q_ptr,
            tla.make_layout_with_tag((q_block, head_dim), io_dtype, tla.arch.zN),
        )
        l1_k = tla.make_tensor(
            l1_k_ptr,
            tla.make_layout_with_tag((head_dim, kv_block), io_dtype, tla.arch.nZ),
        )
        l1_v_ref = tla.make_tensor(
            l1_v_ptr,
            tla.make_layout(
                tla.make_shape(kv_block, head_dim),
                tla.make_stride(head_dim, c1),
            ),
        )
        l1_v = tla.make_tensor_like(l1_v_ptr, l1_v_ref)
        l1_p = tla.make_tensor(
            l1_p_ptr,
            tla.make_layout_with_tag((q_block, kv_block), io_dtype, tla.arch.zN),
        )
        ub_s = tla.make_tensor(
            ub_s_ptr,
            tla.make_layout(
                tla.make_shape(rows_here, kv_block),
                tla.make_stride(kv_block, c1),
            ),
        )
        ub_now_max = tla.make_tensor(
            ub_now_max_ptr,
            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
        )
        ub_last_max = tla.make_tensor(
            ub_last_max_ptr,
            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
        )
        ub_sum = tla.make_tensor(
            ub_sum_ptr,
            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
        )
        ub_dm = tla.make_tensor(
            ub_dm_ptr,
            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
        )
        ub_blk_sum = tla.make_tensor(
            ub_blk_sum_ptr,
            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
        )
        ub_p_row = tla.make_tensor(
            ub_p_ptr,
            tla.make_layout(
                tla.make_shape(q_sub + c1, kv_block),
                tla.make_stride(kv_block, c1),
            ),
        )
        ub_p_zN_full = tla.make_tensor_like(ub_p_ptr, ub_p_row, tla.arch.zNUnAlign)
        ub_p_zN = tla.tile_view(
            ub_p_zN_full,
            tla.make_shape(q_sub, kv_block),
            tla.make_coord(c0, c0),
        )
        ub_p_copy = tla.tile_view(
            ub_p_zN_full,
            tla.make_shape(q_sub, kv_block),
            tla.make_coord(c0, c0),
        )
        l1_p_sub = tla.tile_view(
            l1_p,
            tla.make_shape(q_sub, kv_block),
            tla.make_coord(vec_idx, c0),
        )
        ub_idx_f32 = tla.make_tensor(
            tla.recast_ptr(ub_idx_ptr, dtype=tla.Float32),
            tla.make_layout(tla.make_shape(kv_block), tla.make_stride(c1)),
        )
        ub_otmp_re = tla.make_tensor(
            ub_otmp_ptr,
            tla.make_layout(
                tla.make_shape(rows_here, head_dim),
                tla.make_stride(head_dim, c1),
            ),
        )
        ub_o_re = tla.make_tensor(
            ub_o_ptr,
            tla.make_layout(
                tla.make_shape(rows_here, head_dim),
                tla.make_stride(head_dim, c1),
            ),
        )

        with tla.cube():
            q_gm = tla.make_tensor(
                mem_q.ptr + (row * heads + head_base) * head_dim,
                tla.make_layout(
                    tla.make_shape(q_tile_size, head_dim),
                    tla.make_stride(head_dim, c1),
                ),
            )
            l1_q_valid = tla.tile_view(
                l1_q,
                tla.make_shape(q_tile_size, head_dim),
                tla.make_coord(c0, c0),
            )
            mutex_l1_q.lock(pipe=tla.arch.MTE2)
            tla.copy(l1_q_valid, q_gm)
            mutex_l1_q.unlock(pipe=tla.arch.MTE2)
            mutex_l1_q.lock(pipe=tla.arch.MTE1)

        lo = tla.as_numeric(0)
        hi = tla.as_numeric(width - c1)
        for _bs in tla.range(12):
            mid = (lo + hi) // c2
            if mem_idx[row * width + mid] >= c0:
                lo = mid + c1
            else:
                hi = mid
        valid_tokens = lo
        max_idx = tla.as_numeric(-1)
        if valid_tokens >= c1:
            max_idx = mem_idx[row * width + valid_tokens - c1]
        ctx_tokens = ((max_idx // c16) + c1) * c16
        valid_tiles = (ctx_tokens + kv_block - c1) // kv_block
        if valid_tiles < c1:
            valid_tiles = tla.as_numeric(c1)

        for tile in tla.range(valid_tiles):
            tile_base = tile * kv_block
            kv_tile_size = ctx_tokens - tile_base
            if kv_tile_size > kv_block:
                kv_tile_size = tla.as_numeric(kv_block)

            with tla.cube():
                mutex_l1_k.lock(pipe=tla.arch.MTE2)
                nk16 = kv_tile_size // c16
                for gk in tla.range(nk16):
                    tk = tile_base + gk * c16
                    pk = tk // page_size
                    ok = tk - pk * page_size
                    phk = mem_bt[req * num_pages + pk]
                    srck = phk * kv_blk_stride + ok * kv_tok_stride
                    k_src16 = tla.make_tensor(
                        mem_k.ptr + srck,
                        tla.make_layout(
                            tla.make_shape(head_dim, c16),
                            tla.make_stride(c1, kv_tok_stride),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    k_dst16 = tla.tile_view(
                        l1_k,
                        tla.make_shape(head_dim, c16),
                        tla.make_coord(c0, gk),
                    )
                    tla.copy(k_dst16, k_src16)
                mutex_l1_k.unlock(pipe=tla.arch.MTE2)

                l0c_s = tla.make_tensor(
                    l0c_s_ptr,
                    tla.make_layout_with_tag(
                        (q_block, kv_block), tla.Float32, tla.arch.L0Clayout
                    ),
                )
                l1_q_c0 = tla.tile_view(
                    l1_q,
                    tla.make_shape(q_block, half_dim),
                    tla.make_coord(c0, c0),
                )
                l0_q_c0 = tla.make_tensor_like(l0a_q_ptr, l1_q_c0)
                mutex_l0a_q.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_q_c0, l1_q_c0)
                mutex_l0a_q.unlock(pipe=tla.arch.MTE1)
                l1_k_c0 = tla.tile_view(
                    l1_k,
                    tla.make_shape(half_dim, kv_block),
                    tla.make_coord(c0, c0),
                )
                l0_k_c0 = tla.make_tensor_like(l0b_ptr, l1_k_c0)
                mutex_l1_k.lock(pipe=tla.arch.MTE1)
                mutex_l0b.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_k_c0, l1_k_c0)
                mutex_l0b.unlock(pipe=tla.arch.MTE1)
                mutex_l0a_q.lock(pipe=tla.arch.CUBE)
                mutex_l0b.lock(pipe=tla.arch.CUBE)
                mutex_l0c_s.lock(pipe=tla.arch.CUBE)
                tla.mmad(l0c_s, l0_q_c0, l0_k_c0, init_c=True)
                mutex_l0a_q.unlock(pipe=tla.arch.CUBE)
                mutex_l0b.unlock(pipe=tla.arch.CUBE)

                l1_q_c1 = tla.tile_view(
                    l1_q,
                    tla.make_shape(q_block, half_dim),
                    tla.make_coord(c0, c1),
                )
                l0_q_c1 = tla.make_tensor_like(l0a_q_ptr, l1_q_c1)
                mutex_l0a_q.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_q_c1, l1_q_c1)
                mutex_l0a_q.unlock(pipe=tla.arch.MTE1)
                l1_k_c1 = tla.tile_view(
                    l1_k,
                    tla.make_shape(half_dim, kv_block),
                    tla.make_coord(c1, c0),
                )
                l0_k_c1 = tla.make_tensor_like(l0b_ptr, l1_k_c1)
                mutex_l0b.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_k_c1, l1_k_c1)
                mutex_l0b.unlock(pipe=tla.arch.MTE1)
                mutex_l1_k.unlock(pipe=tla.arch.MTE1)
                mutex_l0a_q.lock(pipe=tla.arch.CUBE)
                mutex_l0b.lock(pipe=tla.arch.CUBE)
                tla.mmad(l0c_s, l0_q_c1, l0_k_c1, init_c=False)
                mutex_l0a_q.unlock(pipe=tla.arch.CUBE)
                mutex_l0b.unlock(pipe=tla.arch.CUBE)
                mutex_l0c_s.unlock(pipe=tla.arch.CUBE)

                tla.cross_core_wait_flag(s_free, tla.arch.FIX)
                mutex_l0c_s.lock(pipe=tla.arch.FIX)
                ub_s_fix = tla.make_tensor(
                    ub_s_ptr,
                    tla.make_layout(
                        tla.make_shape(q_block, kv_block),
                        tla.make_stride(kv_block, c1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    ub_s_fix,
                    l0c_s,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                    ),
                )
                mutex_l0c_s.unlock(pipe=tla.arch.FIX)
                tla.cross_core_set_flag(s_ready, tla.arch.FIX)

            with tla.cube():
                mutex_l1_v.lock(pipe=tla.arch.MTE2)
                nv16 = kv_tile_size // c16
                for gv in tla.range(nv16):
                    tv = tile_base + gv * c16
                    pv = tv // page_size
                    ov = tv - pv * page_size
                    phv = mem_bt[req * num_pages + pv]
                    srcv = phv * kv_blk_stride + ov * kv_tok_stride
                    v_src16 = tla.make_tensor(
                        mem_v.ptr + srcv,
                        tla.make_layout(
                            tla.make_shape(c16, head_dim),
                            tla.make_stride(kv_tok_stride, c1),
                        ),
                    )
                    v_dst16 = tla.tile_view(
                        l1_v,
                        tla.make_shape(c16, head_dim),
                        tla.make_coord(gv, c0),
                    )
                    tla.copy(v_dst16, v_src16)
                mutex_l1_v.unlock(pipe=tla.arch.MTE2)

            with tla.vector():
                tla.cross_core_wait_flag(s_ready, tla.arch.VECTOR)

                idx_gm = tla.make_tensor(
                    mem_idx.ptr + row * width + tile_base,
                    tla.make_layout(
                        tla.make_shape(c1, kv_tile_size),
                        tla.make_stride(width, c1),
                    ),
                )
                ub_idx_t = tla.make_tensor(
                    ub_idx_ptr,
                    tla.make_layout(
                        tla.make_shape(c1, kv_block),
                        tla.make_stride(kv_block, c1),
                    ),
                )
                ub_idx_valid = tla.tile_view(
                    ub_idx_t,
                    tla.make_shape(c1, kv_tile_size),
                    tla.make_coord(c0, c0),
                )
                mutex_ub_idx.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_idx_valid, idx_gm)
                mutex_ub_idx.unlock(pipe=tla.arch.MTE2)

                n0 = kv_tile_size
                if n0 > vl_f32:
                    n0 = tla.as_numeric(vl_f32)
                n1 = kv_tile_size - vl_f32
                if n1 < c0:
                    n1 = tla.as_numeric(0)
                if n1 > vl_f32:
                    n1 = tla.as_numeric(vl_f32)

                mutex_ub_idx.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    one_mask = tla.update_mask(1, dtype=tla.Float32)[0]
                    mask_n0 = tla.update_mask(n0, dtype=tla.Float32)[0]
                    mask_n1 = tla.update_mask(n1, dtype=tla.Float32)[0]
                    min_reg = tla.full(min_value, dtype=tla.Float32)
                    for i0 in tla.range(rows_here):
                        ub_s_i0 = tla.tile_view(
                            ub_s,
                            tla.make_shape(c1, vl_f32),
                            tla.make_coord(i0, c0),
                        )
                        ub_s_i1 = tla.tile_view(
                            ub_s,
                            tla.make_shape(c1, vl_f32),
                            tla.make_coord(i0, c1),
                        )
                        s_reg0 = ub_s_i0.load()
                        s_reg1 = ub_s_i1.load()
                        s_reg0 = tla.mul(s_reg0, softmax_scale, mask=pregFull)
                        s_reg1 = tla.mul(s_reg1, softmax_scale, mask=pregFull)
                        s_reg0 = tla.where(mask_n0, s_reg0, min_reg)
                        s_reg1 = tla.where(mask_n1, s_reg1, min_reg)
                        ub_s_i0.store(s_reg0, mask=pregFull)
                        ub_s_i1.store(s_reg1, mask=pregFull)
                        tile_max_reg = tla.max(s_reg0, s_reg1, mask=pregFull)
                        row_max_reg = tile_max_reg.reduce(
                            tla.ReductionOp.MAX, mask=pregFull
                        )
                        ub_last_max_i = tla.tile_view(
                            ub_last_max, tla.make_shape(1), tla.make_coord(i0)
                        )
                        ub_last_max_i.store(
                            row_max_reg,
                            params=tla.params.UnalignStoreParams(),
                            mask=one_mask,
                        )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE,
                        tla.params.MemType.VEC_LOAD,
                    )
                mutex_ub_idx.unlock(pipe=tla.arch.VECTOR)

                with tla.vec.func(mode="simd"):
                    pregFull2 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    ub_now_max_v = tla.make_tensor(
                        ub_now_max_ptr,
                        tla.make_layout(tla.make_shape(vl_f32), tla.make_stride(c1)),
                    )
                    ub_last_max_v = tla.make_tensor(
                        ub_last_max_ptr,
                        tla.make_layout(tla.make_shape(vl_f32), tla.make_stride(c1)),
                    )
                    ub_sum_v = tla.make_tensor(
                        ub_sum_ptr,
                        tla.make_layout(tla.make_shape(vl_f32), tla.make_stride(c1)),
                    )
                    ub_dm_v = tla.make_tensor(
                        ub_dm_ptr,
                        tla.make_layout(tla.make_shape(vl_f32), tla.make_stride(c1)),
                    )
                    old_max_v = ub_now_max_v.load()
                    blk_max_v = ub_last_max_v.load()
                    glob_max_v = tla.max(old_max_v, blk_max_v, mask=pregFull2)
                    dm_v = tla.exp(
                        tla.sub(old_max_v, glob_max_v, mask=pregFull2),
                        mask=pregFull2,
                    )
                    sum_old_v = ub_sum_v.load()
                    upd_sum_v = tla.mul(sum_old_v, dm_v, mask=pregFull2)
                    ub_now_max_v.store(glob_max_v, mask=pregFull2)
                    ub_dm_v.store(dm_v, mask=pregFull2)
                    ub_upd_sum_v = tla.make_tensor(
                        ub_upd_sum_ptr,
                        tla.make_layout(tla.make_shape(vl_f32), tla.make_stride(c1)),
                    )
                    ub_upd_sum_v.store(upd_sum_v, mask=pregFull2)
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE,
                        tla.params.MemType.VEC_LOAD,
                    )

                mutex_ub_p.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    pregFull3 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=io_dtype)
                    one_mask3 = tla.update_mask(1, dtype=tla.Float32)[0]
                    for i2 in tla.range(rows_here):
                        ub_now_max_i = tla.tile_view(
                            ub_now_max, tla.make_shape(1), tla.make_coord(i2)
                        )
                        max_reg = ub_now_max_i.load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        ub_s_row = tla.tile_view(
                            ub_s,
                            tla.make_shape(c1, 2 * vl_f32),
                            tla.make_coord(i2, c0),
                        )
                        s_odd, s_even = ub_s_row.load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                            )
                        )
                        exp_odd = tla.exp(
                            tla.sub(s_odd, max_reg, mask=pregFull3),
                            mask=pregFull3,
                        )
                        exp_even = tla.exp(
                            tla.sub(s_even, max_reg, mask=pregFull3),
                            mask=pregFull3,
                        )
                        sum_reg = tla.add(exp_odd, exp_even, mask=pregFull3)
                        block_sum = sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull3)
                        ub_blk_sum_i = tla.tile_view(
                            ub_blk_sum, tla.make_shape(1), tla.make_coord(i2)
                        )
                        ub_blk_sum_i.store(
                            block_sum,
                            params=tla.params.UnalignStoreParams(),
                            mask=one_mask3,
                        )
                        p_even = exp_even.to(io_dtype, cast_one, mask=pregFull3)
                        p_odd = exp_odd.to(io_dtype, cast_zero, mask=pregFull3)
                        p_zn = tla.bitwise_or(p_even, p_odd, mask=preg_all_b16)
                        ub_p_zn_i = tla.tile_view(
                            ub_p_zN,
                            tla.make_shape(c1, 2 * vl_f32),
                            tla.make_coord(i2, c0),
                        )
                        ub_p_zn_i.store(
                            p_zn,
                            params=tla.params.BlockStoreParams(block_stride=q_sub + c1),
                            mask=preg_all_b16,
                        )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE,
                        tla.params.MemType.VEC_LOAD,
                    )
                mutex_ub_p.unlock(pipe=tla.arch.VECTOR)

                with tla.vec.func(mode="simd"):
                    pregFull4 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    ub_blk_sum_v = tla.make_tensor(
                        ub_blk_sum_ptr,
                        tla.make_layout(tla.make_shape(vl_f32), tla.make_stride(c1)),
                    )
                    ub_sum_v2 = tla.make_tensor(
                        ub_sum_ptr,
                        tla.make_layout(tla.make_shape(vl_f32), tla.make_stride(c1)),
                    )
                    ub_upd_sum_v2 = tla.make_tensor(
                        ub_upd_sum_ptr,
                        tla.make_layout(tla.make_shape(vl_f32), tla.make_stride(c1)),
                    )
                    blk_sum_v = ub_blk_sum_v.load()
                    new_sum_v = tla.add(ub_upd_sum_v2.load(), blk_sum_v, mask=pregFull4)
                    ub_sum_v2.store(new_sum_v, mask=pregFull4)
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE,
                        tla.params.MemType.VEC_LOAD,
                    )

                tla.cross_core_set_flag(s_free, tla.arch.VECTOR)

                mutex_ub_p.lock(pipe=tla.arch.MTE3)
                tla.copy(l1_p_sub, ub_p_copy)
                mutex_ub_p.unlock(pipe=tla.arch.MTE3)
                tla.cross_core_set_flag(p_ready, tla.arch.MTE3)

                tla.cross_core_wait_flag(o_ready, tla.arch.VECTOR)
                ub_dm_re = tla.make_tensor(
                    ub_dm_ptr,
                    tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
                )
                with tla.vec.func(mode="simd"):
                    pregFull5 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    for i3 in tla.range(rows_here):
                        ub_dm_i = tla.tile_view(
                            ub_dm_re, tla.make_shape(1), tla.make_coord(i3)
                        )
                        dm_reg = ub_dm_i.load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        for j3 in tla.range(num_dim_chunks):
                            acc_tile = tla.tile_view(
                                ub_o_re,
                                tla.make_shape(c1, vl_f32),
                                tla.make_coord(i3, j3),
                            )
                            otmp_tile = tla.tile_view(
                                ub_otmp_re,
                                tla.make_shape(c1, vl_f32),
                                tla.make_coord(i3, j3),
                            )
                            acc_reg = acc_tile.load()
                            otmp_reg = otmp_tile.load()
                            scaled = tla.mul(acc_reg, dm_reg, mask=pregFull5)
                            merged = tla.add(scaled, otmp_reg, mask=pregFull5)
                            acc_tile.store(merged, mask=pregFull5)
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE,
                        tla.params.MemType.VEC_LOAD,
                    )
                if tile + c1 < valid_tiles:
                    tla.cross_core_set_flag(o_free, tla.arch.VECTOR)

            with tla.cube():
                tla.cross_core_wait_flag(p_ready, tla.arch.MTE1)
                l0_p = tla.make_tensor_like(l0a_p_ptr, l1_p)
                mutex_l0a_p.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_p, l1_p)
                mutex_l0a_p.unlock(pipe=tla.arch.MTE1)
                l1_v_pv = tla.make_tensor_like(l1_v_ptr, l1_v)
                l0_v = tla.make_tensor_like(l0b_ptr, l1_v_pv)
                mutex_l1_v.lock(pipe=tla.arch.MTE1)
                mutex_l0b.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_v, l1_v_pv)
                mutex_l0b.unlock(pipe=tla.arch.MTE1)
                mutex_l1_v.unlock(pipe=tla.arch.MTE1)
                l0c_o = tla.make_tensor(
                    l0c_o_ptr,
                    tla.make_layout_with_tag(
                        (q_block, head_dim), tla.Float32, tla.arch.L0Clayout
                    ),
                )
                mutex_l0a_p.lock(pipe=tla.arch.CUBE)
                mutex_l0b.lock(pipe=tla.arch.CUBE)
                mutex_l0c_o.lock(pipe=tla.arch.CUBE)
                tla.mmad(l0c_o, l0_p, l0_v, init_c=True)
                mutex_l0a_p.unlock(pipe=tla.arch.CUBE)
                mutex_l0b.unlock(pipe=tla.arch.CUBE)
                mutex_l0c_o.unlock(pipe=tla.arch.CUBE)

                tla.cross_core_wait_flag(o_free, tla.arch.FIX)
                mutex_l0c_o.lock(pipe=tla.arch.FIX)
                ub_otmp_fix = tla.make_tensor(
                    ub_otmp_ptr,
                    tla.make_layout(
                        tla.make_shape(q_block, head_dim),
                        tla.make_stride(head_dim, c1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    ub_otmp_fix,
                    l0c_o,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                    ),
                )
                mutex_l0c_o.unlock(pipe=tla.arch.FIX)
                tla.cross_core_set_flag(o_ready, tla.arch.FIX)

        with tla.cube():
            mutex_l1_q.unlock(pipe=tla.arch.MTE1)

        with tla.vector():
            ub_sum_out = tla.make_tensor(
                ub_sum_ptr,
                tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
            )
            ub_out_t = tla.make_tensor(
                ub_out_ptr,
                tla.make_layout(
                    tla.make_shape(rows_here, head_dim),
                    tla.make_stride(head_dim, c1),
                ),
            )
            mutex_ub_out.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                pregFull6 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_all_b16_6 = tla.create_mask(pattern=tla.mask.ALL, dtype=io_dtype)
                one_reg = tla.full(1.0, dtype=tla.Float32)
                zero_reg6 = tla.full(0.0, dtype=tla.Float32)
                for i6 in tla.range(rows_here):
                    ub_sum_i6 = tla.tile_view(
                        ub_sum_out, tla.make_shape(1), tla.make_coord(i6)
                    )
                    sum_reg6 = ub_sum_i6.load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                        )
                    )
                    pos_mask = tla.cmp(sum_reg6, 0.0, mode="gt", mask=pregFull6)
                    inv_raw = tla.div(one_reg, sum_reg6, mask=pregFull6)
                    inv_reg = tla.where(pos_mask, inv_raw, zero_reg6)
                    for j6 in tla.range(num_dim_chunks):
                        acc_tile6 = tla.tile_view(
                            ub_o_re,
                            tla.make_shape(c1, vl_f32),
                            tla.make_coord(i6, j6),
                        )
                        acc_reg6 = acc_tile6.load()
                        out_reg6 = tla.mul(acc_reg6, inv_reg, mask=pregFull6)
                        acc_tile6.store(out_reg6, mask=pregFull6)
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE,
                    tla.params.MemType.VEC_LOAD,
                )
                for i7 in tla.range(rows_here):
                    for j7 in tla.range(num_dim_chunks // 2):
                        acc_tile7_0 = tla.tile_view(
                            ub_o_re,
                            tla.make_shape(c1, vl_f32),
                            tla.make_coord(i7, 2 * j7),
                        )
                        acc_tile7_1 = tla.tile_view(
                            ub_o_re,
                            tla.make_shape(c1, vl_f32),
                            tla.make_coord(i7, 2 * j7 + c1),
                        )
                        out_tile7 = tla.tile_view(
                            ub_out_t,
                            tla.make_shape(c1, 2 * vl_f32),
                            tla.make_coord(i7, j7),
                        )
                        acc_reg7_0 = acc_tile7_0.load()
                        acc_reg7_1 = acc_tile7_1.load()
                        out_reg7_0 = acc_reg7_0.to(io_dtype, cast_zero, mask=pregFull6)
                        out_reg7_1 = acc_reg7_1.to(io_dtype, cast_zero, mask=pregFull6)
                        packed7, _ = tla.deinterleave(out_reg7_0, out_reg7_1)
                        out_tile7.store(packed7, mask=preg_all_b16_6)

            mutex_ub_out.unlock(pipe=tla.arch.VECTOR)
            mutex_ub_out.lock(pipe=tla.arch.MTE3)
            if rows_here > c0:
                o_gm = tla.make_tensor(
                    mem_o.ptr + (row * heads + head_base + vec_idx * q_sub) * head_dim,
                    tla.make_layout(
                        tla.make_shape(rows_here, head_dim),
                        tla.make_stride(head_dim, c1),
                    ),
                )
                tla.copy(o_gm, ub_out_t)
            mutex_ub_out.unlock(pipe=tla.arch.MTE3)
            tla.cross_core_set_flag(o_free, tla.arch.MTE3)

    with tla.cube():
        tla.cross_core_wait_flag(s_free, tla.arch.FIX)
        tla.cross_core_wait_flag(o_free, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.pipe_barrier(tla.pipes.ALL)



import torch
import torch_npu  # noqa: F401  -- registers torch.npu
from catlass.tla.runtime import from_dlpack

Q_BLOCK = 128
KV_BLOCK = 128
HEAD_DIM = 256
NPU_ARCH = "--npu-arch 3510"

_EXECUTOR_CACHE = {}


def _tla_tensor(value):
    return from_dlpack(
        value, layout_tag=tla.arch.RowMajor, origin_shape=(value.numel(),)
    ).mark_layout_dynamic()


def run(
    q, k_cache, v_cache, logical_indices, block_table, token_to_req, softmax_scale=None
):
    if q.ndim != 3:
        raise ValueError("q must have shape [rows, query_heads, head_dim]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("k_cache and v_cache must have identical 4D shapes")
    if q.shape[2] != k_cache.shape[3]:
        raise ValueError("q and cache head dimensions must match")
    if q.shape[2] != HEAD_DIM:
        raise ValueError("head_dim must be 256")
    if k_cache.shape[2] != 1:
        raise ValueError("kv_heads must be 1")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q must be float16 or bfloat16")
    if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise ValueError("q, k_cache and v_cache must share one dtype")
    if logical_indices.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise ValueError("logical_indices and block_table must be int32")
    if token_to_req.dtype != torch.int32:
        raise ValueError("token_to_req must be int32")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if v_cache.stride() != k_cache.stride():
        raise ValueError("k_cache and v_cache must share strides")
    kv_blk_stride = int(k_cache.stride(0))
    kv_tok_stride = int(k_cache.stride(1))
    rows, heads, _ = q.shape
    if logical_indices.shape != (rows, logical_indices.shape[1]):
        raise ValueError("logical_indices must contain one row per query")
    if block_table.ndim != 2:
        raise ValueError("block_table must be two-dimensional")
    if token_to_req.shape != (rows,):
        raise ValueError("token_to_req must contain one entry per query row")
    scale_value = 0.0625
    if isinstance(softmax_scale, (int, float)):
        scale_value = float(softmax_scale)

    output = torch.empty_like(q)
    if rows == 0 or logical_indices.shape[1] == 0:
        return output

    page_size = int(k_cache.shape[1])
    num_pages = int(block_table.shape[1])
    width = int(logical_indices.shape[1])
    head_groups = (heads + Q_BLOCK - 1) // Q_BLOCK

    q_t = _tla_tensor(q)
    k_t = _tla_tensor(k_cache)
    v_t = _tla_tensor(v_cache)
    o_t = _tla_tensor(output)
    idx_t = _tla_tensor(logical_indices)
    bt_t = _tla_tensor(block_table)
    req_t = _tla_tensor(token_to_req)
    scale_t = tla.Float32(scale_value)
    rows_t = tla.Int32(rows)
    heads_t = tla.Int32(heads)
    width_t = tla.Int32(width)
    page_t = tla.Int32(page_size)
    pages_t = tla.Int32(num_pages)
    blk_stride_t = tla.Int32(kv_blk_stride)
    tok_stride_t = tla.Int32(kv_tok_stride)
    groups_t = tla.Int32(head_groups)

    key = (str(q.dtype), str(q.device))
    executor = _EXECUTOR_CACHE.get(key)
    if executor is None:
        executor = tla.compile(
            qwen_sparse_attention_kernel,
            q_t,
            k_t,
            v_t,
            o_t,
            idx_t,
            bt_t,
            req_t,
            scale_t,
            rows_t,
            heads_t,
            width_t,
            page_t,
            pages_t,
            blk_stride_t,
            tok_stride_t,
            groups_t,
            HEAD_DIM,
            Q_BLOCK,
            KV_BLOCK,
            options=NPU_ARCH,
        )
        _EXECUTOR_CACHE[key] = executor

    device_properties = torch_npu.npu.get_device_properties(q.device)
    cube_core_num = int(device_properties.cube_core_num)
    if cube_core_num < 1:
        raise RuntimeError("NPU device reports no available Cube cores")
    total_tasks = rows * head_groups
    launch_blocks = min(total_tasks, cube_core_num)
    executor(
        q_t,
        k_t,
        v_t,
        o_t,
        idx_t,
        bt_t,
        req_t,
        scale_t,
        rows_t,
        heads_t,
        width_t,
        page_t,
        pages_t,
        blk_stride_t,
        tok_stride_t,
        groups_t,
        block_num=launch_blocks,
    )
    return output
