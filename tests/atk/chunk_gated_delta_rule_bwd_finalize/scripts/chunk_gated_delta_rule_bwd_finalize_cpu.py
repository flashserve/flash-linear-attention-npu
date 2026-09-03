def chunk_gated_delta_rule_bwd_finalize_golden(
    q, k, v, v_new, do, du, g, beta, h, dh, a, *,
    q_rstd=None, k_rstd=None, beta_raw=None, cu_seqlens=None,
    chunk_indices=None, scale=None, chunk_size=64,
    use_qk_l2_norm_in_kernel=False, use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False, use_gate_in_kernel=False, state_v_first=False, use_exp2=True,
):
    """CPU reference for the fused gated-delta-rule backward finalize range.

    This matches the upstream chain from ``chunk_bwd_dqkwg`` through
    ``fused_beta_sigmoid_bwd``. Compared with upstream FLA, sequence and head
    axes are swapped: vectors use ``[B, H, T, D]``, states use
    ``[B, H, NT, K, V]``, and ``a`` uses ``[B, HV, T, BT]``. The function
    returns ``dq, dk, dv, dbeta, dg`` in that NPU-facing layout.
    """
    import math
    import torch

    dim = 128
    if int(chunk_size) != 64:
        raise ValueError("chunk_size must be 64.")
    if scale is None:
        scale = 1.0 / math.sqrt(dim)
    if (cu_seqlens is None) != (chunk_indices is None):
        raise ValueError("cu_seqlens and chunk_indices must be both None or both provided.")
    if use_qk_l2_norm_in_kernel and (q_rstd is None or k_rstd is None):
        raise ValueError("q_rstd and k_rstd are required when Q/K L2Norm backward is enabled.")
    if use_beta_sigmoid_in_kernel and beta_raw is None:
        raise ValueError("beta_raw is required when beta sigmoid backward is enabled.")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True.")
    if bool(use_gate_in_kernel):
        raise ValueError("use_gate_in_kernel only supports False.")
    if not bool(use_exp2):
        raise ValueError("use_exp2 only supports True.")
    if g.dtype != beta.dtype:
        raise ValueError("g and beta must use the same dtype.")

    named_tensors = {
        "q": q, "k": k, "v": v, "v_new": v_new, "do": do, "du": du,
        "g": g, "beta": beta, "h": h, "dh": dh, "a": a,
    }
    for name, tensor in named_tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor for the CPU golden.")

    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError("q and k must have the same [B, HK, T, K] shape.")
    if v.ndim != 4 or v_new.shape != v.shape or do.shape != v.shape or du.shape != v.shape:
        raise ValueError("v, v_new, do and du must have the same [B, HV, T, V] shape.")
    batch, key_heads, total_tokens, key_dim = q.shape
    value_heads = v.shape[1]
    if v.shape[0] != batch or v.shape[2] != total_tokens:
        raise ValueError("q/k and value tensors must share B and T.")
    if key_dim != dim or v.shape[-1] != dim:
        raise ValueError(f"K and V dimensions must both be {dim}.")
    if value_heads % key_heads != 0:
        raise ValueError("HV must be divisible by HK.")
    scalar_shape = (batch, value_heads, total_tokens)
    if g.shape != scalar_shape or beta.shape != scalar_shape:
        raise ValueError("g and beta must have shape [B, HV, T].")
    if beta_raw is not None:
        if tuple(beta_raw.shape) != scalar_shape:
            raise ValueError("beta_raw must have shape [B, HV, T].")
        if beta_raw.device.type != "cpu":
            raise ValueError("beta_raw must be a CPU tensor for the CPU golden.")
    norm_shape = (batch, key_heads, total_tokens)
    for name, tensor in (("q_rstd", q_rstd), ("k_rstd", k_rstd)):
        if tensor is not None:
            if tuple(tensor.shape) != norm_shape or tensor.dtype != torch.float32:
                raise ValueError(f"{name} must be a float32 tensor with shape {norm_shape}.")
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must be a CPU tensor for the CPU golden.")

    def as_int_list(value, name):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim != 1:
                raise ValueError(f"{name} must be a 1-D tensor or sequence.")
            value = value.detach().cpu().tolist()
        return [int(item) for item in value]

    cu_list = as_int_list(cu_seqlens, "cu_seqlens")
    chunk_list = as_int_list(chunk_indices, "chunk_indices")
    tasks = []
    if cu_list is None:
        state_chunk_num = (total_tokens + chunk_size - 1) // chunk_size
        for batch_idx in range(batch):
            for chunk_idx in range(state_chunk_num):
                token_start = chunk_idx * chunk_size
                tasks.append((
                    batch_idx, token_start,
                    min(chunk_size, total_tokens - token_start), chunk_idx,
                ))
    else:
        if batch != 1:
            raise ValueError("varlen mode requires B == 1.")
        if len(cu_list) < 2 or cu_list[0] != 0 or cu_list[-1] != total_tokens:
            raise ValueError("cu_seqlens must start at 0 and end at T.")
        if any(left > right for left, right in zip(cu_list, cu_list[1:])):
            raise ValueError("cu_seqlens must be nondecreasing.")
        if len(chunk_list) % 2 != 0:
            raise ValueError("chunk_indices must contain (sequence, chunk) pairs.")
        for task_idx in range(len(chunk_list) // 2):
            seq_idx = chunk_list[2 * task_idx]
            chunk_idx = chunk_list[2 * task_idx + 1]
            token_start = cu_list[seq_idx] + chunk_idx * chunk_size
            chunk_len = min(chunk_size, cu_list[seq_idx + 1] - token_start)
            tasks.append((0, token_start, chunk_len, task_idx))
        state_chunk_num = len(tasks)

    expected_state_shape = (batch, value_heads, state_chunk_num, dim, dim)
    if tuple(h.shape) != expected_state_shape or tuple(dh.shape) != expected_state_shape:
        raise ValueError(f"h and dh must have shape {expected_state_shape}.")
    if tuple(a.shape) != (batch, value_heads, total_tokens, chunk_size):
        raise ValueError("a must have shape [B, HV, T, 64].")

    head_ratio = value_heads // key_heads
    work_dtype = q.dtype
    g_fp32 = g.float()
    beta_fp32 = beta.float()
    g_exp = torch.exp2(g_fp32)
    bg = beta_fp32 * g_exp
    hv_to_hk = torch.arange(value_heads, dtype=torch.int64) // head_ratio
    k_hv = k[:, hv_to_hk]
    kbg = (k_hv.float() * bg[:, :, :, None]).to(work_dtype)
    vb = (v.float() * beta_fp32[:, :, :, None]).to(work_dtype)
    kb = (k_hv.float() * beta_fp32[:, :, :, None]).to(work_dtype)

    matrix_shape = a.shape
    hv_vector_shape = (batch, value_heads, total_tokens, dim)
    d_a_u_lower = torch.zeros(matrix_shape, dtype=work_dtype)
    d_a_w0 = torch.zeros(matrix_shape, dtype=work_dtype)
    dvb = torch.empty_like(v)
    d_a0 = torch.zeros(matrix_shape, dtype=work_dtype)
    d_a1 = torch.zeros(matrix_shape, dtype=work_dtype)
    a2 = torch.zeros(matrix_shape, dtype=work_dtype)
    dv = torch.empty_like(v)
    db_v_partial = torch.empty(scalar_shape, dtype=torch.float32)
    d_a2 = torch.zeros(matrix_shape, dtype=work_dtype)
    dkbg0 = torch.empty(hv_vector_shape, dtype=work_dtype)
    d_a = torch.zeros(matrix_shape, dtype=work_dtype)
    db_v = torch.empty(scalar_shape, dtype=torch.float32)
    dg_prepare = torch.empty(scalar_shape, dtype=torch.float32)
    dkb = torch.empty(hv_vector_shape, dtype=work_dtype)
    dkb_t = torch.empty(hv_vector_shape, dtype=work_dtype)
    state_term = torch.empty((batch, value_heads, h.shape[2]), dtype=torch.float32)
    ds0 = torch.zeros(matrix_shape, dtype=work_dtype)
    dk_prepare = torch.empty(hv_vector_shape, dtype=work_dtype)
    db_prepare = torch.empty(scalar_shape, dtype=torch.float32)
    dbeta = torch.empty(scalar_shape, dtype=torch.float32)
    ds = torch.zeros(matrix_shape, dtype=work_dtype)
    do_g = torch.empty_like(v)
    v_decay = torch.empty_like(v)
    dq_hv = torch.empty(hv_vector_shape, dtype=work_dtype)
    dk_base = torch.empty(hv_vector_shape, dtype=work_dtype)
    dk_intra = torch.empty(hv_vector_shape, dtype=work_dtype)
    dk_raw_hv = torch.empty(hv_vector_shape, dtype=work_dtype)
    dg = torch.empty(scalar_shape, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)

    for batch_idx, token_start, chunk_len, state_chunk_idx in tasks:
        token_end = token_start + chunk_len
        for hv in range(value_heads):
            du_chunk = du[batch_idx, hv, token_start:token_end].float()
            vb_chunk = vb[batch_idx, hv, token_start:token_end].float()
            h_chunk = h[batch_idx, hv, state_chunk_idx].float()
            dh_chunk = dh[batch_idx, hv, state_chunk_idx].float()
            if state_v_first:
                h_chunk = h_chunk.transpose(-1, -2)
                dh_chunk = dh_chunk.transpose(-1, -2)
            d_a_u = torch.matmul(du_chunk, vb_chunk.transpose(-1, -2)).to(work_dtype)
            d_a_u_lower[batch_idx, hv, token_start:token_end, :chunk_len] = torch.tril(
                d_a_u, diagonal=-1
            )
            dw0 = torch.matmul(du_chunk, h_chunk.transpose(-1, -2)).to(work_dtype)
            kbg_chunk = kbg[batch_idx, hv, token_start:token_end].float()
            d_a_w0[batch_idx, hv, token_start:token_end, :chunk_len] = torch.matmul(
                dw0.float(), kbg_chunk.transpose(-1, -2)
            ).to(work_dtype)
            a_chunk = a[batch_idx, hv, token_start:token_end, :chunk_len].float()
            dvb_chunk = torch.matmul(a_chunk.transpose(-1, -2), du_chunk).to(work_dtype)
            dvb[batch_idx, hv, token_start:token_end] = dvb_chunk
            d_a0_chunk = (
                d_a_u_lower[batch_idx, hv, token_start:token_end, :chunk_len].float()
                - torch.tril(
                    d_a_w0[batch_idx, hv, token_start:token_end, :chunk_len].float(),
                    diagonal=-1,
                )
            ).to(work_dtype)
            d_a0[batch_idx, hv, token_start:token_end, :chunk_len] = d_a0_chunk
            d_a1_chunk = torch.matmul(
                d_a0_chunk.float(), a_chunk.transpose(-1, -2)
            ).to(work_dtype)
            d_a1[batch_idx, hv, token_start:token_end, :chunk_len] = d_a1_chunk
            k_chunk = k_hv[batch_idx, hv, token_start:token_end].float()
            a2[batch_idx, hv, token_start:token_end, :chunk_len] = torch.matmul(
                k_chunk, k_chunk.transpose(-1, -2)
            ).to(work_dtype)
            dv[batch_idx, hv, token_start:token_end] = (
                dvb_chunk.float() * beta_fp32[batch_idx, hv, token_start:token_end, None]
            ).to(work_dtype)
            db_v_partial[batch_idx, hv, token_start:token_end] = (
                dvb_chunk.float() * v[batch_idx, hv, token_start:token_end].float()
            ).sum(dim=-1)
            d_a2[batch_idx, hv, token_start:token_end, :chunk_len] = torch.matmul(
                a_chunk.transpose(-1, -2), d_a1_chunk.float()
            ).to(work_dtype)
            dkbg0[batch_idx, hv, token_start:token_end] = torch.matmul(
                a_chunk.transpose(-1, -2), dw0.float()
            ).to(work_dtype)
            gate_chunk = torch.exp2(
                g_fp32[batch_idx, hv, token_start:token_end, None]
                - g_fp32[batch_idx, hv, None, token_start:token_end]
            ).to(work_dtype)
            d_a_chunk = torch.tril(
                -(d_a2[batch_idx, hv, token_start:token_end, :chunk_len].float()
                  * gate_chunk.float()), diagonal=-1,
            ).to(work_dtype)
            d_a[batch_idx, hv, token_start:token_end, :chunk_len] = d_a_chunk
            ad_a_chunk = (
                d_a_chunk.float()
                * a2[batch_idx, hv, token_start:token_end, :chunk_len].float()
                * beta_fp32[batch_idx, hv, token_start:token_end, None]
            ).to(work_dtype)
            dkbg0_chunk = dkbg0[batch_idx, hv, token_start:token_end].float()
            g_exp_chunk = g_exp[batch_idx, hv, token_start:token_end]
            bg_chunk = bg[batch_idx, hv, token_start:token_end]
            db0_chunk = -(dkbg0_chunk * k_chunk * g_exp_chunk[:, None]).sum(dim=-1)
            dg0_chunk = -(dkbg0_chunk * k_chunk * bg_chunk[:, None]).sum(dim=-1)
            db_v[batch_idx, hv, token_start:token_end] = (
                db0_chunk + db_v_partial[batch_idx, hv, token_start:token_end]
            )
            dg_prepare[batch_idx, hv, token_start:token_end] = (
                dg0_chunk + ad_a_chunk.float().sum(dim=-1) - ad_a_chunk.float().sum(dim=-2)
            )
            kb_chunk = kb[batch_idx, hv, token_start:token_end].float()
            dkb[batch_idx, hv, token_start:token_end] = torch.matmul(
                d_a_chunk.float(), k_chunk
            ).to(work_dtype)
            dkb_t[batch_idx, hv, token_start:token_end] = torch.matmul(
                d_a_chunk.float().transpose(-1, -2), kb_chunk
            ).to(work_dtype)
            state_term[batch_idx, hv, state_chunk_idx] = (
                h_chunk * dh_chunk
            ).sum() * torch.exp2(g_fp32[batch_idx, hv, token_end - 1])
            ds0[batch_idx, hv, token_start:token_end, :chunk_len] = torch.matmul(
                do[batch_idx, hv, token_start:token_end].float(),
                v_new[batch_idx, hv, token_start:token_end].float().transpose(-1, -2),
            ).to(work_dtype)
            dk_prepare_chunk = (
                -dkbg0[batch_idx, hv, token_start:token_end].float() * bg_chunk[:, None]
                + dkb[batch_idx, hv, token_start:token_end].float()
                * beta_fp32[batch_idx, hv, token_start:token_end, None]
                + dkb_t[batch_idx, hv, token_start:token_end].float()
            ).to(work_dtype)
            dk_prepare[batch_idx, hv, token_start:token_end] = dk_prepare_chunk
            db_prepare_chunk = db_v[batch_idx, hv, token_start:token_end] + (
                dkb[batch_idx, hv, token_start:token_end].float() * k_chunk
            ).sum(dim=-1)
            db_prepare[batch_idx, hv, token_start:token_end] = db_prepare_chunk
            if use_beta_sigmoid_in_kernel:
                beta_sigmoid = torch.sigmoid(
                    beta_raw[batch_idx, hv, token_start:token_end].float()
                )
                dbeta[batch_idx, hv, token_start:token_end] = (
                    db_prepare_chunk * beta_sigmoid * (1.0 - beta_sigmoid)
                    * (2.0 if allow_neg_eigval else 1.0)
                )
            else:
                dbeta[batch_idx, hv, token_start:token_end] = db_prepare_chunk
            ds_chunk = (
                torch.tril(
                    ds0[batch_idx, hv, token_start:token_end, :chunk_len].float()
                    * gate_chunk.float()
                ) * scale
            ).to(work_dtype)
            ds[batch_idx, hv, token_start:token_end, :chunk_len] = ds_chunk
            do_g_chunk = (
                do[batch_idx, hv, token_start:token_end].float()
                * g_exp_chunk[:, None] * scale
            ).to(work_dtype)
            do_g[batch_idx, hv, token_start:token_end] = do_g_chunk
            decay_chunk = torch.exp2(
                g_fp32[batch_idx, hv, token_end - 1]
                - g_fp32[batch_idx, hv, token_start:token_end]
            )
            v_decay_chunk = (
                v_new[batch_idx, hv, token_start:token_end].float() * decay_chunk[:, None]
            ).to(work_dtype)
            v_decay[batch_idx, hv, token_start:token_end] = v_decay_chunk
            q_chunk = q[batch_idx, hv // head_ratio, token_start:token_end].float()
            dq_hv[batch_idx, hv, token_start:token_end] = (
                torch.matmul(do_g_chunk.float(), h_chunk.transpose(-1, -2))
                + torch.matmul(ds_chunk.float(), k_chunk)
            ).to(work_dtype)
            dk_base[batch_idx, hv, token_start:token_end] = torch.matmul(
                v_decay_chunk.float(),
                dh_chunk.transpose(-1, -2),
            ).to(work_dtype)
            dk_intra[batch_idx, hv, token_start:token_end] = torch.matmul(
                ds_chunk.float().transpose(-1, -2), q_chunk
            ).to(work_dtype)
            dk_raw_chunk = (
                dk_base[batch_idx, hv, token_start:token_end].float()
                + dk_intra[batch_idx, hv, token_start:token_end].float()
                + dk_prepare_chunk.float()
            ).to(work_dtype)
            dk_raw_hv[batch_idx, hv, token_start:token_end] = dk_raw_chunk
            base_dot = (
                dk_base[batch_idx, hv, token_start:token_end].float() * k_chunk
            ).sum(dim=-1)
            dg_chunk = (
                (dq_hv[batch_idx, hv, token_start:token_end].float() * q_chunk).sum(dim=-1)
                - base_dot
                - (dk_intra[batch_idx, hv, token_start:token_end].float() * k_chunk).sum(dim=-1)
                + dg_prepare[batch_idx, hv, token_start:token_end]
            )
            dg_chunk[-1] += state_term[batch_idx, hv, state_chunk_idx] + base_dot.sum()
            dg[batch_idx, hv, token_start:token_end] = torch.flip(
                torch.cumsum(torch.flip(dg_chunk, dims=(0,)), dim=0), dims=(0,)
            )

        for hk in range(key_heads):
            hv_start = hk * head_ratio
            dq_acc = dq_hv[batch_idx, hv_start, token_start:token_end].clone()
            dk_acc = dk_raw_hv[batch_idx, hv_start, token_start:token_end].clone()
            for hv in range(hv_start + 1, min(hv_start + head_ratio, value_heads)):
                dq_acc = (
                    dq_acc.float() + dq_hv[batch_idx, hv, token_start:token_end].float()
                ).to(work_dtype)
                dk_acc = (
                    dk_acc.float() + dk_raw_hv[batch_idx, hv, token_start:token_end].float()
                ).to(work_dtype)
            dq[batch_idx, hk, token_start:token_end] = dq_acc
            dk[batch_idx, hk, token_start:token_end] = dk_acc

    if use_qk_l2_norm_in_kernel:
        q_fp32, dq_fp32 = q.float(), dq.float()
        q_dot = (dq_fp32 * q_fp32).sum(dim=-1, keepdim=True)
        dq = ((dq_fp32 - q_fp32 * q_dot) * q_rstd[..., None]).to(q.dtype)
        k_fp32, dk_fp32 = k.float(), dk.float()
        k_dot = (dk_fp32 * k_fp32).sum(dim=-1, keepdim=True)
        dk = ((dk_fp32 - k_fp32 * k_dot) * k_rstd[..., None]).to(k.dtype)

    return dq, dk, dv, dbeta.to(beta.dtype), dg.to(g.dtype)
