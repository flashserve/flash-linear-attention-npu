# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ctypes backed Python wrappers for FLA NPU Ascend C operators.

This file intentionally contains only concrete operator wrappers and their ABI
quirks.  Shared descriptor, workspace and stream handling lives in ``_runtime``
so a new operator developer only needs to mirror the matching ``aclnn_*.h``
signature here.
"""

from __future__ import annotations

import ctypes

from ._kda_policy import kda_fwd_optional_output_mask
from ._runtime import (
    call_aclnn as _runtime_call_aclnn,
    chunk_num as _chunk_num,
    empty as _empty,
    empty_like as _empty_like,
    optional_bool as _optional_bool,
    optional_float as _optional_float,
    optional_int as _optional_int,
    shape as _shape,
    zeros as _zeros,
)

# Most aclnn functions only receive pointer-sized descriptors and scalar ctypes
# objects, so ctypes can call them without explicit argtypes.  Functions with C
# strings or otherwise ambiguous scalar conversion are listed here to prevent
# ctypes from narrowing or mis-converting arguments.
_GET_WORKSPACE_ARGTYPES = {
    "aclnnCausalConv1dBwd": [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "aclnnSolveTri": [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "aclnnChunkKdaFwd": [
        *([ctypes.c_void_p] * 10),
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_bool,
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_bool,
        *([ctypes.c_void_p] * 11),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "aclnnKdaGateCumsum": [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "aclnnChunkLocalCumsum": [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_bool,
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "aclnnChunkScaledDotKkt": [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
}


def _call_aclnn(name: str, build_args, outputs):
    return _runtime_call_aclnn(
        name,
        build_args,
        outputs,
        get_workspace_argtypes=_GET_WORKSPACE_ARGTYPES.get(name),
    )


def npu_fast_gelu_custom(self):
    out = _empty_like(self)
    return _call_aclnn(
        "aclnnFastGelu",
        lambda ctx: [ctx.tensor(self, "self"), ctx.tensor(out, "out")],
        out,
    )


def npu_fast_gelu_custom_backward(grad, self):
    out = _empty_like(grad)
    return _call_aclnn(
        "aclnnFastGeluBackward",
        lambda ctx: [ctx.tensor(grad, "grad"), ctx.tensor(self, "self"), ctx.tensor(out, "out")],
        out,
    )


def npu_prepare_wy_repr_bwd_full(
    k,
    v,
    beta,
    A,
    dA,
    dw,
    du,
    g,
    chunk_size,
    *,
    cu_seqlens=None,
    chunk_indices=None,
):
    dk = _empty_like(k)
    dv = _empty_like(v)
    dbeta = _empty_like(beta)
    dg = _empty_like(g)
    outputs = (dk, dv, dbeta, dg)
    return _call_aclnn(
        "aclnnPrepareWyReprBwdFull",
        lambda ctx: [
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(beta, "beta"),
            ctx.tensor(A, "A"),
            ctx.tensor(dA, "dA"),
            ctx.tensor(dw, "dw"),
            ctx.tensor(du, "du"),
            ctx.tensor(g, "g"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(dk, "dk"),
            ctx.tensor(dv, "dv"),
            ctx.tensor(dbeta, "dbeta"),
            ctx.tensor(dg, "dg"),
        ],
        outputs,
    )


def npu_chunk_gated_delta_rule_bwd_dhu(
    q,
    k,
    w,
    d_o,
    dv,
    scale,
    chunk_size,
    *,
    g=None,
    gK=None,
    h0=None,
    dht=None,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    transpose_state_layout=False,
):
    q_shape = _shape(q)
    dv_shape = _shape(dv)
    B, _, T, K = q_shape
    Hv, V = dv_shape[1], dv_shape[3]
    NT = _chunk_num(T, int(chunk_size), chunk_indices)
    dh = _empty((B, Hv, NT, K, V), q)
    dh0 = _empty((B, Hv, NT, K, V), q) if h0 is not None else None
    dv2 = _empty_like(dv)
    outputs = (dh, dh0, dv2)
    return _call_aclnn(
        "aclnnChunkGatedDeltaRuleBwdDhu",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(w, "w"),
            ctx.tensor(d_o, "d_o"),
            ctx.tensor(dv, "dv"),
            ctx.tensor(g, "g"),
            ctx.tensor(gK, "gK"),
            ctx.tensor(h0, "h0"),
            ctx.tensor(dht, "dht"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(dh, "dh"),
            ctx.tensor(dh0, "dh0"),
            ctx.tensor(dv2, "dv2"),
        ],
        outputs,
    )


def npu_chunk_bwd_dv_local(
    q,
    k,
    d_o,
    g,
    scale,
    chunk_size,
    *,
    g_gamma=None,
    A=None,
    cu_seqlens=None,
    chunk_indices=None,
):
    out = _empty_like(d_o)
    return _call_aclnn(
        "aclnnChunkBwdDvLocal",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(d_o, "d_o"),
            ctx.tensor(g, "g"),
            ctx.tensor(g_gamma, "g_gamma"),
            ctx.tensor(A, "A"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(out, "out"),
        ],
        out,
    )


def npu_prepare_wy_repr_bwd_da(
    k,
    v,
    beta,
    A,
    dw,
    du,
    g,
    *,
    chunk_size,
    cu_seqlens=None,
    chunk_indices=None,
):
    out = _empty_like(A)
    return _call_aclnn(
        "aclnnPrepareWyReprBwdDa",
        lambda ctx: [
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(beta, "beta"),
            ctx.tensor(A, "A"),
            ctx.tensor(dw, "dw"),
            ctx.tensor(du, "du"),
            ctx.tensor(g, "g"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(out, "dA"),
        ],
        out,
    )


def npu_chunk_bwd_dqkwg(
    q,
    k,
    v,
    g,
    h,
    dox,
    dh,
    dv,
    chunk_size,
    *,
    cu_seqlens=None,
    chunk_indices=None,
    w=None,
    g_gamma=None,
    scale=None,
    use_exp2=None,
    transpose_state_layout=None,
):
    q_shape = _shape(q)
    value_num_heads = int(v.shape[1])
    dq = _empty_like(q)
    dk = _empty_like(k)
    dw = _empty((q_shape[0], value_num_heads, q_shape[2], q_shape[3]), q)
    dg = _empty_like(g)
    outputs = (dq, dk, dw, dg)
    return _call_aclnn(
        "aclnnChunkBwdDqkwg",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(g, "g"),
            ctx.tensor(h, "h"),
            ctx.tensor(dox, "dox"),
            ctx.tensor(dh, "dh"),
            ctx.tensor(dv, "dv"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctx.tensor(w, "w"),
            ctx.tensor(g_gamma, "g_gamma"),
            ctypes.c_float(_optional_float(scale, 1.0)),
            ctypes.c_int64(int(chunk_size)),
            ctypes.c_bool(_optional_bool(use_exp2, False)),
            ctypes.c_bool(_optional_bool(transpose_state_layout, False)),
            ctx.tensor(dq, "dq"),
            ctx.tensor(dk, "dk"),
            ctx.tensor(dw, "dw"),
            ctx.tensor(dg, "dg"),
        ],
        outputs,
    )


def npu_chunk_fwd_o(
    q,
    k,
    v,
    h,
    scale,
    *,
    g=None,
    g_gamma=None,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=None,
    transpose_state_layout=False,
):
    del g_gamma, transpose_state_layout
    chunk_size = _optional_int(chunk_size, 64)
    out = _empty_like(v)
    return _call_aclnn(
        "aclnnChunkFwdO",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(h, "h"),
            ctx.tensor(g, "g"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(chunk_size),
            ctx.tensor(out, "out"),
        ],
        out,
    )


def npu_chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g=None,
    *,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=None,
    cu_seqlens=None,
    chunk_indices=None,
    state_v_first=False,
):
    import torch

    if g is None and gk is None:
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: either g or gk must be provided.")
    output_final_state = _optional_bool(output_final_state, False)
    state_v_first = _optional_bool(state_v_first, False)
    chunk_size = _optional_int(chunk_size, 64)
    B, _, T, K = _shape(k)
    _, HV, _, V = _shape(u)
    cu = None if cu_seqlens is None else tuple(int(value) for value in cu_seqlens)
    indices = None if chunk_indices is None else tuple(int(value) for value in chunk_indices)
    if indices is None and cu is not None:
        indices = _kda_build_chunk_indices(cu, chunk_size)
    NT = _kda_total_chunks(B, T, chunk_size, cu, indices)
    N = len(cu) - 1 if cu is not None else B
    state_tail = (V, K) if state_v_first else (K, V)
    if initial_state is not None and _shape(initial_state) != (N, HV, *state_tail):
        raise RuntimeError(
            "npu_chunk_gated_delta_rule_fwd_h: initial_state shape does not match state_v_first."
        )
    h_out = _empty((B, HV, NT, *state_tail), k)
    v_new_out = _empty_like(u)
    if output_final_state:
        if initial_state is not None:
            final_state_out = _empty((N, HV, *state_tail), initial_state)
        else:
            final_state_out = _empty((N, HV, *state_tail), k, dtype=torch.float32)
    else:
        final_state_out = None
    outputs = (h_out, v_new_out, final_state_out if output_final_state else None)
    return _call_aclnn(
        "aclnnChunkGatedDeltaRuleFwdH",
        lambda ctx: [
            ctx.tensor(k, "k"),
            ctx.tensor(w, "w"),
            ctx.tensor(u, "u"),
            ctx.tensor(g, "g"),
            ctx.tensor(gk, "gk"),
            ctx.tensor(initial_state, "initial_state"),
            ctypes.c_bool(output_final_state),
            ctypes.c_int64(chunk_size),
            ctx.int_array(cu),
            ctx.int_array(indices),
            ctypes.c_bool(state_v_first),
            ctx.tensor(h_out, "h"),
            ctx.tensor(v_new_out, "v_new"),
            ctx.tensor(final_state_out, "final_state"),
        ],
        outputs,
    )


def npu_recompute_w_u_fwd(
    k,
    v,
    beta,
    A,
    chunk_size,
    *,
    g=None,
    gk=None,
    cu_seqlens=None,
    chunk_indices=None,
):
    w_shape = list(_shape(v))
    w_shape[3] = int(k.shape[3])
    w_out = _empty(w_shape, v, dtype=k.dtype)
    u_out = _empty_like(v)
    outputs = (w_out, u_out)
    return _call_aclnn(
        "aclnnRecomputeWUFwd",
        lambda ctx: [
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(beta, "beta"),
            ctx.tensor(A, "A"),
            ctx.tensor(g, "g"),
            ctx.tensor(gk, "gk"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(w_out, "w"),
            ctx.tensor(u_out, "u"),
        ],
        outputs,
    )


def _chunk_local_cumsum_output_dtype(g, output_dtype):
    import torch

    if output_dtype is None:
        return "float32", torch.float32
    if isinstance(output_dtype, torch.dtype):
        if output_dtype in (torch.float, torch.float32):
            return "float32", torch.float32
        if output_dtype in (torch.float16, torch.half):
            return "float16", torch.float16
        if output_dtype == torch.bfloat16:
            return "bfloat16", torch.bfloat16
        raise TypeError(f"Unsupported chunk_local_cumsum output_dtype: {output_dtype}.")

    normalized = str(output_dtype).removeprefix("torch.").lower()
    if normalized in {"float", "float32"}:
        return "float32", torch.float32
    if normalized in {"half", "float16"}:
        return "float16", torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return "bfloat16", torch.bfloat16
    if normalized in {"same", "input", "none"}:
        return normalized, g.dtype
    raise TypeError(f"Unsupported chunk_local_cumsum output_dtype: {output_dtype}.")


def npu_chunk_local_cumsum(
    g,
    chunk_size,
    *,
    cu_seqlens=None,
    chunk_indices_out=None,
    reverse=False,
    scale=1.0,
    head_first=True,
    output_dtype="float32",
):
    output_dtype_name, out_dtype = _chunk_local_cumsum_output_dtype(g, output_dtype)
    g_contig = g.contiguous()
    out = _empty(_shape(g_contig), g_contig, dtype=out_dtype)
    output_dtype_buffer = ctypes.create_string_buffer(output_dtype_name.encode("utf-8"))
    return _call_aclnn(
        "aclnnChunkLocalCumsum",
        lambda ctx: [
            ctx.tensor(g_contig, "g"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices_out),
            ctypes.c_int64(int(chunk_size)),
            ctypes.c_bool(bool(reverse)),
            ctypes.c_double(float(scale)),
            ctypes.c_bool(bool(head_first)),
            ctypes.cast(output_dtype_buffer, ctypes.c_char_p),
            ctx.tensor(out, "out"),
        ],
        out,
    )


def npu_chunk_scaled_dot_kkt(
    k,
    g,
    beta,
    *,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=64,
):
    import torch

    k_contig = k.contiguous()
    g_contig = g.contiguous()
    beta_contig = beta.contiguous()
    B, Hk, T, _ = _shape(k_contig)
    out = _empty((B, Hk, T, int(chunk_size)), k_contig, dtype=torch.float32)
    return _call_aclnn(
        "aclnnChunkScaledDotKkt",
        lambda ctx: [
            ctx.tensor(k_contig, "k"),
            ctx.tensor(g_contig, "g"),
            ctx.tensor(beta_contig, "beta"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(out, "out"),
        ],
        out,
    )


def _infer_causal_conv1d_y(x, head_num: int, run_mode: int):
    x_dim = x.dim()
    if run_mode == 0 and head_num > 0:
        if x_dim == 3:
            b, s, d_model = _shape(x)
            return _empty((b, head_num, s, d_model // head_num), x)
        if x_dim == 2:
            s, d_model = _shape(x)
            return _empty((head_num, s, d_model // head_num), x)
    return _empty_like(x)


def npu_causal_conv1d(
    x,
    weight,
    bias=None,
    conv_states=None,
    *,
    query_start_loc=None,
    cache_indices=None,
    initial_state_mode=None,
    num_accepted_tokens=None,
    activation_mode=0,
    pad_slot_id=-1,
    run_mode=0,
    head_num=0,
):
    out = _infer_causal_conv1d_y(x, int(head_num), int(run_mode))
    return _call_aclnn(
        "aclnnCausalConv1d",
        lambda ctx: [
            ctx.tensor(x, "x"),
            ctx.tensor(weight, "weight"),
            ctx.tensor(bias, "bias"),
            ctx.tensor(conv_states, "conv_states"),
            ctx.int_array(query_start_loc),
            ctx.int_array(cache_indices),
            ctx.int_array(initial_state_mode),
            ctx.int_array(num_accepted_tokens),
            ctypes.c_int64(int(activation_mode)),
            ctypes.c_int64(int(pad_slot_id)),
            ctypes.c_int64(int(run_mode)),
            ctypes.c_int64(int(head_num)),
            ctx.tensor(out, "out"),
        ],
        out,
    )


def npu_causal_conv1d_bwd(
    x,
    y,
    weight,
    dy,
    initial_state=None,
    dht=None,
    *,
    query_start_loc=None,
    activation=0,
    input_layout="BSND",
):
    input_layout = str(input_layout)
    width, dim = int(weight.shape[0]), int(weight.shape[1])
    if input_layout == "BNSD":
        batch = int(x.shape[0])
        dx_shape = _shape(x)
    elif input_layout in {"NTD", "TND"}:
        if query_start_loc is None:
            raise RuntimeError(f"query_start_loc is required for {input_layout} input.")
        batch = len(query_start_loc) - 1
        dx_shape = _shape(x)
    else:
        batch = int(x.shape[0])
        dx_shape = _shape(x)
    dx = _empty(dx_shape, x)
    dw = _empty((width, dim), weight)
    db = _empty((dim,), weight)
    dh0 = _empty((batch, width, dim), x)
    outputs = (dx, dw, db, dh0)
    layout_buffer = ctypes.create_string_buffer(input_layout.encode("utf-8"))
    return _call_aclnn(
        "aclnnCausalConv1dBwd",
        lambda ctx: [
            ctx.tensor(x, "x"),
            ctx.tensor(y, "y"),
            ctx.tensor(weight, "weight"),
            ctx.tensor(dy, "dy"),
            ctx.tensor(initial_state, "initial_state"),
            ctx.tensor(dht, "dht"),
            ctx.int_array(query_start_loc),
            ctypes.c_int64(int(activation)),
            ctypes.cast(layout_buffer, ctypes.c_char_p),
            ctx.tensor(dx, "dx"),
            ctx.tensor(dw, "dw"),
            ctx.tensor(db, "db"),
            ctx.tensor(dh0, "dh0"),
        ],
        outputs,
    )


def _kda_ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _kda_build_chunk_indices(cu_seqlens, chunk_size: int):
    if cu_seqlens is None:
        return None
    cu = tuple(int(value) for value in cu_seqlens)
    indices = []
    for seq in range(len(cu) - 1):
        seq_len = cu[seq + 1] - cu[seq]
        for chunk in range(_kda_ceil_div(seq_len, chunk_size)):
            indices.extend((seq, chunk))
    return tuple(indices)


def _kda_total_chunks(batch: int, seqlen: int, chunk_size: int, cu_seqlens, chunk_indices) -> int:
    del batch
    if chunk_indices is not None:
        return len(tuple(chunk_indices)) // 2
    if cu_seqlens is None:
        return _kda_ceil_div(seqlen, chunk_size)
    cu = tuple(int(value) for value in cu_seqlens)
    return sum(_kda_ceil_div(cu[i + 1] - cu[i], chunk_size) for i in range(len(cu) - 1))


def npu_chunk_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    chunk_size=64,
    *,
    layout="BSND",
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
):
    import torch

    layout = str(layout)
    if layout not in {"BSND", "BNSD", "TND", "NTD"}:
        raise RuntimeError("npu_chunk_kda_fwd: layout must be uppercase and one of BSND, BNSD, TND, NTD.")
    chunk_size = int(chunk_size)
    if chunk_size not in {64, 128}:
        raise RuntimeError("npu_chunk_kda_fwd: chunk_size must be 64 or 128.")

    is_rank3 = layout in {"TND", "NTD"}
    is_sequence_major = layout in {"BSND", "TND"}
    q_shape, k_shape, v_shape, g_shape, beta_shape = map(
        _shape, (q, k, v, g, beta)
    )
    expected_rank = 3 if is_rank3 else 4
    if any(len(shape) != expected_rank for shape in (q_shape, k_shape, v_shape, g_shape)):
        raise RuntimeError("npu_chunk_kda_fwd: q/k/v/g rank does not match layout.")
    if len(beta_shape) != expected_rank - 1 or q_shape != k_shape:
        raise RuntimeError("npu_chunk_kda_fwd: beta rank must match layout and q/k shapes must be identical.")

    if layout == "TND":
        batch, seqlen, h_num, k_dim = 1, q_shape[0], q_shape[1], q_shape[2]
        hv_num, v_dim = v_shape[1], v_shape[2]
        expected_v = (seqlen, hv_num, v_dim)
        expected_g = (seqlen, hv_num, k_dim)
        expected_beta = (seqlen, hv_num)
    elif layout == "NTD":
        batch, h_num, seqlen, k_dim = 1, q_shape[0], q_shape[1], q_shape[2]
        hv_num, v_dim = v_shape[0], v_shape[2]
        expected_v = (hv_num, seqlen, v_dim)
        expected_g = (hv_num, seqlen, k_dim)
        expected_beta = (hv_num, seqlen)
    elif layout == "BSND":
        batch, seqlen, h_num, k_dim = q_shape
        hv_num, v_dim = v_shape[2], v_shape[3]
        expected_v = (batch, seqlen, hv_num, v_dim)
        expected_g = (batch, seqlen, hv_num, k_dim)
        expected_beta = (batch, seqlen, hv_num)
    else:
        batch, h_num, seqlen, k_dim = q_shape
        hv_num, v_dim = v_shape[1], v_shape[3]
        expected_v = (batch, hv_num, seqlen, v_dim)
        expected_g = (batch, hv_num, seqlen, k_dim)
        expected_beta = (batch, hv_num, seqlen)
    if v_shape != expected_v or g_shape != expected_g or beta_shape != expected_beta:
        raise RuntimeError("npu_chunk_kda_fwd: v/g/beta shapes do not match the selected layout.")
    if h_num <= 0 or hv_num < h_num or hv_num % h_num != 0 or h_num > 128 or hv_num > 128:
        raise RuntimeError("npu_chunk_kda_fwd: H/HV must satisfy 0 < H <= HV <= 128 and HV % H == 0.")
    if q.dtype not in {torch.float16, torch.bfloat16} or k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError("npu_chunk_kda_fwd: q/k/v must use the same float16 or bfloat16 dtype.")
    if g.dtype not in {torch.float32, torch.bfloat16} or beta.dtype not in {torch.float32, torch.bfloat16}:
        raise RuntimeError("npu_chunk_kda_fwd: g and beta must be float32 or bfloat16.")
    if k_dim < 16 or k_dim > 256 or k_dim % 16 or v_dim < 16 or v_dim > 256 or v_dim % 16:
        raise RuntimeError("npu_chunk_kda_fwd: K/V must be multiples of 16 with K,V <= 256.")

    use_gate_in_kernel = _optional_bool(use_gate_in_kernel, False)
    safe_gate = _optional_bool(safe_gate, False)
    disable_recompute = _optional_bool(disable_recompute, False)
    return_intermediate_states = _optional_bool(return_intermediate_states, False)
    output_final_state = _optional_bool(output_final_state, False)
    state_v_first = _optional_bool(state_v_first, False)
    if use_gate_in_kernel:
        if A_log is None or _shape(A_log) != (hv_num,) or A_log.dtype != torch.float32:
            raise RuntimeError("npu_chunk_kda_fwd: A_log must be float32 [HV] when use_gate_in_kernel=True.")
        if dt_bias is not None and (_shape(dt_bias) != (hv_num * k_dim,) or dt_bias.dtype != torch.float32):
            raise RuntimeError("npu_chunk_kda_fwd: dt_bias must be float32 [HV*K].")
    lower_bound = _optional_float(lower_bound, -5.0)
    if use_gate_in_kernel and safe_gate and not (-5.0 <= lower_bound < 0.0):
        raise RuntimeError("npu_chunk_kda_fwd: lower_bound must be in [-5, 0) for safe gate.")

    cu = None if cu_seqlens is None else tuple(int(value) for value in cu_seqlens)
    if cu is not None:
        if len(cu) < 2 or cu[0] != 0 or cu[-1] != seqlen or any(a > b for a, b in zip(cu, cu[1:])):
            raise RuntimeError("npu_chunk_kda_fwd: cu_seqlens must be nondecreasing, start at 0 and end at T.")
        if len(cu) - 1 > 1024:
            raise RuntimeError("npu_chunk_kda_fwd: varlen input supports at most 1024 sequences.")
        if not is_rank3 and batch != 1:
            raise RuntimeError("npu_chunk_kda_fwd: rank4 varlen input requires B=1.")
    seq_num = len(cu) - 1 if cu is not None else batch
    canonical_indices = _kda_build_chunk_indices(cu, chunk_size)
    indices = canonical_indices if chunk_indices is None else tuple(int(value) for value in chunk_indices)
    if indices is not None and indices != canonical_indices:
        raise RuntimeError("npu_chunk_kda_fwd: chunk_indices must use canonical sequence-major order.")
    total_chunks = _kda_total_chunks(batch, seqlen, chunk_size, cu, indices)

    state_shape = (
        (seq_num, hv_num, v_dim, k_dim)
        if state_v_first
        else (seq_num, hv_num, k_dim, v_dim)
    )
    if initial_state is not None and (_shape(initial_state) != state_shape or initial_state.dtype != torch.float32):
        raise RuntimeError("npu_chunk_kda_fwd: initial_state shape/dtype does not match state_v_first.")

    attn_shape = (
        (seqlen, hv_num, v_dim)
        if is_rank3
        else (batch, seqlen, hv_num, v_dim)
    )
    matrix_shape = (
        (hv_num, seqlen, chunk_size)
        if is_rank3
        else (batch, hv_num, seqlen, chunk_size)
    )
    k_shape_head = (
        (hv_num, seqlen, k_dim)
        if is_rank3
        else (batch, hv_num, seqlen, k_dim)
    )
    v_shape_head = (
        (hv_num, seqlen, v_dim)
        if is_rank3
        else (batch, hv_num, seqlen, v_dim)
    )
    h_shape = (
        ((total_chunks, hv_num, v_dim, k_dim) if state_v_first
         else (total_chunks, hv_num, k_dim, v_dim))
        if is_rank3
        else ((batch, total_chunks, hv_num, v_dim, k_dim) if state_v_first
              else (batch, total_chunks, hv_num, k_dim, v_dim))
    )

    output_mask = kda_fwd_optional_output_mask(
        output_final_state=output_final_state,
        use_gate_in_kernel=use_gate_in_kernel,
        disable_recompute=disable_recompute,
        return_intermediate_states=return_intermediate_states,
    )
    attn_out = _empty(attn_shape, q)
    final_state = _empty(state_shape, q, dtype=torch.float32) if output_mask[1] else None
    gk_out = _empty(k_shape_head, q, dtype=torch.float32) if output_mask[2] else None
    aqk = _empty(matrix_shape, q)
    akk = _empty(matrix_shape, q)
    w = _empty(k_shape_head, q) if output_mask[5] else None
    u = _empty(v_shape_head, q) if output_mask[6] else None
    qg = _empty(k_shape_head, q) if output_mask[7] else None
    kg = _empty(k_shape_head, q) if output_mask[8] else None
    v_new = _empty(v_shape_head, q) if output_mask[9] else None
    h = _empty(h_shape, q) if output_mask[10] else None

    outputs = (attn_out, final_state, gk_out, aqk, akk, w, u, qg, kg, v_new, h)
    layout_buffer = ctypes.create_string_buffer(layout.encode("utf-8"))
    _call_aclnn(
        "aclnnChunkKdaFwd",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(g, "g"),
            ctx.tensor(beta, "beta"),
            ctx.tensor(A_log, "A_log"),
            ctx.tensor(dt_bias, "dt_bias"),
            ctx.tensor(initial_state, "initial_state"),
            ctx.int_array(cu),
            ctx.int_array(indices),
            ctypes.cast(layout_buffer, ctypes.c_char_p),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(chunk_size),
            ctypes.c_bool(safe_gate),
            ctypes.c_double(lower_bound),
            ctypes.c_bool(use_gate_in_kernel),
            ctypes.c_bool(state_v_first),
            ctx.tensor(attn_out, "attn_out"),
            ctx.tensor(final_state, "final_state"),
            ctx.tensor(gk_out, "gk"),
            ctx.tensor(aqk, "Aqk"),
            ctx.tensor(akk, "Akk"),
            ctx.tensor(w, "w"),
            ctx.tensor(u, "u"),
            ctx.tensor(qg, "qg"),
            ctx.tensor(kg, "kg"),
            ctx.tensor(v_new, "v_new"),
            ctx.tensor(h, "h"),
        ],
        outputs,
    )
    initial_state_out = initial_state
    return (*outputs, initial_state_out)


def npu_kda_gate_cumsum(
    g,
    chunk_size,
    *,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    use_gate_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
):
    import torch

    out = _empty(_shape(g), g, dtype=torch.float32)
    return _call_aclnn(
        "aclnnKdaGateCumsum",
        lambda ctx: [
            ctx.tensor(g, "g"),
            ctx.tensor(A_log, "A_log"),
            ctx.tensor(dt_bias, "dt_bias"),
            ctx.int_array(None if cu_seqlens is None else tuple(int(value) for value in cu_seqlens)),
            ctypes.c_int64(int(chunk_size)),
            ctypes.c_bool(_optional_bool(use_gate_in_kernel, False)),
            ctypes.c_bool(_optional_bool(safe_gate, False)),
            ctypes.c_double(_optional_float(lower_bound, -5.0)),
            ctx.tensor(out, "gk"),
        ],
        out,
    )


def npu_chunk_kda_fwd_intra_sub_chunk(
    q,
    k,
    g,
    beta,
    scale,
    chunk_size,
    *,
    cu_seqlens=None,
    chunk_indices=None,
):
    """Safe-gate diagonal intra sub-chunk: Aqk diag blocks + fp32 Akkd=(I-L)^-1.

    Layout is BNSD:
      q/k [B,H,T,K], g [B,HV,T,K], beta [B,HV,T],
      aqk [B,HV,T,BT], akkd [B,HV,T,16].
    GVA: HV >= H and HV % H == 0 (i_h = i_hv // (HV/H)), same as GPU Triton.
    """
    import torch

    B, H, T, K = q.shape
    if q.shape != k.shape:
        raise ValueError("q/k must share shape [B,H,T,K]")
    if K != 128:
        raise ValueError("npu_chunk_kda_fwd_intra_sub_chunk: K must be 128 (DESIGN v1)")
    if g.ndim != 4 or g.shape[0] != B or g.shape[2] != T or g.shape[3] != K:
        raise ValueError("g must be [B,HV,T,K] matching q on B/T/K")
    HV = int(g.shape[1])
    if HV < H or (HV % H) != 0:
        raise ValueError(f"GVA requires HV>=H and HV%H==0, got H={H} HV={HV}")
    if tuple(beta.shape) != (B, HV, T):
        raise ValueError("beta must be [B,HV,T]")
    BT = int(chunk_size)
    BC = 16
    if BT not in (32, 64, 128):
        raise ValueError("chunk_size must be 32, 64 or 128")
    # Aqk is diagonal-block sparse in this stage; zero so off-block columns
    # are not left as recycled allocator garbage after larger prior calls.
    aqk = _zeros((B, HV, T, BT), q, dtype=q.dtype)
    akkd = _zeros((B, HV, T, BC), q, dtype=torch.float32)
    cu = None if cu_seqlens is None else tuple(int(v) for v in cu_seqlens)
    idx = None if chunk_indices is None else tuple(int(v) for v in chunk_indices)
    return _call_aclnn(
        "aclnnChunkKdaFwdIntraSubChunk",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(g, "g"),
            ctx.tensor(beta, "beta"),
            ctx.int_array(cu),
            ctx.int_array(idx),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(BT),
            ctx.tensor(aqk, "aqk"),
            ctx.tensor(akkd, "akkd"),
        ],
        (aqk, akkd),
    )


def npu_solve_tri(x, *, cu_seqlens=None, chunk_indices=None, layout="bsnd"):
    x_contig = x.contiguous()
    out = _empty_like(x_contig)
    layout_arg = ctypes.c_char_p(str(layout).encode("utf-8"))
    return _call_aclnn(
        "aclnnSolveTri",
        lambda ctx: [
            ctx.tensor(x_contig, "x"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            layout_arg,
            ctx.tensor(out, "out"),
        ],
        out,
    )


ASCENDC_CTYPES_OPS = {
    name: value
    for name, value in globals().items()
    if name.startswith("npu_") and callable(value)
}
