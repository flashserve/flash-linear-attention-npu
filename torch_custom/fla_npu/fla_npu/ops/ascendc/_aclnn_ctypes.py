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
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_bool,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "aclnnRecurrentKda": [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_double,
        ctypes.c_bool,
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
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    transpose_state_layout=False,
):
    import torch

    if g is None and gk is None:
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: either g or gk must be provided.")
    save_new_value = _optional_bool(save_new_value, True)
    use_exp2 = _optional_bool(use_exp2, False)
    transpose_state_layout = _optional_bool(transpose_state_layout, False)
    if not save_new_value:
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: save_new_value must be True.")
    if use_exp2:
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: use_exp2 must be False.")
    if transpose_state_layout:
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: transpose_state_layout must be False.")

    output_final_state = _optional_bool(output_final_state, False)
    chunk_size = _optional_int(chunk_size, 64)
    B, _, T, K = _shape(k)
    _, HV, _, V = _shape(u)
    NT = _chunk_num(T, chunk_size, chunk_indices)
    h_out = _zeros((B, HV, NT, K, V), k)
    v_new_out = _empty_like(u)
    if output_final_state:
        N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
        if initial_state is not None:
            final_state_out = _empty((N, HV, K, V), initial_state)
        else:
            final_state_out = _empty((N, HV, K, V), k, dtype=torch.float32)
    else:
        final_state_out = _empty((1,), k)
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
            ctypes.c_bool(save_new_value),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_bool(use_exp2),
            ctypes.c_bool(transpose_state_layout),
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
            ctx.int_tensor(query_start_loc, x.device),
            ctx.int_tensor(cache_indices, x.device),
            ctx.int_tensor(initial_state_mode, x.device),
            ctx.int_tensor(num_accepted_tokens, x.device),
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
    gk,
    beta,
    scale,
    chunk_size,
    *,
    layout="BSND",
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    return_intermediate=False,
    safe_gate=False,
    transpose_state_layout=False,
):
    import torch

    return_intermediate = _optional_bool(return_intermediate, False)
    layout = str(layout)
    if layout not in {"BSND", "BNSD", "TND", "NTD"}:
        raise RuntimeError("npu_chunk_kda_fwd: layout must be one of BSND, BNSD, TND, NTD and must be uppercase.")
    if _optional_bool(safe_gate, False):
        raise RuntimeError("npu_chunk_kda_fwd: safe_gate=True is not supported.")
    if _optional_bool(transpose_state_layout, False):
        raise RuntimeError("npu_chunk_kda_fwd: transpose_state_layout=True is not supported.")

    chunk_size = int(chunk_size)
    if chunk_size not in {64, 128}:
        raise RuntimeError("npu_chunk_kda_fwd: chunk_size must be 64 or 128.")

    q_shape = _shape(q)
    k_shape = _shape(k)
    v_shape = _shape(v)
    gk_shape = _shape(gk)
    beta_shape = _shape(beta)
    is_tnd = layout == "TND"
    is_ntd = layout == "NTD"
    is_bsnd = layout == "BSND"
    is_bnsd = layout == "BNSD"
    is_rank3 = is_tnd or is_ntd
    is_internal_layout = is_bnsd or is_ntd
    rank_ok = (
        (is_rank3 and len(q_shape) == 3 and len(k_shape) == 3 and len(v_shape) == 3 and
         len(gk_shape) == 3 and len(beta_shape) == 2) or
        (not is_rank3 and len(q_shape) == 4 and len(k_shape) == 4 and len(v_shape) == 4 and
         len(gk_shape) == 4 and len(beta_shape) == 3)
    )
    if not rank_ok:
        raise RuntimeError(
            "npu_chunk_kda_fwd: layout/rank mismatch. TND/NTD require q/k/v/gk rank3 with beta rank2; "
            "BSND/BNSD require q/k/v/gk rank4 with beta rank3."
        )
    if q_shape != k_shape:
        raise RuntimeError("npu_chunk_kda_fwd: q and k must have identical shape.")
    if gk.dtype not in {torch.float32, torch.bfloat16} or beta.dtype not in {torch.float32, torch.bfloat16}:
        raise RuntimeError("npu_chunk_kda_fwd: gk and beta must be float32 or bfloat16.")

    batch = 1 if is_rank3 else q_shape[0]
    seqlen = q_shape[0] if is_tnd else (q_shape[1] if is_ntd else (q_shape[2] if is_bnsd else q_shape[1]))
    h_num = q_shape[1] if is_tnd else (q_shape[0] if is_ntd else (q_shape[1] if is_bnsd else q_shape[2]))
    k_dim = q_shape[2] if is_rank3 else q_shape[3]
    hv_num = v_shape[1] if is_tnd else (v_shape[0] if is_ntd else (v_shape[1] if is_bnsd else v_shape[2]))
    v_dim = v_shape[2] if is_rank3 else v_shape[3]
    if h_num <= 0 or hv_num < h_num:
        raise RuntimeError("npu_chunk_kda_fwd: H and HV must be positive and H must be <= HV.")
    if h_num > 128 or hv_num > 128:
        raise RuntimeError("npu_chunk_kda_fwd: H and HV must be <= 128.")
    if is_tnd and h_num > 1:
        raise RuntimeError(
            "npu_chunk_kda_fwd: TND layout with H > 1 is not supported; use NTD [H,T,D] layout "
            "for multi-head rank3 input."
        )
    if hv_num % h_num != 0:
        raise RuntimeError("npu_chunk_kda_fwd: HV must be divisible by H.")
    split_cube_supported = (
        q.dtype in {torch.float16, torch.bfloat16} and k.dtype == q.dtype and v.dtype == q.dtype and
        k_dim >= 16 and v_dim >= 16 and k_dim % 16 == 0 and v_dim % 16 == 0 and v_dim <= 256
    )
    if not split_cube_supported:
        raise RuntimeError("npu_chunk_kda_fwd: shape is outside the supported split cube/vector template.")
    if (is_tnd and (v_shape[0] != seqlen or gk_shape != (seqlen, hv_num, k_dim) or
                   beta_shape != (seqlen, hv_num))) or (
        is_ntd and (v_shape[1] != seqlen or gk_shape != (hv_num, seqlen, k_dim) or
                    beta_shape != (hv_num, seqlen))
    ) or (
        is_bsnd and (v_shape[0] != batch or v_shape[1] != seqlen or
                     gk_shape != (batch, seqlen, hv_num, k_dim) or beta_shape != (batch, seqlen, hv_num))
    ) or (
        is_bnsd and (v_shape[0] != batch or v_shape[2] != seqlen or
                     gk_shape != (batch, hv_num, seqlen, k_dim) or beta_shape != (batch, hv_num, seqlen))
    ):
        raise RuntimeError("npu_chunk_kda_fwd: v/gk/beta shape mismatch for the selected layout.")

    cu_seqlens_for_call = None if cu_seqlens is None else tuple(int(value) for value in cu_seqlens)
    if cu_seqlens_for_call is not None and (len(cu_seqlens_for_call) < 2 or cu_seqlens_for_call[0] != 0 or
                                            cu_seqlens_for_call[-1] != seqlen):
        raise RuntimeError("npu_chunk_kda_fwd: cu_seqlens must start at 0 and end at sequence length.")
    seq_num = len(cu_seqlens_for_call) - 1 if cu_seqlens_for_call is not None else batch
    chunk_indices_for_call = (
        tuple(int(value) for value in chunk_indices)
        if chunk_indices is not None
        else _kda_build_chunk_indices(cu_seqlens_for_call, chunk_size)
    )
    if chunk_indices_for_call is not None:
        if cu_seqlens_for_call is None:
            raise RuntimeError("npu_chunk_kda_fwd: chunk_indices requires cu_seqlens.")
        if len(chunk_indices_for_call) % 2 != 0:
            raise RuntimeError("npu_chunk_kda_fwd: chunk_indices must contain (seq_id, chunk_id) pairs.")
        expected_chunks = sum(
            (cu_seqlens_for_call[idx + 1] - cu_seqlens_for_call[idx] + chunk_size - 1) // chunk_size
            for idx in range(len(cu_seqlens_for_call) - 1)
        )
        if len(chunk_indices_for_call) // 2 != expected_chunks:
            raise RuntimeError("npu_chunk_kda_fwd: chunk_indices must contain exactly one pair per chunk.")
        for idx in range(0, len(chunk_indices_for_call), 2):
            seq_idx, local_chunk = chunk_indices_for_call[idx:idx + 2]
            if seq_idx < 0 or seq_idx >= len(cu_seqlens_for_call) - 1:
                raise RuntimeError("npu_chunk_kda_fwd: chunk_indices seq_id is out of range.")
            seq_len = cu_seqlens_for_call[seq_idx + 1] - cu_seqlens_for_call[seq_idx]
            seq_chunks = (seq_len + chunk_size - 1) // chunk_size
            if local_chunk < 0 or local_chunk >= seq_chunks:
                raise RuntimeError("npu_chunk_kda_fwd: chunk_indices chunk_id is out of range.")
        canonical_chunk_indices = _kda_build_chunk_indices(cu_seqlens_for_call, chunk_size)
        if chunk_indices_for_call != canonical_chunk_indices:
            raise RuntimeError(
                "npu_chunk_kda_fwd: chunk_indices must use canonical sequence-major chunk order."
            )
    total_chunks = _kda_total_chunks(batch, seqlen, chunk_size, cu_seqlens_for_call, chunk_indices_for_call)
    if cu_seqlens_for_call is not None and seq_num > 1024:
        raise RuntimeError(
            "npu_chunk_kda_fwd: varlen input supports at most 1024 sequences in one call; "
            "split a larger request at sequence boundaries."
        )
    if initial_state is not None:
        initial_state_shape = _shape(initial_state)
        if initial_state.dtype != torch.float32:
            raise RuntimeError("npu_chunk_kda_fwd: initial_state must be float32 when provided.")
        if initial_state_shape != (seq_num, hv_num, k_dim, v_dim):
            raise RuntimeError(
                "npu_chunk_kda_fwd: initial_state must be [seq_num,Hv,K,V], where seq_num is batch "
                "for dense input or len(cu_seqlens)-1 for varlen input."
            )

    o = _empty_like(v)
    final_state_work = _empty((seq_num, hv_num, k_dim, v_dim), q, dtype=torch.float32)
    if is_rank3:
        bnst_shape = (hv_num, seqlen, chunk_size) if is_internal_layout else (seqlen, hv_num, chunk_size)
        bnsd_k_shape = (hv_num, seqlen, k_dim) if is_internal_layout else (seqlen, hv_num, k_dim)
        h_shape = ((hv_num, total_chunks, k_dim, v_dim) if is_internal_layout
                   else (total_chunks, hv_num, k_dim, v_dim))
    else:
        bnst_shape = ((batch, hv_num, seqlen, chunk_size) if is_internal_layout
                      else (batch, seqlen, hv_num, chunk_size))
        bnsd_k_shape = ((batch, hv_num, seqlen, k_dim) if is_internal_layout
                        else (batch, seqlen, hv_num, k_dim))
        h_shape = ((batch, hv_num, total_chunks, k_dim, v_dim) if is_internal_layout
                   else (batch, total_chunks, hv_num, k_dim, v_dim))
    if return_intermediate:
        kernel_aqk = aqk = _empty(bnst_shape, q)
        kernel_akk = akk = _empty(bnst_shape, q)
        kernel_w = w = _empty(bnsd_k_shape, q)
        kernel_u = u = _empty_like(v)
        kernel_qg = qg = _empty(bnsd_k_shape, q)
        kernel_kg = kg = _empty(bnsd_k_shape, q)
        kernel_v_new = v_new = _empty_like(v)
        kernel_h = h = _empty(h_shape, q)
    else:
        aqk, akk, w, u = (_empty((0,), q) for _ in range(4))
        qg, kg, v_new, h = (_empty((0,), q) for _ in range(4))
        kernel_aqk, kernel_akk, kernel_w, kernel_u = aqk, akk, w, u
        kernel_qg, kernel_kg, kernel_v_new, kernel_h = qg, kg, v_new, h
    empty = _empty((0,), q)
    final_state = final_state_work if _optional_bool(output_final_state, False) else _empty((0,), q, dtype=torch.float32)
    g = gk if gk.dtype == torch.float32 else gk.to(torch.float32)
    initial_state_out = initial_state if initial_state is not None else empty
    user_outputs = (o, final_state, g, aqk, akk, w, u, qg, kg, v_new, h, initial_state_out)
    kernel_outputs = (o, final_state_work, kernel_aqk, kernel_akk, kernel_w, kernel_u,
                      kernel_qg, kernel_kg, kernel_v_new, kernel_h)
    layout_buffer = ctypes.create_string_buffer(layout.encode("utf-8"))
    _call_aclnn(
        "aclnnChunkKdaFwd",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(gk, "gk"),
            ctx.tensor(beta, "beta"),
            ctx.tensor(initial_state, "initial_state"),
            ctx.int_array(cu_seqlens_for_call),
            ctx.int_array(chunk_indices_for_call),
            ctypes.cast(layout_buffer, ctypes.c_char_p),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(chunk_size),
            ctypes.c_bool(True),
            ctypes.c_int64(total_chunks),
            ctx.tensor(o, "o"),
            ctx.tensor(final_state_work, "final_state"),
            ctx.tensor(kernel_aqk, "aqk"),
            ctx.tensor(kernel_akk, "akk"),
            ctx.tensor(kernel_w, "w"),
            ctx.tensor(kernel_u, "u"),
            ctx.tensor(kernel_qg, "qg"),
            ctx.tensor(kernel_kg, "kg"),
            ctx.tensor(kernel_v_new, "v_new"),
            ctx.tensor(kernel_h, "h"),
        ],
        kernel_outputs,
    )
    return user_outputs


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
    layout="BSND",
):
    import torch

    out = _empty(_shape(g), g, dtype=torch.float32)
    layout = str(layout)
    if layout not in ("BSND", "BNSD", "TND", "NTD"):
        raise ValueError("layout must be uppercase and one of BSND, BNSD, TND or NTD")
    layout_buffer = ctypes.create_string_buffer(layout.encode("utf-8"))
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
            ctypes.cast(layout_buffer, ctypes.c_char_p),
            ctx.tensor(out, "gk"),
        ],
        out,
    )

def npu_recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    *,
    cu_seqlens,
    ssm_state_indices=None,
    A_log=None,
    dt_bias=None,
    num_accepted_tokens=None,
    layout="BSND",
    scale=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    state_v_first=True,
):
    import torch

    layout = str(layout)
    if layout not in ("BSND", "TND"):
        raise RuntimeError("npu_recurrent_kda: layout must be BSND or TND.")
    is_tnd = layout == "TND"
    q_shape = _shape(q)
    k_shape = _shape(k)
    v_shape = _shape(v)
    g_shape = _shape(g)
    beta_shape = _shape(beta)
    expected_q_rank = 3 if is_tnd else 4
    if (
        len(q_shape) != expected_q_rank
        or len(k_shape) != expected_q_rank
        or len(v_shape) != expected_q_rank
        or len(g_shape) != expected_q_rank
        or len(beta_shape) != (2 if is_tnd else 3)
    ):
        raise RuntimeError(
            "npu_recurrent_kda: layout/rank mismatch. TND expects q/k [T,H,K], v [T,HV,V], "
            "g [T,HV,K], beta [T,HV]; BSND expects q/k [B,T,H,K], v [B,T,HV,V], "
            "g [B,T,HV,K], beta [B,T,HV]."
        )
    if q_shape != k_shape:
        raise RuntimeError("npu_recurrent_kda: q and k must have identical shape.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise RuntimeError("npu_recurrent_kda: q/k/v currently support bfloat16 only.")
    if any(tensor.device != q.device for tensor in (k, v, g, beta)):
        raise RuntimeError("npu_recurrent_kda: q/k/v/g/beta must be on the same device.")
    if g.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("npu_recurrent_kda: g must use FP16, BF16 or FP32.")
    if beta.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("npu_recurrent_kda: beta must use FP16, BF16 or FP32.")

    if is_tnd:
        total_tokens, heads, key_dim = q_shape
        value_heads, value_dim = v_shape[1], v_shape[2]
        dense_seq_len = total_tokens
        batch = 1
        value_shape_ok = (
            v_shape[0] == total_tokens
            and g_shape == (total_tokens, value_heads, key_dim)
            and beta_shape == (total_tokens, value_heads)
        )
    else:
        batch, dense_seq_len, heads, key_dim = q_shape
        value_heads, value_dim = v_shape[2], v_shape[3]
        total_tokens = batch * dense_seq_len
        value_shape_ok = (
            v_shape[0] == batch
            and v_shape[1] == dense_seq_len
            and g_shape == (batch, dense_seq_len, value_heads, key_dim)
            and beta_shape == (batch, dense_seq_len, value_heads)
        )
    if not value_shape_ok:
        raise RuntimeError("npu_recurrent_kda: v/g/beta shape mismatch.")
    if min(total_tokens, dense_seq_len, heads, value_heads, key_dim, value_dim) <= 0:
        raise RuntimeError("npu_recurrent_kda: all shape dimensions must be positive.")
    if value_heads % heads != 0:
        raise RuntimeError("npu_recurrent_kda: HV must be divisible by H.")
    if (key_dim, value_dim) not in ((128, 128), (128, 256)):
        raise RuntimeError("npu_recurrent_kda: K/V currently support only K=128,V=128 or K=128,V=256.")
    if (not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.dim() != 1 or
            cu_seqlens.numel() < 2):
        raise RuntimeError(
            "npu_recurrent_kda: cu_seqlens must be a 1D device tensor with at least two elements."
        )
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("npu_recurrent_kda: cu_seqlens must be INT32 or INT64.")
    if cu_seqlens.device != q.device:
        raise RuntimeError("npu_recurrent_kda: cu_seqlens must be on the same device as q.")
    seq_num = int(cu_seqlens.shape[0]) - 1

    state_v_first = _optional_bool(state_v_first, True)
    if not state_v_first:
        raise RuntimeError("npu_recurrent_kda: state_v_first=False is not supported.")
    expected_tail = (value_heads, value_dim, key_dim)
    if initial_state is None:
        state_shape = (seq_num, *expected_tail)
        initial_state_work = _zeros(state_shape, q, dtype=torch.float32)
    else:
        if initial_state.dtype not in (torch.float32, torch.bfloat16):
            raise RuntimeError("npu_recurrent_kda: initial_state must be a mutable FP32 or BF16 state tensor.")
        if initial_state.device != q.device:
            raise RuntimeError("npu_recurrent_kda: initial_state must be on the same device as q.")
        state_shape = _shape(initial_state)
        if len(state_shape) != 4 or state_shape[0] <= 0 or state_shape[1:] != expected_tail:
            raise RuntimeError("npu_recurrent_kda: initial_state must be [state_capacity,HV,V,K].")
        initial_state_work = initial_state

    if ssm_state_indices is not None:
        packed_1d = ssm_state_indices.dim() == 1 and int(ssm_state_indices.shape[0]) >= total_tokens
        speculative_2d = (
            ssm_state_indices.dim() == 2
            and int(ssm_state_indices.shape[0]) == seq_num
            and int(ssm_state_indices.shape[1]) > 0
        )
        if ssm_state_indices.dtype not in (torch.int32, torch.int64) or not (packed_1d or speculative_2d):
            raise RuntimeError(
                "npu_recurrent_kda: ssm_state_indices must be INT32/INT64 packed [T] "
                "or speculative [seq_num,max_step]."
            )
        if ssm_state_indices.device != q.device:
            raise RuntimeError("npu_recurrent_kda: ssm_state_indices must be on the same device as q.")
    elif state_shape[0] != seq_num:
        raise RuntimeError(
            "npu_recurrent_kda: without ssm_state_indices, state_capacity must equal seq_num."
        )
    if num_accepted_tokens is not None:
        if ssm_state_indices is None:
            raise RuntimeError("npu_recurrent_kda: num_accepted_tokens requires ssm_state_indices.")
        if num_accepted_tokens.dtype not in (torch.int32, torch.int64) or _shape(num_accepted_tokens) != (seq_num,):
            raise RuntimeError("npu_recurrent_kda: num_accepted_tokens must be INT32/INT64 [seq_num].")
        if num_accepted_tokens.device != q.device:
            raise RuntimeError("npu_recurrent_kda: num_accepted_tokens must be on the same device as q.")

    use_gate = _optional_bool(use_gate_in_kernel, False)
    safe = _optional_bool(safe_gate, False)
    lower = _optional_float(lower_bound, -5.0)
    if use_gate:
        if A_log is None or A_log.dtype != torch.float32 or _shape(A_log) != (value_heads,):
            raise RuntimeError("npu_recurrent_kda: A_log must be FP32 [HV] when use_gate_in_kernel=True.")
        if A_log.device != q.device:
            raise RuntimeError("npu_recurrent_kda: A_log must be on the same device as q.")
        if safe and not -5.0 <= lower < 0.0:
            raise RuntimeError("npu_recurrent_kda: lower_bound must be in [-5,0) when safe_gate=True.")
        if dt_bias is not None:
            valid_bias_shape = _shape(dt_bias) in ((value_heads * key_dim,), (value_heads, key_dim))
            if dt_bias.dtype != torch.float32 or not valid_bias_shape:
                raise RuntimeError("npu_recurrent_kda: dt_bias must be FP32 [HV*K] or [HV,K].")
            if dt_bias.device != q.device:
                raise RuntimeError("npu_recurrent_kda: dt_bias must be on the same device as q.")
    else:
        if safe:
            raise RuntimeError("npu_recurrent_kda: safe_gate only applies when use_gate_in_kernel=True.")
        if A_log is not None or dt_bias is not None:
            raise RuntimeError("npu_recurrent_kda: A_log and dt_bias must be None when use_gate_in_kernel=False.")

    out = _empty_like(v)
    final_state_work = initial_state_work
    final_state = final_state_work if _optional_bool(output_final_state, False) else _empty((0,), initial_state_work)
    user_outputs = (out, final_state)
    kernel_outputs = (out, final_state_work)
    scale_value = _optional_float(scale, key_dim ** -0.5)
    layout_buffer = ctypes.create_string_buffer(layout.encode("utf-8"))

    def build_args(ctx):
        return [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(g, "g"),
            ctx.tensor(beta, "beta"),
            ctx.tensor(initial_state_work, "initial_state"),
            ctx.tensor(cu_seqlens, "cu_seqlens"),
            ctx.tensor(ssm_state_indices, "ssm_state_indices"),
            ctx.tensor(A_log, "A_log"),
            ctx.tensor(dt_bias, "dt_bias"),
            ctx.tensor(num_accepted_tokens, "num_accepted_tokens"),
            ctypes.cast(layout_buffer, ctypes.c_char_p),
            ctypes.c_double(float(scale_value)),
            ctypes.c_bool(_optional_bool(output_final_state, False)),
            ctypes.c_bool(_optional_bool(use_qk_l2norm_in_kernel, False)),
            ctypes.c_bool(use_gate),
            ctypes.c_bool(_optional_bool(use_beta_sigmoid_in_kernel, False)),
            ctypes.c_bool(_optional_bool(allow_neg_eigval, False)),
            ctypes.c_bool(safe),
            ctypes.c_double(lower),
            ctypes.c_bool(state_v_first),
            ctx.tensor(out, "out"),
        ]

    _call_aclnn(
        "aclnnRecurrentKda",
        build_args,
        kernel_outputs,
    )
    return user_outputs


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
