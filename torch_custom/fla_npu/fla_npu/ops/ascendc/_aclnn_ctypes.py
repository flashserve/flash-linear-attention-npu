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
    if g is None:
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: g cannot be None.")
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
        like = initial_state if initial_state is not None else h_out
        final_state_out = _empty((N, HV, K, V), like)
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
