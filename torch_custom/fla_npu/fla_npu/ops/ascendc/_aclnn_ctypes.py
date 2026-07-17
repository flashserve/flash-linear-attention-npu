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
import operator

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
    "aclnnRecurrentGatedDeltaRule": [
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
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "aclnnKdaLayoutSwap12": [
        ctypes.c_void_p,
        ctypes.c_void_p,
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


def _flatten_int_values(values, name):
    if values is None:
        return None
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()

    flattened = []

    def append(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                append(item)
            return
        try:
            flattened.append(operator.index(value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{name} must contain integers.") from exc

    append(values)
    return tuple(flattened)


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
    if g is None:
        raise RuntimeError("npu_chunk_gated_delta_rule_bwd_dhu: g must be provided.")
    if gK is not None:
        raise RuntimeError("npu_chunk_gated_delta_rule_bwd_dhu: gK is reserved and must be None.")
    if h0 is not None:
        raise RuntimeError("npu_chunk_gated_delta_rule_bwd_dhu: h0 is reserved and must be None.")
    if dht is not None:
        raise RuntimeError("npu_chunk_gated_delta_rule_bwd_dhu: dht is reserved and must be None.")
    if _optional_bool(use_exp2, False):
        raise RuntimeError("npu_chunk_gated_delta_rule_bwd_dhu: use_exp2 must be False.")
    if _optional_bool(transpose_state_layout, False):
        raise RuntimeError("npu_chunk_gated_delta_rule_bwd_dhu: transpose_state_layout must be False.")

    q_shape = _shape(q)
    dv_shape = _shape(dv)
    B, _, T, K = q_shape
    Hv, V = dv_shape[1], dv_shape[3]
    NT = _chunk_num(T, int(chunk_size), chunk_indices)
    dh = _empty((B, Hv, NT, K, V), q)
    dh0 = None
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
    if g_gamma is not None:
        raise RuntimeError("npu_chunk_bwd_dv_local: g_gamma is reserved and must be None.")
    if A is not None:
        raise RuntimeError("npu_chunk_bwd_dv_local: A is reserved and must be None.")
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
    if w is not None:
        raise RuntimeError("npu_chunk_bwd_dqkwg: w is reserved and must be None.")
    if g_gamma is not None:
        raise RuntimeError("npu_chunk_bwd_dqkwg: g_gamma is reserved and must be None.")
    if _optional_bool(use_exp2, False):
        raise RuntimeError("npu_chunk_bwd_dqkwg: use_exp2 must be False.")
    if _optional_bool(transpose_state_layout, False):
        raise RuntimeError("npu_chunk_bwd_dqkwg: transpose_state_layout must be False.")
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
    if g is None:
        raise RuntimeError("npu_chunk_fwd_o: g is required by the Ascend C kernel.")
    if g_gamma is not None:
        raise RuntimeError("npu_chunk_fwd_o: g_gamma is reserved and must be None.")
    if _optional_bool(transpose_state_layout, False):
        raise RuntimeError("npu_chunk_fwd_o: transpose_state_layout must be False.")
    if (cu_seqlens is None) != (chunk_indices is None):
        raise RuntimeError("npu_chunk_fwd_o: cu_seqlens and chunk_indices must be provided together.")
    chunk_size = _optional_int(chunk_size, 64)
    if chunk_size not in (64, 128):
        raise RuntimeError("npu_chunk_fwd_o: chunk_size must be 64 or 128.")
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
    if chunk_size not in (64, 128):
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: chunk_size must be 64 or 128.")
    if (cu_seqlens is None) != (chunk_indices is None):
        raise RuntimeError(
            "npu_chunk_gated_delta_rule_fwd_h: cu_seqlens and chunk_indices must be provided together."
        )
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
    if g is None:
        raise RuntimeError("npu_recompute_wu_fwd: g is required by the Ascend C kernel.")
    if gk is not None:
        raise RuntimeError("npu_recompute_wu_fwd: gk is reserved and must be None.")
    if int(chunk_size) not in (64, 128):
        raise RuntimeError("npu_recompute_wu_fwd: chunk_size must be 64 or 128.")
    if (cu_seqlens is None) != (chunk_indices is None):
        raise RuntimeError("npu_recompute_wu_fwd: cu_seqlens and chunk_indices must be provided together.")
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


# Public spelling follows the registered operator name; keep the historical
# extra-underscore spelling as a compatibility alias.
npu_recompute_wu_fwd = npu_recompute_w_u_fwd


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
    import torch

    if conv_states is None:
        raise RuntimeError("npu_causal_conv1d: conv_states is required and is updated in place.")
    if x.dim() not in (2, 3):
        raise RuntimeError("npu_causal_conv1d: x must be [T,D] or [B,T,D].")
    if weight.dim() != 2:
        raise RuntimeError("npu_causal_conv1d: weight must be [W,D].")
    width, dim = (int(value) for value in weight.shape)
    if width not in (2, 3, 4):
        raise RuntimeError("npu_causal_conv1d: convolution width W must be 2, 3 or 4.")
    if int(x.shape[-1]) != dim or dim % 16 != 0:
        raise RuntimeError("npu_causal_conv1d: x/weight dimension D must match and be a multiple of 16.")
    if x.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
        raise RuntimeError("npu_causal_conv1d: x and weight must have the same FP16 or BF16 dtype.")
    if conv_states.dim() != 3 or int(conv_states.shape[2]) != dim:
        raise RuntimeError("npu_causal_conv1d: conv_states must be [D_s,L_s,D].")
    if int(conv_states.shape[0]) <= 0 or int(conv_states.shape[1]) < width - 1:
        raise RuntimeError("npu_causal_conv1d: D_s must be positive and L_s must be at least W-1.")
    if conv_states.dtype != x.dtype:
        raise RuntimeError("npu_causal_conv1d: conv_states dtype must equal x dtype.")
    if bias is not None and (bias.dim() != 1 or tuple(bias.shape) != (dim,) or bias.dtype != x.dtype):
        raise RuntimeError("npu_causal_conv1d: bias must be [D] with the same dtype as x.")

    activation_mode = int(activation_mode)
    run_mode = int(run_mode)
    head_num = int(head_num)
    if activation_mode not in (0, 1):
        raise RuntimeError("npu_causal_conv1d: activation_mode must be 0 or 1.")
    if run_mode not in (0, 1):
        raise RuntimeError("npu_causal_conv1d: run_mode must be 0 or 1.")
    if head_num < 0:
        raise RuntimeError("npu_causal_conv1d: head_num must be non-negative.")
    if head_num > 0:
        if run_mode != 0:
            raise RuntimeError("npu_causal_conv1d: head_num is supported only when run_mode=0.")
        if dim % head_num != 0 or (dim // head_num) % 16 != 0:
            raise RuntimeError("npu_causal_conv1d: head_num must divide D and D/head_num must be a multiple of 16.")

    query_values = None if query_start_loc is None else tuple(int(value) for value in query_start_loc)
    if x.dim() == 2 and run_mode == 0 and query_values is None:
        raise RuntimeError("npu_causal_conv1d: query_start_loc is required for rank-2 forward input.")
    if query_values is not None:
        total_tokens = int(x.shape[0]) if x.dim() == 2 else int(x.shape[0]) * int(x.shape[1])
        if len(query_values) < 2 or query_values[0] != 0 or query_values[-1] != total_tokens:
            raise RuntimeError("npu_causal_conv1d: query_start_loc must start at 0 and end at T.")
        if any(right < left for left, right in zip(query_values, query_values[1:])):
            raise RuntimeError("npu_causal_conv1d: query_start_loc must be non-decreasing.")
        batch = len(query_values) - 1
        if x.dim() == 3 and batch != int(x.shape[0]):
            raise RuntimeError("npu_causal_conv1d: query_start_loc must contain B+1 entries.")
    else:
        batch = int(x.shape[0])

    def _optional_values(name, values):
        if values is None:
            return None
        result = tuple(int(value) for value in values)
        if len(result) != batch:
            raise RuntimeError(f"npu_causal_conv1d: {name} must contain B entries.")
        return result

    cache_values = _optional_values("cache_indices", cache_indices)
    initial_values = _optional_values("initial_state_mode", initial_state_mode)
    accepted_values = _optional_values("num_accepted_tokens", num_accepted_tokens)
    if cache_values is None and int(conv_states.shape[0]) < batch:
        raise RuntimeError("npu_causal_conv1d: D_s must be at least B when cache_indices is absent.")
    if cache_values is not None:
        state_count = int(conv_states.shape[0])
        if any(value != int(pad_slot_id) and not 0 <= value < state_count for value in cache_values):
            raise RuntimeError("npu_causal_conv1d: cache_indices must select a valid state slot or pad_slot_id.")
    if initial_values is not None:
        if run_mode != 0 or any(value not in (0, 1) for value in initial_values):
            raise RuntimeError("npu_causal_conv1d: initial_state_mode requires run_mode=0 and values 0 or 1.")
    if accepted_values is not None:
        if run_mode != 1 or width != 4:
            raise RuntimeError("npu_causal_conv1d: num_accepted_tokens requires run_mode=1 and W=4.")
        if x.dim() == 3:
            token_limits = (int(x.shape[1]),) * batch
            required_state_len = width - 1 + int(x.shape[1]) - 1
            if int(conv_states.shape[1]) < required_state_len:
                raise RuntimeError("npu_causal_conv1d: rank-3 speculative update requires L_s >= (W-1)+(T-1).")
        elif query_values is None:
            token_limits = (1,) * batch
        else:
            token_limits = tuple(right - left for left, right in zip(query_values, query_values[1:]))
        if any(value < 0 or value > limit for value, limit in zip(accepted_values, token_limits)):
            raise RuntimeError("npu_causal_conv1d: num_accepted_tokens must be within each sequence token count.")

    out = _infer_causal_conv1d_y(x, int(head_num), int(run_mode))
    return _call_aclnn(
        "aclnnCausalConv1d",
        lambda ctx: [
            ctx.tensor(x, "x"),
            ctx.tensor(weight, "weight"),
            ctx.tensor(bias, "bias"),
            ctx.tensor(conv_states, "conv_states"),
            ctx.int_array(query_values),
            ctx.int_array(cache_values),
            ctx.int_array(initial_values),
            ctx.int_array(accepted_values),
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
    import torch

    input_layout = str(input_layout)
    if input_layout not in {"BSH", "BSND", "BNSD", "TND", "NTD"}:
        raise RuntimeError("npu_causal_conv1d_bwd: input_layout must be BSH, BSND, BNSD, TND or NTD.")
    activation = int(activation)
    if activation not in (0, 1, 2):
        raise RuntimeError("npu_causal_conv1d_bwd: activation must be 0, 1 or 2.")
    if weight.dim() != 2:
        raise RuntimeError("npu_causal_conv1d_bwd: weight must be [W,D].")
    width, dim = int(weight.shape[0]), int(weight.shape[1])
    if width not in (2, 3, 4):
        raise RuntimeError("npu_causal_conv1d_bwd: convolution width W must be 2, 3 or 4.")
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    if x.dtype not in supported_dtypes or weight.dtype != x.dtype or dy.dtype != x.dtype:
        raise RuntimeError("npu_causal_conv1d_bwd: x, weight and dy must share an FP32, FP16 or BF16 dtype.")

    if input_layout == "BNSD":
        if x.dim() != 3 or dy.dim() != 4:
            raise RuntimeError("npu_causal_conv1d_bwd: BNSD requires x [B,T,D] and dy [B,H_v,T,V].")
        batch = int(x.shape[0])
        if tuple(dy.shape[:1]) != tuple(x.shape[:1]) or int(dy.shape[2]) != int(x.shape[1]):
            raise RuntimeError("npu_causal_conv1d_bwd: BNSD dy batch/time dimensions must match x.")
        if int(dy.shape[1]) * int(dy.shape[3]) != dim or int(dy.shape[3]) % 16 != 0:
            raise RuntimeError("npu_causal_conv1d_bwd: BNSD requires D=H_v*V and V a multiple of 16.")
        dx_shape = _shape(x)
    elif input_layout in {"NTD", "TND"}:
        if query_start_loc is None:
            raise RuntimeError(f"query_start_loc is required for {input_layout} input.")
        query_values = tuple(int(value) for value in query_start_loc)
        if len(query_values) < 2 or query_values[0] != 0 or query_values[-1] != int(x.shape[0]):
            raise RuntimeError("npu_causal_conv1d_bwd: query_start_loc must start at 0 and end at total T.")
        if any(right < left for left, right in zip(query_values, query_values[1:])):
            raise RuntimeError("npu_causal_conv1d_bwd: query_start_loc must be non-decreasing.")
        batch = len(query_values) - 1
        if x.dim() != 2:
            raise RuntimeError(f"npu_causal_conv1d_bwd: {input_layout} requires x [T,D].")
        if input_layout == "TND":
            if dy.dim() != 2 or tuple(dy.shape) != tuple(x.shape):
                raise RuntimeError("npu_causal_conv1d_bwd: TND requires dy to match x [T,D].")
        elif dy.dim() != 3 or int(dy.shape[1]) != int(x.shape[0]) or int(dy.shape[0]) * int(dy.shape[2]) != dim:
            raise RuntimeError("npu_causal_conv1d_bwd: NTD requires dy [H_v,T,V] with D=H_v*V.")
        if input_layout == "NTD" and int(dy.shape[2]) % 16 != 0:
            raise RuntimeError("npu_causal_conv1d_bwd: NTD head dimension V must be a multiple of 16.")
        dx_shape = _shape(x)
    else:
        if x.dim() != 3 or dy.dim() != 3 or tuple(dy.shape) != tuple(x.shape):
            raise RuntimeError("npu_causal_conv1d_bwd: BSH/BSND requires matching x and dy [B,T,D].")
        batch = int(x.shape[0])
        dx_shape = _shape(x)
    if int(x.shape[-1]) != dim:
        raise RuntimeError("npu_causal_conv1d_bwd: x and weight dimension D must match.")
    if input_layout not in {"BNSD", "NTD"} and dim % 16 != 0:
        raise RuntimeError("npu_causal_conv1d_bwd: D must be a multiple of 16.")
    if activation != 0 and y is None:
        raise RuntimeError("npu_causal_conv1d_bwd: y is required when activation is 1 or 2.")
    if y is not None and (tuple(y.shape) != tuple(dy.shape) or y.dtype != x.dtype):
        raise RuntimeError("npu_causal_conv1d_bwd: y must match dy shape and dtype.")
    for name, value in (("initial_state", initial_state), ("dht", dht)):
        if value is not None and (tuple(value.shape) != (batch, width, dim) or value.dtype != x.dtype):
            raise RuntimeError(f"npu_causal_conv1d_bwd: {name} must be [B,W,D] with the same dtype as x.")

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
    import math
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
    if any(int(dim) <= 0 for shape in (q_shape, k_shape, v_shape, gk_shape, beta_shape) for dim in shape):
        raise RuntimeError("npu_chunk_kda_fwd: all input dimensions must be positive.")
    if not math.isfinite(float(scale)):
        raise RuntimeError("npu_chunk_kda_fwd: scale must be finite.")
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
    if cu_seqlens_for_call is not None and any(
        right < left for left, right in zip(cu_seqlens_for_call, cu_seqlens_for_call[1:])
    ):
        raise RuntimeError("npu_chunk_kda_fwd: cu_seqlens must be non-decreasing.")
    if cu_seqlens_for_call is not None and not is_rank3 and batch != 1:
        raise RuntimeError("npu_chunk_kda_fwd: rank-4 varlen input requires physical B=1.")
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

    if g.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("npu_kda_gate_cumsum: g must use FP16, BF16 or FP32.")
    if g.dim() not in (3, 4) or any(int(dim) <= 0 for dim in g.shape):
        raise RuntimeError("npu_kda_gate_cumsum: g must have a positive rank-3 or rank-4 shape.")
    chunk_size = int(chunk_size)
    if chunk_size not in (32, 64, 128):
        raise RuntimeError("npu_kda_gate_cumsum: chunk_size must be 32, 64 or 128.")
    out = _empty(_shape(g), g, dtype=torch.float32)
    layout = str(layout)
    if layout not in ("BSND", "BNSD", "TND", "NTD"):
        raise ValueError("layout must be uppercase and one of BSND, BNSD, TND or NTD")
    if (g.dim() == 4) != (layout in ("BSND", "BNSD")):
        raise RuntimeError("npu_kda_gate_cumsum: BSND/BNSD require rank 4 and TND/NTD require rank 3.")
    time = int(g.shape[2] if layout == "BNSD" else g.shape[1] if layout in ("BSND", "NTD") else g.shape[0])
    value_heads = int(g.shape[1] if layout == "BNSD" else g.shape[0] if layout == "NTD" else g.shape[-2])
    key_dim = int(g.shape[-1])
    if key_dim > 256:
        raise RuntimeError("npu_kda_gate_cumsum: K must be <= 256.")
    cu_values = None if cu_seqlens is None else tuple(int(value) for value in cu_seqlens)
    if cu_values is not None:
        if len(cu_values) < 2 or cu_values[0] != 0 or cu_values[-1] != time:
            raise RuntimeError("npu_kda_gate_cumsum: cu_seqlens must start at 0 and end at T.")
        if any(right < left for left, right in zip(cu_values, cu_values[1:])):
            raise RuntimeError("npu_kda_gate_cumsum: cu_seqlens must be non-decreasing.")
        if g.dim() == 4 and int(g.shape[0]) != 1:
            raise RuntimeError("npu_kda_gate_cumsum: rank-4 varlen input requires physical B=1.")
    use_gate = _optional_bool(use_gate_in_kernel, False)
    safe = _optional_bool(safe_gate, False)
    lower = _optional_float(lower_bound, -5.0)
    if use_gate:
        if A_log is None or A_log.dtype != torch.float32 or tuple(A_log.shape) != (value_heads,):
            raise RuntimeError("npu_kda_gate_cumsum: A_log must be FP32 [H_v] for raw-gate mode.")
        if not safe:
            raise RuntimeError("npu_kda_gate_cumsum: raw-gate mode requires safe_gate=True.")
        if not -5.0 <= lower < 0.0:
            raise RuntimeError("npu_kda_gate_cumsum: lower_bound must be in [-5,0).")
        if dt_bias is not None:
            valid_bias_shape = tuple(dt_bias.shape) in ((value_heads * key_dim,), (value_heads, key_dim))
            if dt_bias.dtype != torch.float32 or not valid_bias_shape:
                raise RuntimeError("npu_kda_gate_cumsum: dt_bias must be FP32 [H_v*K] or [H_v,K].")
    else:
        if safe:
            raise RuntimeError("npu_kda_gate_cumsum: safe_gate applies only to raw-gate mode.")
        if A_log is not None or dt_bias is not None:
            raise RuntimeError("npu_kda_gate_cumsum: A_log and dt_bias must be None in step-gate mode.")
    layout_buffer = ctypes.create_string_buffer(layout.encode("utf-8"))
    return _call_aclnn(
        "aclnnKdaGateCumsum",
        lambda ctx: [
            ctx.tensor(g, "g"),
            ctx.tensor(A_log, "A_log"),
            ctx.tensor(dt_bias, "dt_bias"),
            ctx.int_array(cu_values),
            ctypes.c_int64(chunk_size),
            ctypes.c_bool(use_gate),
            ctypes.c_bool(safe),
            ctypes.c_double(lower),
            ctypes.cast(layout_buffer, ctypes.c_char_p),
            ctx.tensor(out, "gk"),
        ],
        out,
    )


def npu_kda_layout_swap12(x, *, dependency=None):
    import torch

    x_shape = list(_shape(x))
    if len(x_shape) < 3:
        raise RuntimeError("npu_kda_layout_swap12: x rank must be at least 3.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("npu_kda_layout_swap12: x must use FP16, BF16 or FP32.")
    if any(int(dim) <= 0 for dim in x_shape):
        raise RuntimeError("npu_kda_layout_swap12: all x dimensions must be positive.")
    if len(x_shape) == 3:
        x_shape[0], x_shape[1] = x_shape[1], x_shape[0]
    else:
        x_shape[1], x_shape[2] = x_shape[2], x_shape[1]
    out = _empty(tuple(x_shape), x)
    return _call_aclnn(
        "aclnnKdaLayoutSwap12",
        lambda ctx: [
            ctx.tensor(x, "x"),
            ctx.tensor(dependency, "dependency"),
            ctx.tensor(out, "y"),
        ],
        out,
    )


def npu_recurrent_gated_delta_rule(
    query,
    key,
    value,
    beta,
    state_ref,
    actual_seq_lengths,
    ssm_state_indices,
    *,
    g=None,
    gk=None,
    num_accepted_tokens=None,
    scale_value=1.0,
):
    out = _empty_like(value)
    outputs = (out, state_ref)
    return _call_aclnn(
        "aclnnRecurrentGatedDeltaRule",
        lambda ctx: [
            ctx.tensor(query, "query"),
            ctx.tensor(key, "key"),
            ctx.tensor(value, "value"),
            ctx.tensor(beta, "beta"),
            ctx.tensor(state_ref, "state_ref"),
            ctx.tensor(actual_seq_lengths, "actual_seq_lengths"),
            ctx.tensor(ssm_state_indices, "ssm_state_indices"),
            ctx.tensor(g, "g"),
            ctx.tensor(gk, "gk"),
            ctx.tensor(num_accepted_tokens, "num_accepted_tokens"),
            ctypes.c_float(float(scale_value)),
            ctx.tensor(out, "out"),
        ],
        outputs,
    )


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
    import torch

    if g.dim() < 3 or any(int(dim) <= 0 for dim in g.shape):
        raise RuntimeError("npu_chunk_local_cumsum: g must have positive shape [B,H_v,T,...].")
    if g.dtype != torch.float32:
        raise RuntimeError("npu_chunk_local_cumsum: g and out support float32 only.")
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise RuntimeError("npu_chunk_local_cumsum: chunk_size must be a positive power of two.")
    output_dtype = str(output_dtype)
    if output_dtype not in {"float32", "torch.float", "torch.float32"}:
        raise RuntimeError("npu_chunk_local_cumsum: output_dtype must select float32.")
    if not _optional_bool(head_first, True):
        raise RuntimeError("npu_chunk_local_cumsum: head_first=False is not supported by the Ascend C kernel.")
    if (cu_seqlens is None) != (chunk_indices_out is None):
        raise RuntimeError("npu_chunk_local_cumsum: cu_seqlens and chunk_indices_out must be provided together.")
    cu_values = _flatten_int_values(cu_seqlens, "npu_chunk_local_cumsum: cu_seqlens")
    index_values = _flatten_int_values(chunk_indices_out, "npu_chunk_local_cumsum: chunk_indices_out")
    if cu_values is not None:
        if int(g.shape[0]) != 1:
            raise RuntimeError("npu_chunk_local_cumsum: varlen mode requires physical B=1.")
        if len(cu_values) < 2:
            raise RuntimeError("npu_chunk_local_cumsum: cu_seqlens must contain at least two integer values.")
        if not index_values or len(index_values) % 2 != 0:
            raise RuntimeError("npu_chunk_local_cumsum: chunk_indices_out must contain flattened integer pairs.")
    output_dtype_buffer = ctypes.create_string_buffer(output_dtype.encode("utf-8"))
    out = _empty_like(g, dtype=torch.float32)
    return _call_aclnn(
        "aclnnChunkLocalCumsum",
        lambda ctx: [
            ctx.tensor(g, "g"),
            ctx.int_array(cu_values),
            ctx.int_array(index_values),
            ctypes.c_int64(int(chunk_size)),
            ctypes.c_bool(_optional_bool(reverse, False)),
            ctypes.c_double(float(scale)),
            ctypes.c_bool(True),
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

    k_shape = _shape(k)
    if len(k_shape) != 4:
        raise RuntimeError("npu_chunk_scaled_dot_kkt: k must have shape [B,H_k,T,K].")
    if k.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("npu_chunk_scaled_dot_kkt: k must use FP16 or BF16.")
    if g.dim() != 3 or beta.dim() != 3 or tuple(g.shape) != tuple(beta.shape):
        raise RuntimeError("npu_chunk_scaled_dot_kkt: g and beta must have the same [B,H_v,T] shape.")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise RuntimeError("npu_chunk_scaled_dot_kkt: g and beta must use FP32.")
    batch, key_heads, time, _ = (int(value) for value in k_shape)
    if int(g.shape[0]) != batch or int(g.shape[2]) != time or int(g.shape[1]) % key_heads != 0:
        raise RuntimeError("npu_chunk_scaled_dot_kkt: g/beta must match B,T and H_v must be divisible by H_k.")
    chunk_size = int(chunk_size)
    if chunk_size not in (16, 32, 64, 128):
        raise RuntimeError("npu_chunk_scaled_dot_kkt: chunk_size must be 16, 32, 64 or 128.")
    if (cu_seqlens is None) != (chunk_indices is None):
        raise RuntimeError("npu_chunk_scaled_dot_kkt: cu_seqlens and chunk_indices must be provided together.")
    if cu_seqlens is not None:
        if batch != 1:
            raise RuntimeError("npu_chunk_scaled_dot_kkt: varlen mode requires physical B=1.")
        cu_values = tuple(int(value) for value in cu_seqlens)
        index_values = tuple(int(value) for value in chunk_indices)
        if len(cu_values) < 2 or cu_values[0] != 0 or cu_values[-1] != time:
            raise RuntimeError("npu_chunk_scaled_dot_kkt: cu_seqlens must start at 0 and end at T.")
        if any(right < left for left, right in zip(cu_values, cu_values[1:])):
            raise RuntimeError("npu_chunk_scaled_dot_kkt: cu_seqlens must be non-decreasing.")
        expected_indices = []
        for seq_id, (left, right) in enumerate(zip(cu_values, cu_values[1:])):
            for local_chunk in range((right - left + chunk_size - 1) // chunk_size):
                expected_indices.extend((seq_id, local_chunk))
        if index_values != tuple(expected_indices):
            raise RuntimeError("npu_chunk_scaled_dot_kkt: chunk_indices must list every varlen chunk in sequence-major order.")
    out = _empty((batch, key_heads, time, chunk_size), k, dtype=torch.float32)
    return _call_aclnn(
        "aclnnChunkScaledDotKkt",
        lambda ctx: [
            ctx.tensor(k, "k"),
            ctx.tensor(g, "g"),
            ctx.tensor(beta, "beta"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(out, "A"),
        ],
        out,
    )


def npu_solve_tri(x, *, cu_seqlens=None, chunk_indices=None, layout="bsnd"):
    layout = str(layout)
    if layout not in {"bhtd", "bsnd", "tnd"}:
        raise RuntimeError("npu_solve_tri: layout must be lowercase bhtd, bsnd or tnd.")
    expected_rank = 3 if layout == "tnd" else 4
    if x.dim() != expected_rank:
        raise RuntimeError(f"npu_solve_tri: layout {layout} requires a rank-{expected_rank} input.")
    if (cu_seqlens is None) != (chunk_indices is None):
        raise RuntimeError("npu_solve_tri: cu_seqlens and chunk_indices must be provided together.")
    if layout == "tnd" and cu_seqlens is None:
        raise RuntimeError("npu_solve_tri: tnd layout requires cu_seqlens and chunk_indices.")
    if layout != "tnd" and cu_seqlens is not None:
        raise RuntimeError("npu_solve_tri: varlen indices are only valid with tnd layout.")
    if x.shape[-1] not in (16, 32, 64, 128):
        raise RuntimeError("npu_solve_tri: the last dimension C must be 16, 32, 64 or 128.")
    x_contig = x.contiguous()
    out = _empty_like(x_contig)
    layout_arg = ctypes.c_char_p(layout.encode("utf-8"))
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
