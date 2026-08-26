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
import sys

from ._kda_policy import kda_fwd_optional_output_mask
from ._runtime import (
    ACL_FORMAT_NCDHW,
    ACL_FORMAT_NCHW,
    ACL_FORMAT_NCL,
    ACL_FORMAT_ND,
    acl_format as _acl_format,
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

RGDR_DIM_ALIGNMENT = 16
RGDR_MAX_DIM = 128

# Most aclnn functions only receive pointer-sized descriptors and scalar ctypes
# objects, so ctypes can call them without explicit argtypes.  Functions with C
# strings or otherwise ambiguous scalar conversion are listed here to prevent
# ctypes from narrowing or mis-converting arguments.
_GET_WORKSPACE_ARGTYPES = {
    "aclnnPrepareWyReprBwd": [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "aclnnCausalConv1dBwd": [
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # yOptional
        ctypes.c_void_p,  # weight
        ctypes.c_void_p,  # dy
        ctypes.c_void_p,  # initialStateOptional
        ctypes.c_void_p,  # dhtOptional
        ctypes.c_void_p,  # queryStartLocOptional
        ctypes.c_int64,  # activation
        ctypes.c_char_p,  # inputLayoutOptional
        ctypes.c_void_p,  # dxOut
        ctypes.c_void_p,  # dwOutOptional
        ctypes.c_void_p,  # dbOutOptional
        ctypes.c_void_p,  # dh0OutOptional
        ctypes.POINTER(ctypes.c_uint64),  # workspaceSize
        ctypes.POINTER(ctypes.c_void_p),  # executor
    ],
    "aclnnChunkGatedDeltaRuleBwdDhu": [
        ctypes.c_void_p,  # q
        ctypes.c_void_p,  # k
        ctypes.c_void_p,  # w
        ctypes.c_void_p,  # dO
        ctypes.c_void_p,  # dv
        ctypes.c_void_p,  # gOptional
        ctypes.c_void_p,  # gkOptional
        ctypes.c_void_p,  # h0Optional
        ctypes.c_void_p,  # dhtOptional
        ctypes.c_void_p,  # cuSeqlensOptional
        ctypes.c_void_p,  # chunkIndicesOptional
        ctypes.c_double,  # scale
        ctypes.c_int64,  # chunkSize
        ctypes.c_bool,  # useExp2
        ctypes.c_void_p,  # dhOut
        ctypes.c_void_p,  # dh0Out
        ctypes.c_void_p,  # dv2Out
        ctypes.POINTER(ctypes.c_uint64),  # workspaceSize
        ctypes.POINTER(ctypes.c_void_p),  # executor
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
    "aclnnChunkKdaBwdIntra": [
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
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_bool,
        ctypes.c_char_p,
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
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "aclnnRecurrentKda": [
        ctypes.c_void_p,  # query
        ctypes.c_void_p,  # key
        ctypes.c_void_p,  # value
        ctypes.c_void_p,  # gate
        ctypes.c_void_p,  # beta
        ctypes.c_void_p,  # initialStateRef
        ctypes.c_void_p,  # cuSeqlensOptional
        ctypes.c_void_p,  # ssmStateIndicesOptional
        ctypes.c_void_p,  # aLogOptional
        ctypes.c_void_p,  # dtBiasOptional
        ctypes.c_void_p,  # numAcceptedTokensOptional
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_bool,  # outputFinalState
        ctypes.c_bool,  # inplaceFinalState
        ctypes.c_bool,  # useQkL2normInKernel
        ctypes.c_bool,  # useGateInKernel
        ctypes.c_bool,  # useBetaSigmoidInKernel
        ctypes.c_bool,  # allowNegEigval
        ctypes.c_bool,  # safeGate
        ctypes.c_double,
        ctypes.c_bool,  # stateVFirst
        ctypes.c_void_p,  # attnOut
        ctypes.c_void_p,  # finalState
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    "aclnnRecurrentGatedDeltaRule": [
        ctypes.c_void_p,  # query
        ctypes.c_void_p,  # key
        ctypes.c_void_p,  # value
        ctypes.c_void_p,  # beta
        ctypes.c_void_p,  # stateRef
        ctypes.c_void_p,  # actualSeqLengths
        ctypes.c_void_p,  # ssmStateIndices
        ctypes.c_void_p,  # g
        ctypes.c_void_p,  # gk
        ctypes.c_void_p,  # numAcceptedTokens
        ctypes.c_float,  # scaleValue
        ctypes.c_void_p,  # out
        ctypes.POINTER(ctypes.c_uint64),  # workspaceSize
        ctypes.POINTER(ctypes.c_void_p),  # executor
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
    "aclnnChunkFwdH": [
        *([ctypes.c_void_p] * 6),
        ctypes.c_bool,
        ctypes.c_int64,
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
}


def _call_aclnn(
    name: str,
    build_args,
    outputs,
):
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


def npu_prepare_wy_repr_bwd(
    k,
    v,
    beta,
    A,
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
        "aclnnPrepareWyReprBwd",
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
    import torch

    q_shape = _shape(q)
    dv_shape = _shape(dv)
    B, _, T, K = q_shape
    Hv, V = dv_shape[1], dv_shape[3]
    if (g is None) == (gK is None):
        raise ValueError("Exactly one of g and gK must be provided.")
    if gK is not None and not _optional_bool(use_exp2, True):
        raise ValueError("use_exp2 must be true when gK is provided.")
    if any(tensor.dtype != q.dtype for tensor in (k, w, d_o, dv)):
        raise ValueError("q, k, w, d_o and dv must have the same dtype.")
    gate = g if g is not None else gK
    if gate.dtype not in (q.dtype, torch.float32):
        raise ValueError("g or gK must be float32 or have the same dtype as q and k.")
    if g is not None and _shape(g) != (B, Hv, T):
        raise ValueError(f"g must have shape {(B, Hv, T)}, got {_shape(g)}.")
    if gK is not None and _shape(gK) != (B, Hv, T, K):
        raise ValueError(f"gK must have shape {(B, Hv, T, K)}, got {_shape(gK)}.")
    NT = _chunk_num(T, int(chunk_size), chunk_indices)
    dh = _empty((B, Hv, NT, K, V), q)
    dh0 = _empty((B, Hv, NT, K, V), q) if h0 is not None else None
    dv2 = _empty_like(dv)
    outputs = (dh, dh0, dv2)

    def logical_tensor(ctx, tensor, name):
        if tensor is None:
            return ctx.tensor(tensor, name)
        return ctx.tensor(tensor, name, storage_shape_override=_shape(tensor))

    return _call_aclnn(
        "aclnnChunkGatedDeltaRuleBwdDhu",
        lambda ctx: [
            logical_tensor(ctx, q, "q"),
            logical_tensor(ctx, k, "k"),
            logical_tensor(ctx, w, "w"),
            logical_tensor(ctx, d_o, "d_o"),
            logical_tensor(ctx, dv, "dv"),
            logical_tensor(ctx, g, "g"),
            logical_tensor(ctx, gK, "gK"),
            logical_tensor(ctx, h0, "h0"),
            logical_tensor(ctx, dht, "dht"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(int(chunk_size)),
            ctypes.c_bool(_optional_bool(use_exp2, gK is not None)),
            logical_tensor(ctx, dh, "dh"),
            logical_tensor(ctx, dh0, "dh0"),
            logical_tensor(ctx, dv2, "dv2"),
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


def _chunk_fwd_h_ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _chunk_fwd_h_build_chunk_indices(cu_seqlens, chunk_size: int):
    if cu_seqlens is None:
        return None
    cu = tuple(int(value) for value in cu_seqlens)
    indices = []
    for sequence, (begin, end) in enumerate(zip(cu, cu[1:])):
        for chunk in range(_chunk_fwd_h_ceil_div(end - begin, chunk_size)):
            indices.extend((sequence, chunk))
    return tuple(indices)


def _chunk_fwd_h_total_chunks(seqlen: int, chunk_size: int, cu_seqlens, chunk_indices) -> int:
    if chunk_indices is not None:
        return len(tuple(chunk_indices)) // 2
    if cu_seqlens is None:
        return _chunk_fwd_h_ceil_div(seqlen, chunk_size)
    cu = tuple(int(value) for value in cu_seqlens)
    return sum(
        _chunk_fwd_h_ceil_div(end - begin, chunk_size)
        for begin, end in zip(cu, cu[1:])
    )


def npu_chunk_fwd_h(
    k,
    w,
    u,
    *,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    state_v_first=False,
):
    import torch

    op_name = "npu_chunk_fwd_h"
    if (g is None) == (gk is None):
        raise RuntimeError(f"{op_name}: exactly one of g and gk must be provided.")
    output_final_state = _optional_bool(output_final_state, False)
    save_new_value = _optional_bool(save_new_value, True)
    use_exp2 = _optional_bool(use_exp2, False)
    state_v_first = _optional_bool(state_v_first, False)
    chunk_size = _optional_int(chunk_size, 64)
    if chunk_size != 64:
        raise RuntimeError(f"{op_name}: chunk_size must be 64.")
    if not save_new_value:
        raise RuntimeError(f"{op_name}: save_new_value must be True.")
    if len(_shape(k)) != 4 or len(_shape(w)) != 4 or len(_shape(u)) != 4:
        raise RuntimeError(f"{op_name}: k, w and u must be rank-4 BNSD tensors.")

    batch, k_heads, seqlen, k_dim = _shape(k)
    _, v_heads, _, v_dim = _shape(u)
    if batch <= 0 or k_heads <= 0 or v_heads <= 0 or seqlen <= 0:
        raise RuntimeError(f"{op_name}: B, HK, HV and T must all be positive.")
    if k.dtype != torch.bfloat16 or w.dtype != k.dtype or u.dtype != k.dtype:
        raise RuntimeError(f"{op_name}: k, w and u must all use bfloat16.")
    if k_dim != 128 or v_dim != 128:
        raise RuntimeError(f"{op_name}: K and V must both be 128.")
    if _shape(w) != (batch, v_heads, seqlen, k_dim) or _shape(u) != (
        batch,
        v_heads,
        seqlen,
        v_dim,
    ):
        raise RuntimeError(f"{op_name}: w/u must be [B, HV, T, K/V].")

    gate_dtype = g.dtype if g is not None else gk.dtype
    if gate_dtype not in {torch.bfloat16, torch.float32}:
        raise RuntimeError(f"{op_name}: g/gk must use bfloat16 or float32.")
    if g is not None:
        if v_heads < k_heads or v_heads % k_heads != 0:
            raise RuntimeError(f"{op_name}: g-only mode requires HV >= HK and HV % HK == 0.")
        if _shape(g) != (batch, v_heads, seqlen):
            raise RuntimeError(f"{op_name}: g must be [B, HV, T].")
    else:
        if k_heads != v_heads:
            raise RuntimeError(f"{op_name}: gk-only mode requires prepared kg to have HV heads.")
        if _shape(gk) != (batch, v_heads, seqlen, k_dim):
            raise RuntimeError(f"{op_name}: gk must be [B, HV, T, K].")

    cu = None if cu_seqlens is None else tuple(int(value) for value in cu_seqlens)
    if cu is not None:
        if batch != 1:
            raise RuntimeError(f"{op_name}: variable-length BNSD input requires B=1.")
        if len(cu) < 2 or cu[0] != 0 or cu[-1] != seqlen or any(a >= b for a, b in zip(cu, cu[1:])):
            raise RuntimeError(
                f"{op_name}: cu_seqlens must be strictly increasing, start at 0 and end at T."
            )
    canonical_indices = _chunk_fwd_h_build_chunk_indices(cu, chunk_size)
    indices = canonical_indices if chunk_indices is None else tuple(int(value) for value in chunk_indices)
    if indices is not None and indices != canonical_indices:
        raise RuntimeError(f"{op_name}: chunk_indices must use canonical sequence-major order.")

    total_chunks = _chunk_fwd_h_total_chunks(seqlen, chunk_size, cu, indices)
    sequences = len(cu) - 1 if cu is not None else batch
    state_tail = (v_dim, k_dim) if state_v_first else (k_dim, v_dim)
    if initial_state is not None:
        if _shape(initial_state) != (sequences, v_heads, *state_tail):
            raise RuntimeError(f"{op_name}: initial_state shape does not match state_v_first.")
        if initial_state.dtype not in {torch.bfloat16, torch.float32}:
            raise RuntimeError(f"{op_name}: initial_state must use bfloat16 or float32.")

    h_out = _empty((batch, v_heads, total_chunks, *state_tail), k)
    v_new_out = _empty(_shape(u), u)
    if output_final_state:
        state_template = initial_state if initial_state is not None else k
        state_dtype = initial_state.dtype if initial_state is not None else torch.float32
        final_state_out = _empty(
            (sequences, v_heads, *state_tail), state_template, dtype=state_dtype
        )
    else:
        final_state_out = None
    outputs = (h_out, v_new_out, final_state_out if output_final_state else None)

    # ChunkFwdH 的公开布局契约是 ND。标准连续 rank-4/5 NPU tensor 的物理存储
    # 仍是行主序，但通用 runtime 会按维数推断 NCHW/NCDHW，因此这里由本算子
    # 显式覆盖 descriptor 元数据，不触发格式转换或额外数据搬运。
    def nd_tensor(ctx, tensor, name):
        if tensor is None:
            return ctx.tensor(None, name)
        loaded_torch_npu = sys.modules.get("torch_npu")
        if loaded_torch_npu is not None:
            try:
                actual_format = int(loaded_torch_npu.get_npu_format(tensor))
            except Exception as exc:
                raise RuntimeError(
                    f"{op_name}: cannot determine the real NPU format of {name}."
                ) from exc
        else:
            actual_format = _acl_format(tensor)
        standard_formats = {
            ACL_FORMAT_NCHW,
            ACL_FORMAT_ND,
            ACL_FORMAT_NCDHW,
            ACL_FORMAT_NCL,
        }
        if actual_format not in standard_formats:
            raise RuntimeError(
                f"{op_name}: {name} must use a standard contiguous-compatible layout; "
                f"private NPU format {actual_format} is not supported."
            )
        storage_shape = (
            _shape(tensor)
            if tensor.is_contiguous() and int(tensor.storage_offset()) == 0
            else None
        )
        return ctx.tensor(
            tensor,
            name,
            acl_format_override=ACL_FORMAT_ND,
            storage_shape_override=storage_shape,
        )

    return _call_aclnn(
        "aclnnChunkFwdH",
        lambda ctx: [
            nd_tensor(ctx, k, "k"),
            nd_tensor(ctx, w, "w"),
            nd_tensor(ctx, u, "u"),
            nd_tensor(ctx, g, "g"),
            nd_tensor(ctx, gk, "gk"),
            nd_tensor(ctx, initial_state, "initial_state"),
            ctypes.c_bool(output_final_state),
            ctypes.c_int64(chunk_size),
            ctypes.c_bool(save_new_value),
            ctx.int_array(cu),
            ctx.int_array(indices),
            ctypes.c_bool(use_exp2),
            ctypes.c_bool(state_v_first),
            nd_tensor(ctx, h_out, "h"),
            nd_tensor(ctx, v_new_out, "v_new"),
            nd_tensor(ctx, final_state_out, "final_state"),
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


def npu_recurrent_gated_delta_rule(
    query,
    key,
    value,
    state,
    *,
    beta,
    scale=1.0,
    actual_seq_lengths,
    ssm_state_indices,
    num_accepted_tokens=None,
    g=None,
    gk=None,
):
    """
    Run recurrent GDN and update ``state`` in place.

    ``g`` and ``gk`` are independent optional gates. Passing ``None`` for
    either input disables that decay term by using an effective factor of one,
    but one of ``g`` or ``gk`` must be provided.
    """

    op_name = "npu_recurrent_gated_delta_rule"
    if g is None and gk is None:
        raise RuntimeError(f"{op_name}: either g or gk must be provided.")

    import math
    import torch

    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "state": state,
        "beta": beta,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
    }
    optional_tensors = {
        "g": g,
        "gk": gk,
        "num_accepted_tokens": num_accepted_tokens,
    }

    for name, tensor in (*tensors.items(), *optional_tensors.items()):
        if tensor is None:
            if name in tensors:
                raise TypeError(f"{op_name}: {name} must be a torch.Tensor, got None.")
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{op_name}: {name} must be a torch.Tensor, got {type(tensor)!r}.")
        if tensor.device.type != "npu":
            raise TypeError(f"{op_name}: {name} must be a torch NPU tensor, got device {tensor.device}.")

    dtype_requirements = {
        "query": (torch.bfloat16,),
        "key": (torch.bfloat16,),
        "value": (torch.bfloat16,),
        "beta": (torch.bfloat16,),
        "state": (torch.bfloat16, torch.float32),
        "actual_seq_lengths": (torch.int32,),
        "ssm_state_indices": (torch.int32,),
        "g": (torch.float32,),
        "gk": (torch.float32,),
        "num_accepted_tokens": (torch.int32,),
    }
    for name, supported_dtypes in dtype_requirements.items():
        tensor = tensors.get(name, optional_tensors.get(name))
        if tensor is not None and tensor.dtype not in supported_dtypes:
            supported = ", ".join(str(dtype) for dtype in supported_dtypes)
            raise TypeError(
                f"{op_name}: {name} dtype must be one of ({supported}), got {tensor.dtype}."
            )

    shapes = {
        name: _shape(tensor)
        for name, tensor in (*tensors.items(), *optional_tensors.items())
        if tensor is not None
    }
    expected_ranks = {
        "query": 3,
        "key": 3,
        "value": 3,
        "beta": 2,
        "state": 4,
        "actual_seq_lengths": 1,
        "ssm_state_indices": 1,
        "g": 2,
        "gk": 3,
        "num_accepted_tokens": 1,
    }
    for name, expected_rank in expected_ranks.items():
        shape = shapes.get(name)
        if shape is not None and len(shape) != expected_rank:
            raise RuntimeError(
                f"{op_name}: {name} must be rank {expected_rank}, got shape {shape}."
            )

    query_shape = shapes["query"]
    key_shape = shapes["key"]
    value_shape = shapes["value"]
    state_shape = shapes["state"]
    actual_seq_lengths_shape = shapes["actual_seq_lengths"]
    if query_shape != key_shape:
        raise RuntimeError(
            f"{op_name}: key shape must equal query shape, got query={query_shape}, key={key_shape}."
        )

    total_tokens, key_heads, key_dim = query_shape
    value_tokens, value_heads, value_dim = value_shape
    state_blocks = state_shape[0]
    positive_dims = {
        "T": total_tokens,
        "Nk": key_heads,
        "Nv": value_heads,
        "Dk": key_dim,
        "Dv": value_dim,
        "state blocks": state_blocks,
    }
    for name, size in positive_dims.items():
        if size <= 0:
            raise RuntimeError(f"{op_name}: {name} must be positive, got {size}.")

    if value_tokens != total_tokens:
        raise RuntimeError(
            f"{op_name}: value T dimension must be {total_tokens}, got {value_tokens}."
        )
    expected_beta_shape = (total_tokens, value_heads)
    if shapes["beta"] != expected_beta_shape:
        raise RuntimeError(
            f"{op_name}: beta shape must be {expected_beta_shape}, got {shapes['beta']}."
        )
    expected_state_shape = (state_blocks, value_heads, value_dim, key_dim)
    if state_shape != expected_state_shape:
        raise RuntimeError(
            f"{op_name}: state shape must be {expected_state_shape}, got {state_shape}."
        )
    if actual_seq_lengths_shape[0] < 2:
        raise RuntimeError(
            f"{op_name}: actual_seq_lengths must contain the prefix entry and at least one sequence length."
        )
    if shapes["ssm_state_indices"] != (total_tokens,):
        raise RuntimeError(
            f"{op_name}: ssm_state_indices shape must be ({total_tokens},), "
            f"got {shapes['ssm_state_indices']}."
        )

    batch_size = actual_seq_lengths_shape[0] - 1
    optional_shapes = {
        "g": (total_tokens, value_heads),
        "gk": (total_tokens, value_heads, key_dim),
        "num_accepted_tokens": (batch_size,),
    }
    for name, expected_shape in optional_shapes.items():
        shape = shapes.get(name)
        if shape is not None and shape != expected_shape:
            raise RuntimeError(
                f"{op_name}: {name} shape must be {expected_shape}, got {shape}."
            )

    if key_heads > 256 or value_heads > 256 or key_dim > RGDR_MAX_DIM or value_dim > RGDR_MAX_DIM:
        raise RuntimeError(
            f"{op_name}: Nk and Nv must be <= 256 and Dk and Dv must be <= {RGDR_MAX_DIM}, "
            f"got Nk={key_heads}, Nv={value_heads}, Dk={key_dim}, Dv={value_dim}."
        )
    if key_dim % RGDR_DIM_ALIGNMENT != 0 or value_dim % RGDR_DIM_ALIGNMENT != 0:
        raise RuntimeError(
            f"{op_name}: Dk and Dv must be multiples of {RGDR_DIM_ALIGNMENT} for 32B-aligned BF16 rows, "
            f"got Dk={key_dim}, Dv={value_dim}."
        )
    if value_heads % key_heads != 0:
        raise RuntimeError(
            f"{op_name}: Nv must be an integer multiple of Nk, got Nv={value_heads}, Nk={key_heads}."
        )

    scale = _optional_float(scale, 1.0)
    if not math.isfinite(scale):
        raise ValueError(f"{op_name}: scale must be finite, got {scale}.")

    def nd_tensor(ctx, tensor, name):
        if tensor is None:
            return ctx.tensor(tensor, name)
        storage_shape = _shape(tensor) if tensor.is_contiguous() else None
        return ctx.tensor(
            tensor,
            name,
            acl_format_override=ACL_FORMAT_ND,
            storage_shape_override=storage_shape,
        )

    out = _empty(_shape(value), value)
    return _call_aclnn(
        "aclnnRecurrentGatedDeltaRule",
        lambda ctx: [
            nd_tensor(ctx, query, "query"),
            nd_tensor(ctx, key, "key"),
            nd_tensor(ctx, value, "value"),
            nd_tensor(ctx, beta, "beta"),
            nd_tensor(ctx, state, "state"),
            nd_tensor(ctx, actual_seq_lengths, "actual_seq_lengths"),
            nd_tensor(ctx, ssm_state_indices, "ssm_state_indices"),
            nd_tensor(ctx, g, "g"),
            nd_tensor(ctx, gk, "gk"),
            nd_tensor(ctx, num_accepted_tokens, "num_accepted_tokens"),
            ctypes.c_float(scale),
            nd_tensor(ctx, out, "out"),
        ],
        out,
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
    B, _, T, _ = _shape(k_contig)
    _, Hv, _ = _shape(g_contig)
    out = _empty((B, Hv, T, int(chunk_size)), k_contig, dtype=torch.float32)
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

def npu_recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    *,
    cu_seqlens=None,
    ssm_state_indices=None,
    A_log=None,
    dt_bias=None,
    num_accepted_tokens=None,
    layout="BSND",
    scale=None,
    output_final_state=False,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    state_v_first=False,
):
    import torch

    layout = str(layout)
    if layout not in ("BSND", "TND"):
        raise RuntimeError("npu_recurrent_kda: layout must be BSND or TND.")
    is_tnd = layout == "TND"
    q_shape, k_shape, v_shape = _shape(q), _shape(k), _shape(v)
    g_shape, beta_shape = _shape(g), _shape(beta)
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
        batch, dense_seq_len = 1, total_tokens
        value_heads, value_dim = v_shape[1], v_shape[2]
        value_shape_ok = (
            v_shape[0] == total_tokens
            and g_shape == (total_tokens, value_heads, key_dim)
            and beta_shape == (total_tokens, value_heads)
        )
    else:
        batch, dense_seq_len, heads, key_dim = q_shape
        total_tokens = batch * dense_seq_len
        value_heads, value_dim = v_shape[2], v_shape[3]
        value_shape_ok = (
            v_shape[:2] == (batch, dense_seq_len)
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

    if cu_seqlens is None:
        seq_num = batch if not is_tnd else 1
    else:
        if (
            not isinstance(cu_seqlens, torch.Tensor)
            or cu_seqlens.dim() != 1
            or cu_seqlens.numel() < 2
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise RuntimeError("npu_recurrent_kda: cu_seqlens must be a 1D INT32 or INT64 tensor.")
        if cu_seqlens.device != q.device:
            raise RuntimeError("npu_recurrent_kda: cu_seqlens must be on the same device as q.")
        seq_num = int(cu_seqlens.shape[0]) - 1

    state_v_first = _optional_bool(state_v_first, False)
    expected_tail = (
        (value_heads, value_dim, key_dim)
        if state_v_first
        else (value_heads, key_dim, value_dim)
    )
    inplace = _optional_bool(inplace_final_state, True)
    if initial_state is None:
        if inplace:
            raise RuntimeError("npu_recurrent_kda: inplace_final_state=True requires initial_state.")
        state_shape = (seq_num, *expected_tail)
        initial_state_work = _zeros(state_shape, q, dtype=torch.float32)
    else:
        if initial_state.dtype not in (torch.float32, torch.bfloat16):
            raise RuntimeError("npu_recurrent_kda: initial_state must use FP32 or BF16.")
        if initial_state.device != q.device:
            raise RuntimeError("npu_recurrent_kda: initial_state must be on the same device as q.")
        state_shape = _shape(initial_state)
        if len(state_shape) != 4 or state_shape[0] <= 0 or state_shape[1:] != expected_tail:
            layout_desc = "[state_capacity,HV,V,K]" if state_v_first else "[state_capacity,HV,K,V]"
            raise RuntimeError(f"npu_recurrent_kda: initial_state must be {layout_desc}.")
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
        raise RuntimeError("npu_recurrent_kda: without ssm_state_indices, state_capacity must equal seq_num.")
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
    elif safe or A_log is not None or dt_bias is not None:
        raise RuntimeError("npu_recurrent_kda: A_log, dt_bias and safe_gate require use_gate_in_kernel=True.")

    out = _empty_like(v)
    final_state_work = initial_state_work if inplace else _empty(state_shape, initial_state_work)
    final_state_arg = final_state_work
    output_final = _optional_bool(output_final_state, False)
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
            ctypes.c_bool(output_final),
            ctypes.c_bool(inplace),
            ctypes.c_bool(_optional_bool(use_qk_l2norm_in_kernel, False)),
            ctypes.c_bool(use_gate),
            ctypes.c_bool(_optional_bool(use_beta_sigmoid_in_kernel, False)),
            ctypes.c_bool(_optional_bool(allow_neg_eigval, False)),
            ctypes.c_bool(safe),
            ctypes.c_double(lower),
            ctypes.c_bool(state_v_first),
            ctx.tensor(out, "attn_out"),
            ctx.tensor(final_state_arg, "final_state"),
        ]

    _call_aclnn(
        "aclnnRecurrentKda",
        build_args,
        (out, initial_state_work, final_state_arg),
    )
    final_state = final_state_work if output_final else None
    return out, final_state

# Dense BSND Host code transposes all ten inputs to BNSD. Its four internal
# outputs alias the transposed gradient inputs, so bounding this input footprint
# bounds the dominant per-call workspace without changing kernel/tiling.
_KDA_BSND_TRANSPOSE_WORKSPACE_BUDGET_BYTES = 960 * 1024 * 1024


def _chunk_kda_bwd_intra_bsnd_segment_tokens(
    tensors,
    seqlen,
    chunk_size,
    *,
    workspace_budget_bytes=None,
):
    """Return a chunk-aligned BSND segment length for bounded transposes."""

    if workspace_budget_bytes is None:
        workspace_budget_bytes = _KDA_BSND_TRANSPOSE_WORKSPACE_BUDGET_BYTES
    total_transpose_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors
    )
    if total_transpose_bytes <= workspace_budget_bytes:
        return int(seqlen)

    bytes_per_token = total_transpose_bytes // int(seqlen)
    budget_tokens = int(workspace_budget_bytes) // bytes_per_token
    aligned_tokens = (budget_tokens // int(chunk_size)) * int(chunk_size)
    return min(int(seqlen), max(int(chunk_size), aligned_tokens))


def npu_chunk_kda_bwd_intra(
    q,
    k,
    gk,
    beta,
    dAqk,
    dAkk,
    dq,
    dk,
    db,
    dg,
    *,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=64,
    safe_gate=True,
    layout="BSND",
):
    """Run the safe-gate KDA intra-chunk backward kernel.

    BNSD is the native performance layout. BSND is converted through the same
    layout-swap operator used by the existing KDA forward path for dense input.
    Varlen uses zero-copy TND [T,H,D], with BSND [1,T,H,D] accepted as a
    storage-compatible form. q/k must be BF16; beta accepts BF16 or FP32.
    """
    import torch

    tensors = {
        "q": q,
        "k": k,
        "gk": gk,
        "beta": beta,
        "dAqk": dAqk,
        "dAkk": dAkk,
        "dq": dq,
        "dk": dk,
        "db": db,
        "dg": dg,
    }
    layout = str(layout)
    if layout not in {"BSND", "BNSD", "TND"}:
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: supports dense BSND/BNSD or varlen TND."
        )
    if not bool(safe_gate):
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: safe_gate=False is reserved but not supported in v1."
        )
    chunk_size = int(chunk_size)
    if chunk_size != 64:
        raise RuntimeError("npu_chunk_kda_bwd_intra: chunk_size must be 64.")
    expected_dtypes = {
        "q": torch.bfloat16,
        "k": torch.bfloat16,
        "gk": torch.float32,
        "dAqk": torch.float32,
        "dAkk": torch.float32,
        "dq": torch.float32,
        "dk": torch.float32,
        "db": torch.float32,
        "dg": torch.float32,
    }
    for name, tensor in tensors.items():
        if name == "beta":
            if tensor.dtype not in {torch.bfloat16, torch.float32}:
                raise RuntimeError(
                    "npu_chunk_kda_bwd_intra: beta must be torch.bfloat16 "
                    "or torch.float32."
                )
        elif tensor.dtype != expected_dtypes[name]:
            raise RuntimeError(
                f"npu_chunk_kda_bwd_intra: {name} must be {expected_dtypes[name]}."
            )
        if tensor.device != q.device:
            raise RuntimeError(
                f"npu_chunk_kda_bwd_intra: {name} must be on the same device as q."
            )
        if not tensor.is_contiguous():
            raise RuntimeError(
                f"npu_chunk_kda_bwd_intra: {name} must be contiguous; implicit copies are disabled."
            )
    if chunk_indices is not None and cu_seqlens is None:
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: chunk_indices requires cu_seqlens."
        )
    is_varlen = cu_seqlens is not None
    if is_varlen and layout == "BNSD":
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: varlen supports TND or BSND, not BNSD."
        )
    if not is_varlen and layout == "TND":
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: TND requires cu_seqlens."
        )

    q_shape = _shape(q)
    expected_rank = 3 if layout == "TND" else 4
    if len(q_shape) != expected_rank:
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: q rank does not match layout."
        )
    if layout == "TND":
        seqlen, heads, head_dim = q_shape
        batch = 1
        scalar_shape = (seqlen, heads)
        matrix_shape = (seqlen, heads, chunk_size)
    elif layout == "BSND":
        batch, seqlen, heads, head_dim = q_shape
        scalar_shape = (batch, seqlen, heads)
        matrix_shape = (batch, seqlen, heads, chunk_size)
    else:
        batch, heads, seqlen, head_dim = q_shape
        scalar_shape = (batch, heads, seqlen)
        matrix_shape = (batch, heads, seqlen, chunk_size)
    if batch <= 0 or heads <= 0 or seqlen <= 0:
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: B/H/T must be positive."
        )
    if (is_varlen and head_dim != 128) or (
        not is_varlen and head_dim not in {64, 128, 256}
    ):
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: varlen supports K=128; "
            "dense supports K=64, 128 or 256."
        )
    if is_varlen and layout == "BSND" and batch != 1:
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: varlen BSND compatibility requires B=1."
        )
    for name in ("k", "gk", "dq", "dk", "dg"):
        if _shape(tensors[name]) != q_shape:
            raise RuntimeError(
                f"npu_chunk_kda_bwd_intra: {name} must have the same shape as q."
            )
    if _shape(beta) != scalar_shape or _shape(db) != scalar_shape:
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: beta/db shape must match the selected layout."
        )
    if _shape(dAqk) != matrix_shape or _shape(dAkk) != matrix_shape:
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: dAqk/dAkk shape must match the selected layout."
        )

    cu_seqlens_arg = None
    chunk_indices_arg = None
    if is_varlen:
        cu_seqlens_arg = tuple(int(value) for value in cu_seqlens)
        if not 2 <= len(cu_seqlens_arg) <= 65:
            raise RuntimeError(
                "npu_chunk_kda_bwd_intra: cu_seqlens must contain 2..65 entries."
            )
        if cu_seqlens_arg[0] != 0 or cu_seqlens_arg[-1] != seqlen:
            raise RuntimeError(
                "npu_chunk_kda_bwd_intra: cu_seqlens must start at 0 and end at T."
            )
        canonical_chunks = []
        for seq, (begin, end) in enumerate(
            zip(cu_seqlens_arg[:-1], cu_seqlens_arg[1:])
        ):
            if begin < 0 or end < begin:
                raise RuntimeError(
                    "npu_chunk_kda_bwd_intra: cu_seqlens must be nondecreasing."
                )
            for local_chunk in range((end - begin + chunk_size - 1) // chunk_size):
                canonical_chunks.extend((seq, local_chunk))
        if not canonical_chunks:
            raise RuntimeError(
                "npu_chunk_kda_bwd_intra: varlen input has no non-empty sequence."
            )
        if chunk_indices is not None:
            chunk_indices_arg = tuple(int(value) for value in chunk_indices)
            if chunk_indices_arg != tuple(canonical_chunks):
                raise RuntimeError(
                    "npu_chunk_kda_bwd_intra: chunk_indices must use canonical "
                    "sequence-major order."
                )

    dq_out = _empty_like(dq)
    dk_out = _empty_like(dk)
    db_out = _empty_like(db)
    dg_out = _empty_like(dg)
    outputs = (dq_out, dk_out, db_out, dg_out)
    layout_buffer = ctypes.create_string_buffer(layout.encode("utf-8"))

    # The custom op consumes dense BSND/BNSD as an ND tensor. Standard contiguous
    # rank-4 NPU tensors can carry an NCHW tag despite row-major storage, so
    # override descriptor metadata without a format conversion or data copy.
    def nd_tensor(ctx, tensor, name):
        return ctx.tensor(
            tensor,
            name,
            acl_format_override=ACL_FORMAT_ND,
            storage_shape_override=_shape(tensor),
        )

    input_tensors = (
        q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg
    )
    input_names = (
        "q", "k", "gk", "beta", "dAqk",
        "dAkk", "dq", "dk", "db", "dg",
    )
    output_names = ("dq_out", "dk_out", "db_out", "dg_out")

    def launch(call_inputs, call_outputs):
        def build_args(ctx):
            return [
                *(
                    nd_tensor(ctx, tensor, name)
                    for tensor, name in zip(call_inputs, input_names)
                ),
                ctx.int_array(cu_seqlens_arg),
                ctx.int_array(chunk_indices_arg),
                ctypes.c_int64(chunk_size),
                ctypes.c_bool(True),
                ctypes.cast(layout_buffer, ctypes.c_char_p),
                *(
                    nd_tensor(ctx, tensor, name)
                    for tensor, name in zip(call_outputs, output_names)
                ),
            ]

        return _call_aclnn(
            "aclnnChunkKdaBwdIntra",
            build_args,
            call_outputs,
        )

    if not is_varlen and layout == "BSND" and batch == 1:
        segment_tokens = _chunk_kda_bwd_intra_bsnd_segment_tokens(
            input_tensors,
            seqlen,
            chunk_size,
        )
        if segment_tokens < seqlen:
            # Intra-chunk math has no dependency across chunk boundaries. B=1
            # makes every token slice physically contiguous; each launch writes
            # directly into disjoint views of the final full-size outputs.
            for begin in range(0, seqlen, segment_tokens):
                length = min(segment_tokens, seqlen - begin)
                segment_inputs = tuple(
                    tensor.narrow(1, begin, length) for tensor in input_tensors
                )
                segment_outputs = tuple(
                    tensor.narrow(1, begin, length) for tensor in outputs
                )
                launch(segment_inputs, segment_outputs)
            return outputs

    return launch(input_tensors, outputs)


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
