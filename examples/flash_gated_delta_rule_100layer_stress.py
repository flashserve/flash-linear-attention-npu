#!/usr/bin/env python3
"""Pure-asynchronous 100-layer GDN reproduction workload.

The workload intentionally performs no host/device synchronization while steps
are being submitted. All outputs and input gradients remain on the NPU until
every requested step has finished launching. Parameter gradient norms use the
same global L2 definition as ``clip_grad_norm_`` and are clipped on device.
Finite checks are queued only at the end and then copied to the host once.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
from typing import Callable, Optional

import torch
import torch_npu
from torch.utils.checkpoint import checkpoint

from fla_npu.ops.ascendc import (  # noqa: E402
    npu_causal_conv1d as ascendc_causal_conv1d,
    npu_causal_conv1d_bwd as ascendc_causal_conv1d_bwd,
    npu_chunk_bwd_dqkwg as ascendc_chunk_bwd_dqkwg,
    npu_chunk_bwd_dv_local as ascendc_chunk_bwd_dv_local,
    npu_chunk_fwd_o as ascendc_chunk_fwd_o,
    npu_chunk_gated_delta_rule_bwd_dhu as ascendc_chunk_gated_delta_rule_bwd_dhu,
    npu_chunk_gated_delta_rule_fwd_h as ascendc_chunk_gated_delta_rule_fwd_h,
    npu_prepare_wy_repr_bwd_da as ascendc_prepare_wy_repr_bwd_da,
    npu_prepare_wy_repr_bwd_full as ascendc_prepare_wy_repr_bwd_full,
    npu_recompute_w_u_fwd as ascendc_recompute_w_u_fwd,
    npu_solve_tri as ascendc_solve_tri,
)
from fla_npu.ops.triton import (  # noqa: E402
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    l2norm_bwd,
    l2norm_fwd,
)


_TRACE_APPLY = False
_STAGE_GRAD_RECORDS: Optional[
    dict[str, list[Optional[torch.Tensor]]]
] = None
_DEFAULT_VARLEN_CHUNK_SIZES = (16, 32, 64, 128, 608 * 2)


DEFAULT_CU_SEQLENS = [
    0,
    796,
    1560,
    2262,
    2914,
    3535,
    4137,
    4734,
    5319,
    5893,
    6415,
    6925,
    7422,
    7898,
    8358,
    8802,
    9234,
    9656,
    10073,
    10488,
    10878,
    11226,
    11550,
    11860,
    12162,
    12462,
    12758,
    13047,
    13333,
    13613,
    13893,
    14173,
    14451,
    14728,
    15004,
    15279,
    15551,
    15822,
    16089,
    16354,
    16616,
    16876,
    17135,
    17394,
    17647,
    17899,
    18151,
    18401,
    18650,
    18896,
    19138,
    19376,
    19611,
    19842,
    20072,
    20302,
    20530,
    20756,
    20981,
    21204,
    21419,
    21633,
    21844,
    22041,
    32768,
]


def parse_cu_seqlens(value: str, tokens: int) -> list[int]:
    offsets = DEFAULT_CU_SEQLENS if not value.strip() else [
        int(item.strip()) for item in value.split(",") if item.strip()
    ]
    if len(offsets) < 2:
        raise ValueError("cu_seqlens must contain at least two offsets")
    if offsets[0] != 0 or offsets[-1] != tokens:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at --tokens={tokens}, "
            f"got ({offsets[0]}, {offsets[-1]})"
        )
    if any(right <= left for left, right in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be strictly increasing")
    return offsets


def build_chunk_indices(offsets: list[int], chunk_size: int) -> tuple[list[list[int]], list[int]]:
    pairs: list[list[int]] = []
    flattened: list[int] = []
    for sequence_index, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        chunk_count = (end - begin + chunk_size - 1) // chunk_size
        for chunk_index in range(chunk_count):
            pairs.append([sequence_index, chunk_index])
            flattened.extend([sequence_index, chunk_index])
    return pairs, flattened


def prepare_varlen_metadata(
    offsets: list[int],
    chunk_size: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, list[int]],
]:
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)
    tensor_indices: dict[str, torch.Tensor] = {}
    list_indices: dict[str, list[int]] = {}
    required_sizes = set(_DEFAULT_VARLEN_CHUNK_SIZES)
    required_sizes.add(chunk_size)
    for size in sorted(required_sizes):
        pairs, flattened = build_chunk_indices(offsets, size)
        tensor_indices[str(size)] = torch.tensor(pairs, dtype=torch.int64, device=device)
        list_indices[str(size)] = flattened
    return cu_seqlens, tensor_indices, list_indices


def install_stage_tracing() -> None:
    global _TRACE_APPLY

    _TRACE_APPLY = True
    names = (
        "l2norm_fwd",
        "l2norm_bwd",
        "chunk_local_cumsum",
        "chunk_scaled_dot_kkt_fwd",
        "ascendc_causal_conv1d",
        "ascendc_causal_conv1d_bwd",
        "solve_tri_ascendc",
        "ascendc_solve_tri",
        "recompute_w_u",
        "ascendc_recompute_w_u_fwd",
        "gated_delta_rule_fwd",
        "gated_delta_rule_bwd",
        "ascendc_chunk_gated_delta_rule_fwd_h",
        "ascendc_chunk_fwd_o",
        "ascendc_chunk_bwd_dv_local",
        "ascendc_chunk_gated_delta_rule_bwd_dhu",
        "ascendc_chunk_bwd_dqkwg",
        "ascendc_prepare_wy_repr_bwd_da",
        "ascendc_prepare_wy_repr_bwd_full",
    )
    namespace = globals()
    for name in names:
        function = namespace[name]

        @functools.wraps(function)
        def traced(*args, __function=function, __name=name, **kwargs):
            print(f"stage enter: {__name}", flush=True)
            result = __function(*args, **kwargs)
            print(f"stage return: {__name}", flush=True)
            return result

        namespace[name] = traced


def trace_apply(message: str) -> None:
    if _TRACE_APPLY:
        print(message, flush=True)


def record_stage_grad(name: str, tensor: Optional[torch.Tensor]) -> None:
    if _STAGE_GRAD_RECORDS is not None:
        record = None if tensor is None else tensor.detach().clone()
        _STAGE_GRAD_RECORDS.setdefault(name, []).append(record)


def _chunk_tensor(
    chunk_indices: Optional[dict[str, torch.Tensor]],
    chunk_size: int,
) -> Optional[torch.Tensor]:
    if chunk_indices is None:
        return None
    return chunk_indices.get(str(chunk_size))


def _chunk_list(
    chunk_indices_list: Optional[dict[str, list[int]]],
    chunk_size: int,
) -> Optional[list[int]]:
    if chunk_indices_list is None:
        return None
    return chunk_indices_list.get(str(chunk_size))


def solve_tri_ascendc(
    A: torch.Tensor,
    *,
    cu_seqlens: Optional[list[int]],
    chunk_indices: Optional[list[int]],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    A_in = A.to(output_dtype).contiguous()
    if cu_seqlens is None:
        return ascendc_solve_tri(A_in, layout="bsnd")
    if chunk_indices is None:
        raise ValueError("solve_tri varlen path requires chunk_indices.")
    return ascendc_solve_tri(
        A_in.squeeze(0),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        layout="tnd",
    ).unsqueeze(0)


def recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: Optional[list[int]],
    chunk_indices: Optional[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    return ascendc_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )

    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )

    # A is FP32 here. The adapter casts the solve_tri input to k.dtype, while
    # the 64x64 kernel keeps its internal MCH/MXR calculations in FP32 and
    # returns k.dtype.
    A = solve_tri_ascendc(
        A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        output_dtype=k.dtype,
    )

    g_head = g.transpose(1, 2).contiguous()
    beta_head = beta.transpose(1, 2).contiguous().float()
    A_head = A.transpose(1, 2).contiguous()

    w, u = recompute_w_u(
        k,
        v,
        beta_head,
        A_head,
        g_head,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    h, v_new, _final_state = ascendc_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g_head,
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )

    o = ascendc_chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g_head,
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )

    return g_head.transpose(1, 2).contiguous(), o.transpose(1, 2).contiguous(), A_head


def gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    do: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g_head = g.transpose(1, 2).contiguous()
    beta_head = beta.transpose(1, 2).contiguous().float()

    w, u = recompute_w_u(
        k,
        v,
        beta_head,
        A,
        g_head,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    do_head = do.transpose(1, 2).contiguous()

    h, v_new, _ = ascendc_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g_head,
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )

    dv = ascendc_chunk_bwd_dv_local(
        q,
        k,
        do_head,
        g_head,
        scale,
        chunk_size,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    diagnostic_h0 = (
        torch.zeros_like(h) if _STAGE_GRAD_RECORDS is not None else None
    )
    dh, dh0, dv = ascendc_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do_head,
        dv,
        scale,
        chunk_size,
        g=g_head,
        gK=None,
        h0=diagnostic_h0,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )

    dq, dk, dw, dg = ascendc_chunk_bwd_dqkwg(
        q,
        k,
        v_new,
        g_head,
        h,
        do_head,
        dh,
        dv,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )

    dA = ascendc_prepare_wy_repr_bwd_da(
        k,
        v,
        beta_head.float(),
        A,
        dw,
        dv,
        g_head.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dk2, dv, db, dg2 = ascendc_prepare_wy_repr_bwd_full(
        k,
        v,
        beta_head,
        A,
        dA,
        dw,
        dv,
        g_head,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()

    dk.add_(dk2)
    dg.add_(dg2)

    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )

    record_stage_grad("gdr_dq", dq)
    record_stage_grad("gdr_dk", dk)
    record_stage_grad("gdr_dv", dv)
    record_stage_grad("gdr_db", db)
    record_stage_grad("gdr_dg", dg)
    record_stage_grad("gdr_dh", dh)
    record_stage_grad("gdr_dh0", dh0)
    return dq, dk, dv, db, dg


class StressCausalConv1dFunction(torch.autograd.Function):
    """Reproduction-only causal conv node using NTD output layout."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        head_num: int,
        query_start_loc: list[int],
        activation_mode: int,
    ) -> torch.Tensor:
        trace_apply("stage enter: StressCausalConv1dFunction.forward")
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"stress causal conv expects [1, T, D] input, got {tuple(x.shape)}")
        if head_num <= 0:
            raise ValueError(f"stress causal conv requires NTD output head_num > 0, got {head_num}")

        op_weight = weight.transpose(-1, -2).contiguous()
        width, dim = op_weight.shape
        op_x = x.reshape(x.shape[1], x.shape[2]).contiguous()
        conv_states = torch.zeros(
            len(query_start_loc) - 1,
            width - 1,
            dim,
            dtype=x.dtype,
            device=x.device,
        )
        preactivation = ascendc_causal_conv1d(
            op_x,
            op_weight,
            None,
            conv_states,
            query_start_loc=query_start_loc,
            initial_state_mode=None,
            activation_mode=0,
            pad_slot_id=-1,
            run_mode=0,
            head_num=head_num,
        )
        y = torch.nn.functional.silu(preactivation) if activation_mode != 0 else preactivation

        if _STAGE_GRAD_RECORDS is None:
            ctx.save_for_backward(x, op_weight, preactivation)
        else:
            bwd_initial_state = torch.zeros(
                len(query_start_loc) - 1,
                width,
                dim,
                dtype=x.dtype,
                device=x.device,
            )
            ctx.save_for_backward(
                x,
                op_weight,
                preactivation,
                bwd_initial_state,
            )
        ctx.query_start_loc = query_start_loc
        ctx.activation_mode = activation_mode
        trace_apply("stage return: StressCausalConv1dFunction.forward")
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        trace_apply("stage enter: StressCausalConv1dFunction.backward")
        x, op_weight, preactivation, *optional_state = ctx.saved_tensors
        bwd_initial_state = optional_state[0] if optional_state else None
        op_x = x.reshape(x.shape[1], x.shape[2]).contiguous()
        dx, dw, db, dh0 = ascendc_causal_conv1d_bwd(
            x=op_x,
            y=preactivation if ctx.activation_mode != 0 else None,
            weight=op_weight,
            dy=dy.contiguous(),
            initial_state=bwd_initial_state,
            dht=None,
            query_start_loc=ctx.query_start_loc,
            activation=ctx.activation_mode,
            input_layout="NTD",
        )
        record_stage_grad("causal_conv_dx", dx)
        record_stage_grad("causal_conv_dw", dw)
        record_stage_grad("causal_conv_db", db)
        record_stage_grad("causal_conv_dh0", dh0)
        trace_apply("stage return: StressCausalConv1dFunction.backward")
        return dx.reshape_as(x), dw.transpose(0, 1).contiguous(), None, None, None


class StressGatedDeltaRuleFunction(torch.autograd.Function):
    """Reproduction-only GDR node exposing one apply boundary."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        cu_seqlens: torch.Tensor,
        cu_seqlens_list: list[int],
        chunk_indices: dict[str, torch.Tensor],
        chunk_indices_list: dict[str, list[int]],
        chunk_size: int,
        use_qk_l2norm_in_kernel: bool,
    ) -> torch.Tensor:
        trace_apply("stage enter: StressGatedDeltaRuleFunction.forward")
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A = gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            cu_seqlens,
            cu_seqlens_list,
            chunk_indices,
            chunk_indices_list,
            chunk_size,
        )

        ctx.save_for_backward(q, k, v, g, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        ctx.chunk_size = chunk_size
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        trace_apply("stage return: StressGatedDeltaRuleFunction.forward")
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        trace_apply("stage enter: StressGatedDeltaRuleFunction.backward")
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, db, dg = gated_delta_rule_bwd(
            q,
            k,
            v,
            g,
            beta,
            A,
            ctx.scale,
            do,
            ctx.cu_seqlens,
            ctx.cu_seqlens_list,
            ctx.chunk_indices,
            ctx.chunk_indices_list,
            ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)
        record_stage_grad("gdr_wrapper_dq", dq)
        record_stage_grad("gdr_wrapper_dk", dk)
        trace_apply("stage return: StressGatedDeltaRuleFunction.backward")
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def make_layer(
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    beta_weight: torch.Tensor,
    beta_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    chunk_indices: dict[str, torch.Tensor],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
    residual_scale: float,
    norm_eps: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def layer(state: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, dim = state.shape
        hidden = state.transpose(1, 2).contiguous().view(
            batch,
            tokens,
            heads * dim,
        )

        # Head-wise diagonal projections make every layer numerically distinct
        # without introducing the cost of a full 2048x2048 model projection.
        mixed_qkv = torch.cat(
            (
                hidden.mul(q_weight),
                hidden.mul(k_weight),
                hidden.mul(v_weight),
            ),
            dim=-1,
        )
        mixed_qkv = StressCausalConv1dFunction.apply(
            mixed_qkv,
            conv_weight,
            3 * heads,
            cu_seqlens_list,
            1,
        )
        q = mixed_qkv[:heads].unsqueeze(0).contiguous()
        k = mixed_qkv[heads : 2 * heads].unsqueeze(0).contiguous()
        v = mixed_qkv[2 * heads :].unsqueeze(0).contiguous()
        gate_input = state.float().mean(dim=-1).transpose(1, 2)
        beta = torch.sigmoid(gate_input.mul(beta_weight).add(beta_bias)).to(state.dtype)
        gate = torch.nn.functional.logsigmoid(
            gate_input.mul(gate_weight).add(gate_bias)
        ).to(state.dtype)
        beta = beta.contiguous()
        gate = gate.contiguous()
        output = StressGatedDeltaRuleFunction.apply(
            q,
            k,
            v,
            gate,
            beta,
            dim ** -0.5,
            cu_seqlens,
            cu_seqlens_list,
            chunk_indices,
            chunk_indices_list,
            chunk_size,
            True,
        )
        output = output.transpose(1, 2).contiguous()
        residual = torch.add(state, output, alpha=residual_scale)
        hidden = residual.transpose(1, 2).contiguous().view(
            batch,
            tokens,
            heads * dim,
        )
        hidden = torch_npu.npu_rms_norm(hidden, norm_weight, norm_eps)[0]
        return hidden.view(batch, tokens, heads, dim).transpose(1, 2).contiguous()

    return layer


def tensor_md5(tensor: torch.Tensor) -> str:
    host = tensor.float().cpu().contiguous()
    return hashlib.md5(host.numpy().tobytes()).hexdigest()


def tensor_binary_md5(tensor: torch.Tensor) -> str:
    host = tensor.cpu().contiguous().view(torch.uint8)
    return hashlib.md5(host.numpy().tobytes()).hexdigest()


def global_grad_norm_and_clip_(
    parameters: list[torch.nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Return the pre-clip global FP32 L2 norm and clip gradients in place."""
    grads = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grads:
        raise RuntimeError("no parameter gradients were produced")

    per_parameter_norms = torch.stack(
        [torch.linalg.vector_norm(grad.float(), 2) for grad in grads]
    )
    total_norm = torch.linalg.vector_norm(per_parameter_norms, 2)
    clip_coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for grad in grads:
        grad.mul_(clip_coefficient.to(dtype=grad.dtype))
    return total_norm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--conv-width", type=int, default=4)
    parser.add_argument("--layers", type=int, default=100)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Clip parameter gradients to this global L2 norm after each backward.",
    )
    parser.add_argument(
        "--cu-seqlens",
        default="",
        help="Comma-separated offsets. Empty uses the exact 32768-token reproduction offsets.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable activation checkpointing; 100 layers may require substantially more HBM.",
    )
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument(
        "--md5",
        action="store_true",
        help="Copy full retained outputs/gradients after all steps and print MD5 values.",
    )
    parser.add_argument(
        "--trace-stages",
        action="store_true",
        help="Print Python call boundaries without reading NPU tensors or synchronizing.",
    )
    parser.add_argument(
        "--replay-step-inputs",
        action="store_true",
        help=(
            "Reuse identical input and upstream tensors for every step and compare "
            "outputs plus every pre-clip parameter gradient for exact determinism."
        ),
    )
    parser.add_argument(
        "--train-norm-weight",
        action="store_true",
        help=(
            "Include RMSNorm weights in global grad norm. Disabled by default because "
            "npu_rms_norm weight gradients are nondeterministic on the target stack."
        ),
    )
    parser.add_argument(
        "--check-stage-grad-binary",
        action="store_true",
        help=(
            "For a one-layer replay, retain direct backward stage outputs and compare "
            "their original-dtype bytes after all steps finish."
        ),
    )
    args = parser.parse_args()

    for name in (
        "tokens",
        "heads",
        "dim",
        "chunk_size",
        "conv_width",
        "layers",
        "steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive")
    if args.check_stage_grad_binary:
        if args.forward_only:
            raise ValueError("--check-stage-grad-binary requires backward")
        if not args.replay_step_inputs:
            raise ValueError(
                "--check-stage-grad-binary requires --replay-step-inputs"
            )
        if args.layers != 1:
            raise ValueError("--check-stage-grad-binary requires --layers=1")
        if args.steps < 2:
            raise ValueError("--check-stage-grad-binary requires --steps >= 2")

    offsets = parse_cu_seqlens(args.cu_seqlens, args.tokens)
    device = torch.device(f"npu:{args.device}")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    try:
        package_version = importlib.metadata.version("flash-linear-attention-npu")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source-tree"
    try:
        triton_ascend_version = importlib.metadata.version("triton-ascend")
    except importlib.metadata.PackageNotFoundError:
        triton_ascend_version = "unknown"

    cu_seqlens, chunk_indices, chunk_indices_list = prepare_varlen_metadata(
        offsets,
        args.chunk_size,
        device,
    )
    if args.trace_stages:
        install_stage_tracing()
    if args.check_stage_grad_binary:
        global _STAGE_GRAD_RECORDS
        _STAGE_GRAD_RECORDS = {}

    layers: list[Callable[[torch.Tensor], torch.Tensor]] = []
    parameters: list[torch.nn.Parameter] = []
    parameter_names: list[str] = []

    def make_parameter(
        name: str,
        shape: tuple[int, ...],
        parameter_dtype: torch.dtype,
        scale: float,
        bias: float = 0.0,
        trainable: bool = True,
    ) -> torch.nn.Parameter:
        tensor = torch.randn(
            shape,
            dtype=parameter_dtype,
            device=device,
        ).mul_(scale)
        if bias:
            tensor.add_(bias)
        parameter = torch.nn.Parameter(
            tensor,
            requires_grad=not args.forward_only and trainable,
        )
        if parameter.requires_grad:
            parameters.append(parameter)
            parameter_names.append(name)
        return parameter

    for layer_index in range(args.layers):
        hidden_size = args.heads * args.dim
        projection_shape = (1, 1, hidden_size)
        gate_shape = (1, 1, args.heads)
        layers.append(
            make_layer(
                q_weight=make_parameter(
                    f"layers.{layer_index}.q_weight",
                    projection_shape,
                    dtype,
                    0.1,
                    1.0,
                ),
                k_weight=make_parameter(
                    f"layers.{layer_index}.k_weight",
                    projection_shape,
                    dtype,
                    0.1,
                    1.0,
                ),
                v_weight=make_parameter(
                    f"layers.{layer_index}.v_weight",
                    projection_shape,
                    dtype,
                    0.1,
                    0.5,
                ),
                conv_weight=make_parameter(
                    f"layers.{layer_index}.conv_weight",
                    (3 * hidden_size, args.conv_width),
                    dtype,
                    0.1,
                ),
                beta_weight=make_parameter(
                    f"layers.{layer_index}.beta_weight",
                    gate_shape,
                    torch.float32,
                    0.1,
                    1.0,
                ),
                beta_bias=make_parameter(
                    f"layers.{layer_index}.beta_bias",
                    gate_shape,
                    torch.float32,
                    0.1,
                ),
                gate_weight=make_parameter(
                    f"layers.{layer_index}.gate_weight",
                    gate_shape,
                    torch.float32,
                    0.1,
                    1.0,
                ),
                gate_bias=make_parameter(
                    f"layers.{layer_index}.gate_bias",
                    gate_shape,
                    torch.float32,
                    0.1,
                ),
                norm_weight=make_parameter(
                    f"layers.{layer_index}.norm_weight",
                    (args.heads * args.dim,),
                    dtype,
                    0.01,
                    1.0,
                    trainable=args.train_norm_weight,
                ),
                cu_seqlens=cu_seqlens,
                cu_seqlens_list=offsets,
                chunk_indices=chunk_indices,
                chunk_indices_list=chunk_indices_list,
                chunk_size=args.chunk_size,
                residual_scale=args.residual_scale,
                norm_eps=args.norm_eps,
            )
        )

    retained_outputs: list[torch.Tensor] = []
    retained_grads: list[torch.Tensor] = []
    retained_grad_norms: list[torch.Tensor] = []
    reference_parameter_grads: Optional[list[torch.Tensor]] = None
    parameter_grad_equal_checks: list[torch.Tensor] = []
    parameter_grad_max_diffs: list[torch.Tensor] = []
    trainable_parameter_count = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )

    print(
        "config:",
        f"fla_npu={package_version}",
        f"triton_ascend={triton_ascend_version}",
        f"device={device}",
        f"dtype={args.dtype}",
        f"tokens={args.tokens}",
        f"heads={args.heads}",
        f"dim={args.dim}",
        f"chunk_size={args.chunk_size}",
        f"conv_width={args.conv_width}",
        f"sequences={len(offsets) - 1}",
        f"layers={args.layers}",
        f"steps={args.steps}",
        f"checkpoint={not args.no_checkpoint and not args.forward_only}",
        f"backward={not args.forward_only}",
        f"trainable_parameters={trainable_parameter_count}",
        f"max_grad_norm={args.max_grad_norm}",
        f"replay_step_inputs={args.replay_step_inputs}",
        f"train_norm_weight={args.train_norm_weight}",
        f"check_stage_grad_binary={args.check_stage_grad_binary}",
        "causal_conv_output_layout=NTD",
        "apply_wrappers=causal_conv1d,gated_delta_rule",
        flush=True,
    )

    replay_input: Optional[torch.Tensor] = None
    replay_upstream: Optional[torch.Tensor] = None
    if args.replay_step_inputs:
        replay_input = torch.randn(
            1,
            args.heads,
            args.tokens,
            args.dim,
            dtype=dtype,
            device=device,
        )

    for step_index in range(args.steps):
        if replay_input is None:
            input_state = torch.randn(
                1,
                args.heads,
                args.tokens,
                args.dim,
                dtype=dtype,
                device=device,
                requires_grad=not args.forward_only,
            )
        else:
            input_state = replay_input.clone().requires_grad_(not args.forward_only)
        state = input_state
        for layer in layers:
            if args.no_checkpoint or args.forward_only:
                state = layer(state)
            else:
                state = checkpoint(
                    layer,
                    state,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )

        if not args.forward_only:
            if replay_upstream is None:
                upstream = torch.randn_like(state)
                if args.replay_step_inputs:
                    replay_upstream = upstream
            else:
                upstream = replay_upstream
            state.backward(upstream)
            if input_state.grad is None:
                raise RuntimeError("input gradient was not produced")
            retained_grads.append(input_state.grad.detach())
            if args.replay_step_inputs:
                current_parameter_grads = [
                    parameter.grad.detach()
                    for parameter in parameters
                    if parameter.grad is not None
                ]
                if len(current_parameter_grads) != len(parameters):
                    raise RuntimeError(
                        "determinism check requires every parameter to produce a gradient"
                    )
                if reference_parameter_grads is None:
                    reference_parameter_grads = [
                        grad.clone() for grad in current_parameter_grads
                    ]
                else:
                    parameter_grad_equal_checks.append(
                        torch.stack(
                            [
                                torch.eq(grad, reference).all()
                                for grad, reference in zip(
                                    current_parameter_grads,
                                    reference_parameter_grads,
                                )
                            ]
                        )
                    )
                    parameter_grad_max_diffs.append(
                        torch.stack(
                            [
                                (grad.float() - reference.float()).abs().max()
                                for grad, reference in zip(
                                    current_parameter_grads,
                                    reference_parameter_grads,
                                )
                            ]
                        )
                    )
            retained_grad_norms.append(
                global_grad_norm_and_clip_(parameters, args.max_grad_norm).detach()
            )
            for parameter in parameters:
                parameter.grad = None

        retained_outputs.append(state.detach())
        print(f"queued step {step_index + 1}/{args.steps}", flush=True)
        del state
        del input_state

    # These reductions are submitted only after all requested steps. The first
    # host copy below is the workload's only intentional host/device wait.
    finite_checks: list[torch.Tensor] = []
    magnitudes: list[torch.Tensor] = []
    for output in retained_outputs:
        finite_checks.append(torch.isfinite(output).all())
        magnitudes.append(output.float().abs().max())
    for grad in retained_grads:
        finite_checks.append(torch.isfinite(grad).all())
        magnitudes.append(grad.float().abs().max())
    for grad_norm in retained_grad_norms:
        finite_checks.append(torch.isfinite(grad_norm))
        magnitudes.append(grad_norm.float())

    replay_equal_checks: list[torch.Tensor] = []
    if args.replay_step_inputs and args.steps > 1:
        for step_index in range(1, args.steps):
            checks = [
                torch.eq(retained_outputs[step_index], retained_outputs[0]).all(),
            ]
            if retained_grads:
                checks.extend(
                    (
                        torch.eq(retained_grads[step_index], retained_grads[0]).all(),
                        torch.eq(
                            retained_grad_norms[step_index],
                            retained_grad_norms[0],
                        ),
                        parameter_grad_equal_checks[step_index - 1].all(),
                    )
                )
            replay_equal_checks.append(torch.stack(checks))

    finite_host = torch.stack(finite_checks).cpu().tolist()
    magnitude_host = torch.stack(magnitudes).cpu().tolist()
    replay_equal_host = (
        torch.stack(replay_equal_checks).cpu().tolist()
        if replay_equal_checks
        else []
    )
    parameter_grad_equal_host = (
        torch.stack(parameter_grad_equal_checks).cpu().tolist()
        if parameter_grad_equal_checks
        else []
    )
    parameter_grad_max_diff_host = (
        torch.stack(parameter_grad_max_diffs).cpu().tolist()
        if parameter_grad_max_diffs
        else []
    )

    all_finite = True
    for step_index in range(args.steps):
        output_finite = bool(finite_host[step_index])
        output_absmax = float(magnitude_host[step_index])
        all_finite = all_finite and output_finite
        message = (
            f"step={step_index + 1} output_finite={output_finite} "
            f"output_absmax={output_absmax:.8g}"
        )
        if retained_grads:
            grad_index = args.steps + step_index
            grad_norm_index = 2 * args.steps + step_index
            grad_finite = bool(finite_host[grad_index])
            grad_absmax = float(magnitude_host[grad_index])
            grad_norm_finite = bool(finite_host[grad_norm_index])
            grad_norm = float(magnitude_host[grad_norm_index])
            all_finite = all_finite and grad_finite and grad_norm_finite
            message += (
                f" grad_finite={grad_finite} grad_absmax={grad_absmax:.8g}"
                f" grad_norm={grad_norm:.8g}"
                f" grad_norm_finite={grad_norm_finite}"
            )
        print(message)

    all_deterministic = True
    for replay_index, checks in enumerate(replay_equal_host, start=2):
        output_equal = bool(checks[0])
        message = f"replay_step={replay_index} output_equal={output_equal}"
        step_deterministic = output_equal
        if retained_grads:
            input_grad_equal = bool(checks[1])
            grad_norm_equal = bool(checks[2])
            parameter_grads_equal = bool(checks[3])
            step_deterministic = (
                step_deterministic
                and input_grad_equal
                and grad_norm_equal
                and parameter_grads_equal
            )
            message += (
                f" input_grad_equal={input_grad_equal}"
                f" grad_norm_equal={grad_norm_equal}"
                f" parameter_grads_equal={parameter_grads_equal}"
            )
            if not parameter_grads_equal:
                equal_row = parameter_grad_equal_host[replay_index - 2]
                mismatch_indices = [
                    index
                    for index, equal in enumerate(equal_row)
                    if not equal
                ]
                first_mismatch = mismatch_indices[0]
                worst_mismatch = max(
                    mismatch_indices,
                    key=lambda index: parameter_grad_max_diff_host[
                        replay_index - 2
                    ][index],
                )
                max_diff = parameter_grad_max_diff_host[replay_index - 2][
                    first_mismatch
                ]
                worst_max_diff = parameter_grad_max_diff_host[replay_index - 2][
                    worst_mismatch
                ]
                mismatch_types = sorted(
                    {
                        parameter_names[index].rsplit(".", 1)[-1]
                        for index in mismatch_indices
                    }
                )
                message += (
                    f" parameter_mismatch_count={len(mismatch_indices)}"
                    f" parameter_mismatch_types={','.join(mismatch_types)}"
                    f" first_parameter_mismatch={parameter_names[first_mismatch]}"
                    f" first_parameter_maxdiff={max_diff:.8g}"
                    f" worst_parameter_mismatch={parameter_names[worst_mismatch]}"
                    f" worst_parameter_maxdiff={worst_max_diff:.8g}"
                )
        all_deterministic = all_deterministic and step_deterministic
        print(message)

    if args.check_stage_grad_binary:
        if _STAGE_GRAD_RECORDS is None:
            raise RuntimeError("stage gradient recording was not initialized")
        for name in sorted(_STAGE_GRAD_RECORDS):
            tensors = _STAGE_GRAD_RECORDS[name]
            if len(tensors) != args.steps:
                raise RuntimeError(
                    f"{name} produced {len(tensors)} records for {args.steps} steps"
                )
            if all(tensor is None for tensor in tensors):
                print(f"stage_grad={name} value=None binary_equal=True")
                continue
            if any(tensor is None for tensor in tensors):
                all_deterministic = False
                print(f"stage_grad={name} optional_presence_equal=False")
                continue
            present_tensors = [
                tensor for tensor in tensors if tensor is not None
            ]
            hashes = [tensor_binary_md5(tensor) for tensor in present_tensors]
            binary_equal = all(item == hashes[0] for item in hashes[1:])
            all_deterministic = all_deterministic and binary_equal
            print(
                f"stage_grad={name}"
                f" dtype={present_tensors[0].dtype}"
                f" shape={tuple(present_tensors[0].shape)}"
                f" binary_equal={binary_equal}"
                f" md5={','.join(hashes)}"
            )

    if args.md5:
        for step_index, output in enumerate(retained_outputs, start=1):
            message = f"step={step_index} output_md5={tensor_md5(output)}"
            if retained_grads:
                message += f" grad_md5={tensor_md5(retained_grads[step_index - 1])}"
            print(message)

    if not all_finite:
        result = "NONFINITE"
    elif not all_deterministic:
        result = "NONDETERMINISTIC"
    else:
        result = "PASS"
    print(
        "result:",
        result,
        f"deterministic={all_deterministic}",
        f"max_memory_allocated={torch.npu.max_memory_allocated(args.device)}",
    )
    return 0 if all_finite and all_deterministic else 2


if __name__ == "__main__":
    raise SystemExit(main())
