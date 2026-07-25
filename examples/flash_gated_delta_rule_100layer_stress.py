#!/usr/bin/env python3
"""Pure-asynchronous 100-layer GDN reproduction workload.

The workload intentionally performs no host/device synchronization while steps
are being submitted. All outputs and input gradients remain on the NPU until
every requested step has finished launching; finite checks are queued only at
the end and then copied to the host once.
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

    dh, _dh0, dv = ascendc_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do_head,
        dv,
        scale,
        chunk_size,
        g=g_head,
        gK=None,
        h0=None,
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

        ctx.save_for_backward(x, op_weight, preactivation)
        ctx.query_start_loc = query_start_loc
        ctx.activation_mode = activation_mode
        trace_apply("stage return: StressCausalConv1dFunction.forward")
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        trace_apply("stage enter: StressCausalConv1dFunction.backward")
        x, op_weight, preactivation = ctx.saved_tensors
        op_x = x.reshape(x.shape[1], x.shape[2]).contiguous()
        dx, dw, _db, _dh0 = ascendc_causal_conv1d_bwd(
            x=op_x,
            y=preactivation if ctx.activation_mode != 0 else None,
            weight=op_weight,
            dy=dy.contiguous(),
            initial_state=None,
            dht=None,
            query_start_loc=ctx.query_start_loc,
            activation=ctx.activation_mode,
            input_layout="NTD",
        )
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

    layers: list[Callable[[torch.Tensor], torch.Tensor]] = []
    for _ in range(args.layers):
        hidden_size = args.heads * args.dim
        projection_shape = (1, 1, hidden_size)
        gate_shape = (1, 1, args.heads)
        layers.append(
            make_layer(
                q_weight=torch.randn(
                    projection_shape,
                    dtype=dtype,
                    device=device,
                ).mul(0.1).add(1.0),
                k_weight=torch.randn(
                    projection_shape,
                    dtype=dtype,
                    device=device,
                ).mul(0.1).add(1.0),
                v_weight=torch.randn(
                    projection_shape,
                    dtype=dtype,
                    device=device,
                ).mul(0.1).add(0.5),
                conv_weight=torch.randn(
                    3 * hidden_size,
                    args.conv_width,
                    dtype=dtype,
                    device=device,
                ).mul(0.1),
                beta_weight=torch.randn(
                    gate_shape,
                    dtype=torch.float32,
                    device=device,
                ).mul(0.1).add(1.0),
                beta_bias=torch.randn(
                    gate_shape,
                    dtype=torch.float32,
                    device=device,
                ).mul(0.1),
                gate_weight=torch.randn(
                    gate_shape,
                    dtype=torch.float32,
                    device=device,
                ).mul(0.1).add(1.0),
                gate_bias=torch.randn(
                    gate_shape,
                    dtype=torch.float32,
                    device=device,
                ).mul(0.1),
                norm_weight=torch.randn(
                    args.heads * args.dim,
                    dtype=dtype,
                    device=device,
                ).mul(0.01).add(1.0),
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
        "causal_conv_output_layout=NTD",
        "apply_wrappers=causal_conv1d,gated_delta_rule",
        flush=True,
    )

    for step_index in range(args.steps):
        input_state = torch.randn(
            1,
            args.heads,
            args.tokens,
            args.dim,
            dtype=dtype,
            device=device,
            requires_grad=not args.forward_only,
        )
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
            upstream = torch.randn_like(state)
            state.backward(upstream)
            if input_state.grad is None:
                raise RuntimeError("input gradient was not produced")
            retained_grads.append(input_state.grad.detach())

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

    finite_host = torch.stack(finite_checks).cpu().tolist()
    magnitude_host = torch.stack(magnitudes).cpu().tolist()

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
            grad_finite = bool(finite_host[grad_index])
            grad_absmax = float(magnitude_host[grad_index])
            all_finite = all_finite and grad_finite
            message += f" grad_finite={grad_finite} grad_absmax={grad_absmax:.8g}"
        print(message)

    if args.md5:
        for step_index, output in enumerate(retained_outputs, start=1):
            message = f"step={step_index} output_md5={tensor_md5(output)}"
            if retained_grads:
                message += f" grad_md5={tensor_md5(retained_grads[step_index - 1])}"
            print(message)

    print(
        "result:",
        "PASS" if all_finite else "NONFINITE",
        f"max_memory_allocated={torch.npu.max_memory_allocated(args.device)}",
    )
    return 0 if all_finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
