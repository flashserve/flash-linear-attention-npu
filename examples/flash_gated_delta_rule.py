import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

# Large default smoke shapes can exceed Triton-NPU's default launch-grid limit.
os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from fla.ops.triton.triton_core.causal_conv1d import causal_conv1d_triton
from fla.ops.triton.triton_core.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.triton.triton_core.cumsum import chunk_local_cumsum
from fla.ops.triton.triton_core.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.triton.triton_core.solve_tril_fast import solve_tril_npu as solve_tril
from fla.ops.triton.triton_core.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


_disable_compile = getattr(getattr(torch, "compiler", None), "disable", lambda fn: fn)
_DEFAULT_VARLEN_CHUNK_SIZES = (16, 32, 64, 128, 608 * 2)


def _make_gate(shape: tuple[int, ...], dtype: torch.dtype, device: str, gate_function: str) -> torch.Tensor:
    if gate_function == "logsigmoid":
        return F.logsigmoid(torch.randn(*shape, dtype=dtype, device=device))
    if gate_function == "negative_linear":
        lo, hi = -5e-2, -5e-5
        steps = shape[1] if len(shape) >= 2 else shape[-1]
        g_t = torch.linspace(hi, lo, steps, dtype=torch.float32, device=device).to(dtype)
        if len(shape) == 3:
            return g_t.view(1, steps, 1).expand(*shape).contiguous()
        return g_t.view(1, 1, steps, 1).expand(*shape).contiguous()
    if gate_function == "zeros":
        return torch.zeros(*shape, dtype=dtype, device=device)
    raise ValueError(f"Unsupported gate_function: {gate_function}")


def cdiv_torch(a, b):
    return (a + b - 1) // b


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat(
        [torch.arange(n) for n in cdiv_torch(prepare_lens(cu_seqlens), chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_indices_list(cu_seqlens: list[int] | torch.LongTensor, chunk_size: int) -> list[int]:
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = [int(x) for x in cu_seqlens.detach().cpu().tolist()]

    indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        length = int(cu_seqlens[seq_idx + 1]) - int(cu_seqlens[seq_idx])
        if length <= 0:
            continue
        for chunk_idx in range((length + chunk_size - 1) // chunk_size):
            indices.extend([seq_idx, chunk_idx])
    return indices


def _as_int_list(value: Optional[list[int] | torch.Tensor]) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    return [int(x) for x in value]


def _as_chunk_dict(
    value: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor],
    chunk_size: int,
) -> Dict[str, Optional[torch.LongTensor]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {str(chunk_size): value}


def _as_chunk_list_dict(
    value: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor],
    chunk_size: int,
) -> Dict[str, Optional[list[int]]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): _as_int_list(v) for k, v in value.items()}
    return {str(chunk_size): _as_int_list(value)}


def _next_power_of_2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _cumsum_block_t(g: torch.Tensor, chunk_size: int) -> int:
    # Keep this aligned with fla.ops.triton.triton_core.cumsum.chunk_local_cumsum_scalar.
    h = int(g.shape[-1])
    return _next_power_of_2((1 << 17) // max(1, h * int(chunk_size)))


def _ensure_varlen_metadata(
    g: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    cu_seqlens_list: Optional[list[int]],
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor],
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor],
    chunk_size: int,
) -> tuple[
    Optional[torch.LongTensor],
    Optional[list[int]],
    Optional[Dict[str, Optional[torch.LongTensor]]],
    Optional[Dict[str, Optional[list[int]]]],
]:
    if cu_seqlens is None:
        return None, None, None, None

    cu_seqlens = cu_seqlens.to(device=g.device, dtype=torch.int64)
    cu_seqlens_list = _as_int_list(cu_seqlens_list) or _as_int_list(cu_seqlens)
    assert cu_seqlens_list is not None

    tensor_indices = _as_chunk_dict(chunk_indices, chunk_size)
    list_indices = _as_chunk_list_dict(chunk_indices_list, chunk_size)

    required_sizes = set(_DEFAULT_VARLEN_CHUNK_SIZES)
    required_sizes.add(int(chunk_size))
    required_sizes.add(_cumsum_block_t(g, chunk_size))

    for size in required_sizes:
        key = str(size)
        if key not in tensor_indices or tensor_indices[key] is None:
            tensor_indices[key] = prepare_chunk_indices(cu_seqlens, size)
        if key not in list_indices or list_indices[key] is None:
            list_indices[key] = prepare_chunk_indices_list(cu_seqlens_list, size)

    return cu_seqlens, cu_seqlens_list, tensor_indices, list_indices


def _chunk_tensor(
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]],
    chunk_size: int,
) -> Optional[torch.LongTensor]:
    if chunk_indices is None:
        return None
    return chunk_indices.get(str(chunk_size))


def _chunk_list(
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]],
    chunk_size: int,
) -> Optional[list[int]]:
    if chunk_indices_list is None:
        return None
    return chunk_indices_list.get(str(chunk_size))


def flash_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
    chunk_size: int = 64,
):
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )

    # A is the WY lower-triangular representation before inversion.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )

    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        output_dtype=k.dtype,
    )

    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    A = A.transpose(1, 2).contiguous()

    w, u = torch.ops.npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    if not output_final_state:
        final_state = None

    o = torch.ops.npu.npu_chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g,
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )

    g = g.transpose(1, 2).contiguous()
    o = o.transpose(1, 2).contiguous()
    return g, o, A, final_state


def flash_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
    chunk_size: int = 64,
):
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()

    w, u = torch.ops.npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    do = do.transpose(1, 2).contiguous()

    h, v_new, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )

    dv = torch.ops.npu.npu_chunk_bwd_dv_local(
        q,
        k,
        do,
        g,
        scale,
        chunk_size,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dh, dh0, dv = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do,
        dv,
        scale,
        chunk_size,
        g=g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    dh0 = None

    dq, dk, dw, dg = torch.ops.npu.npu_chunk_bwd_dqkwg(
        q,
        k,
        v_new,
        g,
        h,
        do,
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

    dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
        k,
        v,
        beta.float(),
        A,
        dw,
        dv,
        g.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dk2, dv, db, dg2 = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        dv,
        g,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()

    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype}, should be float32")

    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )

    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_list: Optional[list[int]] = None,
        chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
        chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, final_state = flash_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
        )

        ctx.save_for_backward(q, k, v, g, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.initial_state = initial_state
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = flash_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=ctx.initial_state,
            do=do,
            dht=dht,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_list=ctx.cu_seqlens_list,
            chunk_indices=ctx.chunk_indices,
            chunk_indices_list=ctx.chunk_indices_list,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@_disable_compile
def flash_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor] = None,
    chunk_size: int = 64,
    head_first: bool = False,
):
    r"""
    Flash-linear-attention NPU port of xtuner's GDN entry.

    This port keeps the xtuner NPU layout:
        q, k: [B, H, T, K]
        v:    [B, H, T, V]
        g:    [B, T, H]
        beta: [B, T, H]

    It returns:
        o: [B, T, H, V]
        final_state: [N, H, K, V] when output_final_state=True, else None
    """
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype}, k current type is {k.dtype}, "
            f"v current type is {v.dtype}; they should be equal"
        )
    if q.dtype == torch.float32:
        raise ValueError("ChunkGatedDeltaRuleFunction does not support float32. Please use float16/bfloat16.")
    if beta.ndim != 3 or g.ndim != 3:
        raise ValueError("g and beta must be rank-3 tensors with shape [B, T, H].")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors with shape [B, H, T, D].")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError(f"q/k/v shape prefixes must match, got {q.shape}, {k.shape}, {v.shape}.")
    if g.shape != beta.shape:
        raise ValueError(f"g and beta shapes must match, got {g.shape} and {beta.shape}.")
    if g.shape[0] != q.shape[0] or g.shape[1] != q.shape[2] or g.shape[2] != q.shape[1]:
        raise ValueError(
            "Expected q/k/v in [B, H, T, D] and g/beta in [B, T, H]; "
            f"got q={tuple(q.shape)}, g={tuple(g.shape)}."
        )

    if head_first:
        warnings.warn(
            "head_first is kept only for API compatibility. This NPU port always expects q/k/v as [B, H, T, D].",
            stacklevel=2,
        )
    if chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError(f"chunk_size must be a power of 2, got {chunk_size}.")

    if cu_seqlens is not None:
        cu_seqlens, cu_seqlens_list, chunk_indices, chunk_indices_list = _ensure_varlen_metadata(
            g=g,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
        )
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using cu_seqlens. "
                "Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens_list) - 1:
            raise ValueError(
                "The number of initial states is expected to match the number of input sequences, "
                f"got initial_state.shape[0]={initial_state.shape[0]} and sequences={len(cu_seqlens_list) - 1}."
            )
    else:
        cu_seqlens_list = None
        chunk_indices = None
        chunk_indices_list = None

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_list,
        use_qk_l2norm_in_kernel,
        chunk_size,
    )
    return o, final_state


class QwenStyleRMSNormGated(nn.Module):
    """RMSNorm + gated SiLU."""

    def __init__(self, head_v_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_v_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        inp_dtype = hidden_states.dtype
        hidden_states = torch_npu.npu_rms_norm(hidden_states, weight, self.variance_epsilon)[0]
        hidden_states = hidden_states * torch.nn.functional.silu(gate.to(hidden_states.dtype))
        return hidden_states.to(inp_dtype)


class DemoGatedDeltaNet(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        *,
        num_value_heads: int,
        num_key_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_dim: int,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_v_heads = num_value_heads
        self.num_k_heads = num_key_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                "num_value_heads must be an integer multiple of num_key_heads "
                f"for the current grouped-value smoke path, got {self.num_v_heads} and {self.num_k_heads}."
            )
        self.head_k_dim = key_head_dim
        self.head_v_dim = value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = conv_kernel_dim
        self.activation = hidden_act
        self.chunk_size_default = chunk_size

        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.rand(self.num_v_heads) * 15.99 + 0.01))
        self.in_proj_qkv = nn.Linear(self.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.norm = QwenStyleRMSNormGated(self.head_v_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: Optional[torch.LongTensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError("DemoGatedDeltaNet 仅对齐 xtuner bs=1 路径")

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        weight = self.conv1d.weight.squeeze(1)
        bias = self.conv1d.bias

        if cu_seqlens is not None and cu_seqlens.device != mixed_qkv.device:
            cu_seqlens = cu_seqlens.to(mixed_qkv.device)

        mixed_qkv, _ = causal_conv1d_triton(
            mixed_qkv,
            weight=weight,
            H=2 * self.num_k_heads + self.num_v_heads,
            bias=bias,
            residual=None,
            initial_state=None,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        )

        query, key, value = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim).transpose(1, 2).contiguous()
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim).transpose(1, 2).contiguous()
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim).transpose(1, 2).contiguous()

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)

        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=1)
            key = key.repeat_interleave(repeat, dim=1)

        cu_list = cu_seqlens.detach().tolist() if cu_seqlens is not None else None
        cu_seqlens_list: Optional[list[int]] = cu_list if cu_list is not None else None

        core_attn_out, _ = flash_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=None,
            chunk_indices_list=None,
            chunk_size=self.chunk_size_default,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


__all__ = [
    "flash_gated_delta_rule",
    "flash_chunk_gated_delta_rule_fwd",
    "flash_chunk_gated_delta_rule_bwd",
    "prepare_chunk_indices",
    "prepare_chunk_indices_list",
    "DemoGatedDeltaNet",
    "QwenStyleRMSNormGated",
]


def _build_mean_1k_cu_seqlens(
    total_tokens: int,
    chunk_size: int,
    device: str,
    mean_len: int = 1024,
) -> torch.LongTensor:
    num_seqs = max(1, round(total_tokens / mean_len))
    if num_seqs == 1:
        lengths = [total_tokens]
    else:
        target = max(1, total_tokens // num_seqs)
        delta = max(chunk_size, (target // 4 // chunk_size) * chunk_size)
        low = max(1, target - delta)
        high = max(1, target + delta)
        lengths = [low if i % 2 == 0 else high for i in range(num_seqs)]

        diff = total_tokens - sum(lengths)
        i = len(lengths) - 1
        while diff != 0:
            if diff > 0:
                add = min(diff, max(1, delta))
                lengths[i] += add
                diff -= add
            else:
                sub = min(-diff, max(0, lengths[i] - 1))
                if sub == 0:
                    i = (i - 1) % len(lengths)
                    continue
                lengths[i] -= sub
                diff += sub
            i = (i - 1) % len(lengths)

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def _main():
    import argparse

    import torch_npu

    import fla_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", default="", help="Example/ST case name for CI logs")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32, help="legacy alias for --query-heads and --value-heads")
    parser.add_argument("--query-heads", type=int, default=None)
    parser.add_argument("--value-heads", type=int, default=None)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--dim", type=int, default=128, help="legacy alias for --key-dim")
    parser.add_argument("--key-dim", type=int, default=None)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--mean-len", type=int, default=1024)
    parser.add_argument("--gate-source", choices=["g", "gk", "g+gk"], default="g")
    parser.add_argument("--gate-function", choices=["logsigmoid", "negative_linear", "zeros"], default="logsigmoid")
    parser.add_argument("--initial-state", choices=["none", "zeros", "random"], default="none")
    parser.add_argument("--output-final-state", action="store_true")
    parser.add_argument("--qk-l2norm", dest="qk_l2norm", action="store_true", default=True)
    parser.add_argument("--no-qk-l2norm", dest="qk_l2norm", action="store_false")
    parser.add_argument("--varlen", dest="varlen", action="store_true", default=True)
    parser.add_argument("--no-varlen", dest="varlen", action="store_false")
    parser.add_argument(
        "--demo-model",
        action="store_true",
        help="运行 DemoGatedDeltaNet（从 cu_seqlens 迁移到 tensor 设备及 causal_conv 起始的完整链路）代替裸张量 attn 冒烟",
    )
    parser.add_argument("--conv-kernel", type=int, default=4, help="depthwise causal conv kernel size")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    query_heads = args.query_heads if args.query_heads is not None else args.heads
    value_heads = args.value_heads if args.value_heads is not None else args.heads
    key_dim = args.key_dim if args.key_dim is not None else args.dim
    value_dim = args.value_dim
    batch = args.batch
    device = f"npu:{args.device}"
    positive_values = {
        "batch": batch,
        "tokens": args.tokens,
        "query_heads": query_heads,
        "value_heads": value_heads,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "chunk_size": args.chunk_size,
        "mean_len": args.mean_len,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if args.varlen and batch != 1:
        raise ValueError("varlen smoke currently requires batch=1. Use --no-varlen for B > 1 cases.")
    if value_heads % query_heads != 0:
        raise ValueError(
            "value_heads must be an integer multiple of query_heads for the current grouped-value smoke path, "
            f"got query_heads={query_heads}, value_heads={value_heads}."
        )
    if args.gate_source != "g":
        raise NotImplementedError(
            f"gate_source={args.gate_source} is reserved in the Example/ST schema, but the current NPU "
            "fwd_h path only supports g and requires gk=None."
        )

    if args.demo_model:
        if (
            args.gate_source != "g"
            or args.gate_function != "logsigmoid"
            or args.initial_state != "none"
            or args.output_final_state
            or not args.qk_l2norm
        ):
            raise ValueError(
                "demo_model smoke currently uses its built-in gate, state and qk-l2norm path. "
                "Use demo_model=false for gate/initial_state/output_final_state/qk_l2norm case coverage."
            )
        if batch != 1:
            raise ValueError("demo_model smoke currently requires batch=1.")
        hidden_size = query_heads * key_dim
        if hidden_size <= 0:
            raise ValueError("hidden_size = query_heads * key_dim 非法")
        cu_seqlens = None
        if args.varlen:
            cu_seqlens = _build_mean_1k_cu_seqlens(
                total_tokens=args.tokens,
                chunk_size=args.chunk_size,
                device=device,
                mean_len=args.mean_len,
            )
        x = torch.randn(batch, args.tokens, hidden_size, dtype=dtype, device=device, requires_grad=True)
        net = DemoGatedDeltaNet(
            hidden_size,
            num_value_heads=value_heads,
            num_key_heads=query_heads,
            key_head_dim=key_dim,
            value_head_dim=value_dim,
            conv_kernel_dim=args.conv_kernel,
            chunk_size=args.chunk_size,
        ).to(device=device, dtype=dtype)
        torch.npu.synchronize()
        out = net(x, cu_seqlens=cu_seqlens)
        torch.npu.synchronize()
        print("demo_model forward:", tuple(out.shape), out.dtype, "finite:", torch.isfinite(out.float()).all().item())
        loss = out.float().square().mean()
        loss.backward()
        torch.npu.synchronize()
        print("demo_model backward ok")
        print(
            "x.grad finite:",
            torch.isfinite(x.grad.float()).all().item() if x.grad is not None else None,
            "norm:",
            float(x.grad.float().norm()) if x.grad is not None else None,
        )
        return

    q = torch.randn(batch, query_heads, args.tokens, key_dim, dtype=dtype, device=device)
    k = torch.randn(batch, query_heads, args.tokens, key_dim, dtype=dtype, device=device)
    v = torch.randn(batch, value_heads, args.tokens, value_dim, dtype=dtype, device=device)
    beta = torch.rand(batch, args.tokens, value_heads, dtype=dtype, device=device).sigmoid()
    g = _make_gate((batch, args.tokens, value_heads), dtype, device, args.gate_function)

    cu_seqlens = None
    if args.varlen:
        cu_seqlens = _build_mean_1k_cu_seqlens(
            total_tokens=args.tokens,
            chunk_size=args.chunk_size,
            device=device,
            mean_len=args.mean_len,
        )
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        print(
            "varlen:",
            "num_seqs=", int(lens.numel()),
            "mean_len=", float(lens.float().mean().item()),
            "min_len=", int(lens.min().item()),
            "max_len=", int(lens.max().item()),
        )

    print(
        "config:",
        f"case={args.case_name or '<direct>'}",
        f"B={batch}",
        f"QH={query_heads}",
        f"VH={value_heads}",
        f"T={args.tokens}",
        f"K={key_dim}",
        f"V={value_dim}",
        f"chunk_size={args.chunk_size}",
        f"dtype={args.dtype}",
        f"varlen={args.varlen}",
        f"gate_source={args.gate_source}",
        f"gate_function={args.gate_function}",
        f"initial_state={args.initial_state}",
        f"output_final_state={args.output_final_state}",
        f"qk_l2norm={args.qk_l2norm}",
        f"device={device}",
    )

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    beta.requires_grad_(True)
    g.requires_grad_(True)

    torch.npu.synchronize()
    attn_q = q
    attn_k = k
    if query_heads != value_heads:
        repeat = value_heads // query_heads
        attn_q = q.repeat_interleave(repeat, dim=1)
        attn_k = k.repeat_interleave(repeat, dim=1)
        print("grouped value heads:", f"repeat={repeat}", f"attn_heads={attn_q.shape[1]}")
    initial_state = None
    if args.initial_state != "none":
        state_count = len(cu_seqlens) - 1 if cu_seqlens is not None else batch
        state_shape = (state_count, value_heads, key_dim, value_dim)
        if args.initial_state == "zeros":
            initial_state = torch.zeros(state_shape, dtype=dtype, device=device)
        else:
            initial_state = torch.randn(state_shape, dtype=dtype, device=device)
        print("initial_state:", tuple(initial_state.shape), initial_state.dtype, initial_state.device)
    o, final_state = flash_gated_delta_rule(
        attn_q,
        attn_k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=args.output_final_state,
        use_qk_l2norm_in_kernel=args.qk_l2norm,
        cu_seqlens=cu_seqlens,
        chunk_size=args.chunk_size,
    )
    torch.npu.synchronize()

    print("forward ok")
    print("o:", tuple(o.shape), o.dtype, o.device, "finite:", torch.isfinite(o.float()).all().item())
    print("final_state:", None if final_state is None else tuple(final_state.shape))

    loss = o.float().square().mean()
    loss.backward()
    torch.npu.synchronize()
    print("backward ok")
    for name, tensor in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
        grad = tensor.grad
        print(name, "grad finite:", torch.isfinite(grad.float()).all().item(), "norm:", grad.float().norm().item())


if __name__ == "__main__":
    _main()
