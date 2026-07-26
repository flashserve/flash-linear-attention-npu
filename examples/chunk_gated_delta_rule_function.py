"""Complete autograd adapter for the AscendC chunk gated-delta-rule chain.

The public entry is ``ChunkGatedDeltaRuleFunction.apply`` with this signature::

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

The adapter implements the full forward and backward chains, including GVA
head reduction. The current v26.6.0 ``bwd_dhu`` kernel does not consume
``h0``/``dht`` or produce ``dh0``. This wrapper therefore supports
``initial_state`` as a constant forward input, but rejects initial-state
gradients and final-state-gradient propagation instead of silently returning
incorrect gradients. Other unsupported combinations are rejected before the
first compute kernel is launched.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from fla_npu.ops.ascendc import (
    npu_chunk_bwd_dqkwg as ascendc_chunk_bwd_dqkwg,
    npu_chunk_bwd_dv_local as ascendc_chunk_bwd_dv_local,
    npu_chunk_fwd_o as ascendc_chunk_fwd_o,
    npu_chunk_gated_delta_rule_bwd_dhu as ascendc_chunk_gated_delta_rule_bwd_dhu,
    npu_chunk_gated_delta_rule_fwd_h as ascendc_chunk_gated_delta_rule_fwd_h,
    npu_prepare_wy_repr_bwd_da as ascendc_prepare_wy_repr_bwd_da,
    npu_prepare_wy_repr_bwd_full as ascendc_prepare_wy_repr_bwd_full,
    npu_recompute_w_u_fwd as ascendc_recompute_w_u_fwd,
    solve_tri as ascendc_solve_tri,
)
from fla_npu.ops.triton import (
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    l2norm_bwd,
    l2norm_fwd,
)


TensorIndexDict = Dict[str, Optional[torch.Tensor]]
ListIndexDict = Dict[str, Optional[list[int]]]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_CHUNK_SIZES = (64, 128)
_SUPPORTED_VALUE_DIMS = (128, 256)
_REQUIRED_METADATA_CHUNK_SIZES = (16, 32, 64, 128, 608 * 2)


def _require_npu_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "npu":
        raise ValueError(f"{name} must be on NPU, got device={tensor.device}")


def _as_int_list(
    value: Optional[list[int] | tuple[int, ...] | torch.Tensor],
    *,
    name: str,
) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be rank-1, got {tuple(value.shape)}")
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"{name} must be int32/int64, got dtype={value.dtype}"
            )
        return [int(item) for item in value.detach().cpu().tolist()]
    return [int(item) for item in value]


def _expected_chunk_metadata(
    offsets: list[int],
    chunk_size: int,
) -> tuple[list[list[int]], list[int]]:
    pairs: list[list[int]] = []
    flattened: list[int] = []
    for sequence_index, (left, right) in enumerate(
        zip(offsets, offsets[1:])
    ):
        count = (right - left + chunk_size - 1) // chunk_size
        for chunk_index in range(count):
            pairs.append([sequence_index, chunk_index])
            flattened.extend((sequence_index, chunk_index))
    return pairs, flattened


def _validate_offsets(offsets: list[int], total_tokens: int) -> None:
    if len(offsets) < 2:
        raise ValueError("cu_seqlens must contain at least [0, total_tokens]")
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError(
            "cu_seqlens must start at 0 and end at T, "
            f"got first={offsets[0]}, last={offsets[-1]}, T={total_tokens}"
        )
    if any(right <= left for left, right in zip(offsets, offsets[1:])):
        raise ValueError(
            "GDR varlen does not support empty/decreasing sequences; "
            "cu_seqlens must be strictly increasing"
        )


def _normalize_varlen_metadata(
    *,
    g: torch.Tensor,
    total_tokens: int,
    batch: int,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor],
    cu_seqlens_list: Optional[list[int]],
    chunk_indices: Optional[TensorIndexDict | torch.Tensor],
    chunk_indices_list: Optional[
        ListIndexDict | list[int] | tuple[int, ...] | torch.Tensor
    ],
) -> tuple[
    Optional[torch.Tensor],
    Optional[list[int]],
    Optional[TensorIndexDict],
    Optional[ListIndexDict],
]:
    auxiliary_present = any(
        value is not None
        for value in (
            cu_seqlens_list,
            chunk_indices,
            chunk_indices_list,
        )
    )
    if cu_seqlens is None:
        if auxiliary_present:
            raise ValueError(
                "cu_seqlens_list/chunk_indices metadata must be None when "
                "cu_seqlens is None"
            )
        return None, None, None, None

    _require_npu_tensor("cu_seqlens", cu_seqlens)
    if cu_seqlens.ndim != 1:
        raise ValueError(
            "cu_seqlens must be rank-1, "
            f"got shape={tuple(cu_seqlens.shape)}"
        )
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "cu_seqlens must be int32/int64, "
            f"got dtype={cu_seqlens.dtype}"
        )
    if batch != 1:
        raise ValueError(
            f"GDR varlen mode supports only B=1, got B={batch}"
        )
    if cu_seqlens_list is not None:
        offsets = [int(item) for item in cu_seqlens_list]
        if cu_seqlens.numel() != len(offsets):
            raise ValueError(
                "cu_seqlens and cu_seqlens_list must have the same length, "
                f"got {cu_seqlens.numel()} and {len(offsets)}"
            )
    else:
        offsets = _as_int_list(cu_seqlens, name="cu_seqlens")
    assert offsets is not None
    _validate_offsets(offsets, total_tokens)
    cu_tensor = cu_seqlens.to(device=g.device, dtype=torch.int64).contiguous()

    if isinstance(chunk_indices, dict):
        tensor_indices = dict(chunk_indices)
    elif isinstance(chunk_indices, torch.Tensor):
        tensor_indices = {str(chunk_size): chunk_indices}
    elif chunk_indices is None:
        tensor_indices = {}
    else:
        raise TypeError("chunk_indices must be a Tensor, dict, or None")

    if isinstance(chunk_indices_list, dict):
        list_indices: ListIndexDict = {}
        for key, value in chunk_indices_list.items():
            list_indices[str(key)] = _as_int_list(
                value,
                name=f"chunk_indices_list[{key!r}]",
            )
    elif chunk_indices_list is None:
        list_indices = {}
    else:
        list_indices = {
            str(chunk_size): _as_int_list(
                chunk_indices_list,
                name="chunk_indices_list",
            )
        }

    required_sizes = set(_REQUIRED_METADATA_CHUNK_SIZES)
    required_sizes.add(chunk_size)
    for size in sorted(required_sizes):
        key = str(size)
        expected_pairs, expected_flat = _expected_chunk_metadata(offsets, size)

        supplied_tensor = tensor_indices.get(key)
        if supplied_tensor is None:
            tensor_indices[key] = torch.tensor(
                expected_pairs,
                dtype=torch.int64,
                device=g.device,
            )
        else:
            _require_npu_tensor(f"chunk_indices[{key!r}]", supplied_tensor)
            if supplied_tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    f"chunk_indices[{key!r}] must be int32/int64"
                )
            expected_shape = (len(expected_pairs), 2)
            if tuple(supplied_tensor.shape) != expected_shape:
                raise ValueError(
                    f"chunk_indices[{key!r}] must have shape "
                    f"{expected_shape}, got {tuple(supplied_tensor.shape)}"
                )
            tensor_indices[key] = supplied_tensor.to(
                device=g.device,
                dtype=torch.int64,
            ).contiguous()

        supplied_list = list_indices.get(key)
        if supplied_list is None:
            list_indices[key] = expected_flat
        elif [int(item) for item in supplied_list] != expected_flat:
            raise ValueError(
                f"chunk_indices_list[{key!r}] does not match cu_seqlens"
            )
        else:
            list_indices[key] = [int(item) for item in supplied_list]

    return cu_tensor, offsets, tensor_indices, list_indices


def _chunk_tensor(
    chunk_indices: Optional[TensorIndexDict],
    chunk_size: int,
) -> Optional[torch.Tensor]:
    if chunk_indices is None:
        return None
    value = chunk_indices.get(str(chunk_size))
    if value is None:
        raise ValueError(f"missing tensor chunk metadata for size {chunk_size}")
    return value


def _chunk_list(
    chunk_indices_list: Optional[ListIndexDict],
    chunk_size: int,
) -> Optional[list[int]]:
    if chunk_indices_list is None:
        return None
    value = chunk_indices_list.get(str(chunk_size))
    if value is None:
        raise ValueError(f"missing list chunk metadata for size {chunk_size}")
    return value


def _validate_forward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
    ):
        _require_npu_tensor(name, tensor)
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"q/k/v support only FP16/BF16, got q.dtype={q.dtype}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"q/k/v dtypes must match, got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if g.dtype not in (*_SUPPORTED_DTYPES, torch.float32):
        raise ValueError(f"g must be FP16/BF16/FP32, got {g.dtype}")
    if beta.dtype not in (*_SUPPORTED_DTYPES, torch.float32):
        raise ValueError(f"beta must be FP16/BF16/FP32, got {beta.dtype}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [B,H,T,D] rank-4 tensors")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g/beta must be [B,T,HV] rank-3 tensors")
    if tuple(q.shape) != tuple(k.shape):
        raise ValueError(
            f"q and k shapes must match, got {tuple(q.shape)}, {tuple(k.shape)}"
        )

    batch, key_heads, tokens, key_dim = q.shape
    value_batch, value_heads, value_tokens, value_dim = v.shape
    if batch <= 0 or tokens <= 0 or key_heads <= 0 or value_heads <= 0:
        raise ValueError("B, H, and T dimensions must be positive")
    if value_batch != batch or value_tokens != tokens:
        raise ValueError(
            "q/k/v B and T dimensions must match, "
            f"got q={tuple(q.shape)}, v={tuple(v.shape)}"
        )
    if value_heads % key_heads != 0:
        raise ValueError(
            "GVA requires HV to be divisible by HK, "
            f"got HK={key_heads}, HV={value_heads}"
        )
    if key_dim != 128:
        raise ValueError(
            f"the current AscendC GDR chain supports only K=128, got K={key_dim}"
        )
    if value_dim not in _SUPPORTED_VALUE_DIMS:
        raise ValueError(
            "the current AscendC GDR chain supports V=128 or V=256, "
            f"got V={value_dim}"
        )
    expected_gate_shape = (batch, tokens, value_heads)
    if tuple(g.shape) != expected_gate_shape:
        raise ValueError(
            f"g must have shape [B,T,HV]={expected_gate_shape}, "
            f"got {tuple(g.shape)}"
        )
    if tuple(beta.shape) != expected_gate_shape:
        raise ValueError(
            f"beta must have shape [B,T,HV]={expected_gate_shape}, "
            f"got {tuple(beta.shape)}"
        )
    if chunk_size not in _SUPPORTED_CHUNK_SIZES:
        raise ValueError(
            f"chunk_size must be 64 or 128, got {chunk_size}"
        )
    if not isinstance(output_final_state, bool):
        raise TypeError("output_final_state must be bool")
    if not isinstance(use_qk_l2norm_in_kernel, bool):
        raise TypeError("use_qk_l2norm_in_kernel must be bool")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
    ):
        raise ValueError(f"scale must be a finite number, got {scale!r}")
    if float(scale) <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    if initial_state is not None:
        _require_npu_tensor("initial_state", initial_state)
        if initial_state.device != q.device or initial_state.dtype != q.dtype:
            raise ValueError(
                "initial_state must have the same dtype/device as q; "
                "FP32 state is not supported by the complete backward chain"
            )
        if initial_state.ndim != 4:
            raise ValueError(
                "initial_state must be [N,HV,K,V], "
                f"got {tuple(initial_state.shape)}"
            )
        if tuple(initial_state.shape[1:]) != (
            value_heads,
            key_dim,
            value_dim,
        ):
            raise ValueError(
                "initial_state trailing shape must be [HV,K,V], expected "
                f"{(value_heads, key_dim, value_dim)}, "
                f"got {tuple(initial_state.shape[1:])}"
            )

    return batch, key_heads, value_heads, tokens, key_dim, value_dim


def _expand_k_for_kkt(
    k: torch.Tensor,
    *,
    value_heads: int,
) -> torch.Tensor:
    key_heads = k.shape[1]
    if key_heads == value_heads:
        return k
    return k.repeat_interleave(value_heads // key_heads, dim=1).contiguous()


def _reduce_gva_key_gradient(
    gradient: torch.Tensor,
    *,
    key_heads: int,
) -> torch.Tensor:
    value_heads = gradient.shape[1]
    if value_heads == key_heads:
        return gradient
    group_size = value_heads // key_heads
    return gradient.reshape(
        gradient.shape[0],
        key_heads,
        group_size,
        gradient.shape[2],
        gradient.shape[3],
    ).sum(dim=2)


def _solve_tri(
    A: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    cu_seqlens_list: Optional[list[int]],
    chunk_indices_list: Optional[list[int]],
) -> torch.Tensor:
    A_input = A.to(output_dtype).contiguous()
    if cu_seqlens_list is None:
        return ascendc_solve_tri(A_input, layout="bsnd")
    if chunk_indices_list is None:
        raise ValueError("varlen solve_tri requires chunk_indices_list")
    return ascendc_solve_tri(
        A_input.squeeze(0),
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list,
        layout="tnd",
    ).unsqueeze(0)


def _recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta_head: torch.Tensor,
    A_head: torch.Tensor,
    g_head: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens_list: Optional[list[int]],
    chunk_indices_list: Optional[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    return ascendc_recompute_w_u_fwd(
        k,
        v,
        beta_head,
        A_head,
        chunk_size,
        g=g_head,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list,
    )


def _forward_chain(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor],
    cu_seqlens_list: Optional[list[int]],
    chunk_indices: Optional[TensorIndexDict],
    chunk_indices_list: Optional[ListIndexDict],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    cumulative_g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )
    k_for_kkt = _expand_k_for_kkt(k, value_heads=v.shape[1])
    A = chunk_scaled_dot_kkt_fwd(
        k=k_for_kkt,
        g=cumulative_g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = _solve_tri(
        A,
        output_dtype=k.dtype,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices_list=_chunk_list(chunk_indices_list, chunk_size),
    )

    g_head = cumulative_g.transpose(1, 2).contiguous()
    beta_head = beta.transpose(1, 2).contiguous().float()
    A_head = A.transpose(1, 2).contiguous()
    chunk_list = _chunk_list(chunk_indices_list, chunk_size)
    w, u = _recompute_w_u(
        k,
        v,
        beta_head,
        A_head,
        g_head,
        chunk_size=chunk_size,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices_list=chunk_list,
    )
    h, v_new, final_state = ascendc_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g_head,
        gk=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_list,
        use_exp2=False,
        transpose_state_layout=False,
    )
    output_head = ascendc_chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g_head,
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_list,
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )
    output = output_head.transpose(1, 2).contiguous()
    if not output_final_state:
        final_state = None
    return cumulative_g, output, A_head, final_state


def _backward_chain(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_g: torch.Tensor,
    beta: torch.Tensor,
    A_head: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: Optional[torch.Tensor],
    dht: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    cu_seqlens_list: Optional[list[int]],
    chunk_indices: Optional[TensorIndexDict],
    chunk_indices_list: Optional[ListIndexDict],
    chunk_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    batch, key_heads, tokens, _ = q.shape
    value_heads = v.shape[1]
    value_dim = v.shape[-1]
    if do is None:
        do = torch.zeros(
            batch,
            tokens,
            value_heads,
            value_dim,
            dtype=v.dtype,
            device=v.device,
        )
    expected_do = (batch, tokens, value_heads, value_dim)
    if tuple(do.shape) != expected_do:
        raise ValueError(
            f"do must have shape {expected_do}, got {tuple(do.shape)}"
        )
    if do.dtype != v.dtype or do.device != v.device:
        raise ValueError("do must have the same dtype/device as v")
    if dht is not None:
        raise NotImplementedError(
            "final_state gradient is unsupported by the current v26.6.0 "
            "AscendC chunk_gated_delta_rule_bwd_dhu kernel: the kernel "
            "ignores dht. Set output_final_state=False, or do not include "
            "final_state in the differentiated loss."
        )

    g_head = cumulative_g.transpose(1, 2).contiguous()
    beta_head = beta.transpose(1, 2).contiguous().float()
    do_head = do.transpose(1, 2).contiguous()
    chunk_list = _chunk_list(chunk_indices_list, chunk_size)
    w, u = _recompute_w_u(
        k,
        v,
        beta_head,
        A_head,
        g_head,
        chunk_size=chunk_size,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices_list=chunk_list,
    )
    h, v_new, _ = ascendc_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g_head,
        gk=None,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_list,
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
        A=A_head,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_list,
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
        # v26.6.0 ignores both state tensors in the kernel. Passing the
        # documented 4D h0 also hits an erroneous 5D check in the legacy
        # torch wrapper, so omit both on supported gradient paths.
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_list,
        use_exp2=False,
        transpose_state_layout=False,
    )
    dq_value, dk_value, dw, dg = ascendc_chunk_bwd_dqkwg(
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
        chunk_indices=chunk_list,
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )
    dA = ascendc_prepare_wy_repr_bwd_da(
        k,
        v,
        beta_head,
        A_head,
        dw,
        dv,
        g_head.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_list,
    )
    dk_wy, dv, dbeta, dg_wy = ascendc_prepare_wy_repr_bwd_full(
        k,
        v,
        beta_head,
        A_head,
        dA,
        dw,
        dv,
        g_head,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_list,
    )

    dq = _reduce_gva_key_gradient(dq_value, key_heads=key_heads)
    dk = _reduce_gva_key_gradient(dk_value, key_heads=key_heads)
    dk.add_(dk_wy)
    dg = dg.transpose(1, 2).contiguous()
    dg_wy = dg_wy.transpose(1, 2).contiguous()
    dg.add_(dg_wy)
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )
    dbeta = dbeta.transpose(1, 2).contiguous()
    # v26.6.0 does not implement dh0. Forward rejects a differentiable
    # initial_state, so None is the only correct result on supported paths.
    dh0 = None
    return dq, dk, dv, dg, dbeta, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Full autograd binding for the AscendC GDR operator chain."""

    @staticmethod
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
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_list: Optional[list[int]] = None,
        chunk_indices: Optional[TensorIndexDict | torch.Tensor] = None,
        chunk_indices_list: Optional[
            ListIndexDict | list[int] | tuple[int, ...] | torch.Tensor
        ] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        (
            batch,
            _key_heads,
            _value_heads,
            tokens,
            _key_dim,
            _value_dim,
        ) = _validate_forward_inputs(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            chunk_size,
        )
        (
            cu_seqlens,
            cu_seqlens_list,
            chunk_indices,
            chunk_indices_list,
        ) = _normalize_varlen_metadata(
            g=g,
            total_tokens=tokens,
            batch=batch,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
        )
        expected_states = (
            len(cu_seqlens_list) - 1
            if cu_seqlens_list is not None
            else batch
        )
        if initial_state is not None and initial_state.shape[0] != expected_states:
            raise ValueError(
                "initial_state sequence dimension does not match the input, "
                f"expected={expected_states}, got={initial_state.shape[0]}"
            )
        if initial_state is not None and ctx.needs_input_grad[6]:
            raise NotImplementedError(
                "initial_state.requires_grad=True is unsupported by the "
                "current v26.6.0 AscendC "
                "chunk_gated_delta_rule_bwd_dhu kernel: the kernel does not "
                "produce dh0. Pass a detached initial_state until state "
                "backward is implemented."
            )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        if initial_state is not None:
            initial_state = initial_state.contiguous()
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd = None
            k_rstd = None

        cumulative_g, output, A_head, final_state = _forward_chain(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=float(scale),
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
        )

        saved = [q, k, v, cumulative_g, beta, A_head]
        if q_rstd is not None:
            saved.extend((q_rstd, k_rstd))
        if initial_state is not None:
            saved.append(initial_state)
        ctx.save_for_backward(*saved)
        ctx.has_l2norm = use_qk_l2norm_in_kernel
        ctx.has_initial_state = initial_state is not None
        ctx.scale = float(scale)
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        ctx.chunk_size = chunk_size
        ctx.set_materialize_grads(False)
        return output.to(q.dtype), final_state

    @staticmethod
    def backward(
        ctx,
        do: Optional[torch.Tensor],
        dht: Optional[torch.Tensor],
    ):
        saved = list(ctx.saved_tensors)
        q = saved.pop(0)
        k = saved.pop(0)
        v = saved.pop(0)
        cumulative_g = saved.pop(0)
        beta = saved.pop(0)
        A_head = saved.pop(0)
        if ctx.has_l2norm:
            q_rstd = saved.pop(0)
            k_rstd = saved.pop(0)
        else:
            q_rstd = None
            k_rstd = None
        initial_state = saved.pop(0) if ctx.has_initial_state else None

        dq, dk, dv, dg, dbeta, dh0 = _backward_chain(
            q=q,
            k=k,
            v=v,
            cumulative_g=cumulative_g,
            beta=beta,
            A_head=A_head,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_list=ctx.cu_seqlens_list,
            chunk_indices=ctx.chunk_indices,
            chunk_indices_list=ctx.chunk_indices_list,
            chunk_size=ctx.chunk_size,
        )
        if ctx.has_l2norm:
            assert q_rstd is not None and k_rstd is not None
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        # The order must match all 14 forward inputs exactly:
        # q, k, v, g, beta, scale, initial_state, output_final_state,
        # cu_seqlens, cu_seqlens_list, chunk_indices, chunk_indices_list,
        # use_qk_l2norm_in_kernel, chunk_size.
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(cumulative_g),
            dbeta.to(beta),
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


__all__ = ["ChunkGatedDeltaRuleFunction"]
