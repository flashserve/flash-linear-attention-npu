"""Complete autograd adapter for the AscendC causal-conv1d operators.

The public entry is ``AscendCCausalConv1dFunction.apply`` with this signature::

    y, final_state = AscendCCausalConv1dFunction.apply(
        x,
        weight,
        head_num,
        bias,
        residual,
        initial_state,
        activation,
        cu_seqlens,
        output_final_state,
    )

Supported layouts:

* fixed length: ``x`` is ``[B, T, D]`` and ``y`` is ``[B, H, T, Dh]``;
* varlen: ``x`` is ``[T, D]`` or ``[1, T, D]`` and ``y`` is
  ``[1, H, T, Dh]``;
* ``initial_state`` accepts ``[N, D, W]`` or ``[N, W, D]``;
* ``final_state`` is always returned as ``[N, D, W]``.

Decode/cache-update modes are deliberately not exposed by this training
adapter. Unsupported dtype, shape, activation, state, and varlen combinations
are rejected before an AscendC kernel is launched.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from fla_npu.ops.ascendc import (
    npu_causal_conv1d as ascendc_causal_conv1d,
    npu_causal_conv1d_bwd as ascendc_causal_conv1d_bwd,
)


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_WIDTHS = (2, 3, 4)


def _require_npu_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "npu":
        raise ValueError(f"{name} must be on NPU, got device={tensor.device}")


def _as_int_list(
    value: Optional[list[int] | tuple[int, ...] | torch.Tensor],
) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(
                f"cu_seqlens must be rank-1, got shape={tuple(value.shape)}"
            )
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"cu_seqlens must be int32/int64, got dtype={value.dtype}"
            )
        return [int(item) for item in value.detach().cpu().tolist()]
    return [int(item) for item in value]


def _activation_mode(activation: Optional[str]) -> int:
    if activation is None or activation == "":
        return 0
    if activation in ("silu", "swish"):
        return 1
    raise ValueError(
        "AscendCCausalConv1dFunction supports activation=None, 'silu', "
        f"or 'swish', got {activation!r}"
    )


def _validate_offsets(offsets: list[int], total_tokens: int) -> None:
    if len(offsets) < 2:
        raise ValueError("cu_seqlens must contain at least [0, total_tokens]")
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError(
            "cu_seqlens must start at 0 and end at total_tokens, "
            f"got first={offsets[0]}, last={offsets[-1]}, "
            f"total_tokens={total_tokens}"
        )
    if any(right <= left for left, right in zip(offsets, offsets[1:])):
        raise ValueError(
            "empty or decreasing varlen sequences are not supported; "
            "cu_seqlens must be strictly increasing"
        )


def _validate_forward_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    head_num: int,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    activation: Optional[str],
    cu_seqlens: Optional[torch.Tensor],
    output_final_state: bool,
) -> tuple[int, int, int, Optional[list[int]], bool, str]:
    _require_npu_tensor("x", x)
    _require_npu_tensor("weight", weight)
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "AscendCCausalConv1dFunction supports only FP16/BF16, "
            f"got x.dtype={x.dtype}"
        )
    if weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError(
            "weight must have the same dtype and device as x, "
            f"got x=({x.dtype}, {x.device}), "
            f"weight=({weight.dtype}, {weight.device})"
        )
    if x.ndim not in (2, 3):
        raise ValueError(
            f"x must have shape [B,T,D], [1,T,D], or [T,D], got {tuple(x.shape)}"
        )
    if weight.ndim != 2:
        raise ValueError(
            "weight must use caller layout [D,W], "
            f"got shape={tuple(weight.shape)}"
        )
    if not isinstance(head_num, int) or isinstance(head_num, bool) or head_num <= 0:
        raise ValueError(f"head_num must be a positive int, got {head_num!r}")
    if not isinstance(output_final_state, bool):
        raise TypeError(
            "output_final_state must be bool, "
            f"got {type(output_final_state).__name__}"
        )

    dim = int(x.shape[-1])
    width = int(weight.shape[-1])
    if weight.shape[0] != dim:
        raise ValueError(
            f"weight must have shape [D,W] with D={dim}, got {tuple(weight.shape)}"
        )
    if width not in _SUPPORTED_WIDTHS:
        raise ValueError(
            f"causal-conv kernel width must be one of {_SUPPORTED_WIDTHS}, got {width}"
        )
    if dim % head_num != 0:
        raise ValueError(
            f"x feature dim D={dim} must be divisible by head_num={head_num}"
        )
    head_dim = dim // head_num
    if head_dim % 16 != 0:
        raise ValueError(
            "BNSD/NTD head dimension must be a multiple of 16, "
            f"got D/head_num={head_dim}"
        )
    if x.shape[-2] <= 0:
        raise ValueError("empty token dimensions are not supported")

    offsets = _as_int_list(cu_seqlens)
    is_varlen = offsets is not None
    if is_varlen:
        if x.ndim == 3 and x.shape[0] != 1:
            raise ValueError(
                "varlen mode requires x.shape[0] == 1 for rank-3 input, "
                f"got shape={tuple(x.shape)}"
            )
        total_tokens = int(x.shape[-2])
        assert offsets is not None
        _validate_offsets(offsets, total_tokens)
        num_sequences = len(offsets) - 1
    else:
        if x.ndim != 3:
            raise ValueError(
                "rank-2 x is supported only in varlen mode with cu_seqlens"
            )
        num_sequences = int(x.shape[0])

    for name, optional in (("bias", bias), ("residual", residual)):
        if optional is not None:
            _require_npu_tensor(name, optional)
            if optional.dtype != x.dtype or optional.device != x.device:
                raise ValueError(
                    f"{name} must have the same dtype and device as x"
                )
    if bias is not None and tuple(bias.shape) != (dim,):
        raise ValueError(f"bias must have shape [{dim}], got {tuple(bias.shape)}")
    if residual is not None and tuple(residual.shape) != tuple(x.shape):
        raise ValueError(
            "residual must use the same logical layout and shape as x, "
            f"got residual={tuple(residual.shape)}, x={tuple(x.shape)}"
        )

    state_layout = "none"
    if initial_state is not None:
        _require_npu_tensor("initial_state", initial_state)
        if initial_state.dtype != x.dtype or initial_state.device != x.device:
            raise ValueError(
                "initial_state must have the same dtype and device as x"
            )
        if initial_state.ndim != 3 or initial_state.shape[0] != num_sequences:
            raise ValueError(
                "initial_state must be rank-3 and match the sequence count, "
                f"got shape={tuple(initial_state.shape)}, "
                f"num_sequences={num_sequences}"
            )
        if tuple(initial_state.shape[1:]) == (dim, width):
            state_layout = "NDW"
        elif tuple(initial_state.shape[1:]) == (width, dim):
            state_layout = "NWD"
        else:
            raise ValueError(
                "initial_state must be [N,D,W] or [N,W,D] with "
                f"N={num_sequences}, D={dim}, W={width}; "
                f"got {tuple(initial_state.shape)}"
            )

    _activation_mode(activation)
    return width, dim, num_sequences, offsets, is_varlen, state_layout


def _to_kernel_state(
    initial_state: Optional[torch.Tensor],
    state_layout: str,
) -> Optional[torch.Tensor]:
    if initial_state is None:
        return None
    if state_layout == "NDW":
        return initial_state.transpose(1, 2).contiguous()
    if state_layout == "NWD":
        return initial_state.contiguous()
    raise AssertionError(f"unexpected state layout: {state_layout}")


def _from_kernel_state_gradient(
    dh0: Optional[torch.Tensor],
    state_layout: str,
) -> Optional[torch.Tensor]:
    if dh0 is None or state_layout == "none":
        return None
    if state_layout == "NDW":
        return dh0.transpose(1, 2).contiguous()
    if state_layout == "NWD":
        return dh0.contiguous()
    raise AssertionError(f"unexpected state layout: {state_layout}")


def _flat_x(
    x: torch.Tensor,
    *,
    is_varlen: bool,
) -> torch.Tensor:
    if not is_varlen:
        return x.contiguous()
    return x.reshape(-1, x.shape[-1]).contiguous()


def _flat_to_head_layout(
    x: torch.Tensor,
    head_num: int,
    *,
    is_varlen: bool,
) -> torch.Tensor:
    head_dim = x.shape[-1] // head_num
    if is_varlen:
        flat = x.reshape(-1, x.shape[-1])
        return (
            flat.reshape(flat.shape[0], head_num, head_dim)
            .transpose(0, 1)
            .contiguous()
            .unsqueeze(0)
        )
    return (
        x.reshape(x.shape[0], x.shape[1], head_num, head_dim)
        .transpose(1, 2)
        .contiguous()
    )


def _head_to_flat_layout(
    x: torch.Tensor,
    *,
    is_varlen: bool,
    original_shape: tuple[int, ...],
) -> torch.Tensor:
    if is_varlen:
        head = x.squeeze(0)
        flat = (
            head.transpose(0, 1)
            .reshape(head.shape[1], head.shape[0] * head.shape[-1])
            .contiguous()
        )
        return flat.reshape(original_shape)
    return (
        x.transpose(1, 2)
        .reshape(original_shape)
        .contiguous()
    )


def _final_state_ndw(
    x: torch.Tensor,
    *,
    width: int,
    initial_state_nwd: Optional[torch.Tensor],
    offsets: Optional[list[int]],
) -> torch.Tensor:
    dim = x.shape[-1]
    if offsets is None:
        sequences = [x[index] for index in range(x.shape[0])]
    else:
        flat = x.reshape(-1, dim)
        sequences = [
            flat[left:right]
            for left, right in zip(offsets, offsets[1:])
        ]

    states = []
    for index, sequence in enumerate(sequences):
        history = sequence.transpose(0, 1).contiguous()
        if initial_state_nwd is not None:
            previous = initial_state_nwd[index].transpose(0, 1).contiguous()
            history = torch.cat((previous, history), dim=-1)
        if history.shape[-1] < width:
            history = F.pad(history, (width - history.shape[-1], 0))
        states.append(history[:, -width:])
    return torch.stack(states, dim=0)


class AscendCCausalConv1dFunction(torch.autograd.Function):
    """Autograd binding for AscendC causal-conv forward and backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        head_num: int,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        activation: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        (
            width,
            dim,
            num_sequences,
            offsets,
            is_varlen,
            state_layout,
        ) = _validate_forward_inputs(
            x,
            weight,
            head_num,
            bias,
            residual,
            initial_state,
            activation,
            cu_seqlens,
            output_final_state,
        )
        activation_mode = _activation_mode(activation)
        op_x = _flat_x(x, is_varlen=is_varlen)
        op_weight = weight.transpose(0, 1).contiguous()
        initial_state_nwd = _to_kernel_state(initial_state, state_layout)
        state_len = width - 1
        if initial_state_nwd is None:
            conv_states = torch.zeros(
                num_sequences,
                state_len,
                dim,
                dtype=x.dtype,
                device=x.device,
            )
            initial_state_mode = None
        else:
            conv_states = initial_state_nwd[:, -state_len:, :].contiguous()
            initial_state_mode = [1] * num_sequences

        raw_output = ascendc_causal_conv1d(
            op_x,
            op_weight,
            bias,
            conv_states,
            query_start_loc=offsets,
            cache_indices=None,
            initial_state_mode=initial_state_mode,
            num_accepted_tokens=None,
            activation_mode=0,
            pad_slot_id=-1,
            run_mode=0,
            head_num=head_num,
        )
        preactivation = (
            raw_output.unsqueeze(0) if is_varlen else raw_output
        )
        if residual is not None:
            preactivation = preactivation + _flat_to_head_layout(
                residual,
                head_num,
                is_varlen=is_varlen,
            )
        output = (
            F.silu(preactivation)
            if activation_mode != 0
            else preactivation
        )
        final_state = (
            _final_state_ndw(
                x,
                width=width,
                initial_state_nwd=initial_state_nwd,
                offsets=offsets,
            )
            if output_final_state
            else None
        )

        saved = [x, op_weight, preactivation]
        if initial_state_nwd is not None:
            saved.append(initial_state_nwd)
        ctx.save_for_backward(*saved)
        ctx.has_initial_state = initial_state_nwd is not None
        ctx.has_bias = bias is not None
        ctx.has_residual = residual is not None
        ctx.activation_mode = activation_mode
        ctx.head_num = head_num
        ctx.offsets = offsets
        ctx.is_varlen = is_varlen
        ctx.original_x_shape = tuple(x.shape)
        ctx.state_layout = state_layout
        ctx.output_final_state = output_final_state
        ctx.set_materialize_grads(False)
        return output, final_state

    @staticmethod
    def backward(
        ctx,
        dy: Optional[torch.Tensor],
        dht: Optional[torch.Tensor],
    ):
        saved = list(ctx.saved_tensors)
        x = saved.pop(0)
        op_weight = saved.pop(0)
        preactivation = saved.pop(0)
        initial_state_nwd = saved.pop(0) if ctx.has_initial_state else None
        if dy is None:
            dy = torch.zeros_like(preactivation)
        if tuple(dy.shape) != tuple(preactivation.shape):
            raise ValueError(
                "dy must match the forward output shape, "
                f"got dy={tuple(dy.shape)}, output={tuple(preactivation.shape)}"
            )
        if dy.dtype != x.dtype or dy.device != x.device:
            raise ValueError("dy must have the same dtype and device as x")

        op_x = _flat_x(x, is_varlen=ctx.is_varlen)
        op_dy = dy.squeeze(0).contiguous() if ctx.is_varlen else dy.contiguous()
        op_y = (
            preactivation.squeeze(0).contiguous()
            if ctx.is_varlen
            else preactivation.contiguous()
        )
        dht_nwd = None
        if dht is not None:
            expected = (
                len(ctx.offsets) - 1
                if ctx.offsets is not None
                else x.shape[0]
            )
            expected_shape = (expected, x.shape[-1], op_weight.shape[0])
            if tuple(dht.shape) != expected_shape:
                raise ValueError(
                    "final-state gradient must match [N,D,W], "
                    f"expected={expected_shape}, got={tuple(dht.shape)}"
                )
            if dht.dtype != x.dtype or dht.device != x.device:
                raise ValueError(
                    "final-state gradient must have the same dtype/device as x"
                )
            dht_nwd = dht.transpose(1, 2).contiguous()

        dx, dw, db, dh0 = ascendc_causal_conv1d_bwd(
            x=op_x,
            y=op_y if ctx.activation_mode != 0 else None,
            weight=op_weight,
            dy=op_dy,
            initial_state=initial_state_nwd,
            dht=dht_nwd,
            query_start_loc=ctx.offsets,
            activation=ctx.activation_mode,
            input_layout="NTD" if ctx.is_varlen else "BNSD",
        )
        dx = dx.reshape(ctx.original_x_shape)
        dw = dw.transpose(0, 1).contiguous()
        db = db if ctx.has_bias else None
        dr = None
        if ctx.has_residual:
            if ctx.activation_mode == 0:
                dresidual_head = dy
            else:
                sigmoid = torch.sigmoid(preactivation)
                dresidual_head = (
                    dy
                    * sigmoid
                    * (1.0 + preactivation * (1.0 - sigmoid))
                )
            dr = _head_to_flat_layout(
                dresidual_head,
                is_varlen=ctx.is_varlen,
                original_shape=ctx.original_x_shape,
            )
        dh0 = _from_kernel_state_gradient(dh0, ctx.state_layout)
        # x, weight, head_num, bias, residual, initial_state, activation,
        # cu_seqlens, output_final_state.
        return dx, dw, None, db, dr, dh0, None, None, None


__all__ = ["AscendCCausalConv1dFunction"]
