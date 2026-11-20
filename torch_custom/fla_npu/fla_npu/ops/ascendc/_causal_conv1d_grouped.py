"""Bias-free causal convolutions on independent dim-last feature tensors."""

from operator import index

import torch
import torch.nn.functional as F


class _TrainingConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, boundaries):
        from . import causal_conv1d_fn

        preactivation = causal_conv1d_fn(
            x, weight, None, query_start_loc_cpu=boundaries, activation=None
        )
        ctx.save_for_backward(x, weight, preactivation)
        ctx.boundaries = boundaries
        return F.silu(preactivation)

    @staticmethod
    def backward(ctx, grad):
        from . import causal_conv1d_bwd

        x, weight, preactivation = ctx.saved_tensors
        activation = (
            getattr(causal_conv1d_bwd, "SILU_BF16_GRAD", None)
            if x.dtype == torch.bfloat16
            else None
        )
        if activation is None:
            grad = torch.ops.aten.silu_backward(grad, preactivation)
            preactivation = None
            activation = 0
        dx, dw, _, _ = causal_conv1d_bwd(
            x,
            preactivation,
            weight,
            grad.contiguous(),
            query_start_loc=ctx.boundaries,
            activation=activation,
            input_layout="TND",
        )
        return dx, dw, None


def causal_conv1d_grouped(inputs, weights, cu_seqlens=None):
    """Run independent bias-free causal convolutions followed by SiLU.

    Each [T, C] tensor has its own [W, C] filter. Channel counts and widths can
    differ across groups. The groups share token positions and document
    boundaries. Inputs and weights must share a device. Each pair follows the
    dtype, width and channel-alignment contract of causal_conv1d_fn. The SiLU
    gradient is rounded to the input dtype before convolution backward.
    """
    if not inputs or len(inputs) != len(weights):
        raise ValueError("Expected an input and a filter for each convolution")
    first = inputs[0]
    if first.ndim != 2:
        raise ValueError("Expected [T, C] input tensors")
    tokens = first.size(0)
    if tokens <= 0:
        raise ValueError("Expected at least one input token")
    for x, weight in zip(inputs, weights):
        if (
            x.ndim != 2
            or weight.ndim != 2
            or x.size(0) != tokens
            or x.size(1) != weight.size(1)
            or x.device != first.device
            or weight.device != x.device
            or x.dtype != weight.dtype
        ):
            raise ValueError("Expected matching [T, C], [W, C] tensors on one device")
    if cu_seqlens is None:
        boundaries = (0, tokens)
    elif isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected one-dimensional integer document offsets")
        boundaries = tuple(cu_seqlens.tolist())
    else:
        boundaries = tuple(index(offset) for offset in cu_seqlens)
    if (
        len(boundaries) < 2
        or boundaries[0] != 0
        or boundaries[-1] != tokens
        or any(b < a for a, b in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("Offsets must start at zero, end at T and be nondecreasing")
    return tuple(
        _TrainingConvolution.apply(x.contiguous(), weight.contiguous(), boundaries)
        for x, weight in zip(inputs, weights)
    )
