"""Ascend C backed FLA NPU operators.

The raw custom operators are registered under ``torch.ops.npu`` by
``import fla_npu``.  This module provides stable Python import paths and a
compatibility shim for older ``torch_npu.ops`` call sites.
"""

from __future__ import annotations

import functools
import types
from typing import Callable

_ASCENDC_OPS = (
    "npu_fast_gelu_custom",
    "npu_fast_gelu_custom_backward",
    "npu_causal_conv1d",
    "npu_causal_conv1d_bwd",
    "npu_prepare_wy_repr_bwd_full",
    "npu_chunk_gated_delta_rule_bwd_dhu",
    "npu_chunk_bwd_dv_local",
    "npu_prepare_wy_repr_bwd_da",
    "npu_chunk_bwd_dqkwg",
    "npu_chunk_fwd_o",
    "npu_chunk_gated_delta_rule_fwd_h",
    "npu_recompute_w_u_fwd",
    "npu_solve_tri",
)

BACKWARD_OPS = {
    "fast_gelu_custom": "fast_gelu_custom_backward",
    "npu_fast_gelu_custom": "npu_fast_gelu_custom_backward",
    "causal_conv1d": "causal_conv1d_bwd",
    "npu_causal_conv1d": "npu_causal_conv1d_bwd",
}


def _torch_npu_namespace():
    import torch

    return torch.ops.npu


def _get_torch_op(name: str):
    namespace = _torch_npu_namespace()
    if not hasattr(namespace, name):
        raise AttributeError(
            f"torch.ops.npu.{name} is not registered. Import fla_npu first and "
            "make sure the custom extension loaded successfully."
        )
    return getattr(namespace, name)


def _make_raw_wrapper(name: str) -> Callable:
    @functools.wraps(_get_torch_op)
    def wrapper(*args, **kwargs):
        return _get_torch_op(name)(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = f"Call torch.ops.npu.{name}."
    return wrapper


def _strip_npu_prefix(name: str) -> str:
    return name[4:] if name.startswith("npu_") else name


def _has_tensor_requiring_grad(*values) -> bool:
    try:
        import torch
    except Exception:
        return False

    for value in values:
        if isinstance(value, torch.Tensor) and value.requires_grad:
            return True
    return False


class _FastGeluCustomFunction:
    @staticmethod
    def apply(input_tensor):
        import torch

        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, self):
                ctx.save_for_backward(self)
                return _get_torch_op("npu_fast_gelu_custom")(self)

            @staticmethod
            def backward(ctx, grad):
                (self,) = ctx.saved_tensors
                return _get_torch_op("npu_fast_gelu_custom_backward")(grad, self)

        return Function.apply(input_tensor)


def fast_gelu_custom(input_tensor):
    """FastGELU with automatic binding to its custom backward operator."""

    if _has_tensor_requiring_grad(input_tensor):
        return _FastGeluCustomFunction.apply(input_tensor)
    return _get_torch_op("npu_fast_gelu_custom")(input_tensor)


def causal_conv1d(
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
    """Causal conv1d with automatic backward binding for prefill mode.

    Decode/speculative modes mutate cache state and are left on the raw op path.
    """

    can_bind_backward = (
        run_mode == 0
        and activation_mode == 0
        and query_start_loc is None
        and cache_indices is None
        and initial_state_mode is None
        and num_accepted_tokens is None
        and _has_tensor_requiring_grad(x, weight, bias)
    )
    if not can_bind_backward:
        return _get_torch_op("npu_causal_conv1d")(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=initial_state_mode,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=activation_mode,
            pad_slot_id=pad_slot_id,
            run_mode=run_mode,
            head_num=head_num,
        )

    import torch

    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_, weight_, bias_, conv_states_):
            y = _get_torch_op("npu_causal_conv1d")(
                x=x_,
                weight=weight_,
                bias=bias_,
                conv_states=conv_states_,
                query_start_loc=query_start_loc,
                cache_indices=None,
                initial_state_mode=None,
                num_accepted_tokens=None,
                activation_mode=activation_mode,
                pad_slot_id=pad_slot_id,
                run_mode=run_mode,
                head_num=head_num,
            )
            tensors = [x_, weight_]
            ctx.has_bias = bias_ is not None
            if bias_ is not None:
                tensors.append(bias_)
            ctx.save_for_backward(*tensors)
            return y

        @staticmethod
        def backward(ctx, grad):
            saved = list(ctx.saved_tensors)
            x_ = saved.pop(0)
            weight_ = saved.pop(0)
            bias_ = saved.pop(0) if ctx.has_bias else None
            dx, dw, db, _ = _get_torch_op("npu_causal_conv1d_bwd")(
                x=x_,
                y=None if ctx.activation_mode == 0 else None,
                weight=weight_,
                dy=grad,
                initial_state=None,
                dht=None,
                query_start_loc=None,
                activation=0,
                input_layout="BSH",
            )
            return dx, dw, (db if bias_ is not None else None), None

    return Function.apply(x, weight, bias, conv_states)


def install_torch_npu_ops_compat() -> None:
    """Expose wrappers through the legacy ``torch_npu.ops`` namespace."""

    try:
        import torch_npu
    except Exception:
        return

    ops = getattr(torch_npu, "ops", None)
    if ops is None:
        ops = types.SimpleNamespace()
        setattr(torch_npu, "ops", ops)

    for name in _ASCENDC_OPS:
        setattr(ops, name, globals()[name])
        setattr(ops, _strip_npu_prefix(name), globals()[_strip_npu_prefix(name)])


for _name in _ASCENDC_OPS:
    globals()[_name] = _make_raw_wrapper(_name)
    globals()[_strip_npu_prefix(_name)] = globals()[_name]

globals()["fast_gelu_custom"] = fast_gelu_custom
globals()["causal_conv1d"] = causal_conv1d

__all__ = [
    "BACKWARD_OPS",
    "install_torch_npu_ops_compat",
    *sorted(set(_ASCENDC_OPS)),
    *sorted({_strip_npu_prefix(name) for name in _ASCENDC_OPS}),
]
