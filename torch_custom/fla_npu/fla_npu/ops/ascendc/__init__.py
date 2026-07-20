# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend C backed FLA NPU operators.

This module provides stable Python import paths backed by Python ctypes aclnn
calls.  Compatibility helpers for legacy torch_npu/torch.ops.npu call sites are
kept opt-in and are not installed during normal import.
"""

from __future__ import annotations

import functools
import inspect
import types
import warnings
from typing import Callable, Optional

from ._aclnn_ctypes import ASCENDC_CTYPES_OPS

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
    "npu_chunk_local_cumsum",
    "npu_chunk_scaled_dot_kkt",
    "npu_solve_tri",
    "npu_chunk_kda_fwd",
    "npu_kda_gate_cumsum",
    "npu_recurrent_kda",
)

BACKWARD_OPS = {
    "fast_gelu_custom": "fast_gelu_custom_backward",
    "npu_fast_gelu_custom": "npu_fast_gelu_custom_backward",
    "causal_conv1d": "causal_conv1d_bwd",
    "npu_causal_conv1d": "npu_causal_conv1d_bwd",
}

# ctypes 直接写 tensor storage 时，PyTorch 无法从 Python 调用自动发现副作用。
# 这里集中声明被修改的参数，由 direct-op wrapper 负责 grad 限制和版本计数。
MUTATED_ARGUMENTS = {
    "causal_conv1d": ("conv_states",),
    "npu_causal_conv1d": ("conv_states",),
}

_LEGACY_TORCH_OPS_WARNING = (
    "torch.ops.npu.{name} is a legacy FLA NPU compatibility API. This call path "
    "depends on the PyTorch/torch_npu dispatcher ABI and will not be supported "
    "in a future fla_npu release. Use fla_npu.ops.ascendc.{public_name}(...) "
    "or the decoupled Ascend C API instead."
)

_DIRECT_RUNTIME_READY = False
_DIRECT_RUNTIME_ERROR: Optional[Exception] = None


def _prepare_direct_runtime(*, raise_on_error: bool = True) -> None:
    """Prepare embedded OPP paths and load custom op_api libraries."""

    global _DIRECT_RUNTIME_ERROR, _DIRECT_RUNTIME_READY
    if _DIRECT_RUNTIME_READY:
        return

    try:
        import fla_npu

        fla_npu.load_ascendc_opapi_libraries()
    except Exception as exc:
        _DIRECT_RUNTIME_ERROR = exc
        if raise_on_error:
            raise RuntimeError(
                "Unable to initialize fla_npu Ascend C op_api libraries. "
                "Please source the CANN set_env.sh before importing "
                "fla_npu.ops.ascendc or calling Ascend C operators."
            ) from exc
    else:
        _DIRECT_RUNTIME_ERROR = None
        _DIRECT_RUNTIME_READY = True


def _torch_npu_namespace():
    import torch

    return torch.ops.npu


def _ensure_legacy_torch_ops_loaded() -> None:
    import fla_npu

    is_loaded = getattr(fla_npu, "is_legacy_torch_ops_loaded", lambda: False)
    if not is_loaded():
        fla_npu.load_legacy_torch_ops()


def _get_torch_op(name: str):
    namespace = _torch_npu_namespace()
    if not hasattr(namespace, name):
        _ensure_legacy_torch_ops_loaded()
        namespace = _torch_npu_namespace()
    if not hasattr(namespace, name):
        raise AttributeError(
            f"torch.ops.npu.{name} is not registered. Call "
            "fla_npu.load_legacy_torch_ops() first if you need the legacy "
            "torch.ops.npu compatibility path."
        )
    return _unwrap_legacy_torch_op(getattr(namespace, name))


@functools.lru_cache(maxsize=None)
def _get_direct_op(name: str):
    _prepare_direct_runtime()
    try:
        op = ASCENDC_CTYPES_OPS[name]
    except KeyError as exc:
        raise AttributeError(f"fla_npu.ops.ascendc has no ctypes Ascend C op {name}.") from exc
    return _wrap_mutable_direct_op(name, op)


def _wrap_mutable_direct_op(name: str, op: Callable) -> Callable:
    mutated_names = MUTATED_ARGUMENTS.get(name, ())
    if not mutated_names:
        return op

    signature = inspect.signature(op)

    @functools.wraps(op)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        mutated_tensors = [bound.arguments[arg_name] for arg_name in mutated_names]

        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Mutable Ascend C operators require the torch Python runtime.") from exc

        mutated_tensors = [tensor for tensor in mutated_tensors if isinstance(tensor, torch.Tensor)]
        requiring_grad = [
            arg_name for arg_name in mutated_names if getattr(bound.arguments[arg_name], "requires_grad", False)
        ]
        if requiring_grad:
            names = ", ".join(requiring_grad)
            raise RuntimeError(
                f"{name} mutates {names} in place through ctypes. Mutable state tensors "
                "must not require gradients; use a functional state API for training."
            )

        result = op(*args, **kwargs)
        if mutated_tensors:
            torch.autograd.graph.increment_version(mutated_tensors)
        return result

    return wrapper


def _warn_legacy_torch_op(name: str) -> None:
    warnings.warn(
        _LEGACY_TORCH_OPS_WARNING.format(
            name=name,
            public_name=_strip_npu_prefix(name),
        ),
        FutureWarning,
        stacklevel=3,
    )


def _unwrap_legacy_torch_op(op):
    return getattr(op, "_fla_npu_original_op", op)


class _LegacyTorchOpOverloadWarningWrapper:
    _fla_npu_legacy_warning_wrapper = True

    def __init__(self, name: str, overload):
        self._fla_npu_name = name
        self._fla_npu_original_op = overload

    def __call__(self, *args, **kwargs):
        _warn_legacy_torch_op(self._fla_npu_name)
        return self._fla_npu_original_op(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._fla_npu_original_op, name)

    def __repr__(self) -> str:
        return repr(self._fla_npu_original_op)


class _LegacyTorchOpWarningWrapper:
    _fla_npu_legacy_warning_wrapper = True

    def __init__(self, name: str, op):
        self._fla_npu_name = name
        self._fla_npu_original_op = op
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = getattr(op, "__doc__", None)

    def __call__(self, *args, **kwargs):
        _warn_legacy_torch_op(self._fla_npu_name)
        return self._fla_npu_original_op(*args, **kwargs)

    def __getattr__(self, name: str):
        value = getattr(self._fla_npu_original_op, name)
        if callable(value):
            return _LegacyTorchOpOverloadWarningWrapper(self._fla_npu_name, value)
        return value

    def __repr__(self) -> str:
        return repr(self._fla_npu_original_op)


def _make_raw_wrapper(name: str) -> Callable:
    @functools.wraps(_get_direct_op)
    def wrapper(*args, **kwargs):
        return _get_direct_op(name)(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = f"Call the direct Ascend C binding for {name}."
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
                return _get_direct_op("npu_fast_gelu_custom")(self)

            @staticmethod
            def backward(ctx, grad):
                (self,) = ctx.saved_tensors
                return _get_direct_op("npu_fast_gelu_custom_backward")(grad, self)

        return Function.apply(input_tensor)


def fast_gelu_custom(input_tensor):
    """FastGELU with automatic binding to its custom backward operator."""

    if _has_tensor_requiring_grad(input_tensor):
        return _FastGeluCustomFunction.apply(input_tensor)
    return _get_direct_op("npu_fast_gelu_custom")(input_tensor)


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

    ``conv_states`` is mutable state in every mode and must not require gradients.
    Decode/speculative modes are left on the raw op path; prefill only binds
    gradients for ``x``, ``weight`` and ``bias``.
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
        return _get_direct_op("npu_causal_conv1d")(
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
            y = _get_direct_op("npu_causal_conv1d")(
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
            ctx.activation_mode = activation_mode
            ctx.save_for_backward(*tensors)
            return y

        @staticmethod
        def backward(ctx, grad):
            saved = list(ctx.saved_tensors)
            x_ = saved.pop(0)
            weight_ = saved.pop(0)
            bias_ = saved.pop(0) if ctx.has_bias else None
            dx, dw, db, _ = _get_direct_op("npu_causal_conv1d_bwd")(
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


def install_legacy_torch_ops_warning() -> None:
    """Warn when users call legacy ``torch.ops.npu`` FLA NPU operators."""

    namespace = _torch_npu_namespace()
    for name in _ASCENDC_OPS:
        if not hasattr(namespace, name):
            continue
        current = getattr(namespace, name)
        if getattr(current, "_fla_npu_legacy_warning_wrapper", False):
            continue
        setattr(namespace, name, _LegacyTorchOpWarningWrapper(name, current))


for _name in _ASCENDC_OPS:
    globals()[_name] = _make_raw_wrapper(_name)
    globals()[_strip_npu_prefix(_name)] = globals()[_name]

globals()["fast_gelu_custom"] = fast_gelu_custom
globals()["causal_conv1d"] = causal_conv1d

_prepare_direct_runtime(raise_on_error=False)

__all__ = [
    "BACKWARD_OPS",
    "MUTATED_ARGUMENTS",
    "install_legacy_torch_ops_warning",
    "install_torch_npu_ops_compat",
    *sorted(set(_ASCENDC_OPS)),
    *sorted({_strip_npu_prefix(name) for name in _ASCENDC_OPS}),
]
