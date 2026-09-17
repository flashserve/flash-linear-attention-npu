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
import os
import sys
import types
import warnings
from typing import Callable, Optional

from ._aclnn_ctypes import ASCENDC_CTYPES_OPS

_ASCENDC_OPS = (
    "npu_fast_gelu_custom",
    "npu_fast_gelu_custom_backward",
    "npu_causal_conv1d",
    "npu_causal_conv1d_fn",
    "npu_causal_conv1d_update",
    "npu_causal_conv1d_bwd",
    "npu_prepare_wy_repr_bwd_full",
    "npu_prepare_wy_repr_bwd",
    "npu_chunk_gated_delta_rule_bwd_dhu",
    "npu_chunk_bwd_dv_local",
    "npu_prepare_wy_repr_bwd_da",
    "npu_chunk_bwd_dqkwg",
    "npu_chunk_fwd_o",
    "npu_chunk_gated_delta_rule_fwd_h",
    "npu_recompute_w_u_fwd",
    "npu_recurrent_gated_delta_rule",
    "npu_chunk_local_cumsum",
    "npu_chunk_scaled_dot_kkt",
    "npu_solve_tri",
    "npu_chunk_kda_fwd",
    "npu_chunk_kda_bwd_intra",
    "npu_kda_gate_cumsum",
    "npu_recurrent_kda",
)

# Operators the launcher carries that the ctypes reference does not define.
# Landing here is deliberate, never accidental: with no same-kernel reference to
# compare against, the operator's parity scenario has to bring its own (a torch
# implementation, or recorded golden tensors).  stable_coverage.py fails on an
# operator missing from ctypes *without* being listed here, and on one listed
# here that ctypes still defines, so the two lists cannot drift apart.
_LAUNCHER_ONLY_OPS: tuple[str, ...] = ()

BACKWARD_OPS = {
    "fast_gelu_custom": "fast_gelu_custom_backward",
    "npu_fast_gelu_custom": "npu_fast_gelu_custom_backward",
    "causal_conv1d": "causal_conv1d_bwd",
    "causal_conv1d_fn": "causal_conv1d_bwd",
    "npu_causal_conv1d": "npu_causal_conv1d_bwd",
    "npu_causal_conv1d_fn": "npu_causal_conv1d_bwd",
}

# ctypes 直接写 tensor storage 时，PyTorch 无法从 Python 调用自动发现副作用。
# 这里集中声明被修改的参数，由 direct-op wrapper 负责 grad 限制和版本计数。
MUTATED_ARGUMENTS = {
    "causal_conv1d": ("conv_states",),
    "causal_conv1d_fn": ("conv_states",),
    "causal_conv1d_update": ("conv_state",),
    "npu_causal_conv1d": ("conv_states",),
    "npu_causal_conv1d_fn": ("conv_states",),
    "npu_causal_conv1d_update": ("conv_state",),
    "npu_recurrent_kda": ("initial_state",),
    "recurrent_gated_delta_rule": ("state",),
    "npu_recurrent_gated_delta_rule": ("state",),
}

# Some mutable operators only write the state tensor for some argument values,
# and the value comes straight from the caller's own arguments.  Declaring that
# flag as ``(argument_name, default)`` lets the mutation wrapper read it from
# the positional/keyword arguments directly, instead of running
# ``inspect.signature().bind(...)`` (with ``apply_defaults``) on every call.
# Only operators whose "did this call mutate?" answer is not derivable from a
# single argument need the older ``MUTATION_PREDICATES`` lambda escape hatch.
MUTATION_FLAGS = {
    # Ascend950/910b: `inplace_final_state=False` makes the operator write into
    # a scratch state and return it, leaving the caller's tensor untouched.
    "npu_recurrent_kda": ("inplace_final_state", True),
}

# Escape hatch for mutation conditions that need more than one argument.
MUTATION_PREDICATES: dict[str, Callable[[dict], bool]] = {}

_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _mutation_plan(name: str, signature: inspect.Signature):
    """Precompute how to decide the mutation contract for one operator.

    Declared flags are validated against the real signature here, once, so a
    wrong default fails loudly at first use instead of silently skipping (or
    adding) a version bump on the hot path.

    Called once per wrapped operator.  It is deliberately *not* cached by
    signature: hashing a 20-argument ``inspect.Signature`` costs ~30us, which
    would land on every call and dwarf everything this avoids.

    Returns ``(mutated_args, flag)``, each entry ``(name, position, default)``:
    ``position`` is the index a positional call puts the argument at, or None
    for an argument that can only be passed by keyword, and ``default`` is what
    the signature supplies when the call omits it.  Reading a call this way
    replaces ``Signature.bind(..., apply_defaults=True)``, which cost ~12us on
    a decode operator's fifteen parameters and ran on every call: a decode step
    reaches these wrappers a few dozen times.
    """

    positional_names = [
        parameter.name for parameter in signature.parameters.values()
        if parameter.kind in _POSITIONAL_KINDS
    ]

    def reader(argument: str):
        parameter = signature.parameters.get(argument)
        if parameter is None:
            raise RuntimeError(
                f"{name}: {argument!r} is not an argument of this operator")
        position = None
        if parameter.kind in _POSITIONAL_KINDS:
            position = positional_names.index(parameter.name)
        default = parameter.default
        return (argument, position,
                None if default is inspect.Parameter.empty else default)

    mutated_args = tuple(reader(argument)
                         for argument in MUTATED_ARGUMENTS.get(name, ()))

    flag = None
    declared = MUTATION_FLAGS.get(name)
    if declared is not None:
        flag_name, flag_default = declared
        parameter = signature.parameters.get(flag_name)
        if parameter is None:
            raise RuntimeError(
                f"{name}: MUTATION_FLAGS names unknown argument {flag_name!r}")
        if (parameter.default is not inspect.Parameter.empty
                and bool(parameter.default) != bool(flag_default)):
            raise RuntimeError(
                f"{name}: MUTATION_FLAGS[{flag_name!r}] default "
                f"{flag_default!r} disagrees with the operator signature "
                f"default {parameter.default!r}")
        flag = reader(flag_name)
    return mutated_args, flag


def _argument_value(reader, args, kwargs):
    """One declared argument's value, with the signature's default filled in."""

    name, position, default = reader
    if position is not None and len(args) > position:
        return args[position]
    if name in kwargs:
        return kwargs[name]
    return default


def _resolve_mutation(plan, signature, predicate, args, kwargs):
    """Return ``(mutated_tensors, used_fast_path)`` for one call.

    The mutated tensors and the declared flag are read straight out of the
    caller's arguments, in every call form (positional, keyword or mixed), so a
    decode step never pays for a full ``signature.bind``.  The bind remains the
    reference for the operators that still need a lambda predicate, which is
    the only case that has to see every argument's default at once.
    """

    mutated_args, flag = plan
    if flag is not None and not _argument_value(flag, args, kwargs):
        return [], True
    if predicate is None:
        return ([_argument_value(reader, args, kwargs)
                 for reader in mutated_args], True)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    if not predicate(bound.arguments):
        return [], False
    return [bound.arguments[argument] for argument, _, _ in mutated_args], False

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
    stable_op = _get_stable_op(name)
    if stable_op is not None:
        _note_backend(name, "stable")
        # A backend that applies the in-place contract in its own frame is
        # returned as it is.  Wrapping it here would bump the version counter a
        # second time for one call, and the wrapper costs ~8us on the decode
        # path (`_stable`'s four hot wrappers declare the contract; the ctypes
        # reference never does).
        if getattr(stable_op, "_fla_npu_inplace_contract", False):
            return stable_op
        return _wrap_mutable_direct_op(name, stable_op)
    try:
        op = ASCENDC_CTYPES_OPS[name]
    except KeyError as exc:
        if name in _LAUNCHER_ONLY_OPS:
            raise AttributeError(
                f"{name} is carried only by the Stable-ABI launcher, and this "
                "install has none (a wheel built with "
                "FLA_NPU_BUILD_STABLE_ABI=0 ships no launcher).") from exc
        raise AttributeError(f"fla_npu.ops.ascendc has no ctypes Ascend C op {name}.") from exc
    if _validate_requested():
        _note_backend(name, "ctypes", "FLA_NPU_STABLE_VALIDATE=1")
    elif _stable_backend_selected():
        reason = ("the stable launcher is unusable"
                  if _LAUNCHER_FAILURE is not None
                  else "not carried by the stable launcher")
        _note_backend(name, "ctypes", reason)
    else:
        _note_backend(name, "ctypes")
    return _wrap_mutable_direct_op(name, op)


def _abi_mode() -> str:
    return (os.environ.get("FLA_NPU_STABLE_ABI") or "").strip().lower()


# Which backend serves each operator, and why anything fell back.  Recorded at
# resolution time (once per operator, not per call) so both a diagnosis and the
# CI assertion "no fallback inside the legal domain" can read it.
BACKENDS: dict[str, str] = {}
FALLBACKS: dict[str, int] = {}


def _trace_enabled() -> bool:
    value = os.environ.get("FLA_NPU_STABLE_TRACE")
    return value is not None and value.upper() in {"1", "TRUE", "YES", "ON"}


def _note_backend(name: str, backend: str, reason: str | None = None) -> None:
    """Record (and optionally report) which backend answers *name*.

    ``FLA_NPU_STABLE_TRACE=1`` turns this into a per-operator line on stderr.  The
    interesting case is a fallback: the launcher exists but does not carry the
    operator, so ctypes answers instead -- that changes the dependency
    footprint, and it must never happen silently.
    """

    BACKENDS[name] = backend
    if reason is not None:
        FALLBACKS[name] = FALLBACKS.get(name, 0) + 1
    if _trace_enabled():
        detail = f" ({reason})" if reason else ""
        print(f"[fla-npu] {name}: {backend}{detail}", file=sys.stderr)


# The launcher failing to load is not an error the package can repair, and every
# operator would repeat it, so it is reported once per process and remembered
# for the backend record.
_LAUNCHER_FAILURE: BaseException | None = None


def _note_launcher_unusable(exc: BaseException) -> None:
    """Report once why the launcher is not answering, then let ctypes serve.

    Degrading to the reference keeps the results correct, which is exactly why
    this needs to be loud: a wheel whose launcher cannot load still installs and
    still passes accuracy tests, it only loses the host-side speedup.  Warning
    once (not per operator, not per call) keeps the diagnosis close to the cause
    without spamming a decode loop.
    """

    global _LAUNCHER_FAILURE
    if _LAUNCHER_FAILURE is not None:
        return
    _LAUNCHER_FAILURE = exc
    warnings.warn(
        f"fla_npu: the Stable-ABI launcher is unusable in this environment, so "
        f"every operator falls back to the ctypes reference. Results stay "
        f"correct but the host-side speedup is lost. Cause: {exc}",
        RuntimeWarning, stacklevel=3)


def _validate_requested() -> bool:
    """Whether the caller asked for full input validation.

    The launcher checks what is free (the dispatcher schema) and otherwise lets
    the operator report its own illegal-domain errors, which is what keeps the
    hot path cheap.  FLA_NPU_STABLE_VALIDATE=1 switches to the ctypes reference
    for the whole call: it performs the full Python validation and drives the
    same kernel, so results stay bit-identical while illegal inputs produce a
    precise message instead of an aclnn status.  It is a diagnosis switch, not
    a performance mode.
    """

    value = os.environ.get("FLA_NPU_STABLE_VALIDATE")
    return value is not None and value.upper() in {"1", "TRUE", "YES", "ON"}


def _stable_backend_selected() -> bool:
    """Whether the ABI-free backend should be tried first.

    Default order is stable -> ctypes: the stable launcher carries neither the
    CPython ABI nor the libtorch C++ ABI, so it is the only backend that keeps a
    wheel usable across Python and torch versions.  ``FLA_NPU_STABLE_ABI=ctypes``
    forces the reference path.
    """

    # `ctypes` has to mean ctypes: the flag used to be ignored for the shipped
    # backend, so FLA_NPU_STABLE_ABI=ctypes still picked stable and the
    # documented "force the reference path" escape hatch did nothing.
    if _validate_requested():
        return False
    return _abi_mode() != "ctypes"


def _get_stable_op(name: str):
    """Return the Stable-ABI backend entry for *name*, else None.

    ``FLA_NPU_STABLE_ABI`` selects the backend: unset / ``stable`` uses
    ``libfla_npu_stable.so`` via torch.ops (falling back to ctypes for anything it
    does not carry), and ``ctypes`` forces the Python reference path.  An
    operator declared in ``_LAUNCHER_ONLY_OPS`` has no reference to fall back to,
    so it stays on the launcher even when validation was requested.
    """

    if not _stable_backend_selected() and name not in _LAUNCHER_ONLY_OPS:
        return None
    try:
        from . import _stable
    except Exception as exc:
        _note_launcher_unusable(exc)
        return None
    try:
        # load() raises with the cause -- an aoti_torch_* symbol this torch does
        # not export, a stamp that does not match this glue, a library built for
        # another platform.  available() below turns all of those into a bare
        # False, which is how the fallback to ctypes used to happen silently.
        _stable.load()
    except Exception as exc:
        _note_launcher_unusable(exc)
        return None
    if not _stable.available():
        # Loaded, but the registration this glue addresses is missing: a
        # launcher built from another revision, not an operator it lacks.
        _note_launcher_unusable(RuntimeError(
            "the launcher registered no ops this package can address"))
        return None
    return getattr(_stable, name, None)


# Both of these are module handles and plain functions, so caching them is
# safe; the wrapper used to `import torch` and walk
# `torch.autograd.graph.increment_version` on every call, and a decode step
# reaches these wrappers a few dozen times.
_TORCH = None
_INCREMENT_VERSION = None


def _torch_runtime():
    """The torch module, imported once."""

    global _TORCH
    if _TORCH is None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "Mutable Ascend C operators require the torch Python runtime."
            ) from exc
        _TORCH = torch
    return _TORCH


def _increment_version():
    """``torch.autograd.graph.increment_version``, resolved once."""

    global _INCREMENT_VERSION
    if _INCREMENT_VERSION is None:
        _INCREMENT_VERSION = _torch_runtime().autograd.graph.increment_version
    return _INCREMENT_VERSION


def _wrap_mutable_direct_op(name: str, op: Callable) -> Callable:
    mutated_names = MUTATED_ARGUMENTS.get(name, ())
    if not mutated_names:
        return op

    signature = inspect.signature(op)
    # Validate and precompute the mutation plan once, at wrap time.
    plan = _mutation_plan(name, signature)
    predicate = MUTATION_PREDICATES.get(name)

    @functools.wraps(op)
    def wrapper(*args, **kwargs):
        torch = _torch_runtime()

        raw_mutated, _used_fast_path = _resolve_mutation(
            plan, signature, predicate, args, kwargs)

        mutated_tensors = [
            tensor for tensor in raw_mutated if isinstance(tensor, torch.Tensor)
        ]
        for tensor in mutated_tensors:
            if tensor.requires_grad:
                raise RuntimeError(
                    f"{name} mutates state tensors in place. Mutable state tensors "
                    "must not require gradients; use a functional state API for training."
                )

        result = op(*args, **kwargs)
        if mutated_tensors:
            _increment_version()(mutated_tensors)
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
    # Bound on the first call, then called directly.  `_get_direct_op` is
    # lru_cached and records which backend answers this operator, so resolving
    # once keeps that bookkeeping identical while taking a Python call and a
    # cache lookup off every invocation of a hot operator.
    bound: list[Callable] = []

    @functools.wraps(_get_direct_op)
    def wrapper(*args, **kwargs):
        if not bound:
            bound.append(_get_direct_op(name))
            # The backend is fixed for the life of the process, so hand the
            # module-level names straight to it: this frame plus the
            # *args/**kwargs re-expansion is ~3us of the host time per call,
            # measured in probes/wrapper_mystery.py (M8 vs M9), on a decode
            # path that walks these operators some thirty times a step.
            resolved = bound[0]
            globals()[name] = resolved
            public_name = _strip_npu_prefix(name)
            if globals().get(public_name) is wrapper:
                globals()[public_name] = resolved
        return bound[0](*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = f"Call the direct Ascend C binding for {name}."
    # The public signature comes from whichever backend carries the operator:
    # the ctypes reference when it exists, otherwise the launcher's own
    # wrapper.  Without the fallback a launcher-only operator would expose
    # (*args, **kwargs), which is what help() and every introspection tool
    # would then report.
    source = ASCENDC_CTYPES_OPS.get(name)
    if source is None:
        try:
            from . import _stable

            source = getattr(_stable, name, None)
        except Exception:
            source = None
    if source is not None:
        wrapper.__signature__ = inspect.signature(source)
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
    """Deprecated causal conv1d API kept for source compatibility.

    This preserves the pre-``bd55f7c`` signature and automatic backward binding
    for the supported prefill path. ``conv_states`` is mutable state in every
    mode and must not require gradients.
    """

    warnings.warn(
        "fla_npu.ops.ascendc.causal_conv1d is a deprecated compatibility API "
        "and will be removed in 2027/02. Use causal_conv1d_fn or "
        "causal_conv1d_update instead.",
        FutureWarning,
        stacklevel=2,
    )
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
    globals().setdefault(_strip_npu_prefix(_name), globals()[_name])

_prepare_direct_runtime(raise_on_error=False)

__all__ = [
    "BACKENDS",
    "BACKWARD_OPS",
    "FALLBACKS",
    "MUTATED_ARGUMENTS",
    "MUTATION_FLAGS",
    "install_legacy_torch_ops_warning",
    "install_torch_npu_ops_compat",
    *sorted(set(_ASCENDC_OPS)),
    *sorted({_strip_npu_prefix(name) for name in _ASCENDC_OPS}),
]
