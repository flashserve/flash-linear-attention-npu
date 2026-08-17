"""Use Ascend C KDA forward behind the Triton-Ascend KDA model API.

The adapter preserves the public and autograd-facing contract of
``triton_ascend_kernels.attention.fla.kda.chunk_kda``. Its low-level KDA
forward and forward L2Norm implementation are replaced. The third-party
Triton backward remains active and consumes the saved tensors returned by
this adapter.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple


_TARGET_MODULES = (
    "triton_ascend_kernels.attention.fla.kda.chunk_fwd",
    "triton_ascend_kernels.attention.fla.kda.chunk",
)
_L2NORM_TARGET_MODULE = "triton_ascend_kernels.attention.fla.kda.chunk"
_EXPECTED_PARAMETERS = {
    "q",
    "k",
    "v",
    "g",
    "beta",
    "scale",
    "initial_state",
    "output_final_state",
    "cu_seqlens",
    "cu_seqlens_cpu",
    "chunk_indices",
    "chunk_size",
    "safe_gate",
    "lower_bound",
    "use_gate_in_kernel",
    "A_log",
    "dt_bias",
    "disable_recompute",
    "return_intermediate_states",
    "transpose_state_layout",
}

_ORIGINALS: Dict[str, Callable[..., Any]] = {}
_L2NORM_ORIGINALS: Dict[str, Callable[..., Any]] = {}


def _debug_synchronize(stage: str) -> None:
    """Synchronize adapter boundaries only for the A5 acceptance probe."""

    if os.environ.get("FLA_NPU_KDA_ADAPTER_DEBUG_SYNC") != "1":
        return
    import torch

    print(json.dumps({"stage": f"{stage}_sync_begin"}), flush=True)
    torch.npu.synchronize()
    print(json.dumps({"stage": f"{stage}_sync_done"}), flush=True)


def _install_triton_extra_ascend_compat() -> bool:
    """Bridge the upstream eager-import name used by the pinned repository."""

    try:
        importlib.import_module("triton.language.extra.ascend.libdevice")
        return False
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if not missing.startswith("triton.language.extra.ascend"):
            raise

    extra = importlib.import_module("triton.language.extra")
    cann = importlib.import_module("triton.language.extra.cann")
    libdevice = importlib.import_module("triton.language.extra.cann.libdevice")
    sys.modules.setdefault("triton.language.extra.ascend", cann)
    sys.modules.setdefault("triton.language.extra.ascend.libdevice", libdevice)
    if not hasattr(extra, "ascend"):
        extra.ascend = cann
    return True


def _host_int_tuple(
    value: Optional[Sequence[int]],
    *,
    cpu_value: Optional[Sequence[int]] = None,
) -> Optional[Tuple[int, ...]]:
    source = cpu_value if cpu_value is not None else value
    if source is None:
        return None
    if hasattr(source, "detach"):
        source = source.detach().cpu().tolist()
    return tuple(int(item) for item in source)


def _head_major_to_sequence_major(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(
            "The Triton-Ascend KDA adapter expects rank-4 BNSD intermediates "
            f"from fla_npu, but received shape {tuple(tensor.shape)}."
        )
    return tensor.permute(0, 2, 1, 3).contiguous()


def _load_ascendc_ops():
    from fla_npu.ops.ascendc import chunk_kda_fwd

    return chunk_kda_fwd


def _load_optimized_l2norm_fwd():
    """Load the packaged fixed-grid L2Norm used by the model adapter."""

    try:
        from fla_npu.ops.triton import l2norm_fwd
    except ModuleNotFoundError as error:
        # Source-tree tests do not materialize setup.py's package_dir mapping.
        if not (error.name or "").startswith(
            "fla_npu.ops.triton.triton_core"
        ):
            raise
        from fla.ops.triton.triton_core.l2norm import l2norm_fwd

    return l2norm_fwd


def triton_ascend_chunk_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
    chunk_size=64,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    disable_recompute=False,
    return_intermediate_states=False,
    transpose_state_layout=False,
):
    """Drop-in replacement for Triton-Ascend ``chunk_kda_fwd``.

    Inputs remain BSND, matching the model-side implementation. Public
    ``fla_npu`` intermediates are head-major, so outputs 2 through 9 are
    converted back to the sequence-major tensors expected by the existing
    Triton backward.
    """

    if q.dim() != 4:
        raise RuntimeError(
            "The Triton-Ascend KDA adapter currently expects BSND rank-4 "
            f"inputs, but q has shape {tuple(q.shape)}."
        )

    chunk_kda_fwd = _load_ascendc_ops()
    host_cu = _host_int_tuple(cu_seqlens, cpu_value=cu_seqlens_cpu)
    if host_cu is not None:
        # Both implementations require canonical sequence-major chunk order.
        # Let the fla_npu host wrapper derive it once from the host metadata.
        host_chunk_indices = None
    else:
        host_chunk_indices = _host_int_tuple(chunk_indices)

    # Both pinned low-level interfaces retain h when recomputation is disabled.
    export_h = bool(return_intermediate_states or disable_recompute)
    outputs = list(
        chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            float(scale),
            int(chunk_size),
            layout="BSND",
            initial_state=initial_state,
            output_final_state=bool(output_final_state),
            cu_seqlens=host_cu,
            chunk_indices=host_chunk_indices,
            safe_gate=bool(safe_gate),
            lower_bound=lower_bound,
            use_gate_in_kernel=bool(use_gate_in_kernel),
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=bool(disable_recompute),
            return_intermediate_states=export_h,
            state_v_first=bool(transpose_state_layout),
        )
    )
    _debug_synchronize("adapter_core")
    if len(outputs) != 12:
        raise RuntimeError(
            f"fla_npu chunk_kda_fwd returned {len(outputs)} values; expected 12."
        )

    # Aqk/Akk and saved backward tensors are BNSD from fla_npu but BSND in the
    # Triton-Ascend autograd contract.
    for index in range(2, 10):
        outputs[index] = _head_major_to_sequence_major(outputs[index])
    _debug_synchronize("adapter_layout_exports")

    if disable_recompute:
        required = {
            "gk": outputs[2],
            "w": outputs[5],
            "u": outputs[6],
            "qg": outputs[7],
            "kg": outputs[8],
            "v_new": outputs[9],
            "h": outputs[10],
        }
        missing = [name for name, tensor in required.items() if tensor is None]
        if missing:
            raise RuntimeError(
                "The Triton-Ascend backward requires saved tensors when "
                f"disable_recompute=True; missing {', '.join(missing)}."
            )

    return tuple(outputs)


def _validate_target(function: Callable[..., Any], module_name: str) -> None:
    parameters = set(inspect.signature(function).parameters)
    missing = sorted(_EXPECTED_PARAMETERS - parameters)
    if missing:
        raise RuntimeError(
            f"{module_name}.chunk_kda_fwd is incompatible with this adapter; "
            f"missing parameters: {', '.join(missing)}."
        )


def install_triton_ascend_kda_adapter() -> bool:
    """Patch the third-party KDA forward in place.

    Returns ``True`` when a new patch was installed and ``False`` when the
    adapter was already active.
    """

    if _ORIGINALS:
        return False

    # Register the packaged OPP path before importing Triton-Ascend. Importing
    # Triton initializes the NPU runtime, after which newly registered custom
    # operator paths are not guaranteed to be visible to dynamic kernels.
    _load_ascendc_ops()
    _install_triton_extra_ascend_compat()
    modules = [importlib.import_module(name) for name in _TARGET_MODULES]
    originals = {}
    for name, module in zip(_TARGET_MODULES, modules):
        original = getattr(module, "chunk_kda_fwd")
        _validate_target(original, name)
        originals[name] = original

    l2norm_module = modules[_TARGET_MODULES.index(_L2NORM_TARGET_MODULE)]
    original_l2norm = getattr(l2norm_module, "l2norm_fwd")
    optimized_l2norm = _load_optimized_l2norm_fwd()

    _ORIGINALS.update(originals)
    _L2NORM_ORIGINALS[_L2NORM_TARGET_MODULE] = original_l2norm
    for module in modules:
        module.chunk_kda_fwd = triton_ascend_chunk_kda_fwd
    l2norm_module.l2norm_fwd = optimized_l2norm
    return True


def remove_triton_ascend_kda_adapter() -> bool:
    """Restore the original third-party forward functions."""

    if not _ORIGINALS:
        return False
    for name, original in tuple(_ORIGINALS.items()):
        module = importlib.import_module(name)
        if getattr(module, "chunk_kda_fwd", None) is triton_ascend_chunk_kda_fwd:
            module.chunk_kda_fwd = original
    for name, original in tuple(_L2NORM_ORIGINALS.items()):
        module = importlib.import_module(name)
        module.l2norm_fwd = original
    _ORIGINALS.clear()
    _L2NORM_ORIGINALS.clear()
    return True


def is_triton_ascend_kda_adapter_installed() -> bool:
    return bool(_ORIGINALS)
