"""Shared KDA host policy for forward outputs and optimized backward inputs."""

from __future__ import annotations

from typing import Tuple


FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT = "0f0f0c97af39343855b43bbbaddcedfda5cb9d77"
FLA_ORG_KDA_FWD_ALIGNMENT_SOURCE = (
    "https://github.com/fla-org/flash-linear-attention/blob/"
    f"{FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT}/fla/ops/kda/chunk_fwd.py"
)


def kda_fwd_optional_output_mask(
    *,
    output_final_state: bool,
    use_gate_in_kernel: bool,
    disable_recompute: bool,
    return_intermediate_states: bool,
) -> Tuple[bool, ...]:
    """Return the visibility mask for the low-level 12-value FLA interface."""

    return (
        True,
        output_final_state,
        not use_gate_in_kernel or disable_recompute,
        True,
        True,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute or return_intermediate_states,
        True,
    )


def _select_kda_bwd_optimized(implementation, q_rstd, k_rstd, disable_recompute):
    if implementation not in ("auto", "legacy", "optimized"):
        raise ValueError("implementation must be auto, legacy or optimized")
    if (q_rstd is None) != (k_rstd is None):
        raise ValueError("q_rstd and k_rstd must be supplied together")
    if implementation == "legacy":
        if q_rstd is not None:
            raise ValueError("legacy does not support L2Norm backward")
        return False
    return True


def _canonical_kda_bwd_metadata(cu, indices, total_tokens, chunk_size=64):
    """Remove empty sequences without changing token or global chunk storage."""
    import operator

    def host_list(value):
        if hasattr(value, "device") and value.device.type != "cpu":
            raise ValueError("optimized metadata must be Host-resident; device readback is not implicit")
        if hasattr(value,"detach"):
            value=value.detach().flatten().tolist()
        return tuple(operator.index(x) for x in value)

    def chunk_pairs(boundaries):
        return tuple(value for seq,(a,b) in enumerate(zip(boundaries,boundaries[1:]))
                     for chunk in range((b-a+chunk_size-1)//chunk_size) for value in (seq,chunk))

    if cu is None:
        if indices is not None:
            raise ValueError("chunk_indices requires cu_seqlens")
        return None, None, (total_tokens + chunk_size - 1) // chunk_size
    cu = host_list(cu)
    if len(cu) < 2 or cu[0] != 0 or cu[-1] != total_tokens or any(a > b for a,b in zip(cu,cu[1:])):
        raise ValueError("invalid cu_seqlens")
    original = chunk_pairs(cu)
    if indices is not None and host_list(indices) != original:
        raise ValueError("chunk_indices must use canonical sequence-major order")
    compact = (cu[0],) + tuple(b for a,b in zip(cu,cu[1:]) if b > a)
    indices = chunk_pairs(compact)
    return compact, indices, len(indices) // 2


def _prepare_kda_bwd_optimized(args):
    """A5 V2 shares the single-layout contract, with its existing shape limits."""
    import torch

    _validate_kda_bwd_a5_options(args)
    a = _prepare_kda_bwd_arch22(args)
    q = a["q"]
    index = q.device.index if q.device.index is not None else torch.npu.current_device()
    if q.device.type != "npu" or "Ascend950" not in torch.npu.get_device_name(index):
        raise ValueError("optimized KDA backward requires Ascend950/A5")
    if not a["safe_gate"] or not a["use_gate_in_kernel"]:
        raise ValueError("A5 V2 requires safe_gate=True and use_gate_in_kernel=True")
    if not -5.0 <= a["lower_bound"] < 0.0:
        raise ValueError("A5 V2 requires -5<=lower_bound<0")
    h = a["_output_gate_shape"][-2]
    if not a["disable_recompute"] and (h > 256 or h % 8):
        raise ValueError("optimized recompute currently requires H<=256 and H divisible by 8")
    return a


def _is_kda_bwd_arch22(q):
    """Select the A2/A3 composition before applying A5 implementation policy."""
    import torch

    if q.device.type != "npu":
        return False
    index = q.device.index
    if index is None:
        index = torch.npu.current_device()
    name = str(torch.npu.get_device_name(index))
    return name.startswith(("Ascend910B", "Ascend910C", "Ascend910_93"))


def _validate_kda_recompute_layout(input_layout):
    """Reserved attribute; it does not change the rank-4 tensor contract."""
    if input_layout not in ("BNSD", "NTD"):
        raise ValueError("Recompute input_layout must be BNSD or NTD")
    return input_layout


def _validate_kda_bwd_a5_options(args):
    from ._parameters import optional_bool

    if args["implementation"] not in ("auto", "optimized"):
        raise ValueError("A5 single-layout backward supports auto/optimized; legacy is not supported")
    if (args.get("beta_eff") is not None or
            optional_bool(args["use_beta_sigmoid_in_kernel"], False) or
            optional_bool(args["allow_neg_eigval"], False)):
        raise ValueError("A5 beta sigmoid backward is reserved: beta_eff=None and both flags False required")


def _prepare_kda_bwd_arch22(args):
    """Shared single-layout tensor contract; A5 adds V2 eligibility checks."""
    import math
    import torch
    import torch_npu
    from ._parameters import (
        optional_bool, optional_float, ACL_FORMAT_ND, ACL_FORMAT_NCHW,
        ACL_FORMAT_NCDHW, ACL_FORMAT_NCL,
    )

    a = dict(args)
    if a["implementation"] not in ("auto", "legacy", "optimized"):
        raise ValueError("implementation must be auto, legacy or optimized")
    for name, default in (("disable_recompute", True), ("safe_gate", True),
                          ("use_gate_in_kernel", True), ("use_exp2", True),
                          ("state_v_first", False), ("use_beta_sigmoid_in_kernel", False),
                          ("allow_neg_eigval", False)):
        a[name] = optional_bool(a[name], default)
    a["lower_bound"] = optional_float(a["lower_bound"], -5.0)
    a["scale"] = float(a["scale"])
    a["chunk_size"] = int(a["chunk_size"])
    if not math.isfinite(a["scale"]) or not math.isfinite(a["lower_bound"]):
        raise ValueError("scale and lower_bound must be finite")
    if a["chunk_size"] != 64 or not a["use_exp2"] or a["state_v_first"]:
        raise ValueError("KDA backward requires C=64, exp2 and K-first state")
    if a["initial_state"] is not None or a["dht"] is not None:
        raise ValueError("initial_state/dht are not supported")
    if (a["q_rstd"] is None) != (a["k_rstd"] is None):
        raise ValueError("q_rstd and k_rstd must be supplied together")
    if a["allow_neg_eigval"] and not a["use_beta_sigmoid_in_kernel"]:
        raise ValueError("allow_neg_eigval requires use_beta_sigmoid_in_kernel")

    variable = a["cu_seqlens"] is not None
    layout = a["layout"]
    if layout not in ("BNSD", "BSND", "NTD", "TND"):
        raise ValueError("layout must be BNSD, BSND, NTD or TND")
    packed = layout in ("NTD", "TND")
    shape = tuple(a["v"].shape)
    if len(shape) != (3 if packed else 4):
        raise ValueError("layout does not match Token rank")
    if packed and not variable:
        raise ValueError("3D calls require cu_seqlens")
    if variable and not packed and shape[0] != 1:
        raise ValueError("4D varlen calls require B=1")
    tm = layout in ("BSND", "TND")
    b = 1 if packed else shape[0]
    first = 0 if packed else 1
    h, t = (shape[first + 1], shape[first]) if tm else shape[first:first + 2]
    if min(b, h, t) <= 0 or shape[-1] != 128:
        raise ValueError("KDA backward requires positive B/H/T and K=V=128")
    head = (h, t, 128) if packed else (b, h, t, 128)
    gate_token = (t, h, 128) if packed else (b, t, h, 128)
    scalar = head[:-1]
    matrix = (*scalar, 64)
    cu, indices, nc = _canonical_kda_bwd_metadata(a["cu_seqlens"], a["chunk_indices"], t)
    state = (nc, h, 128, 128) if packed else (b, nc, h, 128, 128)
    formats = {ACL_FORMAT_ND, ACL_FORMAT_NCHW, ACL_FORMAT_NCDHW, ACL_FORMAT_NCL}
    bf16, fp32 = (torch.bfloat16,), (torch.float32,)
    mixed = (torch.bfloat16, torch.float32)

    def check(name, expected, dtypes, optional=False):
        x = a[name]
        if optional and x is None:
            return
        if x is None or tuple(x.shape) != expected or x.dtype not in dtypes:
            raise ValueError(f"{name}: expected shape {expected}, dtype {dtypes}")
        if x.device != a["q"].device or not x.is_contiguous():
            raise ValueError(f"{name} must be contiguous on q.device")
        if int(torch_npu.get_npu_format(x)) not in formats:
            raise ValueError(f"{name}: private NPU storage formats are not supported")

    for name in ("q", "k"):
        check(name, head if a["q_rstd"] is not None else shape, bf16)
    check("v", shape, bf16)
    check("d_o", gate_token, bf16)
    check("beta", gate_token[:-1], mixed)
    if a["use_beta_sigmoid_in_kernel"]:
        check("beta_eff", scalar, fp32)
    elif a.get("beta_eff") is not None:
        raise ValueError("beta_eff requires use_beta_sigmoid_in_kernel=True")
    for name in ("Aqk", "Akk"):
        check(name, matrix, bf16)
    for name in ("q_rstd", "k_rstd"):
        check(name, scalar, fp32, optional=True)
    saved = ("gk", "w", "qg", "kg", "v_new", "h")
    if a["disable_recompute"]:
        for name in saved:
            check(name, state if name == "h" else head, fp32 if name == "gk" else bf16)
    elif any(a[name] is not None for name in saved):
        raise ValueError("recompute mode requires gk/w/qg/kg/v_new/h to be None")
    if a["use_gate_in_kernel"]:
        check("raw_g", gate_token, mixed)
        check("A_log", (h,), fp32)
    elif any(a[name] is not None for name in ("raw_g", "A_log", "dt_bias")):
        raise ValueError("raw_g/A_log/dt_bias require use_gate_in_kernel=True")
    bias = a["dt_bias"]
    if bias is not None:
        if (bias.dtype != torch.float32 or bias.numel() != h * 128
                or not bias.is_contiguous() or bias.device != a["q"].device):
            raise ValueError("dt_bias must be contiguous FP32 with H*K elements on q.device")
        a["dt_bias"] = bias.view(h, 128)
    a.update(cu_seqlens=cu, chunk_indices=indices,
             _output_token_shape=shape, _output_gate_shape=gate_token,
             _output_beta_shape=gate_token[:-1])
    return a
