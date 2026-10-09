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
    return implementation == "optimized" or q_rstd is not None or not disable_recompute


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
    import torch
    import torch_npu
    import math
    from ._runtime import (
        optional_bool as _optional_bool, optional_float as _optional_float,
        ACL_FORMAT_ND, ACL_FORMAT_NCHW, ACL_FORMAT_NCDHW, ACL_FORMAT_NCL,
    )

    args = dict(args)
    for name, default in (("disable_recompute",True),("safe_gate",True),
                          ("use_gate_in_kernel",False),("use_exp2",True),("state_v_first",False)):
        args[name] = _optional_bool(args[name],default)
    args["lower_bound"] = _optional_float(args["lower_bound"],-5.0)
    q = args["q"]
    if q.device.type != "npu" or "Ascend950" not in torch.npu.get_device_name(q.device):
        raise ValueError("optimized KDA backward requires Ascend950/A5")
    packed = args["cu_seqlens"] is not None
    shape = tuple(q.shape)
    if len(shape) != (3 if packed else 4):
        raise ValueError("optimized expects dense [B,H,T,D] or packed [H,T,D]")
    b,h,t,d = (1,*shape) if packed else shape
    if min(b,h,t) <= 0 or d != 128:
        raise ValueError("optimized requires positive B/H/T and K=V=128; T=0 is not supported")
    if not math.isfinite(float(args["scale"])):
        raise ValueError("scale must be finite")
    if (int(args["chunk_size"]) != 64 or not args["safe_gate"] or
            not args["use_gate_in_kernel"] or not args["use_exp2"] or args["state_v_first"]):
        raise ValueError("optimized requires C=64, safe gate, gate-in-kernel, exp2 and K-first states")
    if args["initial_state"] is not None or args["dht"] is not None:
        raise ValueError("optimized initial_state/dht are not supported")
    if not args["disable_recompute"] and (h > 256 or h % 8):
        raise ValueError("optimized recompute currently requires H<=256 and H divisible by 8")
    cu, indices, nc = _canonical_kda_bwd_metadata(args["cu_seqlens"], args["chunk_indices"], t)
    token = shape
    scalar = shape[:-1]
    state = (nc,h,128,128) if packed else (b,nc,h,128,128)

    def check(name, expected, dtypes, optional=False):
        x = args[name]
        if optional and x is None:
            return
        if x is None or tuple(x.shape) != expected or x.dtype not in dtypes:
            raise ValueError(f"optimized {name}: expected shape {expected}, dtype {dtypes}")
        if x.device != q.device or not x.is_contiguous():
            raise ValueError(f"optimized {name}: expected contiguous tensor on q.device")
        standard_formats = {ACL_FORMAT_ND, ACL_FORMAT_NCHW,
                            ACL_FORMAT_NCDHW, ACL_FORMAT_NCL}
        if int(torch_npu.get_npu_format(x)) not in standard_formats:
            raise ValueError(f"optimized {name}: private NPU storage formats are not supported")

    bf16 = (torch.bfloat16,)
    fp32 = (torch.float32,)
    mixed = (torch.bfloat16,torch.float32)
    for name in ("q","k","v","d_o"):
        check(name,token,bf16)
    check("beta",scalar,mixed)
    for name in ("Aqk","Akk"):
        check(name,(*scalar,64),bf16)
    check("raw_g",token,mixed)
    check("A_log",(h,),mixed)
    for name in ("q_rstd","k_rstd"):
        check(name,scalar,fp32,optional=True)
    saved = ("gk","w","qg","kg","v_new","h")
    if args["disable_recompute"]:
        for name in saved:
            check(name,state if name == "h" else token,fp32 if name == "gk" else bf16)
    elif any(args[name] is not None for name in saved):
        raise ValueError("recompute mode requires gk/w/qg/kg/v_new/h to be None")
    bias = args["dt_bias"]
    if bias is not None:
        if bias.dtype != torch.float32 or bias.numel() != h*128 or not bias.is_contiguous() or bias.device != q.device:
            raise ValueError("dt_bias must be contiguous FP32 with H*128 elements on q.device")
        bias = bias.view(h,128)

    args.update(dt_bias=bias, cu_seqlens=cu, chunk_indices=indices)
    return args
