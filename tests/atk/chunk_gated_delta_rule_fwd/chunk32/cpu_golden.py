"""Helpers shared with the GPU-emulating forward golden."""

from __future__ import annotations

import torch
import torch.nn.functional as F

# fla.ops.utils.constant.RCP_LN2, FP32 hex 0x3FB8AA3B.
RCP_LN2 = 1.4426950216


def _repeat_k_for_gva(k: torch.Tensor, hv: int) -> torch.Tensor:
    """``[B, T, H, K]`` to ``[B, T, HV, K]``."""
    heads = k.shape[2]
    if hv == heads:
        return k
    if hv % heads != 0:
        raise ValueError(f"GVA requires HV % H == 0, got H={heads}, HV={hv}")
    return k.repeat_interleave(hv // heads, dim=2)


def _pad_time(x: torch.Tensor, pad: int, time_dim: int = 1) -> torch.Tensor:
    if pad <= 0:
        return x
    pads = [0, 0] * x.ndim
    idx = x.ndim - 1 - time_dim
    pads[2 * idx + 1] = pad
    return F.pad(x, pads)
