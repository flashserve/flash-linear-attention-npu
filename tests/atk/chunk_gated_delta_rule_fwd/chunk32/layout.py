"""FLA ``[B, T, H, ...]`` versus the rank-4 head-major tensors this suite builds."""

from __future__ import annotations

import torch

LAYOUTS = ("bsnd", "bnsd", "tnd", "ntd")


def normalize_layout(layout: str) -> str:
    name = layout.strip().lower()
    if name not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    return name


def to_fla(x: torch.Tensor, layout: str) -> torch.Tensor:
    """User layout to FLA ``[B, T, H, ...]``."""
    layout = normalize_layout(layout)
    if layout == "bsnd":
        return x
    if layout == "bnsd":
        return x.transpose(1, 2).contiguous()
    if layout == "tnd":
        return x.unsqueeze(0).contiguous()
    return x.transpose(0, 1).unsqueeze(0).contiguous()


def from_fla(x: torch.Tensor | None, layout: str) -> torch.Tensor | None:
    """FLA ``[B, T, H, ...]`` back to the user layout."""
    if x is None:
        return None
    layout = normalize_layout(layout)
    if layout == "bsnd":
        return x.contiguous()
    if layout == "bnsd":
        return x.transpose(1, 2).contiguous()
    if x.shape[0] != 1:
        raise ValueError(f"{layout} requires packed batch size 1, got B={x.shape[0]}")
    packed = x.squeeze(0)
    if layout == "tnd":
        return packed.contiguous()
    return packed.transpose(0, 1).contiguous()


def check_cu_seqlens(x_fla: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
    if x_fla.shape[0] != 1:
        raise ValueError(f"varlen requires packed B=1, got B={x_fla.shape[0]}")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must be [N+1], got {tuple(cu_seqlens.shape)}")
    total = int(cu_seqlens[-1])
    if total != x_fla.shape[1]:
        raise ValueError(f"cu_seqlens[-1]={total} does not match packed T={x_fla.shape[1]}")
    if int(cu_seqlens[0]) != 0:
        raise ValueError("cu_seqlens[0] must be 0")
    diffs = cu_seqlens[1:] - cu_seqlens[:-1]
    if (diffs <= 0).any():
        raise ValueError("cu_seqlens must be strictly increasing")
