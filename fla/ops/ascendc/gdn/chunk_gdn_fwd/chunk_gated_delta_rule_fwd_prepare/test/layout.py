"""Layout conversion between FLA token-first and NPU BSND / BNSD / TND / NTD.

FLA compute layout is always ``[B, T, H, ...]``. Variable length uses packed
``B=1`` plus ``cu_seqlens`` of shape ``[N+1]``.

NPU-side names (same as SolveTri / KDA tests):

- BSND  ``[B, T, H, D]``     fixed, or packed varlen ``[1, total_T, H, D]``
- BNSD  ``[B, H, T, D]``     BSND transpose; chunk contiguous on T
- TND   ``[total_T, H, D]``  varlen, chunk not contiguous in H
- NTD   ``[H, total_T, D]``  varlen, TND transpose, chunk contiguous in T
"""

from __future__ import annotations

import torch

LAYOUTS = ("bsnd", "bnsd", "tnd", "ntd")

# tensors that follow [B, T, H, ...] (or [B, T, H] for gates / rstd)
_LAYOUT_KEYS = {
    "q", "k", "v", "g", "beta",
    "q_hat", "k_hat", "q_rstd", "k_rstd",
    "beta_out", "g_cumsum", "w", "u", "A",
}


def normalize_layout(layout: str) -> str:
    name = layout.strip().lower()
    if name not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    return name


def parse_layouts(text: str | None) -> list[str]:
    """Comma-separated layouts, or ``all``."""
    if text is None or str(text).strip() == "":
        return ["bsnd"]
    raw = str(text).strip().lower()
    if raw == "all":
        return list(LAYOUTS)
    return [normalize_layout(part) for part in raw.split(",") if part.strip()]


def parse_seqlens(text: str | None) -> list[int] | None:
    if text is None or str(text).strip() == "":
        return None
    lens = [int(x) for x in str(text).replace(" ", "").split(",") if x]
    if not lens or any(n <= 0 for n in lens):
        raise ValueError(f"seqlens must be positive ints, got {text!r}")
    return lens


def seqlens_to_cu(seqlens: list[int], device=None, dtype=torch.long) -> torch.Tensor:
    cu = torch.zeros(len(seqlens) + 1, dtype=dtype, device=device)
    cu[1:] = torch.tensor(seqlens, dtype=dtype, device=device).cumsum(0)
    return cu


def packed_time_dim(x: torch.Tensor, layout: str) -> int:
    layout = normalize_layout(layout)
    if layout == "bsnd":
        return int(x.shape[1])
    if layout == "bnsd":
        return int(x.shape[2])
    if layout == "tnd":
        return int(x.shape[0])
    return int(x.shape[1])  # ntd: [H, T, ...]


def layout_skip_reason(layout: str, batch: int, varlen: bool) -> str | None:
    layout = normalize_layout(layout)
    if layout in ("tnd", "ntd") and batch != 1 and not varlen:
        return f"{layout.upper()} requires B=1 or packed varlen"
    return None


def to_fla(x: torch.Tensor, layout: str) -> torch.Tensor:
    """User layout → FLA ``[B, T, H, ...]`` (B=1 for TND/NTD)."""
    layout = normalize_layout(layout)
    if layout == "bsnd":
        return x
    if layout == "bnsd":
        # [B, H, T, ...] -> [B, T, H, ...]
        return x.transpose(1, 2).contiguous()
    if layout == "tnd":
        return x.unsqueeze(0).contiguous()
    # ntd: [H, T, ...] -> [1, T, H, ...]
    return x.transpose(0, 1).unsqueeze(0).contiguous()


def from_fla(x: torch.Tensor | None, layout: str) -> torch.Tensor | None:
    """FLA ``[B, T, H, ...]`` → user layout."""
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


def convert_dict_to_fla(tensors: dict, layout: str) -> dict:
    out = {}
    for key, value in tensors.items():
        if key in _LAYOUT_KEYS and isinstance(value, torch.Tensor):
            out[key] = to_fla(value, layout)
        else:
            out[key] = value
    return out


def convert_dict_from_fla(tensors: dict, layout: str) -> dict:
    out = {}
    for key, value in tensors.items():
        if key in _LAYOUT_KEYS and isinstance(value, torch.Tensor):
            out[key] = from_fla(value, layout)
        else:
            out[key] = value
    return out


def check_cu_seqlens(x_fla: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
    if x_fla.shape[0] != 1:
        raise ValueError(
            f"varlen requires packed B=1, got B={x_fla.shape[0]}. "
            "Flatten sequences on T, or use --layout tnd/ntd."
        )
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must be [N+1], got {tuple(cu_seqlens.shape)}")
    total = int(cu_seqlens[-1])
    if total != x_fla.shape[1]:
        raise ValueError(
            f"cu_seqlens[-1]={total} does not match packed T={x_fla.shape[1]}"
        )
    if int(cu_seqlens[0]) != 0:
        raise ValueError("cu_seqlens[0] must be 0")
    diffs = cu_seqlens[1:] - cu_seqlens[:-1]
    if (diffs <= 0).any():
        raise ValueError("cu_seqlens must be strictly increasing")
