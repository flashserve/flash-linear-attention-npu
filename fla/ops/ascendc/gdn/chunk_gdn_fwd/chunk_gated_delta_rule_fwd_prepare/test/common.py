"""Input helpers for local GDN prepare golden / cases."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from layout import convert_dict_from_fla, normalize_layout, seqlens_to_cu
else:
    from .layout import convert_dict_from_fla, normalize_layout, seqlens_to_cu


def generate_inputs(
    *,
    batch: int,
    seq_len: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k: int,
    head_v: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    use_qk_l2norm_in_kernel: bool = True,
    layout: str = "bsnd",
    seqlens: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Build inputs in the requested layout.

    Fixed BSND is ``[B, T, H, D]``; BNSD is ``[B, H, T, D]``.
    ``seqlens`` packs sequences on T (FLA varlen, ``B=1``).

    When ``use_qk_l2norm_in_kernel`` is False, ``k`` is L2-normalized along the
    last dim. The kernel will not normalize it again.
    """
    layout = normalize_layout(layout)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    cu = None
    if seqlens is not None:
        batch = 1
        seq_len = sum(seqlens)
        cu = seqlens_to_cu(seqlens, device="cpu", dtype=torch.long)
    elif layout in ("tnd", "ntd"):
        if batch != 1:
            raise ValueError(
                f"{layout.upper()} is a packed layout; use batch=1 or pass seqlens"
            )

    q = torch.randn(batch, seq_len, num_k_heads, head_k, generator=gen, dtype=torch.float32)
    k = torch.randn(batch, seq_len, num_k_heads, head_k, generator=gen, dtype=torch.float32)
    if not use_qk_l2norm_in_kernel:
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
    v = torch.randn(batch, seq_len, num_v_heads, head_v, generator=gen, dtype=torch.float32)
    if use_gate_in_kernel:
        gate = torch.randn(batch, seq_len, num_v_heads, generator=gen, dtype=torch.float32)
        a_log = torch.randn(num_v_heads, generator=gen, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(num_v_heads, generator=gen, dtype=torch.float32) * 0.1
    else:
        gate = torch.log(torch.rand(batch, seq_len, num_v_heads, generator=gen).clamp_min(1e-4))
        a_log = None
        dt_bias = None
    if use_beta_sigmoid_in_kernel:
        beta = torch.randn(batch, seq_len, num_v_heads, generator=gen, dtype=torch.float32)
    else:
        beta = torch.rand(batch, seq_len, num_v_heads, generator=gen, dtype=torch.float32)

    fla = {
        "q": q,
        "k": k,
        "v": v,
        "g": gate,
        "beta": beta,
    }
    out = convert_dict_from_fla(fla, layout)
    for name in out:
        out[name] = out[name].to(device=device, dtype=dtype)
    if a_log is not None:
        out["A_log"] = a_log.to(device=device, dtype=dtype)
        out["dt_bias"] = dt_bias.to(device=device, dtype=dtype)
    if cu is not None:
        out["cu_seqlens"] = cu.to(device=device)
    return out
