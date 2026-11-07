"""Layout helpers used by the GPU-emulating forward golden."""

from __future__ import annotations

import torch

L2NORM_EPS = 1e-6


def _to_chunks(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """``[B, T, H, ...]`` to ``[B, H, NT, BT, ...]``. ``T`` is already padded."""
    batch, tokens = x.shape[:2]
    if tokens % chunk_size != 0:
        raise ValueError(f"T={tokens} is not divisible by chunk_size={chunk_size}")
    n_chunks = tokens // chunk_size
    y = x.reshape(batch, n_chunks, chunk_size, *x.shape[2:])
    perm = [0, 3, 1, 2, *range(4, y.ndim)]
    return y.permute(*perm).contiguous()


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().sum(dim=-1) + L2NORM_EPS)
    return (x_f * rstd.unsqueeze(-1)).to(x.dtype)


def _apply_qk_beta(
    q: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True.")
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)
    if use_beta_sigmoid_in_kernel:
        scale = 2.0 if allow_neg_eigval else 1.0
        beta = scale * torch.sigmoid(beta.float())
    return q, k, beta
