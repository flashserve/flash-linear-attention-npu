"""Independent CPU golden for ``gdn_recoverssm_verify``.

The implementation follows the frozen recurrence one token at a time in
float64. It intentionally does not import the NPU runtime or kernel sources.
"""

from __future__ import annotations

import math

import torch


QK_L2_NORM_EPS = 1.0e-6


def _require_cpu(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got {tensor.device}")


def _require_layout(name: str, tensor: torch.Tensor) -> None:
    if any(step <= 0 for step in tensor.stride()):
        raise ValueError(f"{name} must have positive strides")


def _logical(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()


def _softplus(x: torch.Tensor, beta: float, threshold: float) -> torch.Tensor:
    scaled = beta * x
    return torch.where(
        scaled <= threshold,
        torch.log1p(torch.exp(scaled.clamp_max(80.0))) / beta,
        x,
    )


def gdn_recoverssm_verify_reference(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replay_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(out, replay_cache_after)`` using FP64 CPU arithmetic."""

    tensors = {
        "A_log": A_log,
        "a": a,
        "b": b,
        "dt_bias": dt_bias,
        "q": q,
        "k": k,
        "v": v,
        "checkpoint_state": checkpoint_state,
        "replay_cache": replay_cache,
        "query_start_loc": query_start_loc,
        "state_indices": state_indices,
    }
    for name, tensor in tensors.items():
        _require_cpu(name, tensor)
        _require_layout(name, tensor)

    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape:
        raise ValueError("q and k must have matching [1,T,HK,K] shapes")
    total_tokens, hk, key_dim = q.shape[1:]
    if v.ndim != 4 or v.shape[:2] != (1, total_tokens):
        raise ValueError("v must have shape [1,T,HV,V]")
    hv, value_dim = v.shape[2:]
    if hk <= 0 or hv % hk != 0 or hv // hk > 16:
        raise ValueError("HV/HK must be an integer in [1,16]")
    if not (1 <= key_dim <= 128 and 1 <= value_dim <= 128):
        raise ValueError("K and V must be in [1,128]")
    spec_query_len = replay_cache.shape[2]
    if not 1 <= spec_query_len <= 32:
        raise ValueError("replay_cache window must be in [1,32]")
    if A_log.shape != (hv,) or dt_bias.shape != (hv,):
        raise ValueError("A_log and dt_bias must be [HV]")
    if a.numel() != total_tokens * hv or b.numel() != total_tokens * hv:
        raise ValueError("a and b must contain T*HV elements")
    if checkpoint_state.ndim != 4 or checkpoint_state.shape[1:] != (
        hv,
        value_dim,
        key_dim,
    ):
        raise ValueError("checkpoint_state must be [P,HV,V,K]")
    if replay_cache.ndim != 4 or replay_cache.shape[1:] != (
        hv,
        spec_query_len,
        value_dim + key_dim + 1,
    ):
        raise ValueError("replay_cache must be [P,HV,L,V+K+1]")
    if query_start_loc.ndim != 1 or query_start_loc.numel() < 1:
        raise ValueError("query_start_loc must be a non-empty vector")
    rows = state_indices.numel()
    if state_indices.ndim != 1 or query_start_loc.numel() != rows + 1:
        raise ValueError("query_start_loc must have B+1 entries")
    if not math.isfinite(float(scale if scale is not None else key_dim**-0.5)):
        raise ValueError("scale must be finite")
    if not math.isfinite(softplus_beta) or softplus_beta <= 0:
        raise ValueError("softplus_beta must be finite and positive")
    if not math.isfinite(softplus_threshold):
        raise ValueError("softplus_threshold must be finite")

    offsets = [int(value) for value in query_start_loc.tolist()]
    pages = [int(value) for value in state_indices.tolist()]
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError("query_start_loc must start at 0 and end at T")
    if any(offsets[index] > offsets[index + 1] for index in range(rows)):
        raise ValueError("query_start_loc must be nondecreasing")
    if any(offsets[row + 1] - offsets[row] > spec_query_len for row in range(rows)):
        raise ValueError("a request span exceeds spec_query_len")
    positive_pages = [page for page in pages if page > 0]
    if any(page >= replay_cache.shape[0] for page in positive_pages):
        raise ValueError("positive state_indices must be in range")
    if len(set(positive_pages)) != len(positive_pages):
        raise ValueError("positive state_indices must be unique")

    a64 = _logical(a).reshape(total_tokens, hv)
    b64 = _logical(b).reshape(total_tokens, hv)
    alog64 = _logical(A_log)
    bias64 = _logical(dt_bias)
    q64 = _logical(q)[0]
    k64 = _logical(k)[0]
    v64 = _logical(v)[0]
    state64 = _logical(checkpoint_state)
    replay64 = _logical(replay_cache)

    if use_qk_l2norm_in_kernel:
        q64 = q64 * torch.rsqrt(q64.square().sum(dim=-1, keepdim=True) + QK_L2_NORM_EPS)
        k64 = k64 * torch.rsqrt(k64.square().sum(dim=-1, keepdim=True) + QK_L2_NORM_EPS)
    query_scale = key_dim**-0.5 if scale is None else float(scale)
    q64 = q64 * query_scale

    ratio = hv // hk
    q64 = q64.repeat_interleave(ratio, dim=1)
    k64 = k64.repeat_interleave(ratio, dim=1)
    x = a64 + bias64
    decay = torch.exp(-torch.exp(alog64) * _softplus(x, softplus_beta, softplus_threshold))
    beta = torch.sigmoid(b64)

    out64 = torch.zeros_like(v64)
    for row, page in enumerate(pages):
        if page <= 0:
            continue
        state = state64[page].clone()
        for token in range(offsets[row], offsets[row + 1]):
            key = k64[token]
            state = state * decay[token, :, None, None]
            correction = beta[token, :, None] * (
                v64[token] - torch.einsum("hvk,hk->hv", state, key)
            )
            state = state + correction[:, :, None] * key[:, None, :]
            out64[token] = torch.einsum("hvk,hk->hv", state, q64[token])
            position = token - offsets[row]
            replay64[page, :, position, :value_dim] = correction
            replay64[page, :, position, value_dim : value_dim + key_dim] = key
            replay64[page, :, position, value_dim + key_dim] = decay[token]

    return out64.unsqueeze(0), replay64


__all__ = ["gdn_recoverssm_verify_reference"]
