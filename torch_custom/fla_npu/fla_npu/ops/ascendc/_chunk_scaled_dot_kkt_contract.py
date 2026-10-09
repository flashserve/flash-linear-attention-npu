"""Input contract shared by the ctypes and Stable-ABI KKT entry points."""


def validate(k, g, beta, cu_seqlens, chunk_indices, chunk_size):
    import torch

    op = "npu_chunk_scaled_dot_kkt"
    for name, tensor, rank, layout in (
        ("k", k, 4, "[B,Hk,T,K]"),
        ("g", g, 3, "[B,Hv,T]"),
        ("beta", beta, 3, "[B,Hv,T]"),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{op}: {name} must be a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.ndim != rank:
            raise ValueError(f"{op}: {name} must have shape {layout}, got {tuple(tensor.shape)}.")

    if k.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"{op}: k must use float16 or bfloat16, got {k.dtype}.")
    for name, tensor in (("g", g), ("beta", beta)):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{op}: {name} must use float32, got {tensor.dtype}.")

    B, Hk, T, K = (int(dim) for dim in k.shape)
    _, Hv, _ = (int(dim) for dim in g.shape)
    if any(dim <= 0 or dim > 2**31 - 1 for dim in (B, Hk, T, K)) or K < 8:
        raise ValueError(f"{op}: k requires positive B/Hk/T, 8 <= K <= 2^31-1, "
                         f"and every dimension <= 2^31-1; got {tuple(k.shape)}.")
    if tuple(g.shape) != (B, Hv, T) or Hv <= 0 or Hv > 2**31 - 1:
        raise ValueError(f"{op}: g must have shape [B,Hv,T] with positive Hv, "
                         f"matching k B/T; got {tuple(g.shape)} for k {tuple(k.shape)}.")
    if tuple(beta.shape) != (B, Hv, T):
        raise ValueError(f"{op}: beta must match g shape {(B, Hv, T)}, got {tuple(beta.shape)}.")
    if Hv % Hk != 0:
        raise ValueError(f"{op}: Hv must be divisible by Hk, got Hv={Hv}, Hk={Hk}.")
    if not isinstance(chunk_size, int) or chunk_size not in (16, 32, 64, 128):
        raise ValueError(f"{op}: chunk_size must be one of 16, 32, 64, 128, got {chunk_size!r}.")
    if (cu_seqlens is None) != (chunk_indices is None):
        raise ValueError(f"{op}: cu_seqlens and chunk_indices must be provided together.")
    if cu_seqlens is not None:
        if len(cu_seqlens) < 2:
            raise ValueError(f"{op}: cu_seqlens must have at least 2 elements.")
        if len(chunk_indices) == 0 or len(chunk_indices) % 2:
            raise ValueError(f"{op}: chunk_indices must contain non-empty [seq, chunk] pairs.")
    return B, Hv, T
