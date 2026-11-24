"""SHAPE-ONLY PLACEHOLDER: not a numerical KDA backward reference.

The public legacy implementation returns FP32 dq/dk/db/dg, unlike A5 V2.
"""
import torch


def chunk_kda_bwd_cpu(**values):
    q, k, v, beta = (values[name] for name in ("q", "k", "v", "beta"))
    heads = int(q.shape[0 if q.ndim == 3 else 1])
    zero = lambda shape, dtype: torch.zeros(tuple(shape), dtype=dtype, device="cpu")
    return (zero(q.shape, torch.float32), zero(k.shape, torch.float32),
            zero(v.shape, v.dtype), zero(beta.shape, torch.float32),
            zero(q.shape, torch.float32), None,
            zero((heads,), torch.float32),
            None if values["dt_bias"] is None else zero((heads, 128), torch.float32))
