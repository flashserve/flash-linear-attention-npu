"""CPU reference from FLAN 05b530df (tests/atk/chunk_gated_delta_rule_bwd/executor_chunk_gated_delta_rule_bwd.py)."""

import math
from typing import Any

import torch

from gdn_atk import gate_values, packed_metadata


CHUNK_SIZE = 64


DIM = 128


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _randn(shape, seed: int, scale: float, dtype: torch.dtype, device: torch.device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.randn(tuple(shape), generator=generator, dtype=torch.float32) * float(scale)
    return value.to(dtype).to(device)


def _rand(shape, seed: int, low: float, high: float, dtype: torch.dtype, device: torch.device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.rand(tuple(shape), generator=generator, dtype=torch.float32)
    value = value * (float(high) - float(low)) + float(low)
    return value.to(dtype).to(device)


def build_inputs(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch, key_heads, value_heads, tokens = (
        int(spec[name]) for name in ("B", "HK", "HV", "T")
    )
    g_dtype = (
        torch.float32
        if spec.get("g_dtype", spec.get("scalar_dtype", "fp32")) == "fp32"
        else torch.bfloat16
    )
    beta_dtype = (
        torch.float32
        if spec.get("beta_dtype", spec.get("scalar_dtype", "fp32")) == "fp32"
        else torch.bfloat16
    )
    seed = int(spec.get("seed", 20260911))
    seqlens, cu_seqlens, chunk_indices = packed_metadata(spec)

    q = _randn((batch, key_heads, tokens, DIM), seed + 1, 0.12, torch.bfloat16, device)
    k = _randn((batch, key_heads, tokens, DIM), seed + 2, 0.12, torch.bfloat16, device)
    v = _randn((batch, value_heads, tokens, DIM), seed + 3, 0.05, torch.bfloat16, device)
    g = gate_values(batch, value_heads, tokens, seqlens, seed + 4, g_dtype, device)
    use_beta_sigmoid = _as_bool(spec.get("use_beta_sigmoid", False))
    beta_raw = (
        _randn((batch, value_heads, tokens), seed + 5, 0.5, beta_dtype, device)
        if use_beta_sigmoid
        else None
    )
    beta = torch.sigmoid(beta_raw.float()).to(beta_dtype) if beta_raw is not None else _rand(
        (batch, value_heads, tokens), seed + 5, 0.1, 0.9, beta_dtype, device
    )
    a = _randn((batch, value_heads, tokens, CHUNK_SIZE), seed + 6, 0.02, torch.bfloat16, device)
    d_o = _randn((batch, value_heads, tokens, DIM), seed + 7, 0.12, torch.bfloat16, device)

    use_qk_l2norm = _as_bool(spec.get("use_qk_l2norm", False))
    q_rstd = torch.rsqrt((q.float() * q.float()).sum(-1) + 1.0e-6) if use_qk_l2norm else None
    k_rstd = torch.rsqrt((k.float() * k.float()).sum(-1) + 1.0e-6) if use_qk_l2norm else None

    sequences = batch if seqlens is None else len(seqlens)
    with_state = _as_bool(spec.get("with_state", False))
    with_dht = _as_bool(spec.get("with_dht", False))
    initial_state = (
        _randn((sequences, value_heads, DIM, DIM), seed + 8, 0.02, torch.bfloat16, device)
        if with_state
        else None
    )
    dht = (
        _randn((sequences, value_heads, DIM, DIM), seed + 9, 0.02, torch.bfloat16, device)
        if with_dht
        else None
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A": a,
        "d_o": d_o,
        "q_rstd": q_rstd,
        "k_rstd": k_rstd,
        "beta_raw": beta_raw,
        "initial_state": initial_state,
        "dht": dht,
        "seqlens": seqlens,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "scale": float(spec.get("scale", 1.0 / math.sqrt(DIM))),
    }
