# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic inputs shared by the CPU, device and performance checks."""
import torch

from .reference import reference


def make_case(frames, heads, tokens, dtype, *, text=True, strided=True, seed=431):
    rng = torch.Generator().manual_seed(seed)

    def feature(normalize):
        value = torch.randn(frames, tokens, heads, 128, generator=rng)
        if normalize:
            value = torch.nn.functional.normalize(value, dim=-1)
        value = value.to(dtype).transpose(1, 2)
        return value if strided else value.contiguous()

    q, k, v = feature(True), feature(True), feature(False)
    beta = torch.rand(frames, heads, tokens, generator=rng).to(dtype)
    alpha = torch.rand(frames, heads, 128, generator=rng) * 0.2 + 0.79
    row = torch.arange(frames)
    bounds = torch.stack(((row - 1).clamp_min(0), (row + 1).clamp_max(frames - 1)), 1)
    state = torch.randn(heads, 128, 128, generator=rng) * 0.025 if text else None
    return q, k, v, beta, alpha, bounds, state


def on_device(case, device):
    return tuple(x.to(device) if i != 5 and x is not None else x for i, x in enumerate(case))


def check(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("public output must preserve the feature shape and dtype")
    diff = actual.cpu().float() - expected.float()
    rel = (diff.norm() / expected.float().norm().clamp_min(1e-12)).item()
    peak = (diff.abs().max() / expected.float().abs().max().clamp_min(1e-12)).item()
    limits = (0.007, 0.04) if actual.dtype == torch.bfloat16 else (0.0002, 0.0004)
    if not (rel < limits[0] and peak < limits[1]):
        raise AssertionError(f"relative L2 {rel}, peak-normalized error {peak}, limits {limits}")
    return rel, peak


def expected(case, silu=True, norm=True):
    return reference(*case, silu=silu, qk_l2norm=norm)

