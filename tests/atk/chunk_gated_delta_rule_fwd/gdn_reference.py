#!/usr/bin/env python3
"""GDN 大融合 core 的纯 CPU FP64 golden 实现。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass(frozen=True)
class GdnCase:
    batch: int
    k_heads: int
    v_heads: int
    tokens: int
    key_dim: int
    value_dim: int
    chunk_size: int
    scale: float
    scenario: str
    cu_seqlens: Optional[tuple[int, ...]] = None

    @property
    def output_final_state(self) -> bool:
        return self.scenario in {"state_initial_final", "state_zero_final"}

    @property
    def uses_initial_state(self) -> bool:
        return self.scenario.startswith("state_")

    @property
    def sequence_count(self) -> int:
        return len(self.cu_seqlens) - 1 if self.cu_seqlens is not None else self.batch

    def validate(self) -> None:
        if self.key_dim != 128:
            raise ValueError(f"GDN Phase6 要求 K=128，实际为 {self.key_dim}")
        if self.value_dim not in (128, 256):
            raise ValueError(f"GDN Phase6 要求 V=128/256，实际为 {self.value_dim}")
        if self.chunk_size not in (64, 128):
            raise ValueError(f"chunk_size 必须为 64/128，实际为 {self.chunk_size}")
        if self.k_heads <= 0 or self.v_heads <= 0 or self.v_heads % self.k_heads != 0:
            raise ValueError(
                f"GVA 约束要求 Hk 能整除 Hv：Hk={self.k_heads}, Hv={self.v_heads}"
            )
        if self.scenario not in {
            "dense",
            "varlen",
            "state_initial_final",
            "state_initial_only",
            "state_zero_final",
        }:
            raise ValueError(f"未知场景：{self.scenario}")
        if self.cu_seqlens is None:
            return
        if self.batch != 1:
            raise ValueError("变长 BNSD 输入要求物理 batch=1")
        if (
            len(self.cu_seqlens) < 2
            or self.cu_seqlens[0] != 0
            or self.cu_seqlens[-1] != self.tokens
        ):
            raise ValueError(
                f"cu_seqlens 必须从 0 开始并以 T={self.tokens} 结束：{self.cu_seqlens}"
            )
        if any(left >= right for left, right in zip(self.cu_seqlens, self.cu_seqlens[1:])):
            raise ValueError(f"cu_seqlens 必须严格递增：{self.cu_seqlens}")


def canonical_chunk_indices(
    cu_seqlens: Optional[Iterable[int]], chunk_size: int
) -> Optional[list[int]]:
    if cu_seqlens is None:
        return None
    values = [int(value) for value in cu_seqlens]
    result: list[int] = []
    for sequence, (begin, end) in enumerate(zip(values, values[1:])):
        for local_chunk in range(math.ceil((end - begin) / chunk_size)):
            result.extend((sequence, local_chunk))
    return result


def sequence_ranges(case: GdnCase):
    if case.cu_seqlens is None:
        for batch_index in range(case.batch):
            yield batch_index, batch_index, 0, case.tokens
        return
    for sequence, (begin, end) in enumerate(zip(case.cu_seqlens, case.cu_seqlens[1:])):
        yield 0, sequence, int(begin), int(end)


def chunk_ranges(case: GdnCase):
    for batch_index, sequence, begin, end in sequence_ranges(case):
        for chunk_begin in range(begin, end, case.chunk_size):
            chunk_end = min(chunk_begin + case.chunk_size, end)
            yield batch_index, sequence, chunk_begin, chunk_end


def deterministic_initial_state(case: GdnCase) -> Optional[torch.Tensor]:
    if not case.uses_initial_state:
        return None
    shape = (case.sequence_count, case.v_heads, case.key_dim, case.value_dim)
    if case.scenario == "state_zero_final":
        return torch.zeros(shape, dtype=torch.float32)
    values = torch.arange(math.prod(shape), dtype=torch.float32)
    return (((values.remainder(257) - 128) * 1.0e-4).reshape(shape)).contiguous()


def effective_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    public_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """在 CPU 上冻结三条路径完全一致的实际算子输入。"""

    q = q.detach().cpu().to(public_dtype).contiguous()
    k = k.detach().cpu().to(public_dtype).contiguous()
    v = v.detach().cpu().to(public_dtype).contiguous()
    raw_g = raw_g.detach().cpu().float().contiguous()
    raw_beta = raw_beta.detach().cpu().float().contiguous()
    g = (-torch.sigmoid(raw_g) * 0.1).to(torch.float32).contiguous()
    beta = torch.sigmoid(raw_beta).to(public_dtype).contiguous()
    return q, k, v, g, beta


def validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    case: GdnCase,
) -> None:
    case.validate()
    if tuple(q.shape) != (case.batch, case.k_heads, case.tokens, case.key_dim):
        raise ValueError(f"q shape 不匹配：{tuple(q.shape)}")
    if tuple(k.shape) != tuple(q.shape):
        raise ValueError(f"k shape 不匹配：{tuple(k.shape)}")
    if tuple(v.shape) != (case.batch, case.v_heads, case.tokens, case.value_dim):
        raise ValueError(f"v shape 不匹配：{tuple(v.shape)}")
    expected_gate = (case.batch, case.tokens, case.v_heads)
    if tuple(g.shape) != expected_gate or tuple(beta.shape) != expected_gate:
        raise ValueError(
            f"g/beta 应为 {expected_gate}，实际为 {tuple(g.shape)}/{tuple(beta.shape)}"
        )


def mask_a_contract(a: torch.Tensor, case: GdnCase) -> torch.Tensor:
    """只保留每个真实 chunk 的有效方阵，清零无定义 tail padding。"""

    masked = torch.zeros_like(a)
    for batch_index, _sequence, begin, end in chunk_ranges(case):
        valid = end - begin
        masked[batch_index, :, begin:end, :valid] = a[
            batch_index, :, begin:end, :valid
        ]
    return masked


def _compute_cumsum_and_a(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    case: GdnCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.float64
    k_compute = k.to(compute_dtype)
    g_compute = g.to(compute_dtype)
    beta_compute = beta.to(compute_dtype)
    g_cumsum = torch.zeros(
        (case.batch, case.tokens, case.v_heads), dtype=compute_dtype
    )
    a = torch.zeros(
        (case.batch, case.v_heads, case.tokens, case.chunk_size),
        dtype=compute_dtype,
    )
    head_ratio = case.v_heads // case.k_heads

    for batch_index, _sequence, begin, end in chunk_ranges(case):
        valid = end - begin
        eye = torch.eye(valid, dtype=compute_dtype)
        lower = torch.tril(torch.ones((valid, valid), dtype=torch.bool), diagonal=-1)
        for value_head in range(case.v_heads):
            key_head = value_head // head_ratio
            gate = torch.cumsum(
                g_compute[batch_index, begin:end, value_head], dim=0
            )
            g_cumsum[batch_index, begin:end, value_head] = gate
            key = k_compute[batch_index, key_head, begin:end]
            score = key @ key.transpose(0, 1)
            gate_scale = torch.exp(gate[:, None] - gate[None, :])
            raw = score * gate_scale * beta_compute[
                batch_index, begin:end, value_head
            ][:, None]
            raw = torch.where(lower, raw, torch.zeros_like(raw))
            inverse = torch.linalg.inv(eye + raw)
            a[batch_index, value_head, begin:end, :valid] = inverse

    return g_cumsum, a


def _compute_recurrent_outputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    case: GdnCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.float64
    q_compute = q.to(compute_dtype)
    k_compute = k.to(compute_dtype)
    v_compute = v.to(compute_dtype)
    g_compute = g.to(compute_dtype)
    beta_compute = beta.to(compute_dtype)
    output = torch.zeros(
        (case.batch, case.v_heads, case.tokens, case.value_dim),
        dtype=compute_dtype,
    )
    final_state = torch.zeros(
        (case.sequence_count, case.v_heads, case.key_dim, case.value_dim),
        dtype=compute_dtype,
    )
    head_ratio = case.v_heads // case.k_heads

    for batch_index, sequence, begin, end in sequence_ranges(case):
        for value_head in range(case.v_heads):
            key_head = value_head // head_ratio
            if initial_state is None:
                state = torch.zeros(
                    (case.key_dim, case.value_dim), dtype=compute_dtype
                )
            else:
                state = initial_state[sequence, value_head].to(compute_dtype).clone()
            for token in range(begin, end):
                state = state * torch.exp(g_compute[batch_index, token, value_head])
                key = k_compute[batch_index, key_head, token]
                delta = v_compute[batch_index, value_head, token] - key @ state
                delta = delta * beta_compute[batch_index, token, value_head]
                state = state + torch.outer(key, delta)
                value = (q_compute[batch_index, key_head, token] @ state) * case.scale
                output[batch_index, value_head, token] = value
            final_state[sequence, value_head] = state

    return output, final_state


def run_golden_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    case: GdnCase,
    public_dtype: torch.dtype,
):
    """以 FP64 内部计算返回与融合算子同序的 3/4 个逻辑输出。"""

    validate_inputs(q, k, v, g, beta, case)
    initial_state = deterministic_initial_state(case)
    output, final_state = _compute_recurrent_outputs(
        q, k, v, g, beta, initial_state, case
    )
    g_cumsum, a = _compute_cumsum_and_a(k, g, beta, case)
    a = mask_a_contract(a, case)
    # 双标杆 golden 使用 FP64 完成内部计算，但 ATK 比较前要求三路
    # 输出 dtype 一致，因此按真实算子输出契约回传。
    output = output.to(public_dtype)
    final_state = final_state.float()
    g_cumsum = g_cumsum.float()
    a = a.to(public_dtype)
    if case.output_final_state:
        return output, final_state, g_cumsum, a
    return output, g_cumsum, a


def output_names(case: GdnCase) -> tuple[str, ...]:
    if case.output_final_state:
        return "o", "final_state", "g_cumsum", "A"
    return "o", "g_cumsum", "A"
