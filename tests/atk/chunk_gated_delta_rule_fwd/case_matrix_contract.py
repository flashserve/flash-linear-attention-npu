#!/usr/bin/env python3
"""不依赖 ATK 运行时的 GDN 双标杆 500-case shape/场景矩阵。"""

from __future__ import annotations

import itertools
import random


SCENARIOS = (
    "dense",
    "varlen",
    "state_initial_final",
    "state_initial_only",
    "state_zero_final",
)
CASES_PER_SCENARIO = 100
HEAD_PAIRS = (
    (1, 1),
    (1, 2),
    (1, 4),
    (2, 2),
    (2, 4),
    (2, 8),
    (4, 4),
    (4, 8),
)
TOKEN_EDGES = {
    64: (
        1,
        2,
        3,
        31,
        32,
        63,
        64,
        65,
        95,
        127,
        128,
        129,
        191,
        193,
        255,
        257,
        319,
        383,
        511,
        513,
        639,
        767,
        895,
        1023,
        1025,
    ),
    128: (
        1,
        2,
        3,
        63,
        64,
        127,
        128,
        129,
        191,
        255,
        256,
        257,
        383,
        385,
        511,
        513,
        639,
        767,
        895,
        1023,
        1024,
        1025,
        1151,
        1279,
        1281,
    ),
}


def _varlen_metadata(chunk_size: int, index: int) -> tuple[int, str]:
    rng = random.Random(20260827 + index * 3571 + chunk_size)
    sequence_count = (1, 2, 3, 4, 5, 8)[(index // 2) % 6]
    edges = TOKEN_EDGES[chunk_size]
    lengths = []
    for sequence in range(sequence_count):
        edge = edges[(index * 7 + sequence * 3) % len(edges)]
        # 变长矩阵强调 chunk 边界和多序列重置；限制单序列长度，避免
        # CPU FP64 golden 因极端总 T 把精度矩阵退化成压力测试。
        edge = min(edge, 2 * chunk_size + 63)
        if (index + sequence) % 5 == 4:
            edge = rng.randint(3, 2 * chunk_size + 31)
        lengths.append(max(1, edge))
    offsets = [0, *itertools.accumulate(lengths)]
    return offsets[-1], ",".join(str(value) for value in offsets)


def case_contract(scenario: str, local_index: int):
    if scenario not in SCENARIOS:
        raise ValueError(f"未知场景：{scenario}")
    if not 0 <= local_index < CASES_PER_SCENARIO:
        raise ValueError(f"local_index 超出范围：{local_index}")
    use_varlen = scenario == "varlen" or (
        scenario.startswith("state_") and local_index % 2 == 1
    )
    shape_index = local_index if scenario in {"dense", "varlen"} else local_index // 2
    chunk_size = (64, 128)[shape_index % 2]
    value_dim = (128, 256)[(shape_index // 2) % 2]
    k_heads, v_heads = HEAD_PAIRS[
        (shape_index * 5 + shape_index // len(HEAD_PAIRS)) % len(HEAD_PAIRS)
    ]
    if use_varlen:
        tokens, cu_spec = _varlen_metadata(chunk_size, shape_index)
        return 1, k_heads, v_heads, tokens, value_dim, chunk_size, True, cu_spec
    batch = (1, 2, 3, 4)[(shape_index // 4) % 4]
    edges = TOKEN_EDGES[chunk_size]
    tokens = edges[(shape_index * 7 + shape_index // len(edges)) % len(edges)]
    # 其余维度的组合周期是 64；第二轮加入不与 chunk 边界对齐的偏移，
    # 保证 100 条定长合同不重复，同时继续保留首轮的精确边界值。
    tokens += (shape_index // 64) * (4 * chunk_size + 17)
    return batch, k_heads, v_heads, tokens, value_dim, chunk_size, False, "none"
