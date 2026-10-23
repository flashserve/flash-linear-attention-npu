# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Stable-only baseline/candidate comparison for representative overlay operators.

Run the same script in fresh, isolated baseline/candidate installations::

    python test_stable_parity.py dump /tmp/parity-before.json
    python test_stable_parity.py dump /tmp/parity-after.json
    python test_stable_parity.py compare /tmp/parity-before.json /tmp/parity-after.json

Both runs use Stable ABI. FLA_NPU_PARITY_OPS selects a comma-separated subset
of RECIPES; unknown names fail. The shared before/after verifier records public
signatures, exact inputs/outputs, optional returns, mutation/version counters
and errors. Missing calls, mismatched inputs, non-finite results and failed
recipes fail closed. These representative recipes supplement full public API
and ATK checks; they are not a 37-operator numerical acceptance claim.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from typing import Callable
from pathlib import Path


torch = None
ascendc_ops = None
_ascendc_mod = None


def _load_runtime() -> None:
    """Import NPU dependencies only for ``dump``; ``compare`` is JSON-only."""

    global torch, ascendc_ops, _ascendc_mod
    if torch is not None:
        return
    import torch as torch_module
    from fla_npu.ops import ascendc as ascendc_module

    torch = torch_module
    ascendc_ops = ascendc_module
    _ascendc_mod = ascendc_module


# ---------------------------------------------------------------------------
# 配方：每个算子一组确定性输入（固定 seed，保证两轮跑的是同一批数据）
# ---------------------------------------------------------------------------
def _chunk_local_cumsum() -> list[torch.Tensor]:
    torch.manual_seed(20260717)
    x = torch.ones(1, 2, 128, dtype=torch.float32)
    return ascendc_ops.npu_chunk_local_cumsum(x.npu(), chunk_size=64)


def _chunk_fwd_o() -> list[torch.Tensor]:
    batch, k_heads, v_heads, tokens, v_dim = 2, 4, 4, 129, 128
    k_dim = chunk_size = 128
    chunk_size = 64
    torch.manual_seed(20260717)
    q = torch.randn(batch, k_heads, tokens, k_dim, dtype=torch.bfloat16) * 0.1
    k = torch.randn(batch, k_heads, tokens, k_dim, dtype=torch.bfloat16) * 0.1
    v = torch.randn(batch, v_heads, tokens, v_dim, dtype=torch.bfloat16) * 0.1
    h = torch.randn(batch, math.ceil(tokens / chunk_size), v_heads, k_dim, v_dim,
                    dtype=torch.bfloat16) * 0.1
    g = torch.zeros(batch, v_heads, tokens, dtype=torch.float32)
    return ascendc_ops.npu_chunk_fwd_o(
        q.npu(), k.npu(), v.npu(), h.npu(), 1.0 / math.sqrt(k_dim), g=g.npu(), g_gamma=None,
        cu_seqlens=None, chunk_indices=None, chunk_size=chunk_size,
        transpose_state_layout=False)


def _chunk_gated_delta_rule_fwd_h() -> list[torch.Tensor]:
    batch, k_heads, v_heads, tokens, k_dim, v_dim = 1, 2, 2, 64, 128, 64
    torch.manual_seed(20260717)
    k = torch.randn(batch, k_heads, tokens, k_dim, dtype=torch.bfloat16) * 0.1
    w = torch.randn(batch, v_heads, tokens, k_dim, dtype=torch.bfloat16) * 0.1
    u = torch.randn(batch, v_heads, tokens, v_dim, dtype=torch.bfloat16) * 0.1
    g = (-torch.rand(batch, v_heads, tokens) * 0.1).float()
    return ascendc_ops.npu_chunk_gated_delta_rule_fwd_h(
        k.npu(), w.npu(), u.npu(), g=g.npu(), chunk_size=64)


def _chunk_gated_delta_rule_fwd() -> list[torch.Tensor]:
    batch, k_heads, v_heads, tokens, k_dim = 1, 2, 2, 64, 128
    torch.manual_seed(20260717)
    q = torch.randn(batch, k_heads, tokens, k_dim, dtype=torch.bfloat16) * 0.1
    k = torch.randn(batch, k_heads, tokens, k_dim, dtype=torch.bfloat16) * 0.1
    v = torch.randn(batch, v_heads, tokens, k_dim, dtype=torch.bfloat16) * 0.1
    g = (-torch.rand(batch, v_heads, tokens) * 0.1).float()
    beta = torch.sigmoid(torch.randn(batch, v_heads, tokens)).float()
    return ascendc_ops.npu_chunk_gated_delta_rule_fwd(
        q.transpose(1, 2).contiguous().npu(), k.transpose(1, 2).contiguous().npu(),
        v.transpose(1, 2).contiguous().npu(), g.transpose(1, 2).contiguous().npu(),
        beta.transpose(1, 2).contiguous().npu(), layout="BSND", use_exp2=True,
        chunk_size=64, output_final_state=True)


def _fast_gelu_custom() -> list[torch.Tensor]:
    torch.manual_seed(20260717)
    x = torch.randn(2, 16, 128, dtype=torch.bfloat16)
    return ascendc_ops.npu_fast_gelu_custom(x.npu())


RECIPES: dict[str, Callable[[], object]] = {
    "npu_chunk_local_cumsum": _chunk_local_cumsum,
    "npu_chunk_fwd_o": _chunk_fwd_o,
    "npu_chunk_gated_delta_rule_fwd_h": _chunk_gated_delta_rule_fwd_h,
    "npu_chunk_gated_delta_rule_fwd": _chunk_gated_delta_rule_fwd,
    "npu_fast_gelu_custom": _fast_gelu_custom,
}


def _selected_recipes() -> dict[str, Callable[[], object]]:
    raw = (__import__("os").environ.get("FLA_NPU_PARITY_OPS") or "").strip()
    if not raw:
        return RECIPES
    names = list(dict.fromkeys(name.strip() for name in raw.split(",") if name.strip()))
    unknown = sorted(set(names) - set(RECIPES))
    if unknown:
        raise ValueError(f"Unknown FLA_NPU_PARITY_OPS recipes: {', '.join(unknown)}")
    return {name: RECIPES[name] for name in names}


def _verification_tool():
    """Reuse the main Stable-only verifier; compare stays torch/NPU-free."""

    path = Path(__file__).resolve().parents[3] / "scripts/verify_stable_only_results.py"
    spec = importlib.util.spec_from_file_location("overlay_stable_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path: str) -> int:
    names = tuple(_selected_recipes())
    return _verification_tool().record(Path(path), Path(__file__), ["run", *names])


def compare(left_path: str, right_path: str) -> int:
    names = tuple(_selected_recipes())
    return _verification_tool().compare(Path(left_path), Path(right_path), names)


def run(names: list[str]) -> int:
    unknown = sorted(set(names) - set(RECIPES))
    if not names or unknown:
        raise ValueError(f"Expected known recipe names, got {names!r}")
    _load_runtime()
    for name in names:
        RECIPES[name]()
        torch.npu.synchronize()
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "run":
        return run(argv[2:])
    if len(argv) >= 3 and argv[1] == "dump":
        return dump(argv[2])
    if len(argv) >= 4 and argv[1] == "compare":
        return compare(argv[2], argv[3])
    print(__doc__)
    print("可用配方：")
    for name, recipe in sorted(RECIPES.items()):
        print(f"  {name:38s} {inspect.getdoc(recipe) or ''}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
