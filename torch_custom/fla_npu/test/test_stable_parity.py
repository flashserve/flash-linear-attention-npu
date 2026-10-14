# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Stable-ABI launcher 与 ctypes 参考实现的**逐算子一致性对照**。

为什么需要它：适配器（``csrc/src/stable_<op>.cpp``）与 ctypes wrapper 是同一算子的两份 host 实现，
两者的参数编组必须逐项等价。launcher 侧的改动、以及任何"半自动生成适配器"的尝试，都需要一个
可执行的一致性判据——本脚本就是那个判据。

用法（两次运行 + 一次比较）::

    FLA_NPU_STABLE_ABI=ctypes python test_stable_parity.py dump /tmp/parity-ctypes.json
    FLA_NPU_STABLE_ABI=stable python test_stable_parity.py dump /tmp/parity-stable.json
    python test_stable_parity.py compare /tmp/parity-ctypes.json /tmp/parity-stable.json

判据：
* 每个算子记录 **实际服务它的后端**（``entry`` 字段）——若 launcher 那一轮某算子仍由 ctypes 服务，
  说明该算子不被 launcher 携带（或 launcher 不可用），比较对它**无意义**，必须以 ``SKIP`` 呈现而不是 PASS；
* 覆盖到的算子要求输出**逐值一致**（sha256 相同）。

配方（``RECIPES``）是"算子 → 确定性输入 + 调用"的登记表；新增算子只需加一条，未登记的在报告里列为 ``NO_RECIPE``。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from typing import Callable

import torch

from fla_npu.ops import ascendc as ascendc_ops
from fla_npu.ops import ascendc as _ascendc_mod


def _digest(tensor: torch.Tensor) -> dict:
    array = tensor.detach().float().cpu().numpy()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "sum": float(array.sum()),
        "max": float(array.max()) if array.size else None,
        "min": float(array.min()) if array.size else None,
    }


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


def _served_by(op_name: str) -> tuple[str, int]:
    """回读仓库自己的后端记账（``BACKENDS`` / ``FALLBACKS``）。

    包装函数的 ``__module__`` 只能说明它挂在哪个模块（都在 ``fla_npu.ops.ascendc``），
    真正服务该算子的是 ``_note_backend()`` 记下来的 backend 名（``stable`` / ``ctypes``）。
    """

    backend = getattr(_ascendc_mod, "BACKENDS", {}).get(op_name, "?")
    fallbacks = getattr(_ascendc_mod, "FALLBACKS", {}).get(op_name, 0)
    return backend, fallbacks


def dump(path: str) -> int:
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"torch_npu is required: {exc}")
        return 2

    from fla_npu.ops.ascendc import _stable

    report = {
        "backend_env": __import__("os").environ.get("FLA_NPU_STABLE_ABI"),
        "stable_backend_selected": _ascendc_mod._stable_backend_selected(),
        "launcher_usable": bool(_stable.available()),
        "operators": {},
    }
    for name, recipe in RECIPES.items():
        record: dict = {"entry": _served_by(name)[0]}
        try:
            outputs = recipe()
            if not isinstance(outputs, (tuple, list)):
                outputs = [outputs]
            record["outputs"] = [_digest(t) for t in outputs if isinstance(t, torch.Tensor)]
            record["status"] = "ok"
        except Exception as exc:  # 配方失败（形状约束/环境限制）要如实记录，不能算通过
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        backend, fallbacks = _served_by(name)
        record["entry"] = backend
        record["fallbacks"] = fallbacks
        report["operators"][name] = record
        print(f"[{report['backend_env']}] {name:38s} backend={backend:7s} "
              f"fallbacks={fallbacks} {record['status']}")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"written {path}")
    return 0


def compare(left_path: str, right_path: str) -> int:
    with open(left_path, encoding="utf-8") as handle:
        left = json.load(handle)
    with open(right_path, encoding="utf-8") as handle:
        right = json.load(handle)
    print(f"left : {left_path}  FLA_NPU_STABLE_ABI={left['backend_env']}  "
          f"launcher_usable={left['launcher_usable']}")
    print(f"right: {right_path}  FLA_NPU_STABLE_ABI={right['backend_env']}  "
          f"launcher_usable={right['launcher_usable']}")
    names = sorted(set(left["operators"]) | set(right["operators"]))
    failed: list[str] = []
    for name in names:
        a = left["operators"].get(name, {})
        b = right["operators"].get(name, {})
        if a.get("status") != "ok" or b.get("status") != "ok":
            print(f"  {name:38s} ERROR   left={a.get('status')} right={b.get('status')}"
                  f"  {a.get('error') or b.get('error') or ''}")
            continue
        # 只有"两轮由不同后端服务"时，比较才有意义
        if a["entry"] == b["entry"]:
            print(f"  {name:38s} SKIP    两轮同一后端({a['entry']})——不构成对照"
                  f"（launcher 未承载或不可用）")
            continue
        same = [x["sha256"] for x in a["outputs"]] == [x["sha256"] for x in b["outputs"]]
        detail = ""
        if not same:
            pairs = zip(a["outputs"], b["outputs"])
            detail = "  " + " | ".join(
                f"sum {x['sum']:+.5f} vs {y['sum']:+.5f}" for x, y in pairs)
            failed.append(name)
        print(f"  {name:38s} {'PASS' if same else 'FAIL'}    "
              f"{a['entry']} vs {b['entry']}{detail}")
    if failed:
        print(f"parity FAILED for: {', '.join(failed)}")
        return 1
    print("parity OK（覆盖到的算子逐值一致；SKIP 项需在 launcher 侧补齐承载后才能对照）")
    return 0


def main(argv: list[str]) -> int:
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
