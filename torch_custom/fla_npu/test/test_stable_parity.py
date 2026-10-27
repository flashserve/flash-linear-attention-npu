# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Stable-ABI 修改前后及历史后端的逐算子一致性对照。

Stable-only 迁移验证：在修改前/后的包环境分别设置不同的
FLA_NPU_PARITY_REVISION，运行 dump，再用 compare-versions 比较。
此模式要求两个实际 package source hash 不同、双方均走 Stable 且无回退；
原 compare 命令继续保留跨历史后端对照，不能用相同后端自证通过。

为什么需要它：适配器（``csrc/src/stable_<op>.cpp``）与 ctypes wrapper 是同一算子的两份 host 实现，
两者的参数编组必须逐项等价。launcher 侧的改动、以及任何"半自动生成适配器"的尝试，都需要一个
可执行的一致性判据——本脚本就是那个判据。

用法（两次运行 + 一次比较）::

    FLA_NPU_STABLE_ABI=ctypes python test_stable_parity.py dump /tmp/parity-ctypes.json
    FLA_NPU_STABLE_ABI=stable python test_stable_parity.py dump /tmp/parity-stable.json
    python test_stable_parity.py compare /tmp/parity-ctypes.json /tmp/parity-stable.json

可用 ``FLA_NPU_PARITY_OPS=npu_op_a,npu_op_b`` 显式限定两轮配方；未知名称会
直接报错，compare 也会拒绝两份报告选择不同的配方集合。

判据：
* 每个算子记录 **实际服务它的后端**（``entry`` 字段）——若 launcher 那一轮某算子仍由 ctypes 服务，
  说明该算子不被 launcher 携带（或 launcher 不可用），比较对它**无意义**，必须以 ``SKIP`` 呈现而不是 PASS；
* 覆盖到的算子要求输出**逐值一致**（sha256 相同）。

比较默认是严格模式：任何 ``ERROR``、``SKIP`` 或预期后端未命中都会返回 1。
仅生成探索性报告时可设置 ``FLA_NPU_PARITY_ALLOW_INCOMPLETE=1``。

配方（``RECIPES``）是"算子 → 确定性输入 + 调用"的登记表；新增算子只需加一条，未登记的在报告里列为 ``NO_RECIPE``。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Callable


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


def _selected_recipes() -> dict[str, Callable[[], object]]:
    raw = (__import__("os").environ.get("FLA_NPU_PARITY_OPS") or "").strip()
    if not raw:
        return RECIPES
    names = list(dict.fromkeys(name.strip() for name in raw.split(",") if name.strip()))
    unknown = sorted(set(names) - set(RECIPES))
    if unknown:
        raise ValueError(f"Unknown FLA_NPU_PARITY_OPS recipes: {', '.join(unknown)}")
    return {name: RECIPES[name] for name in names}


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

    _load_runtime()
    from fla_npu.ops.ascendc import _stable

    try:
        recipes = _selected_recipes()
    except ValueError as exc:
        print(exc)
        return 2

    report = {
        "backend_env": __import__("os").environ.get("FLA_NPU_STABLE_ABI"),
        "stable_backend_selected": (getattr(_ascendc_mod, "_stable_backend_selected", lambda: True)()),
        "package_source_sha256": hashlib.sha256(Path(_stable.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "revision": __import__("os").environ.get("FLA_NPU_PARITY_REVISION"),
        "launcher_usable": bool(_stable.available()),
        # 全部已发布算子：compare 用它报告 NO_RECIPE（未登记配方的算子），
        # 否则"覆盖率"无从判断，测试会因为"只跑了有配方的几个"而假通过。
        "published_ops": [name for name in getattr(_ascendc_mod, "_ASCENDC_OPS", ())],
        "requested_ops": sorted(recipes),
        "operators": {},
    }
    for name, recipe in recipes.items():
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


def compare(left_path: str, right_path: str, *, versions: bool = False) -> int:
    with open(left_path, encoding="utf-8") as handle:
        left = json.load(handle)
    with open(right_path, encoding="utf-8") as handle:
        right = json.load(handle)
    print(f"left : {left_path}  FLA_NPU_STABLE_ABI={left['backend_env']}  "
          f"launcher_usable={left['launcher_usable']}")
    print(f"right: {right_path}  FLA_NPU_STABLE_ABI={right['backend_env']}  "
          f"launcher_usable={right['launcher_usable']}")
    if sorted(left.get("requested_ops") or left["operators"]) != sorted(
            right.get("requested_ops") or right["operators"]):
        print("parity FAILED: the two reports selected different recipe sets")
        return 1
    if versions:
        if (not left.get("revision") or not right.get("revision")
                or left["revision"] == right["revision"]
                or not left.get("package_source_sha256") or not right.get("package_source_sha256")
                or left["package_source_sha256"] == right["package_source_sha256"]):
            print("parity FAILED: version comparison requires distinct revisions and actual package source hashes")
            return 1
        if not left.get("launcher_usable") or not right.get("launcher_usable"):
            print("parity FAILED: version comparison requires usable Stable launchers")
            return 1
    names = sorted(set(left["operators"]) | set(right["operators"]))
    environ = __import__("os").environ
    allow_incomplete = (environ.get("FLA_NPU_PARITY_ALLOW_INCOMPLETE") or "").upper() in {
        "1", "TRUE", "YES", "ON"
    }
    legacy_strict = environ.get("FLA_NPU_PARITY_STRICT")
    strict = not allow_incomplete
    if legacy_strict is not None:
        strict = legacy_strict.upper() in {"1", "TRUE", "YES", "ON"}
    if versions:
        strict = True
    failed: list[str] = []      # 数值不一致
    errored: list[str] = []     # 配方失败（环境/形状）
    vacuous: list[str] = []     # 两轮同一后端 ⇒ 不构成对照
    wrong_backend: list[str] = []
    expected_left = "ctypes" if left.get("backend_env") == "ctypes" else "stable"
    expected_right = "ctypes" if right.get("backend_env") == "ctypes" else "stable"
    if versions:
        expected_left = expected_right = "stable"
    for name in names:
        a = left["operators"].get(name, {})
        b = right["operators"].get(name, {})
        if a.get("status") != "ok" or b.get("status") != "ok":
            errored.append(name)
            print(f"  {name:38s} ERROR   left={a.get('status')} right={b.get('status')}"
                  f"  {a.get('error') or b.get('error') or ''}")
            continue
        if a.get("entry") != expected_left or b.get("entry") != expected_right:
            wrong_backend.append(name)
            print(f"  {name:38s} BACKEND left={a.get('entry')} (expected {expected_left}) "
                  f"right={b.get('entry')} (expected {expected_right})")
            continue
        # 只有"两轮由不同后端服务"时，比较才有意义。
        if versions and (a.get("fallbacks", 0) or b.get("fallbacks", 0)):
            wrong_backend.append(name)
            print(f"  {name} BACKEND: fallback observed in Stable version comparison")
            continue
        if a["entry"] == b["entry"] and not versions:
            vacuous.append(name)
            print(f"  {name:38s} SKIP    两轮同一后端({a['entry']})——不构成对照"
                  f"（launcher 未承载或不可用）")
            continue
        left_outputs = [
            (x.get("shape"), x.get("dtype"), x.get("sha256")) for x in a.get("outputs", [])
        ]
        right_outputs = [
            (x.get("shape"), x.get("dtype"), x.get("sha256")) for x in b.get("outputs", [])
        ]
        same = bool(left_outputs) and left_outputs == right_outputs
        detail = ""
        if not same:
            pairs = zip(a["outputs"], b["outputs"])
            detail = "  " + " | ".join(
                f"sum {x.get('sum')!r} vs {y.get('sum')!r}" for x, y in pairs)
            failed.append(name)
        print(f"  {name:38s} {'PASS' if same else 'FAIL'}    "
              f"{a['entry']} vs {b['entry']}{detail}")

    published = sorted(set(left.get("published_ops") or []) | set(right.get("published_ops") or []))
    no_recipe = [name for name in published if name not in left["operators"] and name not in right["operators"]]
    if no_recipe:
        print(f"  {'（未登记配方）':36s} NO_RECIPE  {len(no_recipe)}/{len(published)}: "
              f"{', '.join(no_recipe[:6])}{' …' if len(no_recipe) > 6 else ''}")

    print(f"  汇总: PASS={len(names) - len(failed) - len(errored) - len(vacuous) - len(wrong_backend)} "
          f"FAIL={len(failed)} ERROR={len(errored)} SKIP={len(vacuous)} "
          f"BACKEND={len(wrong_backend)} NO_RECIPE={len(no_recipe)}  "
          f"严格模式={'开' if strict else '关'}")
    if failed:
        print(f"parity FAILED for: {', '.join(failed)}")
        return 1
    if strict and (errored or vacuous or wrong_backend):
        # 严格模式：任何"没测到"的算子都算失败——launcher 不可用/未承载会让所有用例
        # 走同一后端，此时"parity OK"是假通过，必须由调用方（CI）看见非零退出码。
        print("parity FAILED (strict)：存在 ERROR、SKIP 或 BACKEND 错配 —— "
              "说明没有真正对照到指定后端"
              f"（launcher_usable={right.get('launcher_usable')}）")
        return 1
    print("parity OK（覆盖到的算子逐值一致；SKIP/NO_RECIPE 需在 launcher 侧补齐承载或补配方）"
          + ("；严格模式已开启" if strict else ""))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "dump":
        return dump(argv[2])
    if len(argv) >= 4 and argv[1] == "compare-versions":
        return compare(argv[2], argv[3], versions=True)
    if len(argv) >= 4 and argv[1] == "compare":
        return compare(argv[2], argv[3])
    print(__doc__)
    print("可用配方：")
    for name, recipe in sorted(RECIPES.items()):
        print(f"  {name:38s} {inspect.getdoc(recipe) or ''}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
