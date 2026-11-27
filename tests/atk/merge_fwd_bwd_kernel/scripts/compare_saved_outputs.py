#!/usr/bin/env python3
"""比对 ATK --save_data output 落下的 npu_dut / cpu_lowp / cpu_benchmark。

ATK 26.11.11 的 cv_fused 汇总表经常只给 cpu_cpu_lowp 打分，npu_npu_dut 精度列是 "-"。
真正的 DUT 证据在 output/npu_npu_dut/.../output_0.pt。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

LIMITS = {"max_re_ratio": 5.0, "avg_re_ratio": 1.5, "rmse_ratio": 1.5}
ROLES = ("npu_npu_dut", "cpu_cpu_lowp", "cpu_benchmark")


def _stats(actual: torch.Tensor, golden: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    golden_f = golden.float()
    diff = (actual_f - golden_f).abs()
    rel = diff / golden_f.abs().clamp_min(1e-12)
    return {
        "max_abs": float(diff.max()),
        "max_re": float(rel.max()),
        "avg_re": float(rel.mean()),
        "rmse": float(diff.pow(2).mean().sqrt()),
        "mag": float(actual_f.abs().mean()),
    }


def _ratio(actual: float, benchmark: float) -> float:
    if benchmark == 0:
        return 0.0 if actual == 0 else float("inf")
    return actual / benchmark


def _find_cases(output_root: Path) -> list[int]:
    npu = output_root / "npu_npu_dut"
    if not npu.is_dir():
        raise FileNotFoundError(f"找不到 NPU 输出目录：{npu}")
    cases = []
    for path in npu.rglob("output_0.pt"):
        try:
            cases.append(int(path.parent.name))
        except ValueError:
            continue
    return sorted(set(cases))


def _load(output_root: Path, role: str, case_id: int) -> torch.Tensor:
    matches = list((output_root / role).rglob(f"{case_id}/output_0.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{role} case {case_id}: {matches}")
    return torch.load(matches[0], map_location="cpu", weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path, help="ATK 结果里的 output/ 目录")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    cases = _find_cases(output_root)
    if not cases:
        raise SystemExit(f"没有找到用例：{output_root}")
    failed = 0
    print(
        f"{'id':>4} {'max_r':>8} {'avg_r':>8} {'rmse_r':>8} "
        f"{'npu_max_abs':>12} {'lowp_max_abs':>12} {'mag':>10} pass"
    )
    for case_id in cases:
        npu, lowp, gold = (_load(output_root, role, case_id) for role in ROLES)
        npu_s = _stats(npu, gold)
        lowp_s = _stats(lowp, gold)
        max_r = _ratio(npu_s["max_re"], lowp_s["max_re"])
        avg_r = _ratio(npu_s["avg_re"], lowp_s["avg_re"])
        rmse_r = _ratio(npu_s["rmse"], lowp_s["rmse"])
        ok = (
            max_r <= LIMITS["max_re_ratio"]
            and avg_r <= LIMITS["avg_re_ratio"]
            and rmse_r <= LIMITS["rmse_ratio"]
            and torch.isfinite(npu.float()).all().item()
        )
        failed += int(not ok)
        print(
            f"{case_id:4d} {max_r:8.4f} {avg_r:8.4f} {rmse_r:8.4f} "
            f"{npu_s['max_abs']:12.4e} {lowp_s['max_abs']:12.4e} "
            f"{npu_s['mag']:10.4e} {ok}"
        )
    print(f"cv_fused {LIMITS}: {len(cases) - failed}/{len(cases)} pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
