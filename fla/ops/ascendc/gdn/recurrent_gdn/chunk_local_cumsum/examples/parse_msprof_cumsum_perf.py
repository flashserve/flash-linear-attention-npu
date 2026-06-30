#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import argparse
import csv
import json
import math
from pathlib import Path


def find_latest_csv(case_dir: Path):
    candidates = list(case_dir.rglob("OpBasicInfo.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def get_field(row, needles):
    for key, value in row.items():
        key_norm = key.strip().lower()
        if all(needle in key_norm for needle in needles):
            return value
    return None


def to_float(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_basic_info(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    durations = []
    kernels = []
    block_dims = []
    op_types = []
    for row in rows:
        duration = to_float(get_field(row, ["task", "duration"]))
        if duration is None:
            continue
        durations.append(duration)
        kernel = get_field(row, ["kernel"]) or get_field(row, ["op", "name"]) or get_field(row, ["name"])
        op_type = get_field(row, ["op", "type"]) or get_field(row, ["type"])
        block_dim = get_field(row, ["block", "dim"])
        if kernel:
            kernels.append(str(kernel))
        if op_type:
            op_types.append(str(op_type))
        if block_dim:
            block_dims.append(str(block_dim))
    if not durations:
        raise RuntimeError(f"Task Duration column not found in {csv_path}")
    return {
        "op_basic_info_csv": str(csv_path),
        "task_duration_us": sum(durations),
        "max_task_duration_us": max(durations),
        "kernel_count": len(durations),
        "kernels": kernels,
        "block_dims": block_dims,
        "op_types": op_types,
    }


def load_summary(case_dir: Path):
    with (case_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return {case["name"]: case for case in summary["cases"]}


def main():
    parser = argparse.ArgumentParser(description="Parse msprof op output for ChunkLocalCumsum perf comparison.")
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()

    profile_root = Path(args.profile_root).resolve()
    case_dir = Path(args.case_dir).resolve()
    cases_by_name = load_summary(case_dir)
    case_names = args.cases or sorted(cases_by_name)
    results = []
    for name in case_names:
        case = cases_by_name[name]
        row = {
            "case": name,
            "shape": [case["B"], case["T"], case["H"]],
            "chunk_size": case["chunk_size"],
            "reverse": bool(case["reverse"]),
            "scale": case["scale"],
            "is_varlen": bool(case["is_varlen"]),
            "block_t": case["block_t"],
        }
        for impl in ("triton", "ascendc"):
            impl_case_dir = profile_root / impl / name
            csv_path = find_latest_csv(impl_case_dir)
            if csv_path is None:
                row[impl] = {"error": f"OpBasicInfo.csv not found under {impl_case_dir}"}
                continue
            row[impl] = parse_basic_info(csv_path)
            row[impl]["profiling_dir"] = str(csv_path.parent)
        triton_us = row.get("triton", {}).get("task_duration_us")
        ascendc_us = row.get("ascendc", {}).get("task_duration_us")
        if triton_us and ascendc_us and ascendc_us > 0:
            row["speedup_triton_over_ascendc"] = triton_us / ascendc_us
            row["faster"] = "ascendc" if ascendc_us < triton_us else "triton"
        else:
            row["speedup_triton_over_ascendc"] = None
            row["faster"] = "n/a"
        results.append(row)

    valid = [r for r in results if isinstance(r.get("triton"), dict) and isinstance(r.get("ascendc"), dict)
             and "task_duration_us" in r["triton"] and "task_duration_us" in r["ascendc"]]
    avg_speedup = None
    if valid:
        avg_speedup = sum(r["speedup_triton_over_ascendc"] for r in valid) / len(valid)
    payload = {
        "source": "msprof op",
        "metric": "Task Duration(us)",
        "profile_root": str(profile_root),
        "case_count": len(results),
        "valid_case_count": len(valid),
        "average_speedup_triton_over_ascendc": avg_speedup,
        "cases": results,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# ChunkLocalCumsum Triton vs AscendC 性能对比",
        "",
        "## 摘要",
        "",
        f"- 数据来源: `msprof op`",
        f"- 指标: `Task Duration(us)`",
        f"- profiling 根目录: `{profile_root}`",
        f"- 有效 case 数: {len(valid)}/{len(results)}",
    ]
    if avg_speedup is not None and math.isfinite(avg_speedup):
        lines.append(f"- 平均 speedup `triton_us / ascendc_us`: {avg_speedup:.3f}x")
    lines.extend([
        "",
        "## 明细",
        "",
        "| case | shape | chunk | reverse | scale | varlen | Triton us | AscendC us | speedup(T/A) | faster |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for row in results:
        triton_us = row.get("triton", {}).get("task_duration_us")
        ascendc_us = row.get("ascendc", {}).get("task_duration_us")
        speedup = row.get("speedup_triton_over_ascendc")
        lines.append(
            "| {case} | {shape} | {chunk_size} | {reverse} | {scale} | {is_varlen} | {tu} | {au} | {sp} | {faster} |".format(
                case=row["case"],
                shape=row["shape"],
                chunk_size=row["chunk_size"],
                reverse=str(row["reverse"]).lower(),
                scale=row["scale"],
                is_varlen=str(row["is_varlen"]).lower(),
                tu="n/a" if triton_us is None else f"{triton_us:.3f}",
                au="n/a" if ascendc_us is None else f"{ascendc_us:.3f}",
                sp="n/a" if speedup is None else f"{speedup:.3f}",
                faster=row["faster"],
            )
        )
    lines.extend([
        "",
        "## 原始数据",
        "",
        f"- JSON: `{out_json}`",
        f"- profiling: `{profile_root}`",
    ])
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
