#!/usr/bin/env python3
"""Run the frozen A2 GDN Phase 3 accuracy or production performance matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


CASES = (
    ("D_BF16_C64", "bf16", 1, 4, 8, 1024, 64, "", False, False),
    ("D_FP16_C64", "fp16", 1, 4, 8, 1025, 64, "", False, False),
    ("D_BF16_C128", "bf16", 1, 4, 8, 1024, 128, "", False, False),
    ("D_FP16_C128", "fp16", 1, 4, 8, 1025, 128, "", False, False),
    ("V_BF16_C64", "bf16", 1, 4, 8, 259, 64, "0,1,66,259", False, False),
    ("V_FP16_C64", "fp16", 1, 4, 8, 259, 64, "0,1,66,259", False, False),
    ("V_BF16_C128", "bf16", 1, 4, 8, 259, 128, "0,1,130,259", False, False),
    ("V_FP16_C128", "fp16", 1, 4, 8, 259, 128, "0,1,130,259", False, False),
    ("STATE_D_BF16_C64", "bf16", 2, 4, 4, 1024, 64, "", True, True),
)

PHASE_VARIANTS = (
    "phase2_one_aclnn_fused_kkt_solve",
    "phase3_one_aclnn_fused_cumsum_kkt",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def latency_summary(samples: list[float]) -> dict:
    return {
        "iterations": len(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 0.90),
        "min_ms": min(samples),
        "samples_ms": samples,
    }


def benchmark_command(
    benchmark: Path,
    args,
    case: tuple,
    output: Path,
    standalone_variant: str | None = None,
) -> list[str]:
    (
        case_id, dtype, batch, key_heads, value_heads, tokens, chunk_size, cu_seqlens,
        initial_state, output_final_state,
    ) = case
    command = [
        sys.executable, str(benchmark),
        "--device", str(args.device),
        "--dtype", dtype,
        "--batch", str(batch),
        "--key-heads", str(key_heads),
        "--value-heads", str(value_heads),
        "--tokens", str(tokens),
        "--chunk-size", str(chunk_size),
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--case-id", case_id,
        "--output", str(output),
    ]
    if args.mode == "accuracy":
        command.append("--phase3-accuracy-only")
    elif standalone_variant is not None:
        command.extend(("--standalone-variant", standalone_variant))
    if cu_seqlens:
        command.extend(("--cu-seqlens", cu_seqlens))
    if initial_state:
        command.append("--initial-state")
    if output_final_state:
        command.append("--output-final-state")
    return command


def aggregate_runs(runs: list[dict]) -> dict:
    aggregate = {}
    for variant in PHASE_VARIANTS:
        variant_runs = [run for run in runs if run["variant"] == variant]
        samples = [
            sample
            for run in variant_runs
            for sample in run["report"]["variants"][variant]["latency"]["samples_ms"]
        ]
        batch_medians = [
            run["report"]["variants"][variant]["latency"]["median_ms"]
            for run in variant_runs
        ]
        signatures = {
            (
                run["report"]["variants"][variant]["workspace_max_bytes"],
                run["report"]["variants"][variant]["workspace_sum_bytes"],
                run["report"]["variants"][variant]["peak_allocated_delta_bytes"],
            )
            for run in variant_runs
        }
        aggregate[variant] = {
            "run_count": len(variant_runs),
            "latency": latency_summary(samples),
            "batch_medians_ms": batch_medians,
            "batch_median_summary": latency_summary(batch_medians),
            "finiteness_all_runs": all(
                run["report"]["finiteness"]["all_finite"] for run in variant_runs
            ),
            "workspace_consistent": len(signatures) == 1,
            "workspace_signatures": [
                {
                    "workspace_max_bytes": item[0],
                    "workspace_sum_bytes": item[1],
                    "peak_allocated_delta_bytes": item[2],
                }
                for item in sorted(signatures)
            ],
        }
    phase2 = aggregate[PHASE_VARIANTS[0]]["latency"]
    phase3 = aggregate[PHASE_VARIANTS[1]]["latency"]
    return {
        "variants": aggregate,
        "phase3_vs_phase2_median_change_pct":
            (phase3["median_ms"] / phase2["median_ms"] - 1.0) * 100.0,
        "phase3_vs_phase2_p90_change_pct":
            (phase3["p90_ms"] / phase2["p90_ms"] - 1.0) * 100.0,
        "phase3_vs_phase2_min_change_pct":
            (phase3["min_ms"] / phase2["min_ms"] - 1.0) * 100.0,
    }


def run_accuracy_case(benchmark: Path, args, case: tuple) -> dict:
    case_id = case[0]
    output = args.output_dir / f"{case_id}.json"
    subprocess.run(
        benchmark_command(benchmark, args, case, output),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    if not report["accuracy"]["phase3_vs_phase2"]["bit_exact"]:
        raise AssertionError(f"{case_id}: Phase 3 is not bit exact with Phase 2")
    if not all(item["all_finite"] for item in report["finiteness"].values()):
        raise AssertionError(f"{case_id}: Phase 2/3 produced non-finite output/state")
    return report


def run_performance_case(benchmark: Path, args, case: tuple) -> dict:
    case_id = case[0]
    case_dir = args.output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    expected_contract = None
    for round_index in range(args.rounds):
        order = PHASE_VARIANTS if round_index % 2 == 0 else tuple(reversed(PHASE_VARIANTS))
        for position, variant in enumerate(order):
            output = case_dir / f"round_{round_index + 1}_{position + 1}_{variant}.json"
            subprocess.run(
                benchmark_command(benchmark, args, case, output, variant),
                check=True,
                stdout=subprocess.DEVNULL,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            if report["measurement"]["standalone_variant"] != variant:
                raise AssertionError(f"{case_id}: standalone variant mismatch")
            if expected_contract is None:
                expected_contract = report["contract"]
            elif report["contract"] != expected_contract:
                raise AssertionError(f"{case_id}: contract changed between runs")
            if not report["finiteness"]["all_finite"]:
                raise AssertionError(f"{case_id}: {variant} produced non-finite output/state")
            runs.append({
                "round": round_index + 1,
                "position": position + 1,
                "variant": variant,
                "output": str(output),
                "report": report,
            })
    aggregate = aggregate_runs(runs)
    if not all(item["workspace_consistent"] for item in aggregate["variants"].values()):
        raise AssertionError(f"{case_id}: workspace or peak changed between runs")
    return {
        "contract": expected_contract,
        "measurement_mode": "standalone_clean_process_balanced",
        "round_count": args.rounds,
        "run_order": [
            {key: run[key] for key in ("round", "position", "variant", "output")}
            for run in runs
        ],
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", 0)))
    parser.add_argument("--mode", choices=("accuracy", "performance"), default="accuracy")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "performance" and os.environ.get("ASCEND_LAUNCH_BLOCKING") not in (None, "", "0"):
        raise RuntimeError("Production performance requires ASCEND_LAUNCH_BLOCKING to be unset.")
    if args.mode == "performance" and (args.rounds < 2 or args.rounds % 2):
        raise ValueError("performance mode requires a positive even --rounds >= 2")
    selected = set(args.case)
    unknown = selected - {case[0] for case in CASES}
    if unknown:
        raise ValueError(f"Unknown case IDs: {sorted(unknown)}")

    benchmark = Path(__file__).with_name("benchmark_gdn_core_ablation.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "device": args.device,
        "mode": args.mode,
        "warmup_rounds": args.warmup,
        "iterations_per_variant": args.iterations,
        "rounds": args.rounds if args.mode == "performance" else None,
        "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
        "cases": {},
    }
    for case in CASES:
        case_id = case[0]
        if selected and case_id not in selected:
            continue
        print(f"[{case_id}] running", flush=True)
        if args.mode == "accuracy":
            summary["cases"][case_id] = run_accuracy_case(benchmark, args, case)
        else:
            summary["cases"][case_id] = run_performance_case(benchmark, args, case)
        print(f"[{case_id}] PASS", flush=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
