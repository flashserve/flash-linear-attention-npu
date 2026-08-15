# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------

"""Run the frozen A2 GDN Phase 2 production performance matrix."""

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
    ("S_B4_H4", "bf16", 4, 4, 4, 4096, 64, "", False, False),
    ("S_H1", "fp16", 1, 1, 1, 4096, 64, "", False, False),
    ("S_H16", "bf16", 1, 16, 16, 4096, 128, "", False, False),
    ("S_H32", "fp16", 1, 32, 32, 1024, 64, "", False, False),
    ("L_DENSE_T32768", "bf16", 1, 4, 8, 32768, 64, "", False, False),
    ("L_VARLEN_T32768", "fp16", 1, 4, 8, 32768, 128, "0,8191,16384,32768", False, False),
    ("STATE_D_BF16_C64", "bf16", 2, 4, 4, 1024, 64, "", True, True),
)

PHASE_VARIANTS = (
    "phase1_one_aclnn_six_kernels",
    "phase2_one_aclnn_fused_kkt_solve",
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


def aggregate_standalone_runs(runs: list[dict]) -> dict:
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
        workspace_signatures = {
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
            "workspace_consistent": len(workspace_signatures) == 1,
            "workspace_signatures": [
                {
                    "workspace_max_bytes": signature[0],
                    "workspace_sum_bytes": signature[1],
                    "peak_allocated_delta_bytes": signature[2],
                }
                for signature in sorted(workspace_signatures)
            ],
        }
    phase1 = aggregate[PHASE_VARIANTS[0]]["latency"]
    phase2 = aggregate[PHASE_VARIANTS[1]]["latency"]
    return {
        "variants": aggregate,
        "phase2_vs_phase1_median_change_pct": (
            (phase2["median_ms"] / phase1["median_ms"] - 1.0) * 100.0
        ),
        "phase2_vs_phase1_p90_change_pct": (
            (phase2["p90_ms"] / phase1["p90_ms"] - 1.0) * 100.0
        ),
        "phase2_vs_phase1_min_change_pct": (
            (phase2["min_ms"] / phase1["min_ms"] - 1.0) * 100.0
        ),
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
        sys.executable,
        str(benchmark),
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
    if standalone_variant is None:
        command.append("--paired-only")
    else:
        command.extend(("--standalone-variant", standalone_variant))
    if cu_seqlens:
        command.extend(("--cu-seqlens", cu_seqlens))
    if initial_state:
        command.append("--initial-state")
    if output_final_state:
        command.append("--output-final-state")
    return command


def run_standalone_case(benchmark: Path, args, case: tuple) -> dict:
    case_id = case[0]
    case_dir = args.output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    expected_contract = None
    for round_index in range(args.standalone_rounds):
        order = PHASE_VARIANTS if round_index % 2 == 0 else tuple(reversed(PHASE_VARIANTS))
        for position, variant in enumerate(order):
            output = case_dir / f"round_{round_index + 1}_{position + 1}_{variant}.json"
            command = benchmark_command(benchmark, args, case, output, variant)
            print(f"[{case_id}] round {round_index + 1} {variant}", flush=True)
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            report = json.loads(output.read_text(encoding="utf-8"))
            if report["measurement"]["standalone_variant"] != variant:
                raise AssertionError(f"{case_id}: standalone variant mismatch")
            if expected_contract is None:
                expected_contract = report["contract"]
            elif report["contract"] != expected_contract:
                raise AssertionError(f"{case_id}: contract changed between standalone runs")
            if variant == PHASE_VARIANTS[1] and not report["finiteness"]["all_finite"]:
                raise AssertionError(f"{case_id}: Phase 2 produced non-finite output/state")
            runs.append({
                "round": round_index + 1,
                "position": position + 1,
                "variant": variant,
                "output": str(output),
                "report": report,
            })
    aggregate = aggregate_standalone_runs(runs)
    if not all(item["workspace_consistent"] for item in aggregate["variants"].values()):
        raise AssertionError(f"{case_id}: workspace or peak allocation changed between runs")
    return {
        "contract": expected_contract,
        "measurement_mode": "standalone_clean_process_balanced",
        "round_count": args.standalone_rounds,
        "run_order": [
            {
                "round": run["round"],
                "position": run["position"],
                "variant": run["variant"],
                "output": run["output"],
            }
            for run in runs
        ],
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", 0)))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--measurement-mode", choices=("paired", "standalone"), default="paired")
    parser.add_argument("--standalone-rounds", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("ASCEND_LAUNCH_BLOCKING") not in (None, "", "0"):
        raise RuntimeError("Production performance acceptance requires ASCEND_LAUNCH_BLOCKING to be unset.")
    selected = set(args.case)
    unknown = selected - {case[0] for case in CASES}
    if unknown:
        raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
    if args.measurement_mode == "standalone" and (
        args.standalone_rounds < 2 or args.standalone_rounds % 2
    ):
        raise ValueError("standalone mode requires a positive even --standalone-rounds >= 2")

    benchmark = Path(__file__).with_name("benchmark_gdn_core_ablation.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "device": args.device,
        "warmup_rounds": args.warmup,
        "iterations_per_variant": args.iterations,
        "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
        "measurement_mode": args.measurement_mode,
        "standalone_rounds": args.standalone_rounds if args.measurement_mode == "standalone" else None,
        "cases": {},
    }
    for case in CASES:
        case_id = case[0]
        if selected and case_id not in selected:
            continue
        if args.measurement_mode == "standalone":
            summary["cases"][case_id] = run_standalone_case(benchmark, args, case)
            print(f"[{case_id}] PASS", flush=True)
            continue
        output = args.output_dir / f"{case_id}.json"
        command = benchmark_command(benchmark, args, case, output)
        print(f"[{case_id}] running", flush=True)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        report = json.loads(output.read_text(encoding="utf-8"))
        summary["cases"][case_id] = {
            "contract": report["contract"],
            "accuracy": report["accuracy"],
            "stage_accuracy": report["stage_accuracy"],
            "paired_latency": report["paired_latency"],
            "variants": report["variants"],
            "stage_variants": report["stage_variants"],
        }
        print(f"[{case_id}] PASS", flush=True)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
