#!/usr/bin/env python3
"""A/B benchmark for the Phase 3 local-cumsum + KKT fusion."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import fla_npu
from fla_npu.ops import ascendc
from fla_npu.ops.ascendc import _runtime as ascendc_runtime


def canonical_chunk_indices(cu_seqlens: tuple[int, ...], chunk_size: int) -> tuple[int, ...]:
    indices = []
    for seq, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for local_chunk in range(math.ceil((end - begin) / chunk_size)):
            indices.extend((seq, local_chunk))
    return tuple(indices)


def local_cumsum_block_t(chunk_size: int) -> int:
    value = (1 << 17) // chunk_size
    return 1 << (max(value, 1) - 1).bit_length()


def local_cumsum_chunk_indices(
    cu_seqlens: tuple[int, ...], chunk_size: int
) -> torch.Tensor:
    block_t = local_cumsum_block_t(chunk_size)
    rows = []
    for seq, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for local_block in range(math.ceil((end - begin) / block_t)):
            rows.append((seq, local_block))
    return torch.tensor(rows, dtype=torch.int64)


def make_inputs(args) -> dict:
    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    k = (torch.randn(args.batch, args.heads, args.tokens, 128) * 0.2).to(dtype).npu()
    raw_g = (torch.randn(args.batch, args.heads, args.tokens) * 0.02).float().npu()
    beta = torch.sigmoid(torch.randn(args.batch, args.heads, args.tokens)).float().npu()
    cu_seqlens = None
    cu_seqlens_npu = None
    local_indices_npu = None
    fused_indices = None
    if args.cu_seqlens:
        cu_seqlens = tuple(int(value) for value in args.cu_seqlens.split(","))
        if cu_seqlens[0] != 0 or cu_seqlens[-1] != args.tokens:
            raise ValueError("cu_seqlens must start at 0 and end at --tokens")
        if args.batch != 1:
            raise ValueError("varlen physical layout requires --batch 1")
        cu_seqlens_npu = torch.tensor(cu_seqlens, dtype=torch.int64).npu()
        local_indices_npu = local_cumsum_chunk_indices(cu_seqlens, args.chunk_size).npu()
        fused_indices = canonical_chunk_indices(cu_seqlens, args.chunk_size)
    return {
        "k": k,
        "raw_g": raw_g,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_npu": cu_seqlens_npu,
        "local_indices_npu": local_indices_npu,
        "fused_indices": fused_indices,
        "chunk_size": args.chunk_size,
    }


def run_baseline(inputs: dict):
    g_cumsum = ascendc.chunk_local_cumsum(
        inputs["raw_g"],
        inputs["chunk_size"],
        cu_seqlens=inputs["cu_seqlens_npu"],
        chunk_indices_out=inputs["local_indices_npu"],
        reverse=False,
        scale=1.0,
        head_first=True,
        output_dtype="float32",
    )
    A = ascendc.chunk_scaled_dot_kkt(
        inputs["k"],
        g_cumsum,
        inputs["beta"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["fused_indices"],
        chunk_size=inputs["chunk_size"],
    )
    return g_cumsum, A


def run_fused(inputs: dict):
    return ascendc.chunk_cumsum_kkt(
        inputs["k"],
        inputs["raw_g"],
        inputs["beta"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["fused_indices"],
        chunk_size=inputs["chunk_size"],
    )


def clear_allocator_state() -> None:
    ascendc_runtime._RECENT_LAUNCH_STORAGE.clear()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()


@contextmanager
def capture_workspaces():
    runtime = ascendc_runtime.runtime()
    original_call = runtime.call
    records = []

    def wrapped(name, args, device, **kwargs):
        workspace = original_call(name, args, device, **kwargs)
        size = 0 if workspace is None else workspace.numel() * workspace.element_size()
        records.append({"name": name, "bytes": int(size)})
        return workspace

    runtime.call = wrapped
    try:
        yield records
    finally:
        runtime.call = original_call


def measure_once(function, inputs: dict) -> dict:
    clear_allocator_state()
    torch.npu.reset_peak_memory_stats()
    baseline = int(torch.npu.memory_allocated())
    with capture_workspaces() as workspaces:
        outputs = function(inputs)
        torch.npu.synchronize()
    peak = int(torch.npu.max_memory_allocated())
    result = {
        "aclnn_call_count": len(workspaces),
        "workspaces": workspaces,
        "workspace_max_bytes": max((item["bytes"] for item in workspaces), default=0),
        "workspace_sum_bytes": sum(item["bytes"] for item in workspaces),
        "peak_allocated_delta_bytes": max(0, peak - baseline),
    }
    del outputs
    clear_allocator_state()
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def summarize_samples(samples: list[float]) -> dict:
    return {
        "iterations": len(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 0.9),
        "min_ms": min(samples),
        "samples_ms": samples,
    }


def run_synchronized(function, inputs: dict) -> None:
    outputs = function(inputs)
    torch.npu.synchronize()
    del outputs
    ascendc_runtime._RECENT_LAUNCH_STORAGE.clear()


def measure_paired(inputs: dict, warmup: int, iterations: int) -> dict:
    functions = {"baseline_two_aclnn": run_baseline, "fused_one_aclnn": run_fused}
    samples = {name: [] for name in functions}
    names = tuple(functions)
    clear_allocator_state()
    for iteration in range(warmup):
        order = names if iteration % 2 == 0 else tuple(reversed(names))
        for name in order:
            run_synchronized(functions[name], inputs)
    for iteration in range(iterations):
        order = names if iteration % 2 == 0 else tuple(reversed(names))
        for name in order:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            outputs = functions[name](inputs)
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end)))
            del outputs
            ascendc_runtime._RECENT_LAUNCH_STORAGE.clear()
    summaries = {name: summarize_samples(values) for name, values in samples.items()}
    baseline = summaries["baseline_two_aclnn"]
    fused = summaries["fused_one_aclnn"]
    clear_allocator_state()
    return {
        "method": "paired_alternating_npu_events",
        "order": "even: baseline->fused; odd: fused->baseline",
        "warmup_rounds": warmup,
        "results": summaries,
        "fused_vs_baseline_median_change_pct":
            (fused["median_ms"] / baseline["median_ms"] - 1.0) * 100.0,
        "fused_vs_baseline_p90_change_pct":
            (fused["p90_ms"] / baseline["p90_ms"] - 1.0) * 100.0,
    }


def compare_outputs(inputs: dict) -> dict:
    baseline = run_baseline(inputs)
    fused = run_fused(inputs)
    torch.npu.synchronize()
    baseline_cpu = tuple(tensor.cpu() for tensor in baseline)
    fused_cpu = tuple(tensor.cpu() for tensor in fused)
    result = {}
    for name, expected, actual in zip(("g_cumsum", "A_raw"), baseline_cpu, fused_cpu):
        finite = bool(torch.isfinite(actual).all())
        exact = bool(torch.equal(actual, expected))
        result[name] = {
            "finite": finite,
            "bit_exact": exact,
            "max_abs": float((actual - expected).abs().max()),
            "numel": actual.numel(),
        }
        if not finite or not exact:
            raise AssertionError(f"{name} failed finite/bit-exact gate: {result[name]}")
    del baseline, fused, baseline_cpu, fused_cpu
    clear_allocator_state()
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_libraries() -> list[dict]:
    libraries = []
    for library in fla_npu.load_ascendc_opapi_libraries():
        path = Path(str(library._name)).resolve()
        libraries.append({"path": str(path), "sha256": file_sha256(path)})
    return libraries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=1025)
    parser.add_argument("--chunk-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--cu-seqlens")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iterations < 2:
        parser.error("--warmup must be >=1 and --iterations must be >=2")
    torch.npu.set_device(args.device)
    inputs = make_inputs(args)
    accuracy = compare_outputs(inputs)
    variants = {
        "baseline_two_aclnn": measure_once(run_baseline, inputs),
        "fused_one_aclnn": measure_once(run_fused, inputs),
    }
    if variants["baseline_two_aclnn"]["aclnn_call_count"] != 2:
        raise AssertionError("baseline must issue exactly two ACLNN calls")
    if variants["fused_one_aclnn"]["aclnn_call_count"] != 1:
        raise AssertionError("fused variant must issue exactly one ACLNN call")
    payload = {
        "case": {
            "layout": "varlen" if args.cu_seqlens else "dense",
            "dtype": args.dtype,
            "batch": args.batch,
            "heads": args.heads,
            "tokens": args.tokens,
            "chunk_size": args.chunk_size,
            "cu_seqlens": inputs["cu_seqlens"],
            "seed": args.seed,
        },
        "environment": {
            "device": args.device,
            "device_name": torch.npu.get_device_name(args.device),
            "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
            "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
            "installed_opapi_libraries": installed_libraries(),
        },
        "accuracy": accuracy,
        "variants": variants,
        "latency": measure_paired(inputs, args.warmup, args.iterations),
        "profiler_evidence": {
            "baseline_npu_kernel_count": 2,
            "fused_npu_kernel_count": 1,
            "source": "separate dense_fp16_c64 profiler smoke",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
