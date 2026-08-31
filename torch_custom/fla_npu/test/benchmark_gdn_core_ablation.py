# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------

"""Compare the legacy six-ACLNN GDN core with the final Phase6 fused core."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
import torch_npu

from fla_npu.ops import ascendc
from fla_npu.ops.ascendc import _runtime as ascendc_runtime


LEGACY_NAME = "legacy_six_aclnn"
PHASE6_NAME = "phase6_one_aclnn_fused_core"


def canonical_chunks(cu_seqlens: list[int] | None, chunk_size: int) -> list[int] | None:
    if cu_seqlens is None:
        return None
    result = []
    for sequence, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for local_chunk in range(math.ceil((end - begin) / chunk_size)):
            result.extend((sequence, local_chunk))
    return result


def make_inputs(args) -> dict:
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    q = (torch.randn(args.batch, args.key_heads, args.tokens, 128, dtype=dtype) * 0.05).npu()
    k = (torch.randn(args.batch, args.key_heads, args.tokens, 128, dtype=dtype) * 0.05).npu()
    repeat = args.value_heads // args.key_heads
    if repeat > 1 and not args.keep_grouped_qk:
        q = q.repeat_interleave(repeat, dim=1).contiguous()
        k = k.repeat_interleave(repeat, dim=1).contiguous()
    v = (
        torch.randn(args.batch, args.value_heads, args.tokens, args.value_dim, dtype=dtype) * 0.05
    ).npu()
    beta = torch.sigmoid(
        torch.randn(args.batch, args.tokens, args.value_heads, dtype=dtype, device="npu")
    )
    g = -torch.rand(
        args.batch,
        args.tokens,
        args.value_heads,
        dtype=torch.float32,
        device="npu",
    ) * 0.1

    cu_seqlens = None
    if args.cu_seqlens:
        cu_seqlens = [int(value) for value in args.cu_seqlens.split(",")]
        if args.batch != 1 or cu_seqlens[0] != 0 or cu_seqlens[-1] != args.tokens:
            raise ValueError("varlen requires batch=1 and cu_seqlens=[0,...,tokens]")
        if any(end <= begin for begin, end in zip(cu_seqlens, cu_seqlens[1:])):
            raise ValueError("cu_seqlens must be strictly increasing")

    chunk_indices = canonical_chunks(cu_seqlens, args.chunk_size)
    cu_seqlens_tensor = None
    chunk_indices_tensor = None
    if cu_seqlens is not None:
        cu_seqlens_tensor = torch.tensor(cu_seqlens, device="npu", dtype=torch.int64)
        chunk_indices_tensor = torch.tensor(
            chunk_indices,
            device="npu",
            dtype=torch.int64,
        ).view(-1, 2)

    initial_state = None
    if args.initial_state:
        sequence_count = args.batch if cu_seqlens is None else len(cu_seqlens) - 1
        initial_state = (
            torch.randn(
                sequence_count,
                args.value_heads,
                128,
                args.value_dim,
                dtype=torch.float32,
            ) * 0.01
        ).npu()

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "cu_seqlens_tensor": cu_seqlens_tensor,
        "chunk_indices_tensor": chunk_indices_tensor,
        "chunk_size": args.chunk_size,
        "scale": float(args.scale),
        "initial_state": initial_state,
        "output_final_state": args.output_final_state,
    }


def run_legacy(inputs: dict):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    cu_seqlens = inputs["cu_seqlens"]
    chunk_indices = inputs["chunk_indices"]
    chunk_size = inputs["chunk_size"]
    beta = inputs["beta"].transpose(1, 2).contiguous().float()
    g = ascendc.chunk_local_cumsum(
        inputs["g"].transpose(1, 2).contiguous(),
        chunk_size=chunk_size,
        cu_seqlens=inputs["cu_seqlens_tensor"],
        chunk_indices_out=inputs["chunk_indices_tensor"],
        head_first=True,
    )
    a_raw = ascendc.chunk_scaled_dot_kkt(
        k,
        g,
        beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    if cu_seqlens is None:
        a = ascendc.solve_tri(a_raw.to(k.dtype), layout="bhtd")
    else:
        a_token_first = a_raw.transpose(1, 2).contiguous().squeeze(0)
        a_token_first = ascendc.solve_tri(
            a_token_first.to(k.dtype),
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            layout="tnd",
        )
        a = a_token_first.unsqueeze(0).transpose(1, 2).contiguous()
    w, u = ascendc.recompute_w_u_fwd(
        k,
        v,
        beta,
        a,
        chunk_size,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = ascendc.chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=inputs["initial_state"],
        output_final_state=inputs["output_final_state"],
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    output = ascendc.chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        inputs["scale"],
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )
    return output, g.transpose(1, 2).contiguous(), a, final_state


def ascendc_gdn_function(name: str):
    if hasattr(ascendc, name):
        return getattr(ascendc, name)
    prefixed_name = f"npu_{name}"
    if hasattr(ascendc, prefixed_name):
        return getattr(ascendc, prefixed_name)
    raise AttributeError(f"module 'fla_npu.ops.ascendc' has no {name!r} or {prefixed_name!r}")


def run_phase6(inputs: dict):
    function = ascendc_gdn_function("gdn_core_fwd_phase6")
    output, final_state, g_cumsum, a = function(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=inputs["output_final_state"],
        chunk_size=inputs["chunk_size"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        scale=inputs["scale"],
    )
    return output, g_cumsum, a, final_state


def tensor_finiteness(tensor: torch.Tensor | None) -> dict:
    if tensor is None:
        return {
            "present": False,
            "element_count": 0,
            "non_finite_count": 0,
            "all_finite": True,
        }
    non_finite_count = int((~torch.isfinite(tensor.float())).sum().cpu())
    return {
        "present": True,
        "element_count": tensor.numel(),
        "non_finite_count": non_finite_count,
        "all_finite": non_finite_count == 0,
    }


def valid_a_chunks(tensor: torch.Tensor, inputs: dict):
    cu_seqlens = inputs["cu_seqlens"]
    sequences = [(batch, 0, tensor.shape[2]) for batch in range(tensor.shape[0])]
    if cu_seqlens is not None:
        sequences = [(0, begin, end) for begin, end in zip(cu_seqlens, cu_seqlens[1:])]
    for batch, begin, end in sequences:
        for chunk_begin in range(begin, end, inputs["chunk_size"]):
            chunk_end = min(chunk_begin + inputs["chunk_size"], end)
            chunk_len = chunk_end - chunk_begin
            yield tensor[batch, :, chunk_begin:chunk_end, :chunk_len]


def result_finiteness(outputs, inputs: dict) -> dict:
    valid_a_element_count = 0
    valid_a_non_finite_count = 0
    for chunk in valid_a_chunks(outputs[2], inputs):
        valid_a_element_count += chunk.numel()
        valid_a_non_finite_count += int((~torch.isfinite(chunk.float())).sum().cpu())
    components = {
        "output": tensor_finiteness(outputs[0]),
        "g_cumsum": tensor_finiteness(outputs[1]),
        "valid_a": {
            "present": True,
            "element_count": valid_a_element_count,
            "non_finite_count": valid_a_non_finite_count,
            "all_finite": valid_a_non_finite_count == 0,
        },
        "final_state": tensor_finiteness(outputs[3]),
    }
    return {
        "all_finite": all(component["all_finite"] for component in components.values()),
        "components": components,
    }


def compare_tensor(expected: torch.Tensor | None, actual: torch.Tensor | None) -> dict:
    if expected is None or actual is None:
        equal = expected is None and actual is None
        return {
            "bit_exact": equal,
            "element_count": 0,
            "mismatch_count": 0 if equal else 1,
            "max_abs": 0.0 if equal else float("inf"),
        }
    expected_cpu = expected.cpu()
    actual_cpu = actual.cpu()
    if expected_cpu.shape != actual_cpu.shape or expected_cpu.dtype != actual_cpu.dtype:
        return {
            "bit_exact": False,
            "element_count": expected_cpu.numel(),
            "mismatch_count": expected_cpu.numel(),
            "max_abs": float("inf"),
            "expected_shape": list(expected_cpu.shape),
            "actual_shape": list(actual_cpu.shape),
            "expected_dtype": str(expected_cpu.dtype),
            "actual_dtype": str(actual_cpu.dtype),
        }
    mismatch_count = int((expected_cpu != actual_cpu).sum())
    max_abs = float((expected_cpu.float() - actual_cpu.float()).abs().max())
    return {
        "bit_exact": mismatch_count == 0,
        "element_count": expected_cpu.numel(),
        "mismatch_count": mismatch_count,
        "max_abs": max_abs,
    }


def compare_valid_a(expected: torch.Tensor, actual: torch.Tensor, inputs: dict) -> dict:
    expected_chunks = list(valid_a_chunks(expected, inputs))
    actual_chunks = list(valid_a_chunks(actual, inputs))
    if len(expected_chunks) != len(actual_chunks):
        return {
            "bit_exact": False,
            "element_count": 0,
            "mismatch_count": 1,
            "max_abs": float("inf"),
        }
    element_count = 0
    mismatch_count = 0
    max_abs = 0.0
    for expected_chunk, actual_chunk in zip(expected_chunks, actual_chunks):
        comparison = compare_tensor(expected_chunk, actual_chunk)
        element_count += comparison["element_count"]
        mismatch_count += comparison["mismatch_count"]
        max_abs = max(max_abs, comparison["max_abs"])
    return {
        "bit_exact": mismatch_count == 0,
        "element_count": element_count,
        "mismatch_count": mismatch_count,
        "max_abs": max_abs,
    }


def compare_results(expected, actual, inputs: dict) -> dict:
    components = {
        "output": compare_tensor(expected[0], actual[0]),
        "g_cumsum": compare_tensor(expected[1], actual[1]),
        "valid_a": compare_valid_a(expected[2], actual[2], inputs),
        "final_state": compare_tensor(expected[3], actual[3]),
    }
    return {
        "bit_exact": all(component["bit_exact"] for component in components.values()),
        "components": components,
    }


def clear_allocator_state() -> None:
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()


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
        "peak_allocated_delta_bytes": max(0, peak - baseline),
        "aclnn_call_count": len(workspaces),
        "workspaces": workspaces,
        "workspace_max_bytes": max((item["bytes"] for item in workspaces), default=0),
        "workspace_sum_bytes": sum(item["bytes"] for item in workspaces),
    }
    del outputs
    clear_allocator_state()
    return result


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
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "samples_ms": samples,
    }


def percentage_summary(samples: list[float]) -> dict:
    return {
        "iterations": len(samples),
        "mean_pct": statistics.fmean(samples),
        "median_pct": statistics.median(samples),
        "p90_pct": percentile(samples, 0.90),
        "min_pct": min(samples),
        "samples_pct": samples,
    }


def run_synchronized(function, inputs: dict) -> None:
    outputs = function(inputs)
    torch.npu.synchronize()
    del outputs


def measure_paired_latency(inputs: dict, warmup: int, iterations: int) -> dict:
    functions = {
        LEGACY_NAME: run_legacy,
        PHASE6_NAME: run_phase6,
    }
    samples = {LEGACY_NAME: [], PHASE6_NAME: []}
    clear_allocator_state()
    for iteration in range(warmup):
        order = (LEGACY_NAME, PHASE6_NAME) if iteration % 2 == 0 else (PHASE6_NAME, LEGACY_NAME)
        for name in order:
            run_synchronized(functions[name], inputs)

    for iteration in range(iterations):
        order = (LEGACY_NAME, PHASE6_NAME) if iteration % 2 == 0 else (PHASE6_NAME, LEGACY_NAME)
        for name in order:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            outputs = functions[name](inputs)
            end.record()
            end.synchronize()
            elapsed = float(start.elapsed_time(end))
            if elapsed <= 0.0:
                raise RuntimeError(f"{name} produced a non-positive NPU Event duration: {elapsed}")
            samples[name].append(elapsed)
            del outputs

    legacy_summary = latency_summary(samples[LEGACY_NAME])
    phase6_summary = latency_summary(samples[PHASE6_NAME])
    pairwise_delta_ms = [
        phase6 - legacy
        for legacy, phase6 in zip(samples[LEGACY_NAME], samples[PHASE6_NAME])
    ]
    pairwise_change_pct = [
        (phase6 / legacy - 1.0) * 100.0
        for legacy, phase6 in zip(samples[LEGACY_NAME], samples[PHASE6_NAME])
    ]
    result = {
        "method": "ab_ba_alternating_npu_events",
        "order": (
            f"even rounds: {LEGACY_NAME} -> {PHASE6_NAME}; "
            f"odd rounds: {PHASE6_NAME} -> {LEGACY_NAME}"
        ),
        "warmup_rounds": warmup,
        "measurement_rounds": iterations,
        "results": {
            LEGACY_NAME: legacy_summary,
            PHASE6_NAME: phase6_summary,
        },
        "phase6_vs_legacy_median_change_pct": (
            phase6_summary["median_ms"] / legacy_summary["median_ms"] - 1.0
        ) * 100.0,
        "pairwise_phase6_minus_legacy_ms": latency_summary(pairwise_delta_ms),
        "pairwise_phase6_vs_legacy_change_pct": percentage_summary(pairwise_change_pct),
    }
    clear_allocator_state()
    return result


def trace_summary(trace_path: Path) -> dict:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    duration_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("ph") == "X"
    ]
    categories = Counter(str(event.get("cat", "")) for event in duration_events)
    arg_keys = Counter(
        key
        for event in duration_events
        for key in (event.get("args") or {}).keys()
    )
    device_events = []
    for event in duration_events:
        args = event.get("args") or {}
        if "Task Type" in args or "task type" in args:
            device_events.append(event)
    return {
        "device_kernel_count": len(device_events),
        "duration_event_count": len(duration_events),
        "duration_categories": dict(categories.most_common(20)),
        "duration_arg_keys": dict(arg_keys.most_common(20)),
        "trace_path": str(trace_path),
    }


def profile_variant(name: str, function, inputs: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{name}.json"
    clear_allocator_state()
    outputs = function(inputs)
    torch.npu.synchronize()
    del outputs
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        ),
    ) as profiler:
        outputs = function(inputs)
        torch.npu.synchronize()
        del outputs
    profiler.export_chrome_trace(str(trace_path.resolve()))
    clear_allocator_state()
    return trace_summary(trace_path)


def contract_report(args, inputs: dict) -> dict:
    return {
        "device": args.device,
        "batch": args.batch,
        "logical_key_heads": args.key_heads,
        "value_heads": args.value_heads,
        "physical_qk_heads": int(inputs["q"].shape[1]),
        "tokens": args.tokens,
        "k_dim": 128,
        "v_dim": args.value_dim,
        "chunk_size": args.chunk_size,
        "dtype": args.dtype,
        "scale": inputs["scale"],
        "cu_seqlens": inputs["cu_seqlens"],
        "initial_state": args.initial_state,
        "output_final_state": args.output_final_state,
        "seed": args.seed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", 0)))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--key-heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=8)
    parser.add_argument(
        "--keep-grouped-qk",
        action="store_true",
        help=(
            "Keep Q/K at --key-heads instead of expanding them to --value-heads. "
            "Use this for grouped-value model shapes."
        ),
    )
    parser.add_argument("--value-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--scale",
        type=float,
        default=128**-0.5,
        help="Attention scale passed as a Python float.",
    )
    parser.add_argument("--cu-seqlens", default="")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--initial-state", action="store_true")
    parser.add_argument("--output-final-state", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--output", type=Path, default=Path("gdn_core_phase6_benchmark.json"))
    args = parser.parse_args()

    if min(args.batch, args.key_heads, args.value_heads, args.tokens) <= 0:
        parser.error("batch, key-heads, value-heads and tokens must be positive")
    if args.value_heads % args.key_heads:
        parser.error("value-heads must be divisible by key-heads")
    if args.warmup <= 0 or args.iterations <= 0:
        parser.error("warmup and iterations must be positive")
    if args.warmup % 2 or args.iterations % 2:
        parser.error("warmup and iterations must be even for balanced AB/BA measurement")
    return args


def main() -> None:
    args = parse_args()
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    inputs = make_inputs(args)

    legacy = run_legacy(inputs)
    phase6 = run_phase6(inputs)
    torch.npu.synchronize()
    accuracy = compare_results(legacy, phase6, inputs)
    finiteness = {
        LEGACY_NAME: result_finiteness(legacy, inputs),
        PHASE6_NAME: result_finiteness(phase6, inputs),
    }
    if not accuracy["bit_exact"]:
        raise AssertionError(f"Phase6 is not bit exact with the legacy six ACLNN path: {accuracy}")
    if not all(result["all_finite"] for result in finiteness.values()):
        raise AssertionError(f"legacy/Phase6 produced non-finite output: {finiteness}")
    del legacy, phase6
    clear_allocator_state()

    functions = {
        LEGACY_NAME: run_legacy,
        PHASE6_NAME: run_phase6,
    }
    expected_calls = {
        LEGACY_NAME: 6,
        PHASE6_NAME: 1,
    }
    variants = {}
    for name, function in functions.items():
        result = measure_once(function, inputs)
        if result["aclnn_call_count"] != expected_calls[name]:
            raise AssertionError(
                f"{name}: expected {expected_calls[name]} ACLNN calls, "
                f"observed {result['aclnn_call_count']}"
            )
        if args.profile:
            result["profile"] = profile_variant(
                name,
                function,
                inputs,
                args.output.parent / "traces",
            )
        variants[name] = result

    paired_latency = measure_paired_latency(
        inputs,
        args.warmup,
        args.iterations,
    )
    report = {
        "schema_version": 1,
        "case_id": args.case_id,
        "measurement": {
            "method": "legacy_vs_phase6_ab_ba_npu_events",
            "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
            "warmup_rounds": args.warmup,
            "measurement_rounds": args.iterations,
            "profile_enabled": args.profile,
        },
        "contract": contract_report(args, inputs),
        "accuracy": {"phase6_vs_legacy": accuracy},
        "finiteness": finiteness,
        "expected_aclnn_call_count": expected_calls,
        "variants": variants,
        "paired_latency": paired_latency,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
