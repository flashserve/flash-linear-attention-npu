# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------

"""Benchmark GDN core ablation variants on one reproducible input contract."""

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


STANDALONE_VARIANTS = (
    "legacy_six_aclnn",
    "phase1_one_aclnn_six_kernels",
    "phase2_one_aclnn_fused_kkt_solve",
    "phase3_one_aclnn_fused_cumsum_kkt",
    "phase4_one_aclnn_fused_fwd_ho",
    "phase5_one_aclnn_fused_recompute_wu_ho",
    "phase6_one_aclnn_fused_core",
)


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
    if args.value_heads % args.key_heads:
        raise ValueError("value_heads must be divisible by key_heads")
    repeat = args.value_heads // args.key_heads
    if repeat > 1:
        q = q.repeat_interleave(repeat, dim=1).contiguous()
        k = k.repeat_interleave(repeat, dim=1).contiguous()
    v = (
        torch.randn(args.batch, args.value_heads, args.tokens, args.value_dim, dtype=dtype) * 0.05
    ).npu()
    beta = torch.sigmoid(
        torch.randn(args.batch, args.tokens, args.value_heads, dtype=dtype, device="npu")
    )
    g = -torch.rand(
        args.batch, args.tokens, args.value_heads, dtype=torch.float32, device="npu"
    ) * 0.1
    cu_seqlens = None
    if args.cu_seqlens:
        cu_seqlens = [int(value) for value in args.cu_seqlens.split(",")]
        if args.batch != 1 or cu_seqlens[0] != 0 or cu_seqlens[-1] != args.tokens:
            raise ValueError("varlen requires batch=1 and cu_seqlens=[0,...,tokens]")
    chunk_indices = canonical_chunks(cu_seqlens, args.chunk_size)
    cu_seqlens_tensor = None
    chunk_indices_tensor = None
    if cu_seqlens is not None:
        cu_seqlens_tensor = torch.tensor(cu_seqlens, device="npu", dtype=torch.int64)
        chunk_indices_tensor = torch.tensor(
            chunk_indices, device="npu", dtype=torch.int64
        ).view(-1, 2)
    initial_state = None
    if args.initial_state:
        sequence_count = args.batch if cu_seqlens is None else len(cu_seqlens) - 1
        initial_state = (
            torch.randn(sequence_count, args.value_heads, 128, args.value_dim, dtype=torch.float32) * 0.01
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


def run_pipeline(inputs: dict, *, fused_kkt_solve: bool):
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
    if fused_kkt_solve:
        a = ascendc.chunk_kkt_solve_tri(
            k,
            g,
            beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )
    else:
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
        initial_state=inputs["initial_state"],
        output_final_state=inputs["output_final_state"],
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=False,
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


def make_kkt_solve_inputs(inputs: dict) -> dict:
    beta = inputs["beta"].transpose(1, 2).contiguous().float()
    g = ascendc.chunk_local_cumsum(
        inputs["g"].transpose(1, 2).contiguous(),
        chunk_size=inputs["chunk_size"],
        cu_seqlens=inputs["cu_seqlens_tensor"],
        chunk_indices_out=inputs["chunk_indices_tensor"],
        head_first=True,
    )
    torch.npu.synchronize()
    return {
        "k": inputs["k"],
        "g": g,
        "beta": beta,
        "cu_seqlens": inputs["cu_seqlens"],
        "chunk_indices": inputs["chunk_indices"],
        "chunk_size": inputs["chunk_size"],
    }


def run_legacy_kkt_solve(inputs: dict):
    a_raw = run_kkt_stage(inputs)
    if inputs["cu_seqlens"] is None:
        return ascendc.solve_tri(a_raw.to(inputs["k"].dtype), layout="bhtd")
    a_token_first = a_raw.transpose(1, 2).contiguous().squeeze(0)
    a_token_first = ascendc.solve_tri(
        a_token_first.to(inputs["k"].dtype),
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        layout="tnd",
    )
    return a_token_first.unsqueeze(0).transpose(1, 2).contiguous()


def run_kkt_stage(inputs: dict):
    return ascendc.chunk_scaled_dot_kkt(
        inputs["k"],
        inputs["g"],
        inputs["beta"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        chunk_size=inputs["chunk_size"],
    )


def run_fused_kkt_solve_stage(inputs: dict):
    return ascendc.chunk_kkt_solve_tri(
        inputs["k"],
        inputs["g"],
        inputs["beta"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        chunk_size=inputs["chunk_size"],
    )


def run_legacy(inputs: dict):
    return run_pipeline(inputs, fused_kkt_solve=False)


def run_fused_kkt_solve(inputs: dict):
    return run_pipeline(inputs, fused_kkt_solve=True)


def run_composite_with(function, inputs: dict):
    return_values = function(
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
    output, final_state, g_cumsum, a = return_values
    return output, g_cumsum, a, final_state


def ascendc_gdn_function(name: str):
    if hasattr(ascendc, name):
        return getattr(ascendc, name)
    prefixed_name = f"npu_{name}"
    if hasattr(ascendc, prefixed_name):
        return getattr(ascendc, prefixed_name)
    raise AttributeError(f"module 'fla_npu.ops.ascendc' has no {name!r} or {prefixed_name!r}")


def run_composite(inputs: dict):
    return run_composite_with(ascendc_gdn_function("gdn_core_fwd"), inputs)


def run_composite_phase1(inputs: dict):
    return run_composite_with(ascendc_gdn_function("gdn_core_fwd_phase1"), inputs)


def run_composite_phase2(inputs: dict):
    return run_composite_with(ascendc_gdn_function("gdn_core_fwd_phase2"), inputs)


def run_composite_phase3(inputs: dict):
    return run_composite_with(ascendc_gdn_function("gdn_core_fwd_phase3"), inputs)


def run_composite_phase4(inputs: dict):
    return run_composite_with(ascendc_gdn_function("gdn_core_fwd_phase4"), inputs)


def run_composite_phase5(inputs: dict):
    return run_composite_with(ascendc_gdn_function("gdn_core_fwd_phase5"), inputs)


def run_composite_phase6(inputs: dict):
    return run_composite_with(ascendc_gdn_function("gdn_core_fwd_phase6"), inputs)


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


def standalone_finiteness(outputs, inputs: dict) -> dict:
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


def compare_results(expected, actual, inputs: dict) -> dict:
    output_equal = torch.equal(expected[0].cpu(), actual[0].cpu())
    g_equal = torch.equal(expected[1].cpu(), actual[1].cpu())
    a_equal = all(
        torch.equal(left.cpu(), right.cpu())
        for left, right in zip(valid_a_chunks(expected[2], inputs), valid_a_chunks(actual[2], inputs))
    )
    max_abs = float((expected[0].float() - actual[0].float()).abs().max().cpu())
    expected_state = expected[3]
    actual_state = actual[3]
    state_equal = (
        expected_state is None and actual_state is None
        or expected_state is not None
        and actual_state is not None
        and torch.equal(expected_state.cpu(), actual_state.cpu())
    )
    return {
        "bit_exact": output_equal and g_equal and a_equal and state_equal,
        "output_bit_exact": output_equal,
        "g_bit_exact": g_equal,
        "valid_a_bit_exact": a_equal,
        "final_state_bit_exact": state_equal,
        "output_max_abs": max_abs,
    }


def cpu_inverse_reference(raw: torch.Tensor, inputs: dict) -> torch.Tensor:
    raw_cpu = raw.float().cpu()
    reference = torch.zeros_like(raw_cpu)
    sequences = [(batch, 0, raw.shape[2]) for batch in range(raw.shape[0])]
    if inputs["cu_seqlens"] is not None:
        sequences = [(0, begin, end) for begin, end in zip(
            inputs["cu_seqlens"], inputs["cu_seqlens"][1:]
        )]
    for batch, begin, end in sequences:
        for chunk_begin in range(begin, end, inputs["chunk_size"]):
            valid = min(inputs["chunk_size"], end - chunk_begin)
            identity = torch.eye(valid, dtype=torch.float64)
            for head in range(raw.shape[1]):
                block = raw_cpu[batch, head, chunk_begin:chunk_begin + valid, :valid].double()
                inverse = torch.linalg.inv(identity + block)
                reference[batch, head, chunk_begin:chunk_begin + valid, :valid] = inverse.float()
    return reference


def compare_a(expected: torch.Tensor, actual: torch.Tensor, inputs: dict,
              raw: torch.Tensor | None = None) -> dict:
    expected_chunks = list(valid_a_chunks(expected, inputs))
    actual_chunks = list(valid_a_chunks(actual, inputs))
    expected_finite = all(bool(torch.isfinite(chunk.float()).all()) for chunk in expected_chunks)
    actual_finite = all(bool(torch.isfinite(chunk.float()).all()) for chunk in actual_chunks)
    if not actual_finite:
        return {
            "passed": False,
            "comparison": "non_finite_fused_output",
            "valid_a_bit_exact": False,
            "valid_a_max_abs": float("inf"),
        }
    if expected_finite:
        bit_exact = all(
            torch.equal(left.cpu(), right.cpu())
            for left, right in zip(expected_chunks, actual_chunks)
        )
        max_abs = max(
            (float((left.float() - right.float()).abs().max().cpu())
             for left, right in zip(expected_chunks, actual_chunks)),
            default=0.0,
        )
        return {
            "passed": bit_exact,
            "comparison": "bit_exact_two_kernel_baseline",
            "valid_a_bit_exact": bit_exact,
            "valid_a_max_abs": max_abs,
        }
    if raw is None:
        raise ValueError("raw KKT output is required when the solve_tri baseline is non-finite")
    reference = cpu_inverse_reference(raw, inputs)
    reference_values = torch.cat([chunk.flatten() for chunk in valid_a_chunks(reference, inputs)])
    actual_values = torch.cat([
        chunk.float().cpu().flatten() for chunk in actual_chunks
    ])
    max_abs = float((reference_values - actual_values).abs().max())
    cosine = float(torch.nn.functional.cosine_similarity(reference_values, actual_values, dim=0))
    passed = max_abs <= 5e-3 and cosine >= 0.999
    return {
        "passed": passed,
        "comparison": "cpu_inverse_fallback",
        "baseline_non_finite_count": sum(
            int((~torch.isfinite(chunk.float())).sum().cpu()) for chunk in expected_chunks
        ),
        "valid_a_bit_exact": False,
        "valid_a_max_abs": max_abs,
        "valid_a_cosine": cosine,
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


def measure_latency(function, inputs: dict, warmup: int, iterations: int) -> dict:
    clear_allocator_state()
    for _ in range(warmup):
        run_synchronized(function, inputs)
    samples = []
    for _ in range(iterations):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        outputs = function(inputs)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del outputs
    result = latency_summary(samples)
    clear_allocator_state()
    return result


def measure_paired_latency(
    first_name: str,
    first_function,
    second_name: str,
    second_function,
    inputs: dict,
    warmup: int,
    iterations: int,
) -> dict:
    functions = {
        first_name: first_function,
        second_name: second_function,
    }
    samples = {first_name: [], second_name: []}
    clear_allocator_state()
    for iteration in range(warmup):
        order = (first_name, second_name) if iteration % 2 == 0 else (second_name, first_name)
        for name in order:
            run_synchronized(functions[name], inputs)

    for iteration in range(iterations):
        order = (first_name, second_name) if iteration % 2 == 0 else (second_name, first_name)
        for name in order:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            outputs = functions[name](inputs)
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end)))
            del outputs

    first_summary = latency_summary(samples[first_name])
    second_summary = latency_summary(samples[second_name])
    pairwise_delta_ms = [
        second - first
        for first, second in zip(samples[first_name], samples[second_name])
    ]
    pairwise_change_pct = [
        (second / first - 1.0) * 100.0
        for first, second in zip(samples[first_name], samples[second_name])
    ]
    result = {
        "method": "paired_alternating_npu_events",
        "order": (
            f"even rounds: {first_name} -> {second_name}; "
            f"odd rounds: {second_name} -> {first_name}"
        ),
        "warmup_rounds": warmup,
        "results": {
            first_name: first_summary,
            second_name: second_summary,
        },
        "second_vs_first_median_change_pct": (
            (second_summary["median_ms"] / first_summary["median_ms"] - 1.0) * 100.0
        ),
        "pairwise_second_minus_first_ms": latency_summary(pairwise_delta_ms),
        "pairwise_second_vs_first_change_pct": percentage_summary(pairwise_change_pct),
    }
    clear_allocator_state()
    return result


def balanced_variant_order(names: tuple[str, ...], iteration: int) -> tuple[str, ...]:
    """Rotate every variant through every position, then repeat in reverse order."""
    block = iteration // len(names)
    base = names if block % 2 == 0 else tuple(reversed(names))
    offset = iteration % len(names)
    return base[offset:] + base[:offset]


def measure_balanced_latency(
    functions: dict[str, object],
    inputs: dict,
    warmup: int,
    iterations: int,
) -> dict:
    names = tuple(functions)
    samples = {name: [] for name in names}
    clear_allocator_state()
    for iteration in range(warmup):
        for name in balanced_variant_order(names, iteration):
            try:
                run_synchronized(functions[name], inputs)
            except Exception as error:
                raise RuntimeError(
                    f"balanced warmup failed at iteration={iteration}, variant={name}"
                ) from error

    for iteration in range(iterations):
        for name in balanced_variant_order(names, iteration):
            try:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                outputs = functions[name](inputs)
                end.record()
                end.synchronize()
                samples[name].append(float(start.elapsed_time(end)))
                del outputs
            except Exception as error:
                raise RuntimeError(
                    f"balanced measurement failed at iteration={iteration}, variant={name}"
                ) from error

    summaries = {name: latency_summary(samples[name]) for name in names}
    baseline_name = names[0]
    versus_baseline = {}
    versus_previous = {}
    for index, name in enumerate(names[1:], start=1):
        baseline_pct = [
            (current / baseline - 1.0) * 100.0
            for baseline, current in zip(samples[baseline_name], samples[name])
        ]
        previous_name = names[index - 1]
        previous_pct = [
            (current / previous - 1.0) * 100.0
            for previous, current in zip(samples[previous_name], samples[name])
        ]
        versus_baseline[name] = {
            "median_change_pct": (
                summaries[name]["median_ms"] / summaries[baseline_name]["median_ms"] - 1.0
            ) * 100.0,
            "roundwise_change_pct": percentage_summary(baseline_pct),
        }
        versus_previous[name] = {
            "previous_variant": previous_name,
            "median_change_pct": (
                summaries[name]["median_ms"] / summaries[previous_name]["median_ms"] - 1.0
            ) * 100.0,
            "roundwise_change_pct": percentage_summary(previous_pct),
        }

    result = {
        "method": "balanced_rotating_npu_events",
        "order": (
            "rotate all variants through every position; reverse the base order "
            "after each complete rotation"
        ),
        "warmup_rounds": warmup,
        "variant_order": names,
        "results": summaries,
        "versus_baseline": versus_baseline,
        "versus_previous": versus_previous,
    }
    clear_allocator_state()
    return result


def trace_summary(trace_path: Path) -> dict:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    duration_events = [
        event for event in events
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
        category = str(event.get("cat", "")).lower()
        task_type = str(args.get("Task Type", args.get("task type", ""))).lower()
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
    function(inputs)
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        ),
    ) as profiler:
        function(inputs)
        torch.npu.synchronize()
    profiler.export_chrome_trace(str(trace_path.resolve()))
    clear_allocator_state()
    return trace_summary(trace_path)


def contract_report(args, inputs: dict) -> dict:
    return {
        "device": args.device,
        "batch": args.batch,
        "logical_key_heads": args.key_heads,
        "value_heads": args.value_heads,
        "physical_qk_heads": args.value_heads,
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


def run_standalone(args, inputs: dict) -> None:
    functions = {
        "legacy_six_aclnn": run_legacy,
        "phase1_one_aclnn_six_kernels": run_composite_phase1,
        "phase2_one_aclnn_fused_kkt_solve": run_composite_phase2,
        "phase3_one_aclnn_fused_cumsum_kkt": run_composite_phase3,
        "phase4_one_aclnn_fused_fwd_ho": run_composite_phase4,
        "phase5_one_aclnn_fused_recompute_wu_ho": run_composite_phase5,
        "phase6_one_aclnn_fused_core": run_composite_phase6,
    }
    function = functions[args.standalone_variant]
    outputs = function(inputs)
    torch.npu.synchronize()
    finiteness = standalone_finiteness(outputs, inputs)
    del outputs
    clear_allocator_state()

    result = measure_once(function, inputs)
    expected_aclnn_calls = 6 if args.standalone_variant == "legacy_six_aclnn" else 1
    if result["aclnn_call_count"] != expected_aclnn_calls:
        raise AssertionError(
            f"{args.standalone_variant}: expected {expected_aclnn_calls} ACLNN calls, "
            f"observed {result['aclnn_call_count']}"
        )
    result["latency"] = measure_latency(function, inputs, args.warmup, args.iterations)
    if args.profile:
        result["profile"] = profile_variant(
            args.standalone_variant, function, inputs, args.output.parent / "traces"
        )

    report = {
        "case_id": args.case_id,
        "measurement": {
            "method": "standalone_clean_process_npu_events",
            "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
            "warmup_rounds": args.warmup,
            "iterations_per_variant": args.iterations,
            "standalone_variant": args.standalone_variant,
        },
        "contract": contract_report(args, inputs),
        "finiteness": finiteness,
        "expected_aclnn_call_count": {args.standalone_variant: expected_aclnn_calls},
        "variants": {args.standalone_variant: result},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_phase3_accuracy(args, inputs: dict) -> None:
    phase2 = run_composite_phase2(inputs)
    phase3 = run_composite_phase3(inputs)
    torch.npu.synchronize()
    comparison = compare_results(phase2, phase3, inputs)
    finiteness = {
        "phase2_one_aclnn_fused_kkt_solve": standalone_finiteness(phase2, inputs),
        "phase3_one_aclnn_fused_cumsum_kkt": standalone_finiteness(phase3, inputs),
    }
    if not comparison["bit_exact"]:
        raise AssertionError(f"Phase 3 is not bit exact with Phase 2: {comparison}")
    if not all(item["all_finite"] for item in finiteness.values()):
        raise AssertionError(f"Phase 2/3 produced non-finite output/state: {finiteness}")
    report = {
        "case_id": args.case_id,
        "measurement": {"method": "phase2_phase3_accuracy_only"},
        "contract": contract_report(args, inputs),
        "accuracy": {"phase3_vs_phase2": comparison},
        "finiteness": finiteness,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_phase4_accuracy(args, inputs: dict) -> None:
    phase3 = run_composite_phase3(inputs)
    phase4 = run_composite_phase4(inputs)
    torch.npu.synchronize()
    comparison = compare_results(phase3, phase4, inputs)
    finiteness = {
        "phase3_one_aclnn_fused_cumsum_kkt": standalone_finiteness(phase3, inputs),
        "phase4_one_aclnn_fused_fwd_ho": standalone_finiteness(phase4, inputs),
    }
    if not comparison["bit_exact"]:
        raise AssertionError(f"Phase 4 is not bit exact with Phase 3: {comparison}")
    if not all(item["all_finite"] for item in finiteness.values()):
        raise AssertionError(f"Phase 3/4 produced non-finite output/state: {finiteness}")
    report = {
        "case_id": args.case_id,
        "measurement": {"method": "phase3_phase4_accuracy_only"},
        "contract": contract_report(args, inputs),
        "accuracy": {"phase4_vs_phase3": comparison},
        "finiteness": finiteness,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_phase5_accuracy(args, inputs: dict) -> None:
    phase4 = run_composite_phase4(inputs)
    phase5 = run_composite_phase5(inputs)
    torch.npu.synchronize()
    comparison = compare_results(phase4, phase5, inputs)
    finiteness = {
        "phase4_one_aclnn_fused_fwd_ho": standalone_finiteness(phase4, inputs),
        "phase5_one_aclnn_fused_recompute_wu_ho": standalone_finiteness(phase5, inputs),
    }
    if not comparison["bit_exact"]:
        raise AssertionError(f"Phase 5 is not bit exact with Phase 4: {comparison}")
    if not all(item["all_finite"] for item in finiteness.values()):
        raise AssertionError(f"Phase 4/5 produced non-finite output/state: {finiteness}")
    report = {
        "case_id": args.case_id,
        "measurement": {"method": "phase4_phase5_accuracy_only"},
        "contract": contract_report(args, inputs),
        "accuracy": {"phase5_vs_phase4": comparison},
        "finiteness": finiteness,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_phase6_accuracy(args, inputs: dict) -> None:
    phase5 = run_composite_phase5(inputs)
    phase6 = run_composite_phase6(inputs)
    torch.npu.synchronize()
    comparison = compare_results(phase5, phase6, inputs)
    finiteness = {
        "phase5_one_aclnn_fused_recompute_wu_ho": standalone_finiteness(phase5, inputs),
        "phase6_one_aclnn_fused_core": standalone_finiteness(phase6, inputs),
    }
    if not comparison["bit_exact"]:
        raise AssertionError(f"Phase 6 P0a is not bit exact with Phase 5: {comparison}")
    if not all(item["all_finite"] for item in finiteness.values()):
        raise AssertionError(f"Phase 5/6 produced non-finite output/state: {finiteness}")
    report = {
        "case_id": args.case_id,
        "measurement": {"method": "phase5_phase6_p0a_accuracy_only"},
        "contract": contract_report(args, inputs),
        "accuracy": {"phase6_vs_phase5": comparison},
        "finiteness": finiteness,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_phase4_paired(args, inputs: dict) -> None:
    functions = {
        "phase3_one_aclnn_fused_cumsum_kkt": run_composite_phase3,
        "phase4_one_aclnn_fused_fwd_ho": run_composite_phase4,
    }
    finiteness = {}
    workspaces = {}
    for name, function in functions.items():
        outputs = function(inputs)
        torch.npu.synchronize()
        finiteness[name] = standalone_finiteness(outputs, inputs)
        del outputs
        clear_allocator_state()
        workspaces[name] = measure_once(function, inputs)
        if workspaces[name]["aclnn_call_count"] != 1:
            raise AssertionError(
                f"{name}: expected 1 ACLNN call, "
                f"observed {workspaces[name]['aclnn_call_count']}"
            )
    if not all(item["all_finite"] for item in finiteness.values()):
        raise AssertionError(f"Phase 3/4 produced non-finite output/state: {finiteness}")

    paired = measure_paired_latency(
        "phase3_one_aclnn_fused_cumsum_kkt",
        run_composite_phase3,
        "phase4_one_aclnn_fused_fwd_ho",
        run_composite_phase4,
        inputs,
        args.warmup,
        args.iterations,
    )
    report = {
        "case_id": args.case_id,
        "measurement": {
            "method": "phase3_phase4_paired_alternating_npu_events",
            "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
        },
        "contract": contract_report(args, inputs),
        "finiteness": finiteness,
        "variants": workspaces,
        "paired_latency": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_phase5_paired(args, inputs: dict) -> None:
    functions = {
        "phase4_one_aclnn_fused_fwd_ho": run_composite_phase4,
        "phase5_one_aclnn_fused_recompute_wu_ho": run_composite_phase5,
    }
    finiteness = {}
    workspaces = {}
    for name, function in functions.items():
        outputs = function(inputs)
        torch.npu.synchronize()
        finiteness[name] = standalone_finiteness(outputs, inputs)
        del outputs
        clear_allocator_state()
        workspaces[name] = measure_once(function, inputs)
        if workspaces[name]["aclnn_call_count"] != 1:
            raise AssertionError(
                f"{name}: expected 1 ACLNN call, "
                f"observed {workspaces[name]['aclnn_call_count']}"
            )
    if not all(item["all_finite"] for item in finiteness.values()):
        raise AssertionError(f"Phase 4/5 produced non-finite output/state: {finiteness}")

    paired = measure_paired_latency(
        "phase4_one_aclnn_fused_fwd_ho",
        run_composite_phase4,
        "phase5_one_aclnn_fused_recompute_wu_ho",
        run_composite_phase5,
        inputs,
        args.warmup,
        args.iterations,
    )
    report = {
        "case_id": args.case_id,
        "measurement": {
            "method": "phase4_phase5_paired_alternating_npu_events",
            "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
        },
        "contract": contract_report(args, inputs),
        "finiteness": finiteness,
        "variants": workspaces,
        "paired_latency": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_phase6_paired(args, inputs: dict) -> None:
    functions = {
        "phase5_one_aclnn_fused_recompute_wu_ho": run_composite_phase5,
        "phase6_one_aclnn_fused_core": run_composite_phase6,
    }
    finiteness = {}
    workspaces = {}
    for name, function in functions.items():
        outputs = function(inputs)
        torch.npu.synchronize()
        finiteness[name] = standalone_finiteness(outputs, inputs)
        del outputs
        clear_allocator_state()
        workspaces[name] = measure_once(function, inputs)
        if workspaces[name]["aclnn_call_count"] != 1:
            raise AssertionError(
                f"{name}: expected 1 ACLNN call, "
                f"observed {workspaces[name]['aclnn_call_count']}"
            )
    if (not all(item["all_finite"] for item in finiteness.values()) and
            not args.allow_nonfinite_for_performance):
        raise AssertionError(f"Phase 5/6 produced non-finite output/state: {finiteness}")

    paired = measure_paired_latency(
        "phase5_one_aclnn_fused_recompute_wu_ho",
        run_composite_phase5,
        "phase6_one_aclnn_fused_core",
        run_composite_phase6,
        inputs,
        args.warmup,
        args.iterations,
    )
    report = {
        "case_id": args.case_id,
        "measurement": {
            "method": "phase5_phase6_paired_alternating_npu_events",
            "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
            "allow_nonfinite_for_performance": args.allow_nonfinite_for_performance,
        },
        "contract": contract_report(args, inputs),
        "finiteness": finiteness,
        "variants": workspaces,
        "paired_latency": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_all_phase_paired(args, inputs: dict) -> None:
    functions = {
        "legacy_six_aclnn": run_legacy,
        "phase1_one_aclnn_six_kernels": run_composite_phase1,
        "phase2_one_aclnn_fused_kkt_solve": run_composite_phase2,
        "phase3_one_aclnn_fused_cumsum_kkt": run_composite_phase3,
        "phase4_one_aclnn_fused_fwd_ho": run_composite_phase4,
    }
    expected_calls = {
        "legacy_six_aclnn": 6,
        "phase1_one_aclnn_six_kernels": 1,
        "phase2_one_aclnn_fused_kkt_solve": 1,
        "phase3_one_aclnn_fused_cumsum_kkt": 1,
        "phase4_one_aclnn_fused_fwd_ho": 1,
    }
    if args.all_phase_skip_phase1:
        del functions["phase1_one_aclnn_six_kernels"]
        del expected_calls["phase1_one_aclnn_six_kernels"]
    order_cycle = 2 * len(functions)
    if args.warmup % order_cycle or args.iterations % order_cycle:
        raise ValueError(
            "all-phase balanced measurement requires warmup and iterations "
            f"to be divisible by the forward/reverse order cycle ({order_cycle})"
        )

    reference = run_legacy(inputs)
    torch.npu.synchronize()
    finiteness = {
        "legacy_six_aclnn": standalone_finiteness(reference, inputs),
    }
    accuracy = {}
    for name, function in functions.items():
        if name != "legacy_six_aclnn":
            outputs = function(inputs)
            torch.npu.synchronize()
            accuracy[name] = compare_results(reference, outputs, inputs)
            finiteness[name] = standalone_finiteness(outputs, inputs)
            del outputs
            clear_allocator_state()
    del reference
    clear_allocator_state()

    variants = {}
    for name, function in functions.items():
        variants[name] = measure_once(function, inputs)
        if variants[name]["aclnn_call_count"] != expected_calls[name]:
            raise AssertionError(
                f"{name}: expected {expected_calls[name]} ACLNN calls, "
                f"observed {variants[name]['aclnn_call_count']}"
            )

    failed_accuracy = {
        name: result for name, result in accuracy.items() if not result["bit_exact"]
    }
    if failed_accuracy:
        raise AssertionError(f"all-phase accuracy failed: {failed_accuracy}")
    if not all(result["all_finite"] for result in finiteness.values()):
        raise AssertionError(f"all-phase finiteness failed: {finiteness}")

    balanced = measure_balanced_latency(
        functions,
        inputs,
        args.warmup,
        args.iterations,
    )
    report = {
        "case_id": args.case_id,
        "measurement": {
            "method": "all_phase_balanced_rotating_npu_events",
            "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
            "warmup_rounds": args.warmup,
            "iterations_per_variant": args.iterations,
        },
        "contract": contract_report(args, inputs),
        "accuracy": accuracy,
        "finiteness": finiteness,
        "expected_aclnn_call_count": expected_calls,
        "variants": variants,
        "balanced_latency": balanced,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", 0)))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--key-heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=8)
    parser.add_argument("--value-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--scale",
        type=float,
        default=128**-0.5,
        help="Attention scale passed as a Python float; defaults to the historical 1/sqrt(128).",
    )
    parser.add_argument("--cu-seqlens", default="")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--initial-state", action="store_true")
    parser.add_argument("--output-final-state", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--paired-only",
        action="store_true",
        help="Skip fixed-order latency and use only paired Phase 1/2 and stage measurements.",
    )
    parser.add_argument(
        "--standalone-variant",
        choices=STANDALONE_VARIANTS,
        default="",
        help="Run exactly one versioned composite variant in this process.",
    )
    parser.add_argument(
        "--phase3-accuracy-only",
        action="store_true",
        help="Compare only the immutable Phase 2 and Phase 3 core checkpoints.",
    )
    parser.add_argument(
        "--phase4-accuracy-only",
        action="store_true",
        help="Compare only the immutable Phase 3 checkpoint and the Phase 4 pilot.",
    )
    parser.add_argument(
        "--phase5-accuracy-only",
        action="store_true",
        help="Compare only the Phase 4 checkpoint and the Phase 5 D+(E+F) pilot.",
    )
    parser.add_argument(
        "--phase6-accuracy-only",
        action="store_true",
        help="Compare only the Phase 5 checkpoint and the restricted Phase 6 P0a pilot.",
    )
    parser.add_argument(
        "--phase4-paired-only",
        action="store_true",
        help="Measure only Phase 3 versus Phase 4 with alternating in-process NPU events.",
    )
    parser.add_argument(
        "--phase5-paired-only",
        action="store_true",
        help="Measure only Phase 4 versus Phase 5 with alternating in-process NPU events.",
    )
    parser.add_argument(
        "--phase6-paired-only",
        action="store_true",
        help="Measure only Phase 5 versus Phase 6 with alternating in-process NPU events.",
    )
    parser.add_argument(
        "--allow-nonfinite-for-performance",
        action="store_true",
        help=(
            "Record rather than reject non-finite outputs in paired Phase 5/6 performance-only "
            "runs. This never constitutes an accuracy pass."
        ),
    )
    parser.add_argument(
        "--all-phase-paired-only",
        action="store_true",
        help=(
            "Measure Phase 0 and immutable Phase 1/2/3/4 on one input with "
            "balanced in-process NPU events."
        ),
    )
    parser.add_argument(
        "--all-phase-skip-phase1",
        action="store_true",
        help=(
            "Exclude the historically unsafe repeated Phase 1 variant from an all-phase run. "
            "This is intended only for varlen performance evidence."
        ),
    )
    parser.add_argument("--case-id", default="")
    parser.add_argument("--output", type=Path, default=Path("gdn_core_ablation.json"))
    args = parser.parse_args()

    selected_modes = sum(bool(value) for value in (
        args.paired_only,
        args.standalone_variant,
        args.phase3_accuracy_only,
        args.phase4_accuracy_only,
        args.phase5_accuracy_only,
        args.phase6_accuracy_only,
        args.phase4_paired_only,
        args.phase5_paired_only,
        args.phase6_paired_only,
        args.all_phase_paired_only,
    ))
    if selected_modes > 1:
        parser.error(
            "the accuracy, paired, standalone and all-phase modes are mutually exclusive"
        )
    if args.all_phase_skip_phase1 and not args.all_phase_paired_only:
        parser.error("--all-phase-skip-phase1 requires --all-phase-paired-only")

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    inputs = make_inputs(args)
    if args.standalone_variant:
        run_standalone(args, inputs)
        return
    if args.phase3_accuracy_only:
        run_phase3_accuracy(args, inputs)
        return
    if args.phase4_accuracy_only:
        run_phase4_accuracy(args, inputs)
        return
    if args.phase5_accuracy_only:
        run_phase5_accuracy(args, inputs)
        return
    if args.phase6_accuracy_only:
        run_phase6_accuracy(args, inputs)
        return
    if args.phase4_paired_only:
        run_phase4_paired(args, inputs)
        return
    if args.phase5_paired_only:
        run_phase5_paired(args, inputs)
        return
    if args.phase6_paired_only:
        run_phase6_paired(args, inputs)
        return
    if args.all_phase_paired_only:
        run_all_phase_paired(args, inputs)
        return
    kkt_solve_inputs = make_kkt_solve_inputs(inputs)
    legacy = run_legacy(inputs)
    composite = run_composite(inputs)
    composite_phase1 = run_composite_phase1(inputs)
    composite_phase2 = run_composite_phase2(inputs)
    composite_phase3 = run_composite_phase3(inputs)
    fused = run_fused_kkt_solve(inputs)
    torch.npu.synchronize()
    accuracy = {
        "composite_one_aclnn": compare_results(legacy, composite, inputs),
        "phase1_one_aclnn_six_kernels": compare_results(legacy, composite_phase1, inputs),
        "phase2_one_aclnn_fused_kkt_solve": compare_results(legacy, composite_phase2, inputs),
        "phase3_one_aclnn_fused_cumsum_kkt": compare_results(legacy, composite_phase3, inputs),
        "fused_kkt_solve": compare_results(legacy, fused, inputs),
    }
    for name, comparison in accuracy.items():
        if not comparison["bit_exact"]:
            raise AssertionError(f"{name} is not bit exact: {comparison}")
    del legacy, composite, composite_phase1, composite_phase2, composite_phase3, fused

    stage_raw = run_kkt_stage(kkt_solve_inputs)
    if kkt_solve_inputs["cu_seqlens"] is None:
        stage_legacy = ascendc.solve_tri(stage_raw.to(kkt_solve_inputs["k"].dtype), layout="bhtd")
    else:
        stage_token_first = stage_raw.transpose(1, 2).contiguous().squeeze(0)
        stage_token_first = ascendc.solve_tri(
            stage_token_first.to(kkt_solve_inputs["k"].dtype),
            cu_seqlens=kkt_solve_inputs["cu_seqlens"],
            chunk_indices=kkt_solve_inputs["chunk_indices"],
            layout="tnd",
        )
        stage_legacy = stage_token_first.unsqueeze(0).transpose(1, 2).contiguous()
    stage_fused = run_fused_kkt_solve_stage(kkt_solve_inputs)
    torch.npu.synchronize()
    stage_accuracy = compare_a(stage_legacy, stage_fused, inputs, stage_raw)
    if not stage_accuracy["passed"]:
        raise AssertionError(f"fused KKT + solve_tri failed accuracy: {stage_accuracy}")
    del stage_raw, stage_legacy, stage_fused

    variants = {
        "legacy_six_aclnn": run_legacy,
        "composite_one_aclnn": run_composite,
        "phase1_one_aclnn_six_kernels": run_composite_phase1,
        "phase2_one_aclnn_fused_kkt_solve": run_composite_phase2,
        "phase3_one_aclnn_fused_cumsum_kkt": run_composite_phase3,
        "fused_kkt_solve": run_fused_kkt_solve,
    }
    results = {}
    for name, function in variants.items():
        result = measure_once(function, inputs)
        if not args.paired_only:
            result["latency"] = measure_latency(function, inputs, args.warmup, args.iterations)
        if args.profile:
            result["profile"] = profile_variant(name, function, inputs, args.output.parent / "traces")
        results[name] = result

    stage_variants = {
        "legacy_kkt_then_solve_tri": run_legacy_kkt_solve,
        "fused_kkt_solve_tri": run_fused_kkt_solve_stage,
    }
    stage_results = {}
    for name, function in stage_variants.items():
        result = measure_once(function, kkt_solve_inputs)
        if not args.paired_only:
            result["latency"] = measure_latency(function, kkt_solve_inputs, args.warmup, args.iterations)
        if args.profile:
            result["profile"] = profile_variant(
                f"stage_{name}", function, kkt_solve_inputs, args.output.parent / "traces"
            )
        stage_results[name] = result

    expected_stage_calls = {
        "legacy_kkt_then_solve_tri": 2,
        "fused_kkt_solve_tri": 1,
    }
    for name, expected in expected_stage_calls.items():
        actual = stage_results[name]["aclnn_call_count"]
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected} ACLNN calls, observed {actual}")

    expected_calls = {
        "legacy_six_aclnn": 6,
        "composite_one_aclnn": 1,
        "phase1_one_aclnn_six_kernels": 1,
        "phase2_one_aclnn_fused_kkt_solve": 1,
        "phase3_one_aclnn_fused_cumsum_kkt": 1,
        "fused_kkt_solve": 5,
    }
    for name, expected in expected_calls.items():
        actual = results[name]["aclnn_call_count"]
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected} ACLNN calls, observed {actual}")

    paired_latency = {
        "core_phase1_vs_phase2": measure_paired_latency(
            "phase1_one_aclnn_six_kernels",
            run_composite_phase1,
            "phase2_one_aclnn_fused_kkt_solve",
            run_composite_phase2,
            inputs,
            args.warmup,
            args.iterations,
        ),
        "core_phase2_vs_phase3": measure_paired_latency(
            "phase2_one_aclnn_fused_kkt_solve",
            run_composite_phase2,
            "phase3_one_aclnn_fused_cumsum_kkt",
            run_composite_phase3,
            inputs,
            args.warmup,
            args.iterations,
        ),
        "stage_legacy_vs_fused": measure_paired_latency(
            "legacy_kkt_then_solve_tri",
            run_legacy_kkt_solve,
            "fused_kkt_solve_tri",
            run_fused_kkt_solve_stage,
            kkt_solve_inputs,
            args.warmup,
            args.iterations,
        ),
    }

    report = {
        "case_id": args.case_id,
        "measurement": {
            "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
            "warmup_rounds": args.warmup,
            "iterations_per_variant": args.iterations,
            "paired_only": args.paired_only,
        },
        "contract": contract_report(args, inputs),
        "accuracy": accuracy,
        "stage_accuracy": stage_accuracy,
        "expected_aclnn_call_count": expected_calls,
        "expected_stage_aclnn_call_count": expected_stage_calls,
        "variants": results,
        "stage_variants": stage_results,
        "paired_latency": paired_latency,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
