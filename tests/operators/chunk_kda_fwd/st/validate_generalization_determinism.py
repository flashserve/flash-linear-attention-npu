#!/usr/bin/env python3
"""Run 100 generalized KDA forward cases with 50-run determinism checks."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.reference.chunk_kda_reference import chunk_kda_forward_reference


LAYOUTS = ("BSND", "BNSD", "TND", "NTD")
HEADS = ((1, 1), (1, 2), (2, 2), (2, 4))
TOKENS = (17, 31, 63, 64, 65, 95, 127, 129)


def _case_specs(count, seed):
    rng = random.Random(seed)
    specs = []
    for case_id in range(count):
        layout = LAYOUTS[case_id % len(LAYOUTS)]
        chunk_size = 64 if (case_id // 32) % 2 == 0 else 128
        h, hv = HEADS[(case_id // 5) % len(HEADS)]
        t = TOKENS[(case_id * 5 + case_id // 7) % len(TOKENS)]
        if layout in {"TND", "NTD"}:
            b = 1
        else:
            b = 1 + int(case_id % 19 == 0)
        varlen = (
            layout in {"TND", "NTD"} and (case_id // 4) % 3 != 0
        )
        cu = None
        if varlen:
            seq_num = 2 + case_id % 3
            cuts = sorted(rng.sample(range(1, t), seq_num - 1))
            cu = [0, *cuts, t]
        output_policy = (case_id // 4) % 4
        specs.append(
            {
                "id": case_id,
                "layout": layout,
                "B": b,
                "T": t,
                "H": h,
                "HV": hv,
                "K": 128,
                "V": 256 if case_id % 5 == 0 else 128,
                "chunk_size": chunk_size,
                "dtype": (
                    "bfloat16" if (case_id // 4) % 2 == 0 else "float16"
                ),
                "safe_gate": bool((case_id // 8) % 2),
                "use_gate_in_kernel": bool((case_id // 16) % 2),
                "state_v_first": bool((case_id // 64) % 2),
                "initial_state": bool(case_id % 3),
                "disable_recompute": output_policy >= 2,
                "return_intermediate_states": output_policy % 2 == 1,
                "cu_seqlens": cu,
                "seed": seed + case_id * 97,
            }
        )
    return specs


def _layout_inputs(torch, spec, q, k, v, g, beta):
    layout = spec["layout"]
    if layout == "BSND":
        return q, k, v, g, beta
    if layout == "BNSD":
        return (
            q.permute(0, 2, 1, 3).contiguous(),
            k.permute(0, 2, 1, 3).contiguous(),
            v.permute(0, 2, 1, 3).contiguous(),
            g.permute(0, 2, 1, 3).contiguous(),
            beta.permute(0, 2, 1).contiguous(),
        )
    if layout == "TND":
        return q[0], k[0], v[0], g[0], beta[0]
    return (
        q[0].permute(1, 0, 2).contiguous(),
        k[0].permute(1, 0, 2).contiguous(),
        v[0].permute(1, 0, 2).contiguous(),
        g[0].permute(1, 0, 2).contiguous(),
        beta[0].permute(1, 0).contiguous(),
    )


def _chunk_cumsum(torch, step_gate, chunk_size, cu_seqlens):
    gk = torch.empty_like(step_gate, dtype=torch.float32)
    if cu_seqlens is None:
        spans = [
            (b, start, min(start + chunk_size, step_gate.shape[1]))
            for b in range(step_gate.shape[0])
            for start in range(0, step_gate.shape[1], chunk_size)
        ]
    else:
        spans = [
            (0, start, min(start + chunk_size, end))
            for seq_start, end in zip(cu_seqlens, cu_seqlens[1:])
            for start in range(seq_start, end, chunk_size)
        ]
    for b, start, end in spans:
        gk[b, start:end] = torch.cumsum(
            step_gate[b, start:end].float(), dim=0
        ) / math.log(2.0)
    return gk


def _head_major(tensor, rank3):
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    return tensor[0] if rank3 else tensor


def _expected_outputs(torch, spec, reference, gk, initial_state):
    rank3 = spec["layout"] in {"TND", "NTD"}
    state_v_first = spec["state_v_first"]
    final_state = reference.final_state
    h = reference.h
    if state_v_first:
        final_state = final_state.transpose(-1, -2).contiguous()
        h = h.transpose(-1, -2).contiguous()
    if rank3:
        h = h[0]
    keep_saved = spec["disable_recompute"]
    outputs = (
        reference.o[0] if rank3 else reference.o,
        final_state,
        (
            _head_major(gk, rank3)
            if not spec["use_gate_in_kernel"] or keep_saved
            else None
        ),
        _head_major(reference.Aqk, rank3),
        _head_major(reference.Akk, rank3),
        _head_major(reference.w, rank3) if keep_saved else None,
        _head_major(reference.u, rank3) if keep_saved else None,
        _head_major(reference.qg, rank3) if keep_saved else None,
        _head_major(reference.kg, rank3) if keep_saved else None,
        _head_major(reference.v_new, rank3) if keep_saved else None,
        (
            h
            if keep_saved or spec["return_intermediate_states"]
            else None
        ),
        initial_state,
    )
    return outputs


def _assert_close(torch, case_name, actual, expected, *, rtol, atol):
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{case_name}: optional output mismatch")
        return 0.0
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    actual_finite = torch.isfinite(actual_cpu)
    expected_finite = torch.isfinite(expected_cpu)
    if not actual_finite.all() or not expected_finite.all():
        raise AssertionError(
            f"{case_name}: non-finite values, "
            f"actual={int((~actual_finite).sum().item())}, "
            f"expected={int((~expected_finite).sum().item())}"
        )
    try:
        torch.testing.assert_close(
            actual_cpu,
            expected_cpu,
            rtol=rtol,
            atol=atol,
            msg=case_name,
        )
    except AssertionError:
        diff = (actual_cpu - expected_cpu).abs()
        tolerance = atol + rtol * expected_cpu.abs()
        bad = diff > tolerance
        first_indices = bad.nonzero(as_tuple=False)[:16]
        samples = []
        for index in first_indices:
            key = tuple(int(value) for value in index.tolist())
            samples.append(
                {
                    "index": list(key),
                    "actual": float(actual_cpu[key].item()),
                    "expected": float(expected_cpu[key].item()),
                    "abs_diff": float(diff[key].item()),
                }
            )
        largest_samples = []
        largest_count = min(16, diff.numel())
        _, largest_indices = torch.topk(
            diff.reshape(-1), k=largest_count
        )
        for flat_index in largest_indices.tolist():
            remainder = int(flat_index)
            key_values = []
            for size in reversed(diff.shape):
                key_values.append(remainder % size)
                remainder //= size
            key = tuple(reversed(key_values))
            largest_samples.append(
                {
                    "index": list(key),
                    "actual": float(actual_cpu[key].item()),
                    "expected": float(expected_cpu[key].item()),
                    "abs_diff": float(diff[key].item()),
                }
            )
        regions = []
        if actual_cpu.ndim == 3:
            row_edges = sorted({0, min(64, actual_cpu.shape[1]), actual_cpu.shape[1]})
            col_edges = sorted({0, min(128, actual_cpu.shape[2]), actual_cpu.shape[2]})
            for head in range(actual_cpu.shape[0]):
                for row_begin, row_end in zip(row_edges, row_edges[1:]):
                    for col_begin, col_end in zip(col_edges, col_edges[1:]):
                        region_diff = diff[
                            head, row_begin:row_end, col_begin:col_end
                        ]
                        if region_diff.numel() == 0:
                            continue
                        regions.append(
                            {
                                "head": head,
                                "rows": [row_begin, row_end],
                                "cols": [col_begin, col_end],
                                "max_abs": float(region_diff.max().item()),
                                "mean_abs": float(region_diff.mean().item()),
                            }
                        )
        print(
            json.dumps(
                {
                    "accuracy_failure": case_name,
                    "shape": list(actual_cpu.shape),
                    "dtype": str(actual.dtype).removeprefix("torch."),
                    "mismatched_elements": int(bad.sum().item()),
                    "max_abs": float(diff.max().item()),
                    "samples": samples,
                    "largest_samples": largest_samples,
                    "regions": regions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        raise
    return float((actual_cpu - expected_cpu).abs().max().item())


def _cpu_snapshot(outputs):
    return tuple(
        None
        if tensor is None
        else tensor.detach().cpu().contiguous()
        for tensor in outputs
    )


def _binary_equal_cpu(actual, baseline):
    for value, expected in zip(actual, baseline):
        if value is None or expected is None:
            if value is not expected:
                return False
            continue
        if not value.equal(expected):
            return False
    return True


def _binary_diff_details(torch, actual, baseline):
    details = []
    for output_index, (value, expected) in enumerate(
        zip(actual, baseline)
    ):
        if value is None or expected is None:
            if value is not expected:
                details.append(
                    {
                        "output": output_index,
                        "optional_output_mismatch": True,
                    }
                )
            continue
        value_cpu = value.detach().cpu().contiguous()
        expected_cpu = expected.detach().cpu().contiguous()
        unequal = value_cpu != expected_cpu
        unequal_count = int(unequal.sum().item())
        if unequal_count == 0:
            continue
        first_index_tensors = unequal.nonzero(as_tuple=False)[:16]
        first_indices = first_index_tensors.tolist()
        samples = []
        for index_tensor in first_index_tensors:
            index = tuple(int(item) for item in index_tensor.tolist())
            samples.append(
                {
                    "index": list(index),
                    "actual": float(value_cpu[index].float().item()),
                    "baseline": float(
                        expected_cpu[index].float().item()
                    ),
                }
            )
        diff = value_cpu.float() - expected_cpu.float()
        details.append(
            {
                "output": output_index,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "unequal_elements": unequal_count,
                "max_abs": float(diff.abs().max().item()),
                "first_indices": first_indices,
                "samples": samples,
            }
        )
    return details


def _run_case(
    torch,
    op,
    spec,
    device,
    repeats,
):
    torch.manual_seed(spec["seed"])
    B, T, H, HV, K, V = (
        spec[name] for name in ("B", "T", "H", "HV", "K", "V")
    )
    dtype = torch.bfloat16 if spec["dtype"] == "bfloat16" else torch.float16
    q = (torch.randn(B, T, H, K) * 0.03).to(dtype)
    k = (torch.randn(B, T, H, K) * 0.03).to(dtype)
    v = (torch.randn(B, T, HV, V) * 0.03).to(dtype)
    beta = torch.sigmoid(torch.randn(B, T, HV, dtype=torch.float32))
    raw = torch.randn(B, T, HV, K, dtype=torch.float32) * 0.2 - 1.0
    a_log = torch.randn(HV, dtype=torch.float32) * 0.03
    dt_bias = torch.randn(HV * K, dtype=torch.float32) * 0.05
    lower_bound = -5.0
    if spec["use_gate_in_kernel"]:
        shifted = raw + dt_bias.reshape(1, 1, HV, K)
        if spec["safe_gate"]:
            step_gate = lower_bound * torch.sigmoid(
                torch.exp(a_log).reshape(1, 1, HV, 1) * shifted
            )
        else:
            step_gate = -torch.exp(a_log).reshape(
                1, 1, HV, 1
            ) * torch.nn.functional.softplus(shifted)
        gate_input = raw
    else:
        step_gate = -torch.rand(B, T, HV, K, dtype=torch.float32) * 0.01
        gate_input = step_gate
    cu = spec["cu_seqlens"]
    cu_tensor = None if cu is None else torch.tensor(cu, dtype=torch.int64)
    gk = _chunk_cumsum(torch, step_gate, spec["chunk_size"], cu)
    seq_num = B if cu is None else len(cu) - 1
    initial_ref = (
        torch.randn(seq_num, HV, K, V, dtype=torch.float32) * 0.01
        if spec["initial_state"]
        else None
    )
    reference = chunk_kda_forward_reference(
        q,
        k,
        v,
        gk,
        beta,
        scale=K ** -0.5,
        chunk_size=spec["chunk_size"],
        initial_state=initial_ref,
        output_final_state=True,
        cu_seqlens=cu_tensor,
    )

    initial_input = initial_ref
    if initial_input is not None and spec["state_v_first"]:
        initial_input = initial_input.transpose(-1, -2).contiguous()
    expected = _expected_outputs(torch, spec, reference, gk, initial_input)
    ql, kl, vl, gl, bl = _layout_inputs(
        torch, spec, q, k, v, gate_input, beta
    )
    npu_inputs = [
        tensor.to(device)
        for tensor in (ql, kl, vl, gl, bl)
    ]
    initial_npu = None if initial_input is None else initial_input.to(device)
    a_log_npu = a_log.to(device) if spec["use_gate_in_kernel"] else None
    dt_bias_npu = dt_bias.to(device) if spec["use_gate_in_kernel"] else None

    kwargs = {
        "layout": spec["layout"],
        "initial_state": initial_npu,
        "output_final_state": True,
        "cu_seqlens": cu,
        "chunk_indices": None,
        "safe_gate": spec["safe_gate"],
        "lower_bound": lower_bound,
        "use_gate_in_kernel": spec["use_gate_in_kernel"],
        "A_log": a_log_npu,
        "dt_bias": dt_bias_npu,
        "disable_recompute": spec["disable_recompute"],
        "return_intermediate_states": spec["return_intermediate_states"],
        "state_v_first": spec["state_v_first"],
    }
    baseline = op(
        *npu_inputs,
        K ** -0.5,
        spec["chunk_size"],
        **kwargs,
    )
    torch.npu.synchronize()
    non_finite = []
    for index, (actual, expected_value) in enumerate(zip(baseline, expected)):
        if actual is None or expected_value is None:
            continue
        actual_count = int(
            (~torch.isfinite(actual.detach())).sum().item()
        )
        expected_count = int(
            (~torch.isfinite(expected_value.detach())).sum().item()
        )
        if actual_count or expected_count:
            actual_bad = (
                (~torch.isfinite(actual.detach()))
                .nonzero(as_tuple=False)[:16]
                .cpu()
                .tolist()
            )
            non_finite.append(
                {
                    "output": index,
                    "shape": list(actual.shape),
                    "actual": actual_count,
                    "expected": expected_count,
                    "actual_first_indices": actual_bad,
                }
            )
    if non_finite:
        print(
            json.dumps(
                {"case": spec, "non_finite": non_finite},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    tolerance = 6e-3 if spec["dtype"] == "bfloat16" else 3e-3
    max_abs = {
        str(index): _assert_close(
            torch,
            f"case {spec['id']} output {index}",
            actual,
            expected_value,
            rtol=tolerance,
            atol=tolerance,
        )
        for index, (actual, expected_value) in enumerate(zip(baseline, expected))
    }
    baseline_binary = _cpu_snapshot(baseline)
    for repeat in range(1, repeats):
        actual = op(
            *npu_inputs,
            K ** -0.5,
            spec["chunk_size"],
            **kwargs,
        )
        actual_binary = _cpu_snapshot(actual)
        if not _binary_equal_cpu(actual_binary, baseline_binary):
            details = _binary_diff_details(
                torch, actual_binary, baseline_binary
            )
            print(
                json.dumps(
                    {
                        "case": spec,
                        "repeat": repeat + 1,
                        "binary_differences": details,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            raise AssertionError(
                f"case {spec['id']} repeat "
                f"{repeat + 1}/{repeats} differs"
            )
    return {
        "id": spec["id"],
        "layout": spec["layout"],
        "dtype": spec["dtype"],
        "shape": {
            key: spec[key]
            for key in ("B", "T", "H", "HV", "K", "V", "chunk_size")
        },
        "safe_gate": spec["safe_gate"],
        "use_gate_in_kernel": spec["use_gate_in_kernel"],
        "disable_recompute": spec["disable_recompute"],
        "return_intermediate_states": spec["return_intermediate_states"],
        "state_v_first": spec["state_v_first"],
        "initial_state": spec["initial_state"],
        "varlen_sequences": 0 if cu is None else len(cu) - 1,
        "max_abs_by_output": max_abs,
        "binary_deterministic_runs": repeats,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if (
        args.cases < 1
        or args.repeats < 1
        or args.start_case < 0
    ):
        raise ValueError(
            "--cases and --repeats must be positive; "
            "--start-case must be non-negative"
        )

    import torch
    from fla_npu.ops.ascendc import chunk_kda_fwd

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("npu")
    results = []
    specs = _case_specs(args.start_case + args.cases, args.seed)[
        args.start_case:
    ]
    for spec in specs:
        result = _run_case(
            torch,
            chunk_kda_fwd,
            spec,
            device,
            args.repeats,
        )
        results.append(result)
        print(
            f"[{len(results):03d}/{args.cases:03d}] "
            f"{result['layout']} {result['dtype']} {result['shape']} PASS",
            flush=True,
        )

    summary = {
        "cases": args.cases,
        "start_case": args.start_case,
        "repeats_per_case": args.repeats,
        "total_executions": args.cases * args.repeats,
        "seed": args.seed,
        "passed": len(results),
        "failed": 0,
        "status": "PASS",
        "results": results,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
