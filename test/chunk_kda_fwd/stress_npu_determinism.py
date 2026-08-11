#!/usr/bin/env python3
"""Run fixed-input NPU-only binary determinism stress for chunk_kda_fwd."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chunk_kda_fwd repeatedly on one NPU and compare every attn_out "
            "bit against run 0. No CPU or GPU reference is executed."
        )
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case-id", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--tokens",
        type=int,
        help="override T for a shorter boundary stress, for example 128",
    )
    parser.add_argument(
        "--case-json",
        type=Path,
        default=Path(__file__).with_name("atk_chunk_kda_fwd.json"),
    )
    return parser.parse_args()


def _load_spec(path: Path, case_id: int) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))
    for case in cases:
        if int(case["id"]) != case_id:
            continue
        for item in case.get("inputs", []):
            if item.get("name") == "case_spec":
                return json.loads(item["range_values"])
        raise RuntimeError(f"case {case_id} has no case_spec in {path}")
    raise RuntimeError(f"case {case_id} not found in {path}")


def _integer_dtype(torch, tensor):
    return {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
        8: torch.int64,
    }[tensor.element_size()]


def _scalar_record(torch, tensor, index: tuple[int, ...]) -> dict:
    scalar = tensor[index].detach().cpu().contiguous().reshape(1)
    raw = int(scalar.view(_integer_dtype(torch, scalar)).item())
    raw &= (1 << (scalar.element_size() * 8)) - 1
    return {
        "index": list(index),
        "value": float(scalar.float().item()),
        "bits": f"0x{raw:0{scalar.element_size() * 2}x}",
    }


def _flat_to_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    for size in reversed(shape):
        result.append(flat_index % size)
        flat_index //= size
    return tuple(reversed(result))


def _max_abs_record(torch, tensor) -> dict:
    absolute = tensor.detach().abs()
    flat_index = int(absolute.reshape(-1).argmax().item())
    index = _flat_to_index(flat_index, tuple(tensor.shape))
    result = _scalar_record(torch, tensor, index)
    result["max_abs"] = float(absolute[index].float().item())
    del absolute
    return result


def _binary_difference(torch, current, baseline, baseline_bits) -> dict:
    current_bits = current.contiguous().view(_integer_dtype(torch, current))
    unequal = current_bits != baseline_bits
    head_counts = unequal.to(torch.float32).sum(dim=(0, 1, 3))
    mismatch_count = int(head_counts.sum().item())
    if mismatch_count == 0:
        return {"binary_equal": True, "mismatched_elements": 0}

    top_head = int(head_counts.argmax().item())
    top_head_count = int(head_counts[top_head].item())
    head_unequal = unequal[:, :, top_head, :]
    local_flat = int(head_unequal.to(torch.int8).reshape(-1).argmax().item())
    batch, token, channel = _flat_to_index(local_flat, tuple(head_unequal.shape))
    index = (batch, token, top_head, channel)
    max_abs_difference = float((current - baseline).abs().max().float().item())
    result = {
        "binary_equal": False,
        "mismatched_elements": mismatch_count,
        "top_head": top_head,
        "top_head_mismatched_elements": top_head_count,
        "first_mismatch_in_top_head": list(index),
        "current": _scalar_record(torch, current, index),
        "baseline": _scalar_record(torch, baseline, index),
        "max_abs_difference": max_abs_difference,
    }
    del current_bits, unequal, head_counts, head_unequal
    return result


def main() -> int:
    args = _parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")

    import torch
    import torch_npu  # noqa: F401

    from executor_chunk_kda_fwd import _prepare_inputs, _run_positive_npu

    spec = _load_spec(args.case_json.expanduser().resolve(), args.case_id)
    if args.tokens is not None:
        if args.tokens <= 0 or args.tokens % int(spec["chunk_size"]) != 0:
            raise ValueError("--tokens must be a positive multiple of chunk_size")
        spec["T"] = args.tokens
        spec["case_key"] = f"{spec['case_key']}_t{args.tokens}_stress"
    if str(spec["layout"]) != "BSND":
        raise ValueError("this stress diagnostic expects BSND attn_out")

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    low_marker = torch.zeros(1, dtype=torch.bfloat16, device=device)
    fp32_marker = torch.zeros(1, dtype=torch.float32, device=device)
    inputs = _prepare_inputs(spec, low_marker, fp32_marker, high_precision=False)

    print("KDA_NPU_BINARY_STRESS_BEGIN", flush=True)
    print(
        json.dumps(
            {
                "case_id": args.case_id,
                "case_key": spec["case_key"],
                "device": args.device,
                "repeats": args.repeats,
                "seed": int(spec["seed"]),
                "shape": {
                    name: int(spec[name])
                    for name in ("B", "H", "HV", "T", "K", "V", "chunk_size")
                },
                "comparison": "full_attn_out_bitwise_against_run_0",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    baseline = None
    baseline_bits = None
    mismatch_runs = []
    started = time.perf_counter()
    with torch.no_grad():
        for repeat in range(args.repeats):
            launch_started = time.perf_counter()
            outputs = _run_positive_npu(inputs, spec)
            attn_out = outputs[0]
            torch.npu.synchronize()
            if not torch.isfinite(attn_out).all().item():
                raise RuntimeError(f"run {repeat} attn_out contains NaN or Inf")

            record = {
                "run": repeat,
                "elapsed_seconds": time.perf_counter() - launch_started,
                "max_abs": _max_abs_record(torch, attn_out),
            }
            if baseline is None:
                baseline = attn_out.detach().clone()
                baseline_bits = baseline.contiguous().view(
                    _integer_dtype(torch, baseline)
                )
                record.update({"binary_equal": True, "baseline": True})
            else:
                difference = _binary_difference(
                    torch, attn_out, baseline, baseline_bits
                )
                record.update(difference)
                if not difference["binary_equal"]:
                    mismatch_runs.append(repeat)
            print("RUN " + json.dumps(record, sort_keys=True), flush=True)
            del outputs, attn_out
            gc.collect()

    print(
        "SUMMARY "
        + json.dumps(
            {
                "repeats": args.repeats,
                "equal_to_run_0": args.repeats - len(mismatch_runs),
                "different_from_run_0": len(mismatch_runs),
                "mismatch_runs": mismatch_runs,
                "elapsed_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("KDA_NPU_BINARY_STRESS_END", flush=True)
    return 1 if mismatch_runs else 0


if __name__ == "__main__":
    raise SystemExit(main())
