#!/usr/bin/env python3
"""Generated recurrent_kda accuracy and launch-time coverage runner."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import npu_recurrent_kda as recurrent_kda
from tests.reference.recurrent_kda_reference import recurrent_kda_reference


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

SUPPORTED_KV = ((128, 128), (128, 256))
BOOL_COMBOS = tuple(
    {
        "use_gate_in_kernel": use_gate,
        "safe_gate": safe_gate,
        "use_beta_sigmoid_in_kernel": beta_sigmoid,
        "allow_neg_eigval": allow_neg,
        "use_qk_l2norm_in_kernel": qk_norm,
        "output_final_state": output_state,
        "initial_state": initial_state,
    }
    for use_gate in (False, True)
    for safe_gate in ((False, True) if use_gate else (False,))
    for beta_sigmoid in (False, True)
    for allow_neg in (False, True)
    for qk_norm in (False, True)
    for output_state in (False, True)
    for initial_state in (False, True)
)
HEAD_PROFILES = (
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (1, 32),
    (1, 64),
    (1, 128),
    (1, 256),
    (2, 2),
    (2, 4),
    (2, 8),
    (2, 16),
    (2, 32),
    (2, 64),
    (2, 128),
    (2, 256),
    (4, 4),
    (4, 8),
    (4, 16),
    (4, 32),
    (4, 64),
    (4, 128),
    (4, 256),
    (8, 8),
    (8, 16),
    (8, 32),
    (8, 64),
    (8, 128),
    (8, 256),
    (16, 16),
    (16, 32),
    (16, 64),
    (16, 128),
    (16, 256),
    (32, 32),
    (32, 64),
    (32, 128),
    (32, 256),
    (64, 64),
    (64, 128),
    (64, 256),
    (96, 96),
    (128, 128),
    (128, 256),
    (256, 256),
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return float(ordered[idx])


def bucket_power2(value: int, cap: int = 128) -> int:
    value = max(1, int(value))
    return min(cap, 2 ** int(math.ceil(math.log2(value))))


def make_lengths(rng: random.Random, seq_num: int, allow_zero: bool) -> list[int]:
    low = 0 if allow_zero else 1
    lengths = [rng.randint(low, 8) for _ in range(seq_num)]
    if sum(lengths) == 0:
        lengths[rng.randrange(seq_num)] = rng.randint(1, 8)
    return lengths


def make_cu(lengths: list[int]) -> list[int]:
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + int(length))
    return cu


def choose_heads(i: int, rng: random.Random) -> tuple[int, int]:
    if i % 100 == 0:
        return 96, 96
    if i % 73 == 0:
        return rng.choice([(1, 256), (8, 256), (64, 256), (128, 256), (256, 256)])
    return HEAD_PROFILES[i % len(HEAD_PROFILES)]


def choose_seq_num(i: int, rng: random.Random, value_heads: int, value_dim: int) -> int:
    if value_heads <= 8 and value_dim == 128 and i % 53 == 1:
        return 128
    if value_heads >= 128 or value_dim == 256:
        choices = [2, 3, 4, 5, 8, 12]
    elif value_heads >= 32:
        choices = [2, 3, 4, 5, 8, 12, 16, 24]
    else:
        choices = [2, 3, 4, 5, 8, 12, 16, 32, 64, 96]
        if i % 79 == 0:
            return 128
    return rng.choice(choices)


def choose_bool_combo(i: int) -> dict[str, bool]:
    combo_idx = (i + i // len(BOOL_COMBOS)) % len(BOOL_COMBOS)
    return dict(BOOL_COMBOS[combo_idx])


def bool_combo_label_from_values(values: dict[str, bool]) -> str:
    return (
        f"gate={int(values['use_gate_in_kernel'])},"
        f"safe={int(values['safe_gate'])},"
        f"beta_sigmoid={int(values['use_beta_sigmoid_in_kernel'])},"
        f"allow_neg={int(values['allow_neg_eigval'])},"
        f"qk_norm={int(values['use_qk_l2norm_in_kernel'])},"
        f"output_state={int(values['output_final_state'])},"
        f"initial_state={int(values['initial_state'])}"
    )


def bool_combo_label(case: dict[str, Any]) -> str:
    attrs = case["attrs"]
    return bool_combo_label_from_values({
        "use_gate_in_kernel": bool(attrs["use_gate_in_kernel"]),
        "safe_gate": bool(attrs["safe_gate"]),
        "use_beta_sigmoid_in_kernel": bool(attrs["use_beta_sigmoid_in_kernel"]),
        "allow_neg_eigval": bool(attrs["allow_neg_eigval"]),
        "use_qk_l2norm_in_kernel": bool(attrs["use_qk_l2norm_in_kernel"]),
        "output_final_state": bool(attrs["output_final_state"]),
        "initial_state": bool(case["optional"]["initial_state"]),
    })


def choose_shape(i: int, rng: random.Random, layout: str, varlen: bool, ssm: bool) -> dict[str, Any]:
    h, hv = choose_heads(i, rng)
    kdim, vdim = SUPPORTED_KV[(i // 7) % len(SUPPORTED_KV)]
    heavy_shape = hv >= 128 or vdim == 256
    if layout == "BSND" and not varlen:
        batch = rng.choice([1, 2, 3, 4] if not heavy_shape else [1, 2])
        seq_len = 8 if i % 41 == 0 else rng.randint(1, 8)
        return {"B": batch, "T": seq_len, "H": h, "H_v": hv, "K": kdim, "V": vdim,
                "cu_seqlens": None, "seq_num": batch}
    if layout == "BSND":
        seq_num = choose_seq_num(i, rng, hv, vdim)
        lengths = make_lengths(rng, seq_num, allow_zero=not ssm and (i % 5 == 0))
        total = sum(lengths)
        return {"B": 1, "T": total, "H": h, "H_v": hv, "K": kdim, "V": vdim,
                "cu_seqlens": make_cu(lengths), "seq_num": seq_num}
    if layout == "TND" and not varlen:
        total = 8 if i % 43 == 0 else rng.randint(1, 8)
        return {"T_total": total, "H": h, "H_v": hv, "K": kdim, "V": vdim,
                "cu_seqlens": None, "seq_num": 1}
    seq_num = choose_seq_num(i, rng, hv, vdim)
    lengths = make_lengths(rng, seq_num, allow_zero=not ssm and (i % 7 == 0))
    return {"T_total": sum(lengths), "H": h, "H_v": hv, "K": kdim, "V": vdim,
            "cu_seqlens": make_cu(lengths), "seq_num": seq_num}


def generate_case(i: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed * 1009 + i * 9176)
    combo = choose_bool_combo(i)
    layout = "BSND" if i % 2 == 0 else "TND"
    varlen = (i % 3) != 0
    dt_kind = "none"
    if combo["use_gate_in_kernel"]:
        dt_kind = ["none", "matrix", "flat"][(i // 4) % 3]
    ssm = varlen and (i % 11 in (3, 8))
    shape = choose_shape(i, rng, layout, varlen, ssm)
    if (int(shape["K"]), int(shape["V"])) not in SUPPORTED_KV:
        raise AssertionError(f"unsupported K/V generated: K={shape['K']}, V={shape['V']}")
    if int(shape["H_v"]) % int(shape["H"]) != 0:
        raise AssertionError(f"invalid head mapping generated: H={shape['H']}, HV={shape['H_v']}")
    return {
        "id": f"generated_{i:04d}",
        "seed": seed + i,
        "layout": layout,
        "shape": shape,
        "attrs": {
            "layout": layout,
            "scale": None if (i % 4) else rng.choice([0.125, 0.25, 0.5, 0.75]),
            "output_final_state": combo["output_final_state"],
            "use_qk_l2norm_in_kernel": combo["use_qk_l2norm_in_kernel"],
            "use_gate_in_kernel": combo["use_gate_in_kernel"],
            "use_beta_sigmoid_in_kernel": combo["use_beta_sigmoid_in_kernel"],
            "allow_neg_eigval": combo["allow_neg_eigval"],
            "safe_gate": combo["safe_gate"],
            "lower_bound": rng.choice([-5.0, -4.0, -1.0, -0.5, -0.05]) if combo["safe_gate"] else -5.0,
            "state_v_first": True,
        },
        "dtype": {
            "q_k_v": "bfloat16",
            "g": rng.choice(["float32", "bfloat16", "float16"]),
            "beta": rng.choice(["float32", "bfloat16", "float16"]),
            "state": rng.choice(["float32", "bfloat16"]),
        },
        "optional": {
            "initial_state": combo["initial_state"],
            "dt_kind": dt_kind,
            "ssm": ssm,
            "accepted": ssm and (i % 22 == 3),
            "non_contiguous_state": combo["initial_state"] and i % 13 == 5,
        },
    }


def randn(shape, generator, dtype, scale=0.5):
    return (torch.randn(tuple(shape), generator=generator, dtype=torch.float32) * scale).to(dtype).contiguous()


def make_inputs(case: dict[str, Any]):
    s = case["shape"]
    d = case["dtype"]
    attrs = case["attrs"]
    gen = torch.Generator().manual_seed(int(case["seed"]))
    h, hv, kdim, vdim = int(s["H"]), int(s["H_v"]), int(s["K"]), int(s["V"])
    if case["layout"] == "BSND":
        q_shape = (int(s["B"]), int(s["T"]), h, kdim)
        v_shape = (int(s["B"]), int(s["T"]), hv, vdim)
        g_shape = (int(s["B"]), int(s["T"]), hv, kdim)
        beta_shape = (int(s["B"]), int(s["T"]), hv)
    else:
        q_shape = (int(s["T_total"]), h, kdim)
        v_shape = (int(s["T_total"]), hv, vdim)
        g_shape = (int(s["T_total"]), hv, kdim)
        beta_shape = (int(s["T_total"]), hv)
    qkv_scale = 0.5 if attrs["use_qk_l2norm_in_kernel"] else 0.05
    q = randn(q_shape, gen, DTYPES[d["q_k_v"]], scale=qkv_scale)
    k = randn(q_shape, gen, DTYPES[d["q_k_v"]], scale=qkv_scale)
    v = randn(v_shape, gen, DTYPES[d["q_k_v"]], scale=qkv_scale)
    if attrs["use_gate_in_kernel"]:
        g = randn(g_shape, gen, DTYPES[d["g"]], scale=0.5)
    else:
        g = (-(0.1 + torch.rand(g_shape, generator=gen, dtype=torch.float32) * 0.4)).to(DTYPES[d["g"]]).contiguous()
    if attrs["use_beta_sigmoid_in_kernel"]:
        beta = randn(beta_shape, gen, DTYPES[d["beta"]], scale=0.5)
    else:
        beta = torch.rand(beta_shape, generator=gen, dtype=torch.float32).to(DTYPES[d["beta"]]).contiguous()
    initial_state = None
    if case["optional"]["initial_state"]:
        initial_state = randn((int(s["seq_num"]), hv, vdim, kdim), gen, DTYPES[d["state"]], scale=0.02)
    a_log = None
    dt_bias = None
    if attrs["use_gate_in_kernel"]:
        a_log = randn((hv,), gen, torch.float32, scale=0.1)
        if case["optional"]["dt_kind"] == "matrix":
            dt_bias = randn((hv, kdim), gen, torch.float32, scale=0.1)
        elif case["optional"]["dt_kind"] == "flat":
            dt_bias = randn((hv * kdim,), gen, torch.float32, scale=0.1)
    cu = s["cu_seqlens"]
    ssm_state_indices = None
    num_accepted_tokens = None
    if case["optional"]["ssm"]:
        lengths = [b - a for a, b in zip(cu, cu[1:])]
        slots = []
        for seq_idx, length in enumerate(lengths):
            slots.extend([seq_idx] * length)
        ssm_state_indices = torch.tensor(slots, dtype=torch.int32 if case["seed"] % 2 else torch.int64)
        if case["optional"]["accepted"]:
            num_accepted_tokens = torch.tensor(lengths, dtype=torch.int32 if case["seed"] % 3 else torch.int64)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "cu_seqlens": cu,
        "ssm_state_indices": ssm_state_indices,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "num_accepted_tokens": num_accepted_tokens,
    }


def to_device(inputs, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


def make_non_contiguous_last_dim(tensor):
    base_shape = tuple(tensor.shape[:-1]) + (int(tensor.shape[-1]) * 2,)
    base = torch.empty(base_shape, dtype=tensor.dtype, device=tensor.device)
    view = base[..., ::2]
    view.copy_(tensor)
    if view.is_contiguous():
        raise AssertionError("failed to create a non-contiguous state test tensor")
    return view


def call_op(inputs, attrs):
    return recurrent_kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
        **attrs,
    )


def timed_call(inputs, attrs, repeats: int):
    times = []
    out = final_state = None
    for _ in range(repeats):
        try:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            out, final_state = call_op(inputs, attrs)
            end.record()
            torch.npu.synchronize()
            times.append(float(start.elapsed_time(end)))
        except Exception:
            begin = time.perf_counter()
            out, final_state = call_op(inputs, attrs)
            torch.npu.synchronize()
            times.append((time.perf_counter() - begin) * 1000.0)
    return out, final_state, float(statistics.median(times))


def assert_close_with_stats(name: str, actual, expected, rtol: float, atol: float) -> dict[str, Any]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    threshold = atol + rtol * expected_f.abs()
    bad = diff > threshold
    stats = {
        "name": name,
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "bad_count": int(bad.sum().item()) if diff.numel() else 0,
        "numel": int(diff.numel()),
    }
    if stats["bad_count"]:
        idx = int(diff.argmax().item())
        stats["max_actual"] = float(actual_f.reshape(-1)[idx].item())
        stats["max_expected"] = float(expected_f.reshape(-1)[idx].item())
        raise AssertionError(json.dumps(stats, ensure_ascii=False))
    return stats


def branch_keys(case: dict[str, Any]) -> list[str]:
    attrs = case["attrs"]
    opt = case["optional"]
    s = case["shape"]
    keys = [
        f"layout={case['layout']}",
        f"varlen={s['cu_seqlens'] is not None}",
        f"total_bucket={bucket_power2(int(s.get('T_total', s.get('T', 1))))}",
        f"seq_num={s['seq_num']}",
        f"H={s['H']}",
        f"HV={s['H_v']}",
        f"H_ratio={s['H_v'] // s['H']}",
        f"kv={s['K']}x{s['V']}",
        f"gate={attrs['use_gate_in_kernel']}",
        f"safe={attrs['safe_gate']}",
        f"dt={opt['dt_kind']}",
        f"beta_sigmoid={attrs['use_beta_sigmoid_in_kernel']}",
        f"allow_neg={attrs['allow_neg_eigval']}",
        f"qk_norm={attrs['use_qk_l2norm_in_kernel']}",
        f"output_state={attrs['output_final_state']}",
        f"initial_state={opt['initial_state']}",
        f"ssm={opt['ssm']}",
        f"accepted={opt['accepted']}",
        f"non_contiguous_state={opt['non_contiguous_state']}",
        f"g_dtype={case['dtype']['g']}",
        f"beta_dtype={case['dtype']['beta']}",
        f"state_dtype={case['dtype']['state']}",
    ]
    if s["cu_seqlens"]:
        lengths = [b - a for a, b in zip(s["cu_seqlens"], s["cu_seqlens"][1:])]
        keys.append(f"max_segment={max(lengths) if lengths else 0}")
        keys.append(f"min_segment={min(lengths) if lengths else 0}")
    if s["H"] == 96 and s["H_v"] == 96 and s["K"] == 128 and s["V"] == 128:
        keys.append("kimi_h96_d128=True")
    if s["cu_seqlens"] and any((b - a) == 0 for a, b in zip(s["cu_seqlens"], s["cu_seqlens"][1:])):
        keys.append("zero_len=True")
    return keys


def shape_label(case: dict[str, Any]) -> str:
    s = case["shape"]
    if case["layout"] == "BSND":
        prefix = f"BSND[B={s['B']},T={s['T']}"
    else:
        prefix = f"TND[T={s['T_total']}"
    return f"{prefix},H={s['H']},HV={s['H_v']},K={s['K']},V={s['V']}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--state-rtol", type=float, default=None)
    parser.add_argument("--state-atol", type=float, default=None)
    parser.add_argument("--progress", type=int, default=50)
    args = parser.parse_args()

    state_rtol = args.rtol if args.state_rtol is None else args.state_rtol
    state_atol = args.atol if args.state_atol is None else args.state_atol
    expected_bool_combos = {bool_combo_label_from_values(combo) for combo in BOOL_COMBOS}
    expected_layout_bool_combos = {
        f"{layout}|{combo}" for layout in ("BSND", "TND") for combo in expected_bool_combos
    }
    torch.npu.set_device(torch.device(f"npu:{args.device}"))
    device = torch.device(f"npu:{args.device}")
    branch_counts: Counter[str] = Counter()
    bool_combo_counts: Counter[str] = Counter()
    layout_bool_combo_counts: Counter[str] = Counter()
    times: list[float] = []
    out_max_abs: list[float] = []
    state_max_abs: list[float] = []
    slowest: list[tuple[float, str, str, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    for offset in range(args.cases):
        i = args.start_index + offset
        case = generate_case(i, args.seed)
        combo_label = bool_combo_label(case)
        bool_combo_counts[combo_label] += 1
        layout_bool_combo_counts[f"{case['layout']}|{combo_label}"] += 1
        for key in branch_keys(case):
            branch_counts[key] += 1
        try:
            cpu_inputs = make_inputs(case)
            expected = recurrent_kda_reference(**cpu_inputs, **case["attrs"])
            dev_inputs = to_device(cpu_inputs, device)
            if case["optional"]["non_contiguous_state"]:
                dev_inputs["initial_state"] = make_non_contiguous_last_dim(dev_inputs["initial_state"])
            out, final_state, elapsed = timed_call(dev_inputs, case["attrs"], max(1, args.repeats))
            out_stats = assert_close_with_stats("out", out.cpu(), expected[0], rtol=args.rtol, atol=args.atol)
            out_max_abs.append(out_stats["max_abs"])
            if case["attrs"]["output_final_state"]:
                state_stats = assert_close_with_stats(
                    "final_state",
                    final_state.cpu(),
                    expected[1],
                    rtol=state_rtol,
                    atol=state_atol,
                )
                state_max_abs.append(state_stats["max_abs"])
            elif tuple(final_state.shape) != (0,):
                raise AssertionError(f"final_state shape is {tuple(final_state.shape)}, expected (0,)")
            times.append(elapsed)
            slowest.append((elapsed, case["id"], shape_label(case),
                            {**case["attrs"], "dt_kind": case["optional"]["dt_kind"]}))
            slowest = sorted(slowest, reverse=True)[:10]
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "id": case["id"],
                "case": case,
                "shape": shape_label(case),
                "attrs": case["attrs"],
                "optional": case["optional"],
                "error": str(exc)[:500],
            })
            print("FAIL", json.dumps(failures[-1], ensure_ascii=False), flush=True)
            break
        if args.progress and (offset + 1) % args.progress == 0:
            print(f"progress {offset + 1}/{args.cases}", flush=True)
    observed_bool_combos = set(bool_combo_counts)
    observed_layout_bool_combos = set(layout_bool_combo_counts)
    missing_bool_combos = sorted(expected_bool_combos - observed_bool_combos)
    missing_layout_bool_combos = sorted(expected_layout_bool_combos - observed_layout_bool_combos)
    summary = {
        "cases_requested": args.cases,
        "cases_passed": len(times),
        "failures": failures,
        "tolerance": {
            "out": {"rtol": args.rtol, "atol": args.atol},
            "final_state": {"rtol": state_rtol, "atol": state_atol},
        },
        "accuracy": {
            "out_max_abs": max(out_max_abs) if out_max_abs else None,
            "final_state_max_abs": max(state_max_abs) if state_max_abs else None,
        },
        "bool_combo_coverage": {
            "expected": len(expected_bool_combos),
            "observed": len(observed_bool_combos),
            "missing": missing_bool_combos,
            "min_count": min(bool_combo_counts.values()) if bool_combo_counts else 0,
            "max_count": max(bool_combo_counts.values()) if bool_combo_counts else 0,
        },
        "layout_bool_combo_coverage": {
            "expected": len(expected_layout_bool_combos),
            "observed": len(observed_layout_bool_combos),
            "missing_count": len(missing_layout_bool_combos),
            "missing_sample": missing_layout_bool_combos[:20],
        },
        "elapsed_ms": {
            "min": min(times) if times else None,
            "p50": percentile(times, 0.50),
            "p90": percentile(times, 0.90),
            "p95": percentile(times, 0.95),
            "p99": percentile(times, 0.99),
            "max": max(times) if times else None,
        },
        "branch_counts": dict(sorted(branch_counts.items())),
        "slowest": [
            {"elapsed_ms": item[0], "id": item[1], "shape": item[2], "attrs": item[3]}
            for item in slowest
        ],
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if args.cases >= len(BOOL_COMBOS) and missing_bool_combos:
        sys.exit(1)
    if args.cases >= 2 * len(BOOL_COMBOS) and missing_layout_bool_combos:
        sys.exit(1)
    if failures or len(times) != args.cases:
        sys.exit(1)


if __name__ == "__main__":
    main()
