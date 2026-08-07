#!/usr/bin/env python3
"""Validate the A5 BF16 chunk64/K128/V128 non-aligned tail template."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from fla_npu.ops.ascendc import chunk_kda_fwd
from tests.reference.chunk_kda_reference import chunk_kda_forward_reference


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "tests/op_cases/chunk_kda_fwd.json"
CASE_IDS = (
    "chunk_kda_fwd_a5_bf16_sub16_varlen_h96",
    "chunk_kda_fwd_a5_bf16_sub16_dense_h29",
    "chunk_kda_fwd_a5_bf16_sub16_minimal_h1",
)
RCP_LN2 = 1.4426950408889634
OUTPUT_NAMES = (
    "attn_out", "final_state", "gk", "Aqk", "Akk", "w", "u", "qg",
    "kg", "v_new", "h", "initial_state",
)


def _load_cases(selected_ids: set[str] | None) -> tuple[dict, list[dict]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    required = set(CASE_IDS) if selected_ids is None else selected_ids
    cases = [case for case in manifest["cases"] if case["id"] in required]
    found = {case["id"] for case in cases}
    if found != required:
        raise RuntimeError(f"missing case ids: {sorted(required - found)}")
    return manifest, cases


def _spans(case: dict) -> list[tuple[int, int]]:
    cu = case["optional_inputs"].get("cu_seqlens")
    if cu is not None:
        return list(zip(cu, cu[1:]))
    return [(0, int(case["shape"]["T"]))]


def _gate_cumsum(raw_gate, a_log, dt_bias, safe_gate, lower_bound, spans, chunk_size):
    dim = raw_gate.shape[-1]
    x = raw_gate.float() + dt_bias.reshape(-1, dim)[:, None, :].float()
    a = torch.exp(a_log.float())[:, None, None]
    activated = (
        float(lower_bound) * torch.sigmoid(a * x)
        if safe_gate
        else -a * torch.nn.functional.softplus(x)
    )
    result = torch.empty_like(activated)
    for seq_start, seq_end in spans:
        for start in range(seq_start, seq_end, chunk_size):
            end = min(start + chunk_size, seq_end)
            result[:, start:end] = torch.cumsum(
                activated[:, start:end] * RCP_LN2, dim=1
            )
    return result


def _reference_outputs(reference, gk, layout, saved):
    if layout == "NTD":
        values = (
            reference.o.squeeze(0), reference.final_state, gk,
            reference.Aqk.squeeze(0).permute(1, 0, 2).contiguous(),
            reference.Akk.squeeze(0).permute(1, 0, 2).contiguous(),
            reference.w.squeeze(0).permute(1, 0, 2).contiguous(),
            reference.u.squeeze(0).permute(1, 0, 2).contiguous(),
            reference.qg.squeeze(0).permute(1, 0, 2).contiguous(),
            reference.kg.squeeze(0).permute(1, 0, 2).contiguous(),
            reference.v_new.squeeze(0).permute(1, 0, 2).contiguous(),
            reference.h.squeeze(0), None,
        )
    else:
        values = (
            reference.o, reference.final_state, gk.unsqueeze(0),
            reference.Aqk.permute(0, 2, 1, 3).contiguous(),
            reference.Akk.permute(0, 2, 1, 3).contiguous(),
            reference.w.permute(0, 2, 1, 3).contiguous(),
            reference.u.permute(0, 2, 1, 3).contiguous(),
            reference.qg.permute(0, 2, 1, 3).contiguous(),
            reference.kg.permute(0, 2, 1, 3).contiguous(),
            reference.v_new.permute(0, 2, 1, 3).contiguous(),
            reference.h, None,
        )
    if saved:
        return values
    return values[:2] + (None,) + values[3:5] + (None,) * 7


def _tail_recomposition_error(outputs, case, spans) -> float | None:
    attrs = case["attrs"]
    if not attrs["disable_recompute"]:
        return None
    chunk_size = int(case["shape"]["chunk_size"])
    layout = attrs["layout"]
    scale = float(attrs["scale"])
    errors = []
    chunk_base = 0
    for seq_start, seq_end in spans:
        chunk_count = math.ceil((seq_end - seq_start) / chunk_size)
        tail = (seq_end - seq_start) % chunk_size
        if tail:
            tail_start = seq_end - tail
            h_index = chunk_base + chunk_count - 1
            if layout == "NTD":
                qg = outputs[7][:, tail_start:seq_end].float() * scale
                h = outputs[10][h_index].float()
                state = torch.einsum("htk,hkv->htv", qg, h)
                local = torch.einsum(
                    "hts,hsv->htv",
                    outputs[3][:, tail_start:seq_end, :tail].float(),
                    outputs[9][:, tail_start:seq_end].float(),
                )
                actual = outputs[0][tail_start:seq_end].permute(1, 0, 2)
            else:
                qg = outputs[7][0, :, tail_start:seq_end].float() * scale
                h = outputs[10][0, h_index].float()
                state = torch.einsum("htk,hkv->htv", qg, h)
                local = torch.einsum(
                    "hts,hsv->htv",
                    outputs[3][0, :, tail_start:seq_end, :tail].float(),
                    outputs[9][0, :, tail_start:seq_end].float(),
                )
                actual = outputs[0][0, tail_start:seq_end].permute(1, 0, 2)
            expected = (state + local).to(torch.bfloat16)
            errors.append(float((actual.float() - expected.float()).abs().max().item()))
        chunk_base += chunk_count
    return max(errors, default=None)


def _run_case(case, device, tolerance):
    shape, attrs = case["shape"], case["attrs"]
    heads = int(shape["H_k"])
    total_t = int(shape["T"])
    dim = int(shape["K"])
    chunk_size = int(shape["chunk_size"])
    spans = _spans(case)
    torch.manual_seed(int(case["seed"]))
    q_head = (torch.randn(heads, total_t, dim) * 0.02).to(torch.bfloat16).to(device)
    k_head = (torch.randn(heads, total_t, dim) * 0.02).to(torch.bfloat16).to(device)
    v_head = (torch.randn(heads, total_t, dim) * 0.02).to(torch.bfloat16).to(device)
    raw_gate = (torch.randn(heads, total_t, dim) * 0.2).to(device)
    beta_head = torch.sigmoid(torch.randn(heads, total_t)).to(torch.bfloat16).to(device)
    a_log = (torch.randn(heads) * 0.2 - 0.5).to(device)
    dt_bias = (torch.randn(heads * dim) * 0.1).to(device)
    safe_gate = bool(attrs["safe_gate"])
    lower_bound = float(attrs["lower_bound"])
    gk = _gate_cumsum(
        raw_gate, a_log, dt_bias, safe_gate, lower_bound, spans, chunk_size
    )

    q_ref = q_head.permute(1, 0, 2).unsqueeze(0).contiguous()
    k_ref = k_head.permute(1, 0, 2).unsqueeze(0).contiguous()
    v_ref = v_head.permute(1, 0, 2).unsqueeze(0).contiguous()
    gk_ref = gk.permute(1, 0, 2).unsqueeze(0).contiguous()
    beta_ref = beta_head.permute(1, 0).unsqueeze(0).contiguous()
    cu_values = case["optional_inputs"].get("cu_seqlens")
    cu_tensor = (
        None
        if cu_values is None
        else torch.tensor(cu_values, dtype=torch.int64, device=device)
    )
    reference = chunk_kda_forward_reference(
        q_ref, k_ref, v_ref, gk_ref, beta_ref, float(attrs["scale"]),
        chunk_size, output_final_state=bool(attrs["output_final_state"]),
        cu_seqlens=cu_tensor,
    )

    layout = attrs["layout"]
    inputs = (
        (q_head, k_head, v_head, raw_gate, beta_head)
        if layout == "NTD"
        else (
            q_ref, k_ref, v_ref,
            raw_gate.permute(1, 0, 2).unsqueeze(0).contiguous(), beta_ref,
        )
    )
    kwargs = dict(
        layout=layout,
        cu_seqlens=None if cu_values is None else tuple(cu_values),
        output_final_state=bool(attrs["output_final_state"]),
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        disable_recompute=bool(attrs["disable_recompute"]),
        return_intermediate_states=bool(attrs["return_intermediate_states"]),
    )
    outputs = chunk_kda_fwd(
        *inputs, float(attrs["scale"]), chunk_size, **kwargs
    )
    torch.npu.synchronize()
    expected = _reference_outputs(
        reference, gk, layout, bool(attrs["disable_recompute"])
    )
    errors = {}
    for name, actual, target in zip(OUTPUT_NAMES, outputs, expected):
        if actual is None or target is None:
            if actual is not target:
                raise AssertionError(f"{case['id']}: optional output {name} differs")
            errors[name] = None
            continue
        if not torch.isfinite(actual.float()).all().item():
            raise AssertionError(f"{case['id']}: {name} contains NaN or Inf")
        errors[name] = float((actual.float() - target.float()).abs().max().item())
        torch.testing.assert_close(
            actual.float(), target.float(),
            rtol=float(tolerance["rtol"]), atol=float(tolerance["atol"]),
            msg=f"{case['id']}: {name}",
        )
    tail_error = _tail_recomposition_error(outputs, case, spans)
    if tail_error is not None and tail_error > 1e-6:
        raise AssertionError(
            f"{case['id']}: tail recomposition max_abs={tail_error} exceeds 1e-6"
        )
    return {
        "id": case["id"],
        "errors": errors,
        "tail_recomposition_max_abs": tail_error,
        "status": "PASS",
    }, inputs, kwargs


def _check_determinism(inputs, kwargs, scale, chunk_size, repeats):
    chunk_kda_fwd(*inputs, scale, chunk_size, **kwargs)
    torch.npu.synchronize()
    baseline = chunk_kda_fwd(*inputs, scale, chunk_size, **kwargs)
    torch.npu.synchronize()
    baseline = tuple(None if value is None else value.detach().clone() for value in baseline)
    torch.npu.synchronize()
    for repeat in range(1, repeats):
        current = chunk_kda_fwd(*inputs, scale, chunk_size, **kwargs)
        torch.npu.synchronize()
        for name, expected, actual in zip(OUTPUT_NAMES, baseline, current):
            if expected is None or actual is None:
                if expected is not actual:
                    raise AssertionError(f"repeat={repeat}: optional output {name} differs")
            elif not torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)):
                raise AssertionError(f"repeat={repeat}: output {name} is not bitwise equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case-id", action="append", choices=CASE_IDS)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    selected = None if args.case_id is None else set(args.case_id)
    manifest, cases = _load_cases(selected)
    tolerance = manifest["tolerance"]["bfloat16"]
    results = []
    for case in cases:
        result, inputs, kwargs = _run_case(case, device, tolerance)
        expected_repeats = int(case["expect"].get("binary_deterministic_runs", 0))
        if expected_repeats:
            repeats = args.repeats if args.repeats else expected_repeats
            _check_determinism(
                inputs, kwargs, float(case["attrs"]["scale"]),
                int(case["shape"]["chunk_size"]), repeats,
            )
            result["binary_deterministic_runs"] = repeats
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {"passed": len(results), "failed": 0, "status": "PASS", "results": results}
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
