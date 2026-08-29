# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").

import argparse
import json
from pathlib import Path

import torch
import torch_npu

from fla_npu.ops.ascendc import chunk_fwd_h


CHUNK_SIZE = 64
K_DIM = 128
V_DIM = 128
CASE_FILE = Path(__file__).resolve().parents[3] / "op_cases" / "chunk_fwd_h.json"
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _load_case(case_id):
    data = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in data["performance_cases"]}
    if case_id not in cases:
        raise ValueError(f"unknown performance case: {case_id}; available={sorted(cases)}")
    return cases[case_id]


def _canonical_chunk_indices(cu_seqlens):
    result = []
    for sequence, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        chunk_count = (end - begin + CHUNK_SIZE - 1) // CHUNK_SIZE
        for chunk in range(chunk_count):
            result.extend((sequence, chunk))
    return tuple(result)


def _make_varlen_metadata(case):
    raw_cu_seqlens = case.get("cu_seqlens")
    if raw_cu_seqlens is None:
        return None, None
    cu_seqlens = tuple(raw_cu_seqlens)
    if case["batch"] != 1:
        raise ValueError("variable-length performance cases require batch=1")
    if len(cu_seqlens) < 2 or cu_seqlens[0] != 0 or cu_seqlens[-1] != case["seqlen"]:
        raise ValueError("cu_seqlens must contain 0 and seqlen as its endpoints")
    if any(begin >= end for begin, end in zip(cu_seqlens, cu_seqlens[1:])):
        raise ValueError("cu_seqlens must be strictly increasing")
    if not case.get("provide_chunk_indices", False):
        return cu_seqlens, None
    return cu_seqlens, _canonical_chunk_indices(cu_seqlens)


def _make_inputs(case, device):
    torch.manual_seed(case["seed"])
    batch = case["batch"]
    seqlen = case["seqlen"]
    k_heads = case["k_heads"]
    v_heads = case["v_heads"]

    def random_bf16(shape):
        return (torch.randn(shape) * 0.05).to(torch.bfloat16).to(device)

    k = random_bf16((batch, k_heads, seqlen, K_DIM))
    w = random_bf16((batch, v_heads, seqlen, K_DIM))
    u = random_bf16((batch, v_heads, seqlen, V_DIM))
    gate_dtype = DTYPES[case["gate_dtype"]]
    g = None
    gk = None
    if case["mode"] == "g":
        gate_step = -torch.rand(batch, v_heads, seqlen) * 0.02
        g = torch.cumsum(gate_step, dim=2).to(gate_dtype).to(device)
    else:
        gate_step = -torch.rand(batch, v_heads, seqlen, K_DIM) * 0.02
        gk = torch.cumsum(gate_step, dim=2).to(gate_dtype).to(device)

    initial_state = None
    if case["state_dtype"] is not None:
        state = (torch.randn(batch, v_heads, K_DIM, V_DIM) * 0.05).to(
            DTYPES[case["state_dtype"]]
        )
        if case["state_v_first"]:
            state = state.transpose(-1, -2).contiguous()
        initial_state = state.to(device)
    cu_seqlens, chunk_indices = _make_varlen_metadata(case)
    return k, w, u, g, gk, initial_state, cu_seqlens, chunk_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="a5_g_h4_t512")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    case = _load_case(args.case_id)
    inputs = _make_inputs(case, device)

    def launch():
        return chunk_fwd_h(
            inputs[0],
            inputs[1],
            inputs[2],
            g=inputs[3],
            gk=inputs[4],
            initial_state=inputs[5],
            output_final_state=case["output_final_state"],
            chunk_size=CHUNK_SIZE,
            save_new_value=True,
            cu_seqlens=inputs[6],
            chunk_indices=inputs[7],
            use_exp2=case["use_exp2"],
            state_v_first=case["state_v_first"],
        )

    outputs = None
    for _ in range(args.warmup):
        outputs = launch()
    torch.npu.synchronize()
    varlen = ""
    if inputs[6] is not None:
        chunk_count = len(inputs[7]) // 2 if inputs[7] is not None else "auto"
        varlen = f" sequences={len(inputs[6]) - 1} chunks={chunk_count}"
    print(f"[PERF_READY] case={args.case_id} warmup={args.warmup}{varlen}", flush=True)
    for _ in range(args.iterations):
        outputs = launch()
    torch.npu.synchronize()
    if outputs is None:
        raise AssertionError("performance runner did not launch the operator")
    print(f"[PERF_DONE] case={args.case_id} launches={args.iterations}", flush=True)


if __name__ == "__main__":
    main()
