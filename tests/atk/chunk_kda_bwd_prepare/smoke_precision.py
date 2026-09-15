"""Focused A5 smoke/precision checks for chunk_kda_bwd_prepare."""

from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import npu_chunk_kda_bwd_prepare


def _dense_reference(aqk, v_new, d_o, h, scale, state_v_first):
    batch, heads, tokens, chunk_size = aqk.shape
    d_aqk = torch.zeros_like(aqk, dtype=torch.float32, device="cpu")
    dv = torch.empty_like(v_new, dtype=torch.float32, device="cpu")
    dq_raw = torch.empty_like(d_o, dtype=torch.float32, device="cpu")
    aqk_cpu = aqk.float().cpu()
    v_cpu = v_new.float().cpu()
    do_cpu = d_o.float().cpu()
    h_cpu = h.float().cpu()
    for b in range(batch):
        for n in range(heads):
            for chunk, begin in enumerate(range(0, tokens, chunk_size)):
                end = min(begin + chunk_size, tokens)
                rows = end - begin
                tri = torch.tril(torch.ones(rows, rows, dtype=torch.float32))
                d_aqk[b, n, begin:end, :rows] = (
                    do_cpu[b, n, begin:end] @ v_cpu[b, n, begin:end].T
                ) * tri * scale
                h_right = h_cpu[b, n, chunk]
                if not state_v_first:
                    h_right = h_right.T
                dq_raw[b, n, begin:end] = do_cpu[b, n, begin:end] @ h_right
                dv[b, n, begin:end] = (
                    aqk_cpu[b, n, begin:end, :rows].T @ do_cpu[b, n, begin:end]
                )
    return d_aqk, dv.to(torch.bfloat16), dq_raw


def _check(name, actual, expected, atol, rtol):
    actual_cpu = actual.float().cpu()
    expected_cpu = expected.float().cpu()
    diff = (actual_cpu - expected_cpu).abs()
    max_abs = float(diff.max())
    denom = expected_cpu.abs().clamp_min(1.0e-6)
    max_rel = float((diff / denom).max())
    print(f"{name}: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")
    if name == "d_aqk" and max_abs > atol:
        flat_index = int(diff.argmax())
        index = tuple(torch.unravel_index(torch.tensor(flat_index), diff.shape))
        print("max_diff_index=", index,
              "actual=", float(actual_cpu[index]), "expected=", float(expected_cpu[index]))
        print("actual[0,0,:4,:4]=\n", actual_cpu[0, 0, :4, :4])
        print("expected[0,0,:4,:4]=\n", expected_cpu[0, 0, :4, :4])
    return torch.allclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol)


def run_dense(device, tokens, heads, state_v_first, seed):
    torch.manual_seed(seed)
    shape_c = (1, heads, tokens, 64)
    shape_v = (1, heads, tokens, 128)
    chunks = (tokens + 63) // 64
    aqk = torch.randn(shape_c, dtype=torch.bfloat16, device=device)
    v_new = torch.randn(shape_v, dtype=torch.bfloat16, device=device)
    d_o = torch.randn(shape_v, dtype=torch.bfloat16, device=device)
    h = torch.randn((1, heads, chunks, 128, 128), dtype=torch.bfloat16, device=device)
    scale = 0.125
    expected = _dense_reference(aqk, v_new, d_o, h, scale, state_v_first)
    actual = npu_chunk_kda_bwd_prepare(
        aqk,
        v_new,
        d_o,
        h,
        scale=scale,
        chunk_size=64,
        state_v_first=state_v_first,
    )
    torch.npu.synchronize()
    results = (
        # Kernel A intentionally transfers the FP32 L0C result to AIV UB as
        # BF16 before applying the triangular scale mask.
        _check("d_aqk", actual[0], expected[0], atol=2.0e-2, rtol=4.0e-3),
        _check("dv", actual[1], expected[1], atol=1.6e-2, rtol=1.0e-2),
        _check("dq_raw", actual[2], expected[2], atol=2.0e-2, rtol=2.0e-3),
    )
    if not all(results):
        raise AssertionError(f"precision check failed: {results}")


def run_varlen(device, heads, state_v_first, seed):
    torch.manual_seed(seed)
    cu_seqlens = [0, 65, 98]
    chunk_indices = [0, 0, 0, 1, 1, 0]
    tokens = cu_seqlens[-1]
    aqk = torch.randn((heads, tokens, 64), dtype=torch.bfloat16, device=device)
    v_new = torch.randn((heads, tokens, 128), dtype=torch.bfloat16, device=device)
    d_o = torch.randn((heads, tokens, 128), dtype=torch.bfloat16, device=device)
    h = torch.randn((heads, 3, 128, 128), dtype=torch.bfloat16, device=device)
    scale = 0.125
    d_aqk_ref = torch.zeros((heads, tokens, 64), dtype=torch.float32)
    dv_ref = torch.empty((heads, tokens, 128), dtype=torch.float32)
    dq_ref = torch.empty((heads, tokens, 128), dtype=torch.float32)
    aqk_cpu = aqk.float().cpu()
    v_cpu = v_new.float().cpu()
    do_cpu = d_o.float().cpu()
    h_cpu = h.float().cpu()
    state_index = 0
    for seq, (seq_begin, seq_end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        del seq
        for begin in range(seq_begin, seq_end, 64):
            end = min(begin + 64, seq_end)
            rows = end - begin
            tri = torch.tril(torch.ones(rows, rows, dtype=torch.float32))
            for head in range(heads):
                d_aqk_ref[head, begin:end, :rows] = (
                    do_cpu[head, begin:end] @ v_cpu[head, begin:end].T
                ) * tri * scale
                h_right = h_cpu[head, state_index]
                if not state_v_first:
                    h_right = h_right.T
                dq_ref[head, begin:end] = do_cpu[head, begin:end] @ h_right
                dv_ref[head, begin:end] = (
                    aqk_cpu[head, begin:end, :rows].T @ do_cpu[head, begin:end]
                )
            state_index += 1
    actual = npu_chunk_kda_bwd_prepare(
        aqk,
        v_new,
        d_o,
        h,
        scale=scale,
        chunk_size=64,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    torch.npu.synchronize()
    results = (
        # Match the documented BF16 L0C-to-UB direct-transfer contract.
        _check("d_aqk", actual[0], d_aqk_ref, atol=2.0e-2, rtol=4.0e-3),
        _check("dv", actual[1], dv_ref.to(torch.bfloat16), atol=1.6e-2, rtol=1.0e-2),
        _check("dq_raw", actual[2], dq_ref, atol=2.0e-2, rtol=2.0e-3),
    )
    if not all(results):
        raise AssertionError(f"varlen precision check failed: {results}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:6")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if args.varlen:
        run_varlen(args.device, args.heads, args.state_v_first, args.seed)
    else:
        run_dense(args.device, args.tokens, args.heads, args.state_v_first, args.seed)


if __name__ == "__main__":
    main()
