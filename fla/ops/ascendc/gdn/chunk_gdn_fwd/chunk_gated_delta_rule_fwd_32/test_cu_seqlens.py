#!/usr/bin/env python3
"""Packed sequences. One head walks every seqlen in T. final_state is [N, Hv, 128, 128]."""

import math
import os
import sys

os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
    "/home/w00933206/tmp/gdn32_py/fla_npu/opp/vendors/fla_npu_transformer:"
    "/home/w00933206/tmp/gdn32_py/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib:"
    + os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
)
os.environ["FLA_NPU_DISABLE_PTH"] = "1"

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/w00933206/tmp/gdn32_py")

from cpu_golden_fused_infer import run_infer_golden  # noqa: E402
from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd_32  # noqa: E402


def _ok(name, got, golden):
    ref = golden.detach().cpu().float().to(torch.bfloat16).float()
    out = got.detach().cpu().float()
    ratio = (out - ref).abs() / (ref.abs() * (2.0 ** -7) + (2.0 ** -7))
    passed = ratio.max().item() < 5 and ratio.mean().item() < 1.5 and ratio.square().mean().sqrt().item() < 1.5
    print(
        f"{name}: max_ratio={ratio.max().item():.4f} mean_ratio={ratio.mean().item():.4f} "
        f"rms_ratio={ratio.square().mean().sqrt().item():.4f} {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return passed


def _inputs(hk, hv, tokens, seed):
    torch.manual_seed(seed)
    q = torch.randn(1, hk, tokens, 128, dtype=torch.bfloat16)
    k = torch.randn(1, hk, tokens, 128, dtype=torch.bfloat16)
    v = torch.randn(1, hv, tokens, 128, dtype=torch.bfloat16)
    g = (torch.randn(1, hv, tokens, dtype=torch.float32) * 0.02).contiguous()
    beta = torch.randn(1, hv, tokens, dtype=torch.bfloat16)
    return q, k, v, g, beta


def run_packed(hk, hv, tokens, bounds, with_state, explicit_indices, seed):
    q, k, v, g, beta = _inputs(hk, hv, tokens, seed)
    cu = torch.tensor(bounds, dtype=torch.int64)
    nseq = len(bounds) - 1
    state0 = torch.randn(nseq, hv, 128, 128, dtype=torch.bfloat16) * 0.01 if with_state else None
    scale = 1.0 / math.sqrt(128)
    chunks = None
    if explicit_indices:
        chunks = []
        for seq in range(nseq):
            length = bounds[seq + 1] - bounds[seq]
            nchunk = (length + 31) // 32
            for local in range(nchunk):
                chunks.extend((seq, local))
    golden_o, golden_s = run_infer_golden(
        q, k, v, g, beta, scale=scale, initial_state=state0, cu_seqlens=cu, g_layout="BHT",
        use_qk_l2norm=True, use_beta_sigmoid=True,
    )
    out, final_state = npu_chunk_gated_delta_rule_fwd_32(
        q.npu(), k.npu(), v.npu(), g.npu(), beta.npu(),
        initial_state=None if state0 is None else state0.npu(),
        cu_seqlens=cu, chunk_indices=chunks, scale=scale,
    )
    torch.npu.synchronize()
    tag = f"Hk={hk} Hv={hv} T={tokens} cu={bounds} state={with_state} idx={explicit_indices}"
    return _ok(tag + " o", out, golden_o) and _ok(tag + " state", final_state, golden_s)


def run_empty(with_state):
    """Middle sequence has no tokens. Compare the two live sequences with separate goldens."""
    hk, hv, tokens = 2, 2, 64
    bounds = [0, 32, 32, 64]
    q, k, v, g, beta = _inputs(hk, hv, tokens, 11 if with_state else 12)
    state0 = torch.randn(3, hv, 128, 128, dtype=torch.bfloat16) * 0.01 if with_state else None
    scale = 1.0 / math.sqrt(128)
    cu = torch.tensor(bounds, dtype=torch.int64)
    out, final_state = npu_chunk_gated_delta_rule_fwd_32(
        q.npu(), k.npu(), v.npu(), g.npu(), beta.npu(),
        initial_state=None if state0 is None else state0.npu(),
        cu_seqlens=cu, scale=scale,
    )
    torch.npu.synchronize()
    passed = True
    for seq, begin, end in ((0, 0, 32), (2, 32, 64)):
        init = None if state0 is None else state0[seq:seq + 1]
        golden_o, golden_s = run_infer_golden(
            q[:, :, begin:end], k[:, :, begin:end], v[:, :, begin:end],
            g[:, :, begin:end], beta[:, :, begin:end],
            scale=scale, initial_state=init, g_layout="BHT",
            use_qk_l2norm=True, use_beta_sigmoid=True,
        )
        tag = f"empty-middle state={with_state} seq={seq}"
        passed = _ok(tag + " o", out[:, :, begin:end], golden_o) and passed
        passed = _ok(tag + " state", final_state[seq:seq + 1], golden_s) and passed
    slot = final_state[1].detach().cpu().float()
    if state0 is None:
        ref = torch.zeros_like(slot)
    else:
        ref = state0[1].float()
    match = torch.equal(slot, ref)
    print(f"empty-middle state={with_state} slot {'PASS' if match else 'FAIL'}", flush=True)
    return passed and match


def main():
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(0)
    cases = [
        (2, 2, 96, [0, 64, 96], False, False, 1),
        (2, 2, 100, [0, 40, 100], False, True, 2),
        (2, 2, 80, [0, 32, 48, 80], True, False, 3),
        (2, 4, 70, [0, 33, 70], False, True, 4),
    ]
    failed = 0
    for case in cases:
        if not run_packed(*case):
            failed += 1
    if not run_empty(False):
        failed += 1
    if not run_empty(True):
        failed += 1
    print(f"cu_seqlens failed={failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
