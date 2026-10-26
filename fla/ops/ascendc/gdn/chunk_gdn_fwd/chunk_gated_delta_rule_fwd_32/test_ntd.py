#!/usr/bin/env python3
"""NTD / head-major sequences. Shape is [B, H, T, D], the same storage as BNSD.

Each (batch, head) is one task. B*H is larger than the cube count, so a core's
second task issues PrepareLoad before the first task stores final_state.
"""

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


def run_case(batch, heads, tokens, with_state):
    torch.manual_seed(20 + batch + tokens)
    q = torch.randn(batch, heads, tokens, 128, dtype=torch.bfloat16)
    k = torch.randn(batch, heads, tokens, 128, dtype=torch.bfloat16)
    v = torch.randn(batch, heads, tokens, 128, dtype=torch.bfloat16)
    g = (torch.randn(batch, heads, tokens, dtype=torch.float32) * 0.02).contiguous()
    beta = torch.randn(batch, heads, tokens, dtype=torch.bfloat16)
    state0 = torch.randn(batch, heads, 128, 128, dtype=torch.bfloat16) * 0.01 if with_state else None
    scale = 1.0 / math.sqrt(128)
    golden_o, golden_s = run_infer_golden(
        q, k, v, g, beta, scale=scale, initial_state=state0, g_layout="BHT",
        use_qk_l2norm=True, use_beta_sigmoid=True,
    )
    out, final_state = npu_chunk_gated_delta_rule_fwd_32(
        q.npu(), k.npu(), v.npu(), g.npu(), beta.npu(),
        initial_state=None if state0 is None else state0.npu(), scale=scale,
    )
    torch.npu.synchronize()
    tag = f"NTD B={batch} H={heads} T={tokens} state={with_state}"
    return _ok(tag + " o", out, golden_o) and _ok(tag + " state", final_state, golden_s)


def main():
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(0)
    # 3*12 = 36 tasks. T=64 is 64-aligned; T=80 is not 32-aligned.
    cases = [(3, 12, 64, False), (3, 12, 80, True)]
    failed = 0
    for case in cases:
        if not run_case(*case):
            failed += 1
    print(f"ntd failed={failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
