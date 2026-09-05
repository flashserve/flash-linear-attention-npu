"""Single-case perf launcher for ``msprof op``.

This process launches ``chunk_gated_delta_rule_fwd_prepare`` exactly once. Pick one case:

  python test_chunk_gated_delta_rule_fwd_prepare_perf.py --case-id 0   # B=1 HK=HV=96 T=8192
  python test_chunk_gated_delta_rule_fwd_prepare_perf.py --case-id 1   # B=1 HK=HV=96 T=16384
  python test_chunk_gated_delta_rule_fwd_prepare_perf.py --case-id 2   # B=1 HK=16 HV=32 T=11264

msprof (collects one MIX kernel; the binary name is ChunkGatedDeltaRuleFwdPrepare_*_mix_aic)::

  msprof op --application="python test_chunk_gated_delta_rule_fwd_prepare_perf.py --case-id 0" \\
      --kernel-name=ChunkGatedDeltaRuleFwdPrepare* --launch-count=1 --kill=on \\
      --aic-metrics=PipeUtilization --output=./msprof_t8192
"""

from __future__ import annotations

import argparse
import os

import torch
from fla_npu.ops import ascendc as ascendc_ops

CASES = (
    dict(case_id=0, batch=1, hk=96, hv=96, seq_len=8192, head_k=128, head_v=128),
    dict(case_id=1, batch=1, hk=96, hv=96, seq_len=16384, head_k=128, head_v=128),
    dict(case_id=2, batch=1, hk=16, hv=32, seq_len=11264, head_k=128, head_v=128),
)


def setup_npu():
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def run_one(case: dict):
    torch.manual_seed(0)
    torch.npu.manual_seed_all(0)
    device = torch.device("npu")
    dtype = torch.bfloat16
    b, hk, hv, t, k, v = (
        case["batch"], case["hk"], case["hv"],
        case["seq_len"], case["head_k"], case["head_v"],
    )
    q = torch.randn(b, hk, t, k, dtype=dtype, device=device)
    k_t = torch.randn(b, hk, t, k, dtype=dtype, device=device)
    v_t = torch.randn(b, hv, t, v, dtype=dtype, device=device)
    g = torch.randn(b, hv, t, dtype=torch.float32, device=device)
    beta = torch.randn(b, hv, t, dtype=torch.float32, device=device)
    torch.npu.synchronize()
    print(
        f"PERF case{case['case_id']}: B={b} HK={hk} HV={hv} T={t} K={k} V={v} "
        f"chunk=64 l2norm=True gate=False beta_sigmoid=True neg=True exp2=True",
        flush=True,
    )
    outs = ascendc_ops.chunk_gated_delta_rule_fwd_prepare(
        q, k_t, v_t, g, beta,
        chunk_size=64,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        use_exp2=True,
    )
    torch.npu.synchronize()
    n_out = sum(1 for x in outs if x is not None)
    print(f"PERF case{case['case_id']}: done n_out={n_out} u_shape={tuple(outs[7].shape)}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Launch exactly one chunk_gated_delta_rule_fwd_prepare for msprof op.")
    parser.add_argument(
        "--case-id", type=int, required=True, choices=(0, 1, 2),
        help="0: T=8192 G=1; 1: T=16384 G=1; 2: HK=16 HV=32 T=11264 G=2.")
    args = parser.parse_args()
    setup_npu()
    with torch.no_grad():
        run_one(CASES[args.case_id])


if __name__ == "__main__":
    main()
