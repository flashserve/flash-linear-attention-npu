"""Accuracy gate for use_qk_l2norm_in_kernel=False.

Same shape as pipe_acc (G=2 long-seq). q and k are L2-normalized in
``generate_inputs`` before the op; the kernel skips in-kernel L2Norm.
WY on unnormalized k is inf/nan.

  B=1, HK=16, HV=32, T=11264, K=V=128, BT=64, BF16
  l2norm=False, gate=False, sigmoid=True, neg=True

From the test directory::

  python test_chunk_gated_delta_rule_fwd_prepare_l2norm_false.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import GdnCase, case_inputs  # noqa: E402
from test_chunk_gated_delta_rule_fwd_prepare import (  # noqa: E402
    GDN_DIR,
    BT,
    check_against_ref,
    run_npu_prepare,
    setup_npu,
)

B, HK, HV, T, K, V = 1, 16, 32, 11264, 128, 128


def main():
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    setup_npu()
    case = GdnCase(
        case_id=0,
        batch=B,
        hk=HK,
        hv=HV,
        seq_len=T,
        head_k=K,
        head_v=V,
        chunk_size=BT,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
    )
    inp = case_inputs(case, dtype=torch.float32, device=torch.device("cpu"), seed=0, layout="bnsd")
    q = inp["q"].to(torch.bfloat16)
    k = inp["k"].to(torch.bfloat16)
    v = inp["v"].to(torch.bfloat16)
    g = inp["g"].float()
    beta = inp["beta"].float()
    flags = dict(
        chunk_size=BT,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        output_a=os.environ.get("OUTPUT_A", "1") != "0",
    )
    print(f"golden={GDN_DIR}")
    print(f"l2norm-false: B={B} HK={HK} HV={HV} T={T} K={K} V={V} BT={BT} BF16")
    try:
        outs = run_npu_prepare(q, k, v, g, beta, flags=flags)
        check_against_ref(q, k, v, g, beta, outs, flags)
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
