"""Accuracy gate for pipeline edits on chunk_gdn_fwd_prepare.

Single case only (same shape as the simulator dump):

  B=1, HK=HV=8, T=1792, K=V=128, BT=64, BF16

From the test directory::

  python test_chunk_gdn_fwd_prepare_pipe_acc.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_chunk_gdn_fwd_prepare import (  # noqa: E402
    GDN_DIR,
    BT,
    check_against_ref,
    run_npu_prepare,
    setup_npu,
)

B, HK, HV, T, K, V = 1, 8, 8, 1792, 128, 128


def main():
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    setup_npu()
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(B, HK, T, K, dtype=dtype)
    k = torch.randn(B, HK, T, K, dtype=dtype)
    v = torch.randn(B, HV, T, V, dtype=dtype)
    g = torch.randn(B, HV, T, dtype=torch.float32)
    beta = torch.randn(B, HV, T, dtype=torch.float32)
    flags = dict(
        chunk_size=BT,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
    )
    print(f"golden={GDN_DIR}")
    print(f"pipe-acc: B={B} HK={HK} HV={HV} T={T} K={K} V={V} BT={BT} BF16")
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
