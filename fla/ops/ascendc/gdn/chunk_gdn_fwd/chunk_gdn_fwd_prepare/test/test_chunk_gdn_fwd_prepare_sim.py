"""Single-case launcher for ``msprof op simulator`` (pipeline dump).

Exactly one ``chunk_gdn_fwd_prepare`` launch. Shape is fixed:

  B=1, HK=HV=8, T=1792, K=V=128, BT=64, BF16

From repo root::

  msprof op simulator \\
      --application="python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gdn_fwd_prepare/test/test_chunk_gdn_fwd_prepare_sim.py" \\
      --soc-version=Ascend950PR_9579 --timeout=20 \\
      --kernel-name=ChunkGdnFwdPrepare* --launch-count=1
"""

from __future__ import annotations

import os

import torch
from fla_npu.ops import ascendc as ascendc_ops

B, HK, HV, T, K, V = 1, 8, 8, 1792, 128, 128
BT = 64


def setup_npu():
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def main():
    setup_npu()
    device = torch.device("npu")
    dtype = torch.bfloat16
    q = torch.empty(B, HK, T, K, dtype=dtype, device=device)
    k = torch.empty(B, HK, T, K, dtype=dtype, device=device)
    v = torch.empty(B, HV, T, V, dtype=dtype, device=device)
    g = torch.empty(B, HV, T, dtype=torch.float32, device=device)
    beta = torch.empty(B, HV, T, dtype=torch.float32, device=device)
    print(
        f"SIM: B={B} HK={HK} HV={HV} T={T} K={K} V={V} BT={BT} BF16 "
        "l2norm=True gate=False beta_sigmoid=True neg=False exp2=True",
        flush=True,
    )
    with torch.no_grad():
        outs = ascendc_ops.chunk_gdn_fwd_prepare(
            q, k, v, g, beta,
            chunk_size=BT,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=False,
            use_exp2=True,
        )
        torch.npu.synchronize()
    n_out = sum(1 for x in outs if x is not None)
    print(f"SIM: done n_out={n_out} u_shape={tuple(outs[7].shape)}", flush=True)


if __name__ == "__main__":
    main()
