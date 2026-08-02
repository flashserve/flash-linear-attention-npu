#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.
"""NPU single-op test for npu_chunk_kda_fwd_intra_sub_chunk (requires installed wheel + NPU)."""

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Sequence

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_GOLDEN_DIR = os.path.join(
    _REPO_ROOT, "fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/test"
)
sys.path.insert(0, _GOLDEN_DIR)
from test_chunk_kda_fwd_intra_sub_chunk import (  # noqa: E402
    chunk_kda_fwd_intra_sub_chunk_ref,
    prepare_chunk_indices,
)


def _make_gate(B: int, HV: int, T: int, K: int, dtype: torch.dtype, mode: str) -> torch.Tensor:
    if mode == "lin_strong":
        # Strong decaying gate (stresses (I-L)^{-1} magnitude).
        g = -torch.linspace(0, 30, T).view(1, 1, T, 1).expand(B, HV, T, K)
    elif mode == "lin_mild":
        g = -torch.linspace(0, 8, T).view(1, 1, T, 1).expand(B, HV, T, K)
    elif mode == "rand":
        g = torch.randn(B, HV, T, K)
    else:
        raise ValueError(mode)
    return g.to(dtype).contiguous()


def _run_case(
    B: int,
    H: int,
    T: int,
    K: int,
    BT: int,
    *,
    HV: Optional[int] = None,
    varlen: bool = False,
    cu: Optional[Sequence[int]] = None,
    dtype: torch.dtype = torch.bfloat16,
    gate: str = "lin_strong",
    seed: int = 0,
    aqk_tol: float = 5e-2,
    akkd_rel_tol: float = 1e-3,
    ref_dtype: torch.dtype = torch.float64,
):
    import time

    import fla_npu.ops.ascendc as ascendc_ops  # noqa: F401 — OPP path must be ready

    if HV is None:
        HV = H
    assert HV >= H and HV % H == 0

    t0 = time.time()
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, K, dtype=dtype)
    k = torch.randn(B, H, T, K, dtype=dtype)
    g = _make_gate(B, HV, T, K, dtype, gate)
    beta = torch.rand(B, HV, T, dtype=dtype)
    scale = 1.0 / math.sqrt(K)

    cu_t = None
    idx = None
    if varlen or cu is not None:
        assert B == 1
        if cu is None:
            mid = T // 2
            cu = [0, mid, T]
        assert cu[0] == 0 and cu[-1] == T
        cu_t = torch.tensor(list(cu), dtype=torch.long)
        idx = torch.tensor(prepare_chunk_indices(cu_t, BT), dtype=torch.long)

    t_ref0 = time.time()
    # Cube path: Vector fp32 → Cast T → MMAD; golden casts score factors to input dtype.
    aqk_ref, akkd_ref = chunk_kda_fwd_intra_sub_chunk_ref(
        q.float(),
        k.float(),
        g.float(),
        beta.float(),
        scale,
        BT,
        cu_t,
        idx,
        dtype=ref_dtype,
        score_dtype=dtype,
    )
    t_ref = time.time() - t_ref0

    t_npu0 = time.time()
    aqk_n, akkd_n = ascendc_ops.npu_chunk_kda_fwd_intra_sub_chunk(
        q.npu(),
        k.npu(),
        g.npu(),
        beta.npu(),
        scale,
        BT,
        cu_seqlens=None if cu_t is None else cu_t.tolist(),
        chunk_indices=None if idx is None else idx.tolist(),
    )
    torch.npu.synchronize()
    t_npu = time.time() - t_npu0

    aqk_err = (aqk_n.float().cpu() - aqk_ref.float()).abs().max().item()
    akkd_n_c = akkd_n.float().cpu()
    akkd_ref_c = akkd_ref.float()
    akkd_abs = (akkd_n_c - akkd_ref_c).abs()
    akkd_err = akkd_abs.max().item()
    # (I-L)^{-1} can be huge under strong gates; use relative tolerance.
    akkd_rel = (akkd_abs / akkd_ref_c.abs().clamp_min(1.0)).max().item()
    tag = f"cu={list(cu)}" if cu is not None else "dense"
    print(
        f"[case] B={B} H={H} HV={HV} T={T} K={K} BT={BT} {tag} gate={gate} dtype={dtype} ref={ref_dtype} "
        f"aqk_max_err={aqk_err:.6g} akkd_max_err={akkd_err:.6g} akkd_max_rel={akkd_rel:.6g} "
        f"t_ref={t_ref:.2f}s t_npu={t_npu:.2f}s total={time.time()-t0:.2f}s"
    )
    assert aqk_n.shape == (B, HV, T, BT) and akkd_n.shape == (B, HV, T, 16)
    assert torch.isfinite(aqk_n).all() and torch.isfinite(akkd_n).all()
    assert aqk_err < aqk_tol, aqk_err
    assert akkd_rel < akkd_rel_tol, (akkd_err, akkd_rel)


def _bf16_sim_sub(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    b: int,
    h: int,
    t0: int,
    bc: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cube-faithful local golden for one diagonal sub-chunk (score factors in bf16)."""
    mid = bc // 2
    gg = g[b, h, t0 : t0 + bc].float()
    qq = q[b, h, t0 : t0 + bc].float()
    kk = k[b, h, t0 : t0 + bc].float()
    bb = beta[b, h, t0 : t0 + bc].float()
    gm = gg - gg[mid : mid + 1]
    qg = (qq * torch.exp2(gm)).bfloat16().float()
    kpos = (kk * torch.exp2(gm)).bfloat16().float()
    kneg = (kk * torch.exp2(-gm)).bfloat16().float()
    aqk = torch.tril(qg @ kneg.T) * scale
    L = torch.tril(kpos @ kneg.T, -1) * bb.unsqueeze(1)
    akk = -L
    for i in range(2, bc):
        tmp = akk[i].clone()
        tmp[i:] = 0
        for j in range(i):
            acc = tmp[j]
            for p in range(i):
                acc = acc + tmp[p] * akk[p, j]
            tmp[j] = acc
        akk[i] = tmp
    akk = akk + torch.eye(bc, dtype=akk.dtype, device=akk.device)
    return aqk, akk


def _run_case_model_sample(
    B: int,
    H: int,
    T: int,
    K: int,
    BT: int,
    *,
    gate: str = "lin_mild",
    seed: int = 0,
    aqk_tol: float = 5e-2,
    akkd_rel_tol: float = 1e-3,
    t_stride: int = 256,
    h_stride: int = 4,
):
    """Large-T check without full CPU golden (too slow). Sample vs Cube bf16-sim."""
    import time

    import fla_npu.ops.ascendc as ascendc_ops  # noqa: F401

    t0 = time.time()
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    q = torch.randn(B, H, T, K, dtype=dtype)
    k = torch.randn(B, H, T, K, dtype=dtype)
    g = _make_gate(B, H, T, K, dtype, gate)
    beta = torch.rand(B, H, T, dtype=dtype)
    scale = 1.0 / math.sqrt(K)
    BC = 16

    t_npu0 = time.time()
    aqk_n, akkd_n = ascendc_ops.npu_chunk_kda_fwd_intra_sub_chunk(
        q.npu(), k.npu(), g.npu(), beta.npu(), scale, BT
    )
    torch.npu.synchronize()
    t_npu = time.time() - t_npu0

    aqk_c = aqk_n.float().cpu()
    akkd_c = akkd_n.float().cpu()
    assert torch.isfinite(aqk_c).all() and torch.isfinite(akkd_c).all()

    aqk_err = 0.0
    akkd_rel = 0.0
    n_samp = 0
    for h in range(0, H, h_stride):
        for ti in range(0, T - BC + 1, t_stride):
            aqk_s, akk_s = _bf16_sim_sub(q, k, g, beta, scale, 0, h, ti, BC)
            col0 = (ti % BT)
            ae = (aqk_c[0, h, ti : ti + BC, col0 : col0 + BC] - aqk_s).abs().max().item()
            re = ((akkd_c[0, h, ti : ti + BC] - akk_s).abs() / akk_s.abs().clamp_min(1.0)).max().item()
            aqk_err = max(aqk_err, ae)
            akkd_rel = max(akkd_rel, re)
            n_samp += 1

    print(
        f"[model-sample] B={B} H={H} T={T} K={K} BT={BT} gate={gate} n_samp={n_samp} "
        f"aqk_max_err={aqk_err:.6g} akkd_max_rel={akkd_rel:.6g} "
        f"t_npu={t_npu:.2f}s total={time.time()-t0:.2f}s"
    )
    assert aqk_err < aqk_tol, aqk_err
    assert akkd_rel < akkd_rel_tol, akkd_rel


def main():
    # Import fla_npu before torch_npu so ASCEND_CUSTOM_OPP_PATH is set
    # before GE loads tiling SOs (otherwise GetWorkspaceSize -> 561103).
    import fla_npu.ops.ascendc  # noqa: F401
    import torch_npu  # noqa: F401

    device = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    torch.npu.set_device(device)

    # FLA_NPU_ONLY_MODEL=1 → skip smoke/small cases, only model-scale.
    # FLA_NPU_ONLY_GVA=1 → only GVA smoke cases (after rebuild).
    only_model = os.environ.get("FLA_NPU_ONLY_MODEL", "0") == "1"
    only_gva = os.environ.get("FLA_NPU_ONLY_GVA", "0") == "1"

    if only_gva:
        _run_case(1, 2, 64, 128, 64, HV=4)  # GVA 2x
        _run_case(1, 4, 64, 128, 64, HV=8)  # GVA 2x
        _run_case(1, 2, 96, 128, 64, HV=4, varlen=True)
        # Larger GVA: Akkd fp32 accumulate vs fp64 golden (same class as K=256 / long-T).
        _run_case(1, 8, 128, 128, 64, HV=16, gate="lin_mild", akkd_rel_tol=5e-2)
        _run_case(1, 4, 80, 128, 32, HV=8)  # BT=32 + tail + GVA
        print("all GVA cases passed")
        return

    if not only_model:
        # ---- original smoke ----
        _run_case(1, 2, 64, 128, 64)
        _run_case(1, 2, 64, 128, 32)
        _run_case(1, 2, 128, 128, 128)
        _run_case(1, 2, 96, 128, 64, varlen=True)

        # ---- GVA (HV > H) ----
        _run_case(1, 2, 64, 128, 64, HV=4)
        _run_case(1, 4, 64, 128, 64, HV=8)
        _run_case(1, 2, 96, 128, 64, HV=4, varlen=True)
        _run_case(1, 8, 128, 128, 64, HV=16, gate="lin_mild", akkd_rel_tol=5e-2)
        _run_case(1, 4, 80, 128, 32, HV=8)
        # ---- dense: batch / heads / K / tail / BT ----
        _run_case(2, 2, 64, 128, 64)  # B>1
        _run_case(1, 4, 64, 128, 64)  # more heads
        # DESIGN locks K=128 (Catlass L1 tile K=128); K=64/256 not in v1 scope.
        _run_case(1, 2, 48, 128, 64)  # T not multiple of BT (tail)
        _run_case(1, 2, 80, 128, 64)  # T > BT with tail
        _run_case(1, 2, 32, 128, 32)  # BT=32 full
        _run_case(1, 2, 40, 128, 32)  # BT=32 tail
        # Longer / multi-chunk: more forward-sub steps → slightly looser akkd rel vs fp64.
        _run_case(1, 2, 256, 128, 128, gate="lin_mild", akkd_rel_tol=5e-3)
        _run_case(1, 2, 200, 128, 128, gate="lin_mild", akkd_rel_tol=5e-3)

        # ---- gates / dtype ----
        _run_case(1, 2, 64, 128, 64, gate="lin_mild")
        _run_case(1, 2, 64, 128, 64, gate="rand")
        _run_case(1, 2, 64, 128, 64, dtype=torch.float16, akkd_rel_tol=5e-3)

        # ---- varlen: unequal / multi-seq / short / BT variants ----
        _run_case(1, 2, 96, 128, 64, cu=[0, 40, 96])  # unequal lens
        _run_case(1, 2, 120, 128, 64, cu=[0, 16, 80, 120])  # 3 sequences
        _run_case(1, 2, 20, 128, 64, cu=[0, 8, 20])  # short seqs (< BC and < BT)
        _run_case(1, 2, 96, 128, 32, cu=[0, 48, 96])  # varlen BT=32
        _run_case(1, 2, 160, 128, 128, cu=[0, 70, 160], gate="lin_mild", akkd_rel_tol=5e-3)
        _run_case(1, 2, 256, 128, 64, cu=[0, 100, 180, 256], gate="lin_mild", akkd_rel_tol=5e-3)

        _run_case(1, 8, 512, 128, 64, gate="lin_mild", akkd_rel_tol=5e-3)

    # ---- model target shapes (README: B=1,T=8192,H=32,K=128,BT=64) ----
    # Full CPU golden at T=8192 is too slow (~500s+/case). Sample vs Cube bf16-sim instead.
    _run_case_model_sample(1, 16, 2048, 128, 64, gate="lin_mild", akkd_rel_tol=5e-2)
    _run_case_model_sample(1, 32, 4096, 128, 64, gate="lin_mild", akkd_rel_tol=5e-2)
    _run_case_model_sample(1, 32, 8192, 128, 64, gate="lin_mild", akkd_rel_tol=5e-2)
    _run_case_model_sample(1, 32, 8192, 128, 64, gate="lin_strong", akkd_rel_tol=5e-2)
    _run_case_model_sample(1, 32, 8192, 128, 64, gate="rand", aqk_tol=1e-1, akkd_rel_tol=5e-2)

    print("all cases passed")


if __name__ == "__main__":
    main()
