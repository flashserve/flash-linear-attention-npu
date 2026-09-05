"""Numerics test for fused chunk_gated_delta_rule_fwd_prepare vs local CPU golden.

Golden is ``cpu_gdn_fwd_l2norm_to_recompute`` in this test directory.

Default runs ``gdn_cases()`` 1-6 first, then case 0 (T=1792 G=2), case 7
(V=256), case 8 (G=3), case 9 (G=4), case 10 (small packed varlen). The six
GdnCase rows::

    1  B=2  HK=16 HV=32 T=11264  G=2  long fixed
    2  B=1  HK=32 HV=32 T=65536  G=1  packed varlen
    3  B=4  HK=96 HV=96 T=128    G=1  wide heads
    4  B=1  HK=32 HV=32 T=160    G=1  T not multiple of 64
    5  B=6  HK=6  HV=6  T=1084   G=1  tail + few heads
    6  B=1  HK=12 HV=12 T=1084   G=1  mid T

  q_hat / k_hat                         elementwise atol=rtol=2e-2
  q_rstd / k_rstd / beta_out / g_cumsum elementwise atol=rtol=5e-3
  A                                     elementwise atol=rtol=8e-2
  w / u                                 per-chunk tile-max rel <= 8e-2
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch
from fla_npu.ops import ascendc as ascendc_ops

_TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TEST_DIR))
from cases import gdn_cases, describe_case, case_inputs  # noqa: E402
from cpu_golden import cpu_gdn_fwd_l2norm_to_recompute  # noqa: E402

GDN_DIR = _TEST_DIR / "cpu_golden.py"
BT = 64


def setup_npu():
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def assert_close(name, got, ref, atol, rtol):
    g = got.detach().float()
    r = ref.detach().float()
    d = (g - r).abs()
    print(f"  {name:12s} max={d.max().item():.4g} mean={d.mean().item():.4g} "
          f"got_absmax={g.abs().max().item():.4g} ref_absmax={r.abs().max().item():.4g}")
    if not torch.allclose(g, r, atol=atol, rtol=rtol, equal_nan=False):
        raise AssertionError(
            f"{name} mismatch max={d.max().item():.4g} mean={d.mean().item():.4g}"
        )


def _tile_rel_one(got, ref, start, end, dim_last, limit):
    g = got[..., start:end, :dim_last].float()
    r = ref[..., start:end, :dim_last].float()
    denom = r.abs().amax().clamp(min=1e-12)
    return float((g - r).abs().amax() / denom)


def assert_tile_rel(name, got, ref, batch, n_head, t_len, dim, bt=BT, limit=8e-2,
                    cu_seqlens=None):
    g = got.detach()
    r = ref.detach()
    worst = 0.0
    n_bad = 0
    n_tiles = 0
    ranges = []
    if cu_seqlens is None:
        ranges = [(0, t_len)] if batch == g.shape[0] else None
        batches = range(batch)
        seq_ranges = [(b, 0, t_len) for b in batches]
    else:
        cu = [int(x) for x in cu_seqlens.tolist()]
        seq_ranges = [(0, int(cu[i]), int(cu[i + 1])) for i in range(len(cu) - 1)]

    for b, s, e in seq_ranges:
        slen = e - s
        for h in range(n_head):
            for start in range(0, slen, bt):
                end = min(start + bt, slen)
                trel = _tile_rel_one(g[b, h], r[b, h], s + start, s + end, dim, limit)
                worst = max(worst, trel)
                n_tiles += 1
                if trel > limit:
                    n_bad += 1
    print(f"  {name:12s} tile-rel max={worst:.4g} gt{limit}={n_bad}/{n_tiles} "
          f"got_absmax={g.float().abs().max().item():.4g} "
          f"ref_absmax={r.float().abs().max().item():.4g}")
    if n_bad > 0 or (not torch.isfinite(g.float()).all()):
        raise AssertionError(f"{name} tile-rel max={worst:.4g} (need <= {limit})")


def run_npu_prepare(q, k, v, g, beta, *, flags, cu_seqlens=None):
    kw = dict(
        chunk_size=flags["chunk_size"],
        use_qk_l2norm_in_kernel=flags["use_qk_l2norm_in_kernel"],
        use_gate_in_kernel=flags["use_gate_in_kernel"],
        use_beta_sigmoid_in_kernel=flags["use_beta_sigmoid_in_kernel"],
        allow_neg_eigval=flags["allow_neg_eigval"],
        use_exp2=True,
    )
    if cu_seqlens is not None:
        kw["cu_seqlens"] = cu_seqlens.contiguous().npu()
    outs = ascendc_ops.chunk_gated_delta_rule_fwd_prepare(
        q.contiguous().npu(),
        k.contiguous().npu(),
        v.contiguous().npu(),
        g.contiguous().npu(),
        beta.contiguous().npu(),
        **kw,
    )
    torch.npu.synchronize()
    return [t.cpu() if t is not None else None for t in outs]


def check_against_ref(q, k, v, g, beta, outs, flags, cu_seqlens=None):
    q_hat, k_hat, q_rstd, k_rstd, beta_out, g_cumsum, w, u, A = outs
    ref = cpu_gdn_fwd_l2norm_to_recompute(
        q, k, v, g, beta,
        chunk_size=flags["chunk_size"],
        use_qk_l2norm_in_kernel=flags["use_qk_l2norm_in_kernel"],
        use_gate_in_kernel=flags["use_gate_in_kernel"],
        use_beta_sigmoid_in_kernel=flags["use_beta_sigmoid_in_kernel"],
        allow_neg_eigval=flags["allow_neg_eigval"],
        cu_seqlens=cu_seqlens,
        layout="bnsd",
    )
    B, HK, T, K = q.shape
    HV, V = v.shape[1], v.shape[3]
    assert_close("q_hat", q_hat, ref.q, 2e-2, 2e-2)
    assert_close("k_hat", k_hat, ref.k, 2e-2, 2e-2)
    assert_close("q_rstd", q_rstd, ref.q_rstd, 5e-3, 5e-3)
    assert_close("k_rstd", k_rstd, ref.k_rstd, 5e-3, 5e-3)
    assert_close("beta_out", beta_out, ref.beta, 5e-3, 5e-3)
    assert_close("g_cumsum", g_cumsum, ref.g, 5e-3, 5e-3)
    assert_close("A", A, ref.a, 8e-2, 8e-2)
    assert_tile_rel("w", w, ref.w, B, HV, T, K, BT, cu_seqlens=cu_seqlens)
    assert_tile_rel("u", u, ref.u, B, HV, T, V, BT, cu_seqlens=cu_seqlens)


def run_required_case():
    """Original bring-up case: B=1 HK=4 HV=8 T=1792 K=V=128."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    B, HK, HV, T, K, V = 1, 4, 8, 1792, 128, 128
    q = torch.randn(B, HK, T, K, dtype=dtype)
    k = torch.randn(B, HK, T, K, dtype=dtype)
    v = torch.randn(B, HV, T, V, dtype=dtype)
    g = torch.randn(B, HV, T, dtype=torch.float32)
    beta = torch.randn(B, HV, T, dtype=torch.float32)
    flags = dict(
        chunk_size=64,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
    )
    print(f"golden={GDN_DIR}")
    print("case0 required: "
          f"B={B} HK={HK} HV={HV} T={T} K={K} V={V} BT={BT} BF16")
    outs = run_npu_prepare(q, k, v, g, beta, flags=flags)
    check_against_ref(q, k, v, g, beta, outs, flags)
    print("PASS case0")


def run_v256_case():
    """V=256 + GVA ratio 2 + tail T=96; also covers second 128-col U MMAD."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    B, HK, HV, T, K, V = 1, 4, 8, 96, 128, 256
    q = torch.randn(B, HK, T, K, dtype=dtype)
    k = torch.randn(B, HK, T, K, dtype=dtype)
    v = torch.randn(B, HV, T, V, dtype=dtype)
    g = torch.randn(B, HV, T, dtype=torch.float32)
    beta = torch.randn(B, HV, T, dtype=torch.float32)
    flags = dict(
        chunk_size=64,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
    )
    print(f"case7 V=256: B={B} HK={HK} HV={HV} T={T} K={K} V={V} BT={BT}")
    outs = run_npu_prepare(q, k, v, g, beta, flags=flags)
    check_against_ref(q, k, v, g, beta, outs, flags)
    print("PASS case7")


def run_g_ratio_case(case_id, hk, hv, seq_len, v_dim=128):
    """G=3 pack-of-3 / G=4 one-HK-per-pack smokes. K/KKT once per HK."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    B, T, K, V = 1, seq_len, 128, v_dim
    q = torch.randn(B, hk, T, K, dtype=dtype)
    k = torch.randn(B, hk, T, K, dtype=dtype)
    v = torch.randn(B, hv, T, V, dtype=dtype)
    g = torch.randn(B, hv, T, dtype=torch.float32)
    beta = torch.randn(B, hv, T, dtype=torch.float32)
    flags = dict(
        chunk_size=64,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
    )
    print(f"case{case_id} G={hv // hk}: B={B} HK={hk} HV={hv} T={T} K={K} V={V}")
    outs = run_npu_prepare(q, k, v, g, beta, flags=flags)
    check_against_ref(q, k, v, g, beta, outs, flags)
    print(f"PASS case{case_id}")


def run_small_varlen_case():
    """Packed varlen smoke: B=1, two sequences, tail + aligned, G=2."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    seqlens = (96, 64)
    B, HK, HV, T, K, V = 1, 4, 8, sum(seqlens), 128, 128
    q = torch.randn(B, HK, T, K, dtype=dtype)
    k = torch.randn(B, HK, T, K, dtype=dtype)
    v = torch.randn(B, HV, T, V, dtype=dtype)
    g = torch.randn(B, HV, T, dtype=torch.float32)
    beta = torch.randn(B, HV, T, dtype=torch.float32)
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int64)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int64).cumsum(0)
    flags = dict(
        chunk_size=64,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
    )
    print(f"case10 small varlen: B={B} HK={HK} HV={HV} T={T} seqlens={seqlens} V={V}")
    outs = run_npu_prepare(q, k, v, g, beta, flags=flags, cu_seqlens=cu)
    check_against_ref(q, k, v, g, beta, outs, flags, cu_seqlens=cu)
    print("PASS case10")


def run_gdn_case(case):
    skip = case.skip_reason()
    print(describe_case(case), flush=True)
    if skip:
        print(f"  SKIP {skip}")
        return
    inp = case_inputs(case, dtype=torch.float32, device=torch.device("cpu"), seed=0, layout="bnsd")
    q = inp["q"].to(torch.bfloat16)
    k = inp["k"].to(torch.bfloat16)
    v = inp["v"].to(torch.bfloat16)
    g = inp["g"].float()
    beta = inp["beta"].float()
    cu = inp.get("cu_seqlens")
    flags = dict(
        chunk_size=case.chunk_size,
        use_qk_l2norm_in_kernel=case.use_qk_l2norm_in_kernel,
        use_gate_in_kernel=case.use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=case.use_beta_sigmoid_in_kernel,
        allow_neg_eigval=case.allow_neg_eigval,
    )
    outs = run_npu_prepare(q, k, v, g, beta, flags=flags, cu_seqlens=cu)
    check_against_ref(q, k, v, g, beta, outs, flags, cu_seqlens=cu)
    print(f"PASS case{case.case_id}")


def main():
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=int, default=None,
                        help="0=bring-up G=2; 1-6=GdnCase; 7=V=256; 8=G=3; 9=G=4; 10=small varlen; omit=1-6 then 0,7-10")
    args = parser.parse_args()
    setup_npu()
    try:
        if args.case_id is None:
            for case in gdn_cases():
                run_gdn_case(case)
            run_required_case()
            run_v256_case()
            run_g_ratio_case(8, hk=4, hv=12, seq_len=96)
            run_g_ratio_case(9, hk=2, hv=8, seq_len=96)
            run_small_varlen_case()
        elif args.case_id == 0:
            run_required_case()
        elif args.case_id == 7:
            run_v256_case()
        elif args.case_id == 8:
            run_g_ratio_case(8, hk=4, hv=12, seq_len=96)
        elif args.case_id == 9:
            run_g_ratio_case(9, hk=2, hv=8, seq_len=96)
        elif args.case_id == 10:
            run_small_varlen_case()
        else:
            picked = [c for c in gdn_cases() if c.case_id == args.case_id]
            if not picked:
                raise SystemExit(f"unknown case-id={args.case_id}")
            run_gdn_case(picked[0])
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
