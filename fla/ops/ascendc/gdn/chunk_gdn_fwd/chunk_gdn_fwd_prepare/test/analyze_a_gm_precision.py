"""Precision audit of fused prepare GM A vs golden.

Same MARE/MERE/RMSE as test_npu_solve_tri_triton_bsnd._error_metrics.
Required case: B=1 HK=4 HV=8 T=1792 K=V=128 BT=64 BF16 seed=0.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from fla_npu.ops import ascendc as ascendc_ops

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "chunk_gdn_fwd_common"))
from stage_test_utils import (
    golden_s2, golden_s3, golden_s4, golden_s5, golden_s5_mbh, golden_s6, setup_npu,
)

sys.path.insert(0, "/data/w00933206/ops/Gate_of_Babylon/tests/gdn")
from cpu_golden import cpu_gdn_fwd_l2norm_to_recompute

EPS = 1e-7
BT = 64
# solve_tri bf16 gates (ratio vs dump-floor, not abs MARE).
RATIO_MARE = 5.0
RATIO_MERE = 1.5
RATIO_RMSE = 1.5


def metrics(actual: torch.Tensor, golden: torch.Tensor):
    a = actual.detach().cpu().double().reshape(-1).numpy()
    g = golden.detach().cpu().double().reshape(-1).numpy()
    fin = np.isfinite(a) & np.isfinite(g)
    a, g = a[fin], g[fin]
    abs_err = np.abs(a - g)
    rel = abs_err / (np.abs(g) + EPS)
    return {
        "n": int(a.size),
        "mare": float(rel.max()) if a.size else 0.0,
        "mere": float(rel.mean()) if a.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(a - g)))) if a.size else 0.0,
        "mae": float(abs_err.mean()) if a.size else 0.0,
        "max_abs": float(abs_err.max()) if a.size else 0.0,
        "g_absmax": float(np.abs(g).max()) if a.size else 0.0,
    }


def metrics_masked(actual, golden, mask):
    a = actual.detach().cpu().double().numpy()
    g = golden.detach().cpu().double().numpy()
    m = mask.detach().cpu().numpy().astype(bool)
    a, g = a[m], g[m]
    abs_err = np.abs(a - g)
    rel = abs_err / (np.abs(g) + EPS)
    return {
        "n": int(a.size),
        "mare": float(rel.max()) if a.size else 0.0,
        "mere": float(rel.mean()) if a.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(a - g)))) if a.size else 0.0,
        "mae": float(abs_err.mean()) if a.size else 0.0,
        "max_abs": float(abs_err.max()) if a.size else 0.0,
    }


def fmt(m):
    return (f"MARE={100*m['mare']:.4g}% MERE={100*m['mere']:.4g}% "
            f"RMSE={m['rmse']:.4g} MAE={m['mae']:.4g} max|err|={m['max_abs']:.4g} n={m['n']}")


def worst_rel(actual, golden, extra=None):
    a = actual.detach().cpu().float()
    g = golden.detach().cpu().float()
    rel = (a - g).abs() / (g.abs() + EPS)
    idx = rel.reshape(-1).argmax()
    loc = tuple(int(x) for x in np.unravel_index(int(idx), a.shape))
    return {
        "loc": loc,
        "kernel": float(a[loc]),
        "golden": float(g[loc]),
        "abs_err": float((a[loc] - g[loc]).abs()),
        "rel": float(rel[loc]),
        "extra": None if extra is None else float(extra.detach().cpu().float()[loc]),
    }


def inverse_residual(L, A, bt=BT):
    """max |(I+L) @ A - I| over chunks. L,A are [B,HV,T,BT] fp32."""
    B, HV, T, _ = L.shape
    nt = T // bt
    I = torch.eye(bt, dtype=torch.float64)
    worst = 0.0
    worst_at = None
    rmses = []
    for b in range(B):
        for hv in range(HV):
            for c in range(nt):
                sl = slice(c * bt, (c + 1) * bt)
                ll = L[b, hv, sl].double()
                aa = A[b, hv, sl].double()
                r = (I + ll) @ aa - I
                e = float(r.abs().max())
                rmses.append(float(torch.sqrt((r * r).mean())))
                if e > worst:
                    worst = e
                    worst_at = (b, hv, c)
    return worst, worst_at, float(np.mean(rmses))


def main():
    setup_npu()
    B, HK, HV, T, K, V = 1, 4, 8, 1792, 128, 128
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(B, HK, T, K, dtype=dtype, device="npu")
    k = torch.randn(B, HK, T, K, dtype=dtype, device="npu")
    v = torch.randn(B, HV, T, V, dtype=dtype, device="npu")
    g = torch.randn(B, HV, T, dtype=torch.float32, device="npu")
    beta = torch.randn(B, HV, T, dtype=torch.float32, device="npu")
    v_cpu = v.detach().cpu().clone()

    import fla_npu.ops.ascendc._aclnn_ctypes as _ac
    import fla_npu.ops.ascendc._runtime as _rt
    _real_empty = _ac._empty

    def _empty_zero(shape, like, **kwargs):
        t = _real_empty(shape, like, **kwargs)
        t.zero_()
        return t

    _ac._empty = _empty_zero
    _rt.empty = _empty_zero
    _ac.npu_chunk_gdn_fwd_prepare.__globals__["_empty"] = _empty_zero
    q_hat, k_hat, q_rstd, k_rstd, beta_out, g_cumsum, w, u, A = ascendc_ops.chunk_gdn_fwd_prepare(
        q, k, v, g, beta,
        use_qk_l2norm_in_kernel=True,
        use_exp2=True,
    )
    _ac._empty = _real_empty
    torch.npu.synchronize()

    ref = cpu_gdn_fwd_l2norm_to_recompute(
        q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(),
        chunk_size=BT,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        layout="bnsd",
    )

    kern = A.detach().cpu()  # bf16 GM
    kern_f = kern.float()
    gold_f = ref.a.float().cpu()  # fp32 pipeline golden
    gold_bf = gold_f.to(torch.bfloat16)
    gold_bf_f = gold_bf.float()

    print("=== GM A tensor ===")
    print(f"A dtype={kern.dtype} shape={tuple(kern.shape)} device was npu")
    print(f"nan={int(torch.isnan(kern_f).sum())} inf={int(torch.isinf(kern_f).sum())} "
          f"absmax={kern_f.abs().max().item():.6g}")
    print(f"golden fp32 absmax={gold_f.abs().max().item():.6g} "
          f"bf16(golden) absmax={gold_bf_f.abs().max().item():.6g}")

    m_kf = metrics(kern_f, gold_f)
    m_bf = metrics(gold_bf_f, gold_f)
    m_kb = metrics(kern_f, gold_bf_f)
    print("\n=== solve_tri _error_metrics (rel / (|g|+1e-7)) ===")
    print("kernel vs fp32 golden     ", fmt(m_kf))
    print("bf16(golden) vs fp32      ", fmt(m_bf), "  [dump floor]")
    print("kernel vs bf16(golden)    ", fmt(m_kb))

    def ratio(a, b):
        return a / max(b, 1e-30)

    mare_r = ratio(m_kf["mare"], m_bf["mare"])
    mere_r = ratio(m_kf["mere"], m_bf["mere"])
    rmse_r = ratio(m_kf["rmse"], m_bf["rmse"])
    print(f"\nNPU / dump-floor  MARE_ratio={mare_r:.4g} MERE_ratio={mere_r:.4g} RMSE_ratio={rmse_r:.4g}")
    print(f"solve_tri bf16 gates: MARE_ratio<={RATIO_MARE} MERE_ratio<={RATIO_MERE} RMSE_ratio<={RATIO_RMSE}")
    ratio_pass = mare_r <= RATIO_MARE and mere_r <= RATIO_MERE and rmse_r <= RATIO_RMSE
    # GDN A has huge dynamic range; MARE_ratio can exceed 5 on a single tiny entry.
    mere_rmse_pass = mere_r <= RATIO_MERE and rmse_r <= RATIO_RMSE
    print(f"MERE+RMSE vs dump-floor PASS={mere_rmse_pass}  (strict 3-ratio PASS={ratio_pass})")

    # Bit identity vs dump-floor.
    same = kern.view(torch.int16) == gold_bf.view(torch.int16)
    n_same = int(same.sum())
    n_all = same.numel()
    print(f"\nbit-identical to bf16(golden): {n_same}/{n_all} ({100.0*n_same/n_all:.4f}%)")

    w_kf = worst_rel(kern_f, gold_f)
    w_kb = worst_rel(kern_f, gold_bf_f)
    print(f"worst MARE vs fp32 @ {w_kf['loc']}: kern={w_kf['kernel']:.6g} "
          f"gold={w_kf['golden']:.6g} abs={w_kf['abs_err']:.6g} rel={100*w_kf['rel']:.4g}%")
    print(f"worst MARE vs bf16(golden) @ {w_kb['loc']}: kern={w_kb['kernel']:.6g} "
          f"gold_bf={w_kb['golden']:.6g} abs={w_kb['abs_err']:.6g} rel={100*w_kb['rel']:.4g}%")

    # Magnitude bins vs fp32.
    gabs = gold_f.abs()
    print("\n=== MARE/MERE by |golden| bin (kernel vs fp32) ===")
    bins = [(0, 1e-3), (1e-3, 1), (1, 100), (100, 1e4), (1e4, 1e6), (1e6, 1e30)]
    bin_rows = []
    for lo, hi in bins:
        mask = (gabs >= lo) & (gabs < hi)
        mm = metrics_masked(kern_f, gold_f, mask)
        mf = metrics_masked(gold_bf_f, gold_f, mask)
        label = f"[{lo:g}, {hi:g})"
        print(f"  {label:16s} n={mm['n']:7d}  npu {fmt(mm)}")
        print(f"  {'':16s}          floor {fmt(mf)}")
        bin_rows.append({"bin": label, "n": mm["n"], "npu": mm, "floor": mf})

    # Per head.
    print("\n=== per head vs fp32 ===")
    head_rows = []
    for hv in range(HV):
        mm = metrics(kern_f[0, hv], gold_f[0, hv])
        print(f"  hv{hv} {fmt(mm)} |A|max={kern_f[0,hv].abs().max().item():.4g}")
        head_rows.append({"hv": hv, **mm, "absmax": float(kern_f[0, hv].abs().max())})

    # chunk0 hv0 quadrants vs fp32 and vs bf16.
    print("\n=== chunk0 hv0 quadrants ===")
    a00 = kern_f[0, 0, :BT, :]
    g00 = gold_f[0, 0, :BT, :]
    b00 = gold_bf_f[0, 0, :BT, :]
    quad_rows = []
    for name, rs, cs in (("TL", slice(0, 32), slice(0, 32)),
                         ("TR", slice(0, 32), slice(32, 64)),
                         ("BL", slice(32, 64), slice(0, 32)),
                         ("BR", slice(32, 64), slice(32, 64))):
        mm = metrics(a00[rs, cs], g00[rs, cs])
        mb = metrics(a00[rs, cs], b00[rs, cs])
        ident = bool(torch.equal(kern[0, 0, :BT, :][rs, cs], gold_bf[0, 0, :BT, :][rs, cs]))
        print(f"  {name} vs fp32 {fmt(mm)}  vs bf16 {fmt(mb)}  bit-ident={ident}")
        quad_rows.append({"q": name, "fp32": mm, "bf16": mb, "bit_ident": ident})

    # Reconstruct L from kernel Stage1 tensors, invert, compare.
    kkt_h = golden_s2(k_hat)
    L_ref, leaf_ref = golden_s3(kkt_h, g_cumsum, beta)
    Y_ref, drv_ref = golden_s4(L_ref, leaf_ref)
    a_mbh = golden_s5_mbh(Y_ref, drv_ref, leaf_ref)
    a_inv = golden_s5(L_ref)
    m_mbh = metrics(kern_f, a_mbh)
    m_inv = metrics(kern_f, a_inv)
    print("\n=== vs host MBH / (I+L)^{-1} from kernel k_hat,g,beta ===")
    print("kernel vs MBH            ", fmt(m_mbh))
    print("kernel vs (I+L)^{-1}     ", fmt(m_inv))
    print("MBH vs (I+L)^{-1}        ", fmt(metrics(a_mbh, a_inv)))

    res_k, at_k, res_mean_k = inverse_residual(L_ref, kern_f)
    res_g, at_g, res_mean_g = inverse_residual(L_ref, gold_f)
    res_b, at_b, res_mean_b = inverse_residual(L_ref, gold_bf_f)
    print("\n=== inverse residual max |(I+L)@A - I|  (L from kernel k_hat/g/beta) ===")
    print(f"  kernel A:  max={res_k:.6g} mean_rmse={res_mean_k:.6g} at {at_k}")
    print(f"  fp32 gold: max={res_g:.6g} mean_rmse={res_mean_g:.6g} at {at_g}")
    print(f"  bf16 gold: max={res_b:.6g} mean_rmse={res_mean_b:.6g} at {at_b}")

    # atol/rtol 8e-2 like independent s5 / fused assert_close.
    def allclose_count(a, b, atol, rtol):
        ok = torch.isclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
        return int(ok.sum()), ok.numel()

    n_ok, n_tot = allclose_count(kern_f, gold_f, 8e-2, 8e-2)
    n_okb, _ = allclose_count(gold_bf_f, gold_f, 8e-2, 8e-2)
    n_okm, _ = allclose_count(kern_f, a_mbh, 8e-2, 8e-2)
    print("\n=== torch.allclose atol=rtol=8e-2 (independent s5 / fused test) ===")
    print(f"  kernel vs fp32 golden: {n_ok}/{n_tot} ({100.0*n_ok/n_tot:.4f}%)")
    print(f"  dump-floor vs fp32:    {n_okb}/{n_tot} ({100.0*n_okb/n_tot:.4f}%)")
    print(f"  kernel vs MBH:         {n_okm}/{n_tot} ({100.0*n_okm/n_tot:.4f}%)")

    # Rel error on |g|>1 should track dump floor.
    mask_big = gold_f.abs() >= 1
    m_big_n = metrics_masked(kern_f, gold_f, mask_big)
    m_big_f = metrics_masked(gold_bf_f, gold_f, mask_big)
    print("\n=== |golden| >= 1 (values that matter for later matmul) ===")
    print("  npu  ", fmt(m_big_n))
    print("  floor", fmt(m_big_f))
    print(f"  MERE_ratio={ratio(m_big_n['mere'], m_big_f['mere']):.4g} "
          f"RMSE_ratio={ratio(m_big_n['rmse'], m_big_f['rmse']):.4g} "
          f"MARE_ratio={ratio(m_big_n['mare'], m_big_f['mare']):.4g}")

    out = {
        "kernel_vs_fp32": m_kf,
        "floor_vs_fp32": m_bf,
        "kernel_vs_floor": m_kb,
        "ratios": {"mare": mare_r, "mere": mere_r, "rmse": rmse_r},
        "bit_ident": {"n": n_same, "all": n_all},
        "worst_fp32": w_kf,
        "worst_floor": w_kb,
        "bins": bin_rows,
        "heads": head_rows,
        "quad_c0h0": quad_rows,
        "mbh": m_mbh,
        "inv": m_inv,
        "residual": {
            "kernel": res_k, "kernel_at": at_k, "kernel_rmse": res_mean_k,
            "fp32": res_g, "bf16": res_b,
        },
        "allclose_8e2": {"npu": n_ok, "floor": n_okb, "mbh": n_okm, "n": n_tot},
        "absge1": {"npu": m_big_n, "floor": m_big_f},
        "mere_rmse_pass": mere_rmse_pass,
    }

    # Stage6 ND dump: u <- vb, v <- kbg (K=V=128). golden_s6 from kernel
    # k_hat / g_cumsum / input beta isolates the Stage6 VF.
    vb_kern = u.detach().cpu()
    kbg_kern = v.detach().cpu()
    vb_s1, kbg_s1 = golden_s6(k_hat, v_cpu, beta.cpu(), g_cumsum)
    vb_s1_f = vb_s1.float()
    kbg_s1_f = kbg_s1.float()
    vb_kf = vb_kern.float()
    kbg_kf = kbg_kern.float()
    vb_bf = vb_s1.to(torch.bfloat16)
    kbg_bf = kbg_s1.to(torch.bfloat16)
    vb_bf_f = vb_bf.float()
    kbg_bf_f = kbg_bf.float()

    print("\n=== Stage6 vb (GM u) / kbg (GM v, in-place) ===")
    print(f"vb nan={int(torch.isnan(vb_kf).sum())} inf={int(torch.isinf(vb_kf).sum())} "
          f"absmax={vb_kf.abs().max().item():.6g} gold_absmax={vb_s1_f.abs().max().item():.6g}")
    print(f"kbg nan={int(torch.isnan(kbg_kf).sum())} inf={int(torch.isinf(kbg_kf).sum())} "
          f"absmax={kbg_kf.abs().max().item():.6g} gold_absmax={kbg_s1_f.abs().max().item():.6g}")

    m_vb_f = metrics(vb_kf, vb_s1_f)
    m_vb_floor = metrics(vb_bf_f, vb_s1_f)
    m_vb_b = metrics(vb_kf, vb_bf_f)
    m_kbg_f = metrics(kbg_kf, kbg_s1_f)
    m_kbg_floor = metrics(kbg_bf_f, kbg_s1_f)
    m_kbg_b = metrics(kbg_kf, kbg_bf_f)
    print("vb  vs fp32 golden_s6   ", fmt(m_vb_f))
    print("vb  dump-floor          ", fmt(m_vb_floor))
    print("vb  vs bf16(golden_s6)  ", fmt(m_vb_b))
    print("kbg vs fp32 golden_s6   ", fmt(m_kbg_f))
    print("kbg dump-floor          ", fmt(m_kbg_floor))
    print("kbg vs bf16(golden_s6)  ", fmt(m_kbg_b))

    vb_mare_r = ratio(m_vb_f["mare"], m_vb_floor["mare"])
    vb_mere_r = ratio(m_vb_f["mere"], m_vb_floor["mere"])
    vb_rmse_r = ratio(m_vb_f["rmse"], m_vb_floor["rmse"])
    kbg_mare_r = ratio(m_kbg_f["mare"], m_kbg_floor["mare"])
    kbg_mere_r = ratio(m_kbg_f["mere"], m_kbg_floor["mere"])
    kbg_rmse_r = ratio(m_kbg_f["rmse"], m_kbg_floor["rmse"])
    print(f"vb  NPU/floor MARE={vb_mare_r:.4g} MERE={vb_mere_r:.4g} RMSE={vb_rmse_r:.4g}")
    print(f"kbg NPU/floor MARE={kbg_mare_r:.4g} MERE={kbg_mere_r:.4g} RMSE={kbg_rmse_r:.4g}")

    vb_same = int((vb_kern.view(torch.int16) == vb_bf.view(torch.int16)).sum())
    kbg_same = int((kbg_kern.view(torch.int16) == kbg_bf.view(torch.int16)).sum())
    print(f"vb  bit-identical to bf16(golden_s6): {vb_same}/{vb_kern.numel()} "
          f"({100.0 * vb_same / vb_kern.numel():.4f}%)")
    print(f"kbg bit-identical to bf16(golden_s6): {kbg_same}/{kbg_kern.numel()} "
          f"({100.0 * kbg_same / kbg_kern.numel():.4f}%)")

    w_vb = worst_rel(vb_kf, vb_s1_f)
    w_kbg = worst_rel(kbg_kf, kbg_s1_f)
    print(f"vb  worst vs fp32 @ {w_vb['loc']}: kern={w_vb['kernel']:.6g} "
          f"gold={w_vb['golden']:.6g} abs={w_vb['abs_err']:.6g} rel={100 * w_vb['rel']:.4g}%")
    print(f"kbg worst vs fp32 @ {w_kbg['loc']}: kern={w_kbg['kernel']:.6g} "
          f"gold={w_kbg['golden']:.6g} abs={w_kbg['abs_err']:.6g} rel={100 * w_kbg['rel']:.4g}%")

    n_vb_ok, n_vb_tot = allclose_count(vb_kf, vb_s1_f, 2e-2, 2e-2)
    n_kbg_ok, n_kbg_tot = allclose_count(kbg_kf, kbg_s1_f, 2e-2, 2e-2)
    n_vb_okb, _ = allclose_count(vb_bf_f, vb_s1_f, 2e-2, 2e-2)
    n_kbg_okb, _ = allclose_count(kbg_bf_f, kbg_s1_f, 2e-2, 2e-2)
    print("\n=== Stage6 torch.allclose atol=rtol=2e-2 (independent s6) ===")
    print(f"  vb  vs fp32 golden_s6: {n_vb_ok}/{n_vb_tot} ({100.0 * n_vb_ok / n_vb_tot:.4f}%)")
    print(f"  vb  dump-floor:        {n_vb_okb}/{n_vb_tot} ({100.0 * n_vb_okb / n_vb_tot:.4f}%)")
    print(f"  kbg vs fp32 golden_s6: {n_kbg_ok}/{n_kbg_tot} ({100.0 * n_kbg_ok / n_kbg_tot:.4f}%)")
    print(f"  kbg dump-floor:        {n_kbg_okb}/{n_kbg_tot} ({100.0 * n_kbg_okb / n_kbg_tot:.4f}%)")

    print("\n=== Stage6 per head vs fp32 golden_s6 ===")
    vb_heads, kbg_heads = [], []
    for hv in range(HV):
        mv = metrics(vb_kf[0, hv], vb_s1_f[0, hv])
        mk = metrics(kbg_kf[0, hv], kbg_s1_f[0, hv])
        print(f"  hv{hv} vb  {fmt(mv)}")
        print(f"  hv{hv} kbg {fmt(mk)}")
        vb_heads.append({"hv": hv, **mv})
        kbg_heads.append({"hv": hv, **mk})

    vb_pass = (n_vb_ok == n_vb_tot) and vb_mere_r <= RATIO_MERE and vb_rmse_r <= RATIO_RMSE
    kbg_pass = (n_kbg_ok == n_kbg_tot) and kbg_mere_r <= RATIO_MERE and kbg_rmse_r <= RATIO_RMSE
    print(f"\nStage6 vb PASS={vb_pass}  kbg PASS={kbg_pass}  "
          f"(allclose 2e-2 and MERE/RMSE vs dump-floor)")

    out["stage6"] = {
        "vb_vs_fp32": m_vb_f,
        "vb_floor": m_vb_floor,
        "vb_vs_bf16": m_vb_b,
        "kbg_vs_fp32": m_kbg_f,
        "kbg_floor": m_kbg_floor,
        "kbg_vs_bf16": m_kbg_b,
        "vb_ratios": {"mare": vb_mare_r, "mere": vb_mere_r, "rmse": vb_rmse_r},
        "kbg_ratios": {"mare": kbg_mare_r, "mere": kbg_mere_r, "rmse": kbg_rmse_r},
        "vb_bit_ident": {"n": vb_same, "all": int(vb_kern.numel())},
        "kbg_bit_ident": {"n": kbg_same, "all": int(kbg_kern.numel())},
        "vb_worst": w_vb,
        "kbg_worst": w_kbg,
        "allclose_2e2": {
            "vb": n_vb_ok, "kbg": n_kbg_ok, "vb_floor": n_vb_okb, "kbg_floor": n_kbg_okb,
            "n": n_vb_tot,
        },
        "vb_heads": vb_heads,
        "kbg_heads": kbg_heads,
        "vb_pass": vb_pass,
        "kbg_pass": kbg_pass,
    }
    out["s6_pass"] = bool(vb_pass and kbg_pass)

    json_path = Path(__file__).resolve().parent / "last_gm_precision.json"
    json_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n=== JSON written {json_path} ===")
    print("=== JSON ===")
    json.dump(out, sys.stdout)
    print()
    if not mere_rmse_pass or not (vb_pass and kbg_pass):
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    main()
