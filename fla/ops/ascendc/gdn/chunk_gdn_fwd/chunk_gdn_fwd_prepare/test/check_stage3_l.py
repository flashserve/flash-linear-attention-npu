"""Stage0-6 audit vs host golden.

Kernel dumps (until Stage7 owns w/u):
  u  <- bf16 ND vb = v * β  (Stage6; overwrites Stage4 Y NZ)
  w  <- bf16 ND kbg = k' * β * exp2(g')
  A  <- bf16 ND (I+L)^{-1} 64x64 per chunk (Stage5)

L is reconstructed from kernel k_hat / g_cumsum / beta_out (clip=50).

Required case, same flags as test_chunk_gdn_fwd_prepare.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from fla_npu.ops import ascendc as ascendc_ops

sys.path.insert(0, "/data/w00933206/ops/Gate_of_Babylon/tests/gdn")
from cpu_golden import cpu_gdn_fwd_l2norm_to_recompute  # noqa: E402

B, HK, HV, T, K, V, BT = 1, 4, 8, 1792, 128, 128, 64
CLIP = 50.0
EPS = 1e-7


def setup_npu():
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def bf16_storage_as_f32(t: torch.Tensor, shape):
    bits = t.detach().cpu().contiguous().view(torch.int16).numpy()
    return torch.from_numpy(np.array(bits).view(np.float32).reshape(shape).copy())


def metrics(actual, golden, mask=None):
    a = actual.detach().cpu().double().numpy()
    g = golden.detach().cpu().double().numpy()
    if mask is not None:
        m = mask.detach().cpu().numpy().astype(bool)
        if m.ndim == 2:
            a, g = a[..., m], g[..., m]
        else:
            a, g = a[m], g[m]
    else:
        a, g = a.reshape(-1), g.reshape(-1)
    fin = np.isfinite(a) & np.isfinite(g)
    a, g = a[fin], g[fin]
    abs_err = np.abs(a - g)
    rel = abs_err / (np.abs(g) + EPS)
    return {
        "n": int(a.size),
        "max_abs": float(abs_err.max()) if a.size else 0.0,
        "mean_abs": float(abs_err.mean()) if a.size else 0.0,
        "mare": float(rel.max()) if a.size else 0.0,
        "mere": float(rel.mean()) if a.size else 0.0,
        "got_absmax": float(np.abs(a).max()) if a.size else 0.0,
        "ref_absmax": float(np.abs(g).max()) if a.size else 0.0,
        "nan": int((~np.isfinite(actual.detach().cpu().numpy().reshape(-1))).sum()),
    }


def fmt(m):
    return (f"max|err|={m['max_abs']:.4g} mean={m['mean_abs']:.4g} "
            f"MARE={100 * m['mare']:.4g}% MERE={100 * m['mere']:.4g}% "
            f"got_absmax={m['got_absmax']:.4g} ref_absmax={m['ref_absmax']:.4g} n={m['n']}")


def nd64_to_nz_c08(nd: torch.Tensor) -> torch.Tensor:
    """64x64 fp32 ND -> cube NZ C0=8 (Fixpipe isChannelSplit / NzC08ToNd64 inverse).

    nd[..., row, col] with col = fc * 8 + c0  ->  nz[..., fc, row, c0]
    """
    *lead, m, n = nd.shape
    if m != BT or n != BT:
        raise ValueError(f"expected [..., {BT}, {BT}], got {tuple(nd.shape)}")
    return nd.reshape(*lead, BT, 8, 8).transpose(-3, -2).contiguous()


def golden_L_tiles(k_bnsd, g_bnsd, beta_bnsd, clip=True):
    """k [B,HK,T,K], g/beta [B,HV,T] -> L [B,HV,NT,BT,BT] fp32."""
    g_ratio = HV // k_bnsd.shape[1]
    k_hv = k_bnsd.float().repeat_interleave(g_ratio, dim=1)
    nt = T // BT
    k_c = k_hv.reshape(B, HV, nt, BT, -1)
    g_c = g_bnsd.float().reshape(B, HV, nt, BT)
    b_c = beta_bnsd.float().reshape(B, HV, nt, BT)
    kkt = torch.matmul(k_c, k_c.transpose(-1, -2))
    tril = torch.tril(torch.ones(BT, BT, dtype=torch.bool), diagonal=-1)
    gdiff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    if clip:
        gdiff = gdiff.clamp(-CLIP, CLIP)
    gate = torch.exp2(gdiff)
    return kkt * gate * b_c.unsqueeze(-1) * tril.to(kkt.dtype)


def golden_s6(k_hat, v, beta_out, g_cumsum):
    """vb = v*β, kbg = k'*β*exp2(g'). k_hat is [B,HK,...]; repeat to HV."""
    g_ratio = HV // k_hat.shape[1]
    k_hv = k_hat.float().repeat_interleave(g_ratio, dim=1)
    b = beta_out.float().unsqueeze(-1)
    g = g_cumsum.float().unsqueeze(-1)
    vb = v.float() * b
    kbg = k_hv * b * torch.exp2(g)
    return vb, kbg


def main():
    setup_npu()
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(B, HK, T, K, dtype=dtype)
    k = torch.randn(B, HK, T, K, dtype=dtype)
    v = torch.randn(B, HV, T, V, dtype=dtype)
    g = torch.randn(B, HV, T, dtype=torch.float32)
    beta = torch.randn(B, HV, T, dtype=torch.float32)

    outs = ascendc_ops.chunk_gdn_fwd_prepare(
        q.contiguous().npu(),
        k.contiguous().npu(),
        v.contiguous().npu(),
        g.contiguous().npu(),
        beta.contiguous().npu(),
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        use_exp2=True,
    )
    torch.npu.synchronize()
    q_hat, k_hat, q_rstd, k_rstd, beta_out, g_cumsum, w, u, A = [t.cpu() for t in outs]

    ref = cpu_gdn_fwd_l2norm_to_recompute(
        q, k, v, g, beta,
        chunk_size=BT,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        layout="bnsd",
    )

    print("=== Stage1 (sanity; Stage3-5 must not stomp hats/g/beta) ===")
    for name, got, r, atol in (
        ("q_hat", q_hat, ref.q, 2e-2),
        ("k_hat", k_hat, ref.k, 2e-2),
        ("q_rstd", q_rstd, ref.q_rstd, 5e-3),
        ("k_rstd", k_rstd, ref.k_rstd, 5e-3),
        ("beta_out", beta_out, ref.beta, 5e-3),
        ("g_cumsum", g_cumsum, ref.g, 5e-3),
    ):
        d = (got.float() - r.float()).abs()
        ok = torch.allclose(got.float(), r.float(), atol=atol, rtol=atol, equal_nan=False)
        print(f"  {name:10s} max={d.max().item():.4g} mean={d.mean().item():.4g} "
              f"got_absmax={got.float().abs().max().item():.4g} PASS={ok}")

    nt = T // BT
    L_s1 = golden_L_tiles(k_hat, g_cumsum, beta_out, clip=True)
    eye32 = torch.eye(32, dtype=torch.float64)
    eye64 = torch.eye(BT, dtype=torch.float64)
    L_d = L_s1.double()
    XL = torch.linalg.inv(eye32 + L_d[..., 32:, 32:])

    print("\n=== Stage4 Y NZ (skipped: u is Stage6 vb dump until Stage7) ===")

    print("\n=== Stage5 A = (I+L)^{-1} (bf16 ND) ===")
    A_got = A.float().reshape(B, HV, nt, BT, BT)
    A_inv = torch.linalg.inv(eye64 + L_d)
    m_a = metrics(A_got, A_inv)
    print("  vs inv(I+L_s1) ", fmt(m_a), f"nan={m_a['nan']}")
    m_ag = metrics(A.float(), ref.a.float())
    print("  vs golden A    ", fmt(m_ag))
    res = (eye64 + L_d) @ A_got.double() - eye64
    print(f"  |(I+L)@A-I| max={res.abs().max().item():.4g} mean={res.abs().mean().item():.4g}")
    XR = torch.linalg.inv(eye32 + L_d[..., :32, :32])
    print("  A[:32,:32] vs XR", fmt(metrics(A_got[..., :32, :32], XR)))
    print("  A[:32,32:] ~ 0  ", fmt(metrics(A_got[..., :32, 32:],
                                           torch.zeros(B, HV, nt, 32, 32))))
    print("  A[32:,32:] vs XL", fmt(metrics(A_got[..., 32:, 32:], XL)))
    print(f"  A absmax={A_got.abs().max().item():.4g}")

    print("\n=== Stage6 vb (GM u) / kbg (GM w) vs kernel k_hat/g/beta ===")
    vb_ref, kbg_ref = golden_s6(k_hat, v, beta_out, g_cumsum)
    vb_got = u.float()
    kbg_got = w.float()
    vb_bf = vb_ref.to(torch.bfloat16).float()
    kbg_bf = kbg_ref.to(torch.bfloat16).float()
    m_vb = metrics(vb_got, vb_ref)
    m_kbg = metrics(kbg_got, kbg_ref)
    m_vb_b = metrics(vb_got, vb_bf)
    m_kbg_b = metrics(kbg_got, kbg_bf)
    vb_ok = torch.allclose(vb_got, vb_ref, atol=2e-2, rtol=2e-2, equal_nan=False)
    kbg_ok = torch.allclose(kbg_got, kbg_ref, atol=2e-2, rtol=2e-2, equal_nan=False)
    print("  vb  vs fp32 ", fmt(m_vb), f"nan={m_vb['nan']} PASS={vb_ok}")
    print("  vb  vs bf16 ", fmt(m_vb_b))
    print("  kbg vs fp32 ", fmt(m_kbg), f"nan={m_kbg['nan']} PASS={kbg_ok}")
    print("  kbg vs bf16 ", fmt(m_kbg_b))
    print(f"  vb absmax={vb_got.abs().max().item():.4g} ref={vb_ref.abs().max().item():.4g}")
    print(f"  kbg absmax={kbg_got.abs().max().item():.4g} ref={kbg_ref.abs().max().item():.4g}")


if __name__ == "__main__":
    main()
