"""CPU / pure-PyTorch golden: L2norm(q/k) → gate/cumsum(g) → KKT+SolveTri → RecomputeWU.

Matches FLA ``ChunkGatedDeltaRuleFunction.forward`` through
``chunk_gated_delta_rule_fwd_intra`` (chunk.py ~282-284 then 52-80).

Math follows the **Triton** kernels (``exp2`` after ``* RCP_LN2``), not the NPU
``exp()`` path. No Triton / NPU dependency; runs on CPU.

Supports BSND / BNSD / TND / NTD. Compute uses FLA packed ``[B, T, H, ...]``
(``B=1`` + ``cu_seqlens`` for varlen).

- BSND ``q/k``: ``[B, T, H, K]``
- BNSD ``q/k``: ``[B, H, T, K]``
- TND  ``q/k``: ``[T, H, K]``
- NTD  ``q/k``: ``[H, T, K]``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from layout import check_cu_seqlens, from_fla, normalize_layout, to_fla
else:
    from .layout import check_cu_seqlens, from_fla, normalize_layout, to_fla

# fla.ops.utils.constant.RCP_LN2  (1/ln2, FP32 hex 0x3FB8AA3B)
RCP_LN2 = 1.4426950216
L2NORM_EPS = 1e-6


def _repeat_k_for_gva(k: torch.Tensor, hv: int) -> torch.Tensor:
    """Broadcast key heads to value heads: ``[B, T, H, K] -> [B, T, HV, K]``."""
    h = k.shape[2]
    if hv == h:
        return k
    if hv % h != 0:
        raise ValueError(f"GVA requires HV % H == 0, got H={h}, HV={hv}")
    return k.repeat_interleave(hv // h, dim=2)


def _pad_time(x: torch.Tensor, pad: int, time_dim: int = 1) -> torch.Tensor:
    if pad <= 0:
        return x
    pads = [0, 0] * x.ndim
    idx = x.ndim - 1 - time_dim
    pads[2 * idx + 1] = pad
    return F.pad(x, pads)


def cpu_chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    scale: float = RCP_LN2,
    reverse: bool = False,
) -> torch.Tensor:
    """Triton ``chunk_local_cumsum`` golden. ``g`` is ``[B, T, H]``.

    Forward:  ``out[t] = scale * sum_{s in chunk, s<=t} g[s]``
    Reverse:  ``out[t] = scale * sum_{s in chunk, s>=t} g[s]``
    """
    if chunk_size <= 0 or (chunk_size & (chunk_size - 1)) != 0:
        raise ValueError(f"chunk_size must be a power of two, got {chunk_size}")
    b, t, _h = g.shape
    bt = chunk_size
    pad = (bt - t % bt) % bt
    g_f = _pad_time(g.float(), pad, time_dim=1)
    nt = g_f.shape[1] // bt
    chunks = g_f.reshape(b, nt, bt, g_f.shape[-1])
    if reverse:
        out = chunks.flip(2).cumsum(dim=2).flip(2)
    else:
        out = chunks.cumsum(dim=2)
    return (out * scale).reshape(b, nt * bt, g_f.shape[-1])[:, :t]


def cpu_gdn_gate_chunk_cumsum(
    g: torch.Tensor,
    a_log: torch.Tensor,
    chunk_size: int,
    scale: float = RCP_LN2,
    dt_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton ``gdn_gate_chunk_cumsum`` golden.

    ``gate[t] = -exp(A_log) * softplus(g[t] + dt_bias)``
    then chunk-local cumsum with ``scale``.
    """
    x = g.float()
    if dt_bias is not None:
        x = x + dt_bias.float()
    gate = -a_log.float().exp() * F.softplus(x)
    return cpu_chunk_local_cumsum(gate, chunk_size=chunk_size, scale=scale)


def cpu_chunk_kkt_solve(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Triton KKT + ``solve_tril`` golden: ``A = (I + L)^{-1}``.

    ``g`` is already the log2-space chunk cumsum (``* RCP_LN2``)::

        L[i, j] = beta[i] * <k[i], k[j]> * exp2(g[i] - g[j])   if i > j
        L[i, j] = 0                                              if i <= j

    Returns ``A`` of shape ``[B, T, HV, BT]`` (fp32).
    """
    b, t, _h, _k = k.shape
    hv = beta.shape[2]
    bt = chunk_size
    pad = (bt - t % bt) % bt
    k_f = _pad_time(_repeat_k_for_gva(k.float(), hv), pad, time_dim=1)
    g_f = _pad_time(g.float(), pad, time_dim=1)
    beta_f = _pad_time(beta.float(), pad, time_dim=1)
    t_pad = k_f.shape[1]
    nt = t_pad // bt

    k_c = k_f.reshape(b, nt, bt, hv, -1).permute(0, 3, 1, 2, 4)  # [B,HV,NT,BT,K]
    g_c = g_f.reshape(b, nt, bt, hv).permute(0, 3, 1, 2)
    beta_c = beta_f.reshape(b, nt, bt, hv).permute(0, 3, 1, 2)

    kkt = torch.matmul(k_c, k_c.transpose(-1, -2))
    tril = torch.tril(torch.ones(bt, bt, device=k.device, dtype=torch.bool), diagonal=-1)
    gdiff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    # Mask before exp2 so the upper triangle never overflows to inf.
    gate = torch.exp2(gdiff.masked_fill(~tril, 0.0))
    l_mat = kkt * gate * beta_c.unsqueeze(-1) * tril.to(kkt.dtype)

    eye = torch.eye(bt, device=k.device, dtype=torch.float32)
    a_c = torch.linalg.inv(eye + l_mat)
    a_out = a_c.permute(0, 2, 3, 1, 4).reshape(b, t_pad, hv, bt)
    return a_out[:, :t]


def cpu_recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton ``recompute_w_u_fwd`` golden.

        u = A @ (v * beta)
        w = A @ (k * beta * exp2(g))
    """
    b, t, _h, k_dim = k.shape
    hv, v_dim = v.shape[2], v.shape[3]
    bt = a.shape[-1]
    pad = (bt - t % bt) % bt

    k_f = _pad_time(_repeat_k_for_gva(k.float(), hv), pad, time_dim=1)
    v_f = _pad_time(v.float(), pad, time_dim=1)
    beta_f = _pad_time(beta.float(), pad, time_dim=1)
    g_f = _pad_time(g.float(), pad, time_dim=1)
    a_f = _pad_time(a.float(), pad, time_dim=1)
    t_pad = k_f.shape[1]
    nt = t_pad // bt

    a_c = a_f.reshape(b, nt, bt, hv, bt).permute(0, 3, 1, 2, 4)
    v_c = (v_f * beta_f.unsqueeze(-1)).reshape(b, nt, bt, hv, v_dim).permute(0, 3, 1, 2, 4)
    k_c = k_f * beta_f.unsqueeze(-1) * torch.exp2(g_f).unsqueeze(-1)
    k_c = k_c.reshape(b, nt, bt, hv, k_dim).permute(0, 3, 1, 2, 4)

    u = torch.matmul(a_c, v_c).permute(0, 2, 3, 1, 4).reshape(b, t_pad, hv, v_dim)[:, :t]
    w = torch.matmul(a_c, k_c).permute(0, 2, 3, 1, 4).reshape(b, t_pad, hv, k_dim)[:, :t]
    return w, u


def cpu_chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton ``chunk_gated_delta_rule_fwd_intra`` golden. Returns ``(w, u, A)``.

    ``g`` must already be the chunk-local cumsum in log2 space.
    """
    a = cpu_chunk_kkt_solve(k, g, beta, chunk_size)
    w, u = cpu_recompute_w_u(k, v, beta, a, g)
    return w, u, a


def _apply_per_sequence(fn, tensors: tuple[torch.Tensor, ...], cu_seqlens: torch.Tensor, **kwargs):
    if tensors[0].shape[0] != 1:
        raise ValueError("cu_seqlens requires batch size 1")
    outs = []
    for i in range(len(cu_seqlens) - 1):
        s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        sliced = tuple(x[:, s:e] for x in tensors)
        outs.append(fn(*sliced, **kwargs))
    if isinstance(outs[0], torch.Tensor):
        return torch.cat(outs, dim=1)
    return tuple(torch.cat([o[j] for o in outs], dim=1) for j in range(len(outs[0])))


def cpu_l2norm_fwd(
    x: torch.Tensor,
    eps: float = L2NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton ``l2norm_fwd`` golden. Normalize along the last dim.

        rstd = 1 / sqrt(sum_k x_k^2 + eps)
        y    = x * rstd

    Returns ``(y, rstd)`` with ``y`` the same shape as ``x`` and
    ``rstd`` of shape ``x.shape[:-1]`` (fp32).
    """
    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().sum(dim=-1) + eps)
    y = (x_f * rstd.unsqueeze(-1)).to(x.dtype)
    return y, rstd


def cpu_beta_sigmoid(beta: torch.Tensor, allow_neg_eigval: bool = False) -> torch.Tensor:
    """Triton ``fused_beta_sigmoid``: ``sigmoid(beta)`` or ``2 * sigmoid(beta)``."""
    scale = 2.0 if allow_neg_eigval else 1.0
    return scale * torch.sigmoid(beta.float())


@dataclass
class GdnL2normToRecomputeRef:
    """Outputs of the L2norm → Recompute segment."""

    q: torch.Tensor
    k: torch.Tensor
    q_rstd: torch.Tensor | None
    k_rstd: torch.Tensor | None
    g: torch.Tensor
    beta: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    a: torch.Tensor


def cpu_chunk_gated_delta_rule_fwd_wy(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_gate_in_kernel: bool = False,
    a_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cumsum(g) + intra. ``k`` must already be L2-normalized if that path is on.

    Prefer ``cpu_gdn_fwd_l2norm_to_recompute`` for the full operator span.
    """
    if use_gate_in_kernel:
        if a_log is None:
            raise ValueError("a_log is required when use_gate_in_kernel=True")

        def _gate_cumsum(g_s, **_kw):
            return cpu_gdn_gate_chunk_cumsum(
                g_s, a_log, chunk_size=chunk_size, scale=RCP_LN2, dt_bias=dt_bias,
            )
    else:
        def _gate_cumsum(g_s, **_kw):
            return cpu_chunk_local_cumsum(g_s, chunk_size=chunk_size, scale=RCP_LN2)

    if cu_seqlens is not None:
        g_cs = _apply_per_sequence(_gate_cumsum, (g,), cu_seqlens)

        def _intra(k_s, v_s, g_s, beta_s, **_kw):
            return cpu_chunk_gated_delta_rule_fwd_intra(
                k_s, v_s, g_s, beta_s, chunk_size=chunk_size,
            )

        w, u, a = _apply_per_sequence(_intra, (k, v, g_cs, beta), cu_seqlens)
        return g_cs, w, u, a

    g_cs = _gate_cumsum(g)
    w, u, a = cpu_chunk_gated_delta_rule_fwd_intra(k, v, g_cs, beta, chunk_size=chunk_size)
    return g_cs, w, u, a


def cpu_gdn_fwd_l2norm_to_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    a_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    layout: str = "bsnd",
    eps: float = L2NORM_EPS,
) -> GdnL2normToRecomputeRef:
    """CPU golden from L2norm through RecomputeWU (not including fwd_h / fwd_o).

    Pipeline (same as FLA autograd forward up to ``fwd_intra``)::

        q, k  --l2norm-->  q̂, k̂          (if use_qk_l2norm_in_kernel)
        beta  --sigmoid--> β              (if use_beta_sigmoid_in_kernel)
        g     --cumsum---> G              (gate fused iff use_gate_in_kernel)
        k̂,v,G,β --KKT+SolveTri+WY--> w, u, A

    Args:
        q, k / v / g, beta: see ``layout``.
        layout: ``bsnd`` ``[B,T,H,D]``, ``bnsd`` ``[B,H,T,D]``,
            ``tnd`` ``[T,H,D]``, ``ntd`` ``[H,T,D]``.
            Gates (g/beta) drop the last D: BSND ``[B,T,H]``, BNSD ``[B,H,T]``,
            TND ``[T,H]``, NTD ``[H,T]``.
        cu_seqlens: packed offsets ``[N+1]``. Required for multi-sequence varlen.
            FLA compute uses B=1; TND/NTD are always packed.
        a_log / dt_bias: required when ``use_gate_in_kernel=True``, shape ``[HV]``.

    ``q`` is normalized for alignment with FLA; Recompute itself only consumes ``k̂``.
    """
    layout = normalize_layout(layout)
    q = to_fla(q, layout)
    k = to_fla(k, layout)
    v = to_fla(v, layout)
    g = to_fla(g, layout)
    beta = to_fla(beta, layout)
    if cu_seqlens is not None:
        check_cu_seqlens(q, cu_seqlens)

    q_rstd = k_rstd = None
    if use_qk_l2norm_in_kernel:
        q, q_rstd = cpu_l2norm_fwd(q, eps=eps)
        k, k_rstd = cpu_l2norm_fwd(k, eps=eps)

    if use_beta_sigmoid_in_kernel:
        beta = cpu_beta_sigmoid(beta, allow_neg_eigval=allow_neg_eigval)

    g_cs, w, u, a = cpu_chunk_gated_delta_rule_fwd_wy(
        k, v, g, beta,
        chunk_size=chunk_size,
        use_gate_in_kernel=use_gate_in_kernel,
        a_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
    )
    return GdnL2normToRecomputeRef(
        q=from_fla(q, layout),
        k=from_fla(k, layout),
        q_rstd=from_fla(q_rstd, layout),
        k_rstd=from_fla(k_rstd, layout),
        g=from_fla(g_cs, layout),
        beta=from_fla(beta, layout),
        w=from_fla(w, layout),
        u=from_fla(u, layout),
        a=from_fla(a, layout),
    )


def ref_to_tensor_dict(ref: GdnL2normToRecomputeRef) -> dict[str, torch.Tensor | None]:
    """Map the dataclass onto the same keys used by the GPU dump."""
    return {
        "q_hat": ref.q,
        "k_hat": ref.k,
        "q_rstd": ref.q_rstd,
        "k_rstd": ref.k_rstd,
        "beta_out": ref.beta,
        "g_cumsum": ref.g,
        "w": ref.w,
        "u": ref.u,
        "A": ref.a,
    }


def _finite_status(name: str, t: torch.Tensor) -> str:
    finite = torch.isfinite(t).all().item()
    return f"{name}={'ok' if finite else 'NONFINITE'}"


def main() -> None:
    """Run the six explicit cases in ``gdn_cases()`` on the CPU golden."""
    import argparse
    import time

    if __package__ in (None, ""):
        from cases import case_inputs, describe_case, gdn_cases, select_cases
        from layout import layout_skip_reason, parse_layouts
    else:
        from .cases import case_inputs, describe_case, gdn_cases, select_cases
        from .layout import layout_skip_reason, parse_layouts

    # Cases are written in gdn_cases(), not read from CSV.
    cases = gdn_cases()

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--case-id", type=int, default=None, help="run a single case id (1-6)")
    parser.add_argument(
        "--layout",
        default="bsnd",
        help="comma-separated layouts: bsnd,bnsd,tnd,ntd or all",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    layouts = parse_layouts(args.layout)
    if args.case_id is not None:
        cases = select_cases(args.case_id)

    failed = 0
    for case in cases:
        skip = case.skip_reason()
        print(f"\n== CPU {describe_case(case)}", flush=True)
        if skip:
            print(f"   SKIP: {skip}", flush=True)
            failed += 1
            continue
        for layout in layouts:
            why = layout_skip_reason(layout, case.batch, case.varlen)
            if why:
                print(f"   SKIP {layout.upper()}: {why}", flush=True)
                continue
            try:
                inputs = case_inputs(
                    case, dtype=torch.float32, device=device, seed=case.case_id, layout=layout,
                )
                t0 = time.perf_counter()
                ref = cpu_gdn_fwd_l2norm_to_recompute(
                    inputs["q"],
                    inputs["k"],
                    inputs["v"],
                    inputs["g"],
                    inputs["beta"],
                    a_log=inputs.get("A_log"),
                    dt_bias=inputs.get("dt_bias"),
                    cu_seqlens=inputs.get("cu_seqlens"),
                    layout=layout,
                    **case.flags(),
                )
            except Exception as exc:
                failed += 1
                print(f"   FAIL {layout.upper()}: {type(exc).__name__}: {exc}", flush=True)
                continue
            dt = time.perf_counter() - t0
            status = " ".join(
                _finite_status(n, t) for n, t in (("w", ref.w), ("u", ref.u), ("A", ref.a))
            )
            print(
                f"   [{layout.upper()}] q={tuple(ref.q.shape)} w={tuple(ref.w.shape)} "
                f"u={tuple(ref.u.shape)} A={tuple(ref.a.shape)}  {status}  {dt:.1f}s",
                flush=True,
            )
            if inputs.get("cu_seqlens") is not None:
                cu = inputs["cu_seqlens"]
                print(
                    f"   cu_seqlens n={cu.numel()} first={int(cu[0])} last={int(cu[-1])} "
                    f"n_seq={cu.numel() - 1}"
                )
            if not (torch.isfinite(ref.w).all() and torch.isfinite(ref.u).all() and torch.isfinite(ref.a).all()):
                print("   WARN: non-finite values (allow_neg_eigval can make I+L ill-conditioned)")
    if failed:
        raise SystemExit(f"CPU run finished with {failed} failed case(s)")
    print("\nCPU: all requested cases passed")


if __name__ == "__main__":
    main()
