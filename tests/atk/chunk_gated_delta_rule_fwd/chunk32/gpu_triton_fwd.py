"""NVIDIA Triton GPU path for the full GDN chunk forward (outputs ``o`` and ``final_state``).

Calls the same span as ``ChunkGatedDeltaRuleFunction.forward`` (chunk.py ~282-309):
optional ``l2norm_fwd`` on q/k, optional ``fused_beta_sigmoid`` on beta, then
``chunk_gated_delta_rule_fwd`` (cumsum -> fwd_intra -> fwd_h -> fwd_o).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import ensure_fla_on_path
    from layout import check_cu_seqlens, from_fla, normalize_layout, to_fla
else:
    from .common import ensure_fla_on_path
    from .layout import check_cu_seqlens, from_fla, normalize_layout, to_fla


def _apply_qk_beta(
    q: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lines 282-288 of ``ChunkGatedDeltaRuleFunction.forward``."""
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")
    if use_qk_l2norm_in_kernel:
        try:
            from fla.modules.norm.l2norm import l2norm_fwd
        except ImportError:
            from fla.modules.l2norm import l2norm_fwd
        q, _q_rstd = l2norm_fwd(q)
        k, _k_rstd = l2norm_fwd(k)
    if use_beta_sigmoid_in_kernel:
        from fla.ops.common.gate import fused_beta_sigmoid_fwd
        # Same kernel as fused_beta_sigmoid; the autograd wrapper only saves beta for backward.
        beta = fused_beta_sigmoid_fwd(beta, scale=2.0 if allow_neg_eigval else 1.0)
    return q, k, beta


@torch.inference_mode()
def gpu_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    layout: str = "bsnd",
    use_qk_l2norm_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    fla_root: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FLA forward on CUDA; ``o`` is returned in ``layout``.

    ``q``/``k`` are raw bf16 when ``use_qk_l2norm_in_kernel`` (default). ``beta`` is
    bf16 logits when ``use_beta_sigmoid_in_kernel`` (default); the sigmoid result is fp32,
    matching ``fused_beta_sigmoid``. ``allow_neg_eigval`` uses ``2 * sigmoid``.
    """
    if q.device.type != "cuda":
        raise ValueError(f"GPU path expects CUDA tensors, got device={q.device}")
    layout = normalize_layout(layout)
    ensure_fla_on_path(fla_root)

    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
    from fla.ops.utils.index import prepare_chunk_indices

    q, k, v, g, beta = (to_fla(x, layout) for x in (q, k, v, g, beta))
    q, k, beta = _apply_qk_beta(
        q, k, beta,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
    )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    chunk_indices = None
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.long)
        check_cu_seqlens(q, cu_seqlens)
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    _g, o, _A, final_state, _h0, _g_input = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cp_context=None,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
        chunk_size=chunk_size,
    )
    return from_fla(o.to(q.dtype), layout), final_state


@torch.inference_mode()
def gpu_chunk_gated_delta_rule_fwd_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    layout: str = "bsnd",
    use_qk_l2norm_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    fla_root: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Same kernels as ``chunk_gated_delta_rule_fwd`` called one by one, keeping intermediates.

    Intermediates stay in FLA ``[B, T, HV, ...]`` layout (``h`` is ``[B, NT, HV, K, V]``);
    ``o`` is returned in ``layout``.
    """
    layout = normalize_layout(layout)
    ensure_fla_on_path(fla_root)

    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices

    q, k, v, g, beta = (to_fla(x, layout) for x in (q, k, v, g, beta))
    q, k, beta = _apply_qk_beta(
        q, k, beta,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
    )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    chunk_indices = None
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.long)
        check_cu_seqlens(q, cu_seqlens)
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    g_cs = chunk_local_cumsum(
        g, chunk_size=chunk_size, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g_cs, beta=beta, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=chunk_size,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cs, initial_state=initial_state, output_final_state=True,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, state_v_first=state_v_first, chunk_size=chunk_size,
    )
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g_cs, scale=scale, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, state_v_first=state_v_first, chunk_size=chunk_size,
    )
    return {
        "g_cumsum": g_cs, "A": A, "w": w, "u": u, "v_new": v_new, "h": h,
        "o": from_fla(o.to(q.dtype), layout), "final_state": final_state,
    }
