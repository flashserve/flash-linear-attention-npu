"""Chunk32 ATK executor.

The NPU DUT and the mixed-tolerance CPU emulation live here. The case JSON
standard ``mixed_tolerance_bm`` selects that emulation. Determinism and
memory checks use the same NPU entry.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _finite_tuple

# --- layout ---
LAYOUTS = ("bsnd", "bnsd", "tnd", "ntd")


def normalize_layout(layout: str) -> str:
    name = layout.strip().lower()
    if name not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    return name


def to_fla(x: torch.Tensor, layout: str) -> torch.Tensor:
    """User layout to FLA ``[B, T, H, ...]``."""
    layout = normalize_layout(layout)
    if layout == "bsnd":
        return x
    if layout == "bnsd":
        return x.transpose(1, 2).contiguous()
    if layout == "tnd":
        return x.unsqueeze(0).contiguous()
    return x.transpose(0, 1).unsqueeze(0).contiguous()


def from_fla(x: torch.Tensor | None, layout: str) -> torch.Tensor | None:
    """FLA ``[B, T, H, ...]`` back to the user layout."""
    if x is None:
        return None
    layout = normalize_layout(layout)
    if layout == "bsnd":
        return x.contiguous()
    if layout == "bnsd":
        return x.transpose(1, 2).contiguous()
    if x.shape[0] != 1:
        raise ValueError(f"{layout} requires packed batch size 1, got B={x.shape[0]}")
    packed = x.squeeze(0)
    if layout == "tnd":
        return packed.contiguous()
    return packed.transpose(0, 1).contiguous()


def check_cu_seqlens(x_fla: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
    if x_fla.shape[0] != 1:
        raise ValueError(f"varlen requires packed B=1, got B={x_fla.shape[0]}")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must be [N+1], got {tuple(cu_seqlens.shape)}")
    total = int(cu_seqlens[-1])
    if total != x_fla.shape[1]:
        raise ValueError(f"cu_seqlens[-1]={total} does not match packed T={x_fla.shape[1]}")
    if int(cu_seqlens[0]) != 0:
        raise ValueError("cu_seqlens[0] must be 0")
    diffs = cu_seqlens[1:] - cu_seqlens[:-1]
    if (diffs <= 0).any():
        raise ValueError("cu_seqlens must be strictly increasing")

# --- shared numeric helpers ---
RCP_LN2 = 1.4426950216


def _repeat_k_for_gva(k: torch.Tensor, hv: int) -> torch.Tensor:
    """``[B, T, H, K]`` to ``[B, T, HV, K]``."""
    heads = k.shape[2]
    if hv == heads:
        return k
    if hv % heads != 0:
        raise ValueError(f"GVA requires HV % H == 0, got H={heads}, HV={hv}")
    return k.repeat_interleave(hv // heads, dim=2)


def _pad_time(x: torch.Tensor, pad: int, time_dim: int = 1) -> torch.Tensor:
    if pad <= 0:
        return x
    pads = [0, 0] * x.ndim
    idx = x.ndim - 1 - time_dim
    pads[2 * idx + 1] = pad
    return F.pad(x, pads)

L2NORM_EPS = 1e-6


def _to_chunks(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """``[B, T, H, ...]`` to ``[B, H, NT, BT, ...]``. ``T`` is already padded."""
    batch, tokens = x.shape[:2]
    if tokens % chunk_size != 0:
        raise ValueError(f"T={tokens} is not divisible by chunk_size={chunk_size}")
    n_chunks = tokens // chunk_size
    y = x.reshape(batch, n_chunks, chunk_size, *x.shape[2:])
    perm = [0, 3, 1, 2, *range(4, y.ndim)]
    return y.permute(*perm).contiguous()


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().sum(dim=-1) + L2NORM_EPS)
    return (x_f * rstd.unsqueeze(-1)).to(x.dtype)


def _apply_qk_beta(
    q: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True.")
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)
    if use_beta_sigmoid_in_kernel:
        scale = 2.0 if allow_neg_eigval else 1.0
        beta = scale * torch.sigmoid(beta.float())
    return q, k, beta

# --- mixed-tolerance CPU emulation ---
EMU_CHUNK_SIZES = (16, 32)
# Emulated block-dot accumulation: "fp32" (plain fp32 matmul), "fp64" (exact-ish sum, RNE to fp32),
# "fp64_rz" (exact-ish sum, round-toward-zero to fp32, closer to tensor-core truncation).
DOT_MODE = "fp64_rz"


def _round_to_fp32_rz(x: torch.Tensor) -> torch.Tensor:
    y = x.float()
    overshoot = y.double().abs() > x.abs()
    return torch.where(overshoot, torch.nextafter(y, torch.zeros_like(y)), y)


TC_GROUP = 8
TC_BITS = 24


def _tc_dot(a: torch.Tensor, b: torch.Tensor, acc: torch.Tensor | None) -> torch.Tensor:
    """Tensor-core block FMA model: per group of ``TC_GROUP`` products plus the fp32 accumulator,
    exact products are aligned to the largest exponent, truncated to ``TC_BITS`` bits, summed,
    and the sum is rounded toward zero to fp32. Groups are chained sequentially over K."""
    k_dim = a.shape[-1]
    out_shape = (*torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]), a.shape[-2], b.shape[-1])
    c = torch.zeros(out_shape, dtype=torch.float64) if acc is None else acc.double().expand(out_shape)
    for k0 in range(0, k_dim, TC_GROUP):
        p = a[..., :, k0:k0 + TC_GROUP, None].double() * b[..., None, k0:k0 + TC_GROUP, :].double()
        terms = torch.cat([c.unsqueeze(-2), p], dim=-2)
        _m, e = torch.frexp(terms.abs().amax(dim=-2, keepdim=True))
        quantum = torch.ldexp(torch.ones_like(_m), e - TC_BITS)
        c = _round_to_fp32_rz((torch.trunc(terms / quantum) * quantum).sum(dim=-2)).double()
    return c.float()


def _dot(a: torch.Tensor, b: torch.Tensor, acc: torch.Tensor | None = None) -> torch.Tensor:
    """Emulated block dot of ``a`` and ``b`` into ``acc`` (bf16 operands held in fp32, fp32 accumulator)."""
    if DOT_MODE == "tc":
        return _tc_dot(a, b, acc)
    if DOT_MODE == "fp32":
        out = a @ b
    else:
        out = a.double() @ b.double()
        out = out.float() if DOT_MODE == "fp64" else _round_to_fp32_rz(out)
    return out if acc is None else acc + out


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 with round-to-nearest-even and keep computing in fp32."""
    return x.to(torch.bfloat16).float()


def _from_chunks(x: torch.Tensor, t: int) -> torch.Tensor:
    """``[B, HV, NT, BT, ...]`` -> ``[B, T, HV, ...]``."""
    b, hv, nt, bt = x.shape[:4]
    perm = (0, 2, 3, 1, *range(4, x.ndim))
    return x.permute(*perm).reshape(b, nt * bt, hv, *x.shape[4:])[:, :t]


def _solve_tril_16(a: torch.Tensor) -> torch.Tensor:
    """``solve_tril_16x16`` forward substitution on strictly-lower ``a`` ``[..., 16, 16]`` (fp32)."""
    ai = -torch.tril(a, diagonal=-1)
    for i in range(2, 16):
        b_a = -a[..., i, :].clone()
        b_a[..., i:] = 0.0
        b_a = b_a + (b_a.unsqueeze(-1) * ai).sum(dim=-2)
        ai[..., i, :] = b_a
    return ai + torch.eye(16, dtype=a.dtype)


def emu_solve_tril(a: torch.Tensor) -> torch.Tensor:
    """``solve_tril`` for BT=16/32 on ``[..., BT, BT]`` fp32; returns ``(I + A)^-1`` rounded to bf16."""
    bt = a.shape[-1]
    if bt == 16:
        return _bf16(_solve_tril_16(a))
    if bt != 32:
        raise NotImplementedError(f"emulated solve_tril supports BT=16/32, got {bt}")
    ai11 = _solve_tril_16(a[..., :16, :16])
    ai22 = _solve_tril_16(a[..., 16:, 16:])
    ai21 = -((ai22 @ a[..., 16:, :16]) @ ai11)
    out = torch.zeros_like(a)
    out[..., :16, :16] = ai11
    out[..., 16:, 16:] = ai22
    out[..., 16:, :16] = ai21
    return _bf16(out)


def _fwd_single_emu(q, k, v, g, beta, scale, initial_state, chunk_size, keep_stages, state_v_first):
    b, t, _h, k_dim = k.shape
    hv, v_dim = v.shape[2], v.shape[3]
    bt = chunk_size
    pad = (bt - t % bt) % bt
    nt = (t + pad) // bt

    def chunks(x, repeat=False):
        x = x.float()
        if repeat:
            x = _repeat_k_for_gva(x, hv)
        return _to_chunks(_pad_time(x, pad), bt)

    q_c, k_c, v_c = chunks(q, True), chunks(k, True), chunks(v)
    beta_c = chunks(beta)                                       # [B, HV, NT, BT]
    valid = (torch.arange(nt * bt) < t).reshape(nt, bt)         # [NT, BT]

    # chunk_local_cumsum: fp32 cumsum inside each chunk, then * RCP_LN2
    g_c = chunks(g).cumsum(dim=-1) * RCP_LN2

    strict = torch.tril(torch.ones(bt, bt, dtype=torch.bool), diagonal=-1)
    causal = torch.tril(torch.ones(bt, bt, dtype=torch.bool))
    pair_valid = valid[:, :, None] & valid[:, None, :]          # [NT, BT, BT]
    gdiff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)

    # chunk_scaled_dot_kkt_fwd: fp32 output
    a = _dot(k_c, k_c.transpose(-1, -2)) * torch.exp2(gdiff.masked_fill(~strict, 0.0))
    a = (a * beta_c.unsqueeze(-1)).masked_fill(~(strict & pair_valid), 0.0)
    ai = emu_solve_tril(a)

    # recompute_w_u_fwd. Cast k * beta * exp2(g) once, then dot.
    u_c = _bf16(_dot(ai, _bf16(v_c * beta_c.unsqueeze(-1))))
    kb = _bf16(k_c * beta_c.unsqueeze(-1) * torch.exp2(g_c).unsqueeze(-1))
    w_c = _bf16(_dot(ai, kb))

    # chunk_gated_delta_rule_fwd_h + chunk_fwd_o
    if initial_state is None:
        state = torch.zeros(b, hv, k_dim, v_dim, dtype=torch.float32)
    else:
        # Kernel layout is [V, K] when state_v_first; the recurrence itself is [K, V].
        state = initial_state.float()
        if state_v_first:
            state = state.transpose(-1, -2)
        state = state.contiguous()
    o_c = torch.empty(b, hv, nt, bt, v_dim, dtype=torch.float32)
    h_tail = (v_dim, k_dim) if state_v_first else (k_dim, v_dim)
    h_all = torch.empty(b, hv, nt, *h_tail, dtype=torch.bfloat16) if keep_stages else None
    vnew_all = torch.empty(b, hv, nt, bt, v_dim, dtype=torch.float32) if keep_stages else None
    for i_t in range(nt):
        n_valid = min(bt, t - i_t * bt)
        m_t = valid[i_t]
        q_i, k_i, g_i = q_c[:, :, i_t], k_c[:, :, i_t], g_c[:, :, i_t]

        h_i = _bf16(state)
        vv = u_c[:, :, i_t] - _dot(w_c[:, :, i_t], h_i)
        v_new = _bf16(vv)

        g_last = g_i[..., n_valid - 1]
        decay = torch.exp2((g_last.unsqueeze(-1) - g_i).masked_fill(~m_t, 0.0)) * m_t
        state = state * torch.exp2(g_last)[..., None, None]
        state = _dot(k_i.transpose(-1, -2), _bf16(vv * decay.unsqueeze(-1)), state)

        o_inter = _dot(q_i, h_i) * torch.exp2(g_i).unsqueeze(-1)
        attn = _dot(q_i, k_i.transpose(-1, -2)) * torch.exp2(gdiff[:, :, i_t].masked_fill(~causal, 0.0))
        attn = attn.masked_fill(~(causal & pair_valid[i_t]), 0.0)
        o_c[:, :, i_t] = _bf16(o_inter * scale + _dot(_bf16(attn), v_new) * scale)

        if keep_stages:
            stored_h = h_i.transpose(-1, -2) if state_v_first else h_i
            h_all[:, :, i_t] = stored_h.to(torch.bfloat16)
            vnew_all[:, :, i_t] = v_new

    o = _from_chunks(o_c, t).to(torch.bfloat16)
    if state_v_first:
        state = state.transpose(-1, -2).contiguous()
    stages = None
    if keep_stages:
        stages = {
            "g_cumsum": _from_chunks(g_c, t),
            "A": _from_chunks(ai, t).to(torch.bfloat16),
            "w": _from_chunks(w_c, t).to(torch.bfloat16),
            "u": _from_chunks(u_c, t).to(torch.bfloat16),
            "v_new": _from_chunks(vnew_all, t).to(torch.bfloat16),
            "h": h_all.transpose(1, 2).contiguous(),            # [B, NT, HV, K, V] or [B, NT, HV, V, K]
        }
    return o, state, stages


def cpu_chunk_gated_delta_rule_fwd_emu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 32,
    layout: str = "bsnd",
    use_qk_l2norm_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    return_stages: bool = False,
):
    """Mixed-tolerance CPU golden. Returns ``(o, final_state)`` or ``(o, final_state, stages)``.

    Defaults match chunk.py 282-288: raw ``q``/``k`` are L2-normalized (rounded to the
    input dtype) and ``beta`` logits go through sigmoid in fp32. ``allow_neg_eigval``
    uses ``2 * sigmoid(beta)``. ``state_v_first`` stores ``initial_state``, ``h`` and
    ``final_state`` as ``[V, K]`` instead of ``[K, V]``.
    ``o`` is bf16 in ``layout``; ``final_state`` is fp32.
    ``stages`` (FLA ``[B, T, HV, ...]`` layout, packed for varlen) holds the emulated
    ``g_cumsum / A / w / u / v_new / h`` for step-by-step comparison with the kernels.
    """
    if chunk_size not in EMU_CHUNK_SIZES:
        raise NotImplementedError(f"emulation supports chunk_size in {EMU_CHUNK_SIZES}, got {chunk_size}")
    layout = normalize_layout(layout)
    q, k, v, g, beta = (to_fla(x, layout) for x in (q, k, v, g, beta))
    q, k, beta = _apply_qk_beta(
        q, k, beta,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
    )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if cu_seqlens is None:
        o, final_state, stages = _fwd_single_emu(
            q, k, v, g, beta, scale, initial_state, chunk_size, return_stages, state_v_first,
        )
    else:
        check_cu_seqlens(q, cu_seqlens)
        outs, states, stage_list = [], [], []
        for i in range(len(cu_seqlens) - 1):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            h0 = None if initial_state is None else initial_state[i : i + 1]
            o_i, ht_i, st_i = _fwd_single_emu(
                *(x[:, s:e] for x in (q, k, v, g, beta)), scale, h0, chunk_size, return_stages, state_v_first,
            )
            outs.append(o_i)
            states.append(ht_i)
            stage_list.append(st_i)
        o = torch.cat(outs, dim=1)
        final_state = torch.cat(states, dim=0)
        stages = None
        if return_stages:
            stages = {key: torch.cat([st[key] for st in stage_list], dim=1) for key in stage_list[0]}

    o = from_fla(o, layout)
    final_state = final_state if output_final_state else None
    if return_stages:
        return o, final_state, stages
    return o, final_state

# --- case inputs and ATK entry ---
OP_NAME = "chunk_gated_delta_rule_fwd_chunk32"


def _seqlens(spec: dict[str, Any]) -> Optional[list[int]]:
    raw = spec.get("seqlens")
    text = "" if raw is None else str(raw).strip()
    if not text:
        return None
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"seqlens must be positive lengths, got {values}")
    return values


def _state_count(spec: dict[str, Any], seqlens: Optional[list[int]]) -> int:
    return len(seqlens) if seqlens is not None else int(spec["B"])


def _flags(spec: dict[str, Any]) -> tuple[str, bool, bool, bool, bool, bool]:
    layout = str(spec.get("layout", "BNSD")).upper()
    if layout not in {"BNSD", "BSND", "NTD", "TND"}:
        raise ValueError(f"layout must be BNSD, BSND, NTD or TND, got {layout}")
    sequence_major = layout in {"BSND", "TND"}
    allow_neg = bool(spec.get("allow_neg_eigval", False))
    state_v_first = bool(spec.get("state_v_first", False))
    disable_recompute = bool(spec.get("disable_recompute", False))
    out_h = bool(spec.get("out_h", False))
    return layout, sequence_major, allow_neg, state_v_first, disable_recompute, out_h


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(f"expected a scalar, got shape {tuple(value.shape)}")
        return value.detach().cpu().item()
    return value


def _bool(value: Any) -> bool:
    value = _scalar(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(_scalar(value)).strip()


def _spec_from_inputs(input_data: InputDataset) -> dict[str, Any]:
    """Read the typed attributes. Tensors stay with the caller."""

    values = input_data.kwargs
    chunk_size = int(_scalar(values["chunk_size"]))
    if chunk_size != 32:
        raise ValueError(f"chunk32 suite only runs chunk_size 32, got {chunk_size}")
    seqlens = _text(values["seqlens"])
    if seqlens.lower() == "none":
        seqlens = ""
    return {
        "layout": _text(values["layout"]).upper(),
        "state_dtype": _text(values["state_dtype"]).lower(),
        "seqlens": seqlens,
        "allow_neg_eigval": _bool(values["allow_neg_eigval"]),
        "state_v_first": _bool(values["state_v_first"]),
        "disable_recompute": _bool(values["disable_recompute"]),
        "out_h": _bool(values["out_h"]),
        "g_mode": _text(values["g_mode"]).lower(),
        "scale": float(_scalar(values["scale"])),
        "output_final_state": _bool(values["output_final_state"]),
        "chunk_size": chunk_size,
    }


def _apply_gate(g: torch.Tensor, g_mode: str, sequence_major: bool) -> torch.Tensor:
    """``normal`` keeps the sampled gate. The other modes write a fixed pattern."""

    if g_mode == "normal":
        return g
    if g_mode == "strong":
        return torch.full_like(g, -4.0)
    tokens = int(g.shape[1] if sequence_major else g.shape[2])
    if g_mode == "spike":
        if tokens < 2:
            raise ValueError("spike decay needs at least two tokens")
        g = g.clone()
        if sequence_major:
            g[:, 0, :] = -100.0
            g[:, 1, :] = 1.0
        else:
            g[:, :, 0] = -100.0
            g[:, :, 1] = 1.0
        return g
    if g_mode == "mix":
        if tokens < 64:
            raise ValueError("mixed decay needs a second chunk, T>=64")
        g = g.clone()
        if sequence_major:
            g[:, 32:64, :] = -4.0
        else:
            g[:, :, 32:64] = -4.0
        return g
    raise ValueError(f"g_mode must be normal/strong/spike/mix, got {g_mode!r}")


def _initial_state(state_name: str, shape: tuple[int, ...]) -> Optional[torch.Tensor]:
    if state_name in {"", "none"}:
        return None
    if state_name == "zero":
        return torch.zeros(shape, dtype=torch.float32)
    if state_name not in {"bf16", "fp32"}:
        raise ValueError(f"state_dtype must be none/bf16/fp32/zero, got {state_name!r}")
    count = 1
    for dim in shape:
        count *= int(dim)
    values = torch.arange(count, dtype=torch.float32)
    data = ((values.remainder(257) - 128) * 1.0e-4).reshape(shape).contiguous()
    if state_name == "bf16":
        return data.to(torch.bfloat16)
    return data


def build_inputs(spec: dict[str, Any], tensors: tuple, device: torch.device) -> dict[str, Any]:
    """Use the tensors ATK sampled. Both nodes see the same values."""

    q, k, v, g, beta = tensors
    q = q.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    k = k.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    v = v.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    g = g.detach().to(device="cpu", dtype=torch.float32).contiguous()
    beta = beta.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    layout, sequence_major, allow_neg, state_v_first, disable_recompute, out_h = _flags(spec)
    if sequence_major:
        batch, tokens, k_heads, key = (int(dim) for dim in q.shape)
        batch_v, tokens_v, v_heads, value = (int(dim) for dim in v.shape)
        gate_shape = (batch, tokens, v_heads)
    else:
        batch, k_heads, tokens, key = (int(dim) for dim in q.shape)
        batch_v, v_heads, tokens_v, value = (int(dim) for dim in v.shape)
        gate_shape = (batch, v_heads, tokens)
    if k.shape != q.shape or (batch_v, tokens_v) != (batch, tokens):
        raise ValueError(
            f"q/k/v disagree: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if tuple(g.shape) != gate_shape or tuple(beta.shape) != gate_shape:
        raise ValueError(
            f"g/beta must be {gate_shape}, got g={tuple(g.shape)} beta={tuple(beta.shape)}"
        )
    if key != 128 or value != 128:
        raise ValueError(f"K and V must be 128, got K={key} V={value}")
    if k_heads <= 0 or v_heads % k_heads != 0:
        raise ValueError(f"Hv must be a positive multiple of Hk, got Hk={k_heads} Hv={v_heads}")
    spec["B"] = batch
    seqlens = _seqlens(spec)
    if seqlens is None:
        if batch < 1 or tokens < 1:
            raise ValueError(f"B and T must be positive, got B={batch} T={tokens}")
    else:
        if batch != 1:
            raise ValueError("packed input requires B=1")
        if sum(seqlens) != tokens:
            raise ValueError(f"sum(seqlens)={sum(seqlens)} != T={tokens}")
    g = _apply_gate(g, str(spec.get("g_mode", "normal")).lower(), sequence_major)
    state_name = str(spec.get("state_dtype", "none")).lower()
    count = _state_count(spec, seqlens)
    state_shape = (count, v_heads, value, key) if state_v_first else (count, v_heads, key, value)
    initial = _initial_state(state_name, state_shape)

    def move(tensor):
        if tensor is None or device.type == "cpu":
            return tensor
        return tensor.to(device)

    return {
        "q": move(q),
        "k": move(k),
        "v": move(v),
        "g": move(g),
        "beta": move(beta),
        "initial_state": move(initial),
        "seqlens": seqlens,
        "scale": float(spec.get("scale", 128 ** -0.5)),
        "layout": layout,
        "sequence_major": sequence_major,
        "allow_neg_eigval": allow_neg,
        "state_v_first": state_v_first,
        "disable_recompute": disable_recompute,
        "out_h": out_h,
    }


def _cu(seqlens: Optional[list[int]]) -> Optional[torch.Tensor]:
    if seqlens is None:
        return None
    values = [0]
    for length in seqlens:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int64)


def _cpu(tensor: Optional[torch.Tensor], dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.detach().to(device="cpu").contiguous()


def _state_dtype(spec: dict[str, Any]) -> torch.dtype:
    state_name = str(spec.get("state_dtype", "none")).lower()
    return torch.bfloat16 if state_name == "bf16" else torch.float32


def _l2_public(q: torch.Tensor, k: torch.Tensor, layout: str):
    """Hats stay in the input layout. rstd is always ``[B, Hk, T]``."""

    def one(x: torch.Tensor):
        values = x.float()
        rstd = torch.rsqrt(values.square().sum(dim=-1) + 1e-6)
        hat = (values * rstd.unsqueeze(-1)).to(torch.bfloat16)
        return hat, rstd

    q_hat, q_rstd = one(q)
    k_hat, k_rstd = one(k)
    if layout in {"BSND", "TND"}:
        q_rstd = q_rstd.permute(0, 2, 1)
        k_rstd = k_rstd.permute(0, 2, 1)
    return q_hat.contiguous(), k_hat.contiguous(), q_rstd.contiguous(), k_rstd.contiguous()


def _beta_public(beta: torch.Tensor, layout: str, allow_neg: bool) -> torch.Tensor:
    beta_bth = beta.transpose(1, 2) if layout in {"BNSD", "NTD"} else beta
    scale = 2.0 if allow_neg else 1.0
    return (scale * torch.sigmoid(beta_bth.float())).contiguous()


def _expect_public(spec: dict[str, Any], outputs) -> None:
    """The loaded operator has to match the current ten-tuple contract."""

    export_ga = bool(spec.get("disable_recompute", False))
    present = {"o", "beta_eff"}
    if bool(spec.get("output_final_state", True)):
        present.add("final")
    if export_ga:
        present.update({"g_cumsum", "A", "q_hat", "k_hat", "q_rstd", "k_rstd"})
    if bool(spec.get("out_h", False)):
        present.add("h")
    names = (
        "o", "final", "g_cumsum", "A", "beta_eff", "h",
        "q_hat", "k_hat", "q_rstd", "k_rstd",
    )
    bad = [
        name for name, value in zip(names, outputs)
        if (value is None) != (name not in present)
    ]
    if bad:
        raise RuntimeError(
            "npu_chunk_gated_delta_rule_fwd outputs do not match "
            f"disable_recompute={export_ga}: {bad}"
        )


def _pack(spec: dict[str, Any], o, state, g_cumsum=None, a=None, beta_eff=None,
          h=None, q_hat=None, k_hat=None, q_rstd=None, k_rstd=None):
    """Same slot order as the ten-tuple, with disabled exports left as None."""

    export_ga = bool(spec.get("disable_recompute", False))
    # Chunk32 always normalises, so public hats follow disable_recompute.
    # beta_eff is required on, and h follows its own switch.
    export_hats = export_ga
    export_beta = True
    out_h = bool(spec.get("out_h", False))
    state_dtype = _state_dtype(spec)
    return (
        _cpu(o, torch.bfloat16),
        _cpu(state, state_dtype),
        _cpu(g_cumsum, torch.float32) if export_ga else None,
        _cpu(a, torch.bfloat16) if export_ga else None,
        _cpu(beta_eff, torch.float32) if export_beta else None,
        _cpu(h, torch.bfloat16) if out_h else None,
        _cpu(q_hat, torch.bfloat16) if export_hats else None,
        _cpu(k_hat, torch.bfloat16) if export_hats else None,
        _cpu(q_rstd, torch.float32) if export_hats else None,
        _cpu(k_rstd, torch.float32) if export_hats else None,
    )


def _exports_from_stages(inputs: dict[str, Any], stages: dict[str, torch.Tensor]):
    """Public optional tensors from the mixed-tolerance emulation.

    The emulation stores the chunk-local gate sum in log2 units. Dividing by
    ``RCP_LN2`` recovers the natural sum the kernel writes. ``h`` uses the
    same ``[V, K]`` layout as ``final_state`` when ``state_v_first`` is set.
    """

    g_cumsum = (stages["g_cumsum"].float() / RCP_LN2).contiguous()
    a = stages["A"].permute(0, 2, 1, 3).contiguous()
    # The emulation already stores h as [V, K] when state_v_first, matching the kernel.
    h = stages["h"]
    q_hat, k_hat, q_rstd, k_rstd = _l2_public(inputs["q"], inputs["k"], inputs["layout"])
    beta_eff = _beta_public(inputs["beta"], inputs["layout"], inputs["allow_neg_eigval"])
    return g_cumsum, a, beta_eff, h.contiguous(), q_hat, k_hat, q_rstd, k_rstd


def _accuracy_standard(task_result: TaskResult) -> str:
    """Name of the accuracy standard in the case JSON.

    ``mixed_tolerance_bm`` compares the NPU with the CPU emulation.
    """

    config = getattr(task_result, "case_config", None)
    standard = getattr(config, "standard", None)
    if isinstance(standard, dict):
        acc = standard.get("acc")
    else:
        acc = getattr(standard, "acc", None)
    if hasattr(acc, "model_dump"):
        acc = acc.model_dump()
    if isinstance(acc, str):
        return acc
    if isinstance(acc, dict) and acc:
        return next(iter(acc))
    return "mixed_tolerance_bm"


def run_cpu(spec: dict[str, Any], tensors: tuple):
    inputs = build_inputs(spec, tensors, torch.device("cpu"))
    # Public NTD/TND are rank-4 packed tensors, not the emulator's rank-3 names.
    golden_layout = "bsnd" if inputs["sequence_major"] else "bnsd"
    o, final_state, stages = cpu_chunk_gated_delta_rule_fwd_emu(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        scale=inputs["scale"],
        initial_state=inputs["initial_state"],
        cu_seqlens=_cu(inputs["seqlens"]),
        chunk_size=32,
        layout=golden_layout,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=inputs["allow_neg_eigval"],
        state_v_first=inputs["state_v_first"],
        return_stages=True,
    )
    if not inputs["sequence_major"]:
        o = o.transpose(1, 2).contiguous()
    g_cumsum, a, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd = _exports_from_stages(inputs, stages)
    return _pack(
        spec, o, final_state, g_cumsum, a, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd,
    )



def _metadata(seqlens: Optional[list[int]]) -> tuple[Optional[list[int]], Optional[list[int]]]:
    if seqlens is None:
        return None, None
    cu = [0]
    chunks = []
    for sequence, length in enumerate(seqlens):
        cu.append(cu[-1] + int(length))
        for local_chunk in range((int(length) + 31) // 32):
            chunks.extend((sequence, local_chunk))
    return cu, chunks


def run_npu(spec: dict[str, Any], tensors: tuple):
    inputs = build_inputs(spec, tensors, tensors[0].device)
    from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd

    saved_env = {
        name: os.environ.get(name)
        for name in ("TBE_PARALLEL_COMPILE_ENABLE", "PARALLEL_COMPILE")
    }
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    try:
        cu, chunks = _metadata(inputs["seqlens"])
        if inputs["sequence_major"]:
            g, beta = inputs["g"], inputs["beta"]
        else:
            g = inputs["g"].transpose(1, 2).contiguous()
            beta = inputs["beta"].transpose(1, 2).contiguous()
        outputs = npu_chunk_gated_delta_rule_fwd(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            g,
            beta,
            initial_state=inputs["initial_state"],
            output_final_state=bool(spec.get("output_final_state", True)),
            chunk_size=32,
            cu_seqlens=cu,
            chunk_indices=chunks,
            scale=inputs["scale"],
            use_exp2=False,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=inputs["allow_neg_eigval"],
            disable_recompute=inputs["disable_recompute"],
            return_intermediate_states=inputs["out_h"],
            state_v_first=inputs["state_v_first"],
            layout=inputs["layout"],
        )
        torch.npu.synchronize()
        _expect_public(spec, outputs)
        return _pack(spec, *outputs)
    finally:
        for name, previous in saved_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous



def _tensors(input_data: InputDataset) -> tuple:
    values = input_data.kwargs
    return values["q"], values["k"], values["v"], values["g"], values["beta"]


@register("executor_chunk_gated_delta_rule_fwd_chunk32")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(getattr(task_result, "is_benchmark_task", False))
        self.accuracy_standard = _accuracy_standard(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _spec_from_inputs(input_data)
        spec["case_id"] = getattr(self.task_result.case_config, "id", "?")
        tensors = _tensors(input_data)
        case_id = spec["case_id"]
        export = (
            f"ga={int(bool(spec.get('disable_recompute')))} "
            f"h={int(bool(spec.get('out_h')))}"
        )
        if self.device in {"npu", "pyaclnn"}:
            print(
                f"[chunk32-atk] case={case_id} role=dut target=npu_chunk32 "
                f"standard={self.accuracy_standard} {export}",
                flush=True,
            )
            outputs = run_npu(spec, tensors)
        elif self.device == "cpu":
            if self.accuracy_standard != "mixed_tolerance_bm":
                raise RuntimeError(
                    f"{OP_NAME} does not know accuracy standard {self.accuracy_standard!r}"
                )
            print(
                f"[chunk32-atk] case={case_id} role=golden target=cpu_emu_chunk32 {export}",
                flush=True,
            )
            outputs = run_cpu(spec, tensors)
        else:
            raise RuntimeError(
                f"{OP_NAME} supports the NPU DUT and the CPU golden, got device {self.device!r}"
            )
        return _finite_tuple(outputs)
