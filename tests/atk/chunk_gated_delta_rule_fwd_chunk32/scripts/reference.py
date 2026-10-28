"""CPU golden for the migrated chunk32 path of chunk_gated_delta_rule_fwd.

The expected value is the three forward operators' own goldens, chained:

1. ``chunk_gated_delta_rule_fwd_prepare`` — ``cpu_gdn_fwd_l2norm_to_recompute``
2. ``chunk_fwd_h`` — ``executor_chunk_fwd_h._reference``
3. ``chunk_fwd_o`` — the same formula as ``executor_chunk_fwd_o._chunk_fwd_o_ref``,
   extended to packed sequences the way ``test_npu_fwd_o_generalization._reference``
   indexes chunks.

Flags are the natural-exp path those goldens already implement
(``use_exp2=False``, ``torch.exp``). L2 and sigmoid stay on, ``allow_neg_eigval``
stays off. Chunk size is 64, the size those three operators' goldens use.
``w`` and ``u`` are cast to BF16 before ``chunk_fwd_h``, which is the dtype that
golden consumes. The 32-kernel's extra BF16 round-trips of ``D`` and ``u-P``
are not part of this golden. Mixed tolerance accounts for those intentional
low-precision round trips.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch

_ATK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ATK / "chunk_gated_delta_rule_fwd_prepare" / "scripts"))
sys.path.insert(0, str(_ATK / "chunk_fwd_h"))

from cpu_golden import cpu_gdn_fwd_l2norm_to_recompute  # noqa: E402
from executor_chunk_fwd_h import PreparedInputs, _reference  # noqa: E402


GOLDEN_CHUNK = 64
K_DIM = 128


def _cu(seqlens: Optional[list[int]]) -> Optional[torch.Tensor]:
    if seqlens is None:
        return None
    values = [0]
    for length in seqlens:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int64)


def forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    seqlens: Optional[list[int]],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(o, final_state)`` in FP32.

    ``q/k/v/beta`` are the raw BF16 inputs. ``g`` is the raw FP32 gate, not a
    cumsum. ``initial_state`` is ``None``, FP32, or BF16, shaped
    ``[N, Hv, K, V]``.
    """

    cu = _cu(seqlens)
    prepared = cpu_gdn_fwd_l2norm_to_recompute(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        g.cpu(),
        beta.cpu(),
        chunk_size=GOLDEN_CHUNK,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        cu_seqlens=cu,
        layout="bnsd",
        use_exp2=False,
    )
    # chunk_fwd_h's golden loads k/w/u as BF16 and only then widens them.
    k_hat = prepared.k.to(torch.bfloat16).contiguous()
    w = prepared.w.to(torch.bfloat16).contiguous()
    u = prepared.u.to(torch.bfloat16).contiguous()
    g_cumsum = prepared.g.float().contiguous()
    inputs = PreparedInputs(
        k=k_hat,
        w=w,
        u=u,
        g=g_cumsum,
        gk=None,
        initial_state=None if initial_state is None else initial_state.detach().cpu().contiguous(),
        cu_seqlens=None if cu is None else tuple(int(item) for item in cu.tolist()),
        chunk_indices=None,
        seqlens=seqlens,
    )
    h, v_new, final_state = _reference(
        inputs,
        output_final_state=True,
        use_exp2=False,
        state_v_first=False,
    )
    if final_state is None:
        raise RuntimeError("chunk_fwd_h golden did not return final_state")
    o = _chunk_fwd_o(prepared.q.to(torch.bfloat16), k_hat, v_new, h, g_cumsum, seqlens, scale)
    return o.float().contiguous(), final_state.float().contiguous()


def _chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    seqlens: Optional[list[int]],
    scale: float,
) -> torch.Tensor:
    """``chunk_fwd_o`` golden: ``scale * (q_exp @ H + tril(QK * exp(gi-gj)) @ v_new)``.

    Same placement of ``scale`` and ``exp`` as ``executor_chunk_fwd_o._chunk_fwd_o_ref``.
    Packed sequences use the chunk slot order of ``chunk_fwd_h`` (sequence-major).
    """

    batch, _hk, _tokens, _kdim = q.shape
    v_heads = v_new.shape[1]
    value_dim = v_new.shape[-1]
    total_tokens = q.shape[2]
    calc = torch.float64
    out = torch.zeros((batch, v_heads, total_tokens, value_dim), dtype=calc)
    group = max(v_heads // q.shape[1], 1)
    if seqlens is None:
        spans = [(batch_id, 0, total_tokens, 0) for batch_id in range(batch)]
    else:
        spans = []
        cursor = 0
        base = 0
        for length in seqlens:
            length = int(length)
            spans.append((0, cursor, cursor + length, base))
            base += (length + GOLDEN_CHUNK - 1) // GOLDEN_CHUNK
            cursor += length
    for batch_index, begin, end, state_base in spans:
        chunks = (end - begin + GOLDEN_CHUNK - 1) // GOLDEN_CHUNK
        for head in range(v_heads):
            key_head = head // group
            for chunk in range(chunks):
                token_begin = begin + chunk * GOLDEN_CHUNK
                token_end = min(token_begin + GOLDEN_CHUNK, end)
                q_chunk = q[batch_index, key_head, token_begin:token_end].to(calc)
                k_chunk = k[batch_index, key_head, token_begin:token_end].to(calc)
                v_chunk = v_new[batch_index, head, token_begin:token_end].to(calc)
                g_chunk = g[batch_index, head, token_begin:token_end].to(calc)
                state = h[batch_index, state_base + chunk, head].to(calc)
                local = q_chunk @ k_chunk.transpose(0, 1)
                gate = torch.exp(g_chunk[:, None] - g_chunk[None, :])
                mask = torch.tril(torch.ones_like(local))
                intra = (local * gate * mask) @ v_chunk
                inter = (q_chunk * torch.exp(g_chunk)[:, None]) @ state
                out[batch_index, head, token_begin:token_end] = (intra + inter) * float(scale)
    return out


def default_scale() -> float:
    return 1.0 / math.sqrt(K_DIM)
