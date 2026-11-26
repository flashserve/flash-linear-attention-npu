"""CPU reference from FLAN 05b530df (tests/atk/chunk_gdn_bwd_intra/executor_chunk_gdn_bwd_intra.py)."""

from typing import Any

import torch

from _ascendc_common_executor import _chunks


def _work_chunks(inputs: dict[str, Any]):
    """Yield the physical BNSD span for every fixed or varlen chunk."""

    q = inputs["q"]
    chunk_size = int(inputs["chunk_size"])
    cu_seqlens = inputs["cu_seqlens"]
    if cu_seqlens is None:
        for batch in range(q.shape[0]):
            for start, end in _chunks(q.shape[2], chunk_size):
                yield batch, start, end
        return
    for seq, local_chunk in zip(
        inputs["chunk_indices"][::2], inputs["chunk_indices"][1::2]
    ):
        start = cu_seqlens[seq] + local_chunk * chunk_size
        end = min(start + chunk_size, cu_seqlens[seq + 1])
        if start < end:
            yield 0, start, end


def _compute_intermediates(inputs: dict[str, Any]):
    q, k, a = inputs["q"], inputs["k"], inputs["A"]
    g, beta = inputs["g"], inputs["beta"]
    B, HK, T, _ = q.shape
    HV = inputs["v"].shape[1]
    chunk_size = int(inputs["chunk_size"])
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    outputs = [
        torch.zeros((B, HV, T, chunk_size), dtype=calc, device=q.device)
        for _ in range(3)
    ]
    gate = torch.exp2 if inputs["use_exp2"] else torch.exp
    group = HV // HK
    for b, start, end in _work_chunks(inputs):
        for hv in range(HV):
            hk = hv // group
            length = end - start
            g_chunk = g[b, hv, start:end].to(calc)
            beta_chunk = beta[b, hv, start:end].to(calc)
            a_chunk = a[b, hv, start:end, :length].to(calc)
            bg = beta_chunk * gate(g_chunk)
            outputs[0][b, hv, start:end, :length] = a_chunk * bg.unsqueeze(0)
            outputs[1][b, hv, start:end, :length] = (
                a_chunk * beta_chunk.unsqueeze(0)
            )

            score = k[b, hk, start:end].to(calc) @ q[b, hk, start:end].to(calc).t()
            delta = g_chunk.unsqueeze(0) - g_chunk.unsqueeze(1)
            valid = torch.triu(
                torch.ones((length, length), dtype=torch.bool, device=q.device)
            )
            gate_delta = torch.zeros_like(delta)
            gate_delta[valid] = gate(delta[valid])
            outputs[2][b, hv, start:end, :length] = (
                float(inputs["scale"]) * score * gate_delta
            )
    return tuple(outputs)


def _cpu_ref(inputs: dict[str, Any]):
    q, k, v, d_o = inputs["q"], inputs["k"], inputs["v"], inputs["d_o"]
    B, HK, T, _ = q.shape
    HV = v.shape[1]
    chunk_size = int(inputs["chunk_size"])
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    main_dtype = inputs["main_dtype"]
    a_bg_all, a_beta_all, d_all = _compute_intermediates(inputs)
    outputs = [torch.zeros_like(v, dtype=calc) for _ in range(3)]
    group = HV // HK
    for b, start, end in _work_chunks(inputs):
        for hv in range(HV):
            hk = hv // group
            length = end - start
            # Stage 1 writes main dtype workspace before the three Cube MMADs.
            a_bg = a_bg_all[b, hv, start:end, :length].to(main_dtype).to(calc)
            a_beta = a_beta_all[b, hv, start:end, :length].to(main_dtype).to(calc)
            d_local = d_all[b, hv, start:end, :length].to(main_dtype).to(calc)
            outputs[0][b, hv, start:end] = (
                a_bg @ k[b, hk, start:end].to(calc)
            )
            outputs[1][b, hv, start:end] = (
                a_beta @ v[b, hv, start:end].to(calc)
            )
            outputs[2][b, hv, start:end] = (
                d_local @ d_o[b, hv, start:end].to(calc)
            )
    return tuple(outputs)
