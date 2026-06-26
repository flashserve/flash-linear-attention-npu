"""chunk_gated_delta_rule_bwd_dhu CPU 标杆（fp64 / npu / fp32）。"""
from __future__ import annotations

import importlib.util
import os
from typing import List, Optional, Tuple

import torch

_PTA_TEST = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/test_chunk_gated_delta_rule_bwd_dhu.py",
    )
)
_spec = importlib.util.spec_from_file_location("pta_bwd_dhu", _PTA_TEST)
_pta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pta)

create_bwd_dhu_random_inputs = _pta.create_bwd_dhu_random_inputs
create_gate_g = _pta.create_gate_g
effective_scale = _pta.effective_scale
generate_cu_seqlens = _pta.generate_cu_seqlens
prepare_chunk_indices = _pta.prepare_chunk_indices
scale_for_compute_dtype = _pta.scale_for_compute_dtype


def _round_elem(x: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    if elem_dtype == torch.float32:
        return x.to(torch.float32)
    return x.to(elem_dtype).to(torch.float32)


def _matmul_npu_aligned(a: torch.Tensor, b: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    return _round_elem(a, elem_dtype) @ _round_elem(b, elem_dtype)


def chunk_gated_delta_rule_bwd_dhu_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens: Optional[List[int]] = None,
    chunk_indices: Optional[List[int]] = None,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    chunk_size: int = 64,
    golden_mode: str = "fp32",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """GVA 形状 CPU 标杆。golden_mode: fp64 / npu / fp32。"""
    dtype_ = q.dtype
    if golden_mode == "fp64":
        compute_dtype = torch.float64
        elem_dtype = None
    elif golden_mode == "npu":
        compute_dtype = torch.float32
        elem_dtype = dtype_
    elif golden_mode == "fp32":
        compute_dtype = torch.float32
        elem_dtype = None
    else:
        raise ValueError(f"unsupported golden_mode={golden_mode}")

    device = q.device
    B, Hk, T, K = q.shape
    Hv = do.shape[1]
    V = do.shape[-1]
    BT = chunk_size

    if Hk <= 0 or Hv % Hk != 0:
        raise ValueError(f"GVA: Hv % Hk == 0 required, Hk={Hk}, Hv={Hv}")

    hv_per_hk = Hv // Hk
    if cu_seqlens is not None:
        seq_total = cu_seqlens[-1]
        if seq_total > T:
            raise ValueError(f"cu_seqlens[-1]={seq_total} > T={T}")
        NT = len(chunk_indices) // 2
    else:
        NT = (T + BT - 1) // BT

    if scale is None:
        scale = 1.0
    scale_f = float(scale)

    if golden_mode == "npu":
        q = q.to(dtype_)
        k = k.to(dtype_)
        w = w.to(dtype_)
        do = do.to(dtype_)
        dv = dv.to(dtype_)
        if g is not None:
            g = g.float()
    else:
        q = q.to(compute_dtype)
        k = k.to(compute_dtype)
        w = w.to(compute_dtype)
        do = do.to(compute_dtype)
        dv = dv.to(compute_dtype)
        if g is not None:
            g = g.to(compute_dtype)

    def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if elem_dtype is None:
            return a @ b
        return _matmul_npu_aligned(a, b, elem_dtype)

    def _store(x: torch.Tensor) -> torch.Tensor:
        if elem_dtype is None:
            return x
        return _round_elem(x, elem_dtype)

    def _to_compute(x: torch.Tensor) -> torch.Tensor:
        if elem_dtype is None:
            return x.to(compute_dtype)
        return _round_elem(x, elem_dtype)

    chunk_info = []
    for i_t in range(NT):
        if cu_seqlens is not None:
            i_n = chunk_indices[i_t * 2]
            block_idx_in_token = chunk_indices[i_t * 2 + 1]
            bos = cu_seqlens[i_n]
            token_length = cu_seqlens[i_n + 1] - bos
        else:
            i_n = 0
            block_idx_in_token = i_t
            bos = 0
            token_length = T
        start_t = block_idx_in_token * BT
        end_t = min((block_idx_in_token + 1) * BT, token_length)
        global_start_t = bos + start_t
        global_end_t = bos + end_t
        chunk_info.append({
            "i_t": i_t,
            "i_n": i_n,
            "block_idx_in_token": block_idx_in_token,
            "bos": bos,
            "token_length": token_length,
            "block_size_t": end_t - start_t,
            "global_start_t": global_start_t,
            "global_end_t": global_end_t,
        })

    dh = torch.zeros(B, Hv, NT, K, V, device=device, dtype=compute_dtype)
    dv2 = dv.clone() if cu_seqlens is not None else torch.zeros(B, Hv, T, V, device=device, dtype=dtype_)

    if cu_seqlens is None:
        hq = torch.arange(Hv, device=device, dtype=torch.long) // hv_per_hk
        b_dh = torch.zeros(B, Hv, K, V, device=device, dtype=compute_dtype)
        for i_t in range(NT - 1, -1, -1):
            info = chunk_info[i_t]
            gs, ge = info["global_start_t"], info["global_end_t"]
            block_size_t = info["block_size_t"]
            dh[:, :, i_t, :, :] = b_dh

            last_idx = min((info["block_idx_in_token"] + 1) * BT, info["token_length"]) - 1
            global_last_idx = info["bos"] + last_idx

            k_blk = _to_compute(k[:, :, gs:ge, :].index_select(1, hq))
            q_blk = _to_compute(q[:, :, gs:ge, :].index_select(1, hq))
            w_blk = _to_compute(w[:, :, gs:ge, :])
            b_do = _to_compute(do[:, :, gs:ge, :])
            b_dv_existing = _to_compute(dv[:, :, gs:ge, :])

            b_dv = _mm(k_blk, b_dh)
            if g is not None:
                bg_last = g[:, :, global_last_idx].to(torch.float32)
                b_g = g[:, :, gs:ge].to(torch.float32)
                gate_factor = torch.exp(bg_last.unsqueeze(-1) - b_g).unsqueeze(-1)
                m_t = torch.arange(block_size_t, device=device, dtype=torch.float32) < float(block_size_t)
                b_dv = b_dv * gate_factor * m_t.view(1, 1, block_size_t, 1)

            b_dv = b_dv + b_dv_existing
            dv2[:, :, gs:ge, :] = _store(b_dv).to(dtype_)

            b_q_t = q_blk.transpose(-1, -2)
            b_w_t = w_blk.transpose(-1, -2)
            if g is not None:
                bg_last_exp = torch.exp(bg_last)
                b_g_exp = torch.exp(b_g)
                b_dh_for_update = b_dh * bg_last_exp.unsqueeze(-1).unsqueeze(-1)
                b_q_gated = b_q_t * b_g_exp.unsqueeze(-2)
            else:
                b_dh_for_update = b_dh.clone()
                b_q_gated = b_q_t

            term1 = _mm(b_q_gated, b_do) * scale_f
            term2 = _mm(b_w_t, b_dv)
            b_dh = _store(b_dh_for_update + term1 - term2)
    else:
        for b in range(B):
            for i_h in range(Hv):
                hq = i_h // hv_per_hk
                num_tokens = len(cu_seqlens) - 1
                b_dh_buffers = {i_n: torch.zeros(K, V, device=device, dtype=compute_dtype) for i_n in range(num_tokens)}
                for i_t in range(NT - 1, -1, -1):
                    info = chunk_info[i_t]
                    i_n = info["i_n"]
                    b_dh = b_dh_buffers[i_n]
                    dh[b, i_h, i_t] = b_dh
                    gs, ge = info["global_start_t"], info["global_end_t"]
                    block_size_t = info["block_size_t"]
                    last_idx = min((info["block_idx_in_token"] + 1) * BT, info["token_length"]) - 1
                    global_last_idx = info["bos"] + last_idx

                    b_do = _to_compute(do[b, i_h, gs:ge, :])
                    b_dv_existing = _to_compute(dv[b, i_h, gs:ge, :])
                    b_k = _to_compute(k[b, hq, gs:ge, :])
                    b_dv = _mm(b_k, b_dh)

                    if g is not None:
                        bg_last = g[b, i_h, global_last_idx].to(torch.float32)
                        b_g = g[b, i_h, gs:ge].to(torch.float32)
                        bg_last_exp = torch.exp(bg_last)
                        b_g_exp = torch.exp(b_g)
                        m_t = torch.arange(block_size_t, device=device) < block_size_t
                        b_dv = b_dv * torch.exp(bg_last - b_g).unsqueeze(-1) * m_t.unsqueeze(-1).float()
                    else:
                        bg_last_exp = b_g_exp = None

                    b_dv = b_dv + b_dv_existing
                    dv2[b, i_h, gs:ge, :] = _store(b_dv).to(dtype_)

                    b_q = _to_compute(q[b, hq, gs:ge, :])
                    b_w = _to_compute(w[b, i_h, gs:ge, :])
                    b_q_t = b_q.transpose(0, 1)
                    b_w_t = b_w.transpose(0, 1)
                    b_dh_for_update = b_dh.clone()
                    if g is not None:
                        b_dh_for_update = b_dh_for_update * bg_last_exp
                        b_q_gated = b_q_t * b_g_exp.unsqueeze(0)
                    else:
                        b_q_gated = b_q_t

                    term1 = _mm(b_q_gated, b_do) * scale_f
                    term2 = _mm(b_w_t, b_dv)
                    b_dh_buffers[i_n] = _store(b_dh_for_update + term1 - term2)

    return dh, None, dv2
