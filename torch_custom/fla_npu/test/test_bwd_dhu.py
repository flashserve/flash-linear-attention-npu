"""chunk_gated_delta_rule_bwd_dhu CPU 标杆（fp64 / npu / fp32）。"""
from __future__ import annotations

import importlib.util
import os
from typing import List, Optional, Tuple

import torch

_LN2 = 0.69314718055994530942

_SCRIPT_DIR = os.path.dirname(__file__)
_PTA_TEST_CANDIDATES = [
    os.path.abspath(
        os.path.join(
            _SCRIPT_DIR,
            "../../../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/test_chunk_gated_delta_rule_bwd_dhu.py",
        )
    ),
    os.path.abspath(
        os.path.join(
            _SCRIPT_DIR,
            "../../../examples/fast_kernel_launch_example/tests/chunk_gated_delta_rule_bwd_dhu/test_chunk_gated_delta_rule_bwd_dhu.py",
        )
    ),
]
_PTA_TEST = next((path for path in _PTA_TEST_CANDIDATES if os.path.exists(path)), _PTA_TEST_CANDIDATES[0])
try:
    _spec = importlib.util.spec_from_file_location("pta_bwd_dhu", _PTA_TEST)
    _pta = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_pta)

    create_bwd_dhu_random_inputs = _pta.create_bwd_dhu_random_inputs
    create_gate_g = _pta.create_gate_g
    effective_scale = _pta.effective_scale
    generate_cu_seqlens = _pta.generate_cu_seqlens
    prepare_chunk_indices = _pta.prepare_chunk_indices
    scale_for_compute_dtype = _pta.scale_for_compute_dtype
except (FileNotFoundError, ModuleNotFoundError):
    _LOW_PRECISION_INPUT_HALF_RANGE_QK = 6.5e-3
    _LOW_PRECISION_INPUT_HALF_RANGE_WO = 6.5e-3
    _LOW_PRECISION_INPUT_HALF_RANGE_DV = 9e-3
    _LOW_PRECISION_SCALE_FACTOR = 0.92

    def prepare_chunk_indices(cu_seqlens: List[int], chunk_size: int = 64) -> List[int]:
        chunk_indices = []
        for seq_idx in range(len(cu_seqlens) - 1):
            seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
            chunk_num = (seq_len + chunk_size - 1) // chunk_size
            for chunk_idx in range(chunk_num):
                chunk_indices.append(seq_idx)
                chunk_indices.append(chunk_idx)
        return chunk_indices

    def create_gate_g(B: int, Hv: int, T: int, gtype, narrow: bool = False):
        if narrow:
            lo, hi = -1e-2, -1e-6
        else:
            lo, hi = -5e-2, -5e-5
        span = hi - lo
        margin = max(span * 1e-7, 1e-12)
        g_t = torch.linspace(float(hi) - margin, float(lo) + margin, T, dtype=torch.float64)
        return g_t.unsqueeze(0).unsqueeze(0).expand(B, Hv, T).contiguous().to(gtype)

    def generate_cu_seqlens(cu_seqlens_len: int, total_length: int, seg_min: int = 64,
                            seg_max: int = 128) -> List[int]:
        batchsize = cu_seqlens_len - 1
        if batchsize <= 0:
            return [0, total_length]

        lengths = [
            (total_length * (i + 1)) // batchsize - (total_length * i) // batchsize
            for i in range(batchsize)
        ]
        for i in range(batchsize):
            lengths[i] = max(seg_min, min(seg_max, lengths[i]))

        diff = total_length - sum(lengths)
        while diff > 0:
            cand = [i for i in range(batchsize) if lengths[i] < seg_max]
            if not cand:
                break
            i = min(cand, key=lambda j: lengths[j])
            lengths[i] += 1
            diff -= 1
        while diff < 0:
            cand = [i for i in range(batchsize) if lengths[i] > seg_min]
            if not cand:
                break
            i = max(cand, key=lambda j: lengths[j])
            lengths[i] -= 1
            diff += 1

        sorted_l = sorted(lengths)
        seq_lengths: List[int] = []
        i, j = 0, len(sorted_l) - 1
        while i <= j:
            if i == j:
                seq_lengths.append(sorted_l[i])
            else:
                seq_lengths.append(sorted_l[i])
                seq_lengths.append(sorted_l[j])
            i += 1
            j -= 1

        cu_seqlens = [0]
        for seq_len in seq_lengths:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
        return cu_seqlens

    def _rand_symmetric_uniform(shape, dtype: torch.dtype, half_range: float) -> torch.Tensor:
        x = torch.rand(shape, dtype=torch.float32, device="cpu")
        x = (x * 2.0 - 1.0) * float(half_range)
        return x.to(dtype=dtype)

    def create_bwd_dhu_random_inputs(
        B: int, Hk: int, Hv: int, T: int, K: int, V: int, ktype: torch.dtype, gtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        low = ktype in (torch.float16, torch.bfloat16)
        if low:
            hr_qk = _LOW_PRECISION_INPUT_HALF_RANGE_QK
            hr_wo = _LOW_PRECISION_INPUT_HALF_RANGE_WO
            hr_dv = _LOW_PRECISION_INPUT_HALF_RANGE_DV
            narrow_g = True
        else:
            hr_qk = 2e-2
            hr_wo = 2e-2
            hr_dv = 3e-2
            narrow_g = False

        q = _rand_symmetric_uniform((B, Hk, T, K), ktype, half_range=hr_qk)
        k = _rand_symmetric_uniform((B, Hk, T, K), ktype, half_range=hr_qk)
        w = _rand_symmetric_uniform((B, Hv, T, K), ktype, half_range=hr_qk)
        d_o = _rand_symmetric_uniform((B, Hv, T, V), ktype, half_range=hr_wo)
        dv = _rand_symmetric_uniform((B, Hv, T, V), ktype, half_range=hr_dv)
        g = create_gate_g(B, Hv, T, gtype, narrow=narrow_g)
        return q, k, w, d_o, dv, g

    def effective_scale(scale: float, K: int) -> float:
        return float(min(scale, 1.0 / (float(K) ** 0.5)))

    def scale_for_compute_dtype(scale: float, ktype: torch.dtype) -> float:
        if ktype in (torch.float16, torch.bfloat16):
            return float(scale * _LOW_PRECISION_SCALE_FACTOR)
        return float(scale)


def _round_elem(x: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    if elem_dtype == torch.float32:
        return x.to(torch.float32)
    return x.to(elem_dtype).to(torch.float32)


def _matmul_npu_aligned(a: torch.Tensor, b: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    a = _round_elem(a, elem_dtype).contiguous()
    b = _round_elem(b, elem_dtype).contiguous()
    if a.device.type == "npu" and a.dim() > 3:
        batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        if a.shape[:-2] != batch_shape:
            a = a.expand(*batch_shape, *a.shape[-2:]).contiguous()
        if b.shape[:-2] != batch_shape:
            b = b.expand(*batch_shape, *b.shape[-2:]).contiguous()

        m = a.shape[-2]
        n = b.shape[-1]
        out = a.reshape(-1, m, a.shape[-1]) @ b.reshape(-1, b.shape[-2], n)
        return out.reshape(*batch_shape, m, n)
    return a @ b


def _gate_exp2(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x * _LN2)


def _gate_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


def chunk_gated_delta_rule_bwd_dhu_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens: Optional[List[int]] = None,
    chunk_indices: Optional[List[int]] = None,
    g: Optional[torch.Tensor] = None,
    gK: Optional[torch.Tensor] = None,
    h0: Optional[torch.Tensor] = None,
    dht: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    chunk_size: int = 64,
    golden_mode: str = "fp32",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """GVA 形状 CPU 标杆。golden_mode: fp64 / npu / fp32。"""
    del dht
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
        if gK is not None:
            gK = gK.float()
    else:
        q = q.to(compute_dtype)
        k = k.to(compute_dtype)
        w = w.to(compute_dtype)
        do = do.to(compute_dtype)
        dv = dv.to(compute_dtype)
        if g is not None:
            g = g.to(compute_dtype)
        if gK is not None:
            gK = gK.to(compute_dtype)

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
    dh0 = torch.zeros_like(dh) if h0 is not None else None
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

            b_dv = _store(_mm(k_blk, b_dh))
            if g is not None:
                bg_last = g[:, :, global_last_idx].to(torch.float32)
                b_g = g[:, :, gs:ge].to(torch.float32)
                gate_factor = _gate_exp(bg_last.unsqueeze(-1) - b_g).unsqueeze(-1)
                m_t = torch.arange(block_size_t, device=device, dtype=torch.float32) < float(block_size_t)
                b_dv = b_dv * gate_factor * m_t.view(1, 1, block_size_t, 1)

            b_dv = b_dv + b_dv_existing
            dv2[:, :, gs:ge, :] = _store(b_dv).to(dtype_)

            b_q_t = q_blk.transpose(-1, -2)
            b_w_t = w_blk.transpose(-1, -2)
            if g is not None:
                bg_last_exp = _gate_exp(bg_last)
                b_g_exp = _gate_exp(b_g)
                b_dh_for_update = b_dh * bg_last_exp.unsqueeze(-1).unsqueeze(-1)
                b_q_gated = b_q_t * b_g_exp.unsqueeze(-2)
            elif gK is not None:
                bgk_last_exp = _gate_exp2(gK[:, :, global_last_idx, :].to(torch.float32))
                b_dh_for_update = b_dh * bgk_last_exp.unsqueeze(-1)
                b_q_gated = b_q_t
            else:
                b_dh_for_update = b_dh.clone()
                b_q_gated = b_q_t

            term1 = _store(_mm(b_q_gated, b_do)) * scale_f
            term2 = _store(_mm(b_w_t, b_dv))
            b_dh = _store(b_dh_for_update + term1 - term2)

        if dh0 is not None:
            dh0[:, :, 0, :, :] = b_dh
    else:
        hq = torch.arange(Hv, device=device, dtype=torch.long) // hv_per_hk
        num_tokens = len(cu_seqlens) - 1
        b_dh_buffers = torch.zeros(B, Hv, num_tokens, K, V, device=device, dtype=compute_dtype)
        for i_t in range(NT - 1, -1, -1):
            info = chunk_info[i_t]
            i_n = info["i_n"]
            gs, ge = info["global_start_t"], info["global_end_t"]
            block_size_t = info["block_size_t"]
            b_dh = b_dh_buffers[:, :, i_n, :, :]
            dh[:, :, i_t, :, :] = b_dh

            last_idx = min((info["block_idx_in_token"] + 1) * BT, info["token_length"]) - 1
            global_last_idx = info["bos"] + last_idx

            k_blk = _to_compute(k[:, :, gs:ge, :].index_select(1, hq))
            q_blk = _to_compute(q[:, :, gs:ge, :].index_select(1, hq))
            w_blk = _to_compute(w[:, :, gs:ge, :])
            b_do = _to_compute(do[:, :, gs:ge, :])
            b_dv_existing = _to_compute(dv[:, :, gs:ge, :])

            b_dv = _store(_mm(k_blk, b_dh))
            if g is not None:
                bg_last = g[:, :, global_last_idx].to(torch.float32)
                b_g = g[:, :, gs:ge].to(torch.float32)
                gate_factor = _gate_exp(bg_last.unsqueeze(-1) - b_g).unsqueeze(-1)
                m_t = torch.arange(block_size_t, device=device, dtype=torch.float32) < float(block_size_t)
                b_dv = b_dv * gate_factor * m_t.view(1, 1, block_size_t, 1)

            b_dv = b_dv + b_dv_existing
            dv2[:, :, gs:ge, :] = _store(b_dv).to(dtype_)

            b_q_t = q_blk.transpose(-1, -2)
            b_w_t = w_blk.transpose(-1, -2)
            if g is not None:
                bg_last_exp = _gate_exp(bg_last)
                b_g_exp = _gate_exp(b_g)
                b_dh_for_update = b_dh * bg_last_exp.unsqueeze(-1).unsqueeze(-1)
                b_q_gated = b_q_t * b_g_exp.unsqueeze(-2)
            elif gK is not None:
                bgk_last_exp = _gate_exp2(gK[:, :, global_last_idx, :].to(torch.float32))
                b_dh_for_update = b_dh * bgk_last_exp.unsqueeze(-1)
                b_q_gated = b_q_t
            else:
                b_dh_for_update = b_dh.clone()
                b_q_gated = b_q_t

            term1 = _store(_mm(b_q_gated, b_do)) * scale_f
            term2 = _store(_mm(b_w_t, b_dv))
            b_dh_buffers[:, :, i_n, :, :] = _store(b_dh_for_update + term1 - term2)

        if dh0 is not None:
            for info in chunk_info:
                if info["block_idx_in_token"] == 0:
                    dh0[:, :, info["i_t"], :, :] = b_dh_buffers[:, :, info["i_n"], :, :]

    return dh, dh0, dv2
