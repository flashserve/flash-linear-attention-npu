import torch
import torch_npu
from typing import Optional
import math
import random
import os
import numpy as np
from ct import dual,viz
import fla_npu

GLOBAL_SEED = 42
_PRIMARY_GOLDEN_DIR = "/workspace/clx/work_data/ChunkBwdDvLocal"
_FALLBACK_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "work_data", "ChunkBwdDvLocal")
GOLDEN_DIR = _PRIMARY_GOLDEN_DIR if os.path.isdir(os.path.dirname(_PRIMARY_GOLDEN_DIR)) else _FALLBACK_GOLDEN_DIR

def set_global_seed(seed=GLOBAL_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def cdiv(a: torch.LongTensor, b: int):
    return (a + b - 1) // b


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def generate_cu_seqlens(
    cu_seqlens_len: int,
    total_length: int,
    device: Optional[torch.device] = None
) -> torch.LongTensor:
    batchsize = cu_seqlens_len - 1
    remaining = total_length
    seq_lengths = []
    for i in range(batchsize - 1):
        min_len = 1
        max_len = remaining - (batchsize - 1 - i)
        if max_len < min_len:
            max_len = min_len
        seq_len = random.randint(min_len, max_len)
        seq_lengths.append(seq_len)
        remaining -= seq_len
    seq_lengths.append(remaining)
    
    cu_seqlens = [0]
    for seq_len in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
    
    tensor = torch.tensor(cu_seqlens, dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def dtype_short(dt):
    mapping = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
        torch.float64: "fp64",
    }
    return mapping.get(dt, str(dt))

def make_case_name(mode, B, H_qk, T, K, V, chunk_size, scale, ktype, gtype, h_ratio=1, cu_seqlens_len=None):
    k_s = dtype_short(ktype)
    g_s = dtype_short(gtype)
    name = f"{mode}_B{B}_Hqk{H_qk}_T{T}_K{K}_V{V}_cs{chunk_size}_s{scale}_{k_s}_{g_s}_hr{h_ratio}"
    if cu_seqlens_len is not None:
        name += f"_csl{cu_seqlens_len}"
    return name

def save_golden(case_name, **kwargs):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = os.path.join(GOLDEN_DIR, f"{case_name}.pt")
    torch.save(kwargs, path)
    print(f"[Golden] saved to {path}")

def load_golden(case_name):
    path = os.path.join(GOLDEN_DIR, f"{case_name}.pt")
    if os.path.exists(path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[Golden] loaded from {path}")
        return data
    return None

def stop_on_case_failure(case_name, result):
    success = result.get("success") if isinstance(result, dict) else getattr(result, "success", None)
    if success is not True:
        raise AssertionError(f"[{case_name}] failed, stop on first failed case")

def create_tensor(shape, dtype=torch.float16):
    return torch.rand(shape, dtype=dtype)


def chunk_bwd_dv_local_fix(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
    h_ratio: int = 1
) -> torch.Tensor:
    B, H_qk, T, K = k.shape
    H_do = do.shape[1]
    qkv_type = q.dtype
    V = do.shape[3]
    assert H_do == H_qk * h_ratio, f"H_do ({H_do}) must equal H_qk * h_ratio ({H_qk} * {h_ratio} = {H_qk * h_ratio})"
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if cu_seqlens is not None:
        batch_idx = 0
    BT = min(chunk_size, max(16, 2 ** math.ceil(math.log2(T))))
    chunk_per_T = (T + chunk_size - 1) // chunk_size
    NT = chunk_per_T * B
    dv = torch.zeros_like(do).to(torch.float32)
    g_t = g
    for chunk_idx in range(NT):
        i_n = chunk_idx // chunk_per_T
        batch_idx = i_n
        i_t = chunk_idx % chunk_per_T

        chunk_start_token = i_t * chunk_size
        chunk_end_token = min(chunk_start_token + chunk_size, T)
        chunk_len = chunk_end_token - chunk_start_token
        if chunk_len <= 0:
            continue
        for qk_head in range(H_qk):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float32)
            BK = 128
            BK = min(BK, K)
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, qk_head, chunk_start_token:chunk_start_token+chunk_len, i_k:k_end]
                q_normal = q[batch_idx, qk_head, chunk_start_token:chunk_start_token+chunk_len, i_k:k_end]
                b_q = q_normal.transpose(0, 1)
                if chunk_len == 1:
                    matmul_result = torch.sum(b_k * q_normal)
                    b_A[:chunk_len, :chunk_len] += matmul_result
                else:
                    b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q)
            for do_group in range(h_ratio):
                do_head = qk_head * h_ratio + do_group
                b_g = g_t[batch_idx, do_head, chunk_start_token:chunk_start_token+chunk_len].to(torch.float32)
                o_t = i_t * BT + torch.arange(0, BT)
                m_t = o_t < T
                o_t_col = o_t.unsqueeze(1)
                o_t_row = o_t.unsqueeze(0)
                pos_mask = o_t_col <= o_t_row
                m_t_col = m_t.unsqueeze(1)
                valid_mask = m_t_col & m_t
                m_A = pos_mask & valid_mask
                g_i = b_g.unsqueeze(1)
                g_j = b_g.unsqueeze(0)
                g_factor = torch.exp(g_j - g_i) * scale
                b_A_gated = torch.zeros_like(b_A)
                b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor
                b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated))
                b_A_masked = b_A_masked.to(qkv_type)
                BV = 128
                BV = min(BV, V)
                for i_v in range(0, V, BV):
                    v_end = min(i_v + BV, V)
                    v_width = v_end - i_v
                    b_do = do[batch_idx, do_head, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end]
                    b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do)
                    dv[batch_idx, do_head, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end] += b_dv
    return dv

def chunk_bwd_dv_local_fix_high_precision(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
    h_ratio: int = 1
) -> torch.Tensor:
    B, H_qk, T, K = k.shape
    H_do = do.shape[1]
    V = do.shape[3]
    assert H_do == H_qk * h_ratio, f"H_do ({H_do}) must equal H_qk * h_ratio ({H_qk} * {h_ratio} = {H_qk * h_ratio})"
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if cu_seqlens is not None:
        batch_idx = 0
    BT = min(chunk_size, max(16, 2 ** math.ceil(math.log2(T))))
    chunk_per_T = (T + chunk_size - 1) // chunk_size
    NT = chunk_per_T * B
    dv = torch.zeros_like(do).to(torch.float64)
    g_t = g
    for chunk_idx in range(NT):
        i_n = chunk_idx // chunk_per_T
        batch_idx = i_n
        i_t = chunk_idx % chunk_per_T

        chunk_start_token = i_t * chunk_size
        chunk_end_token = min(chunk_start_token + chunk_size, T)
        chunk_len = chunk_end_token - chunk_start_token
        if chunk_len <= 0:
            continue
        for qk_head in range(H_qk):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float64)
            BK = 128
            BK = min(BK, K)
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, qk_head, chunk_start_token:chunk_start_token+chunk_len, i_k:k_end].to(torch.float32)
                q_normal = q[batch_idx, qk_head, chunk_start_token:chunk_start_token+chunk_len, i_k:k_end].to(torch.float32)
                b_q = q_normal.transpose(0, 1)
                if chunk_len == 1:
                    matmul_result = torch.sum(b_k * q_normal)
                    b_A[:chunk_len, :chunk_len] += matmul_result
                else:
                    b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q)
            for do_group in range(h_ratio):
                do_head = qk_head * h_ratio + do_group
                b_g = g_t[batch_idx, do_head, chunk_start_token:chunk_start_token+chunk_len].to(torch.float64)
                o_t = i_t * BT + torch.arange(0, BT)
                m_t = o_t < T
                o_t_col = o_t.unsqueeze(1)
                o_t_row = o_t.unsqueeze(0)
                pos_mask = o_t_col <= o_t_row
                m_t_col = m_t.unsqueeze(1)
                valid_mask = m_t_col & m_t
                m_A = pos_mask & valid_mask
                g_i = b_g.unsqueeze(1)
                g_j = b_g.unsqueeze(0)
                g_factor = torch.exp(g_j - g_i) * scale
                b_A_gated = torch.zeros_like(b_A)
                b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor
                b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated))
                b_A_masked = b_A_masked.to(torch.float32)
                BV = 128
                BV = min(BV, V)
                for i_v in range(0, V, BV):
                    v_end = min(i_v + BV, V)
                    v_width = v_end - i_v
                    b_do = do[batch_idx, do_head, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end].to(torch.float32)
                    b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do)
                    dv[batch_idx, do_head, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end] += b_dv
    return dv


def chunk_bwd_dv_local_variable(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
    h_ratio: int = 1
) -> torch.Tensor:
    B, H_qk, T, K = k.shape
    H_do = do.shape[1]
    qkv_type = q.dtype
    V = do.shape[3]
    assert H_do == H_qk * h_ratio, f"H_do ({H_do}) must equal H_qk * h_ratio ({H_qk} * {h_ratio} = {H_qk * h_ratio})"
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if cu_seqlens is not None:
        batch_idx = 0
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_indices = chunk_indices.view(-1)
    BT = min(chunk_size, max(16, 2 ** math.ceil(math.log2(T))))
    NT = len(chunk_indices) // 2
    dv = torch.zeros_like(do).to(torch.float32)
    g_t = g
    for chunk_idx in range(NT):
        i_n = chunk_indices[chunk_idx * 2].item()
        i_t = chunk_indices[chunk_idx * 2 + 1].item()
        bos = cu_seqlens[i_n].item()
        eos = cu_seqlens[i_n + 1].item()
        T = eos - bos
        chunk_start_token = i_t * chunk_size
        chunk_end_token = min(chunk_start_token + chunk_size, T)
        chunk_len = chunk_end_token - chunk_start_token
        if chunk_len <= 0:
            continue
        global_start = bos + chunk_start_token
        for qk_head in range(H_qk):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float32)
            BK = 128
            BK = min(BK, K)
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, qk_head, global_start:global_start+chunk_len, i_k:k_end]
                q_normal = q[batch_idx, qk_head, global_start:global_start+chunk_len, i_k:k_end]
                b_q = q_normal.transpose(0, 1)
                if chunk_len == 1:
                    matmul_result = torch.sum(b_k * q_normal)
                    b_A[:chunk_len, :chunk_len] += matmul_result
                else:
                    b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q)
            for do_group in range(h_ratio):
                do_head = qk_head * h_ratio + do_group
                b_g = g_t[batch_idx, do_head, global_start:global_start+chunk_len].to(torch.float32)
                o_t = i_t * BT + torch.arange(0, BT)
                m_t = o_t < T
                o_t_col = o_t.unsqueeze(1)
                o_t_row = o_t.unsqueeze(0)
                pos_mask = o_t_col <= o_t_row
                m_t_col = m_t.unsqueeze(1)
                valid_mask = m_t_col & m_t
                m_A = pos_mask & valid_mask
                g_i = b_g.unsqueeze(1)
                g_j = b_g.unsqueeze(0)
                g_factor = torch.exp(g_j - g_i)
                b_A_gated = torch.zeros_like(b_A)
                b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor * scale
                b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated))
                b_A_masked = b_A_masked.to(qkv_type)
                BV = 128
                BV = min(BV, V)
                for i_v in range(0, V, BV):
                    v_end = min(i_v + BV, V)
                    v_width = v_end - i_v
                    b_do = do[batch_idx, do_head, global_start:global_start+chunk_len, i_v:v_end]
                    b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do)
                    dv[batch_idx, do_head, global_start:global_start+chunk_len, i_v:v_end] += b_dv
    return dv

def chunk_bwd_dv_local_variable_high_precision(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
    h_ratio: int = 1
) -> torch.Tensor:
    B, H_qk, T, K = k.shape
    H_do = do.shape[1]
    qkv_type = q.dtype
    V = do.shape[3]
    assert H_do == H_qk * h_ratio, f"H_do ({H_do}) must equal H_qk * h_ratio ({H_qk} * {h_ratio} = {H_qk * h_ratio})"
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if cu_seqlens is not None:
        batch_idx = 0
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_indices = chunk_indices.view(-1)
    BT = min(chunk_size, max(16, 2 ** math.ceil(math.log2(T))))
    NT = len(chunk_indices) // 2
    dv = torch.zeros_like(do).to(torch.float64)
    g_t = g
    for chunk_idx in range(NT):
        i_n = chunk_indices[chunk_idx * 2].item()
        i_t = chunk_indices[chunk_idx * 2 + 1].item()
        bos = cu_seqlens[i_n].item()
        eos = cu_seqlens[i_n + 1].item()
        T = eos - bos
        chunk_start_token = i_t * chunk_size
        chunk_end_token = min(chunk_start_token + chunk_size, T)
        chunk_len = chunk_end_token - chunk_start_token
        if chunk_len <= 0:
            continue
        global_start = bos + chunk_start_token
        for qk_head in range(H_qk):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float64)
            BK = 128
            BK = min(BK, K)
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, qk_head, global_start:global_start+chunk_len, i_k:k_end].to(torch.float32)
                q_normal = q[batch_idx, qk_head, global_start:global_start+chunk_len, i_k:k_end].to(torch.float32)
                b_q = q_normal.transpose(0, 1)
                if chunk_len == 1:
                    matmul_result = torch.sum(b_k * q_normal)
                    b_A[:chunk_len, :chunk_len] += matmul_result
                else:
                    b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q)
            for do_group in range(h_ratio):
                do_head = qk_head * h_ratio + do_group
                b_g = g_t[batch_idx, do_head, global_start:global_start+chunk_len].to(torch.float64)
                o_t = i_t * BT + torch.arange(0, BT)
                m_t = o_t < T
                o_t_col = o_t.unsqueeze(1)
                o_t_row = o_t.unsqueeze(0)
                pos_mask = o_t_col <= o_t_row
                m_t_col = m_t.unsqueeze(1)
                valid_mask = m_t_col & m_t
                m_A = pos_mask & valid_mask
                g_i = b_g.unsqueeze(1)
                g_j = b_g.unsqueeze(0)
                g_factor = torch.exp(g_j - g_i)
                b_A_gated = torch.zeros_like(b_A)
                b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor * scale
                b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated))
                b_A_masked = b_A_masked.to(torch.float32)
                BV = 128
                BV = min(BV, V)
                for i_v in range(0, V, BV):
                    v_end = min(i_v + BV, V)
                    v_width = v_end - i_v
                    b_do = do[batch_idx, do_head, global_start:global_start+chunk_len, i_v:v_end].to(torch.float32)
                    b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do)
                    dv[batch_idx, do_head, global_start:global_start+chunk_len, i_v:v_end] += b_dv
    return dv


def test_chunk_bwd_dv_local_fix(
    B: int,
    H_qk: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    scale: float,
    ktype,
    gtype,
    seed: int = 0,
    h_ratio: int = 1,
    case_name: str = None,
):
    if case_name is None:
        case_name = make_case_name("fix", B, H_qk, T, K, V, chunk_size, scale, ktype, gtype, h_ratio)
    set_global_seed(GLOBAL_SEED)
    H_do = H_qk * h_ratio
    q = create_tensor((B, H_qk, T, K), dtype=ktype)
    k = create_tensor((B, H_qk, T, K), dtype=ktype)
    d_o = create_tensor((B, H_do, T, V), dtype=ktype)
    g = torch.arange(B * H_do * T, 0, -1).reshape((B, H_do, T)).to(gtype)

    golden_data = load_golden(case_name)
    if golden_data is not None:
        dv_golden = golden_data["dv_golden"]
        dv_golden_high_precision = golden_data["dv_golden_high_precision"]
        print(f"[{case_name}] golden loaded, skip computation")
    else:
        cu_seqlens = None
        dv_golden = chunk_bwd_dv_local_fix(q, k, d_o, g, scale, cu_seqlens, chunk_size, h_ratio)
        print(f"[{case_name}] chunk_bwd_dv_local_fix golden done")
        dv_golden_high_precision = chunk_bwd_dv_local_fix_high_precision(q, k, d_o, g, scale, cu_seqlens, chunk_size, h_ratio)
        print(f"[{case_name}] chunk_bwd_dv_local_fix_high_precision golden done")
        save_golden(case_name, dv_golden=dv_golden, dv_golden_high_precision=dv_golden_high_precision)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()

    dv = torch.ops.npu.npu_chunk_bwd_dv_local(
        q_npu, k_npu, d_o_npu, g_npu,
        g_gamma=None,
        A=None,
        cu_seqlens=None,
        chunk_indices=None,
        scale=scale,
        chunk_size=chunk_size
    )
    print(f"[{case_name}] npu op done")
    result = dual(dv.cpu(), dv_golden, dv_golden_high_precision)
    print(f"[{case_name}] H_qk={H_qk}, H_do={H_do}, h_ratio={h_ratio}, result={result}")
    stop_on_case_failure(case_name, result)


def test_chunk_bwd_dv_local_variable(
    B: int,
    H_qk: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    scale: float,
    cu_seqlens_len: int,
    ktype,
    gtype,
    seed: int = 0,
    h_ratio: int = 1,
    case_name: str = None,
):
    if case_name is None:
        case_name = make_case_name("var", B, H_qk, T, K, V, chunk_size, scale, ktype, gtype, h_ratio, cu_seqlens_len)
    set_global_seed(GLOBAL_SEED)
    H_do = H_qk * h_ratio
    q = create_tensor((B, H_qk, T, K), dtype=ktype)
    k = create_tensor((B, H_qk, T, K), dtype=ktype)
    d_o = create_tensor((B, H_do, T, V), dtype=ktype)
    g = torch.arange(B * H_do * T, 0, -1).reshape((B, H_do, T)).to(gtype)

    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    golden_data = load_golden(case_name)
    if golden_data is not None:
        dv_golden = golden_data["dv_golden"]
        dv_golden_high_precision = golden_data["dv_golden_high_precision"]
        print(f"[{case_name}] golden loaded, skip computation")
    else:
        dv_golden = chunk_bwd_dv_local_variable(q, k, d_o, g, scale, cu_seqlens, chunk_size, h_ratio)
        print(f"[{case_name}] chunk_bwd_dv_local_variable golden done")
        dv_golden_high_precision = chunk_bwd_dv_local_variable_high_precision(q, k, d_o, g, scale, cu_seqlens, chunk_size, h_ratio)
        print(f"[{case_name}] chunk_bwd_dv_local_variable_high_precision golden done")
        save_golden(case_name, dv_golden=dv_golden, dv_golden_high_precision=dv_golden_high_precision)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()

    cu_seqlens_list = cu_seqlens.tolist()
    chunk_indices_list = chunk_indices.flatten().tolist()

    dv = torch.ops.npu.npu_chunk_bwd_dv_local(
        q_npu, k_npu, d_o_npu, g_npu,
        g_gamma=None,
        A=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list,
        scale=scale,
        chunk_size=chunk_size
    )
    print(f"[{case_name}] npu op done")
    result = dual(dv.cpu(), dv_golden, dv_golden_high_precision)
    print(f"[{case_name}] H_qk={H_qk}, H_do={H_do}, h_ratio={h_ratio}, result={result}")
    stop_on_case_failure(case_name, result)


if __name__ == "__main__":
    set_global_seed(GLOBAL_SEED)
    ################################## 一阶段泛化用例 (h_ratio=1, H_qk=H_do) ##################################
    # C1
    test_chunk_bwd_dv_local_fix(B=64, H_qk=8, T=1024, K=128, V=128, chunk_size=64, scale=0.088, ktype=torch.float16, gtype=torch.float16, case_name="C1")
    # C2
    test_chunk_bwd_dv_local_fix(B=32, H_qk=16, T=2048, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="C2")
    # C3
    test_chunk_bwd_dv_local_fix(B=16, H_qk=32, T=4096, K=128, V=128, chunk_size=64, scale=0.0442, ktype=torch.float16, gtype=torch.float16, case_name="C3")
    # C4
    test_chunk_bwd_dv_local_fix(B=8, H_qk=32, T=8192, K=128, V=128, chunk_size=64, scale=0.03125, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="C4")
    # A5 reverse-sync regression: the ready/free handshake exceeds one hardware counter window.
    test_chunk_bwd_dv_local_fix(B=2, H_qk=32, T=8192, K=128, V=128, chunk_size=64, scale=0.03125, ktype=torch.bfloat16, gtype=torch.float32, case_name="A5_SYNC_B2")
    # C5
    test_chunk_bwd_dv_local_fix(B=128, H_qk=4, T=1024, K=128, V=128, chunk_size=64, scale=0.088, ktype=torch.float16, gtype=torch.float16, case_name="C5")
    # C6
    test_chunk_bwd_dv_local_fix(B=64, H_qk=8, T=4096, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="C6")
    # C7
    test_chunk_bwd_dv_local_fix(B=32, H_qk=16, T=8192, K=128, V=128, chunk_size=64, scale=0.0442, ktype=torch.float16, gtype=torch.float16, case_name="C7")
    # C8
    test_chunk_bwd_dv_local_fix(B=16, H_qk=32, T=16384, K=128, V=128, chunk_size=64, scale=0.03125, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="C8")
    # C9
    test_chunk_bwd_dv_local_fix(B=64, H_qk=8, T=2048, K=128, V=128, chunk_size=128, scale=0.0625, ktype=torch.float16, gtype=torch.float16, case_name="C9")
    # C10
    test_chunk_bwd_dv_local_fix(B=32, H_qk=16, T=4096, K=128, V=128, chunk_size=128, scale=0.0442, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="C10")
    # C11
    test_chunk_bwd_dv_local_fix(B=16, H_qk=32, T=8192, K=128, V=128, chunk_size=128, scale=0.03125, ktype=torch.float16, gtype=torch.float16, case_name="C11")
    # C12
    test_chunk_bwd_dv_local_fix(B=8, H_qk=32, T=16384, K=128, V=128, chunk_size=128, scale=0.0221, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="C12")
    # C13
    test_chunk_bwd_dv_local_fix(B=1, H_qk=4, T=1024, K=128, V=128, chunk_size=64, scale=0.088, ktype=torch.float16, gtype=torch.float16, case_name="C13")
    # C14
    test_chunk_bwd_dv_local_fix(B=48, H_qk=8, T=2048, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="C14")
    # C15
    test_chunk_bwd_dv_local_fix(B=24, H_qk=16, T=4096, K=128, V=128, chunk_size=64, scale=0.0442, ktype=torch.float16, gtype=torch.float16, case_name="C15")
    # C16
    test_chunk_bwd_dv_local_fix(B=12, H_qk=32, T=8192, K=128, V=128, chunk_size=64, scale=0.03125, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="C16")
    # V1
    test_chunk_bwd_dv_local_variable(B=1, H_qk=16, T=32768, K=128, V=128, chunk_size=64, scale=0.0625, cu_seqlens_len=512, ktype=torch.float16, gtype=torch.float32, case_name="V1")
    # V2
    test_chunk_bwd_dv_local_variable(B=1, H_qk=8, T=65536, K=128, V=128, chunk_size=64, scale=0.0625, cu_seqlens_len=1024, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="V2")
    # V3
    test_chunk_bwd_dv_local_variable(B=1, H_qk=32, T=65536, K=128, V=128, chunk_size=64, scale=0.0442, cu_seqlens_len=1024, ktype=torch.float16, gtype=torch.float32, case_name="V3")
    # V4
    test_chunk_bwd_dv_local_variable(B=1, H_qk=32, T=16384, K=128, V=128, chunk_size=64, scale=0.03125, cu_seqlens_len=256, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="V4")

    # ################################## Vdim 256 泛化用例 #################################################
    test_chunk_bwd_dv_local_fix(B=2, H_qk=2, T=512, K=128, V=256, chunk_size=64, scale=0.0625, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="Vdim256_F1")
    test_chunk_bwd_dv_local_variable(B=1, H_qk=2, T=512, K=128, V=256, chunk_size=64, scale=0.011, cu_seqlens_len=4, ktype=torch.bfloat16, gtype=torch.bfloat16, case_name="Vdim256_V1")

    # # ################################## GVA 泛化用例 (H_do = h_ratio × H_qk) ##################################
    # GVA-F1: h_ratio=2, H_qk=4, H_do=8
    test_chunk_bwd_dv_local_fix(B=2, H_qk=4, T=512, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.bfloat16, gtype=torch.bfloat16, h_ratio=2, case_name="GVA_F1")
    # GVA-F2: h_ratio=2, H_qk=8, H_do=16
    test_chunk_bwd_dv_local_fix(B=2, H_qk=8, T=1024, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.bfloat16, gtype=torch.bfloat16, h_ratio=2, case_name="GVA_F2")
    # GVA-F3: h_ratio=4, H_qk=4, H_do=16
    test_chunk_bwd_dv_local_fix(B=2, H_qk=4, T=512, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.bfloat16, gtype=torch.bfloat16, h_ratio=4, case_name="GVA_F3")
    # GVA-F4: h_ratio=2, H_qk=4, H_do=8, V=256
    test_chunk_bwd_dv_local_fix(B=2, H_qk=4, T=512, K=128, V=256, chunk_size=64, scale=0.0625, ktype=torch.bfloat16, gtype=torch.bfloat16, h_ratio=2, case_name="GVA_F4")
    # GVA-F5: h_ratio=2, H_qk=8, H_do=16, float16
    test_chunk_bwd_dv_local_fix(B=2, H_qk=8, T=1024, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.float16, gtype=torch.float16, h_ratio=2, case_name="GVA_F5")
    # GVA-FIX4: h_ratio=32, H_qk=2, H_do=64, short T, V=256, fp16/fp32 gate
    test_chunk_bwd_dv_local_fix(B=176, H_qk=2, T=24, K=128, V=256, chunk_size=64, scale=0.0625, ktype=torch.float16, gtype=torch.float32, h_ratio=32, case_name="GVA-FIX4")
    # GVA-V1: h_ratio=2, variable length
    test_chunk_bwd_dv_local_variable(B=1, H_qk=4, T=512, K=128, V=128, chunk_size=64, scale=0.0625, cu_seqlens_len=4, ktype=torch.bfloat16, gtype=torch.bfloat16, h_ratio=2, case_name="GVA_V1")
    # GVA-V2: h_ratio=4, variable length
    test_chunk_bwd_dv_local_variable(B=1, H_qk=4, T=512, K=128, V=128, chunk_size=64, scale=0.0625, cu_seqlens_len=4, ktype=torch.bfloat16, gtype=torch.bfloat16, h_ratio=4, case_name="GVA_V2")
