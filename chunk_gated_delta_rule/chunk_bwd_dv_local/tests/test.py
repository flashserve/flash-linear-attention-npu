import torch
import torch_npu
from typing import Optional
import math
import random
import ct

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


def create_tensor(shape, dtype=torch.float16):
    return torch.rand(shape, dtype=dtype)


def chunk_bwd_dv_local_fix(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64
) -> torch.Tensor:
    B, H, T, K = k.shape
    V = do.shape[3]
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
        for i_h in range(H):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float32)
            BK = 128
            BK = min(BK, K)
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len, i_k:k_end].to(torch.float32)
                q_normal = q[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len, i_k:k_end].to(torch.float32)
                b_q = q_normal.transpose(0, 1)
                if chunk_len == 1:
                    matmul_result = torch.sum(b_k * q_normal)
                    b_A[:chunk_len, :chunk_len] += matmul_result
                else:
                    b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q)
            b_g = g_t[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len]
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
                b_do = do[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end].to(torch.float32)
                b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do)
                dv[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end] += b_dv
    return dv


def chunk_bwd_dv_local_variable(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64
) -> torch.Tensor:
    B, H, T, K = k.shape
    V = do.shape[3]
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
        for i_h in range(H):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float32)
            BK = 128
            BK = min(BK, K)
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, i_h, global_start:global_start+chunk_len, i_k:k_end].to(torch.float32)
                q_normal = q[batch_idx, i_h, global_start:global_start+chunk_len, i_k:k_end].to(torch.float32)
                b_q = q_normal.transpose(0, 1)
                if chunk_len == 1:
                    matmul_result = torch.sum(b_k * q_normal)
                    b_A[:chunk_len, :chunk_len] += matmul_result
                else:
                    b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q)
            b_g = g_t[batch_idx, i_h, global_start:global_start+chunk_len]
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
                b_do = do[batch_idx, i_h, global_start:global_start+chunk_len, i_v:v_end].to(torch.float32)
                b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do)
                dv[batch_idx, i_h, global_start:global_start+chunk_len, i_v:v_end] += b_dv
    return dv


def test_chunk_bwd_dv_local_fix(
    B: int,
    H: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    scale: float,
    ktype,
    gtype,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if not hasattr(test_chunk_bwd_dv_local_fix, "call_count"):
        test_chunk_bwd_dv_local_fix.call_count = 1
    else:
        test_chunk_bwd_dv_local_fix.call_count += 1

    q = create_tensor((B, H, T, K), dtype=ktype)
    k = create_tensor((B, H, T, K), dtype=ktype)
    d_o = create_tensor((B, H, T, V), dtype=ktype)
    g = torch.arange(B * H * T, 0, -1).reshape((B, H, T)).to(gtype)

    cu_seqlens = None
    dv_golden = chunk_bwd_dv_local_fix(q, k, d_o, g, scale, cu_seqlens, chunk_size)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()

    dv = torch_npu.npu_chunk_bwd_dv_local(
        q_npu, k_npu, d_o_npu, g_npu,
        g_gamma=None,
        A=None,
        cu_seqlens=None,
        chunk_indices=None,
        scale=scale,
        chunk_size=chunk_size
    )

    ct.single(dv.cpu(), dv_golden)
    print(f"test_chunk_bwd_dv_local_fix 被调用了第 {test_chunk_bwd_dv_local_fix.call_count} 次")


def test_chunk_bwd_dv_local_variable(
    B: int,
    H: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    scale: float,
    cu_seqlens_len: int,
    ktype,
    gtype,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if not hasattr(test_chunk_bwd_dv_local_variable, "call_count"):
        test_chunk_bwd_dv_local_variable.call_count = 1
    else:
        test_chunk_bwd_dv_local_variable.call_count += 1

    q = create_tensor((B, H, T, K), dtype=ktype)
    k = create_tensor((B, H, T, K), dtype=ktype)
    d_o = create_tensor((B, H, T, V), dtype=ktype)
    g = torch.arange(B * H * T, 0, -1).reshape((B, H, T)).to(gtype)

    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    dv_golden = chunk_bwd_dv_local_variable(q, k, d_o, g, scale, cu_seqlens, chunk_size)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()

    cu_seqlens_list = cu_seqlens.tolist()
    chunk_indices_list = chunk_indices.flatten().tolist()

    dv = torch_npu.npu_chunk_bwd_dv_local(
        q_npu, k_npu, d_o_npu, g_npu,
        g_gamma=None,
        A=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list,
        scale=scale,
        chunk_size=chunk_size
    )

    ct.single(dv.cpu(), dv_golden)
    print(f"test_chunk_bwd_dv_local_variable 被调用了第 {test_chunk_bwd_dv_local_variable.call_count} 次")


if __name__ == "__main__":
    # Fix length tests
    test_chunk_bwd_dv_local_fix(B=2, H=2, T=65, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.float16, gtype=torch.float16)
    test_chunk_bwd_dv_local_fix(B=4, H=4, T=128, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.float16, gtype=torch.float32)
    test_chunk_bwd_dv_local_fix(B=8, H=8, T=256, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.float16, gtype=torch.float16)
    test_chunk_bwd_dv_local_fix(B=16, H=16, T=512, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.float16, gtype=torch.float16)
    test_chunk_bwd_dv_local_fix(B=32, H=32, T=1024, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.float16, gtype=torch.float16)
    
    # Variable length tests
    test_chunk_bwd_dv_local_variable(B=1, H=32, T=128, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=2, ktype=torch.float16, gtype=torch.float16)
    test_chunk_bwd_dv_local_variable(B=1, H=16, T=256, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=3, ktype=torch.float16, gtype=torch.float32)
    test_chunk_bwd_dv_local_variable(B=1, H=8, T=512, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=4, ktype=torch.float16, gtype=torch.float16)
    test_chunk_bwd_dv_local_variable(B=1, H=4, T=1024, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=5, ktype=torch.float16, gtype=torch.float16)
    test_chunk_bwd_dv_local_variable(B=1, H=32, T=2048, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=8, ktype=torch.float16, gtype=torch.float16)
