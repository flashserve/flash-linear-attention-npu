import torch
from typing import Optional
import math

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

def chunk_bwd_dv_local_fix(
    q: torch.Tensor,  # [B, H, T, K]
    k: torch.Tensor,  # [B, H, T, K]
    do: torch.Tensor, # [B, H, T, V]
    g: torch.Tensor,  # [B, H, T]
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,  # [batch_size+1]
    chunk_size: int = 64
) -> torch.Tensor:
    B, H, T, K = k.shape
    V = do.shape[3]
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if cu_seqlens is not None:
        batch_idx = 0
    BT = min(chunk_size, max(16, 2 ** math.ceil(math.log2(T)))) # 有风险，T至少要>=64,否则会计算错误
    chunk_per_T = (T + chunk_size -1)// chunk_size
    NT = chunk_per_T * B
    dv = torch.zeros_like(do).to(torch.float32)
    g_t = g
    for chunk_idx in range(NT):
        i_n = chunk_idx // chunk_per_T # 序列编号
        batch_idx = i_n
        i_t = chunk_idx % chunk_per_T # chunk编号

        chunk_start_token = i_t * chunk_size # 当前chunk在序列内的起始token位置
        chunk_end_token = min(chunk_start_token + chunk_size, T) # 结束位置，不超过序列真实长度
        chunk_len = chunk_end_token - chunk_start_token # 当前chunk的有效token数
        if chunk_len <= 0:
            continue
        for i_h in range(H):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float32)
            BK = 128  # 与Triton保持一致
            BK = min(BK, K)  # 确保不超过K
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len, i_k:k_end].to(torch.float32) # [chunk_len, BK]
                q_normal = q[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len, i_k:k_end].to(torch.float32)  # [chunk_len, BK]
                b_q = q_normal.transpose(0, 1)  # [BK, chunk_len]
                if chunk_len == 1:
                    matmul_result = torch.sum(b_k * q_normal)
                    b_A[:chunk_len, :chunk_len] += matmul_result
                else:
                    b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q)# [BT,BT]
            b_g = g_t[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len] # g_t [B, H, T_max] → b_g [chunk_len]
            o_t = i_t * BT + torch.arange(0, BT) # [BT] chunk内的token序号
            m_t = o_t < T # [BT] bool掩码：是否是有效token
            o_t_col = o_t.unsqueeze(1)  # [BT, 1]
            o_t_row = o_t.unsqueeze(0)  # [1, BT]
            pos_mask = o_t_col <= o_t_row  # [BT, BT] 上三角矩阵：只允许当前token看<=自己的token（因果掩码）
            m_t_col = m_t.unsqueeze(1)  # [BT, 1]
            valid_mask = m_t_col & m_t  #  [BT, BT] 有效掩码：只保留有效token的位置，左上角是[有效token,有效token]大小的全1
            # 组合掩码
            m_A = pos_mask & valid_mask  # [BT, BT]
            g_i = b_g.unsqueeze(1)   # [chunk_len, 1]
            g_j = b_g.unsqueeze(0)  # [1, chunk_len]
            g_factor = torch.exp(g_j - g_i)  * scale  # [chunk_len, chunk_len]
            b_A_gated = torch.zeros_like(b_A)
            b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor # [BT, BT] 门控缩放后的注意力核矩阵
            # 应用掩码
            b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated)) # 只保留掩码为 True 的位置的 b_A_gated 值，其余置 0
            b_A_masked = b_A_masked.to(torch.float32) # [BT, BT]
            BV = 128  # 与Triton保持一致
            BV = min(BV, V)  # 确保不超过V
            for i_v in range(0, V, BV):
                v_end = min(i_v + BV, V)
                v_width = v_end - i_v
                b_do = do[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end].to(torch.float32) # do [B, T_max, H, V] → b_do [chunk_len, BV]
                b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do) # b_A_masked 这个 [BT, BT] 的矩阵，只有左上角 [chunk_len, chunk_len] 区域有非 0 值，其余所有区域全是 0
                dv[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end] += b_dv
    return dv

def chunk_bwd_dv_local_variable(
    q: torch.Tensor,  # [B, H, T_max, K]
    k: torch.Tensor,  # [B, H, T_max, K]
    do: torch.Tensor, # [B, H, T_max, V]
    g: torch.Tensor,  # [B, H, T_max]
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,  # [batch_size+1]
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
    BT = min(chunk_size, max(16, 2 ** math.ceil(math.log2(T)))) # 有风险，T至少要>=64,否则会计算错误
    NT = len(chunk_indices) // 2 
    dv = torch.zeros_like(do).to(torch.float32)
    g_t = g
    for chunk_idx in range(NT):
        i_n = chunk_indices[chunk_idx * 2].item() # 序列编号
        i_t = chunk_indices[chunk_idx * 2 + 1].item() # chunk编号
        bos = cu_seqlens[i_n].item() # [0,2048,4096,...]  当前token 在T中开始的位置
        eos = cu_seqlens[i_n + 1].item()
        T = eos - bos
        chunk_start_token = i_t * chunk_size # 当前chunk在序列内的起始token位置
        chunk_end_token = min(chunk_start_token + chunk_size, T) # 结束位置，不超过序列真实长度
        chunk_len = chunk_end_token - chunk_start_token # 当前chunk的有效token数
        if chunk_len <= 0:
            continue
        global_start = bos + chunk_start_token # 当前chunk在T中开始的位置
        for i_h in range(H):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float32)
            BK = 128  # 与Triton保持一致
            BK = min(BK, K)  # 确保不超过K
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, i_h, global_start:global_start+chunk_len, i_k:k_end].to(torch.float32) # [chunk_len, BK]
                q_normal = q[batch_idx, i_h, global_start:global_start+chunk_len, i_k:k_end].to(torch.float32)  # [chunk_len, BK]
                b_q = q_normal.transpose(0, 1)  # [BK, chunk_len]
                if chunk_len == 1:
                    matmul_result = torch.sum(b_k * q_normal)
                    b_A[:chunk_len, :chunk_len] += matmul_result
                else:
                    b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q)# [BT,BT]
            b_g = g_t[batch_idx, i_h, global_start:global_start+chunk_len] # g_t [B, H, T_max] → b_g [chunk_len]
            o_t = i_t * BT + torch.arange(0, BT) # [BT] chunk内的token序号
            m_t = o_t < T # [BT] bool掩码：是否是有效token
            o_t_col = o_t.unsqueeze(1)  # [BT, 1]
            o_t_row = o_t.unsqueeze(0)  # [1, BT]
            pos_mask = o_t_col <= o_t_row   # [BT, BT] 上三角矩阵：只允许当前token看<=自己的token（因果掩码）
            m_t_col = m_t.unsqueeze(1)  # [BT, 1]
            valid_mask = m_t_col & m_t  #  [BT, BT] 有效掩码：只保留有效token的位置，左上角是[有效token,有效token]大小的全1
            # 组合掩码
            m_A = pos_mask & valid_mask  # [BT, BT]
            g_i = b_g.unsqueeze(1)  # [chunk_len, 1]
            g_j = b_g.unsqueeze(0)  # [1, chunk_len]
            g_factor = torch.exp(g_j - g_i)  # [chunk_len, chunk_len]
            b_A_gated = torch.zeros_like(b_A)
            # print(f"==== g_factor.shape = {g_factor.shape} ")
            # for i in range(g_factor.shape[0]):
            #     row_str = ', '.join([f"{g_factor[i, j].item():.6f}" for j in range(g_factor.shape[1])])
            #     print(row_str)
            b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor * scale # [BT, BT] 门控缩放后的注意力核矩阵
            # print(f"==== b_A.shape = {b_A.shape} ",b_A[:chunk_len, :chunk_len])
            # 应用掩码
            b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated)) # 只保留掩码为 True 的位置的 b_A_gated 值，其余置 0
            # print(f"==== b_A_masked.shape = {b_A_masked.shape} ", b_A_masked[:chunk_len, :chunk_len])
            # for i in range(b_A_masked.shape[0]):
            #     row_str = ', '.join([f"{b_A_masked[i, j].item():.6f}" for j in range(b_A_masked.shape[1])])
            #     print(row_str)
            b_A_masked = b_A_masked.to(torch.float32) # [BT, BT]
            BV = 128  # 与Triton保持一致
            BV = min(BV, V)  # 确保不超过V
            for i_v in range(0, V, BV):
                v_end = min(i_v + BV, V)
                v_width = v_end - i_v
                b_do = do[batch_idx, i_h, global_start:global_start+chunk_len, i_v:v_end].to(torch.float32) # do [B, T_max, H, V] → b_do [chunk_len, BV]
                b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do) # b_A_masked 这个 [BT, BT] 的矩阵，只有左上角 [chunk_len, chunk_len] 区域有非 0 值，其余所有区域全是 0
                dv[batch_idx, i_h, global_start:global_start+chunk_len, i_v:v_end] += b_dv
    return dv
