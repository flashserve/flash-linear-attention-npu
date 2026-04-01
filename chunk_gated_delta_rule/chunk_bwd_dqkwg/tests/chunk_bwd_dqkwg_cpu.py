import torch
import torch_npu
import os
from ct import single
import torch
import torch.nn.functional as F
from typing import Tuple
from typing import Optional
import pickle
import math
import sys

def pause():
    print("pause")
    input()

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def cdiv(a: torch.LongTensor
    , b : int):
    torch.empty
    return (a + b - 1) // b

def prepare_chunk_indices_torch(
    cu_seqlens: torch.LongTensor,
    chunkSize: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunkSize).tolist()])
    # print("cu_seqlens is ", cu_seqlens)
    # print("indices is ", indices)

    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

def prepare_chunk_indices(
    cu_seqlens: list[int],
    chunk_size: int
) -> list[int]: 
    """
    基于 cu_seqlens (list[int]) 生成 chunk 索引。
    
    注意：原 PyTorch 版本返回的是 shape [N, 2] 的 Tensor。
    为了保持纯 Python 兼容性，这里返回 list[tuple[start_seq_idx, chunk_idx_in_seq]]。
    如果算子需要扁平化的 list[int] (如 [s0, c0, s1, c1, ...])，请在调用前展开。
    
    逻辑复刻原代码：
    1. 计算每个序列的长度: lens[i] = cu_seqlens[i+1] - cu_seqlens[i]
    2. 计算每个序列需要的 chunk 数: ceil(lens[i] / chunk_size)
    3. 生成对应的 (sequence_id, chunk_id) 对
    """
    indices = []
    
    # 遍历每个序列段
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        length = end - start
        
        if length <= 0:
            continue
            
        # 计算该序列需要多少个 chunk
        # 等价于 cdiv(length, chunk_size)
        num_chunks = (length + chunk_size - 1) // chunk_size
        
        for chunk_id in range(num_chunks):
            # 原逻辑: indices.eq(0).cumsum(0) - 1 对应的是序列索引 i
            # 原逻辑: indices 对应的是 chunk_id
            indices.append((i))
            indices.append((chunk_id))
            
    return indices


def chunk_bwd_dqkwg_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CPU Equivalent of chunk_bwd_kernel_dqkwg.
    """
    q.to(torch.float32)
    k.to(torch.float32)
    v.to(torch.float32)
    do.to(torch.float32)
    h.to(torch.float32)
    dh.to(torch.float32)
    w.to(torch.float32)
    g.to(torch.float32)
    dv.to(torch.float32)
    B, T, H, K = q.shape
    V = v.shape[-1]
    datatype = q.dtype
    gtype = g.dtype
    calctype = torch.float32
    g_gamma = None
    # print(f"h {h.dtype}")
    
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dg = torch.zeros_like(g) if g is not None else None
    dw = torch.zeros_like(w) if w is not None else None
    
    # 辅助函数：处理单个序列的逻辑
    def process_sequence(b_idx, t_start, t_end, seq_idx_in_batch, chunk_start_idx):
        # 计算该序列有多少个块
        seq_len = t_end - t_start
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        # print("H(head)", H, "num_chunks", num_chunks, "b_idx", b_idx, "t_start", t_start, "t_end", t_end, "seq_idx_in_batch", seq_idx_in_batch, "chunk_start_idx", chunk_start_idx)

        
        for h_idx in range(H):
            # 获取当前头的 gamma (如果 USE_G_GAMMA)
            gamma_val = None
            if g_gamma is not None:
                gamma_val = g_gamma[h_idx].item()

            for i_t in range(num_chunks):
                # 块的绝对起始位置
                chunk_start_token_idx = t_start + i_t * chunk_size
                chunk_end_token_idx = min(t_start + (i_t + 1) * chunk_size, t_end)
                actual_chunk_len = chunk_end_token_idx - chunk_start_token_idx

                # 当前块在 h/dh 中的索引 (NT 维度)
                # Triton 代码逻辑: i_tg = i_b * NT + i_t (定长) 或 i_t (变长且 chunk_indices 处理)
                # 这里我们假设 h 形状为 (B, H, NT, K, V) 或者兼容的扁平结构。
                # 为简化，假设标准 FLA 布局 (B, H, NT, K, V)
                # 注意：Triton 中 h 是指向第 i_t 个块的*起始*状态 (即上一个块的输出)
                
                # 切片当前块的数据
                
                q_c = q[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :]  # [BT, K]
                k_c = k[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :]  # [BT, K]
                v_c = v[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :]  # [BT, V]
                do_c = do[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] # [BT, V]

                # 获取状态 (h_prev) 和 状态梯度 (dh_curr)
                # h[..., i_t, ...] 存储的是第 i_t 块之前的状态 (即第 i_t-1 块的输出)
                # print("\th ", h.shape, f"h[{b_idx}, {i_t} + {chunk_start_idx}, {h_idx}, :, :]")
                h_prev = h[b_idx, i_t + chunk_start_idx, h_idx, :, :]  # [K, V]  ## 不对齐的情况??
                dh_curr = dh[b_idx, i_t + chunk_start_idx, h_idx, :, :] # [K, V]

                # -----------------------------------------------------------
                # 1. State Contributions (Inter-chunk)
                # -----------------------------------------------------------
                # Triton: b_dq += dot(b_do, b_h) -> do @ h_prev.T
                # h_prev 是 [K, V], do_c 是 [BT, V] -> [BT, K]
                dq_from_state = do_c.to(torch.float32) @ h_prev.transpose(-1, -2).to(torch.float32)

                dq_from_state = dq_from_state.to(datatype).to(torch.float32)

                # Triton: b_dk += dot(b_v, b_dh) -> v @ dh_curr.T
                # dh_curr 是 [K, V], v_c 是 [BT, V] -> [BT, K]
                dk_from_state = v_c.to(torch.float32) @ dh_curr.transpose(-1, -2).to(torch.float32)
                dk_from_state = dk_from_state.to(datatype).to(torch.float32)
                # Triton: if USE_DW -> b_dw += dot(b_dv, b_h)
                if w is not None and dv is not None:
                    dv_c = dv[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] # [BT, V]
                    # dw_c: [BT, K]
                    dw_c_val = dv_c.to(torch.float32) @ h_prev.transpose(-1, -2).to(torch.float32)
                    dw_c_val = dw_c_val.to(datatype).to(torch.float32)
                    # Triton stores -b_dw
                    dw[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = -dw_c_val

                # -----------------------------------------------------------
                # 2. Gating / Decay Logic Preparation
                # -----------------------------------------------------------
                # 构建 g_c (decay values)
                if g is not None:
                    g_c = g[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx] # [BT]

                    g_last = g[b_idx, min(chunk_start_token_idx + chunk_size, t_end) - 1, h_idx]
                    
                    # Triton: b_dg_last += sum(h * dh)
                    dg_last_accum = (h_prev * dh_curr).sum()

                    dg_last_accum = dg_last_accum * torch.exp(g_last)
                    # Apply decay to state contributions

                    dq_from_state = dq_from_state * torch.exp(g_c)[:, None] * scale
                    dk_from_state = dk_from_state * torch.exp(-g_c + g_last)[:, None]

                    # Accumulate gradients into dg (from state terms)
                    # b_dg += sum(b_dq * b_q)
                    dg_c = (dq_from_state * q_c).sum(dim=-1)
                    # print("ADD0.A", dg_c.to(datatype).to(torch.float32))
                    dg_c = dg_c.to(datatype).to(torch.float32)         #ADD0.A

                    # b_dg -= sum(b_k * b_dk)
                    dg_c -= (k_c * dk_from_state).sum(dim=-1)           #ADD0.B
                    # print("k_c",k_c)
                    # print("dk_from_state",dk_from_state)
                    # print("k_c * dk_from_state",( k_c * dk_from_state)[0])
                    # print("Add0.B", -(k_c * dk_from_state).sum(dim=-1))
                    dg_c = dg_c.to(datatype).to(torch.float32)

                    # b_dg_last += sum(b_dk * b_k)
                    # print(f"dg_last_accum {dg_last_accum} += (dk_from_state * k_c).sum() {(dk_from_state * k_c).sum()}")
                    # if h_idx == 0 and i_t == 31:
                    #     print("     sum0 result", (dk_from_state * k_c).sum())
                    dg_last_accum += (dk_from_state * k_c).sum()
                    # print("dg_last_accum += (dk_from_state * k_c).sum()", dg_last_accum)
                    # pause()
                    

                elif g_gamma is not None:
                    # Scalar decay
                    # b_g = b_gamma * (arange + 1)
                    # b_g_last = b_gamma * actual_chunk_len
                    # 这里模拟 Triton 里的相对 decay 逻辑
                    arange = torch.arange(actual_chunk_len, device=q.device, dtype=q.dtype)
                    g_c = gamma_val * (arange + 1)
                    g_last = gamma_val * actual_chunk_len
                    
                    dq_from_state = dq_from_state * torch.exp(g_c)[:, None] * scale
                    dk_from_state = dk_from_state * torch.exp(-g_c + g_last)[:, None]
                    # USE_G_GAMMA 模式下不需要计算 dg
                else:
                    # No decay
                    # Triton: b_dk *= scale (else block)
                    dk_from_state = dk_from_state * scale
                    dq_from_state = dq_from_state * scale

                # -----------------------------------------------------------
                # 3. Intra-chunk Attention
                # -----------------------------------------------------------
                ds = do_c.to(torch.float32) @ v_c.transpose(-1, -2).to(torch.float32) # [BT, BT]
                ds = ds.to(datatype).to(torch.float32)

                
                # Causal Mask
                i_indices = torch.arange(actual_chunk_len, device=q.device)[:, None]
                j_indices = torch.arange(actual_chunk_len, device=q.device)[None, :]
                mask = i_indices >= j_indices
                
                if g is not None:
                    # Decay: exp(g[i] - g[j])

                    decay_mat = torch.exp(g_c[:, None] - g_c[None, :])
                    # if h_idx == 0 and i_t == 7:
                    #     print("g_c[:, None] - g_c[None, :]", g_c[:, None] - g_c[None, :])
                    #     print("decay_mat = torch.exp(g_c[:, None] - g_c[None, :])", decay_mat)

                    ds = torch.where(mask, ds * decay_mat, torch.zeros_like(ds)) * scale
                    # print("decay_mat",decay_mat)
                    # print("ds",ds)

                    
                    # DG Calculation Part 2 (Intra-chunk)
                    # b_ds2 = b_ds * (q @ k.T)
                    qk_t = q_c.to(torch.float32) @ k_c.transpose(-1, -2).to(torch.float32)
                    qk_t = qk_t.to(datatype).to(torch.float32)


                    ds2 = ds * qk_t

                    # print("ADD0.C : +ds2.sum(dim=1)", ds2.sum(dim=1))
                    # print("ADD0.D : -ds2.sum(dim=0)", ds2.sum(dim=0))
                    dg_c += ds2.sum(dim=1)
                    dg_c -= ds2.sum(dim=0)

                    # dg_c = dg_c_C.to(torch.float16) + dg_c_D.to(torch.float16) + dg_c_A.to(torch.float16) + dg_c_B.to(torch.float16)
                    dg_c = dg_c.to(datatype).to(gtype)

                    # print("dg_c after", dg_c.shape)
                    # pause()
                    
                    # Finalize dg: revcumsum-like logic
                    # Triton: b_dg = where(o_t < T-1, b_dg, b_dg + b_dg_last)
                    # 只有块的最后一个有效 token 加上 dg_last_accum
                    # 注意：Triton 内核中的 revcumsum 通常在单独内核或最后处理，
                    # 但这里代码片段显示的是直接加上。
                    # 实际上 dg 在时间轴上是累积的梯度。
                    # 根据 Triton 代码: b_dg = ... + (idx == last ? b_dg_last : 0)
                    # 这里的 dg_c 仅仅是该位置的梯度 contribution。
                    # 为了完全匹配 Triton 的输出，我们需要把 dg_last_accum 加到块的最后。
                    if actual_chunk_len > 0:
                        dg_c[actual_chunk_len - 1] += dg_last_accum.to(gtype)  ## 实际上是is_last_mask

                    #     print(f"dg_c[{actual_chunk_len - 1}] += {dg_last_accum}")
                    # print("dg_c", dg_c)
                    dg[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx] = dg_c
                    # print("dg_c",dg_c)

                elif g_gamma is not None:

                    decay_mat = torch.exp(g_c[:, None] - g_c[None, :])
                    
                    ds = torch.where(mask, ds * decay_mat, torch.zeros_like(ds)) * scale

                else:
                    ds = torch.where(mask, ds, torch.zeros_like(ds))
                    # 在 else 分支，triton 代码: b_dq *= scale (最后)
                    # 但前面 state part 已经 scale 了。
                    # ds 计算时不乘 scale，最后 dq 乘 scale。
                    # 为了统一，这里先不乘 scale，下面加完后再处理，或者这里乘了下面不再乘。
                    # Triton 代码: b_dk += dot(trans(b_ds), b_q) * scale
                    # b_dq += dot(b_ds, b_k); b_dq *= scale
                    pass # logic handled below

                # -----------------------------------------------------------
                # 4. Final Accumulation for dq, dk
                # -----------------------------------------------------------
                # dq += ds @ k

                dq_intra = ds.to(torch.float32) @ k_c.to(torch.float32)
                # if h_idx == 0 and i_t == 7:
                #     print("ds.to(torch.float32)",ds.to(torch.float32))
                #     print("k_c.to(torch.float32)",k_c.to(torch.float32))
                dq_intra = dq_intra.to(datatype).to(torch.float32)
                # dk += ds.T @ q
                dk_intra = ds.transpose(-1, -2).to(torch.float32) @ q_c.to(torch.float32)
                dk_intra = dk_intra.to(datatype).to(torch.float32)

                
                if g is None and g_gamma is None:
                    # Special scaling for "No Decay" mode based on Triton code
                    dk_intra = dk_intra * scale
                    dq_total = (dq_from_state + dq_intra) * scale # Triton: b_dq *= scale at end
                    dk_total = dk_from_state + dk_intra
                else:
                    dq_total = dq_from_state + dq_intra
                    # if h_idx == 0 and i_t == 7:
                    #     print("dq_from_state",dq_from_state[-1])
                    #     print("dq_intra",dq_intra[-1])
                    #     print("dq_total",dq_total.shape,dq_total[-1])
                    dk_total = dk_from_state + dk_intra

                    # print(h_idx,i_t,"dk_from_state",dk_from_state)
                    # print(h_idx,i_t,"dk_intra",dk_intra)
                    # print(h_idx,i_t,"dk_total",dk_total)


                dq[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = dq_total
                dk[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = dk_total

                # if RANDOM_DATA == False:
                #     pass
                # if h_idx == 0 and i_t == 31:
                #     pass
                #     exit()
                

    # Main Loop
    if cu_seqlens is None:
        # Fixed length padding assumed or B*T
        for b in range(B):
            process_sequence(b, 0, T, b, 0)
    else:
        # Variable length
        chunk_location = torch.zeros(cu_seqlens.shape[0], dtype=torch.int64) #每个seq的chunk起始位置
        #chunk_location tensor([0, 64, 96, 128]) 代表：[0,63] [64,95] [96,127]

        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()
            seq_length = end - start
            # print("seq_length", seq_length)
            if i == 0:
                chunk_start_token_idx = 0
            else:
                chunk_start_token_idx = chunk_location[i]
            # print("chunk_start_token_idx before", chunk_start_token_idx)
            chunk_end_token_idx = chunk_start_token_idx + (seq_length + chunk_size - 1) // chunk_size
            # print("chunk_end_token_idx after", chunk_end_token_idx)
            chunk_location[i + 1] = chunk_end_token_idx

            # 在 Varlen 模式下，q/k/v 通常已经是 (Total_T, ...) 或者是 (1, Total_T, ...)
            # 但这里输入还是 (B, T, ...)，我们需要确认输入格式。
            # 通常 Triton varlen kernel 的输入 q 是 (Total_T, H, K)。
            # 如果输入是 packed (1, Total_T, ...)，b_idx 永远是 0。
            # 如果输入是 padded (B, T, ...)，则需要根据 cu_seqlens 切分。
            # 假设输入已根据 varlen 展平 (Batch=1) 或保持 Padded 格式。
            # 鉴于 Triton 代码 `i_b = i_bh // H`，如果 IS_VARLEN，逻辑略有不同。
            # 为保证通用性，这里假设输入是 Padded (B, T) 且 cu_seqlens 描述有效区域，
            # 或者 B=1 的 Packed 模式。
            if B == 1:
                print(f"start {start}, end {end}")
                # if (i == 0):
                #     continue
                process_sequence(0, start, end, i, chunk_location[i])
            else:
                # 如果是 Padded Batch 且提供了 cu_seqlens，这通常不常见，
                # 但如果发生，通常 cu_seqlens[i] 是第 i 个样本的长度。
                # 简化起见，我们假设 input 是 packed flat tensor 如果 cu_seqlens 存在。
                pass 

    return dq, dk, dw, dg
