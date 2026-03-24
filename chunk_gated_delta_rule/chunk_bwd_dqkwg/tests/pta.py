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
    print(f"h {h.dtype}")
    
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dg = torch.zeros_like(g) if g is not None else None
    dw = torch.zeros_like(w) if w is not None else None
    
    # 辅助函数：处理单个序列的逻辑
    def process_sequence(b_idx, t_start, t_end, seq_idx_in_batch, chunk_start_idx):
        # 计算该序列有多少个块
        seq_len = t_end - t_start
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        # print(f"seq_len {seq_len} = {t_end} - {t_start}")
        # print(f"num_chunks {num_chunks} = ({seq_len} + {chunk_size} - 1) // {chunk_size}")
        print("H(head)", H, "b_idx", b_idx, "t_start", t_start, "t_end", t_end, "seq_idx_in_batch", seq_idx_in_batch, "chunk_start_idx", chunk_start_idx)
        #last_unaligned_chunk = seq_len - num_chunks * chunk_size
        #first_chunk_location = chunk_start_idx#(t_start + chunk_size -1) // chunk_size  ##首chunk的位置
        #print()
        
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
                # print(f"  h_idx {h_idx} i_t {i_t} chunk_start_token_idx {chunk_start_token_idx} chunk_end_token_idx {chunk_end_token_idx} actual_chunk_len {actual_chunk_len}")
                #if (i_t == num_chunks - 1): #最后一个chunk
                
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
                #print(f"q_c {q_c.dtype} = q[{b_idx}, {chunk_start_token_idx}:{chunk_end_token_idx}, {h_idx}, :], {q.dtype}")
                
                # 获取状态 (h_prev) 和 状态梯度 (dh_curr)
                # h[..., i_t, ...] 存储的是第 i_t 块之前的状态 (即第 i_t-1 块的输出)
                h_prev = h[b_idx, i_t + chunk_start_idx, h_idx, :, :]  # [K, V]  ## 不对齐的情况??
                dh_curr = dh[b_idx, i_t + chunk_start_idx, h_idx, :, :] # [K, V]
                #print(f"h[{b_idx}, {i_t+t_start//chunk_size}, {h_idx}, :, :],{h.shape}, {h.dtype}, {h_prev.dtype}")

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
                    # print("add4 tensorDgFinal.GetValue(0) = ", dg_last_accum, "* ",torch.exp(g_last))
                    # print(f"dg_last_accum = (h_prev * dh_curr).sum() {(h_prev * dh_curr).sum()} * torch.exp(g_last) {torch.exp(g_last)}")
                    dg_last_accum = dg_last_accum * torch.exp(g_last)
                    # Apply decay to state contributions

                    # print("dq_from_state before",dq_from_state)
                    # print("g_c",g_c)
                    # print("torch.exp(g_c)[:, None]",torch.exp(g_c)[:, None])
                    dq_from_state = dq_from_state * torch.exp(g_c)[:, None] * scale
                    


                    # print("-g_c ", -g_c)
                    # print("-g_c + g_last",g_last, -g_c + g_last)
                    # print("torch.exp(-g_c + g_last)[:, None] before ", torch.exp(-g_c + g_last)[:, None])
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
                    # print("add4+(dk_from_state * k_c).sum()", (dk_from_state * k_c).sum())
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
                        # print("dg_c[-1] before", dg_c[-1])
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

                if RANDOM_DATA == False:
                    pass
                    if h_idx == 0 and i_t == 7:
                        pass
                        # pause()
                # exit()

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

# -------------------------------------------------------------------------
# 使用示例 / 验证
# -------------------------------------------------------------------------
if __name__ == "__main__":
    RANDOM_DATA = True
    case_number = 21
    if len(sys.argv) > 1:
        regen = sys.argv[1]
        if regen == "random":
            print("[test.py] regenerate all random data!")
            RANDOM_DATA=True

    # 简单的形状参数
    K, V = 128, 128
    calc_type = torch.float32
    isVarLen = False
    chunk_size = 128
    cases = [   #B,H,T,chunk_size,dtype,Gtype,scale,cu_seqlens
        [64,8,1024,64,torch.float16,torch.float16,0.088,None],
        [32,16,2048,64,torch.bfloat16,torch.bfloat16,0.0625,None],
        [16,32,4096,64,torch.float16,torch.float16,0.0442,None],
        [8,32,8192,64,torch.bfloat16,torch.bfloat16,0.03125,None],
        [128,4,1024,64,torch.float16,torch.float16,0.088,None],
        [64,4,4096,128,torch.bfloat16,torch.bfloat16,0.0625,None],
        [32,16,8192,64,torch.float16,torch.float16,0.0442,None],
        [16,32,16384,64,torch.bfloat16,torch.bfloat16,0.03125,None],
        [64,8,2048,128,torch.float16,torch.float16,0.0625,None],
        [32,16,4096,128,torch.bfloat16,torch.bfloat16,0.0442,None],
        [16,32,8192,128,torch.float16,torch.float16,0.03125,None],
        [8,32,8192,128,torch.bfloat16,torch.bfloat16,0.0221,None],  #C12
        [1,4,1024,64,torch.float16,torch.float16,0.088,None],
        [48,8,2048,64,torch.bfloat16,torch.bfloat16,0.0625,None],
        [24,16,4096,64,torch.float16,torch.float16,0.0442,None],
        [12,32,8192,64,torch.bfloat16,torch.bfloat16,0.03125,None],
        [1,16,32768,64,torch.float16,torch.float32,0.0625,torch.tensor([0,16,20000,30000,32768])],      # V1
        [1,8,65536,64,torch.bfloat16,torch.bfloat16,0.0625,torch.tensor([0,16,20000,65536])],
        [1,32,65536,64,torch.float16,torch.float32,0.0442,torch.tensor([0,16,20000,50000,65536])],
        [1,32,262144,64,torch.bfloat16,torch.bfloat16,0.03125,torch.tensor([0,16,20000,50000,65536,210000,262144])],
        [2,4,512,64,torch.bfloat16,torch.float32,0.088,None],  #21 [0,16,128] [0,16,135,512]
        [1,32,16384,64,torch.bfloat16,torch.float32,0.088,None],  #21 [0,16,128]
    ]
    device_id = 4
    

    dtype = torch.float16
    Gtype = torch.float16
    B, H = 4, 8
    T = 1024
    scale = 0.088
    if isVarLen:
        cu_seqlens = torch.cumsum(torch.tensor([0, 3, 64, 63, 66, 260]), dim=0)
    else:
        cu_seqlens = None
    if case_number != -1:
        single_case = cases[case_number-1]  #case_01 => cases[0]
        dtype = single_case[4]
        Gtype = single_case[5]
        B, H = single_case[0], single_case[1]
        chunk_size = single_case[3]
        cu_seqlens = single_case[7]
        cu_seqlens_torch = torch.tensor(cu_seqlens) if cu_seqlens is not None else None

        if single_case[7] is None:
            isVarLen = False
        else:
            isVarLen = True
        # isVarLen == single_case[7] != None
        T = single_case[2]
        scale = single_case[6]

    if isVarLen:
        B = 1  ##变长只支持B=1
        T = cu_seqlens_torch[-1]
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        num_chunks = len(chunk_indices) // 2
        print("chunk_indices",chunk_indices)
    else:
        chunk_indices = None
        num_chunks = T // chunk_size
    
    test_case_name = "test"
    data_path = "/data/huangjunzhe/GDN/data/"
    RUN_CPU = True
    SAVE_FILES = True
    if SAVE_FILES:
        os.makedirs(f'{data_path}/{test_case_name}/in/', exist_ok=True)
        os.makedirs(f'{data_path}/{test_case_name}/out/', exist_ok=True)
    if RANDOM_DATA:
        q = torch.randn(B,H,T,K, dtype=dtype, requires_grad=True) * 5e-7 * 100000 # std≈5e-6#torch.randn([B, T, H, K], dtype=dtype)
        k = torch.randn(B,H,T,K, dtype=dtype, requires_grad=True) * 5e-7 * 100000  # torch.randn([B, T, H, K], dtype=dtype)
        v = torch.randn(B,H,T,V, dtype=dtype, requires_grad=True) * 5e-7 * 10000  # torch.randn([B, T, H, V], dtype=dtype)

        # g = torch.randn(B,T,H, dtype=dtype, requires_grad=True) * 5e-2   # torch.randn([B, T, H], dtype=Gtype)
        g = -torch.sort(torch.rand(B*T*H) * 10, descending=False)[0].reshape((B,H,T)).to(Gtype)    #G必须递减且为负数
        # print("g",g)
        do = torch.randn(B,H,T,V, dtype=dtype, requires_grad=True) * 5e-7 * 100000  # torch.randn([B, T, H, V], dtype=dtype)

        dv = torch.randn(B,H,T,V, dtype=dtype, requires_grad=True) * 5e-7 * 1000000  # torch.randn([B, T, H, V], dtype=dtype)
        w = torch.randn(B,H,T,K, dtype=dtype, requires_grad=True) * 5e-7 * 100000  # torch.randn([B, T, H, K], dtype=dtype)

        h = torch.randn(B, H, num_chunks, K, V, dtype=dtype, requires_grad=True) * 5e-7 * 100000  # torch.randn([B, num_chunks, H, K, V], dtype=dtype)
        dh = torch.randn(B, H, num_chunks, K, V, dtype=dtype, requires_grad=True) * 5e-7 * 1000 # torch.randn([B, num_chunks, H, K, V], dtype=dtype)
        if SAVE_FILES:
            torch.save(q, f"{data_path}/{test_case_name}/in/q_cpu.pt")
            torch.save(k, f"{data_path}/{test_case_name}/in/k_cpu.pt")
            torch.save(v, f"{data_path}/{test_case_name}/in/v_cpu.pt")
            torch.save(g, f"{data_path}/{test_case_name}/in/g_cpu.pt")
            torch.save(do, f"{data_path}/{test_case_name}/in/do_cpu.pt")
            torch.save(dv, f"{data_path}/{test_case_name}/in/dv_cpu.pt")
            torch.save(w, f"{data_path}/{test_case_name}/in/w_cpu.pt")
            torch.save(h, f"{data_path}/{test_case_name}/in/h_cpu.pt")
            torch.save(dh, f"{data_path}/{test_case_name}/in/dh_cpu.pt")
    else:
        # q=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/q_cpu.pt", weights_only=False)
        # k=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/k_cpu.pt", weights_only=False)
        # v=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/v_new_cpu.pt", weights_only=False)
        # w=torch.empty_like(q)
        # g=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/g_cpu.pt", weights_only=False)
        # h=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/h_cpu.pt", weights_only=False)
        # dv=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/dv_cpu.pt", weights_only=False)
        # do=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/do_cpu.pt", weights_only=False)
        # dh=torch.load("/home/huangjunzhe/GDN/data/model/after_cpu/dh_cpu.pt", weights_only=False)
        q=torch.load(f"{data_path}/{test_case_name}/in/q_cpu.pt", weights_only=False)
        k=torch.load(f"{data_path}/{test_case_name}/in/k_cpu.pt", weights_only=False)
        v=torch.load(f"{data_path}/{test_case_name}/in/v_cpu.pt", weights_only=False)
        w=torch.empty_like(q)
        g=torch.load(f"{data_path}/{test_case_name}/in/g_cpu.pt", weights_only=False)
        h=torch.load(f"{data_path}/{test_case_name}/in/h_cpu.pt", weights_only=False)
        dv=torch.load(f"{data_path}/{test_case_name}/in/dv_cpu.pt", weights_only=False)
        do=torch.load(f"{data_path}/{test_case_name}/in/do_cpu.pt", weights_only=False)
        dh=torch.load(f"{data_path}/{test_case_name}/in/dh_cpu.pt", weights_only=False)
    q = q.to(dtype).to(calc_type)
    k = k.to(dtype).to(calc_type)
    v = v.to(dtype).to(calc_type)
    h = h.to(dtype).to(calc_type)
    g = g.to(Gtype).to(calc_type)
    do = do.to(dtype).to(calc_type)
    dh = dh.to(dtype).to(calc_type)
    dv = dv.to(dtype).to(calc_type)
    w = w.to(dtype).to(calc_type)
    print("entering chunk_bwd_dqkwg")
    print(f"q: {q.shape} {dtype} => {q.dtype}")
    print(f"k: {k.shape} {dtype} => {k.dtype}")
    print(f"v: {v.shape} {dtype} => {v.dtype}")
    print(f"w: {w.shape} {dtype} => {w.dtype}")
    print(f"g: {g.shape} {Gtype} => {g.dtype}")
    print(f"h: {h.shape} {dtype} => {h.dtype}")
    print(f"dv: {dv.shape} {dtype} => {dv.dtype}")
    print(f"do: {do.shape} {dtype} => {do.dtype}")
    print(f"dh: {dh.shape} {dtype} => {dh.dtype}")
    if cu_seqlens == None:
        print("cu_seqlens is None")
    else:
        print(f"cu_seqlens: {cu_seqlens_torch.shape} {cu_seqlens_torch.dtype} {cu_seqlens_torch}")
        # print(f"chunk_indices: {chunk_indices.shape} {chunk_indices.dtype} {chunk_indices}")
    print(f"scale: {scale}")
    print(f"chunk_size: {chunk_size}")


    print("==============start NPU=============")
    torch_npu.npu.set_device(device_id)
    q_npu = q.to(dtype).npu()
    k_npu = k.to(dtype).npu()
    v_npu = v.to(dtype).npu()
    w_npu = w.to(dtype).npu()
    g_npu = g.to(Gtype).npu()
    h_npu = h.to(dtype).npu()
    dv_npu = dv.to(dtype).npu()
    do_npu = do.to(dtype).npu()
    dh_npu = dh.to(dtype).npu()
    # cu_seqlens_npu = cu_seqlens if cu_seqlens is not None else None
    chunk_indices_npu = chunk_indices if cu_seqlens is not None else None
    down_tri = q_npu
    dq_npu, dk_npu, dw_npu, dg_npu = torch_npu.npu_chunk_bwd_dqkwg(
        q_npu, k_npu, v_npu, g_npu, h_npu, do_npu, dh_npu, dv_npu, cu_seqlens, chunk_indices_npu, scale, chunk_size
    )
    dq_npu = dq_npu.cpu()
    dk_npu = dk_npu.cpu()
    dw_npu = dw_npu.cpu()
    dg_npu = dg_npu.cpu()

    # print("Output shapes:", dq.shape, dk.shape, dg.shape, dw.shape)
    print("dq_npu", dq_npu.shape, dq_npu.dtype)
    print("dk_npu", dk_npu.shape, dk_npu.dtype)
    print("dw_npu", dw_npu.shape, dw_npu.dtype)
    print("dg_npu", dg_npu.shape, dg_npu.dtype)

    # print("dq_npu[0][0][-1]", dq_npu[0][0][-1])
    if RUN_CPU:
        print("=====start cpu=========")
        q = torch.transpose(q, 1, 2).to(dtype)
        k = torch.transpose(k, 1, 2).to(dtype)
        v = torch.transpose(v, 1, 2).to(dtype)
        w = torch.transpose(w, 1, 2).to(dtype)
        g = torch.transpose(g, 1, 2).to(Gtype)
        h = torch.transpose(h, 1, 2).to(dtype)
        dv = torch.transpose(dv, 1, 2).to(dtype)
        do = torch.transpose(do, 1, 2).to(dtype)
        dh = torch.transpose(dh, 1, 2).to(dtype)

        dq, dk, dw, dg = chunk_bwd_dqkwg_cpu(
            q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens_torch, chunk_size
        )
        # dq = dq.to(dtype)
        # dk = dk.to(dtype)
        # dw = dw.to(dtype)
        # dg = dg.to(Gtype)
        dq = torch.transpose(dq, 1, 2).cpu()
        dk = torch.transpose(dk, 1, 2).cpu()
        dw = torch.transpose(dw, 1, 2).cpu()
        dg = torch.transpose(dg, 1, 2).cpu()
        # print("dq[0][0][-1]", dq[0][0][-1])
        # print("dk", dk)
        # print("dw", dw)
        # print("dg", dg)

        print("dq", dq.cpu().shape, dq.cpu().dtype)
        print("dk", dk.cpu().shape, dk.cpu().dtype)
        print("dw", dw.cpu().shape, dw.cpu().dtype)
        print("dg", dg.cpu().shape, dg.cpu().dtype)

        type_dict = {torch.float16:"float16", torch.float32:"float32",torch.bfloat16:"bf16"}
        single(dq_npu,dq,calc_count=100000,dtype=type_dict[dtype])
        single(dk_npu,dk,calc_count=100000,dtype=type_dict[dtype])
        single(dw_npu,dw,calc_count=100000,dtype=type_dict[dtype])
        single(dg_npu,dg,calc_count=100000,dtype=type_dict[Gtype])
        if SAVE_FILES:
            if RANDOM_DATA:
                torch.save(dq,f"{data_path}/{test_case_name}/out/dq_cpu.pt")
                torch.save(dk,f"{data_path}/{test_case_name}/out/dk_cpu.pt")
                torch.save(dw,f"{data_path}/{test_case_name}/out/dw_cpu.pt")
                torch.save(dg,f"{data_path}/{test_case_name}/out/dg_cpu.pt")

            torch.save(dq_npu,f"{data_path}/{test_case_name}/out/dq_npu.pt")
            torch.save(dk_npu,f"{data_path}/{test_case_name}/out/dk_npu.pt")
            torch.save(dw_npu,f"{data_path}/{test_case_name}/out/dw_npu.pt")
            torch.save(dg_npu,f"{data_path}/{test_case_name}/out/dg_npu.pt")
            print(f"files saved to {data_path}/{test_case_name}/out/ .")