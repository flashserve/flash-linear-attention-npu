import torch
import torch_npu
import os
from typing import Optional
import pickle
import math
# import ct
import random
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
import fla_npu
# import custom_ops

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

def get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices):
    if cu_seqlens != None:
        seqIdx = chunk_indices[idx * 2]
        chunkIdx = chunk_indices[idx * 2 + 1]
        bos = cu_seqlens[seqIdx] + chunkIdx * chunk_size
        eos = bos + chunk_size
        if eos > cu_seqlens[seqIdx + 1]:
            eos = cu_seqlens[seqIdx + 1]
    else:
        bos = idx * chunk_size
        eos = bos + chunk_size
        if eos > T:
            eos = T
    # print(bos, eos)
    return bos,eos

def compute_dv_golden(
    A: torch.Tensor,      # [B, T, VH, chunk_size] - 每个chunk的A值
    du: torch.Tensor,     # [B, T, VH, D] - 上游梯度
    beta: torch.Tensor,   # [B, VH, T] - beta参数
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    VH: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dv computation (变长序列)
    A的形状为 [B, T, VH, chunk_size]
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    # 初始化dv，形状与du相同 [B, T, VH, D]
    dv = torch.zeros_like(du)
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            # print(bos, eos, eos-bos)
        # 遍历所有batch
            for i_h in range(VH):

            # 遍历所有head
                # 获取当前chunk的A向量
                # A形状: [B, T, VH, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,:eos - bos]  # [chunk_size, chunk_size]

                # 获取当前chunk的du
                du_chunk = du[i_b,i_h, bos:eos, :]  # [chunk_size, V]

                # 获取当前chunk的beta
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]

                # 计算 dv_chunk = A_chunk @ du_chunk * beta_chunk
                # 步骤1: b_dv_beta = A_chunk @ du_chunk
                b_dv_beta = torch.matmul(A_chunk.T, du_chunk)  # [chunk_size, V]

                # 步骤2: dv_chunk = b_dv_beta * beta_chunk.unsqueeze(-1)
                dv_chunk = b_dv_beta * beta_chunk[:, None]  # [chunk_size, D]

                # 存储结果
                dv[i_b,i_h, bos:eos, :] = dv_chunk.to(dv.dtype)

    return dv

def compute_dv_golden_high_precision(
    A: torch.Tensor,      # [B, T, VH, chunk_size] - 每个chunk的A值
    du: torch.Tensor,     # [B, T, VH, D] - 上游梯度
    beta: torch.Tensor,   # [B, VH, T] - beta参数
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    VH: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dv computation (变长序列)
    A的形状为 [B, T, VH, chunk_size]
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    # 初始化dv，形状与du相同 [B, T, VH, D]
    dv = torch.zeros_like(du).to(torch.float64)
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            # print(bos, eos, eos-bos)
        # 遍历所有batch
            for i_h in range(VH):

            # 遍历所有head
                # 获取当前chunk的A向量
                # A形状: [B, T, VH, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,:eos - bos]  # [chunk_size, chunk_size]

                # 获取当前chunk的du
                du_chunk = du[i_b,i_h, bos:eos, :]  # [chunk_size, V]

                # 获取当前chunk的beta
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]

                # 计算 dv_chunk = A_chunk @ du_chunk * beta_chunk
                # 步骤1: b_dv_beta = A_chunk @ du_chunk
                b_dv_beta = torch.matmul(A_chunk.T.to(torch.float64), du_chunk.to(torch.float64))  # [chunk_size, V]

                # 步骤2: dv_chunk = b_dv_beta * beta_chunk.unsqueeze(-1)
                dv_chunk = b_dv_beta.to(torch.float64) * beta_chunk[:, None].to(torch.float64)  # [chunk_size, D]

                # 存储结果
                dv[i_b,i_h, bos:eos, :] = dv_chunk.to(dv.dtype)

    return dv

def compute_dk_golden(
    A: torch.Tensor,      # [B, T, VH, chunk_size] - 每个chunk的A值
    dw: torch.Tensor,     # [B, T, KH, D]
    g: torch.Tensor,     # [B, VH, T]
    beta: torch.Tensor,   # [B, VH, T] - beta参数
    dA: torch.Tensor,      # [B, T, VH, chunk_size]
    k: torch.Tensor,     # [B, T, KH, D]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    KH: int,
    VH: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dk computation (变长序列)
    A的形状为 [B, T, VH, chunk_size]
    k/dw使用KH head, 其余使用VH head
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    dk = torch.zeros_like(k)
    ratio = VH // KH
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
        # 遍历所有batch
            for i_h in range(VH):
            # 遍历所有head
                i_h_k = i_h // ratio
                # 获取当前chunk的A向量
                # A形状: [B, T, VH, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,: eos - bos]  # [chunk_size, chunk_size]

                # 获取当前chunk的dw
                dw_chunk = dw[i_b,i_h_k, bos:eos, :]  # [chunk_size, D]

                # 获取当前chunk的beta,g
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]
                g_chunk = g[i_b,i_h, bos:eos]  # [chunk_size]
                g_exp_chunk = torch.exp(g_chunk.to(torch.float32))
                #   k________0
                k_chunk = k[i_b, i_h_k, bos:eos, : ]
                dA_chunk = dA[i_b,i_h, bos:eos,:eos-bos]  # [chunk_size, chunk_size]
                b_dk_beta = torch.matmul(dA_chunk.T.to(torch.float32), k_chunk.to(torch.float32))  # [chunk_size, D]
                #   k________1
                b_kt_beta = k_chunk.T.to(torch.float32) * beta_chunk.to(torch.float32)[None,: ]
                b_k_beta = k_chunk.to(torch.float32) * beta_chunk.to(torch.float32)[:, None]
                #   k________2
                # 步骤1: b_dk_beta_g = A_chunk @ dw_chunk
                b_dk_beta_g = torch.matmul(A_chunk.T.to(torch.float32), dw_chunk.to(torch.float32))  # [chunk_size, D]

                # 步骤2: dk_chunk = b_dk_beta_g * beta_chunk[:, None] * g_exp_chunk[:, None]
                dk_chunk = torch.matmul(dA_chunk.to(torch.float32), b_k_beta.to(k.dtype).to(torch.float32)).to(k.dtype).to(torch.float32)  # [chunk_size, D]
                dk_chunk = dk_chunk.to(k.dtype).to(torch.float32) + (b_dk_beta.to(k.dtype).to(torch.float32) * beta_chunk[:, None].to(torch.float32))
                dk_chunk = dk_chunk.to(k.dtype).to(torch.float32) + b_dk_beta_g.to(k.dtype).to(torch.float32) * (beta_chunk.to(torch.float32) * g_exp_chunk.to(torch.float32))[:,None]  # [chunk_size, D]
                # 存储结果
                dk[i_b,i_h_k, bos:eos, :] += dk_chunk
    return dk

def compute_dk_golden_high_precision(
    A: torch.Tensor,      # [B, T, VH, chunk_size] - 每个chunk的A值
    dw: torch.Tensor,     # [B, T, KH, D]
    g: torch.Tensor,     # [B, VH, T]
    beta: torch.Tensor,   # [B, VH, T] - beta参数
    dA: torch.Tensor,      # [B, T, VH, chunk_size]
    k: torch.Tensor,     # [B, T, KH, D]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    KH: int,
    VH: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dk computation (变长序列)
    A的形状为 [B, T, VH, chunk_size]
    k/dw使用KH head, 其余使用VH head
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    dk = torch.zeros_like(k).to(torch.float64)
    ratio = VH // KH
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
        # 遍历所有batch
            for i_h in range(VH):
            # 遍历所有head
                i_h_k = i_h // ratio
                # 获取当前chunk的A向量
                # A形状: [B, T, VH, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,: eos - bos]  # [chunk_size, chunk_size]

                # 获取当前chunk的dw
                dw_chunk = dw[i_b,i_h_k, bos:eos, :]  # [chunk_size, D]

                # 获取当前chunk的beta,g
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]
                g_chunk = g[i_b,i_h, bos:eos]  # [chunk_size]
                g_exp_chunk = torch.exp(g_chunk.to(torch.float64))
                #   k________0
                k_chunk = k[i_b, i_h_k, bos:eos, : ]
                dA_chunk = dA[i_b,i_h, bos:eos,:eos-bos]  # [chunk_size, chunk_size]
                b_dk_beta = torch.matmul(dA_chunk.T.to(torch.float64), k_chunk.to(torch.float64))  # [chunk_size, D]
                #   k________1
                b_kt_beta = k_chunk.T.to(torch.float64) * beta_chunk.to(torch.float64)[None,: ]
                b_k_beta = k_chunk.to(torch.float64) * beta_chunk.to(torch.float64)[:, None]
                #   k________2
                # 步骤1: b_dk_beta_g = A_chunk @ dw_chunk
                b_dk_beta_g = torch.matmul(A_chunk.T.to(torch.float64), dw_chunk.to(torch.float64))  # [chunk_size, D]

                # 步骤2: dk_chunk = b_dk_beta_g * beta_chunk[:, None] * g_exp_chunk[:, None]
                dk_chunk = torch.matmul(dA_chunk.to(torch.float64), b_k_beta.to(k.dtype).to(torch.float64)).to(k.dtype).to(torch.float64)  # [chunk_size, D]
                dk_chunk = dk_chunk.to(k.dtype).to(torch.float64) + (b_dk_beta.to(k.dtype).to(torch.float64) * beta_chunk[:, None].to(torch.float64))
                dk_chunk = dk_chunk.to(k.dtype).to(torch.float64) + b_dk_beta_g.to(k.dtype).to(torch.float64) * (beta_chunk.to(torch.float64) * g_exp_chunk.to(torch.float64))[:,None]  # [chunk_size, D]
                # 存储结果
                dk[i_b,i_h_k, bos:eos, :] += dk_chunk
    return dk

def compute_dg_golden(
    A: torch.Tensor,      # [B, T, VH, chunk_size] - 每个chunk的A值
    dw: torch.Tensor,     # [B, T, KH, D]
    g: torch.Tensor,     # [B, VH, T]
    beta: torch.Tensor,   # [B, VH, T] - beta参数
    dA: torch.Tensor,      # [B, T, VH, chunk_size]
    k: torch.Tensor,     # [B, T, KH, D]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    KH: int,
    VH: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dg computation (变长序列)
    A的形状为 [B, T, VH, chunk_size]
    k/dw使用KH head, 其余使用VH head
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    dg = torch.zeros_like(g)
    ratio = VH // KH
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
        # 遍历所有batch
            for i_h in range(VH):

            # 遍历所有head
                i_h_k = i_h // ratio
                # 获取当前chunk的A向量
                # A形状: [B, T, VH, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,: eos - bos]  # [chunk_size, chunk_size]

                # 获取当前chunk的dw
                dw_chunk = dw[i_b,i_h_k, bos:eos, :]  # [chunk_size, D]

                # 获取当前chunk的beta,g
                # beta形状: [B, VH, T]
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]
                g_chunk = g[i_b,i_h, bos:eos]  # [chunk_size]
                g_exp_chunk = torch.exp(g_chunk)

                #   g________0
                # 步骤1: b_dk_beta_g = A_chunk @ dw_chunk
                b_dk_beta_g = torch.matmul(A_chunk.T, dw_chunk)

                # 步骤2: b_dg += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None] * b_beta[:, None], 1)
                k_chunk = k[i_b, i_h_k, bos:eos, : ]
                b_kbg = k_chunk * (beta_chunk * g_exp_chunk)[:,None]
                #   g________1
                dA_chunk = dA[i_b,i_h, bos:eos,: eos - bos]  # [chunk_size, chunk_size]
                if k_chunk.size(0) == 1:
                    # 形状 [1, K] -> 计算外积等价于平方和
                    # 结果应为 [1, 1]
                    dot_val = torch.sum(k_chunk * k_chunk, dim=1, keepdim=True)  # [1, 1]
                    b_A = dot_val
                else:
                    # 正常走 matmul 路径
                    k_f32 = k_chunk.contiguous()
                    b_A = torch.matmul(k_f32, k_f32.T.contiguous())
                b_A = b_A * beta_chunk[:,None]
                b_dA_A = dA_chunk.T * b_A

                # test
                dg_chunk = torch.sum(b_dk_beta_g.to(dg.dtype) * b_kbg.to(dg.dtype), dim = 1)# [chunk_size]
                dg_chunk = dg_chunk.to(dg.dtype) +  (torch.sum(b_dA_A, dim = 1) - torch.sum(b_dA_A, dim = 0)).to(dg.dtype)
                # 存储结果
                dg[i_b,i_h, bos:eos] = dg_chunk.to(g.dtype)
    return dg

def compute_dg_golden_high_precision(
    A: torch.Tensor,      # [B, T, VH, chunk_size] - 每个chunk的A值
    dw: torch.Tensor,     # [B, T, KH, D]
    g: torch.Tensor,     # [B, VH, T]
    beta: torch.Tensor,   # [B, VH, T] - beta参数
    dA: torch.Tensor,      # [B, T, VH, chunk_size]
    k: torch.Tensor,     # [B, T, KH, D]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    KH: int,
    VH: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dg computation (变长序列)
    A的形状为 [B, T, VH, chunk_size]
    k/dw使用KH head, 其余使用VH head
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    dg = torch.zeros_like(g).to(torch.float64)
    ratio = VH // KH
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
        # 遍历所有batch
            for i_h in range(VH):

            # 遍历所有head
                i_h_k = i_h // ratio
                # 获取当前chunk的A向量
                # A形状: [B, T, VH, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,: eos - bos]  # [chunk_size, chunk_size]

                # 获取当前chunk的dw
                dw_chunk = dw[i_b,i_h_k, bos:eos, :]  # [chunk_size, D]

                # 获取当前chunk的beta,g
                # beta形状: [B, VH, T]
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]
                g_chunk = g[i_b,i_h, bos:eos]  # [chunk_size]
                g_exp_chunk = torch.exp(g_chunk.to(torch.float64))

                #   g________0
                # 步骤1: b_dk_beta_g = A_chunk @ dw_chunk
                b_dk_beta_g = torch.matmul(A_chunk.T.to(torch.float64), dw_chunk.to(torch.float64))

                # 步骤2: b_dg += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None] * b_beta[:, None], 1)
                k_chunk = k[i_b, i_h_k, bos:eos, : ]
                b_kbg = k_chunk.to(torch.float64) * (beta_chunk.to(torch.float64) * g_exp_chunk.to(torch.float64))[:,None]
                #   g________1
                dA_chunk = dA[i_b,i_h, bos:eos,: eos - bos]  # [chunk_size, chunk_size]
                if k_chunk.size(0) == 1:
                    # 形状 [1, K] -> 计算外积等价于平方和
                    # 结果应为 [1, 1]
                    dot_val = torch.sum(k_chunk.to(torch.float64) * k_chunk.to(torch.float64), dim=1, keepdim=True)  # [1, 1]
                    b_A = dot_val
                else:
                    # 正常走 matmul 路径
                    k_f32 = k_chunk.to(torch.float64).contiguous()
                    b_A = torch.matmul(k_f32, k_f32.T.contiguous())
                # b_A = torch.matmul(k_chunk.to(torch.float64).contiguous(), k_chunk.to(torch.float64).T).to(k.dtype).to(torch.float64).contiguous()  # [chunk_size, chunk_size]
                b_A = b_A.to(torch.float64) * beta_chunk[:,None].to(torch.float64)
                b_dA_A = dA_chunk.to(torch.float64).T * b_A.to(torch.float64)

                # test
                dg_chunk = torch.sum(b_dk_beta_g.to(k.dtype).to(torch.float64) * b_kbg.to(torch.float64), dim = 1)# [chunk_size]
                dg_chunk = dg_chunk.to(dg.dtype).to(torch.float64) +  (torch.sum(b_dA_A, dim = 1) - torch.sum(b_dA_A, dim = 0))
                # 存储结果
                dg[i_b,i_h, bos:eos] = dg_chunk.to(dg.dtype)
    return dg

def compute_dbeta_golden(
    A: torch.Tensor,      # [B, T, VH, chunkSize] - 每个chunk的A值
    dw: torch.Tensor,     # [B, T, KH, D]
    g: torch.Tensor,     # [B, VH, T]
    beta: torch.Tensor,   # [B, VH, T] - beta参数
    dA: torch.Tensor,      # [B, T, VH, chunkSize]
    k: torch.Tensor,     # [B, T, KH, D]
    v: torch.Tensor,      # [B, T, VH, D]
    du: torch.Tensor,     # [B, T, VH, D]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    KH: int,
    VH: int,
    T: int,
    D: int,
    chunkSize: int,  # chunkSize
    NT: int,  # T / chunkSize
) -> torch.Tensor:
    """
    CPU golden implementation for dbeta computation (变长序列)
    A的形状为 [B, T, VH, chunkSize]
    k/dw使用KH head, 其余使用VH head
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    dbeta = torch.zeros_like(beta)
    ratio = VH // KH
    for i_b in range(B):
        for idx in range(NT):
        # 遍历所有batch
            bos,eos = get_bos_eos(idx, T, chunkSize, cu_seqlens, chunk_indices)
            for i_h in range(VH):
            # 遍历所有head
                i_h_k = i_h // ratio
                # 获取当前chunk的A向量
                # A形状: [B, T, VH, chunkSize]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,: eos - bos]  # [chunkSize, chunkSize]
                v_chunk = v[i_b,i_h, bos:eos,:]  # [chunkSize, V]

                # 获取当前chunk的dw
                dw_chunk = dw[i_b,i_h_k, bos:eos, :]  # [chunkSize, D]
                du_chunk = du[i_b,i_h, bos:eos, :]  # [chunkSize, D]


                # 获取当前chunk的beta,g
                # beta形状: [B, VH, T]
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunkSize]
                g_chunk = g[i_b,i_h, bos:eos]  # [chunkSize]
                g_exp_chunk = torch.exp(g_chunk)
                k_chunk = k[i_b, i_h_k, bos:eos, : ]
                #   beta________0
                # 步骤1: b_dk_beta_g = A_chunk @ dw_chunk
                b_dk_beta_g = torch.matmul(A_chunk.T, dw_chunk)

                # 步骤2: b_dbeta += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None], 1)
                tmp = b_dk_beta_g * k_chunk * g_exp_chunk[:, None] # [chunkSize, D]
                #   beta________1

                b_dv_beta = torch.matmul(A_chunk.T, du_chunk)  # [chunkSize, V]
                #   beta________2
                dA_chunk = dA[i_b,i_h, bos:eos,: eos - bos]  # [chunkSize, chunkSize]
                b_dk_beta = torch.matmul(dA_chunk.T, k_chunk)  # [chunkSize, V]
                tmp = b_dk_beta.to(k.dtype) * k_chunk
                # test
                dbeta_chunk = torch.sum(tmp, 1)
                tmp = b_dk_beta_g.to(k.dtype) * k_chunk * g_exp_chunk[:, None] # [chunkSize, D]
                dbeta_chunk = dbeta_chunk.to(k.dtype) + torch.sum(tmp, dim = 1)# [chunkSize]
                dbeta_chunk = dbeta_chunk.to(k.dtype) + torch.sum(b_dv_beta.to(k.dtype) * v_chunk, 1)
                # 存储结果
                dbeta[i_b,i_h, bos:eos] = dbeta_chunk
    return dbeta

def compute_dbeta_golden_high_precision(
    A: torch.Tensor,      # [B, T, VH, chunkSize] - 每个chunk的A值
    dw: torch.Tensor,     # [B, T, KH, D]
    g: torch.Tensor,     # [B, VH, T]
    beta: torch.Tensor,   # [B, VH, T] - beta参数
    dA: torch.Tensor,      # [B, T, VH, chunkSize]
    k: torch.Tensor,     # [B, T, KH, D]
    v: torch.Tensor,      # [B, T, VH, D]
    du: torch.Tensor,     # [B, T, VH, D]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    KH: int,
    VH: int,
    T: int,
    D: int,
    chunkSize: int,  # chunkSize
    NT: int,  # T / chunkSize
) -> torch.Tensor:
    """
    CPU golden implementation for dbeta computation (变长序列)
    A的形状为 [B, T, VH, chunkSize]
    k/dw使用KH head, 其余使用VH head
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    dbeta = torch.zeros_like(beta).to(torch.float64)
    ratio = VH // KH
    for i_b in range(B):
        for idx in range(NT):
        # 遍历所有batch
            bos,eos = get_bos_eos(idx, T, chunkSize, cu_seqlens, chunk_indices)
            for i_h in range(VH):
            # 遍历所有head
                i_h_k = i_h // ratio
                # 获取当前chunk的A向量
                # A形状: [B, T, VH, chunkSize]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,: eos - bos]  # [chunkSize, chunkSize]
                v_chunk = v[i_b,i_h, bos:eos,:]  # [chunkSize, V]

                # 获取当前chunk的dw
                dw_chunk = dw[i_b,i_h_k, bos:eos, :]  # [chunkSize, D]
                du_chunk = du[i_b,i_h, bos:eos, :]  # [chunkSize, D]


                # 获取当前chunk的beta,g
                # beta形状: [B, VH, T]
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunkSize]
                g_chunk = g[i_b,i_h, bos:eos]  # [chunkSize]
                g_exp_chunk = torch.exp(g_chunk.to(torch.float64))
                k_chunk = k[i_b, i_h_k, bos:eos, : ]
                #   beta________0
                # 步骤1: b_dk_beta_g = A_chunk @ dw_chunk
                b_dk_beta_g = torch.matmul(A_chunk.T.to(torch.float64), dw_chunk.to(torch.float64))

                # 步骤2: b_dbeta += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None], 1)
                tmp = b_dk_beta_g * k_chunk.to(torch.float64) * g_exp_chunk.to(torch.float64)[:, None] # [chunkSize, D]
                #   beta________1

                b_dv_beta = torch.matmul(A_chunk.T.to(torch.float64), du_chunk.to(torch.float64))  # [chunkSize, V]
                #   beta________2
                dA_chunk = dA[i_b,i_h, bos:eos,: eos - bos]  # [chunkSize, chunkSize]
                b_dk_beta = torch.matmul(dA_chunk.T.to(torch.float64), k_chunk.to(torch.float64))  # [chunkSize, V]
                tmp = b_dk_beta.to(k.dtype).to(torch.float64) * k_chunk
                # test
                dbeta_chunk = torch.sum(tmp, 1).to(torch.float64)
                tmp = b_dk_beta_g.to(k.dtype).to(torch.float64) * k_chunk.to(torch.float64) * g_exp_chunk.to(torch.float64)[:, None] # [chunkSize, D]
                dbeta_chunk = dbeta_chunk.to(k.dtype).to(torch.float64) + torch.sum(tmp, dim = 1)# [chunkSize]
                dbeta_chunk = dbeta_chunk.to(k.dtype).to(torch.float64) + torch.sum(b_dv_beta.to(k.dtype).to(torch.float64) * v_chunk, 1)
                # 存储结果
                dbeta[i_b,i_h, bos:eos] = dbeta_chunk
    return dbeta

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def cdiv(a: torch.LongTensor
    , b : int):
    torch.empty
    return (a + b - 1) // b

def prepare_cu_seqlens(T: int, L: int = 32, seed: int = 42) -> list[int]:
    """
    直接生成一个长度为 L 的 cu_seqlens 列表 (list[int])：
      - 以 0 开头，以 T 结尾
      - 严格单调递增，无重复
      - 所有值在 [0, T] 范围内
      - 可复现（固定随机种子）
      
    此函数完全避开 torch.Tensor，直接返回 Python 原生 list，
    完美适配 npu 算子对 'Optional[list[int]]' 的类型要求。

    Args:
        T (int): 最大值（总 token 数）
        L (int): 输出列表的长度（必须满足 2 <= L <= T + 1）
        seed (int): 随机种子，默认 42

    Returns:
        list[int]: 例如 [0, 15, 32, ..., T]
    """
    if T < 1:
        raise ValueError("T must be at least 1.")
    if L < 2 or L > T + 1:
        raise ValueError(f"L must satisfy 2 <= L <= T + 1 (got L={L}, T={T}).")

    # 固定随机种子 (使用 Python 标准库)
    random.seed(seed)

    if L == 2:
        # 最简单情况：[0, T]
        return [0, T]

    # 需要在 (0, T) 开区间内选择 L - 2 个不重复的整数作为中间点
    # 候选集合：1, 2, ..., T-1
    # random.sample 直接返回不重复的列表，无需担心重复
    middle_points = random.sample(range(1, T), L - 2)
    
    # 必须排序以保证单调递增
    middle_points.sort()

    # 拼接：0 + 中间点 + T
    # 这里的 0, middle_points 中的元素, T 都是纯 Python int
    cu_seqlens = [0] + middle_points + [T]

    return cu_seqlens

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

def test_prepare_wy_repr_bwd_full(
    B: int,
    KH: int,
    VH: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    ktype,
    btype,
    cu_seqlens = None,
    chunk_indices = None,
    seed: int = 0,
):
    """
    生成随机输入张量，保存到文件，并调用 NPU 的 WY 表示反向算子。

    参数:
        B (int): Batch size
        KH (int): Key head number
        VH (int): Value head number
        T (int): Sequence length
        K (int): Key dimension
        V (int): Value dimension
        chunk_size (int): Chunk size (通常为 T 的因数)
        seed (int): 随机种子，默认为 0
        save_path (str): 保存 pickle 文件的路径，默认为 'data.pkl'

    返回:
        tuple: (dk, dv, dbeta, dg) —— 反向传播的梯度结果（在 NPU 上）
    """
    torch.manual_seed(seed)
    if not hasattr(test_prepare_wy_repr_bwd_full, "call_count"):
        test_prepare_wy_repr_bwd_full.call_count = 1
    else:
        test_prepare_wy_repr_bwd_full.call_count += 1

    # 生成随机张量（float16）
    k = torch.rand(B, KH, T, K, dtype=ktype)
    v = torch.rand(B, VH, T, V, dtype=ktype)
    beta = torch.rand(B, VH, T, dtype=btype)
    A = torch.rand(B, VH, T, chunk_size, dtype=ktype)
    dA = torch.rand(B, VH, T, chunk_size, dtype=ktype)
    dw = torch.rand(B, KH, T, K, dtype=ktype)
    du = torch.rand(B, VH, T, V, dtype=ktype)
    g = torch.rand(B, VH, T, dtype=btype)

    # 将张量移到 NPU 并调用反向算子
    if chunk_indices != None:
        dk, dv, dbeta, dg = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
            k.npu(),
            v.npu(),
            beta.npu(),
            A.npu(),
            dA.npu(),
            dw.npu(),
            du.npu(),
            g.npu(),
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size
        )
    else:
        dk, dv, dbeta, dg = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
            k.npu(),
            v.npu(),
            beta.npu(),
            A.npu(),
            dA.npu(),
            dw.npu(),
            du.npu(),
            g.npu(),
            cu_seqlens=None,
            chunk_indices=None,
            chunk_size=chunk_size
        )
    if chunk_indices!=None:
        NT = len(chunk_indices) // 2
    else:
        NT = (T + chunk_size - 1) // chunk_size
    
    try:
        import ct
        cpu_dv = compute_dv_golden(A, du, beta, cu_seqlens, chunk_indices, B, VH, T, K, chunk_size, NT)
        cpu_dv_high_precision = compute_dv_golden_high_precision(A, du, beta, cu_seqlens, chunk_indices, B, VH, T, K, chunk_size, NT)
        ct.dual(dv.cpu(), cpu_dv_high_precision, cpu_dv)
        cpu_dk = compute_dk_golden(A, dw, g, beta, dA,k, cu_seqlens, chunk_indices, B, KH, VH, T, K, chunk_size, NT)
        cpu_dk_high_precision = compute_dk_golden_high_precision(A, dw, g, beta, dA,k, cu_seqlens, chunk_indices, B, KH, VH, T, K, chunk_size, NT)
        ct.dual(dk.cpu(), cpu_dk_high_precision, cpu_dk)
        cpu_dg = compute_dg_golden(A, dw, g, beta, dA,k, cu_seqlens, chunk_indices, B, KH, VH, T, K, chunk_size, NT)
        cpu_dg_high_precision = compute_dg_golden_high_precision(A, dw, g, beta, dA,k, cu_seqlens, chunk_indices, B, KH, VH, T, K, chunk_size, NT)
        ct.dual(dg.cpu(), cpu_dg_high_precision, cpu_dg)
        cpu_dbeta = compute_dbeta_golden(A, dw, g, beta, dA,k,v,du, cu_seqlens, chunk_indices, B, KH, VH, T, K, chunk_size, NT)
        cpu_dbeta_high_precision = compute_dbeta_golden_high_precision(A, dw, g, beta, dA,k,v,du, cu_seqlens, chunk_indices, B, KH, VH, T, K, chunk_size, NT)
        ct.dual(dbeta.cpu(), cpu_dbeta_high_precision, cpu_dbeta)
    finally:
        pass
    
    print(f"test_prepare_wy_repr_bwd_full 被调用了第 {test_prepare_wy_repr_bwd_full.call_count} 次")
    return dk, dv, dbeta, dg

if __name__ == "__main__":
    # #F1
    # test_prepare_wy_repr_bwd_full(B = 64, KH = 8, VH = 8, T = 1024, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16)
    # #F2
    # test_prepare_wy_repr_bwd_full(B = 32, KH = 16, VH = 16, T = 2048, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F3
    # test_prepare_wy_repr_bwd_full(B = 16, KH = 32, VH = 32, T = 4096, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float32)
    # #F4
    # test_prepare_wy_repr_bwd_full(B = 8, KH = 32, VH = 32, T = 8192, K = 128, V = 128, chunk_size = 128, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F5
    # test_prepare_wy_repr_bwd_full(B = 128, KH = 4, VH = 4, T = 1024, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16)
    # #F6
    # test_prepare_wy_repr_bwd_full(B = 64, KH = 8, VH = 8, T = 2048, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.float32)
    # #F7
    # test_prepare_wy_repr_bwd_full(B = 32, KH = 16, VH = 16, T = 4096, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float16)
    # #F8
    # test_prepare_wy_repr_bwd_full(B = 16, KH = 32, VH = 32, T = 8192, K = 128, V = 128, chunk_size = 128, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F9
    # test_prepare_wy_repr_bwd_full(B = 64, KH = 8, VH = 8, T = 4096, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float16)
    # #F10
    # test_prepare_wy_repr_bwd_full(B = 32, KH = 16, VH = 16, T = 8192, K = 128, V = 128, chunk_size = 128, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F11
    # test_prepare_wy_repr_bwd_full(B = 16, KH = 32, VH = 32, T = 16384, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float32)
    # #F12
    # test_prepare_wy_repr_bwd_full(B = 8, KH = 32, VH = 32, T = 32768, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F13
    # test_prepare_wy_repr_bwd_full(B = 64, KH = 8, VH = 8, T = 1024, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16)
    # #F14
    # test_prepare_wy_repr_bwd_full(B = 32, KH = 16, VH = 16, T = 2048, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F15
    # test_prepare_wy_repr_bwd_full(B = 16, KH = 32, VH = 32, T = 4096, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float32)
    # #F16
    # test_prepare_wy_repr_bwd_full(B = 8, KH = 32, VH = 32, T = 8192, K = 128, V = 128, chunk_size = 128, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F17
    # test_prepare_wy_repr_bwd_full(B = 64, KH = 8, VH = 8, T = 2048, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F18
    # test_prepare_wy_repr_bwd_full(B = 32, KH = 16, VH = 16, T = 4096, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float16)
    #L1
    cu_seqlens = prepare_cu_seqlens(T = 65536, L = 64)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    test_prepare_wy_repr_bwd_full(B = 1, KH = 32, VH = 32, T = 65536, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # #L2
    # cu_seqlens = prepare_cu_seqlens(T = 65536, L = 33)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_prepare_wy_repr_bwd_full(B = 1, KH = 16, VH = 16, T = 65536, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # #L3
    # cu_seqlens = prepare_cu_seqlens(T = 131072, L = 333)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_prepare_wy_repr_bwd_full(B = 1, KH = 8, VH = 8, T = 131072, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # # L4
    # cu_seqlens = prepare_cu_seqlens(T = 262144, L = 567)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_prepare_wy_repr_bwd_full(B = 1, KH = 4, VH = 4, T = 262144, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float32, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # #L5
    # cu_seqlens = prepare_cu_seqlens(T = 32768, L = 7)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_prepare_wy_repr_bwd_full(B = 1, KH = 16, VH = 16, T = 32768, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # #L6
    # cu_seqlens = prepare_cu_seqlens(T = 65536, L = 25)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_prepare_wy_repr_bwd_full(B = 1, KH = 8, VH = 8, T = 65536, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    
    # # 二阶段泛化用例
    # # L7
    # cu_seqlens = prepare_cu_seqlens(T = 16384, L = 128)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_prepare_wy_repr_bwd_full(B = 1, VH = 32, KH = 16, T = 16384, K = 128, V = 256, chunk_size = 64, ktype=torch.float16, btype=torch.float32, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # # L8
    # cu_seqlens = prepare_cu_seqlens(T = 16384, L = 2)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_prepare_wy_repr_bwd_full(B = 1, VH = 63, KH = 21, T = 16384, K = 128, V = 256, chunk_size = 64, ktype=torch.bfloat16, btype=torch.float32, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # # L9
    # cu_seqlens = prepare_cu_seqlens(T = 65536, L = 172)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 128)
    # test_prepare_wy_repr_bwd_full(B = 1, VH = 32, KH = 8, T = 65536, K = 128, V = 256, chunk_size = 128, ktype=torch.bfloat16, btype=torch.float32, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # # L10
    # cu_seqlens = prepare_cu_seqlens(T = 65536, L = 668)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_prepare_wy_repr_bwd_full(B = 1, VH = 32, KH = 16, T = 65536, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float32, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # # L11
    # cu_seqlens = prepare_cu_seqlens(T = 65536, L = 17)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 128)
    # test_prepare_wy_repr_bwd_full(B = 1, VH = 32, KH = 4, T = 65536, K = 128, V = 128, chunk_size = 128, ktype=torch.bfloat16, btype=torch.float32, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # # L12
    # cu_seqlens = prepare_cu_seqlens(T = 262144, L = 32)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_prepare_wy_repr_bwd_full(B = 1, VH = 64, KH = 2, T = 262144, K = 128, V = 256, chunk_size = 64, ktype=torch.float16, btype=torch.float32, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # # F19
    # test_prepare_wy_repr_bwd_full(B = 1, VH = 32, KH = 16, T = 4096, K = 128, V = 256, chunk_size = 64, ktype=torch.float16, btype=torch.float32)
    # # F20
    # test_prepare_wy_repr_bwd_full(B = 16, VH = 63, KH = 21, T = 2048, K = 128, V = 256, chunk_size = 64, ktype=torch.bfloat16, btype=torch.float32)
    # # F21
    # test_prepare_wy_repr_bwd_full(B = 711, VH = 32, KH = 4, T = 196, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float32)
    # # F22
    # test_prepare_wy_repr_bwd_full(B = 176, VH = 64, KH = 2, T = 24, K = 128, V = 256, chunk_size = 64, ktype=torch.bfloat16, btype=torch.float32)
