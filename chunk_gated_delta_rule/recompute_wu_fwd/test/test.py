import torch
import torch_npu
from typing import Optional
import pickle
import math
import ct
import random
torch.npu.utils.set_device(6)

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

def compute_w_golden(
    k: torch.Tensor,     # [B, T, H, D]
    v: torch.Tensor,     # [B, T, H, D]
    beta: torch.Tensor,   # [B, H, T]
    A: torch.Tensor,      # [B, T, H, chunk_size]
    g: torch.Tensor,      # [B, T, H]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    H: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dv computation (变长序列)
    A的形状为 [B, T, H, chunk_size]
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    # 初始化dv，形状与du相同 [B, T, H, D]
    w = torch.zeros_like(k)
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            # print(bos, eos, eos-bos)
        # 遍历所有batch
            for i_h in range(H):
                # print("hello?")
            # 遍历所有head 
                # 获取当前chunk的A向量
                # A形状: [B, T, H, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b,i_h, bos:eos,:eos - bos]
                k_chunk = k[i_b,i_h, bos:eos,:]
                
                # 获取当前chunk的beta
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]
                g_chunk = g[i_b,i_h, bos:eos]  # [chunk_size]
                g_exp_chunk = torch.exp(g_chunk.to(torch.float32))
                beta_g_exp_chunk = beta_chunk.to(torch.float32) * g_exp_chunk.to(torch.float32)
                kbgexp_chunk = k_chunk * beta_g_exp_chunk[:,None]

                # 计算 dv_chunk = A_chunk @ du_chunk * beta_chunk
                # 步骤1: b_dv_beta = A_chunk @ du_chunk
                # print(A_chunk.size())
                b_w = torch.matmul(A_chunk.to(torch.float32), kbgexp_chunk.to(torch.float32))  # [chunk_size, V]
                

                # 存储结果
                w[i_b,i_h, bos:eos, :] = b_w.to(w.dtype)
    
    return w


def compute_u_golden(
    k: torch.Tensor,     # [B, T, H, D]
    v: torch.Tensor,     # [B, T, H, D]
    beta: torch.Tensor,   # [B, H, T]
    A: torch.Tensor,      # [B, T, H, chunk_size]
    g: torch.Tensor,      # [B, T, H]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    B: int,
    H: int,
    T: int,
    D: int,
    chunk_size: int,  # chunk_size
    NT: int,  # T / chunk_size
) -> torch.Tensor:
    """
    CPU golden implementation for dv computation (变长序列)
    A的形状为 [B, T, H, chunk_size]
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_indices
    3. 计算该chunk内的dv: dv_chunk = A_chunk @ du_chunk * beta_chunk
    """
    u = torch.zeros_like(v)
    for i_b in range(B):
        for idx in range(NT):
            bos,eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
        # 遍历所有batch
            for i_h in range(H):
            # 遍历所有head 
                # 获取当前chunk的A向量
                # A形状: [B, T, H, chunk_size]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                u_chunk = u[i_b,i_h, bos:eos,:]
                A_chunk = A[i_b,i_h, bos:eos,: eos - bos]  # [chunk_size, chunk_size]
                
                # 获取当前chunk的dw
                v_chunk = v[i_b,i_h, bos:eos, :]  # [chunk_size, D]
           
                # 获取当前chunk的beta,g
                beta_chunk = beta[i_b,i_h, bos:eos]  # [chunk_size]
                vb_chunk = v_chunk.to(torch.float32) * beta_chunk[:,None].to(torch.float32)
                u_chunk = torch.matmul(A_chunk.to(torch.float32), vb_chunk.to(torch.float32))  # [chunk_size, V]

                # 存储结果
                u[i_b,i_h, bos:eos, :] = u_chunk.to(u.dtype)
    return u


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

def test_recompute_wu_fwd(
    B: int,
    H: int,
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
        H (int): Head number
        T (int): Sequence length
        K (int): Key dimension
        V (int): Value dimension
        chunk_size (int): Chunk size (通常为 T 的因数)
        seed (int): 随机种子，默认为 0
        save_path (str): 保存 pickle 文件的路径，默认为 'data.pkl'

    返回:
        tuple: (dw, du) —— 反向传播的梯度结果（在 NPU 上）
    """
    # torch.manual_seed(42)
    if not hasattr(test_recompute_wu_fwd, "call_count"):
        test_recompute_wu_fwd.call_count = 1
    else:
        test_recompute_wu_fwd.call_count += 1

    # 生成随机张量（float16）
    k = torch.ones(B, H, T, K, dtype=ktype)
    v = torch.ones(B, H, T, V, dtype=ktype)
    beta = torch.ones(B, H, T, dtype=btype)
    A = torch.ones(B, H, T, chunk_size, dtype=ktype)
    g = torch.ones(B, H, T, dtype=btype)

    if chunk_indices!=None:
        NT = len(chunk_indices) // 2
    else:
        NT = (T + chunk_size - 1) // chunk_size

    # 将张量移到 NPU 并调用反向算子
    if chunk_indices != None:
        w, u = torch_npu.npu_recompute_w_u_fwd(
            k.npu(),
            v.npu(),
            beta.npu(),
            A.npu(),
            g.npu(),
            None,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size
        )
    else:
        w, u = torch_npu.npu_recompute_w_u_fwd(
            k.npu(),
            v.npu(),
            beta.npu(),
            A.npu(),
            g.npu(),
            None,
            cu_seqlens=None,
            chunk_indices=None,
            chunk_size=chunk_size
        )
    # cpu_w = compute_w_golden(k, v, beta, A, g, cu_seqlens, chunk_indices, B, H, T, K, chunk_size, NT)
    # ct.isclose(w.cpu(), cpu_w.cpu(), diff_thd=0.1)
    # torch.save(cpu_dv, "cpu_dv.pt")
    
    # cpu_u = compute_u_golden(k, v, beta, A, g, cu_seqlens, chunk_indices, B, H, T, K, chunk_size, NT)
    # ct.isclose(u.cpu(), cpu_u.cpu(), diff_thd=0.1)
    # torch.save(cpu_dk, "cpu_dk.pt")

    
    print(f"test_recompute_wu_fwd 被调用了第 {test_recompute_wu_fwd.call_count} 次")
    return w, u

if __name__ == "__main__":
    # test_recompute_wu_fwd(B = 1, H = 4, T = 2048, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16)
    # #F1
    # test_recompute_wu_fwd(B = 64, H = 8, T = 1024, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16)
    # #F2
    # test_recompute_wu_fwd(B = 32, H = 16, T = 2048, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F3
    # test_recompute_wu_fwd(B = 16, H = 32, T = 4096, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float32)
    # #F4
    # test_recompute_wu_fwd(B = 8, H = 32, T = 8192, K = 128, V = 128, chunk_size = 128, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F5
    # test_recompute_wu_fwd(B = 128, H = 4, T = 1024, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16)
    # #F6
    # test_recompute_wu_fwd(B = 64, H = 8, T = 2048, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.float32)
    # #F7
    # test_recompute_wu_fwd(B = 32, H = 16, T = 4096, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float16)
    # #F8    
    # test_recompute_wu_fwd(B = 16, H = 32, T = 8192, K = 128, V = 128, chunk_size = 128, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F9
    # test_recompute_wu_fwd(B = 64, H = 8, T = 4096, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float16)
    # #F10
    # test_recompute_wu_fwd(B = 32, H = 16, T = 8192, K = 128, V = 128, chunk_size = 128, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F11
    # test_recompute_wu_fwd(B = 16, H = 32, T = 16384, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float32)
    # #F12
    # test_recompute_wu_fwd(B = 8, H = 32, T = 32768, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F13
    # test_recompute_wu_fwd(B = 64, H = 8, T = 1024, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16)
    # #F14
    # test_recompute_wu_fwd(B = 32, H = 16, T = 2048, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F15
    # test_recompute_wu_fwd(B = 16, H = 32, T = 4096, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float32)
    # #F16
    # test_recompute_wu_fwd(B = 8, H = 32, T = 8192, K = 128, V = 128, chunk_size = 128, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F17
    # test_recompute_wu_fwd(B = 64, H = 8, T = 2048, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16)
    # #F18
    # test_recompute_wu_fwd(B = 32, H = 16, T = 4096, K = 128, V = 128, chunk_size = 128, ktype=torch.float16, btype=torch.float16)
    # #L1
    cu_seqlens = prepare_cu_seqlens(T = 65536, L = 64)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    print(cu_seqlens)
    print(chunk_indices)
    test_recompute_wu_fwd(B = 1, H = 32, T = 65536, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # #L2
    # cu_seqlens = prepare_cu_seqlens(T = 65536, L = 33)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_recompute_wu_fwd(B = 1, H = 16, T = 65536, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # #L3
    # cu_seqlens = prepare_cu_seqlens(T = 131072, L = 333)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_recompute_wu_fwd(B = 1, H = 8, T = 131072, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # # L4
    # cu_seqlens = prepare_cu_seqlens(T = 262144, L = 567)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_recompute_wu_fwd(B = 1, H = 4, T = 262144, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float32, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # #L5
    # cu_seqlens = prepare_cu_seqlens(T = 32768, L = 7)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_recompute_wu_fwd(B = 1, H = 16, T = 32768, K = 128, V = 128, chunk_size = 64, ktype=torch.bfloat16, btype=torch.bfloat16, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
    # #L6
    # cu_seqlens = prepare_cu_seqlens(T = 65536, L = 25)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size = 64)
    # test_recompute_wu_fwd(B = 1, H = 8, T = 65536, K = 128, V = 128, chunk_size = 64, ktype=torch.float16, btype=torch.float16, cu_seqlens = cu_seqlens, chunk_indices=chunk_indices)
