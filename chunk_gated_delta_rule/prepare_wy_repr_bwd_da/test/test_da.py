import torch
import torch_npu
from typing import Optional
import math
import ct
import random

torch.npu.utils.set_device(1)

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

def create_incremental_tensor(shape, dtype=torch.float16, start=1, step=1):
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    tensor = torch.arange(
        start, 
        start + total_elements * step, 
        step, 
        dtype=dtype
    ).reshape(shape)
    return tensor

def bool_matrix_lower_tri_to_uint8(chunk_size):
    # 创建下三角矩阵（下三角不包括对角线为1，上三角包括对角线为0）
    bool_matrix = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=-1)
    bool_matrix = ~bool_matrix # 和FA含义一致，0代表保留，1代表屏蔽
    print(f"==== bool_matrix.shape = {bool_matrix.shape} ")
    print("==== bool_matrix ====")
    print(bool_matrix)
    # 将bool矩阵转换为uint8 (0或1)
    uint8_matrix = bool_matrix.to(torch.uint8)
    print(f"==== uint8_matrix.shape = {uint8_matrix.shape} ")
    print("==== uint8_matrix ====")
    print(uint8_matrix)
    # 重塑为 (chunk_size, chunk_size//8, 8) 以便每8个bit打包
    reshaped = uint8_matrix.reshape(chunk_size, chunk_size // 8, 8)
    # 将每8个bit打包成一个uint8
    # bit0 * 1 + bit1 * 2 + bit2 * 4 + ... + bit7 * 128
    
    powers = torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.uint8)
    packed = (reshaped * powers).sum(dim=-1).to(torch.uint8)
    return packed

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
    return bos, eos

def compute_dA_cpu(
    A: torch.Tensor,      # [B, H, T, BT] - 每个chunk的A值
    dw: torch.Tensor,     # [B, H, T, K]
    g: torch.Tensor,     # [B, H, T]
    beta: torch.Tensor,   # [B, H, T] - beta参数
    k: torch.Tensor,     # [B, H, T, K]
    v: torch.Tensor,      # [B, H, T, V]
    du: torch.Tensor,     # [B, H, T, V]
    chunk_indices: list[int],  # 扁平化的chunk索引 [seq_idx0, chunk_idx0, seq_idx1, chunk_idx1, ...]
    cu_seqlens: list[int],  # 累积序列长度
    B: int,
    H: int,
    T: int,
    D: int,
    BT: int,  # BT
    NT: int,  # T / BT
) -> torch.Tensor:
    """
    CPU golden implementation for dv computation (变长序列)
    A的形状为 [B, H, T, BT]
    算法:
    1. 对于每个chunk (由chunk_indices指定)
    2. 获取对应的seq_idx, chunk_idx
    3. 计算该chunk内的dA
    """
    dA = torch.zeros_like(A)
    IS_VARLEN = cu_seqlens is not None
    for idx in range(NT):
        bos, eos = get_bos_eos(idx, T, BT, cu_seqlens, chunk_indices)
        chunk_len = eos - bos
        if IS_VARLEN:
            # 从chunk_indices获取batch索引和chunk索引
            # chunk_indices为扁平化列表: [seq_idx0, chunk_idx0, seq_idx1, chunk_idx1, ...]
            seq_idx = chunk_indices[idx * 2]   # 等价于序列号i_n = tl.load(chunk_indices + idx * 2).to(tl.int32)
            chunk_idx = chunk_indices[idx * 2 + 1]
            i_t = chunk_idx
            T = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        else:
            i_t = idx

        # 并行步骤1~3：m_A
        # 创建因果掩码
        # i_t = tl.load(chunk_indices + idx * 2 + 1).to(tl.int32)
        # o_t = i_t * BT + tl.arange(0, BT)
        # m_t = o_t < T
        # m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)

        o_t = i_t * BT + torch.arange(0, BT, dtype=torch.int32)
        m_t = o_t < T
        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
        # print("==== m_A.shape = ", m_A.shape)
        # print("==== m_A ====")
        # print(m_A)

        for i_b in range(B):
        # 遍历所有batch
            for i_h in range(H):
            # 遍历所有head 
                # 获取当前chunk的dw, k, beta, g
                dw_chunk = dw[i_b, i_h, bos : eos, :]  # [BT, K]
                k_chunk = k[i_b, i_h, bos : eos, :]  # [BT, K]
                # beta形状: [B, H, T]
                beta_chunk = beta[i_b, i_h, bos : eos]  # [BT]
                # g形状: [B, H, T]
                g_chunk = g[i_b, i_h, bos : eos]  # [BT]

                # 获取当前chunk的du, v
                du_chunk = du[i_b, i_h, bos : eos, :]  # [BT, V]
                v_chunk = v[i_b, i_h, bos : eos, :]  # [BT, V]

                # 获取当前chunk的A向量
                # A形状: [B, H, T, BT]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b, i_h, bos : eos, : chunk_len]  # [BT, BT]

                g_exp_chunk = torch.exp(g_chunk.to(torch.float32))

                # 步骤1: b_dA_1
                # b_dA_1 = dw_chunk @ b_k_beta_g.T
                b_k_beta_g = k_chunk.to(torch.float32) * (beta_chunk.to(torch.float32) * g_exp_chunk.to(torch.float32))[:, None]
                if chunk_len == 1:
                    b_dA_1 = torch.sum(dw_chunk.to(torch.float32) * b_k_beta_g.to(torch.float32)).reshape(chunk_len, chunk_len)
                else:
                    b_dA_1 = torch.matmul(dw_chunk.to(torch.float32), b_k_beta_g.T.to(torch.float32))

                # 步骤2: b_dA_2
                # b_dA_2 = du_chunk @ b_v_beta.T
                b_v_beta = v_chunk.to(torch.float32) * beta_chunk.to(torch.float32)[:, None]
                if chunk_len == 1:
                    b_dA_2 = torch.sum(du_chunk.to(torch.float32) * b_v_beta.to(torch.float32)).reshape(chunk_len, chunk_len)
                else:
                    b_dA_2 = torch.matmul(du_chunk.to(torch.float32), b_v_beta.T.to(torch.float32))

                # # 步骤3：b_dA_3
                b_dA_3 = b_dA_1 + b_dA_2

                # 步骤4：b_dA_4
                # b_dA_4 = tl.where(m_A, b_dA_3, 0)
                b_dA_4 = torch.where(m_A[:chunk_len, :chunk_len], b_dA_3.to(torch.float32), 0.0)

                # 步骤5：b_dA_5
                # b_dA_5 = b_dA_4 @ A_chunk.T
                if chunk_len == 1:
                    b_dA_5 = torch.sum(b_dA_4.to(torch.float32) * A_chunk.T.to(torch.float32)).reshape(chunk_len, chunk_len)
                else:
                    b_dA_5 = torch.matmul(b_dA_4.to(torch.float32), A_chunk.T.to(torch.float32))

                # 步骤6：b_dA_6
                # b_dA_6 = A_chunk.T @ b_dA_5
                if chunk_len == 1:
                    b_dA_6 = torch.sum(A_chunk.T.to(torch.float32) * b_dA_5.to(torch.float32)).reshape(chunk_len, chunk_len)
                else:
                    b_dA_6 = torch.matmul(A_chunk.T.to(torch.float32), b_dA_5.to(torch.float32))

                # 并行步骤1~6：b_g_sub_exp
                b_g_sub_exp = torch.exp(g_chunk.to(torch.float32)[:, None] - g_chunk.to(torch.float32)[None, :]) 
                
                # 步骤7：b_dA_7
                # b_dA_7 = -b_dA_6.to(torch.float32) * (g_chunk.to(torch.float32)[:, None] - g_chunk.to(torch.float32)[None, :]) #b_g_sub_exp.to(torch.float32)
                b_dA_7 = -b_dA_6.to(torch.float32) * b_g_sub_exp.to(torch.float32)

                # 步骤8：b_dA
                # b_dA = tl.where(m_A, b_dA_7, 0)
                b_dA = torch.where(m_A[:chunk_len, :chunk_len], b_dA_7.to(torch.float32), 0.0)

                # 存储结果
                dA[i_b, i_h, bos : eos, : chunk_len] = b_dA.T.to(A.dtype)

    return dA

def test_variable():
    B = 1
    T = 16384
    H = 32
    K = 128
    V = 128
    BT=chunk_size=64

    k = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== k.shape = {k.shape} ")
    v = create_tensor((B, H, T, V), dtype=torch.float16)
    print(f"==== v.shape = {v.shape} ")
    beta = create_tensor((B, H, T), dtype=torch.float)
    print(f"==== beta.shape = {beta.shape} ")
    A = create_tensor((B, H, T, BT), dtype=torch.float16)
    print(f"==== A.shape = {A.shape} ")
    dw = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== dw.shape = {dw.shape} ")
    du = create_tensor((B, H, T, V), dtype=torch.float16)
    print(f"==== du.shape = {du.shape} ")
    # g = create_tensor((B, H, T), dtype=torch.float)
    g = torch.arange(-1, -(B * H * T + 1), -1).reshape((B, H, T)).to(torch.float)
    print(f"==== g.shape = {g.shape} ")
    print(f"==== g = {g} ")

    cu_seqlens = prepare_cu_seqlens(T = T, L = 16)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    print(f"==== chunk_indices len = {len(chunk_indices)}, chunk_indices[:20] = {chunk_indices[:20]}")

    k_npu = k.npu()
    v_npu = v.npu()
    beta_npu = beta.npu()
    A_npu = A.npu()
    dw_npu = dw.npu()
    du_npu = du.npu()
    g_npu = g.npu()

    dA_npu = torch_npu.npu_prepare_wy_repr_bwd_da(k_npu, v_npu, beta_npu, A_npu, dw_npu, du_npu, g_npu, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=chunk_size)
    torch.save(dA_npu, "/data/yzq/flash-linear-attention-npu-yzq/chunk_gated_delta_rule/prepare_wy_repr_bwd_da/test/output/test_dA_var_npu.pt")
    print(f"==== dA_npu.shape = {dA_npu.shape} ")
    print(f"==== dA_npu = {dA_npu} ")
    print(f"==== dA_npu.dtype = {dA_npu.dtype} ")
    # 测试dA_npu里是否包含nan值
    print(f"==== dA_npu has NaN: {torch.isnan(dA_npu).any().item()}")

    NT = len(chunk_indices) // 2
    print("==== NT = ", NT)
    dA_cpu = compute_dA_cpu(A, dw, g, beta, k, v, du, chunk_indices, cu_seqlens, B, H, T, K, BT, NT)
    torch.save(dA_cpu, "/data/yzq/flash-linear-attention-npu-yzq/chunk_gated_delta_rule/prepare_wy_repr_bwd_da/test/output/test_dA_var_cpu.pt")

    ct.isclose(dA_npu, dA_cpu, diff_thd=0.1)

def test_fix():
    B = 1
    T = 2048
    H = 4
    K = 128
    V = 128
    BT=chunk_size=64

    k = create_tensor((B, H, T, K), dtype=torch.bfloat16)
    print(f"==== k.shape = {k.shape} ")
    v = create_tensor((B, H, T, V), dtype=torch.bfloat16)
    print(f"==== v.shape = {v.shape} ")
    beta = create_tensor((B, H, T), dtype=torch.float)
    print(f"==== beta.shape = {beta.shape} ")
    A = create_tensor((B, H, T, BT), dtype=torch.bfloat16)
    print(f"==== A.shape = {A.shape} ")
    dw = create_tensor((B, H, T, K), dtype=torch.bfloat16)
    print(f"==== dw.shape = {dw.shape} ")
    du = create_tensor((B, H, T, V), dtype=torch.bfloat16)
    print(f"==== du.shape = {du.shape} ")
    # g = create_tensor((B, H, T), dtype=torch.float)
    g = torch.arange(-1, -(B * H * T + 1), -1).reshape((B, H, T)).to(torch.float)
    print(f"==== g.shape = {g.shape} ")
    print(f"==== g = {g} ")

    k_npu = k.npu()
    v_npu = v.npu()
    beta_npu = beta.npu()
    A_npu = A.npu()
    dw_npu = dw.npu()
    du_npu = du.npu()
    g_npu = g.npu()

    dA_npu = torch_npu.npu_prepare_wy_repr_bwd_da(k_npu, v_npu, beta_npu, A_npu, dw_npu, du_npu, g_npu, cu_seqlens=None, chunk_indices=None, chunk_size=chunk_size)
    # torch.save(dA_npu, "/data/yzq/flash-linear-attention-npu-yzq/chunk_gated_delta_rule/prepare_wy_repr_bwd_da/test/output/test_dA_npu.pt")
    print(f"==== dA_npu.shape = {dA_npu.shape} ")
    print(f"==== dA_npu = {dA_npu} ")
    print(f"==== dA_npu.dtype = {dA_npu.dtype} ")
    # 测试dA_npu里是否包含nan值
    print(f"==== dA_npu has NaN: {torch.isnan(dA_npu).any().item()}")

    chunk_indices = None
    cu_seqlens = None
    NT = (T + BT - 1) // BT
    print("==== NT = ", NT)
    dA_cpu = compute_dA_cpu(A, dw, g, beta, k, v, du, chunk_indices, cu_seqlens, B, H, T, K, BT, NT)
    # torch.save(dA_cpu, "/data/yzq/flash-linear-attention-npu-yzq/chunk_gated_delta_rule/prepare_wy_repr_bwd_da/test/output/test_dA_cpu.pt")

    ct.isclose(dA_npu, dA_cpu, diff_thd=0.1)

def create_tensor(shape, dtype=torch.float16):

    # return create_incremental_tensor(shape,dtype)
    # return torch.ones(shape, dtype=dtype)
    return torch.rand(shape, dtype=dtype)

if __name__ == "__main__":
    torch.manual_seed(0)
    # print("==== test_variable ====")
    # test_variable()
    # print("==== test_fix ====")
    test_fix()
