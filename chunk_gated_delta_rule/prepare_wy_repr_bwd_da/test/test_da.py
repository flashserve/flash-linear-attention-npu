import torch
import torch_npu
from typing import Optional
import math

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def cdiv(a: torch.LongTensor
    , b : int):
    torch.empty
    return (a + b - 1) // b

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    # print("cu_seqlens is ", cu_seqlens)
    # print("indices is ", indices)

    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

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

def create_tensor(shape, dtype=torch.float16):

    # return create_incremental_tensor(shape,dtype)
    # return torch.ones(shape, dtype=dtype)
    return torch.rand(shape, dtype=dtype)

def bool_matrix_to_uint8(chunk_size):
    # 创建反上三角矩阵（上三角为0，下三角为1）
    bool_matrix = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
    bool_matrix = ~bool_matrix
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

def compute_dA_cpu(
    A: torch.Tensor,      # [B, H, T, BT] - 每个chunk的A值
    dw: torch.Tensor,     # [B, H, T, K]
    g: torch.Tensor,     # [B, H, T]
    beta: torch.Tensor,   # [B, H, T] - beta参数
    k: torch.Tensor,     # [B, H, T, K]
    v: torch.Tensor,      # [B, H, T, V]
    du: torch.Tensor,     # [B, H, T, V]
    chunk_indices: torch.Tensor,  # [NT, 2] 包含 (seq_idx, chunk_idx) 索引
    cu_seqlens: torch.Tensor,  # [B+1] 累积序列长度
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
        if IS_VARLEN:
            # 从chunk_indices获取batch索引和chunk索引
            # chunk_indices形状: [NT, 2]
            seq_idx = chunk_indices[idx, 0].item()   # 等价于序列号i_n = tl.load(chunk_indices + idx * 2).to(tl.int32)
            chunk_idx = chunk_indices[idx, 1].item()
            # 获取当前序列的边界
            bos = cu_seqlens[seq_idx].item()      # 序列开始位置 tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = cu_seqlens[seq_idx + 1].item()  # 序列结束位置 tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            seq_len = eos - bos                     # 实际序列长度
            i_t = chunk_idx
            T = seq_len
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
        print("==== m_A.shape = ", m_A.shape)
        print("==== m_A ====")
        print(m_A)

        # 全1因果掩码m_A
        # m_A = torch.triu(torch.ones(BT, BT, device="cpu"), diagonal=1).bool() # [BT, BT]

        for i_b in range(B):
        # 遍历所有batch
            if IS_VARLEN == False:
                bos = i_b * T
            for i_h in range(H):
            # 遍历所有head 
                # 获取当前chunk的dw, k, beta, g
                dw_chunk = dw[i_b, i_h, bos + idx * BT : bos + idx * BT + BT, :]  # [BT, K]
                k_chunk = k[i_b, i_h, bos + idx * BT : bos + idx * BT + BT, : ]  # [BT, K]
                # beta形状: [B, H, T]
                beta_chunk = beta[i_b, i_h, bos + idx * BT:bos + idx * BT + BT]  # [BT]
                # g形状: [B, H, T]
                g_chunk = g[i_b, i_h, bos + idx * BT:bos + idx * BT + BT]  # [BT]

                # 获取当前chunk的du, v
                du_chunk = du[i_b, i_h,  bos + idx * BT : bos + idx * BT + BT, :]  # [BT, V]
                v_chunk = v[i_b, i_h, bos + idx * BT : bos + idx * BT + BT, :]  # [BT, V]

                # 获取当前chunk的A向量
                # A形状: [B, H, T, BT]
                # 我们需要获取这个chunk对应的A向量
                # 注意: A的每个位置存储的是该chunk对应的A向量
                A_chunk = A[i_b, i_h, bos + idx * BT : bos + idx * BT + BT, :]  # [BT, BT]
                
                g_exp_chunk = torch.exp(g_chunk.to(torch.float32))

                # 步骤1: b_dA_1
                # b_dA_1 = dw_chunk @ b_k_beta_g.T
                b_k_beta_g = k_chunk.to(torch.float32) * (beta_chunk.to(torch.float32) * g_exp_chunk.to(torch.float32))[:, None]
                b_dA_1 = torch.matmul(dw_chunk.to(torch.float32), b_k_beta_g.T.to(torch.float32))

                # 步骤2: b_dA_2
                # b_dA_2 = du_chunk @ b_v_beta.T
                b_v_beta = v_chunk.to(torch.float32) * beta_chunk.to(torch.float32)[:, None]
                b_dA_2 = torch.matmul(du_chunk.to(torch.float32), b_v_beta.T.to(torch.float32))

                # # 步骤3：b_dA_3
                b_dA_3 = b_dA_1 + b_dA_2

                # 步骤4：b_dA_4
                # b_dA_4 = tl.where(m_A, b_dA_3, 0)
                b_dA_4 = torch.where(m_A, b_dA_3.to(torch.float32), 0.0)

                # 步骤5：b_dA_5
                # b_dA_5 = b_dA_4 @ A_chunk.T
                b_dA_5 = torch.matmul(b_dA_4.to(torch.float64), A_chunk.T.to(torch.float64))

                # 步骤6：b_dA_6
                # b_dA_6 = A_chunk.T @ b_dA_5
                b_dA_6 = torch.matmul(A_chunk.T.to(torch.float32), b_dA_5.to(torch.float32))

                # 并行步骤1~6：b_g_sub_exp
                b_g_sub_exp = torch.exp(g_chunk.to(torch.float32)[:, None] - g_chunk.to(torch.float32)[None, :]) 
                
                # 步骤7：b_dA_7
                b_dA_7 = -b_dA_6.to(torch.float32) * b_g_sub_exp.to(torch.float32)

                # 步骤8：b_dA
                # b_dA = tl.where(m_A, b_dA_7, 0)
                b_dA = torch.where(m_A, b_dA_7.to(torch.float32), 0.0) # [BT, BT]

                # 存储结果
                dA[i_b, i_h, bos + idx * BT : bos + idx * BT + BT, :] = b_dA_6.to(torch.float16)

    return dA

def test_variable():
    # B, H, T, K, V = 1, 2, 128, 128, 128
    # BT=chunk_size=64

    B = 1;
    T = 2048;
    H = 4;
    K = 128;
    V = 128;
    BT=chunk_size=64

    k = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== k.shape = {k.shape} ")
    v = create_tensor((B, H, T, V), dtype=torch.float16)
    print(f"==== v.shape = {v.shape} ")
    beta = create_tensor((B, H, T), dtype=torch.float)
    print(f"==== beta.shape = {beta.shape} ")
    A = create_tensor((B, H, T, BT), dtype=torch.float16) * 0.1
    print(f"==== A.shape = {A.shape} ")
    dw = create_tensor((B, H, T, K), dtype=torch.float16) * 0.1
    print(f"==== dw.shape = {dw.shape} ")
    du = create_tensor((B, H, T, V), dtype=torch.float16) * 0.1
    print(f"==== du.shape = {du.shape} ")
    g = create_tensor((B, H, T), dtype=torch.float)
    print(f"==== g.shape = {g.shape} ")
    
    cu_seqlens = k.new_tensor([0, 64,128], dtype=torch.long)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    print(f"==== chunk_indices.shape = {chunk_indices.shape} ",chunk_indices)

    k_npu = k.npu()
    v_npu = v.npu()
    beta_npu = beta.npu()
    A_npu = A.npu()
    dw_npu = dw.npu()
    du_npu = du.npu()
    g_npu = g.npu()
    if cu_seqlens is not None:
        cu_seqlens_npu = cu_seqlens.npu()
        chunk_indices_npu = chunk_indices.npu()

    dA = torch_npu.npu_prepare_wy_repr_bwd_da(k_npu, v_npu, beta_npu, A_npu, dw_npu, du_npu, g_npu, lower_tri_matrix=upper_tri_matrix_npu, cu_seqlens=cu_seqlens_npu, chunk_indices = chunk_indices_npu, chunk_size =chunk_size)

    print(f"==== dA.shape = {dA.shape} ")
    print(f"==== dA = {dA} ")
    print(f"==== dA.dtype = {dA.dtype} ")

def test_fix():
    # B, H, T, K, V = 1, 2, 128, 128, 128
    # BT=chunk_size=64

    B = 1;
    T = 2048;
    H = 4;
    K = 128;
    V = 128;
    BT=chunk_size=64

    k = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== k.shape = {k.shape} ")
    v = create_tensor((B, H, T, V), dtype=torch.float16)
    print(f"==== v.shape = {v.shape} ")
    beta = create_tensor((B, H, T), dtype=torch.float)
    print(f"==== beta.shape = {beta.shape} ")
    A = create_tensor((B, H, T, BT), dtype=torch.float16) * 0.1
    print(f"==== A.shape = {A.shape} ")
    dw = create_tensor((B, H, T, K), dtype=torch.float16) * 0.1
    print(f"==== dw.shape = {dw.shape} ")
    du = create_tensor((B, H, T, V), dtype=torch.float16) * 0.1
    print(f"==== du.shape = {du.shape} ")
    g = create_tensor((B, H, T), dtype=torch.float)
    print(f"==== g.shape = {g.shape} ")

    upper_tri_matrix = bool_matrix_to_uint8(chunk_size)
    print(f"==== upper_tri_matrix.shape = {upper_tri_matrix.shape}")
    print("==== upper_tri_matrix ====")
    print(upper_tri_matrix)

    lower_tri_matrix = bool_matrix_lower_tri_to_uint8(chunk_size)
    print(f"==== lower_tri_matrix.shape = {lower_tri_matrix.shape}")
    print("==== lower_tri_matrix ====")
    print(lower_tri_matrix)

    k_npu = k.npu()
    v_npu = v.npu()
    beta_npu = beta.npu()
    A_npu = A.npu()
    dw_npu = dw.npu()
    du_npu = du.npu()
    g_npu = g.npu()
    upper_tri_matrix_npu = upper_tri_matrix.npu()
    lower_tri_matrix_npu = lower_tri_matrix.npu()

    dA_npu = torch_npu.npu_prepare_wy_repr_bwd_da(k_npu, v_npu, beta_npu, A_npu, dw_npu, du_npu, g_npu, lower_tri_matrix=lower_tri_matrix_npu, cu_seqlens=None, chunk_indices=None, chunk_size=chunk_size)
    torch.save(dA_npu, "/data/yzq/ops-transformer_GDN/chunk_gated_delta_rule/prepare_wy_repr_bwd_da/test/output/dA_npu.pt")
    # print(f"==== dA_npu.shape = {dA_npu.shape} ")
    # print(f"==== dA_npu = {dA_npu} ")
    # print(f"==== dA_npu.dtype = {dA_npu.dtype} ")
    # print(f"==== dA_npu.dtype = {dA_npu.dtype} ")

    chunk_indices = None
    cu_seqlens = None
    NT = T // BT
    print("hhhhhhhhhhhh NT: ", NT)
    dA_cpu = compute_dA_cpu(A, dw, g, beta, k, v, du, chunk_indices, cu_seqlens, B, H, T, K, BT, NT)
    torch.save(dA_cpu, "/data/yzq/ops-transformer_GDN/chunk_gated_delta_rule/prepare_wy_repr_bwd_da/test/output/dA_cpu.pt")


if __name__ == "__main__":
    torch.manual_seed(0)
    # print("==== test_variable ====")
    # test_variable()
    print("==== test_fix ====")
    test_fix()
