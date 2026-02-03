import torch
import torch_npu
from typing import Optional
import math
# form ct 

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
    print("cu_seqlens is ", cu_seqlens)
    print("indices is ", indices)

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
                    # print(" golden k * q^T = ",b_A)
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
            g_i = b_g.unsqueeze(1)  # [chunk_len, 1]
            g_j = b_g.unsqueeze(0)  # [1, chunk_len]
            g_factor = torch.exp(g_j - g_i)  * scale  # [chunk_len, chunk_len]
            # print(" golden g_factor = ",g_factor)
            b_A_gated = torch.zeros_like(b_A)
            b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor # [BT, BT] 门控缩放后的注意力核矩阵
            # 应用掩码
            b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated)) # 只保留掩码为 True 的位置的 b_A_gated 值，其余置 0
            # print(" golden b_A_masked = ",b_A_masked)
            b_A_masked = b_A_masked.to(torch.float32) # [BT, BT]
            BV = 128  # 与Triton保持一致
            BV = min(BV, V)  # 确保不超过V
            for i_v in range(0, V, BV):
                v_end = min(i_v + BV, V)
                v_width = v_end - i_v
                b_do = do[batch_idx, i_h, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end].to(torch.float32) # do [B, T_max, H, V] → b_do [chunk_len, BV]
                b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do) # b_A_masked 这个 [BT, BT] 的矩阵，只有左上角 [chunk_len, chunk_len] 区域有非 0 值，其余所有区域全是 0
                # print(" golden b_dv = ",b_dv)
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
    # print("T = ",T)
    # print("BT = ",BT)
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
                    # print(" golden k * q^T = ",b_A)
            b_g = g_t[batch_idx, i_h, global_start:global_start+chunk_len] # g_t [B, H, T_max] → b_g [chunk_len]
            o_t = i_t * BT + torch.arange(0, BT) # [BT] chunk内的token序号
            m_t = o_t < T # [BT] bool掩码：是否是有效token
            o_t_col = o_t.unsqueeze(1)  # [BT, 1]
            o_t_row = o_t.unsqueeze(0)  # [1, BT]
            pos_mask = o_t_col <= o_t_row  # [BT, BT] 上三角矩阵：只允许当前token看<=自己的token（因果掩码）
            m_t_col = m_t.unsqueeze(1)  # [BT, 1]
            valid_mask = m_t_col & m_t  #  [BT, BT] 有效掩码：只保留有效token的位置，左上角是[有效token,有效token]大小的全1
            # 组合掩码
            m_A = pos_mask & valid_mask  # [BT, BT]
            g_i = b_g.unsqueeze(1)  # [chunk_len, 1]
            g_j = b_g.unsqueeze(0)  # [1, chunk_len]
            g_factor = torch.exp(g_j - g_i)  * scale  # [chunk_len, chunk_len]
            # print(" golden g_factor = ",g_factor)
            b_A_gated = torch.zeros_like(b_A)
            b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor # [BT, BT] 门控缩放后的注意力核矩阵
            # 应用掩码
            b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated)) # 只保留掩码为 True 的位置的 b_A_gated 值，其余置 0
            # print(" golden b_A_masked = ",b_A_masked)
            b_A_masked = b_A_masked.to(torch.float32) # [BT, BT]
            BV = 128  # 与Triton保持一致
            BV = min(BV, V)  # 确保不超过V
            for i_v in range(0, V, BV):
                v_end = min(i_v + BV, V)
                v_width = v_end - i_v
                b_do = do[batch_idx, i_h, global_start:global_start+chunk_len, i_v:v_end].to(torch.float32) # do [B, T_max, H, V] → b_do [chunk_len, BV]
                b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do) # b_A_masked 这个 [BT, BT] 的矩阵，只有左上角 [chunk_len, chunk_len] 区域有非 0 值，其余所有区域全是 0
                # print(" golden b_dv = ",b_dv)
                dv[batch_idx, i_h, global_start:global_start+chunk_len, i_v:v_end] += b_dv
    return dv

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

import torch

def compare_tensors_by_ratio(tensor1, tensor2, ratio_threshold=0.01, verbose=True):
    """
    对比两个tensor，检查差值与tensor1对应点位值的比值是否超过阈值
    
    Args:
        tensor1: 第一个tensor（作为基准）
        tensor2: 第二个tensor
        ratio_threshold: 比值阈值，默认0.01（1%）
        verbose: 是否打印详细结果，默认True
    
    Returns:
        bool: 是否所有点位都通过精度检查
    """
    # 检查形状是否相同
    if tensor1.shape != tensor2.shape:
        if verbose:
            print(f"错误: tensor形状不匹配!")
            print(f"  tensor1 shape: {tensor1.shape}")
            print(f"  tensor2 shape: {tensor2.shape}")
        return False
    
    # 计算绝对差值
    diff = torch.abs(tensor1 - tensor2)
    
    # 计算比值：差值 / (|tensor1| + epsilon)，避免除以0
    epsilon = 1e-8  # 小值避免除以0
    ratio = diff / (torch.abs(tensor1) + epsilon)
    
    # 找出超过阈值的点位
    mask = ratio > ratio_threshold
    failed_count = torch.sum(mask).item()
    total_count = tensor1.numel()
    
    if failed_count == 0:
        if verbose:
            print(f"✓ 精度对比成功!")
            print(f"  总点数: {total_count}")
            print(f"  最大相对差: {ratio.max().item():.6f}")
            print(f"  平均相对差: {ratio.mean().item():.6f}")
            print(f"  最大绝对差值: {diff.max().item():.6f}")
        return True
    else:
        if verbose:
            print(f"✗ 精度对比失败!")
            print(f"  总点数: {total_count}")
            print(f"  失败点数: {failed_count} ({failed_count/total_count*100:.2f}%)")
            print(f"  最大相对差: {ratio.max().item():.6f}")
            print(f"  平均相对差: {ratio.mean().item():.6f}")
            print(f"  最大绝对差值: {diff.max().item():.6f}")
            
            # 打印部分失败点位
            print(f"\n失败点位示例 (最多显示10个):")
            failed_indices = torch.nonzero(mask, as_tuple=True)
            display_count = min(10, failed_count)
            
            for i in range(display_count):
                idx = tuple(dim[i].item() for dim in failed_indices)
                val1 = tensor1[idx].item()
                val2 = tensor2[idx].item()
                val_diff = diff[idx].item()
                val_ratio = ratio[idx].item()
                print(f"  位置{idx}:")
                print(f"    tensor1={val1:.6f}, tensor2={val2:.6f}")
                print(f"    差值={val_diff:.6f}, 比值={val_ratio:.6f} ({val_ratio*100:.2f}%)")
        
        return False

if __name__ == "__main__":
    torch.manual_seed(0)
    
    B, H, T, K, V = 2, 1, 65, 128, 128
    chunk_size=64
    scale = 1.0

    q = create_tensor((B, H, T, K), dtype=torch.float16)
    k = create_tensor((B, H, T, K), dtype=torch.float16)
    d_o = create_tensor((B, H, T, V), dtype=torch.float16)
    g = create_tensor((B, H, T), dtype=torch.float16)
    print("q =",q)
    print("k =",k)
    print("d_o =",d_o)
    print("g =",g)
    upper_tri_matrix = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
    cu_seqlens = None
    # cu_seqlens = q.new_tensor([0, 64,128], dtype=torch.long)
    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    # dv_golden = chunk_bwd_dv_local_variable(q, k, d_o, g, scale, cu_seqlens, chunk_size)

    dv_golden =  chunk_bwd_dv_local_fix(q, k, d_o, g, scale, cu_seqlens, chunk_size)


    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()
    upper_tri_matrix_npu = upper_tri_matrix.npu()
    if cu_seqlens is not None:
        cu_seqlens_npu = cu_seqlens.npu()
        chunk_indices_npu = chunk_indices.npu()

    #dv = torch_npu.npu_chunk_bwd_dv_local(q_npu, k_npu, d_o_npu, g_npu,upper_tri_matrix=None, g_gamma=None, A=None,cu_seqlens=cu_seqlens_npu, chunk_indices = chunk_indices_npu, scale=scale, chunk_size =chunk_size)
    dv = torch_npu.npu_chunk_bwd_dv_local(q_npu, k_npu, d_o_npu, g_npu,upper_tri_matrix=upper_tri_matrix_npu, g_gamma=None, A=None,cu_seqlens=None, chunk_indices = None, scale=scale, chunk_size =chunk_size)
    print(f"==== dv_golden.shape = {dv_golden.shape} ",dv_golden)
    print(f"==== dv.shape = {dv.shape} ",dv)

    compare_tensors_by_ratio(dv_golden,dv.cpu())


    

