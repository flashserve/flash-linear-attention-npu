# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
import torch
from dataclasses import dataclass
import numpy as np
import math
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

import math
from typing import Optional

def generate_tensor(shape, data_type, data_max):
    tensor = torch.rand(shape) * (data_max * 2) - data_max
    return tensor.to(data_type)

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
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

def cumsum_cu_seqlens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.nn.functional.pad(
        torch.cumsum(cu_seqlens, dim=0),
        (1, 0),
        value=0
    )

def bool_matrix_to_uint8(chunk_size):
    # 创建反上三角矩阵（上三角为0，下三角为1）
    bool_matrix = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
    bool_matrix = ~bool_matrix
    # print(f"==== bool_matrix.shape = {bool_matrix.shape} ",bool_matrix)
    # 将bool矩阵转换为uint8 (0或1)
    uint8_matrix = bool_matrix.to(torch.uint8)
    # 重塑为 (chunk_size, chunk_size//8, 8) 以便每8个bit打包
    reshaped = uint8_matrix.reshape(chunk_size, chunk_size // 8, 8)
    # 将每8个bit打包成一个uint8
    # bit0 * 1 + bit1 * 2 + bit2 * 4 + ... + bit7 * 128
    powers = torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.uint8)
    packed = (reshaped * powers).sum(dim=-1).to(torch.uint8)
    return packed

def prepare_wy_repr_bwd_full_torch(
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
                # print("b_k.size() = ",b_k.size())
                # print("q_normal.size() = ",q_normal.size())
                b_q = q_normal.transpose(0, 1)  # [BK, chunk_len]
                # print("b_q.size() = ",b_q.size())
                b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q) * scale # [BT,BT]
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
            g_factor = torch.exp(g_j - g_i)  # [chunk_len, chunk_len]
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
                b_do = do[batch_idx, i_h, global_start:global_start+chunk_len, i_v:v_end].to(torch.float32) # do [B, T_max, H, V] → b_do [chunk_len, BV]
                b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do) # b_A_masked 这个 [BT, BT] 的矩阵，只有左上角 [chunk_len, chunk_len] 区域有非 0 值，其余所有区域全是 0
                dv[batch_idx, i_h, global_start:global_start+chunk_len, i_v:v_end] += b_dv
    return dv

@register("executor_prepare_wy_repr_bwd_full")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.qkv_type = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        d_o = input_data.kwargs["d_o"]
        g = input_data.kwargs["g"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs["scale"]

        dv = prepare_wy_repr_bwd_full_torch(q, k, d_o, g, scale, cu_seqlens, chunk_size)
        if self.qkv_type == "bf16":
            dv = dv.to(torch.bfloat16)
        if self.qkv_type == "fp16":
            dv = dv.to(torch.float16)

        return dv

    def init_by_input_data(self, input_data: InputDataset):
        B, H, T, K = input_data.kwargs["q"].shape
        V = input_data.kwargs["d_o"].shape[3]
        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        d_o = input_data.kwargs["d_o"]
        g = input_data.kwargs["g"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        chunk_size = input_data.kwargs["chunk_size"]

        is_fix =  input_data.kwargs["is_fix"]
        self.qkv_type =  input_data.kwargs["qkv_type"]

        is_fix = False
        print("is_fix = ",is_fix)
        print("chunk_size = ",chunk_size)
        if not is_fix:
            # 构造cu_seqlens
            cu_seqlens = cumsum_cu_seqlens(cu_seqlens)
            T = cu_seqlens[-1]
            q = generate_tensor((B, H, T, K), torch.bfloat16, 5)
            k = generate_tensor((B, H, T, K), torch.bfloat16, 5)
            d_o = generate_tensor((B, H, T, V), torch.bfloat16, 5)
            g = generate_tensor((B, H, T), torch.bfloat16, 5)
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        else:
            cu_seqlens = None
            chunk_indices = None
        print("cu_seqlens = ",cu_seqlens)
        print("chunk_indices = ",chunk_indices)

        qkv_type = input_data.kwargs["q"].dtype
        print("qkv_type = ",qkv_type)
        g_type = input_data.kwargs["g"].dtype
        is_mix =  input_data.kwargs["is_mix"]
        if not is_mix:
            g_type = qkv_type
        q = q.to(qkv_type)
        k = k.to(qkv_type)
        d_o = d_o.to(qkv_type)
        g = g.to(g_type)
        upper_tri_matrix = bool_matrix_to_uint8(chunk_size)
        g_gamma = input_data.kwargs["g_gamma"]
        A = input_data.kwargs["A"]
        if self.device == "pyaclnn":
            q = q.npu()
            k = k.npu()
            d_o = d_o.npu()
            g = g.npu()
            g_gamma = g_gamma.npu()
            upper_tri_matrix = upper_tri_matrix.npu()
            A = A.npu()
            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens.npu()
            if chunk_indices is not None:
                chunk_indices = chunk_indices.npu()
            
        input_data.kwargs['q'] = q
        input_data.kwargs['k'] = k
        input_data.kwargs['d_o'] = d_o
        input_data.kwargs['g'] = g
        input_data.kwargs['cu_seqlens'] = cu_seqlens
        input_data.kwargs['chunk_indices'] = chunk_indices
        input_data.kwargs["upper_tri_matrix"] = upper_tri_matrix
        input_data.kwargs.pop("is_mix")
        input_data.kwargs.pop("is_fix")
        input_data.kwargs.pop("qkv_type")


# @register("aclnn_prepare_wy_repr_bwd_full")
# class ChunkBwdDvLocalAclnnApi(AclnnBaseApi):
#     def init_by_input_data(self, input_data: InputDataset):
#         input_args, output_packages = super().init_by_input_data(input_data)
#         input_args.pop()
#         output_packages[:] = [input_args[0]]
#         return input_args, output_packages