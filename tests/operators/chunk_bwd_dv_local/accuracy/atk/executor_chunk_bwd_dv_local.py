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

def chunk_bwd_dv_local_torch(
    q: torch.Tensor,  # [B, H_qk, T_max, K]
    k: torch.Tensor,  # [B, H_qk, T_max, K]
    do: torch.Tensor, # [B, H_do, T_max, V]
    g: torch.Tensor,  # [B, H_do, T_max]
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,  # [batch_size+1]
    chunk_size: int = 64,
    h_ratio: int = 1
) -> torch.Tensor:
    B, H_qk, T, K = k.shape
    H_do = do.shape[1]
    V = do.shape[3]
    assert H_do == H_qk * h_ratio, f"H_do ({H_do}) must equal H_qk * h_ratio ({H_qk} * {h_ratio} = {H_qk * h_ratio})"
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
        for qk_head in range(H_qk):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float32)
            BK = 128
            BK = min(BK, K)
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, qk_head, global_start:global_start+chunk_len, i_k:k_end].to(torch.float32)
                q_normal = q[batch_idx, qk_head, global_start:global_start+chunk_len, i_k:k_end].to(torch.float32)
                b_q = q_normal.transpose(0, 1)
                b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q) * scale
            for do_group in range(h_ratio):
                do_head = qk_head * h_ratio + do_group
                b_g = g_t[batch_idx, do_head, global_start:global_start+chunk_len].to(torch.float32)
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
                b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor
                b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated))
                b_A_masked = b_A_masked.to(torch.float32)
                BV = 128
                BV = min(BV, V)
                for i_v in range(0, V, BV):
                    v_end = min(i_v + BV, V)
                    v_width = v_end - i_v
                    b_do = do[batch_idx, do_head, global_start:global_start+chunk_len, i_v:v_end].to(torch.float32)
                    b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do)
                    dv[batch_idx, do_head, global_start:global_start+chunk_len, i_v:v_end] += b_dv
    return dv

@register("executor_chunk_bwd_dv_local")
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
        h_ratio = input_data.kwargs.get("h_ratio", 1)

        dv = chunk_bwd_dv_local_torch(q, k, d_o, g, scale, cu_seqlens, chunk_size, h_ratio=h_ratio)
        if self.qkv_type == "bf16":
            dv = dv.to(torch.bfloat16)
        if self.qkv_type == "fp16":
            dv = dv.to(torch.float16)

        return dv

    def init_by_input_data(self, input_data: InputDataset):
        B, H_qk, T, K = input_data.kwargs["q"].shape
        V = input_data.kwargs["d_o"].shape[3]
        H_do = input_data.kwargs["d_o"].shape[1]
        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        d_o = input_data.kwargs["d_o"]
        g = input_data.kwargs["g"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        chunk_size = input_data.kwargs["chunk_size"]

        is_fix =  input_data.kwargs["is_fix"]
        self.qkv_type =  input_data.kwargs["qkv_type"]

        h_ratio = input_data.kwargs.get("h_ratio", H_do // H_qk)

        is_fix = False
        print("is_fix = ",is_fix)
        print("chunk_size = ",chunk_size)
        if not is_fix:
            cu_seqlens = cumsum_cu_seqlens(cu_seqlens)
            T = cu_seqlens[-1]
            q = generate_tensor((B, H_qk, T, K), torch.bfloat16, 5)
            k = generate_tensor((B, H_qk, T, K), torch.bfloat16, 5)
            d_o = generate_tensor((B, H_do, T, V), torch.bfloat16, 5)
            g = generate_tensor((B, H_do, T), torch.bfloat16, 5)
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


# @register("aclnn_chunk_bwd_dv_local")
# class ChunkBwdDvLocalAclnnApi(AclnnBaseApi):
#     def init_by_input_data(self, input_data: InputDataset):
#         input_args, output_packages = super().init_by_input_data(input_data)
#         input_args.pop()
#         output_packages[:] = [input_args[0]]
#         return input_args, output_packages