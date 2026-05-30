import torch
import torch_npu
import os
from typing import Optional
import math
import hashlib
from golden import chunk_bwd_dv_local_fix, chunk_bwd_dv_local_variable, prepare_chunk_indices
from utils import generate_cu_seqlens, compare_tensors_by_ratio, create_incremental_tensor, create_tensor, bool_matrix_to_uint8, get_tensor_md5, compare_tensors_md5
import ct
import fla_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

def test_variable():
    B, H, T, K, V = 1, 32, 128, 128, 128
    chunk_size= 64
    scale = 0.011
    cu_seqlens_len = 2

    q = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== q.shape = {q.shape} ")
    k = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== k.shape = {k.shape} ")
    d_o = create_tensor((B, H, T, V), dtype=torch.float16)
    print(f"==== d_o.shape = {d_o.shape} ")
    g = torch.arange(B * H * T, 0, -1).reshape((B, H, T)).to(torch.float16)
    print(f"==== g.shape = {g.shape} ")

    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    print(f"==== cu_seqlens.shape = {cu_seqlens.shape} ",cu_seqlens)

    dv_golden = chunk_bwd_dv_local_variable(q, k, d_o, g, scale, cu_seqlens, chunk_size)
    print(f"==== dv_golden.shape = {dv_golden.shape} ")

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()

    cu_seqlens_list = cu_seqlens.tolist()
    chunk_indices_list = chunk_indices.flatten().tolist()

    dv = torch.ops.npu.npu_chunk_bwd_dv_local(q_npu, k_npu, d_o_npu, g_npu, scale=scale, chunk_size=chunk_size, g_gamma=None, A=None, cu_seqlens=cu_seqlens_list, chunk_indices=chunk_indices_list)
    ct.single(dv.cpu(), dv_golden)


def test_fix():
    B=2
    H=2
    T=65
    K=128
    V=128
    chunk_size=64
    scale=0.0625

    q = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== q.shape = {q.shape} ")
    k = create_tensor((B, H, T, K), dtype=torch.float16)
    print(f"==== k.shape = {k.shape} ")
    d_o = create_tensor((B, H, T, V), dtype=torch.float16)
    print(f"==== d_o.shape = {d_o.shape} ")
    g = torch.arange(B * H * T, 0, -1).reshape((B, H, T)).to(torch.float16)
    print(f"==== g.shape = {g.shape} ")
    cu_seqlens = None
    dv_golden =  chunk_bwd_dv_local_fix(q, k, d_o, g, scale, cu_seqlens, chunk_size)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()
    dv = torch.ops.npu.npu_chunk_bwd_dv_local(q_npu, k_npu, d_o_npu, g_npu, scale=scale, chunk_size=chunk_size, g_gamma=None, A=None, cu_seqlens=None, chunk_indices=None)
    ct.single(dv.cpu(), dv_golden)


if __name__ == "__main__":
    torch.manual_seed(0)
    
    test_variable()
    print("variable test done!")
    test_fix()
    print("fix test done!")

    
