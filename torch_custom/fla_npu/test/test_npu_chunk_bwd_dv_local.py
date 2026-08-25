# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import os
from typing import Optional
import math
import hashlib
from golden import (
    chunk_bwd_dv_local_fix,
    chunk_bwd_dv_local_fix_high_precision,
    chunk_bwd_dv_local_variable,
    chunk_bwd_dv_local_variable_high_precision,
    prepare_chunk_indices,
)
from utils import generate_cu_seqlens, compare_tensors_by_ratio, create_incremental_tensor, create_tensor, bool_matrix_to_uint8, get_tensor_md5, compare_tensors_md5
import ct
from fla_npu.ops import ascendc as ascendc_ops


torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

def test_variable():
    B, H_qk, H_do, T, K, V = 1, 4, 16, 128, 128, 128
    chunk_size= 64
    scale = 0.011
    cu_seqlens_len = 2

    q = create_tensor((B, H_qk, T, K), dtype=torch.float16)
    print(f"==== q.shape = {q.shape} ")
    k = create_tensor((B, H_qk, T, K), dtype=torch.float16)
    print(f"==== k.shape = {k.shape} ")
    d_o = create_tensor((B, H_do, T, V), dtype=torch.float16)
    print(f"==== d_o.shape = {d_o.shape} ")
    g = torch.arange(B * H_do * T, 0, -1).reshape((B, H_do, T)).to(torch.float16)
    print(f"==== g.shape = {g.shape} ")

    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    print(f"==== cu_seqlens.shape = {cu_seqlens.shape} ",cu_seqlens)

    dv_golden = chunk_bwd_dv_local_variable(q, k, d_o, g, scale, cu_seqlens, chunk_size)
    dv_golden_high_precision = chunk_bwd_dv_local_variable_high_precision(
        q, k, d_o, g, scale, cu_seqlens, chunk_size
    )
    print(f"==== dv_golden.shape = {dv_golden.shape} ")
    assert torch.all(dv_golden.abs().sum(dim=(0, 2, 3)) > 0)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()

    cu_seqlens_list = cu_seqlens.tolist()
    chunk_indices_list = chunk_indices.flatten().tolist()

    dv = ascendc_ops.npu_chunk_bwd_dv_local(q_npu, k_npu, d_o_npu, g_npu, scale=scale, chunk_size=chunk_size, g_gamma=None, A=None, cu_seqlens=cu_seqlens_list, chunk_indices=chunk_indices_list)
    result = ct.dual(dv.cpu(), dv_golden, dv_golden_high_precision)
    assert result["success"] is True


def test_fix():
    B=4
    H_qk=4
    H_do=16
    T=198
    K=128
    V=128
    chunk_size=64
    scale=0.0625

    q = create_tensor((B, H_qk, T, K), dtype=torch.bfloat16)
    print(f"==== q.shape = {q.shape} ")
    k = create_tensor((B, H_qk, T, K), dtype=torch.bfloat16)
    print(f"==== k.shape = {k.shape} ")
    d_o = create_tensor((B, H_do, T, V), dtype=torch.bfloat16)
    print(f"==== d_o.shape = {d_o.shape} ")
    g = torch.arange(B * H_do * T, 0, -1).reshape((B, H_do, T)).to(torch.bfloat16)
    print(f"==== g.shape = {g.shape} ")
    cu_seqlens = None
    dv_golden =  chunk_bwd_dv_local_fix(q, k, d_o, g, scale, cu_seqlens, chunk_size)
    dv_golden_high_precision = chunk_bwd_dv_local_fix_high_precision(
        q, k, d_o, g, scale, cu_seqlens, chunk_size
    )
    assert torch.all(dv_golden.abs().sum(dim=(0, 2, 3)) > 0)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()
    dv = ascendc_ops.npu_chunk_bwd_dv_local(q_npu, k_npu, d_o_npu, g_npu, scale=scale, chunk_size=chunk_size, g_gamma=None, A=None, cu_seqlens=None, chunk_indices=None)
    result = ct.dual(dv.cpu(), dv_golden, dv_golden_high_precision)
    assert result["success"] is True


if __name__ == "__main__":
    torch.manual_seed(0)
    
    test_variable()
    print("variable test done!")
    test_fix()
    print("fix test done!")

    
