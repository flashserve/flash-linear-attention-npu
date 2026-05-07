#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu
import ascend_ops
import pytest

def test_grouped_matmul_interface_exist():
    """
    Test that the 'ascend_ops.grouped_matmul' operator is present in torch.ops.
    This existence test asserts that the custom operator registered under the
    'ascend_ops' namespace is discoverable from Python via torch.ops.ascend_ops.grouped_matmul.
    It does not exercise operator functionality — only that the Python binding
    and registration are available.
    """
    print(torch.ops.ascend_ops.grouped_matmul)
    assert hasattr(torch.ops.ascend_ops, "grouped_matmul"), "The 'grouped_matmul' operator is not registered in the 'torch.ops.ascend_ops' namespace."

EPS = 0.001

DTYPES = [
    torch.bfloat16,
]

TEST_CONFIGS = [
    (4, 128, 32, 32, 128),
]

def generate_group_list_tensor(E, M, PERGROUP):
    group_list = torch.zeros(E, dtype=torch.int64)
    total_group = PERGROUP
    
    for i in range(E):
        if total_group <= M:
            group_list[i] = PERGROUP
            total_group += PERGROUP
        else:
            group_list[i] = PERGROUP - (total_group - M)
            break
    
    return group_list

def calculate_pytorch_grouped_matmul(x_list, weight_list, group_list, M, E, data_type):
    x = x_list[0].to(torch.float32)
    weight = weight_list[0].to(torch.float32)
    
    group_size = M // E
    remainder = M % E
    split_sizes = [group_size] * E
    if remainder > 0:
        split_sizes[-1] += remainder
    
    x_splits = torch.split(x, split_sizes, dim=0)
    
    split_results = []
    for x_split in x_splits:
        split_matmul = torch.matmul(x_split, weight)
        split_results.append(split_matmul)
    return torch.cat(split_results, dim=0).to(data_type)

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("config", TEST_CONFIGS)
@pytest.mark.parametrize("data_type", DTYPES)
def test_grouped_matmul_operator(config, data_type):
    """
    Test the functionality of the grouped_matmul operator with different configurations.
    
    Parameters:
        config: Tuple of (E, M, K, N, PERGROUP)
        data_type: Data type
    """
    E, M, K, N, PERGROUP = config
    
    print(f"\nTesting with config: E={E}, M={M}, K={K}, N={N}, PERGROUP={PERGROUP}, dtype={data_type}")
    group_list = generate_group_list_tensor(E, M, PERGROUP)
    
    # Generate input data
    x_cpu = torch.rand(M, K, dtype=data_type)
    x_list = [x_cpu]
    
    weight_cpu = torch.rand(K, N, dtype=data_type)
    weight_list = [weight_cpu]
    
    # Move data to NPU
    x_list_npu = [x_i.npu() for x_i in x_list]
    weight_list_npu = [weight_i.npu() for weight_i in weight_list]
    group_list_npu = group_list.npu()
    
    # Call grouped_matmul operator
    try:
        npu_result = torch.ops.ascend_ops.grouped_matmul(
            x_list_npu,           # Tensor[] x
            weight_list_npu,      # Tensor[] weight
            None,                 # Tensor[]? bias (可选)
            None,                 # Tensor[]? scale (可选)
            None,                 # Tensor[]? offset (可选)
            None,                 # Tensor[]? antiquantScale (可选)
            None,                 # Tensor[]? antiquantOffset (可选)
            group_list_npu,       # Tensor? groupList
            None,                 # Tensor[]? perTokenScale (可选)
            3,                    # int splitItem
            0,                    # int groupType
            1,                    # int groupListType
            0,                    # int actType
            None                  # int[]? tuningConfigOptional (可选)
        )
        npu_result_cpu = npu_result.cpu()
        print(f"Result shape: {npu_result.shape} \n",npu_result_cpu)
        
    except Exception as e:
        pytest.fail(f"Error calling grouped_matmul operator: {e}")
    
    # Calculate expected result using PyTorch
    try:
        expected_result = calculate_pytorch_grouped_matmul(x_list, weight_list, group_list, M, E, data_type)
        print(f"Expected result shape: {expected_result.shape}")
        
    except Exception as e:
        pytest.fail(f"Error calculating expected result with PyTorch: {e}")
    
    # Compare results
    assert npu_result_cpu.shape == expected_result.shape, \
        f"Shape mismatch! NPU: {npu_result_cpu.shape}, PyTorch: {expected_result.shape}"
    
    # Convert to float32 for accurate comparison
    npu_float = npu_result_cpu.to(torch.float32)
    expected_float = expected_result.to(torch.float32)
    
    # Calculate absolute difference
    abs_diff = torch.abs(npu_float - expected_float)
    max_diff = torch.max(abs_diff).item()
    
    assert max_diff < EPS, \
        f"Grouped matmul failed for config {config}, dtype {dtype}. " \
        f"Max diff: {max_diff:.6f}, threshold: {EPS:.6f}"
    
    print(f"✓ Test passed! Max diff: {max_diff:.6f}")

