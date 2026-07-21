import torch
import torch_npu
from typing import Optional
import math
import ct
import random
import fla_npu
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# torch.npu.config.allow_internal_format = False
# torch.npu.set_compile_mode(jit_compile=False)


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
        end = cu_seqlens[i + 1]
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


def create_tensor(shape, dtype=torch.float16):
    # return create_incremental_tensor(shape,dtype)
    # return torch.ones(shape, dtype=dtype)
    return torch.rand(shape, dtype=dtype)


def test_prepare_wy_repr_bwd_da_variable(
    B: int,
    HK: int,
    HV: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    cu_seqlens_len: int,
    ktype,
    gtype,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if not hasattr(test_prepare_wy_repr_bwd_da_variable, "call_count"):
        test_prepare_wy_repr_bwd_da_variable.call_count = 1
    else:
        test_prepare_wy_repr_bwd_da_variable.call_count += 1

    BT = chunk_size
    group_size = HV // HK

    k = create_tensor((B, HK, T, K), dtype=ktype)
    print(f"==== k.shape = {k.shape} ")
    v = create_tensor((B, HV, T, V), dtype=ktype)
    print(f"==== v.shape = {v.shape} ")
    beta = create_tensor((B, HV, T), dtype=gtype)
    print(f"==== beta.shape = {beta.shape} ")
    A = create_tensor((B, HV, T, BT), dtype=ktype)
    print(f"==== A.shape = {A.shape} ")
    dw = create_tensor((B, HV, T, K), dtype=ktype)
    print(f"==== dw.shape = {dw.shape} ")
    du = create_tensor((B, HV, T, V), dtype=ktype)
    print(f"==== du.shape = {du.shape} ")
    # g = create_tensor((B, HV, T), dtype=gtype)
    g = torch.arange(-1, -(B * HV * T + 1), -1).reshape((B, HV, T)).to(gtype)
    print(f"==== g.shape = {g.shape} ")
    # print(f"==== g = {g} ")

    cu_seqlens = prepare_cu_seqlens(T=T, L=cu_seqlens_len)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    print(f"==== chunk_indices len = {len(chunk_indices)}, chunk_indices[:20] = {chunk_indices[:20]}")

    k_npu = k.npu()
    v_npu = v.npu()
    beta_npu = beta.npu()
    A_npu = A.npu()
    dw_npu = dw.npu()
    du_npu = du.npu()
    g_npu = g.npu()

    dA_npu = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
        k_npu, v_npu, beta_npu, A_npu, dw_npu, du_npu, g_npu,
        chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    print(f"==== dA_npu.shape = {dA_npu.shape} ")
    print(f"==== dA_npu.dtype = {dA_npu.dtype} ")
    # print(f"==== dA_npu = {dA_npu} ")
    # 测试dA_npu里是否包含nan值
    print(f"==== dA_npu has NaN: {torch.isnan(dA_npu).any().item()}")

    print(f"test_prepare_wy_repr_bwd_da_variable 被调用了第 {test_prepare_wy_repr_bwd_da_variable.call_count} 次")


def test_prepare_wy_repr_bwd_da_fix(
    B: int,
    HK: int,
    HV: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    ktype,
    gtype,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if not hasattr(test_prepare_wy_repr_bwd_da_fix, "call_count"):
        test_prepare_wy_repr_bwd_da_fix.call_count = 1
    else:
        test_prepare_wy_repr_bwd_da_fix.call_count += 1

    BT = chunk_size
    group_size = HV // HK

    k = create_tensor((B, HK, T, K), dtype=ktype)
    print(f"==== k.shape = {k.shape} ")
    v = create_tensor((B, HV, T, V), dtype=ktype)
    print(f"==== v.shape = {v.shape} ")
    beta = create_tensor((B, HV, T), dtype=gtype)
    print(f"==== beta.shape = {beta.shape} ")
    A = create_tensor((B, HV, T, BT), dtype=ktype)
    print(f"==== A.shape = {A.shape} ")
    dw = create_tensor((B, HV, T, K), dtype=ktype)
    print(f"==== dw.shape = {dw.shape} ")
    du = create_tensor((B, HV, T, V), dtype=ktype)
    print(f"==== du.shape = {du.shape} ")
    # g = create_tensor((B, HV, T), dtype=gtype)
    g = torch.arange(-1, -(B * HV * T + 1), -1).reshape((B, HV, T)).to(gtype)
    print(f"==== g.shape = {g.shape} ")
    # print(f"==== g = {g} ")

    k_npu = k.npu()
    v_npu = v.npu()
    beta_npu = beta.npu()
    A_npu = A.npu()
    dw_npu = dw.npu()
    du_npu = du.npu()
    g_npu = g.npu()

    dA_npu = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
        k_npu, v_npu, beta_npu, A_npu, dw_npu, du_npu, g_npu,
        chunk_size=chunk_size, cu_seqlens=None, chunk_indices=None
    )
    print(f"==== dA_npu.shape = {dA_npu.shape} ")
    print(f"==== dA_npu.dtype = {dA_npu.dtype} ")
    # print(f"==== dA_npu = {dA_npu} ")
    # 测试dA_npu里是否包含nan值
    print(f"==== dA_npu has NaN: {torch.isnan(dA_npu).any().item()}")

    print(f"test_prepare_wy_repr_bwd_da_fix 被调用了第 {test_prepare_wy_repr_bwd_da_fix.call_count} 次")


DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def run_cases_from_json(json_path: str):
    import json
    with open(json_path, "r") as f:
        cases = json.load(f)
    for i, case in enumerate(cases):
        if not case.get("enabled", True):
            print(f"[SKIP] case {i}: {case.get('name', '')}")
            continue
        name = case.get("name", f"case_{i}")
        varlen = case.get("varlen", False)
        B = case["B"]
        HK = case["query_head"]
        HV = case["value_head"]
        T = case["T"]
        K = case["Kdim"]
        V = case["Vdim"]
        chunk_size = case["chunk_size"]
        ktype = DTYPE_MAP[case["dtype"]]
        gtype = DTYPE_MAP[case["gtype"]]
        print(f"\n{'='*60}")
        print(f"[RUN] {name}  varlen={varlen}  B={B} HK={HK} HV={HV} T={T} K={K} V={V} chunk_size={chunk_size}")
        print(f"{'='*60}")
        if varlen:
            cu_seqlens_len = case["mean_len"]
            test_prepare_wy_repr_bwd_da_variable(
                B=B, HK=HK, HV=HV, T=T, K=K, V=V,
                chunk_size=chunk_size, cu_seqlens_len=cu_seqlens_len,
                ktype=ktype, gtype=gtype,
            )
        else:
            test_prepare_wy_repr_bwd_da_fix(
                B=B, HK=HK, HV=HV, T=T, K=K, V=V,
                chunk_size=chunk_size, ktype=ktype, gtype=gtype,
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="prepare_wy_repr_bwd_da performance test")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to JSON case file; if omitted, runs built-in cases")
    parser.add_argument("--device", type=int, default=0,
                        help="NPU device id (default: 0)")
    args = parser.parse_args()

    torch.npu.utils.set_device(args.device)
    torch.manual_seed(0)

    if args.json:
        run_cases_from_json(args.json)
    else:
        # Fix length tests (HK == HV, compatible with old behavior)
        test_prepare_wy_repr_bwd_da_fix(B=1, HK=2, HV=2, T=128, K=128, V=128, chunk_size=64, ktype=torch.float16, gtype=torch.float16)
        test_prepare_wy_repr_bwd_da_fix(B=2, HK=4, HV=4, T=256, K=128, V=128, chunk_size=128, ktype=torch.bfloat16, gtype=torch.bfloat16)
        test_prepare_wy_repr_bwd_da_fix(B=4, HK=8, HV=8, T=512, K=128, V=256, chunk_size=64, ktype=torch.float16, gtype=torch.float32)
        test_prepare_wy_repr_bwd_da_fix(B=8, HK=16, HV=16, T=1024, K=128, V=256, chunk_size=128, ktype=torch.bfloat16, gtype=torch.float32)
        test_prepare_wy_repr_bwd_da_fix(B=1, HK=32, HV=32, T=2048, K=128, V=128, chunk_size=64, ktype=torch.float16, gtype=torch.float32)

        # Variable length tests (HK == HV, compatible with old behavior)
        test_prepare_wy_repr_bwd_da_variable(B=1, HK=4, HV=4, T=128, K=128, V=128, chunk_size=64, cu_seqlens_len=2, ktype=torch.float16, gtype=torch.float16)
        test_prepare_wy_repr_bwd_da_variable(B=1, HK=8, HV=8, T=256, K=128, V=128, chunk_size=128, cu_seqlens_len=3, ktype=torch.bfloat16, gtype=torch.bfloat16)
        test_prepare_wy_repr_bwd_da_variable(B=1, HK=16, HV=16, T=512, K=128, V=256, chunk_size=64, cu_seqlens_len=4, ktype=torch.float16, gtype=torch.float32)
        test_prepare_wy_repr_bwd_da_variable(B=1, HK=32, HV=32, T=1024, K=128, V=256, chunk_size=128, cu_seqlens_len=5, ktype=torch.bfloat16, gtype=torch.float32)
        test_prepare_wy_repr_bwd_da_variable(B=1, HK=32, HV=32, T=2048, K=128, V=128, chunk_size=64, cu_seqlens_len=16, ktype=torch.bfloat16, gtype=torch.float32)
