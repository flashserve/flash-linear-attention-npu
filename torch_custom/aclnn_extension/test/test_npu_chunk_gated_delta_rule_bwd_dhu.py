import torch
import torch_npu
import os
# import custom_ops
import aclnn_extension

os.environ['TBE_PARALLEL_COMPILE_ENABLE'] = '0'
os.environ['PARALLEL_COMPILE'] = '0'

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.set_device(4)


def prepare_chunk_indices_original(cu_seqlens, chunk_size=64):
    """完全保持你原始代码的chunk_indices生成逻辑"""
    chunk_indices = []

    # 遍历每个序列
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size  # 向上取整

        # 添加 [seq_idx, chunk_idx] 对
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx + 1)  # chunk_idx从1开始

    return chunk_indices


def run_case(
    case_name,
    B,
    T,
    H,
    K,
    V,
    chunk_size,
    dtype,
    is_varlen=False,
    cu_seqlens=None
):
    print(f"\n===== {case_name} =====")
    print(f"B={B}, T={T}, H={H}, K={K}, V={V}, chunk_size={chunk_size}, dtype={dtype}, is_varlen={is_varlen}")

    torch.manual_seed(0)

    q = torch.randn(B, H, T, K, dtype=dtype)
    k = torch.randn(B, H, T, K, dtype=dtype)
    w = torch.randn(B, H, T, K, dtype=dtype)
    do = torch.randn(B, H, T, V, dtype=dtype)
    dv = torch.randn(B, H, T, V, dtype=dtype)
    g = torch.randn(B, H, T, dtype=dtype)
    scale = k.shape[-1] ** -0.5

    chunk_indices = None
    if is_varlen:
        chunk_indices = prepare_chunk_indices_original(cu_seqlens, chunk_size)
        print("cu_seqlens =", cu_seqlens)
        print("chunk_indices len =", len(chunk_indices))

    print("before custom op")
    dh, dh0, dv2 = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q.npu(),
        k.npu(),
        w.npu(),
        do.npu(),
        dv.npu(),
        scale=scale,
        chunk_size=chunk_size,
        g=g.npu(),
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices
    )
    print("after custom op")
    print("dh shape =", dh.shape)
    print("dv2 shape =", dv2.shape)
    if dh0 is None:
        print("dh0 = None")
    else:
        print("dh0 shape =", dh0.shape)


if __name__ == "__main__":

    # Case 1: 定长
    run_case(
        case_name="Case 1: fixed length",
        B=1,
        T=32768,
        H=32,
        K=128,
        V=128,
        chunk_size=64,
        dtype=torch.float16,
        is_varlen=False,
        cu_seqlens=None
    )
    print("Case 1 finished")

    # Case 2: 完全保持你原始代码不动的变长case
    run_case(
        case_name="Case 2: varlen original",
        B=1,
        T=32768,
        H=32,
        K=128,
        V=128,
        chunk_size=64,
        dtype=torch.float16,
        is_varlen=True,
        cu_seqlens=[0, 1024, 2048, 4096, 8192, 10240, 20480, 32768]
    )
    print("Case 2 finished")

    # Case 3: 额外补一组 L2 风格变长 case
    run_case(
        case_name="Case 3: varlen L2 style",
        B=1,
        T=65536,
        H=16,
        K=128,
        V=128,
        chunk_size=64,
        dtype=torch.float16,
        is_varlen=True,
        cu_seqlens=[0, 2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536]
    )
    print("Case 3 finished")
