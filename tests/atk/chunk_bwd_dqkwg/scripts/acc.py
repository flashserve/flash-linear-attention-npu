import torch
import sys
import os
from typing import Optional, Tuple

# 使同目录下的标杆模块可被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunk_bwd_dqkwg_cpu import chunk_bwd_dqkwg_cpu
from chunk_bwd_dqkwg_gpu import chunk_bwd_dqkwg_gpu_torch


def create_gate_g(B: int, H: int, T: int, gtype) -> torch.Tensor:
    """生成递减且为负数的 gate 张量，逻辑与 executor 一致。"""
    g = -torch.sort(torch.rand(B * T * H) * 10, descending=False)[0].reshape((B, H, T))
    return g.to(gtype)


def chunk_bwd_dqkwg_cpu_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: Optional[torch.Tensor],
    g: Optional[torch.Tensor],
    dv: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64,
    benchmark: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    CPU 标杆包装函数。
    输入布局与 executor 一致：
        q, k: [B, HK, T, K]；v, do, dv: [B, HV, T, V]
        g: [B, HV, T]；h, dh: [B, HV, num_chunks, K, V]；w: [B, HV, T, K]
    输出布局：
        dq, dk: [B, HK, T, K]；dw: [B, HV, T, K]；dg: [B, HV, T]
    """
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    do_t = do.transpose(1, 2).contiguous()
    dv_t = dv.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous() if g is not None else None
    w_t = w.transpose(1, 2).contiguous() if w is not None else None
    h_t = h.permute(0, 2, 1, 3, 4).contiguous()
    dh_t = dh.permute(0, 2, 1, 3, 4).contiguous()

    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int64) if cu_seqlens is not None else None

    dq, dk, dw, dg = chunk_bwd_dqkwg_cpu(
        q_t, k_t, v_t, do_t, h_t, dh_t, w_t, g_t, dv_t, scale, cu_seqlens_tensor, chunk_size,
        benchmark=benchmark
    )

    dq = dq.transpose(1, 2).contiguous()
    dk = dk.transpose(1, 2).contiguous()
    dw = dw.transpose(1, 2).contiguous()
    if dg is not None:
        dg = dg.transpose(1, 2).contiguous()

    return dq, dk, dw, dg


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    """对比两个张量的精度，输出中文统计信息。"""
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    # 相对误差
    denom = b_f.abs().clamp(min=1e-12)
    rel_diff = (diff / denom)
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()
    # allclose 判断
    atol = 1e-3
    rtol = 1e-3
    is_close = torch.allclose(a_f, b_f, atol=atol, rtol=rtol)
    print(f"  [{name}] 最大绝对误差: {max_abs:.8f}  平均绝对误差: {mean_abs:.8f}")
    print(f"  [{name}] 最大相对误差: {max_rel:.8f}  平均相对误差: {mean_rel:.8f}")
    print(f"  [{name}] allclose(atol={atol}, rtol={rtol}): {'通过' if is_close else '未通过'}")


def run_test(
    B: int, HK: int, HV: int, T: int, K: int, V: int,
    chunk_size: int, qkv_dtype: torch.dtype, g_dtype: torch.dtype,
    fp64: bool, device_id: int = 0
) -> None:
    """单组测试：生成数据 -> 调用 CPU/GPU -> 对比精度。"""
    scale = 1.0 / (K ** 0.5)
    num_chunks = (T + chunk_size - 1) // chunk_size

    print(f"\n测试配置: B={B}, HK={HK}, HV={HV}, T={T}, K={K}, V={V}, "
          f"chunk_size={chunk_size}, qkv={qkv_dtype}, g={g_dtype}, fp64={fp64}")

    torch.manual_seed(42)
    # 生成测试数据（executor 布局）
    q = torch.rand((B, HK, T, K), dtype=qkv_dtype)
    k = torch.rand((B, HK, T, K), dtype=qkv_dtype)
    v = torch.rand((B, HV, T, V), dtype=qkv_dtype)
    do = torch.rand((B, HV, T, V), dtype=qkv_dtype)
    dv = torch.rand((B, HV, T, V), dtype=qkv_dtype)
    w = torch.rand((B, HV, T, K), dtype=qkv_dtype)
    h = torch.rand((B, HV, num_chunks, K, V), dtype=qkv_dtype)
    dh = torch.rand((B, HV, num_chunks, K, V), dtype=qkv_dtype)
    g = create_gate_g(B, HV, T, g_dtype)
    cu_seqlens = None  # 定长模式

    # ------------------------------------------------------------------
    # 1. 调用 CPU 标杆
    # ------------------------------------------------------------------
    print("\n[1] 调用 CPU 标杆 ...")
    # CPU 标杆中 benchmark=True 对应 fp64，benchmark=False 对应输入精度
    cpu_benchmark = fp64
    dq_cpu, dk_cpu, dw_cpu, dg_cpu = chunk_bwd_dqkwg_cpu_torch(
        q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens, chunk_size,
        benchmark=cpu_benchmark
    )
    print(f"  CPU 输出: dq{tuple(dq_cpu.shape)} dk{tuple(dk_cpu.shape)} "
          f"dw{tuple(dw_cpu.shape)} dg{tuple(dg_cpu.shape)}")

    # ------------------------------------------------------------------
    # 2. 调用 GPU 标杆
    # ------------------------------------------------------------------
    print("\n[2] 调用 GPU 标杆 ...")
    try:
        dq_gpu, dk_gpu, dw_gpu, dg_gpu = chunk_bwd_dqkwg_gpu_torch(
            q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens, chunk_size,
            fp64=fp64, device=device_id
        )
    except RuntimeError as e:
        print(f"  GPU 执行失败: {e}")
        return
    print(f"  GPU 输出: dq{tuple(dq_gpu.shape)} dk{tuple(dk_gpu.shape)} "
          f"dw{tuple(dw_gpu.shape)} dg{tuple(dg_gpu.shape)}")

    # ------------------------------------------------------------------
    # 3. 精度对比
    # ------------------------------------------------------------------
    print("\n[3] CPU vs GPU 精度对比:")
    # 统一到 float32 做对比
    dq_cpu_c = dq_cpu.float().cpu()
    dk_cpu_c = dk_cpu.float().cpu()
    dw_cpu_c = dw_cpu.float().cpu()
    dg_cpu_c = dg_cpu.float().cpu()
    dq_gpu_c = dq_gpu.float().cpu()
    dk_gpu_c = dk_gpu.float().cpu()
    dw_gpu_c = dw_gpu.float().cpu()
    dg_gpu_c = dg_gpu.float().cpu()

    compare_tensors("dq", dq_cpu_c, dq_gpu_c)
    compare_tensors("dk", dk_cpu_c, dk_gpu_c)
    compare_tensors("dw", dw_cpu_c, dw_gpu_c)
    compare_tensors("dg", dg_cpu_c, dg_gpu_c)


if __name__ == "__main__":
    print("=" * 60)
    print("chunk_bwd_dqkwg CPU vs GPU 精度对比测试")
    print("=" * 60)

    # 公共 shape 参数
    B = 1
    HK = 4
    HV = 4          # n_ratio = 1
    T = 128
    K = 128
    V = 128
    chunk_size = 64

    # ----------------------------------------------------------
    # 测试 1: 输入精度 (bf16) 对比
    # ----------------------------------------------------------
    print("\n" + "#" * 60)
    print("# 测试 1: 输入精度 bf16 (fp64=False)")
    print("#" * 60)
    run_test(
        B=B, HK=HK, HV=HV, T=T, K=K, V=V,
        chunk_size=chunk_size,
        qkv_dtype=torch.bfloat16,
        g_dtype=torch.float32,
        fp64=False,
    )

    # ----------------------------------------------------------
    # 测试 2: fp64 高精度对比
    # ----------------------------------------------------------
    print("\n" + "#" * 60)
    print("# 测试 2: fp64 高精度 (fp64=True)")
    print("#" * 60)
    run_test(
        B=B, HK=HK, HV=HV, T=T, K=K, V=V,
        chunk_size=chunk_size,
        qkv_dtype=torch.bfloat16,
        g_dtype=torch.float32,
        fp64=True,
    )

    # ----------------------------------------------------------
    # 测试 3: fp16 输入精度对比
    # ----------------------------------------------------------
    print("\n" + "#" * 60)
    print("# 测试 3: 输入精度 fp16 (fp64=False)")
    print("#" * 60)
    run_test(
        B=B, HK=HK, HV=HV, T=T, K=K, V=V,
        chunk_size=chunk_size,
        qkv_dtype=torch.float16,
        g_dtype=torch.float32,
        fp64=False,
    )

    # ----------------------------------------------------------
    # 测试 4: GVA 分组场景 (HV = 2*HK)
    # ----------------------------------------------------------
    print("\n" + "#" * 60)
    print("# 测试 4: GVA 分组场景 n_ratio=2 (fp64=True)")
    print("#" * 60)
    run_test(
        B=B, HK=2, HV=4, T=T, K=K, V=V,
        chunk_size=chunk_size,
        qkv_dtype=torch.bfloat16,
        g_dtype=torch.float32,
        fp64=True,
    )

    print("\n" + "=" * 60)
    print("所有精度对比测试完成")
    print("=" * 60)
