import torch
from typing import Tuple, Optional, Union


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def cdiv(a: torch.LongTensor, b: int):
    return (a + b - 1) // b


def prepare_chunk_indices_torch(
    cu_seqlens: torch.LongTensor,
    chunkSize: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunkSize).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: list[int],
    chunk_size: int
) -> list[int]:
    """
    基于 cu_seqlens (list[int]) 生成 chunk 索引。

    逻辑复刻原代码：
    1. 计算每个序列的长度: lens[i] = cu_seqlens[i+1] - cu_seqlens[i]
    2. 计算每个序列需要的 chunk 数: ceil(lens[i] / chunk_size)
    3. 生成对应的 (sequence_id, chunk_id) 对
    """
    indices = []
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        length = end - start
        if length <= 0:
            continue
        num_chunks = (length + chunk_size - 1) // chunk_size
        for chunk_id in range(num_chunks):
            indices.append(i)
            indices.append(chunk_id)
    return indices


def select_gpu_device(device_id: int = 0) -> torch.device:
    """选择 GPU 设备，返回 torch.device 对象。"""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU (CUDA) 不可用，无法执行 GPU 标杆计算")
    device = torch.device(f'cuda:{device_id}')
    print(f"已选择 GPU 设备: {torch.cuda.get_device_name(device_id)} (cuda:{device_id})")
    return device


def move_tensors_to_device(
    q, k, v, do, h, dh, w, g, dv, cu_seqlens,
    device: torch.device
):
    """将所有输入张量移动到指定设备。"""
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    do = do.to(device)
    h = h.to(device)
    dh = dh.to(device)
    if w is not None:
        w = w.to(device)
    if g is not None:
        g = g.to(device)
    dv = dv.to(device)
    if cu_seqlens is not None:
        if isinstance(cu_seqlens, (list, tuple)):
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int64, device=device)
        else:
            cu_seqlens = cu_seqlens.to(device)
    return q, k, v, do, h, dh, w, g, dv, cu_seqlens


def chunk_bwd_dqkwg_gpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: Optional[torch.Tensor],
    g: Optional[torch.Tensor],
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64,
    fp64: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    GPU 版本的 chunk_bwd_kernel_dqkwg 标杆实现。
    使用 GPU 小算子（matmul、exp、where 等）拼接完成完整计算。

    调用者需事先将所有输入张量移动到 GPU 设备上。
    本函数从输入张量推断设备，不再做设备选择或数据搬运。

    参数:
        fp64:   当为 True 时使用 fp64 精度小算子拼接；
                当为 False 时 calc_type 使用 fp32（与 CPU 标杆一致），
                输出精度 datatype 与输入精度一致。
    """
    # ------------------------------------------------------------------
    # 精度设置：与 CPU 标杆完全一致
    #   fp64=True:  calc_type=datatype=gtype=mmtype=float64
    #   fp64=False: calc_type=float32, datatype=q.dtype, gtype=g.dtype, mmtype=datatype
    # ------------------------------------------------------------------
    if fp64:
        calc_type = torch.float64
        datatype = torch.float64
        gtype = torch.float64
        mmtype = torch.float64
        print("使用 fp64 精度小算子进行拼接")
    else:
        calc_type = torch.float32
        datatype = q.dtype
        gtype = g.dtype if g is not None else q.dtype
        mmtype = datatype
        print(f"使用 fp32 计算精度，输出精度: {q.dtype}")

    # ------------------------------------------------------------------
    # 类型转换：与 CPU 标杆完全一致（注意：原 CPU 代码未赋值回变量，
    # 这里保持同样行为，仅做类型转换调用，不改变原张量）
    # ------------------------------------------------------------------
    q = q.to(calc_type)
    k = k.to(calc_type)
    v = v.to(calc_type)
    do = do.to(calc_type)
    h = h.to(calc_type)
    dh = dh.to(calc_type)
    if g is not None:
        g = g.to(gtype).to(calc_type)
    dv = dv.to(calc_type)

    # 从输入张量推断设备
    device = q.device

    B, T, HK, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    if HK <= 0 or HV <= 0 or HV % HK != 0:
        raise ValueError(f"GVA 要求 HV 能被 HK 整除，当前 HV={HV}, HK={HK}")
    n_ratio = HV // HK

    g_gamma = None

    # 初始化输出张量（在输入设备上分配）
    dq_hv = torch.zeros((B, T, HV, K), dtype=datatype, device=device)
    dk_hv = torch.zeros((B, T, HV, K), dtype=datatype, device=device)
    dg = torch.zeros_like(g) if g is not None else None
    dw = torch.zeros((B, T, HV, K), dtype=datatype, device=device)
    w = torch.zeros((B, T, HV, K), dtype=datatype, device=device)

    # ------------------------------------------------------------------
    # 辅助函数：处理单个序列的逻辑（与 CPU 版本完全一致的算法，运行在 GPU 上）
    # ------------------------------------------------------------------
    def process_sequence(b_idx, t_start, t_end, seq_idx_in_batch, chunk_start_idx):
        seq_len = t_end - t_start
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for h_idx in range(HV):
            hk_idx = h_idx // n_ratio
            gamma_val = None
            if g_gamma is not None:
                gamma_val = g_gamma[h_idx].item()

            for i_t in range(num_chunks):
                chunk_start_token_idx = t_start + i_t * chunk_size
                chunk_end_token_idx = min(t_start + (i_t + 1) * chunk_size, t_end)
                actual_chunk_len = chunk_end_token_idx - chunk_start_token_idx

                # 切片当前块的数据
                q_c = q[b_idx, chunk_start_token_idx:chunk_end_token_idx, hk_idx, :]  # [BT, K]
                k_c = k[b_idx, chunk_start_token_idx:chunk_end_token_idx, hk_idx, :]  # [BT, K]

                v_c = v[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :]  # [BT, V]
                do_c = do[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :]  # [BT, V]

                h_prev = h[b_idx, i_t + chunk_start_idx, h_idx, :, :]  # [K, V]
                dh_curr = dh[b_idx, i_t + chunk_start_idx, h_idx, :, :]  # [K, V]

                # -----------------------------------------------------------
                # 1. State Contributions (Inter-chunk)
                # -----------------------------------------------------------
                dq_from_state = do_c.to(calc_type).to(mmtype) @ h_prev.transpose(-1, -2).to(calc_type).to(mmtype)
                dq_from_state = dq_from_state.to(datatype).to(calc_type)

                dk_from_state = v_c.to(calc_type).to(mmtype) @ dh_curr.transpose(-1, -2).to(calc_type).to(mmtype)
                dk_from_state = dk_from_state.to(datatype).to(calc_type)

                if w is not None and dv is not None:
                    dv_c = dv[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :]  # [BT, V]
                    dw_c_val = dv_c.to(calc_type).to(mmtype) @ h_prev.transpose(-1, -2).to(calc_type).to(mmtype)
                    dw_c_val = dw_c_val.to(datatype).to(calc_type)
                    dw[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = -dw_c_val

                # -----------------------------------------------------------
                # 2. Gating / Decay Logic Preparation
                # -----------------------------------------------------------
                if g is not None:
                    g_c = g[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx]  # [BT]
                    g_last = g[b_idx, min(chunk_start_token_idx + chunk_size, t_end) - 1, h_idx]

                    dg_last_accum = (h_prev * dh_curr).sum()
                    dg_last_accum = dg_last_accum * torch.exp(g_last)

                    dq_from_state = dq_from_state * torch.exp(g_c)[:, None] * scale
                    dk_from_state = dk_from_state * torch.exp(-g_c + g_last)[:, None]

                    dg_c = (dq_from_state * q_c).sum(dim=-1)
                    dg_c = dg_c.to(datatype).to(calc_type)

                    dg_c -= (k_c * dk_from_state).sum(dim=-1)
                    dg_c = dg_c.to(gtype).to(calc_type)

                    dg_last_accum += (dk_from_state * k_c).sum()

                elif g_gamma is not None:
                    arange = torch.arange(actual_chunk_len, device=device, dtype=q.dtype)
                    g_c = gamma_val * (arange + 1)
                    g_last = gamma_val * actual_chunk_len

                    dq_from_state = dq_from_state * torch.exp(g_c)[:, None] * scale
                    dk_from_state = dk_from_state * torch.exp(-g_c + g_last)[:, None]
                else:
                    dk_from_state = dk_from_state * scale
                    dq_from_state = dq_from_state * scale

                # -----------------------------------------------------------
                # 3. Intra-chunk Attention
                # -----------------------------------------------------------
                ds = do_c.to(calc_type).to(mmtype) @ v_c.transpose(-1, -2).to(calc_type).to(mmtype)  # [BT, BT]
                ds = ds.to(datatype).to(calc_type)

                i_indices = torch.arange(actual_chunk_len, device=device)[:, None]
                j_indices = torch.arange(actual_chunk_len, device=device)[None, :]
                mask = i_indices >= j_indices

                if g is not None:
                    decay_mat = torch.exp(g_c[:, None] - g_c[None, :])
                    ds = torch.where(mask, ds * decay_mat, torch.zeros_like(ds)) * scale

                    qk_t = q_c.to(calc_type).to(mmtype) @ k_c.transpose(-1, -2).to(calc_type).to(mmtype)
                    qk_t = qk_t.to(datatype).to(calc_type)

                    ds2 = ds * qk_t

                    dg_c += ds2.sum(dim=1)
                    dg_c = dg_c.to(gtype).to(calc_type)
                    dg_c -= ds2.sum(dim=0)

                    dg_c = dg_c.to(gtype)

                    if actual_chunk_len > 0:
                        dg_c[actual_chunk_len - 1] += dg_last_accum.to(gtype)

                    dg[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx] = dg_c

                elif g_gamma is not None:
                    decay_mat = torch.exp(g_c[:, None] - g_c[None, :])
                    ds = torch.where(mask, ds * decay_mat, torch.zeros_like(ds)) * scale

                else:
                    ds = torch.where(mask, ds, torch.zeros_like(ds))

                # -----------------------------------------------------------
                # 4. Final Accumulation for dq, dk
                # -----------------------------------------------------------
                dq_intra = ds.to(calc_type).to(mmtype) @ k_c.to(calc_type).to(mmtype)
                dq_intra = dq_intra.to(datatype).to(calc_type)

                dk_intra = ds.transpose(-1, -2).to(calc_type).to(mmtype) @ q_c.to(calc_type).to(mmtype)
                dk_intra = dk_intra.to(datatype).to(calc_type)

                if g is None and g_gamma is None:
                    dk_intra = dk_intra * scale
                    dq_total = (dq_from_state + dq_intra) * scale
                    dk_total = dk_from_state + dk_intra
                else:
                    dq_total = dq_from_state + dq_intra
                    dk_total = dk_from_state + dk_intra

                dq_hv[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = dq_total.to(datatype)
                dk_hv[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = dk_total.to(datatype)

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    if cu_seqlens is None:
        for b in range(B):
            process_sequence(b, 0, T, b, 0)
    else:
        chunk_location = torch.zeros(cu_seqlens.shape[0], dtype=torch.int64, device=device)

        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            seq_length = end - start
            if i == 0:
                chunk_start_token_idx = 0
            else:
                chunk_start_token_idx = chunk_location[i]
            chunk_end_token_idx = chunk_start_token_idx + (seq_length + chunk_size - 1) // chunk_size
            chunk_location[i + 1] = chunk_end_token_idx

            if B == 1:
                process_sequence(0, start, end, i, chunk_location[i])
            else:
                pass

    dq = dq_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)
    dk = dk_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)

    return dq, dk, dw, dg


def chunk_bwd_dqkwg_gpu_torch(
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
    fp64: bool = False,
    device: Optional[Union[int, str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    带转置的 GPU 标杆包装函数。
    负责 GPU 设备选择、输入张量搬运、布局转置，然后调用核心计算函数。

    输入张量布局与 executor 一致：
        q, k: [B, HK, T, K]；v, do, dv: [B, HV, T, V]
        g: [B, HV, T]；h, dh: [B, HV, num_chunks, K, V]；w: [B, HV, T, K]
    输出张量布局：
        dq, dk: [B, HK, T, K]；dw: [B, HV, T, K]；dg: [B, HV, T]
    """
    # ------------------------------------------------------------------
    # GPU 设备选择
    # ------------------------------------------------------------------
    if device is None:
        device = select_gpu_device(0)
    elif isinstance(device, int):
        device = select_gpu_device(device)
    elif isinstance(device, str):
        device = torch.device(device)
        if device.type != 'cuda':
            raise RuntimeError(f"设备类型必须为 cuda，当前为 {device.type}")
        print(f"已选择 GPU 设备: {device}")
    elif isinstance(device, torch.device):
        if device.type != 'cuda':
            raise RuntimeError(f"设备类型必须为 cuda，当前为 {device.type}")
        print(f"已选择 GPU 设备: {device}")
    else:
        raise RuntimeError(f"不支持的 device 参数类型: {type(device)}")

    # ------------------------------------------------------------------
    # 布局转置
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 将所有张量移动到 GPU
    # ------------------------------------------------------------------
    q_t, k_t, v_t, do_t, h_t, dh_t, w_t, g_t, dv_t, cu_seqlens_tensor = move_tensors_to_device(
        q_t, k_t, v_t, do_t, h_t, dh_t, w_t, g_t, dv_t, cu_seqlens_tensor, device
    )

    # ------------------------------------------------------------------
    # 调用核心计算函数（输入已在 GPU 上）
    # ------------------------------------------------------------------
    dq, dk, dw, dg = chunk_bwd_dqkwg_gpu(
        q_t, k_t, v_t, do_t, h_t, dh_t, w_t, g_t, dv_t, scale, cu_seqlens_tensor, chunk_size,
        fp64=fp64
    )

    dq = dq.transpose(1, 2).contiguous()
    dk = dk.transpose(1, 2).contiguous()
    dw = dw.transpose(1, 2).contiguous()
    if dg is not None:
        dg = dg.transpose(1, 2).contiguous()

    return dq, dk, dw, dg


if __name__ == "__main__":
    print("=" * 60)
    print("chunk_bwd_dqkwg GPU 标杆实现测试")
    print("=" * 60)

    # ----------------------------------------------------------
    # 测试参数配置
    # ----------------------------------------------------------
    B = 1
    HK = 4
    T = 128
    K = 128
    V = 128
    HV = 4       # n_ratio = HV // HK = 1
    chunk_size = 64
    scale = 1.0 / (K ** 0.5)

    qkv_dtype = torch.bfloat16
    g_dtype = torch.float32   # is_mix=True 时 g 使用 fp32

    num_chunks = (T + chunk_size - 1) // chunk_size

    print(f"\n测试配置:")
    print(f"  B={B}, HK={HK}, HV={HV}, T={T}, K={K}, V={V}")
    print(f"  chunk_size={chunk_size}, num_chunks={num_chunks}")
    print(f"  qkv 精度={qkv_dtype}, g 精度={g_dtype}")
    print(f"  scale={scale:.6f}")

    # ----------------------------------------------------------
    # 生成测试数据
    # 布局与 executor 一致：
    #   q, k: [B, HK, T, K]
    #   v, do, dv: [B, HV, T, V]
    #   g: [B, HV, T]
    #   h, dh: [B, HV, num_chunks, K, V]
    #   w: [B, HV, T, K]
    # ----------------------------------------------------------
    torch.manual_seed(42)

    q = torch.rand((B, HK, T, K), dtype=qkv_dtype)
    k = torch.rand((B, HK, T, K), dtype=qkv_dtype)
    v = torch.rand((B, HV, T, V), dtype=qkv_dtype)
    do = torch.rand((B, HV, T, V), dtype=qkv_dtype)
    dv = torch.rand((B, HV, T, V), dtype=qkv_dtype)
    w = torch.rand((B, HV, T, K), dtype=qkv_dtype)
    h = torch.rand((B, HV, num_chunks, K, V), dtype=qkv_dtype)
    dh = torch.rand((B, HV, num_chunks, K, V), dtype=qkv_dtype)

    # g 必须递减且为负数
    g = -torch.sort(torch.rand(B * T * HV) * 10, descending=False)[0].reshape((B, HV, T)).to(g_dtype)

    cu_seqlens = None   # is_fix=True，定长模式

    # ----------------------------------------------------------
    # 测试 1: fp64=False（calc_type=fp32，与 CPU 标杆一致）
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("测试 1: fp64=False（calc_type=fp32）")
    print("-" * 40)

    dq1, dk1, dw1, dg1 = chunk_bwd_dqkwg_gpu_torch(
        q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens, chunk_size,
        fp64=False
    )

    print(f"  dq 形状: {tuple(dq1.shape)}, 精度: {dq1.dtype}")
    print(f"  dk 形状: {tuple(dk1.shape)}, 精度: {dk1.dtype}")
    print(f"  dw 形状: {tuple(dw1.shape)}, 精度: {dw1.dtype}")
    print(f"  dg 形状: {tuple(dg1.shape)}, 精度: {dg1.dtype}")
    print(f"  dq 数值范围: [{dq1.float().min().item():.6f}, {dq1.float().max().item():.6f}]")
    print(f"  dk 数值范围: [{dk1.float().min().item():.6f}, {dk1.float().max().item():.6f}]")
    print(f"  dw 数值范围: [{dw1.float().min().item():.6f}, {dw1.float().max().item():.6f}]")
    print(f"  dg 数值范围: [{dg1.float().min().item():.6f}, {dg1.float().max().item():.6f}]")
    print("  测试 1 完成")

    # ----------------------------------------------------------
    # 测试 2: fp64=True（使用 fp64 精度小算子拼接）
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("测试 2: fp64=True（使用 fp64 精度）")
    print("-" * 40)

    dq2, dk2, dw2, dg2 = chunk_bwd_dqkwg_gpu_torch(
        q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens, chunk_size,
        fp64=True
    )

    print(f"  dq 形状: {tuple(dq2.shape)}, 精度: {dq2.dtype}")
    print(f"  dk 形状: {tuple(dk2.shape)}, 精度: {dk2.dtype}")
    print(f"  dw 形状: {tuple(dw2.shape)}, 精度: {dw2.dtype}")
    print(f"  dg 形状: {tuple(dg2.shape)}, 精度: {dg2.dtype}")
    print(f"  dq 数值范围: [{dq2.min().item():.6f}, {dq2.max().item():.6f}]")
    print(f"  dk 数值范围: [{dk2.min().item():.6f}, {dk2.max().item():.6f}]")
    print(f"  dw 数值范围: [{dw2.min().item():.6f}, {dw2.max().item():.6f}]")
    print(f"  dg 数值范围: [{dg2.min().item():.6f}, {dg2.max().item():.6f}]")
    print("  测试 2 完成")

    # ----------------------------------------------------------
    # 精度对比
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("精度对比: fp64=False vs fp64=True")
    print("-" * 40)

    dq_diff = (dq1.float() - dq2.float()).abs()
    dk_diff = (dk1.float() - dk2.float()).abs()
    dw_diff = (dw1.float() - dw2.float()).abs()
    dg_diff = (dg1.float() - dg2.float()).abs()

    print(f"  dq 最大绝对差异: {dq_diff.max().item():.8f}")
    print(f"  dk 最大绝对差异: {dk_diff.max().item():.8f}")
    print(f"  dw 最大绝对差异: {dw_diff.max().item():.8f}")
    print(f"  dg 最大绝对差异: {dg_diff.max().item():.8f}")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
