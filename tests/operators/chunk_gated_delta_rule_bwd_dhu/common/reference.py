import math
import torch
import torch_npu
from typing import Optional, Tuple, List
import ct
import fla_npu

# fp16/bf16：在尽量无 NaN 的前提下抬高输出数量级（dv2 中位 ~1e-3；dh 非零元 |x| 最大可至 ~1e-3 量级）
_LOW_PRECISION_INPUT_HALF_RANGE_QK = 6.5e-3
_LOW_PRECISION_INPUT_HALF_RANGE_WO = 6.5e-3
_LOW_PRECISION_INPUT_HALF_RANGE_DV = 9e-3
# 乘在 effective_scale 之后；过小会导致 dh/dv2 过小（仅 1e-4～1e-10）
_LOW_PRECISION_SCALE_FACTOR = 0.92

def prepare_chunk_indices(cu_seqlens, chunk_size=64):
    """chunk indices生成函数(chunk_idx从0开始)"""
    chunk_indices = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx)
    return chunk_indices


def create_gate_g(B: int, Hv: int, T: int, gtype, narrow: bool = False):
    """
    构造 gate g：[B,Hv,T]（与 do/dv/w 的 value 头维一致），沿 T 单调递减，全为负。
    narrow=True 时区间更窄，减轻 fp16/bf16 上 exp / 差分 的数值压力（与 NPU 路径对齐）。
    """
    if narrow:
        lo, hi = -1e-2, -1e-6
    else:
        lo, hi = -5e-2, -5e-5
    span = hi - lo
    margin = max(span * 1e-7, 1e-12)
    g_t = torch.linspace(
        float(hi) - margin,
        float(lo) + margin,
        T,
        dtype=torch.float64,
    )
    g = g_t.unsqueeze(0).unsqueeze(0).expand(B, Hv, T).contiguous().to(gtype)
    return g


def generate_cu_seqlens(
    cu_seqlens_len: int,
    total_length: int,
    seg_min: int = 64,
    seg_max: int = 128,
) -> List[int]:
    """
    生成变长 cu_seqlens：总和为 total_length，尽量使每段长度在 [seg_min, seg_max]（默认 64~128）。

    - 当 64*B <= T <= 128*B 时，必可做到每段都在区间内且总和精确为 T；先公平切分再夹紧，用贪心加减1 修正总和。
    - 否则无法满足全在区间内：先夹紧再尽量修正总和（可能个别段略超出区间）。
    - 最后将 multiset 按「小配大」交错排列，减少相邻段长度相同或过于接近。
    """
    batchsize = cu_seqlens_len - 1
    if batchsize <= 0:
        return [0, total_length]

    B = batchsize
    T = total_length
    # 公平切分作为初值
    lengths = [
        (T * (i + 1)) // B - (T * i) // B
        for i in range(B)
    ]
    for i in range(B):
        lengths[i] = max(seg_min, min(seg_max, lengths[i]))

    diff = T - sum(lengths)
    # 总和不足：优先给当前较短的段 +1（方差增长慢）
    while diff > 0:
        cand = [i for i in range(B) if lengths[i] < seg_max]
        if not cand:
            break
        i = min(cand, key=lambda j: lengths[j])
        lengths[i] += 1
        diff -= 1
    # 总和过多：优先从当前较长的段 -1
    while diff < 0:
        cand = [i for i in range(B) if lengths[i] > seg_min]
        if not cand:
            break
        i = max(cand, key=lambda j: lengths[j])
        lengths[i] -= 1
        diff += 1
    # 仍无法对齐（T 超出 [seg_min*B, seg_max*B] 等）：强制在边界上继续调
    guard = 0
    while diff != 0 and guard < B * (seg_max - seg_min + 64):
        guard += 1
        if diff > 0:
            i = min(range(B), key=lambda j: lengths[j])
            lengths[i] += 1
            diff -= 1
        else:
            i = max(range(B), key=lambda j: lengths[j])
            lengths[i] -= 1
            diff += 1

    # 小配大交错， multiset 不变
    sorted_l = sorted(lengths)
    seq_lengths: List[int] = []
    i, j = 0, len(sorted_l) - 1
    while i <= j:
        if i == j:
            seq_lengths.append(sorted_l[i])
        else:
            seq_lengths.append(sorted_l[i])
            seq_lengths.append(sorted_l[j])
        i += 1
        j -= 1

    cu_seqlens = [0]
    for seq_len in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
    if cu_seqlens[-1] != total_length:
        raise ValueError(
            f"generate_cu_seqlens: 各段之和为 {cu_seqlens[-1]}，与期望 total_length={total_length} 不一致。"
            f" 当前 batchsize={batchsize}、seg 约束 [{seg_min},{seg_max}]，请增加序列条数或放宽/调整 T。"
        )
    return cu_seqlens


def rand_symmetric_uniform(shape, dtype: torch.dtype, half_range: float) -> torch.Tensor:
    """
    在 float32 上生成 [-half_range, half_range] 均匀分布再 cast，减轻 fp16/bf16 上直接 rand 的量化与溢出；
    half_range 为单边幅度（全区间宽度为 2*half_range）。
    """
    if half_range < 0:
        raise ValueError("half_range must be non-negative")
    x = torch.rand(shape, dtype=torch.float32, device="cpu")
    x = (x * 2.0 - 1.0) * float(half_range)
    return x.to(dtype=dtype)


def create_bwd_dhu_random_inputs(
    B: int,
    Hk: int,
    Hv: int,
    T: int,
    K: int,
    V: int,
    ktype: torch.dtype,
    gtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构造数值幅度受控的随机输入。
    - fp32：幅度可稍大，与 golden（float32 累加）一致时 NPU 通常也无 NaN。
    - fp16/bf16：自动收紧幅度并收窄 g，避免 K×V 大块 matmul 在昇腾 fp16 链上溢出为 NaN。
    """
    low = ktype in (torch.float16, torch.bfloat16)
    if low:
        hr_qk = _LOW_PRECISION_INPUT_HALF_RANGE_QK
        hr_wo = _LOW_PRECISION_INPUT_HALF_RANGE_WO
        hr_dv = _LOW_PRECISION_INPUT_HALF_RANGE_DV
        narrow_g = True
    else:
        hr_qk = 2e-2
        hr_wo = 2e-2
        hr_dv = 3e-2
        narrow_g = False

    q = rand_symmetric_uniform((B, Hk, T, K), ktype, half_range=hr_qk)
    k_tensor = rand_symmetric_uniform((B, Hk, T, K), ktype, half_range=hr_qk)
    w = rand_symmetric_uniform((B, Hv, T, K), ktype, half_range=hr_qk)
    d_o = rand_symmetric_uniform((B, Hv, T, V), ktype, half_range=hr_wo)
    dv = rand_symmetric_uniform((B, Hv, T, V), ktype, half_range=hr_dv)
    g = create_gate_g(B, Hv, T, gtype, narrow=narrow_g)
    return q, k_tensor, w, d_o, dv, g


def effective_scale(scale: float, K: int) -> float:
    """不超过 1/sqrt(K) 的缩放，与常见 attention scale 一致并抑制过大 matmul 输出。"""
    cap = 1.0 / math.sqrt(float(K))
    return float(min(scale, cap))


def scale_for_compute_dtype(scale: float, ktype: torch.dtype) -> float:
    """fp16/bf16 算子路径上对 scale 的额外系数：兼顾数值动态范围与长 T 下 cube 累加不溢出。"""
    if ktype in (torch.float16, torch.bfloat16):
        return float(scale * _LOW_PRECISION_SCALE_FACTOR)
    return float(scale)


def assert_no_nan_inf(t: torch.Tensor, name: str) -> None:
    """检查单个浮点/复数张量是否含 NaN/Inf（仅测一项时立即抛出）。"""
    if not (torch.is_floating_point(t) or torch.is_complex(t)):
        return
    if torch.isfinite(t).all():
        return
    n_bad = int((~torch.isfinite(t)).sum().item())
    raise RuntimeError(
        f"{name} 含 NaN 或 Inf，异常元素个数={n_bad}，shape={tuple(t.shape)}，dtype={t.dtype}"
    )


def _nonfinite_detail_line(t: torch.Tensor, name: str) -> Optional[str]:
    """若含非有限值则返回一行说明，否则返回 None。"""
    if not (torch.is_floating_point(t) or torch.is_complex(t)):
        return None
    if torch.isfinite(t).all():
        return None
    n_bad = int((~torch.isfinite(t)).sum().item())
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    return (
        f"  - {name}: 非有限元素={n_bad} (NaN={n_nan}, Inf={n_inf}), "
        f"shape={tuple(t.shape)}, dtype={t.dtype}"
    )


def assert_finite_after_npu_golden_compare(
    dh_npu: torch.Tensor,
    dv2_npu: torch.Tensor,
    dh_golden: torch.Tensor,
    dv2_golden: torch.Tensor,
) -> None:
    """须在完成 dh、dv2 的 NPU 与 CPU(golden) 对比之后调用；四项全部扫描后再统一抛出（若有任一异常）。"""
    pairs = (
        (dh_npu, "dh_npu"),
        (dv2_npu, "dv2_npu"),
        (dh_golden, "dh_golden"),
        (dv2_golden, "dv2_golden"),
    )
    lines: List[str] = []
    for t, nm in pairs:
        ln = _nonfinite_detail_line(t, nm)
        if ln is not None:
            lines.append(ln)
    if lines:
        raise RuntimeError(
            "以下输出含 NaN/Inf（已检查全部四项后再汇总）：\n" + "\n".join(lines)
        )


def chunk_gated_delta_rule_bwd_dhu_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens: Optional[List[int]] = None,
    chunk_indices: Optional[List[int]] = None,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    PyTorch 版 chunk_gated_delta_rule_bwd_dhu（GVA 形状）。
    - q, k: [B, Hk, T, K]
    - w, do, dv, (可选 g): [B, Hv, T, ·]；须 Hv % Hk == 0
    - value 头 h 对应 q/k 头 hq = h // (Hv // Hk)
    支持定长 (cu_seqlens=None) 与变长 (cu_seqlens!=None)。
    """
    device = q.device
    dtype = q.dtype

    B, Hk, T, K = q.shape
    Hv = do.shape[1]
    V = do.shape[-1]
    BT = chunk_size

    if k.shape[1] != Hk or k.shape[2] != T:
        raise ValueError(f"k 须为 [B,Hk,T,K]，与 q 一致；q {q.shape}, k {k.shape}")
    if w.shape[1] != Hv or dv.shape[1] != Hv or do.shape[1] != Hv:
        raise ValueError(f"w/d_o/dv 的 dim1 须为 Hv={Hv}；w {w.shape}, d_o {do.shape}, dv {dv.shape}")
    if Hk <= 0 or Hv % Hk != 0:
        raise ValueError(f"GVA 要求 Hv 为 Hk 的整数倍；Hk={Hk}, Hv={Hv}")

    hv_per_hk = Hv // Hk

    if cu_seqlens is not None:
        seq_total = cu_seqlens[-1]
        if seq_total > T:
            raise ValueError(
                f"cu_seqlens末元 {seq_total} 超过 q 的时间维 T={T}，无法对齐 [B,Hv,T,V] 的 dv2。"
            )
        NT = len(chunk_indices) // 2
    else:
        seq_total = T
        NT = (T + BT - 1) // BT

    if scale is None:
        scale = 1.0

    chunk_info = []
    for i_t in range(NT):
        if cu_seqlens is not None:
            i_n = chunk_indices[i_t * 2]
            block_idx_in_token = chunk_indices[i_t * 2 + 1]
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            token_length = eos - bos
        else:
            i_n = 0
            block_idx_in_token = i_t
            bos = 0
            token_length = T

        start_t = block_idx_in_token * BT
        end_t = min((block_idx_in_token + 1) * BT, token_length)
        block_size_t = end_t - start_t
        global_start_t = bos + start_t
        global_end_t = bos + end_t

        chunk_info.append({
            'i_t': i_t,
            'i_n': i_n,
            'block_idx_in_token': block_idx_in_token,
            'bos': bos,
            'token_length': token_length,
            'start_t': start_t,
            'end_t': end_t,
            'block_size_t': block_size_t,
            'global_start_t': global_start_t,
            'global_end_t': global_end_t,
        })

    dh = torch.zeros(B, Hv, NT, K, V, device=device, dtype=torch.float32)
    # dv2 与 dv/do 对齐为 [B,Hv,T,V]；变长时仅改写 [0, seq_total)，其余保留 dv（padding）
    if cu_seqlens is not None:
        dv2 = dv.clone()
    else:
        dv2 = torch.zeros(B, Hv, T, V, device=device, dtype=dv.dtype)

    if cu_seqlens is None:
        # 定长：对 (B, Hv) 向量化，仅沿 chunk 反向迭代。避免 B×Hv×NT 纯 Python 嵌套（大 B/Hv 时极慢甚至像卡死）。
        hq = torch.arange(Hv, device=device, dtype=torch.long) // hv_per_hk
        b_dh = torch.zeros(B, Hv, K, V, device=device, dtype=torch.float32)
        for i_t in range(NT - 1, -1, -1):
            info = chunk_info[i_t]
            global_start_t = info["global_start_t"]
            global_end_t = info["global_end_t"]
            block_size_t = info["block_size_t"]

            dh[:, :, i_t, :, :] = b_dh

            last_idx = min((info["block_idx_in_token"] + 1) * BT, info["token_length"]) - 1
            global_last_idx = info["bos"] + last_idx

            # 先切时间维再 index_select，避免每个 chunk 物化 [B,Hv,T,K] 整段（大 B/Hv/T 时极慢、像卡死）
            k_blk = k[:, :, global_start_t:global_end_t, :].index_select(1, hq)
            q_blk = q[:, :, global_start_t:global_end_t, :].index_select(1, hq)
            w_blk = w[:, :, global_start_t:global_end_t, :]
            b_do = do[:, :, global_start_t:global_end_t, :]
            b_dv_existing = dv[:, :, global_start_t:global_end_t, :]

            b_dv = torch.matmul(k_blk.to(torch.float), b_dh.to(torch.float))

            if g is not None:
                bg_last = g[:, :, global_last_idx]
                b_g = g[:, :, global_start_t:global_end_t]
                gate_factor = torch.exp(bg_last.unsqueeze(-1) - b_g).unsqueeze(-1)
                m_t = torch.arange(block_size_t, device=device, dtype=torch.float32) < float(block_size_t)
                mask_expanded = m_t.view(1, 1, block_size_t, 1)
                b_dv = b_dv * gate_factor * mask_expanded

            b_dv = b_dv + b_dv_existing.to(torch.float32)
            dv2[:, :, global_start_t:global_end_t, :] = b_dv.to(dv2.dtype)

            b_q_t = q_blk.transpose(-1, -2)
            b_w_t = w_blk.transpose(-1, -2)

            if g is not None:
                bg_last_exp = torch.exp(bg_last)
                b_g_exp = torch.exp(b_g)
                b_dh_for_update = b_dh * bg_last_exp.unsqueeze(-1).unsqueeze(-1)
                b_q_gated = b_q_t * b_g_exp.unsqueeze(-2)
            else:
                b_dh_for_update = b_dh.clone()
                b_q_gated = b_q_t

            term1 = torch.matmul(b_q_gated.to(torch.float), b_do.to(torch.float)) * scale
            term2 = torch.matmul(b_w_t.to(torch.float), b_dv.to(torch.float))
            b_dh = b_dh_for_update + term1 - term2
    else:
        for b in range(B):
            for i_h in range(Hv):
                hq = i_h // hv_per_hk
                num_tokens = len(cu_seqlens) - 1
                b_dh_buffers = {}
                for i_n in range(num_tokens):
                    b_dh_buffers[i_n] = torch.zeros(K, V, device=device, dtype=torch.float32)

                for i_t in range(NT - 1, -1, -1):
                    info = chunk_info[i_t]
                    i_n = info['i_n']
                    b_dh = b_dh_buffers[i_n]

                    dh[b, i_h, i_t] = b_dh

                    global_start_t = info['global_start_t']
                    global_end_t = info['global_end_t']
                    block_size_t = info['block_size_t']

                    last_idx = min((info['block_idx_in_token'] + 1) * BT, info['token_length']) - 1
                    global_last_idx = info['bos'] + last_idx

                    bg_last = bg_last_exp = b_g = b_g_exp = None
                    if g is not None:
                        bg_last = g[b, i_h, global_last_idx]
                        b_g = g[b, i_h, global_start_t:global_end_t]
                        bg_last_exp = torch.exp(bg_last)
                        b_g_exp = torch.exp(b_g)

                    b_do = do[b, i_h, global_start_t:global_end_t, :]
                    b_dv_existing = dv[b, i_h, global_start_t:global_end_t, :]

                    b_k = k[b, hq, global_start_t:global_end_t, :]
                    b_dv = torch.matmul(b_k.to(torch.float), b_dh.to(torch.float))

                    if g is not None:
                        m_t = torch.arange(block_size_t, device=device) < block_size_t
                        gate_factor = torch.exp(bg_last - b_g).unsqueeze(-1)
                        mask_expanded = m_t.unsqueeze(-1).float()
                        b_dv *= gate_factor * mask_expanded

                    b_dv += b_dv_existing.to(torch.float32)
                    dv2[b, i_h, global_start_t:global_end_t, :] = b_dv.to(dv2.dtype)

                    b_q = q[b, hq, global_start_t:global_end_t, :]
                    b_w = w[b, i_h, global_start_t:global_end_t, :]

                    b_q_t = b_q.transpose(0, 1)
                    b_w_t = b_w.transpose(0, 1)
                    b_dh_for_update = b_dh.clone()
                    if g is not None:
                        b_dh_for_update = b_dh_for_update * bg_last_exp

                    b_q_gated = b_q_t
                    if g is not None:
                        b_q_gated = b_q_t * b_g_exp.unsqueeze(0)

                    term1 = torch.matmul(b_q_gated.to(torch.float), b_do.to(torch.float)) * scale
                    term2 = torch.matmul(b_w_t.to(torch.float), b_dv.to(torch.float))

                    new_b_dh = b_dh_for_update + term1 - term2
                    b_dh_buffers[i_n] = new_b_dh

    return dh, None, dv2


def test_fix(
    B: int,
    Hk: int,
    Hv: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    scale: float,
    ktype,
    gtype,
    seed: int = 0,
):
    if Hv % Hk != 0:
        raise ValueError(f"GVA: 要求 Hv % Hk == 0，当前 Hk={Hk}, Hv={Hv}")
    scale = scale_for_compute_dtype(effective_scale(scale, K), ktype)
    print("B, T, Hk, Hv, K, V, chunk_size = ", B, T, Hk, Hv, K, V, chunk_size)

    torch.manual_seed(seed)
    if not hasattr(test_fix, "call_count"):
        test_fix.call_count = 1
    else:
        test_fix.call_count += 1

    q, k, w, d_o, dv, g = create_bwd_dhu_random_inputs(B, Hk, Hv, T, K, V, ktype, gtype)

    dh_golden, _, dv2_golden = chunk_gated_delta_rule_bwd_dhu_torch(
        q, k, w, d_o, dv,
        cu_seqlens=None, chunk_indices=None,
        g=g, scale=scale, chunk_size=chunk_size
    )
    print("dh_golden shape is", dh_golden.shape)
    print("dv2_golden shape is", dv2_golden.shape)

    q_npu = q.npu()
    k_npu = k.npu()
    w_npu = w.npu()
    d_o_npu = d_o.npu()
    dv_npu = dv.npu()
    g_npu = g.npu()

    dh_npu, dh0_npu, dv2_npu = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q_npu, k_npu, w_npu, d_o_npu, dv_npu, 
        scale=scale, 
        chunk_size=chunk_size,
        g=g_npu,
        gK=None, 
        h0=None, 
        dht=None,
        cu_seqlens=None, 
        chunk_indices=None,
        # use_exp2=False, 
        # transpose_state_layout=False
    )

    dh_npu_cpu = dh_npu.detach().cpu()
    dv2_npu_cpu = dv2_npu.detach().cpu()
    print("dh comparison:")
    ct.single(dh_npu_cpu, dh_golden)
    print("dv2 comparison:")
    ct.single(dv2_npu_cpu, dv2_golden)
    assert_finite_after_npu_golden_compare(dh_npu_cpu, dv2_npu_cpu, dh_golden, dv2_golden)
    print(f"test_fix 被调用了第 {test_fix.call_count} 次")


def test_variable(
    B: int,
    Hk: int,
    Hv: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    scale: float,
    cu_seqlens_len: int,
    ktype,
    gtype,
    seed: int = 0,
):
    if Hv % Hk != 0:
        raise ValueError(f"GVA: 要求 Hv % Hk == 0，当前 Hk={Hk}, Hv={Hv}")
    scale = scale_for_compute_dtype(effective_scale(scale, K), ktype)
    print("B, T, Hk, Hv, K, V, chunk_size = ", B, T, Hk, Hv, K, V, chunk_size)
    torch.manual_seed(seed)
    if not hasattr(test_variable, "call_count"):
        test_variable.call_count = 1
    else:
        test_variable.call_count += 1

    q, k, w, d_o, dv, g = create_bwd_dhu_random_inputs(B, Hk, Hv, T, K, V, ktype, gtype)

    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    print(cu_seqlens)
    dh_golden, _, dv2_golden = chunk_gated_delta_rule_bwd_dhu_torch(
        q, k, w, d_o, dv,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        g=g, scale=scale, chunk_size=chunk_size
    )


    q_npu = q.npu()
    k_npu = k.npu()
    w_npu = w.npu()
    d_o_npu = d_o.npu()
    dv_npu = dv.npu()
    g_npu = g.npu()

    dh_npu, dh0_npu, dv2_npu = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q_npu, k_npu, w_npu, d_o_npu, dv_npu, 
        scale=scale, 
        chunk_size=chunk_size,
        g=g_npu,
        gK=None, 
        h0=None, 
        dht=None,
        cu_seqlens=cu_seqlens, 
        chunk_indices=chunk_indices,
        # use_exp2=False, 
        # transpose_state_layout=False
    )

    dh_npu_cpu = dh_npu.detach().cpu()
    dv2_npu_cpu = dv2_npu.detach().cpu()
    print("dh_golden shape is", dh_golden.shape)
    print("dv2_golden shape is", dv2_golden.shape)
    print("dh_npu shape is", dh_npu_cpu.shape)
    print("dv2_npu shape is", dv2_npu_cpu.shape)
    print("dh comparison:")
    ct.single(dh_npu_cpu, dh_golden)
    print("dv2 comparison:")
    ct.single(dv2_npu_cpu, dv2_golden)

    # NPU 与 golden 对比全部完成后，再检查 NaN/Inf，最后落盘
    assert_finite_after_npu_golden_compare(dh_npu_cpu, dv2_npu_cpu, dh_golden, dv2_golden)

    # print(f"test_variable 被调用了第 {test_variable.call_count} 次")

if __name__ == "__main__":
    import os
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

    # GVA smoke: Hk=2, Hv=4
    test_fix(B=1, Hk=2, Hv=4, T=256, K=128, V=128, chunk_size=64, scale=0.088, ktype=torch.bfloat16, gtype=torch.bfloat16)
    # Gate state stays in the vector pipeline for both chunk sizes and g dtypes.
    test_fix(B=2, Hk=2, Hv=4, T=256, K=128, V=128, chunk_size=64, scale=0.088, ktype=torch.bfloat16, gtype=torch.float32)
    test_fix(B=1, Hk=2, Hv=4, T=320, K=128, V=128, chunk_size=128, scale=0.088, ktype=torch.bfloat16, gtype=torch.bfloat16)
    # MHA smoke: Hk=Hv
    test_fix(B=1, Hk=4, Hv=4, T=128, K=128, V=128, chunk_size=64, scale=0.088, ktype=torch.bfloat16, gtype=torch.bfloat16)
    # GVA varlen smoke
    test_variable(B=1, Hk=4, Hv=8, T=512, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=4, ktype=torch.bfloat16, gtype=torch.bfloat16)
