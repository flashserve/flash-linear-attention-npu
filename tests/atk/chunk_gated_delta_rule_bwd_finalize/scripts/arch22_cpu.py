"""A2/A3 Finalize 的独立 CPU 源码精度与 FP64 数学标杆。

仅依赖 NumPy。两个标杆共享已量化的公开输入，不共享输出或计算中间量。
源码精度链保留 BF16 落盘、FP32 Vector 和规约顺序；不声明硬件逐位等价。
FP64 标杆不调用源码精度链，也不把中间量降为 BF16。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

OUTPUT_NAMES = ("dq", "dk", "dv", "dbeta", "dg")


def prepare_inputs(inputs, gate_dtype):
    """把同一份公开 ABI 输入量化一次；不重造随机数据，不收窄值域。"""
    import json
    actual = {}
    main = ("q", "k", "v", "v_new", "do", "du", "h", "dh", "a")
    for name in main:
        actual[name] = cpu_bf16(inputs[name])
    for name in ("g", "beta", "beta_raw"):
        if inputs.get(name) is not None:
            actual[name] = (cpu_bf16(inputs[name]) if gate_dtype == "bf16"
                            else np.array(inputs[name], dtype=np.float32, copy=True))
    for name in ("q_rstd", "k_rstd"):
        if inputs.get(name) is not None:
            actual[name] = np.array(inputs[name], dtype=np.float32, copy=True)
    for name in ("scale", "chunk_size", "use_qk_l2_norm_in_kernel",
                 "use_beta_sigmoid_in_kernel", "use_gate_in_kernel", "state_v_first", "use_exp2"):
        actual[name] = inputs[name]
    for name in ("cu_seqlens", "chunk_indices"):
        value = inputs.get(name)
        if isinstance(value, (bytes, str)):
            value = json.loads(value)
        actual[name] = None if value is None else [int(n) for n in np.asarray(value).reshape(-1)]
    b, hk, t, kd = actual["q"].shape
    hv, vd = actual["v"].shape[1], actual["v"].shape[-1]
    cu = actual["cu_seqlens"]
    lengths = () if cu is None else tuple(y - x for x, y in zip(cu, cu[1:]))
    case = CpuCase("ATK actual inputs", b, hk, hv, t, int(actual["chunk_size"]), kd, vd,
                   bool(actual["use_qk_l2_norm_in_kernel"]), lengths, gate_dtype,
                   bool(actual["state_v_first"]), bool(actual["use_exp2"]),
                   bool(actual["use_beta_sigmoid_in_kernel"]), float(actual["scale"]))
    case.validate()
    data = {name: actual[name] for name in ("q", "k", "v", "do", "g", "beta", "a")}
    data.update(x=actual["v_new"], p=actual["du"])
    for name in ("q_rstd", "k_rstd", "beta_raw"):
        data[name] = actual.get(name)
    permutation = list(range(case.chunks_per_batch))
    if cu is not None:
        if cu[0] != 0 or cu[-1] != t or min(lengths) < 0:
            raise ValueError("CPU reference requires valid cu_seqlens")
        natural = [(seq, local) for seq, length in enumerate(lengths)
                   for local in range((length + case.chunk_size - 1) // case.chunk_size)]
        pairs = actual["chunk_indices"]
        if pairs is None:
            pairs = [n for pair in natural for n in pair]
            actual["chunk_indices"] = pairs
        listed = list(zip(pairs[::2], pairs[1::2]))
        if len(pairs) % 2 or sorted(listed) != sorted(natural):
            raise ValueError("CPU reference requires a complete, unique chunk permutation")
        permutation = [listed.index(pair) for pair in natural]
    for name in ("h", "dh"):
        state = actual[name]
        if case.state_v_first:
            state = np.swapaxes(state, -1, -2)
        # 公开 ABI [B,NT,HV,K,V]；公式按序列自然顺序读取状态。
        data[name] = np.take(state.transpose(0, 2, 1, 3, 4), permutation, axis=2)
    return case, data, actual

CPU_LOWP_SOURCE_COMMIT = "3e3f6ad5547edb6d95424566e01fee22bd832081"
REFERENCE_ARTIFACT_SHA256 = "891b42da51150735eb4306ebfd6b573d8fa58f21950a8febab33a5cc1386a340"

@dataclass(frozen=True)
class CpuCase:
    name: str
    batch: int
    key_heads: int
    value_heads: int
    tokens: int
    chunk_size: int = 64
    key_dim: int = 128
    value_dim: int = 128
    norm: bool = False
    seqlens: tuple[int,...] = ()
    gate_dtype: str = 'fp32'
    state_v_first: bool = False
    use_exp2: bool = False
    beta_sigmoid: bool = False
    scale: float = 0.08838834764831845

    def validate(self):
        if min(self.batch,self.key_heads,self.value_heads,self.tokens)<1 or self.value_heads%self.key_heads:
            raise ValueError('B/HK/HV/T 或分组头非法')
        if self.key_dim!=128 or self.value_dim not in (128,256) or self.chunk_size not in (64,128):
            raise ValueError('仅覆盖 K128/V128,256/BT64,128')
        if self.gate_dtype not in ('bf16','fp32'):
            raise ValueError('gate_dtype 非法')
        if self.seqlens and (self.batch!=1 or min(self.seqlens)<0 or sum(self.seqlens)!=self.tokens):
            raise ValueError('packed 序列边界非法')

    @property
    def ratio(self): return self.value_heads//self.key_heads

    @property
    def chunks_per_batch(self):
        return sum((n+self.chunk_size-1)//self.chunk_size for n in (self.seqlens or (self.tokens,)))


def chunks(case):
    case.validate()
    if case.seqlens:
        start=ci=0
        for length in case.seqlens:
            for local in range(0,length,case.chunk_size):
                yield 0,start+local,min(case.chunk_size,length-local),ci
                ci+=1
            start+=length
    else:
        for b in range(case.batch):
            for ci,start in enumerate(range(0,case.tokens,case.chunk_size)):
                yield b,start,min(case.chunk_size,case.tokens-start),ci


def finalize_cpu_fp64(case,data):
    """独立FP64数学式：同一量化输入，不模拟低精度中间落点。"""
    actual=dict(data)
    if case.use_exp2: actual['g']=data['g'].astype(np.float64)*math.log(2.0)
    outputs=finalize_fp64(case,actual)
    if case.beta_sigmoid:
        raw=data['beta_raw'].astype(np.float64)
        sigmoid=1.0/(1.0+np.exp(-raw))
        outputs['dbeta']=outputs['dbeta']*sigmoid*(1.0-sigmoid)
    return outputs


def chunk_formula(q, k, v, x, do, p, g, beta, a, h, dh, scale: float):
    """一个 value head、一个 chunk；输入可为任意正特征维。"""
    q, k, v, x, do, p, g, beta, a, h, dh = (
        np.asarray(value, dtype=np.float64)
        for value in (q, k, v, x, do, p, g, beta, a, h, dh)
    )
    m = q.shape[0]
    aa = np.exp(g)
    bb = np.exp(g[-1] - g)
    e = np.exp(g[:, None] - g[None, :])
    f = scale * np.tril((do @ x.T) * e)
    dq_h = scale * (aa[:, None] * do) @ h.T
    dk_h = (bb[:, None] * x) @ dh.T
    dq = dq_h + f @ k
    dk1 = dk_h + f.T @ q
    dw = -(p @ h.T)
    j = (beta * aa)[:, None] * k
    y = beta[:, None] * v
    da_raw = dw @ j.T + p @ y.T
    dl = -np.tril(a.T @ np.tril(da_raw, k=-1) @ a.T, k=-1)
    mm = dl * e
    dj = a.T @ dw
    psi = a.T @ p
    nw = beta[:, None] * (mm * (k @ k.T))
    dv = beta[:, None] * psi
    mk = mm @ k
    dk2 = (beta * aa)[:, None] * dj + beta[:, None] * mk + mm.T @ (beta[:, None] * k)
    dbeta = np.sum(dj * (aa[:, None] * k), axis=1)
    dbeta += np.sum(psi * v, axis=1) + np.sum(mk * k, axis=1)
    dg1 = np.sum(dq * q, axis=1) - np.sum(dk1 * k, axis=1)
    dg1[-1] += aa[-1] * np.sum(h * dh) + np.sum(dk_h * k)
    dg2 = np.sum(dj * j, axis=1) + np.sum(nw, axis=1) - np.sum(nw, axis=0)
    dgate = dg1 + dg2
    dg = np.cumsum(dgate[::-1])[::-1]
    return {"dq": dq, "dk": dk1 + dk2, "dv": dv, "dbeta": dbeta, "dg": dg}


def finalize_fp64(case: CpuCase, data: dict[str, np.ndarray | None]) -> dict[str, np.ndarray]:
    case.validate()
    b, hk, hv, t, dk, dv = (
        case.batch, case.key_heads, case.value_heads, case.tokens, case.key_dim, case.value_dim
    )
    required = {
        "q": (b, hk, t, dk), "k": (b, hk, t, dk),
        "v": (b, hv, t, dv), "x": (b, hv, t, dv),
        "do": (b, hv, t, dv), "p": (b, hv, t, dv),
        "g": (b, hv, t), "beta": (b, hv, t),
        "a": (b, hv, t, case.chunk_size),
        "h": (b, hv, case.chunks_per_batch, dk, dv),
        "dh": (b, hv, case.chunks_per_batch, dk, dv),
    }
    for name, shape in required.items():
        value = data.get(name)
        if value is None or value.shape != shape:
            raise ValueError(f"{name}: 期望 {shape}，实际 {None if value is None else value.shape}")
    for name in ("q_rstd", "k_rstd"):
        if case.norm and (data.get(name) is None or data[name].shape != (b, hk, t)):
            raise ValueError(f"{name}: L2Norm 用例需要 [B,HK,T]")
    output = {
        "dq": np.zeros((b, hk, t, dk), dtype=np.float64),
        "dk": np.zeros((b, hk, t, dk), dtype=np.float64),
        "dv": np.zeros((b, hv, t, dv), dtype=np.float64),
        "dbeta": np.zeros((b, hv, t), dtype=np.float64),
        "dg": np.zeros((b, hv, t), dtype=np.float64),
    }
    scale = float(case.scale)
    for batch, start, m, ci in chunks(case):
        tokens = slice(start, start + m)
        for head in range(hv):
            key_head = head // case.ratio
            result = chunk_formula(
                data["q"][batch, key_head, tokens], data["k"][batch, key_head, tokens],
                data["v"][batch, head, tokens], data["x"][batch, head, tokens],
                data["do"][batch, head, tokens], data["p"][batch, head, tokens],
                data["g"][batch, head, tokens], data["beta"][batch, head, tokens],
                data["a"][batch, head, tokens, :m], data["h"][batch, head, ci],
                data["dh"][batch, head, ci], scale,
            )
            output["dq"][batch, key_head, tokens] += result["dq"]
            output["dk"][batch, key_head, tokens] += result["dk"]
            for name in ("dv", "dbeta", "dg"):
                output[name][batch, head, tokens] = result[name]
    if case.norm:
        for key, rstd_key, value_key in (
            ("dq", "q_rstd", "q"), ("dk", "k_rstd", "k")
        ):
            xhat = np.asarray(data[value_key], dtype=np.float64)
            grad = output[key]
            rstd = np.asarray(data[rstd_key], dtype=np.float64)
            output[key] = rstd[..., None] * (
                grad - xhat * np.sum(xhat * grad, axis=-1, keepdims=True)
            )
    return output


def cpu_lowp_precision_ledger():
    """可直接写入执行回执的精度链与未证明边界。"""
    return {
        "source_commit": CPU_LOWP_SOURCE_COMMIT,
        "implementation": "independent_numpy_fp32_bf16_source_trace",
        "formal_bit_exact": False,
        "input_main": "q/k/v/x/do/p/a/h/dh: BF16, NumPy FP32 representation",
        "input_gate": "g/beta/beta_raw: case.gate_dtype BF16 or FP32",
        "state_layout": "canonical data [B,HV,NT,K,V]; state_v_first only changes DUT ABI",
        "stages": [
            {"source": "chunk_gated_delta_rule_bwd_finalize_a2.h:182-230",
             "precision": "S0 GT->FP32; use_exp2: g*=float32(ln(2)); no intermediate BF16"},
            {"source": "dq/chunk_bwd_dqkwg_cube.h:292-453,572-599",
             "precision": "DW/MM3/DS/MM4/MM6/MM5/MM7: BF16 operands, FP32 GEMM, BF16 materialization"},
            {"source": "dq/chunk_bwd_dqkwg_vector.h:455-525;dq/chunk_bwd_dqkwg_common.h:118-133",
             "precision": "H*DH FP32 block accumulation then serial 64-lane fold; DW BF16->FP32 negation->BF16"},
            {"source": "dq/chunk_bwd_dqkwg_vector.h:350-388,620-668",
             "precision": "DS FP32 gated value retained for dg; separate BF16 DS feeds MM6/MM7; row and serial column sums"},
            {"source": "dq/chunk_bwd_dqkwg_vector.h:402-411,902-1006,1290-1474",
             "precision": "DQ/DK state FP32; GQA adds state then same-head MM6/MM7 in increasing HV; BF16 only at group end"},
            {"source": "wy/prepare_wy_repr_bwd_vector.h:481-549,576-658;wy/prepare_wy_repr_bwd_cube.h:77-94",
             "precision": "Kbg/Kbeta/Vb and Dkbg/Dvb/KKT/DA1/DA2/DA4/DA5/DA6T/D/Dkb/DK each materialize BF16"},
            {"source": "wy/prepare_wy_repr_bwd_vector.h:378-386,476-479",
             "precision": "WY internal gType=float: g/beta local FP32 copies use Adds(+0), independently of S0 public GT conversion"},
            {"source": "wy/prepare_wy_repr_bwd_vector.h:702-851",
             "precision": "FP32 dbeta: Dkb*K then (Dkbg*exp(g))*K then Dvb*V; DK local and preceding BF16 group result added, BF16 every HV"},
            {"source": "wy/prepare_wy_repr_bwd_vector.h:854-1024;chunk_gated_delta_rule_bwd_finalize_a2.h:111-159",
             "precision": "D*KKT remains FP32; BT64 full-chunk ceil-half column tree then beta; BT128 per-rowOwned tree then beta then ordered block adds"},
            {"source": "chunk_gated_delta_rule_bwd_finalize_a2.h:318-394",
             "precision": "dg1+dg2 FP32, serial reverse chunk scan from +0; optional FP32 sigmoid chain on dbeta; only then public GT cast"},
            {"source": "chunk_gated_delta_rule_bwd_finalize_a2.h:413-437,498-515",
             "precision": "DK=BF16(dk_DQ_BF16+dk_WY_BF16); optional FP32 norm consumes this BF16; final DQ/DK BF16"},
        ],
        "reduction_boundary": (
            "Explicit source folds, 64-column panels, serial columns, GQA order, "
            "ceil-half row trees and reverse scan are modeled. An individual "
            "WholeReduceSum is np.sum(dtype=float32), without asserting its "
            "instruction-internal tree. BLAS FP32 accumulation may differ from Cube MMA."
        ),
        "transcendental_boundary": "np.exp FP32 / np.divide FP32 are source-order CPU approximations, not Ascend instruction bit emulation",
        "special_value_boundary": "RNE BF16 conversion covers finite IEEE FP32; device FTZ/NaN payload behavior is not established by this model",
        "output": "all arrays np.float32; dq/dk/dv represent BF16; dbeta/dg represent case.gate_dtype",
    }


def cpu_f32(value):
    return np.asarray(value, dtype=np.float32)


def cpu_add(left, right):
    return np.add(left, right, dtype=np.float32)


def cpu_sub(left, right):
    return np.subtract(left, right, dtype=np.float32)


def cpu_mul(left, right):
    return np.multiply(left, right, dtype=np.float32)


def cpu_exp(value):
    return np.exp(cpu_f32(value)).astype(np.float32, copy=False)


def cpu_bf16(value):
    """CAST_RINT：FP32->BF16 最近偶数；返回独立 FP32 表示。"""
    bits = np.array(value, dtype=np.float32, copy=True).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + bias) & np.uint32(0xFFFF0000)
    # 不把极小 NaN payload 的输入误舍入成 Inf；本矩阵只使用有限输入。
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    rounded = np.where(nan, (bits & np.uint32(0xFFFF0000)) | np.uint32(0x00400000), rounded)
    return np.asarray(rounded, dtype=np.uint32).view(np.float32)


def cpu_mm_bf16(left, right):
    """Cube L0C FP32 -> BF16 GM；BLAS 内部 MMA 顺序尚未设备逐位证明。"""
    return cpu_bf16(np.matmul(cpu_f32(left), cpu_f32(right)))


def cpu_whole_reduce(value):
    """单个 WholeReduceSum 的 FP32 近似；不猜测硬件内部加法树。"""
    return np.sum(cpu_f32(value), axis=-1, dtype=np.float32)


def cpu_dq_reduce128(product):
    # dq/vector.h:762-764,1119-1121：先逐列合并两半，再归约 64 项。
    return cpu_whole_reduce(cpu_add(product[..., :64], product[..., 64:128]))


def cpu_dq_reduce_m(value):
    # dq/vector.h:647-660,1127-1135：m>64 时尾段逐项加回前 64 项。
    value = cpu_f32(value)
    count = value.shape[-1]
    if count > 64:
        folded = value[..., :64].copy()
        folded[..., :count - 64] = cpu_add(folded[..., :count - 64], value[..., 64:])
        return cpu_whole_reduce(folded)
    return cpu_whole_reduce(value)


def cpu_wy_reduce_panels(product):
    # wy/vector.h:714-726,764-776,825-837,907-917,958-974。
    # 每个 64 列 WholeReduceSum 独立，之后再归约 partial。gate 尾列显式置零。
    product = cpu_f32(product)
    count = product.shape[-1]
    width = ((count + 63) // 64) * 64
    if count != width:
        padded = np.zeros(product.shape[:-1] + (width,), dtype=np.float32)
        padded[..., :count] = product
        product = padded
    partials = cpu_whole_reduce(product.reshape(product.shape[:-1] + (width // 64, 64)))
    return cpu_whole_reduce(partials)


def cpu_column_serial(product):
    # dq/vector.h:655-663：repeat 的目标行不前进，按 row1,row2,... 原地累加。
    result = cpu_f32(product[0]).copy()
    for row in range(1, product.shape[0]):
        result = cpu_add(product[row], result)
    return result


def cpu_column_ceil_half(product):
    # wy/vector.h:924-929,989-1017：不补到二次幂，按 ceil(remain/2) 折叠。
    work = cpu_f32(product).copy()
    remain = work.shape[0]
    while remain > 1:
        pairs = remain // 2
        next_remain = (remain + 1) // 2
        work[:pairs] = cpu_add(work[:pairs], work[next_remain:next_remain + pairs])
        remain = next_remain
    return work[0].copy()


def cpu_hdh_sum(h, dh):
    # dq/vector.h:427,455-495；common.h:118-133。
    # 先跨 K 分块对齐累加到 4096 项，再按 64 lanes 顺序折叠，最后 WholeReduceSum。
    rows = 16 if h.shape[1] == 256 else 32
    accum = cpu_mul(h[:rows], dh[:rows]).reshape(-1)
    for row in range(rows, 128, rows):
        accum = cpu_add(accum, cpu_mul(h[row:row + rows], dh[row:row + rows]).reshape(-1))
    lanes = accum[:64].copy()
    for start in range(64, accum.size, 64):
        lanes = cpu_add(accum[start:start + 64], lanes)
    return cpu_whole_reduce(lanes)


def cpu_vector_rows(bt, cols, factor):
    # chunk_gated_delta_rule_bwd_finalize_a2.h:111-123；不能把 BT128 的分块写死为 32。
    rows = bt // 2
    while rows >= 8:
        scalar_bytes = ((rows * 4 + 31) // 32) * 32
        if rows * cols * 2 <= 16 * 1024 and factor * rows * cols * 4 + 3 * scalar_bytes + rows * 32 <= 128 * 1024:
            break
        rows //= 2
    return rows


def cpu_chunk_dq(q, k, x, do, p, g, h, dh, scale, qkt):
    """返回独立 FP32 state、BF16 MM6/7/DW 和 FP32 dg_DQ。"""
    # dq/cube.h:292-359：所有 Cube 输出先物化 BF16。
    dw = cpu_bf16(cpu_mul(cpu_mm_bf16(p, h.T), np.float32(-1)))
    ds = cpu_mm_bf16(do, x.T)
    m = q.shape[0]
    # dq/vector.h:350-388：先 min(delta,0)，再 Exp，再乘含 scale 的下三角 mask。
    delta = cpu_add(cpu_mul(g[None, :], np.float32(-1)), g[:, None])
    factor = cpu_mul(cpu_exp(np.minimum(delta, np.float32(0))),
                     np.where(np.tri(m, dtype=bool), scale, np.float32(0)))
    ds_fp32 = cpu_mul(ds, factor)
    ds_bf16 = cpu_bf16(ds_fp32)  # 独立 GM 物化；dg 不从这个舍入值重建。
    score_product = cpu_mul(ds_fp32, qkt)
    dg = cpu_sub(cpu_dq_reduce_m(score_product), cpu_column_serial(score_product))

    # dq/cube.h:365-453；vector.h:402-411,756-766,1094-1150。
    dq_state = cpu_mul(cpu_mm_bf16(do, h.T), cpu_mul(cpu_exp(g), scale)[:, None])
    dg = cpu_add(dg, cpu_dq_reduce128(cpu_mul(dq_state, q)))
    dk_factor = cpu_exp(cpu_add(cpu_mul(g, np.float32(-1)), g[-1]))
    dk_state = cpu_mul(cpu_mm_bf16(x, dh.T), dk_factor[:, None])
    dk_dot = cpu_dq_reduce128(cpu_mul(k, dk_state))
    dg = cpu_sub(dg, dk_dot)
    last = cpu_add(cpu_mul(cpu_hdh_sum(h, dh), cpu_exp(g[-1])), cpu_dq_reduce_m(dk_dot))
    dg[-1] = cpu_add(dg[-1], last)
    return (dq_state, cpu_mm_bf16(ds_bf16, k), dk_state,
            cpu_mm_bf16(ds_bf16.T, q), dw, dg)


def cpu_chunk_wy(k, v, p, g, beta, a, dw, kkt, bt):
    """一个 HV 的 WY；dk_local 尚未执行逐 HV 的公开 BF16 累加。"""
    # wy/vector.h:378-386,476-479：内部 gType 固定 float，局部 copy 仍逐项 Adds(+0)。
    # 不能把这一层提前放进 S0，否则会改变 DQ 消费的 BF16->FP32 负零。
    g, beta = cpu_add(g, np.float32(0)), cpu_add(beta, np.float32(0))
    # wy/vector.h:481-549；cube.h:77-94,698-826：输入缩放和每个 GEMM 都落 BF16。
    eg = cpu_exp(g)
    kbg = cpu_bf16(cpu_mul(k, cpu_mul(beta, eg)[:, None]))
    kbeta = cpu_bf16(cpu_mul(k, beta[:, None]))
    vb = cpu_bf16(cpu_mul(v, beta[:, None]))
    dkbg = cpu_mm_bf16(a.T, dw)
    dvb = cpu_mm_bf16(a.T, p)
    da1 = cpu_mm_bf16(dw, kbg.T)
    da2 = cpu_mm_bf16(p, vb.T)
    # wy/vector.h:576-593；cube.h:1025-1178：严格下三角 -> DA4@A.T -> DA5.T@A。
    da4 = cpu_bf16(np.tril(cpu_add(da1, da2), k=-1))
    da5 = cpu_mm_bf16(da4, a.T)
    da6t = cpu_mm_bf16(da5.T, a)
    # D 是转置后的上三角表示；wy/vector.h:629-658，非 dl 的下三角表示。
    delta = cpu_sub(g[None, :], g[:, None])
    d_fp32 = cpu_mul(cpu_mul(da6t, np.float32(-1)), cpu_exp(np.minimum(delta, np.float32(0))))
    d = cpu_bf16(np.triu(d_fp32, k=1))
    dkb = cpu_mm_bf16(d.T, k)
    dk_mat = cpu_mm_bf16(d, kbeta)

    # wy/vector.h:702-840：dbeta 三项依此顺序；第二项也产生 dg 初值。
    dbeta = cpu_wy_reduce_panels(cpu_mul(dkb, k))
    dkbg_exp = cpu_mul(dkbg, eg[:, None])
    dkbg_dot = cpu_wy_reduce_panels(cpu_mul(dkbg_exp, k))
    dbeta = cpu_add(dbeta, dkbg_dot)
    dg = cpu_mul(dkbg_dot, beta)
    # wy/vector.h:783-804：DK + beta*Dkb + beta*(exp(g)*Dkbg)，之后才加历史组结果。
    dk_local = cpu_add(dk_mat, cpu_mul(dkb, beta[:, None]))
    dk_local = cpu_add(dk_local, cpu_mul(dkbg_exp, beta[:, None]))
    dbeta = cpu_add(dbeta, cpu_wy_reduce_panels(cpu_mul(dvb, v)))
    dv = cpu_bf16(cpu_mul(dvb, beta[:, None]))

    # KKT 是单独 BF16 Cube 结果，不能改写为利用舍入后的 DK/Dkb 反推。
    product = cpu_mul(d, kkt)
    row_negative = cpu_mul(cpu_wy_reduce_panels(cpu_mul(product, beta[None, :])), np.float32(-1))
    dg = cpu_add(dg, row_negative)
    if bt == 64:
        # wy/vector.h:924-936：完整有效 m 行按 ceil-half tree 归约，再乘 beta。
        column_term = cpu_mul(cpu_column_ceil_half(product), beta)
    else:
        # wy/vector.h:981-1024：每个 rowOwned 面板先 tree、再 beta，最后顺序累加面板。
        row_owned = min(cpu_vector_rows(bt, 128, 2), cpu_vector_rows(bt, v.shape[1], 2),
                        cpu_vector_rows(bt, bt, 3))
        column_term = cpu_mul(cpu_column_ceil_half(product[:row_owned]), beta)
        for start in range(row_owned, k.shape[0], row_owned):
            panel_term = cpu_mul(cpu_column_ceil_half(product[start:start + row_owned]), beta)
            column_term = cpu_add(column_term, panel_term)
    dg = cpu_add(dg, column_term)
    # WY gType 固定 float，CopyOutBetaGRows 使用 Adds(0)，尚未公开 GT cast。
    return dk_local, dv, cpu_add(dbeta, np.float32(0)), cpu_add(dg, np.float32(0))


def cpu_gate_tail(dg_dq, dg_wy, dbeta, beta_raw, gate_dtype):
    # a2.h:318-355：两个 FP32 工作区相加；严格反向串行 scan，独立 +0 种子。
    merged = cpu_add(dg_dq, dg_wy)
    scan = np.empty_like(merged, dtype=np.float32)
    acc = np.float32(0)
    for row in range(merged.size - 1, -1, -1):
        acc = cpu_add(acc, merged[row])
        scan[row] = acc
    if beta_raw is not None:
        # a2.h:363-382：-raw -> Exp -> +1 -> 1/x -> (1-s)，依次乘 dbeta*s*(1-s)。
        raw = cpu_add(beta_raw, np.float32(0)) if gate_dtype == "fp32" else cpu_f32(beta_raw)
        denom = cpu_add(cpu_exp(cpu_mul(raw, np.float32(-1))), np.float32(1))
        sigmoid = np.divide(np.ones_like(denom, dtype=np.float32), denom, dtype=np.float32)
        one_minus = cpu_sub(np.float32(1), sigmoid)
        dbeta = cpu_mul(cpu_mul(dbeta, sigmoid), one_minus)
    if gate_dtype == "bf16":
        return cpu_bf16(dbeta), cpu_bf16(scan)
    return cpu_f32(dbeta), scan


def cpu_norm(grad_bf16, x_bf16, rstd):
    # a2.h:413-437：两个 64 列 partial 再相加；先 grad-x*sum，后乘 rstd。
    dot = cpu_wy_reduce_panels(cpu_mul(x_bf16, grad_bf16))
    correction = cpu_mul(x_bf16, dot[:, None])
    return cpu_bf16(cpu_mul(cpu_sub(grad_bf16, correction), rstd[:, None]))


def cpu_chunks(case):
    bt = int(case.chunk_size)
    lengths = tuple(getattr(case, "seqlens", ()) or ())
    if lengths:
        offset = 0
        ci = 0
        for length in lengths:
            for local in range(0, int(length), bt):
                yield 0, offset + local, min(bt, int(length) - local), ci
                ci += 1
            offset += int(length)
    else:
        for batch in range(int(case.batch)):
            for ci, start in enumerate(range(0, int(case.tokens), bt)):
                yield batch, start, min(bt, int(case.tokens) - start), ci


def finalize_cpu_lowp(case, data):
    """独立执行 A2 FP32/BF16 源码链，返回 dq/dk/dv/dbeta/dg。

    主输入固定 BF16；实际 gate_dtype 支持 bf16/fp32。可选属性 use_exp2、
    beta_sigmoid 默认 False。state_v_first 不改变 data 的规范状态布局。
    data['g'] 是公开实际输入，exp2 单位转换在本函数内；beta 是实际 beta，
    beta_raw 在开启 beta_sigmoid 时必填，仅供末尾 dbeta sigmoid 导数使用。
    """
    validate = getattr(case, "validate", None)
    if callable(validate):
        validate()
    b, hk, hv, t, kd, vd, bt = (int(getattr(case, key)) for key in
        ("batch", "key_heads", "value_heads", "tokens", "key_dim", "value_dim", "chunk_size"))
    gate_dtype = str(getattr(case, "gate_dtype", "fp32")).lower()
    if min(b, hk, hv, t) <= 0 or hv % hk or kd != 128 or vd not in (128, 256) or bt not in (64, 128):
        raise ValueError("CPU lowp 合同为正 B/HK/HV/T、HV%HK=0、K128、V128/256、BT64/128")
    if gate_dtype not in ("bf16", "fp32"):
        raise ValueError("gate_dtype 必须为 bf16 或 fp32")
    lengths = tuple(getattr(case, "seqlens", ()) or ())
    if lengths and (b != 1 or min(lengths) < 0 or sum(lengths) != t):
        raise ValueError("packed 合同为 B1、非负 seqlens 且总和等于 T")
    nt = sum((int(n) + bt - 1) // bt for n in (lengths or (t,)))
    shapes = {
        "q": (b, hk, t, kd), "k": (b, hk, t, kd),
        "v": (b, hv, t, vd), "x": (b, hv, t, vd),
        "do": (b, hv, t, vd), "p": (b, hv, t, vd),
        "g": (b, hv, t), "beta": (b, hv, t),
        "a": (b, hv, t, bt), "h": (b, hv, nt, kd, vd), "dh": (b, hv, nt, kd, vd),
    }
    if bool(getattr(case, "norm", False)):
        shapes.update(q_rstd=(b, hk, t), k_rstd=(b, hk, t))
    if bool(getattr(case, "beta_sigmoid", False)):
        shapes["beta_raw"] = (b, hv, t)
    values = {}
    for name, shape in shapes.items():
        value = data.get(name)
        if value is None or tuple(np.shape(value)) != shape:
            raise ValueError(f"{name}: 期望 {shape}，实际 {None if value is None else np.shape(value)}")
        values[name] = cpu_f32(value)
    outputs = {
        "dq": np.zeros((b, hk, t, kd), dtype=np.float32),
        "dk": np.zeros((b, hk, t, kd), dtype=np.float32),
        "dv": np.zeros((b, hv, t, vd), dtype=np.float32),
        "dbeta": np.zeros((b, hv, t), dtype=np.float32),
        "dg": np.zeros((b, hv, t), dtype=np.float32),
    }
    ratio = hv // hk
    scale = np.float32(case.scale)
    for batch, start, m, ci in cpu_chunks(case):
        sl = slice(start, start + m)
        for key_head in range(hk):
            q = cpu_bf16(values["q"][batch, key_head, sl])
            k = cpu_bf16(values["k"][batch, key_head, sl])
            qkt = cpu_mm_bf16(q, k.T)
            kkt = cpu_mm_bf16(k, k.T)
            dq_sum = dk_sum = dk_wy_bf16 = None
            for head in range(key_head * ratio, (key_head + 1) * ratio):
                x, do, p, v = (cpu_bf16(values[key][batch, head, sl]) for key in ("x", "do", "p", "v"))
                h, dh = (cpu_bf16(values[key][batch, head, ci]) for key in ("h", "dh"))
                a = cpu_bf16(values["a"][batch, head, sl, :m])
                # S0 a2.h:206-230；g/beta 的公开 dtype 与内部 FP32 工作区分离。
                g = values["g"][batch, head, sl]
                beta = values["beta"][batch, head, sl]
                if gate_dtype == "bf16":
                    g, beta = cpu_bf16(g), cpu_bf16(beta)
                else:
                    g, beta = cpu_add(g, np.float32(0)), cpu_add(beta, np.float32(0))
                if bool(getattr(case, "use_exp2", False)):
                    g = cpu_mul(g, np.float32(0.6931471805599453))
                dqs, mm6, dks, mm7, dw, dg_dq = cpu_chunk_dq(q, k, x, do, p, g, h, dh, scale, qkt)
                # DQ GVA 在组内按 state、同 head MM6/MM7 交替加；只有组末 BF16。
                dq_sum = dqs.copy() if dq_sum is None else cpu_add(dq_sum, dqs)
                dq_sum = cpu_add(dq_sum, mm6)
                dk_sum = dks.copy() if dk_sum is None else cpu_add(dk_sum, dks)
                dk_sum = cpu_add(dk_sum, mm7)
                dk_local, dv, dbeta, dg_wy = cpu_chunk_wy(k, v, p, g, beta, a, dw, kkt, bt)
                # WY 每个 HV 都写回 BF16；下一 HV 读的就是这个已舍入结果。
                if dk_wy_bf16 is not None:
                    dk_local = cpu_add(dk_local, dk_wy_bf16)
                dk_wy_bf16 = cpu_bf16(dk_local)
                beta_raw = None
                if bool(getattr(case, "beta_sigmoid", False)):
                    beta_raw = values["beta_raw"][batch, head, sl]
                    if gate_dtype == "bf16":
                        beta_raw = cpu_bf16(beta_raw)
                dbeta, dg = cpu_gate_tail(dg_dq, dg_wy, dbeta, beta_raw, gate_dtype)
                outputs["dv"][batch, head, sl] = dv
                outputs["dbeta"][batch, head, sl] = dbeta
                outputs["dg"][batch, head, sl] = dg
            dq_bf16 = cpu_bf16(dq_sum)
            dk_bf16 = cpu_bf16(cpu_add(cpu_bf16(dk_sum), dk_wy_bf16))
            if bool(getattr(case, "norm", False)):
                dq_bf16 = cpu_norm(dq_bf16, q, values["q_rstd"][batch, key_head, sl])
                dk_bf16 = cpu_norm(dk_bf16, k, values["k_rstd"][batch, key_head, sl])
            outputs["dq"][batch, key_head, sl] = dq_bf16
            outputs["dk"][batch, key_head, sl] = dk_bf16
    return outputs
