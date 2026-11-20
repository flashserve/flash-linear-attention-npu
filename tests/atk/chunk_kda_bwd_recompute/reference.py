"""Pure CPU input generation and sole recompute reference.

Extracted from the supplied executor without changing input ranges or GEMM
rounding. Packed sequence chunk boundaries and optional gk follow the API.
No NPU or ATK dependency. Candidate pending CPU alignment and policy acceptance.
"""
from __future__ import annotations
from typing import Any
import torch

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}
_RCP_LN2 = 1.4426950408889634


def _orig_dtype(name: str) -> torch.dtype:
    """把 case_spec 中的 dtype 名称转成 torch dtype。"""
    return _DTYPE_MAP.get(str(name).lower(), torch.bfloat16)


def _calc_dtype(name: str, high_precision: bool) -> torch.dtype:
    """CPU 高精度标杆使用 fp64，其余路径使用用例声明的原始精度。"""
    if high_precision:
        return torch.float64
    return _orig_dtype(name)


def _randn(shape, dtype_name: str, calc_dtype: torch.dtype, device: torch.device, seed: int, scale: float = 0.05):
    """生成确定性正态分布输入；先量化到原始 dtype，再转到计算 dtype。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.randn(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * float(scale)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)


def _rand(shape, dtype_name: str, calc_dtype: torch.dtype, device: torch.device, seed: int, low: float = 0.05, high: float = 0.95):
    """生成确定性均匀分布输入；常用于 beta 这类正值参数。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32)
    data = data * (float(high) - float(low)) + float(low)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)


def _zeros(shape, dtype_name: str, calc_dtype: torch.dtype, device: torch.device):
    """生成指定 dtype/device 的零张量。"""
    data = torch.zeros(tuple(int(x) for x in shape), dtype=calc_dtype, device=device)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype)


def _gate(shape, calc_dtype: torch.dtype, device: torch.device, seed: int):
    """生成沿 T 维单调递减的 GDN gate，避免 exp 溢出。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * 0.01 + 0.001
    data = -torch.cumsum(data, dim=-1)
    return data.to(calc_dtype).to(device)


def _kda_gate(shape, dtype_name: str, calc_dtype: torch.dtype, device: torch.device, seed: int):
    """生成 KDA gate，沿 token 维做 chunk 前的稳定递减。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * 0.01 + 0.001
    data = -torch.cumsum(data, dim=-2)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)



def _bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if value is None:
        return default
    return bool(value)


def _expand_hk_to_hv(tensor: torch.Tensor, hv: int) -> torch.Tensor:
    hk = tensor.shape[1]
    group = max(int(hv) // int(hk), 1)
    if group == 1:
        return tensor
    return tensor.repeat_interleave(group, dim=1)


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    if dtype_name == "fp16":
        dtype_name = "bf16"
    g_dtype = str(spec.get("g_dtype", dtype_name)).lower()
    beta_dtype = str(spec.get("beta_dtype", dtype_name)).lower()
    calc_q = _calc_dtype(dtype_name, high_precision)
    calc_g = _calc_dtype(g_dtype, high_precision)
    calc_beta = _calc_dtype(beta_dtype, high_precision)
    calc_fp = torch.float64 if high_precision else torch.float32
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec.get("chunk_size", 64))
    inputs = {
        "q": _randn((B, HK, T, K), dtype_name, calc_q, device, seed + 1, 1.0),
        "k": _randn((B, HK, T, K), dtype_name, calc_q, device, seed + 2, 1.0),
        "v": _randn((B, HV, T, V), dtype_name, calc_q, device, seed + 3, 1.0),
        "g": _kda_gate((B, HV, T, K), g_dtype, calc_g, device, seed + 4),
        "beta": _rand((B, HV, T), beta_dtype, calc_beta, device, seed + 5, 0.1, 0.9),
        "A": _randn((B, HV, T, chunk_size), dtype_name, calc_q, device, seed + 6, 1.0),
        "A_log": _randn((HV,), "fp32", calc_fp, device, seed + 7, 0.05),
        "chunk_size": chunk_size,
        "use_gate": _bool(spec.get("use_gate", True)),
        "use_exp2": _bool(spec.get("use_exp2", True)),
        "lower_bound": float(spec.get("lower_bound", -5.0)),
    }
    if _bool(spec.get("has_dt_bias", True)):
        inputs["dt_bias"] = _randn((HV, K), "fp32", calc_fp, device, seed + 8, 0.05)
    if spec.get("variable", False):
        lengths = [int(x) for x in spec["lengths"]]
        if B != 1 or any(x < 0 for x in lengths) or sum(lengths) != T:
            raise ValueError("packed lengths must be nonnegative and sum to T with B=1")
        cu = [0]
        indices = []
        for seq, length in enumerate(lengths):
            cu.append(cu[-1] + length)
            for chunk in range((length + chunk_size - 1)//chunk_size):
                indices.extend((seq,chunk))
        inputs["cu_seqlens"] = cu
        inputs["chunk_indices"] = indices
    return inputs


def _fused_ref(inputs: dict[str, Any]):
    q, k, v, g, beta, a = (inputs[name] for name in ("q", "k", "v", "g", "beta", "A"))
    a_log = inputs["A_log"]
    dt_bias = inputs.get("dt_bias")
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    q = q.to(calc)
    k = k.to(calc)
    v = v.to(calc)
    g = g.to(calc)
    beta = beta.to(calc)
    a = a.to(calc)
    a_log = a_log.to(calc)
    if dt_bias is not None:
        dt_bias = dt_bias.to(calc)

    hv = g.shape[1]
    tokens = g.shape[2]
    chunk_size = int(inputs["chunk_size"])
    use_gate = _bool(inputs["use_gate"])
    use_exp2 = _bool(inputs["use_exp2"])
    lower_bound = float(inputs["lower_bound"])

    x = g
    if dt_bias is not None:
        x = x + dt_bias.view(1, hv, 1, -1)
    if use_gate:
        eig = torch.exp(a_log.view(1, hv, 1, 1))
        g_corr = lower_bound * torch.sigmoid(eig * x)
    else:
        g_corr = g
    scale = _RCP_LN2 if use_exp2 else 1.0
    cu = inputs.get("cu_seqlens", [0,tokens])
    ranges = [(start,min(start+chunk_size,hi))
              for lo,hi in zip(cu[:-1],cu[1:])
              for start in range(lo,hi,chunk_size)]
    gk = torch.empty_like(g_corr)
    for start, end in ranges:
        gk[:, :, start:end] = torch.cumsum(g_corr[:, :, start:end], dim=2) * scale

    e2 = torch.exp2(gk) if use_exp2 else torch.exp(gk)
    q_hv = _expand_hk_to_hv(q, hv)
    k_hv = _expand_hk_to_hv(k, hv)
    beta_k = beta.unsqueeze(-1)
    qg = q_hv * e2
    kbg = k_hv * beta_k * e2
    vb = v * beta_k
    orig = inputs["q"].dtype
    mm_dtype = orig if orig != torch.float64 else calc
    kg = torch.empty_like(k_hv)
    w = torch.empty((q.shape[0], hv, tokens, k.shape[-1]), dtype=calc, device=q.device)
    u = torch.empty((v.shape[0], hv, tokens, v.shape[-1]), dtype=calc, device=v.device)
    for start, end in ranges:
        gk_last = gk[:, :, end - 1 : end, :]
        delta = gk_last - gk[:, :, start:end]
        scale_k = torch.exp2(delta) if use_exp2 else torch.exp(delta)
        kg[:, :, start:end] = k_hv[:, :, start:end] * scale_k
        length = end - start
        # 同精度路径按 NPU cube 把 A/kbg/vb 量化到输出 dtype 再乘，避免 fp32 GEMM 把小值域误差压得过低。
        a_tile = a[:, :, start:end, :length].to(mm_dtype)
        kbg_tile = kbg[:, :, start:end].to(mm_dtype)
        vb_tile = vb[:, :, start:end].to(mm_dtype)
        w[:, :, start:end] = torch.matmul(a_tile, kbg_tile).to(calc)
        u[:, :, start:end] = torch.matmul(a_tile, vb_tile).to(calc)

    if inputs.get("return_intermediates", False):
        return {"gk": gk, "qg": qg.to(orig), "kg": kg.to(orig),
                "kbg": kbg.to(orig), "vb": vb.to(orig)}
    if orig == torch.float64:
        return gk if use_gate else None, w, u, qg, kg
    return (
        gk.to(torch.float32) if use_gate else None,
        w.to(orig),
        u.to(orig),
        qg.to(orig),
        kg.to(orig),
    )

