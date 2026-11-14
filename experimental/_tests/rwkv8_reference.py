"""ATK-free RWKV8 inputs/reference shared by manual validation, PTA and dump comparison.

Contract tests compare inputs and FP32/FP64 results with the self-contained ATK executor.
The accuracy golden uses dtype-quantized inputs and FP64 accumulation, then FP32 outputs.
"""
from __future__ import annotations
from typing import Any, NamedTuple, Optional
import torch


def _orig_dtype(name):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _rwkv8_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool) -> dict[str, Any]:
    """结构化造数（与仓外 ascendc/scripts/gen_data.py 同配方）：
    q/k/v ~ randn；w = -rand*2-0.1（保证 decay ∈ (0,1)）；
    kk = L2normalize(randn)，z = -kk，b = kk * randn。
    禁止无约束 randn 造 z/b（delta-rule 状态会指数爆炸）。

    生成在 CPU 上按 seed 确定，先量化到用例 dtype 再转到计算 dtype
    （CPU 高精度标杆 fp64 / 其余路径为用例原始 dtype），保证两个
    ATK 节点看到完全一致的输入。
    """
    dtype_name = str(spec.get("dtype", "fp32"))
    calc_dtype = torch.float64 if high_precision else _orig_dtype(dtype_name)
    seed = int(spec.get("seed", 42))
    B, H, T, K, V = (int(spec[x]) for x in ("B", "H", "T", "K", "V"))

    def _make(shape, offset, factory):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + offset)
        data = factory(shape, gen)
        return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)

    q = _make((B, H, T, K), 1, lambda s, g: torch.randn(s, generator=g))
    w = _make((B, H, T, K), 2, lambda s, g: -torch.rand(s, generator=g) * 2.0 - 0.1)
    k = _make((B, H, T, K), 3, lambda s, g: torch.randn(s, generator=g))
    v = _make((B, H, T, V), 4, lambda s, g: torch.randn(s, generator=g))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 5)
    kk = torch.nn.functional.normalize(torch.randn((B, H, T, K), generator=gen), p=2, dim=-1)
    z = (-kk).to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)
    b = (kk * torch.randn((B, H, T, K), generator=gen)).to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)

    initial_state = None
    if bool(spec.get("initial_state", False)):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 6)
        initial_state = torch.randn((B, H, K, V), generator=gen)
        initial_state = initial_state.to(torch.float64 if high_precision else torch.float32).to(device)

    return {
        "q": q, "w": w, "k": k, "v": v, "z": z, "b": b,
        "initial_state": initial_state,
        "scale": float(spec.get("scale", 1.0)),
        "chunk_len": int(spec.get("chunk_len", 16)),
        "output_chunk_state": bool(spec.get("output_chunk_state", False)),
        "output_sa": bool(spec.get("output_sa", False)),
    }


# CPU reference: BlinkDL/RWKV-LM @ 9521024, RWKV-v8/cuda/wkv7_cuda.cu forward_kernel.

class _Rwkv8Result(NamedTuple):
    o: torch.Tensor
    s: Optional[torch.Tensor]
    sa: Optional[torch.Tensor]


def _wkv7_decay(w: torch.Tensor) -> torch.Tensor:
    """log 域衰减参数 -> 衰减值：decay = exp(-exp(w))，与 wkv7_cuda.cu:21 一致。"""
    return torch.exp(-torch.exp(w))


def fused_recurrent_rwkv8_reference(
    q: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_chunk_state: bool = False,
    output_sa: bool = False,
    chunk_len: int = 16,
    reverse: bool = False,
) -> _Rwkv8Result:
    """逐 token 递推参考实现（einsum 版）。输入 fp64 时全程 fp64 累加，
    否则 fp32 累加；o 回投到输入 dtype，s/sa 为计算精度。"""
    orig_dtype = q.dtype
    calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    q, w, k, v, z, b = (x.to(calc_dtype) for x in (q, w, k, v, z, b))
    B, H, T, K = q.shape
    V = v.shape[-1]

    if initial_state is None:
        state = q.new_zeros(B, H, V, K)
    else:
        state = initial_state.to(calc_dtype).transpose(-1, -2).clone()  # 接口 (K,V) → 内部 (V,K)
    decay = _wkv7_decay(w)  # (B, H, T, K)

    o = torch.empty(B, H, T, V, dtype=calc_dtype)
    sa_out = torch.empty(B, H, T, V, dtype=calc_dtype) if output_sa else None
    s_snaps = []
    for i in range(T):
        t = T - 1 - i if reverse else i
        sa = torch.einsum('bhij,bhj->bhi', state, z[:, :, t])
        if output_sa:
            sa_out[:, :, t] = sa
        state = (state * decay[:, :, t].unsqueeze(-2)
                 + sa.unsqueeze(-1) * b[:, :, t].unsqueeze(-2)
                 + v[:, :, t].unsqueeze(-1) * k[:, :, t].unsqueeze(-2))
        o[:, :, t] = torch.einsum('bhij,bhj->bhi', state, q[:, :, t] * scale)
        if output_chunk_state and (t + 1) % chunk_len == 0:
            s_snaps.append((t // chunk_len, state.transpose(-1, -2)))

    o = o.to(orig_dtype)
    if output_chunk_state:
        s = q.new_zeros(B, H, T // chunk_len, K, V)
        for slot, snap in s_snaps:
            s[:, :, slot] = snap
    else:
        s = None
    return _Rwkv8Result(o, s, sa_out)


def compare_output(actual, golden):
    """mixed_tolerance_bm v1 thresholds used by ATK 26.9.8 for FP32 golden.

    Require >=99% isclose AND every element within the absolute-or-32-ULP
    envelope. Optional state/sa outputs are FP32, regardless of input dtype.
    This is deliberately not the older PTA relative-RMSE smoke criterion.
    """
    rtol, atol, max_abs = {
        torch.float16: (1.95e-3, 1.95e-3, 0.1),
        torch.bfloat16: (1.56e-2, 1.56e-2, 1.0),
        torch.float32: (9.77e-4, 1.53e-5, 0.01),
    }[actual.dtype]
    original_dtype = actual.dtype
    actual, golden = actual.cpu().float(), golden.cpu().float()
    if actual.shape != golden.shape:
        raise AssertionError(f"Output/reference shape mismatch: {actual.shape}/{golden.shape}")
    if not torch.isfinite(actual).all() or not torch.isfinite(golden).all():
        raise AssertionError("Non-finite NPU output or CPU reference")
    if actual.numel() == 0:
        return {"pass": True, "elements": 0}
    error = (actual - golden).abs()
    matched = torch.isclose(actual, golden, rtol=rtol, atol=atol, equal_nan=False)
    rounded = golden.to(original_dtype)
    spacing = (torch.nextafter(rounded, torch.full_like(rounded, float("inf"))) - rounded).float()
    envelope = (error <= max_abs) | (error <= 32 * spacing)
    ratio = float(matched.float().mean())
    return {"pass": ratio >= 0.99 and bool(envelope.all()), "elements": actual.numel(),
            "matched_ratio": ratio, "envelope_failures": int((~envelope).sum()),
            "max_abs_error": float(error.max()), "rtol": rtol, "atol": atol,
            "max_abs_error_limit": max_abs, "required_matched_ratio": 0.99, "ulp_multiple": 32}


def check_outputs(outputs, golden, inputs):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise AssertionError("RWKV8 must return exactly (o, s, sa)")
    B, H, T, K = inputs["q"].shape
    V = inputs["v"].shape[-1]
    contracts = [((B, H, T, V), inputs["q"].dtype),
                 ((B, H, T // inputs["chunk_len"], K, V), torch.float32)
                 if inputs["output_chunk_state"] else None,
                 ((B, H, T, V), torch.float32) if inputs["output_sa"] else None]
    result = {}
    for name, output, reference, contract in zip(("o", "s", "sa"), outputs, golden, contracts):
        if contract is None:
            if output is not None or reference is not None:
                raise AssertionError(f"Disabled output {name} must be None")
            result[name] = {"pass": True, "disabled": True}
        else:
            if (not isinstance(output, torch.Tensor) or tuple(output.shape) != contract[0]
                    or output.dtype != contract[1] or output.device != inputs["q"].device):
                raise AssertionError(f"Output {name} violates shape/dtype/device contract {contract}")
            result[name] = {"shape": list(output.shape), "dtype": str(output.dtype),
                            **compare_output(output, reference)}
    return result


def fused_recurrent_rwkv8_golden(q, w, k, v, z, b, scale=1.0, initial_state=None,
                                 reverse=False, output_chunk_state=False, output_sa=False, chunk_len=16):
    """PR #446 PTA/dump compatibility: retain FP32 accumulation and original o dtype."""
    assert q.shape == w.shape == k.shape == z.shape == b.shape, "k-side shape mismatch"
    assert q.ndim == 4, "expect q (B,H,T,K)"
    assert v.ndim == 4 and v.shape[:3] == q.shape[:3], "v shape mismatch"
    result = fused_recurrent_rwkv8_reference(
        *(x.float() for x in (q, w, k, v, z, b)), scale=scale,
        initial_state=None if initial_state is None else initial_state.float(),
        reverse=reverse, output_chunk_state=output_chunk_state, output_sa=output_sa, chunk_len=chunk_len)
    return result._replace(o=result.o.to(q.dtype))
