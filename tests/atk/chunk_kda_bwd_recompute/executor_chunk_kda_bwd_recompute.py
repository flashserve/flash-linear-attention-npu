"""chunk_kda_bwd_recompute 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
CPU 标杆对齐 kda_gate_wu_fusion_golden.fused_cpu：safe-gate + chunk cumsum + GQA 展开 + A @ kbg/vb。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _RCP_LN2,
    _calc_dtype,
    _case_spec,
    _chunks,
    _finite_tuple,
    _kda_gate,
    _marker_device,
    _rand,
    _randn,
)


OP_NAME = "chunk_kda_bwd_recompute"


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
    gk = torch.empty_like(g_corr)
    for start, end in _chunks(tokens, chunk_size):
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
    for start, end in _chunks(tokens, chunk_size):
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

    if orig == torch.float64:
        return gk, w, u, qg, kg
    return (
        gk.to(torch.float32),
        w.to(orig),
        u.to(orig),
        qg.to(orig),
        kg.to(orig),
    )


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _fused_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops.ascendc import chunk_kda_bwd_recompute

    return chunk_kda_bwd_recompute(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["A"],
        inputs["chunk_size"],
        A_log=inputs["A_log"],
        dt_bias=inputs.get("dt_bias"),
        use_gate_in_kernel=bool(inputs["use_gate"]),
        use_exp2=bool(inputs["use_exp2"]),
        lower_bound=float(inputs["lower_bound"]),
    )


@register("executor_chunk_kda_bwd_recompute")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs)
