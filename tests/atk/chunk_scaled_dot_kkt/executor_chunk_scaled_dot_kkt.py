"""chunk_scaled_dot_kkt 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
    _gate,
    _int_tensor,
    _kda_gate,
    _marker_device,
    _num_chunks,
    _orig_dtype,
    _rand,
    _randn,
    _zeros,
)


OP_NAME = "chunk_scaled_dot_kkt"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K"))
    return {
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "g": _gate((B, HV, T), torch.float64 if high_precision else torch.float32, device, seed + 2),
        "beta": _rand((B, HV, T), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 3, 0.1, 0.9),
        "chunk_size": int(spec["chunk_size"]),
    }


def _chunk_scaled_dot_kkt_ref(inputs):
    k, g, beta = inputs["k"], inputs["g"], inputs["beta"]
    B, HK, T, _ = k.shape
    HV = g.shape[1]
    chunk_size = int(inputs["chunk_size"])
    calc = torch.float64 if k.dtype == torch.float64 else torch.float32
    if beta.shape != g.shape or HV % HK != 0:
        raise ValueError("chunk_scaled_dot_kkt requires beta.shape == g.shape and HV divisible by HK")
    head_ratio = HV // HK
    out = torch.zeros((B, HV, T, chunk_size), dtype=calc, device=k.device)
    for b in range(B):
        for hv in range(HV):
            hk = hv // head_ratio
            for start, end in _chunks(T, chunk_size):
                length = end - start
                k_chunk = k[b, hk, start:end].to(calc)
                g_chunk = g[b, hv, start:end].to(calc)
                beta_chunk = beta[b, hv, start:end].to(calc)
                score = torch.matmul(k_chunk, k_chunk.t())
                gate = torch.exp(torch.clamp(g_chunk[:, None] - g_chunk[None, :], -50.0, 50.0))
                mask = torch.tril(torch.ones((length, length), dtype=calc, device=k.device), diagonal=-1)
                out[b, hv, start:end, :length] = score * gate * beta_chunk[:, None] * mask
    return out.to(torch.float32)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _chunk_scaled_dot_kkt_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.chunk_scaled_dot_kkt(inputs["k"], inputs["g"], inputs["beta"], cu_seqlens=None, chunk_indices=None, chunk_size=inputs["chunk_size"])


@register("executor_chunk_scaled_dot_kkt")
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
