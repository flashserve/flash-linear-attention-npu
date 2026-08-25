"""chunk_fwd_o 的 ATK executor。

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


OP_NAME = "chunk_fwd_o"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    return {
        "q": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 2),
        "v": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 3),
        "g": _gate((B, HV, T), torch.float64 if high_precision else torch.float32, device, seed + 4),
        "h": _randn((B, HV, _num_chunks(T, chunk_size), K, V), dtype_name, calc_dtype, device, seed + 6),
        "chunk_size": chunk_size,
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
    }


def _chunk_fwd_o_ref(inputs):
    q, k, v, h, g = inputs["q"], inputs["k"], inputs["v"], inputs["h"], inputs["g"]
    B, HK, T, _ = q.shape
    HV, V = v.shape[1], v.shape[3]
    chunk_size = int(inputs["chunk_size"])
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    out = torch.zeros((B, HV, T, V), dtype=calc, device=q.device)
    group = max(HV // HK, 1)
    for b in range(B):
        for hv in range(HV):
            hk = hv // group
            for chunk_id, (start, end) in enumerate(_chunks(T, chunk_size)):
                q_chunk = q[b, hk, start:end].to(calc)
                k_chunk = k[b, hk, start:end].to(calc)
                v_chunk = v[b, hv, start:end].to(calc)
                g_chunk = g[b, hv, start:end].to(calc)
                local = torch.matmul(q_chunk, k_chunk.t()) * float(inputs["scale"])
                gate = torch.exp(g_chunk[:, None] - g_chunk[None, :])
                mask = torch.tril(torch.ones_like(local))
                out[b, hv, start:end] = torch.matmul(local * gate * mask, v_chunk) + torch.matmul(q_chunk * float(inputs["scale"]), h[b, hv, chunk_id].to(calc))
    return out.to(v.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _chunk_fwd_o_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.chunk_fwd_o(inputs["q"], inputs["k"], inputs["v"], inputs["h"], inputs["scale"], g=inputs["g"], g_gamma=None, cu_seqlens=None, chunk_indices=None, chunk_size=inputs["chunk_size"], transpose_state_layout=False)


@register("executor_chunk_fwd_o")
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
