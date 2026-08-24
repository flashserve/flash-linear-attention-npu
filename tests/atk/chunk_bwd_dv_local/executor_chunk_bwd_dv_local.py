"""chunk_bwd_dv_local 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
"""

from __future__ import annotations

import math
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
    _calc_dtype,
    _case_spec,
    _chunks,
    _finite_tuple,
    _gate,
    _marker_device,
    _orig_dtype,
    _randn,
)


OP_NAME = "chunk_bwd_dv_local"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    return {
        "q": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 2),
        "g": _gate((B, HV, T), torch.float64 if high_precision else torch.float32, device, seed + 4),
        "do": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 5),
        "chunk_size": chunk_size,
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
    }


def _tensor_from_kwargs(input_data: InputDataset, name: str) -> torch.Tensor:
    value = input_data.kwargs.get(name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{OP_NAME} 缺少输入 Tensor: {name}")
    return value


def _chunk_bwd_dv_local_ref(inputs: dict[str, Any]):
    q, k, do, g = inputs["q"], inputs["k"], inputs["do"], inputs["g"]
    B, HK, T, _ = q.shape
    HV = do.shape[1]
    chunk_size = int(inputs["chunk_size"])
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    out = torch.zeros_like(do, dtype=calc)
    group = max(HV // HK, 1)
    for b in range(B):
        for hv in range(HV):
            hk = hv // group
            for start, end in _chunks(T, chunk_size):
                score = torch.matmul(k[b, hk, start:end].to(calc), q[b, hk, start:end].to(calc).t())
                score = score * float(inputs["scale"])
                g_chunk = g[b, hv, start:end].to(calc)
                gate = torch.exp(g_chunk[None, :] - g_chunk[:, None])
                mask = torch.triu(torch.ones_like(score))
                out[b, hv, start:end] = torch.matmul(score * gate * mask, do[b, hv, start:end].to(calc))
    return out.to(do.dtype)


def run_cpu(input_data: InputDataset, high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    q = _tensor_from_kwargs(input_data, "q").detach().cpu()
    k = _tensor_from_kwargs(input_data, "k").detach().cpu()
    do = _tensor_from_kwargs(input_data, "do").detach().cpu()
    g = _tensor_from_kwargs(input_data, "g").detach().cpu()
    if high_precision:
        q = q.to(torch.float64)
        k = k.to(torch.float64)
        do = do.to(torch.float64)
        g = g.to(torch.float64)
    inputs = {
        "q": q,
        "k": k,
        "do": do,
        "g": g,
        "chunk_size": int(input_data.kwargs["chunk_size"]),
        "scale": float(input_data.kwargs["scale"]),
    }
    return _chunk_bwd_dv_local_ref(inputs)


def run_npu(input_data: InputDataset):
    """运行 NPU DUT。"""
    from fla_npu.ops import ascendc

    return ascendc.chunk_bwd_dv_local(
        input_data.kwargs["q"],
        input_data.kwargs["k"],
        input_data.kwargs["do"],
        input_data.kwargs["g"],
        input_data.kwargs["scale"],
        input_data.kwargs["chunk_size"],
        g_gamma=None,
        A=None,
        cu_seqlens=None,
        chunk_indices=None,
    )


@register("executor_chunk_bwd_dv_local")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if "q" not in input_data.kwargs:
            self.init_by_input_data(input_data)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(input_data)
        elif self.device == "cpu":
            outputs = run_cpu(input_data, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs)

    def init_by_input_data(self, input_data: InputDataset):
        spec = _case_spec(input_data, OP_NAME)
        dtype_name = str(spec.get("dtype", "bf16")).lower()
        device = _marker_device(input_data) if self.device in {"npu", "pyaclnn"} else torch.device("cpu")
        inputs = build_inputs(spec, device, high_precision=False)

        q = inputs["q"].to(_orig_dtype(dtype_name))
        k = inputs["k"].to(_orig_dtype(dtype_name))
        do = inputs["do"].to(_orig_dtype(dtype_name))
        g = inputs["g"].to(torch.float32)

        if self.device in {"npu", "pyaclnn"}:
            q = q.to(device)
            k = k.to(device)
            do = do.to(device)
            g = g.to(device)

        input_data.kwargs["q"] = q.contiguous()
        input_data.kwargs["k"] = k.contiguous()
        input_data.kwargs["do"] = do.contiguous()
        input_data.kwargs["g"] = g.contiguous()
        input_data.kwargs["scale"] = float(inputs["scale"])
        input_data.kwargs["chunk_size"] = int(inputs["chunk_size"])
        input_data.kwargs["g_gamma"] = None
        input_data.kwargs["A"] = None
        input_data.kwargs["cu_seqlens"] = None
        input_data.kwargs["chunk_indices"] = None
