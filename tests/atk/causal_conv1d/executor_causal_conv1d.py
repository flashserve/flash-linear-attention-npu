"""causal_conv1d 的 ATK executor。

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


OP_NAME = "causal_conv1d"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, T, D, W = (int(spec[x]) for x in ("B", "T", "D", "W"))
    return {
        "x": _randn((B, T, D), dtype_name, calc_dtype, device, seed + 1),
        "weight": _randn((W, D), dtype_name, calc_dtype, device, seed + 2),
        "bias": _randn((D,), dtype_name, calc_dtype, device, seed + 3),
        "conv_states": _zeros((B, W - 1, D), dtype_name, calc_dtype, device),
    }


def _causal_conv1d_ref(x, weight, bias):
    calc = torch.float64 if x.dtype == torch.float64 else torch.float32
    y = F.conv1d(
        x.to(calc).permute(0, 2, 1).contiguous(),
        weight.to(calc).t().unsqueeze(1).contiguous(),
        bias=bias.to(calc) if bias is not None else None,
        padding=weight.shape[0] - 1,
        groups=x.shape[-1],
    )
    return y[..., : x.shape[1]].permute(0, 2, 1).contiguous().to(x.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _causal_conv1d_ref(inputs["x"], inputs["weight"], inputs["bias"])


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.causal_conv1d(inputs["x"], inputs["weight"], bias=inputs["bias"], conv_states=inputs["conv_states"], activation_mode=0, run_mode=0, head_num=0)


@register("executor_causal_conv1d")
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
