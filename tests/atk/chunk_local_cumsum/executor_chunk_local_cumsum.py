"""chunk_local_cumsum 的 ATK executor。

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


OP_NAME = "chunk_local_cumsum"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, H, T = (int(spec[x]) for x in ("B", "H", "T"))
    return {
        "g": _randn((B, H, T), dtype_name, calc_dtype, device, seed + 1),
        "chunk_size": int(spec["chunk_size"]),
        "reverse": bool(spec.get("reverse", False)),
        "scale": float(spec.get("scale", 1.0)),
    }


def _chunk_local_cumsum_ref(inputs):
    g = inputs["g"]
    calc = torch.float64 if g.dtype == torch.float64 else torch.float32
    out = torch.zeros_like(g, dtype=calc)
    for start, end in _chunks(g.shape[-1], int(inputs["chunk_size"])):
        chunk = g[..., start:end].to(calc) * float(inputs["scale"])
        if bool(inputs["reverse"]):
            out[..., start:end] = torch.flip(torch.cumsum(torch.flip(chunk, dims=(-1,)), dim=-1), dims=(-1,))
        else:
            out[..., start:end] = torch.cumsum(chunk, dim=-1)
    return out.to(torch.float32)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 高精度 golden。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _chunk_local_cumsum_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.chunk_local_cumsum(inputs["g"], inputs["chunk_size"], cu_seqlens=None, chunk_indices_out=None, reverse=inputs["reverse"], scale=inputs["scale"], head_first=True, output_dtype="float32")


@register("executor_chunk_local_cumsum")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")
