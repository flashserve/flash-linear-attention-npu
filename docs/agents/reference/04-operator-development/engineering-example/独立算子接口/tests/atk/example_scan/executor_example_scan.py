"""example_scan 的 ATK executor（示例）。

注意事项：
  1. 本文件负责四件事：按 case_spec 构造输入、CPU 标杆、NPU DUT 调用、ATK FunctionApi 注册。
     公共基础函数从 tests/atk/common/_ascendc_common_executor.py 引入，算子专属逻辑留在这里。
  2. CPU 标杆用高精度（fp64/fp32）独立计算，NPU 侧保持算子原始 dtype；两边都按 case_spec 生成同一组
     输入（同 seed），不允许"NPU 用 A 数据、golden 用 B 数据"。
  3. 返回槽位由档位决定（未导出时不返回该槽位），golden 与 DUT 的槽位数必须一致，
     否则 ATK 会按 case 比较时错位。
  4. 不在 executor 里自定义精度阈值：标准只用 yaml 的 mixed_tolerance_bm。
  5. 精度失败不要改这里来"绕开"：先按 tests/atk/README.md 的定位路线查值域/无效区/标杆语义。
  6. 随机输入禁止收窄 range；极端值/无效区的处理要在 README 里写明预期语义。
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
    _calc_dtype,
    _case_spec,
    _chunks,
    _finite_tuple,
    _marker_device,
    _randn,
)

OP_NAME = "example_scan"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    """按 case_spec 构造输入；NPU 与 CPU 走同一函数、同一 seed。"""
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260923))
    B, H, T, D = (int(spec[key]) for key in ("B", "H", "T", "D"))
    return {
        "x": _randn((B, H, T, D), dtype_name, calc_dtype, device, seed + 1),
        "g": _randn((B, H, T), dtype_name, calc_dtype, device, seed + 2),
        "initial_state": (
            _randn((B, H, D), dtype_name, calc_dtype, device, seed + 3)
            if spec.get("has_initial_state")
            else None
        ),
        "layout": str(spec.get("layout", "BSND")),
        "scale": float(spec.get("scale", 1.0)),
        "chunk_size": int(spec.get("chunk_size", 64)),
        "epsilon": float(spec.get("epsilon", 1e-06)),
        "keep_saved": bool(spec.get("keep_saved", False)),
        "output_mode": int(spec.get("output_mode", 0)),
    }


def _example_scan_ref(inputs: dict[str, Any]) -> tuple:
    """CPU 标杆：与算子语义逐步对应，中间量用 fp64 累加。"""
    x = inputs["x"].to(torch.float64)
    g = inputs["g"].to(torch.float64)
    chunk_size = int(inputs["chunk_size"])
    scale = float(inputs["scale"])
    epsilon = float(inputs["epsilon"])

    norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) / x.shape[-1] + epsilon)
    scan = torch.zeros_like(g)
    state_parts = []
    for start, end in _chunks(g.shape[-1], chunk_size):
        chunk_scan = torch.cumsum(g[..., start:end] * scale, dim=-1)
        scan[..., start:end] = chunk_scan
        state_parts.append(chunk_scan[..., -1])
    y = x * scan.unsqueeze(-1) * norm

    if not inputs.get("keep_saved", False):
        return (y.to(torch.float32),)
    state = torch.stack(state_parts, dim=-1).to(torch.float32)
    return (y.to(torch.float32), state, norm.squeeze(-1).to(torch.float32))


def run_cpu(spec: dict[str, Any], high_precision: bool = False) -> tuple:
    """CPU 高精度 golden。"""
    return _example_scan_ref(build_inputs(spec, torch.device("cpu"), high_precision=True))


def run_npu(spec: dict[str, Any], input_data: InputDataset) -> tuple:
    """NPU DUT：只走 fla_npu.ops.ascendc 的稳定入口。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.example_scan(
        inputs["x"],
        inputs["g"],
        a_log=None,
        initial_state=inputs["initial_state"],
        cu_seqlens=None,
        chunk_indices=None,
        layout=inputs["layout"],
        scale=inputs["scale"],
        chunk_size=inputs["chunk_size"],
        epsilon=inputs["epsilon"],
        return_saved=inputs["keep_saved"],
    )


@register("executor_example_scan")
class FunctionApi(BaseApi):
    """ATK 执行入口：注册名必须与 yaml 的 api_type 一致。"""

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
