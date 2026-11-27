"""merge_fwd_bwd_kernel 的 ATK executor。

输入构造和 CPU 标杆在 scripts/merge_fwd_bwd_kernel_cpu.py。
NPU DUT 走 fla_npu.ops.ascendc.npu_merge_fwd_bwd_kernel，h 原地写回。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _case_spec, _finite_tuple, _marker_device


OP_NAME = "merge_fwd_bwd_kernel"
_CPU_PY = Path(__file__).resolve().parent / "scripts" / "merge_fwd_bwd_kernel_cpu.py"
_CPU = None


def _cpu():
    global _CPU
    if _CPU is None:
        spec = importlib.util.spec_from_file_location("merge_fwd_bwd_kernel_cpu", _CPU_PY)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"找不到 CPU 标杆：{_CPU_PY}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CPU = module
    return _CPU


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    return _cpu().build_inputs(spec, device, high_precision)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    return _cpu().run_reference(spec, high_precision=high_precision)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops.ascendc import npu_merge_fwd_bwd_kernel

    return npu_merge_fwd_bwd_kernel(
        inputs["h"],
        inputs["ag_hm"],
        inputs["pre_or_post_num_ranks"],
        inputs["rank"],
        forward=inputs["forward"],
        state_v_first=inputs["state_v_first"],
    )


@register("executor_merge_fwd_bwd_kernel")
class FunctionApi(BaseApi):
    """ATK 执行入口。CPU 高精度任务走 FP64，CPU 普通任务走与 DUT 相同的舍入。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _cpu().normalize_spec(_case_spec(input_data, OP_NAME, self.task_result.case_config))
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs)
