"""chunk_kda_bwd_recompute 的 ATK executor。

输入生成和唯一 CPU 数学标杆在 reference.py；本文件负责 ATK 调用与输出检查。
CPU 标杆对齐 kda_gate_wu_fusion_golden.fused_cpu：safe-gate + chunk cumsum + GQA 展开 + A @ kbg/vb。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _case_spec, _marker_device


OP_NAME = "chunk_kda_bwd_recompute"


def _finite_outputs(outputs):
    """过滤 None 输出，并检查浮点输出是否有限。

    自定义 OPP 会盖掉 aclnnIsFinite（561103），NPU 输出先拷到 CPU 再检查。
    """
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    visible = []
    for output in outputs:
        if output is None or not isinstance(output, torch.Tensor):
            continue
        probe = output.detach()
        if probe.device.type != "cpu":
            probe = probe.cpu()
        if probe.is_floating_point() and not torch.isfinite(probe.float()).all().item():
            raise RuntimeError("输出包含 NaN 或 Inf")
        visible.append(output)
    return tuple(visible)


from reference import build_inputs, _fused_ref


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
        cu_seqlens=torch.tensor(inputs["cu_seqlens"],dtype=torch.int64) if "cu_seqlens" in inputs else None,
        chunk_indices=torch.tensor(inputs["chunk_indices"],dtype=torch.int64) if "chunk_indices" in inputs else None,
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
        return _finite_outputs(outputs)
