#!/usr/bin/env python3
"""ATK ``cv_fused_double_benchmark`` 的三路角色合同。"""

from __future__ import annotations


RUNTIME_ROLE_ORDER = ("dut", "benchmark", "golden")
DUT_NODE_NAME = "phase6"
DUT_NODE_NAMES = (DUT_NODE_NAME, "npu_dut", "npu_dut_1")
METRIC_TENSOR_PAIRS = {
    "Actual": ("dut", "golden"),
    "Benchmark": ("benchmark", "golden"),
}


def role_for_atk_task(
    device: str,
    node_name: str,
    is_benchmark_task: bool,
) -> str:
    """映射 NPU DUT、CPU FP64 benchmark 和 GPU Triton golden。"""

    device = str(device).strip().lower()
    node_name = str(node_name).strip()
    if device == "cpu":
        if not is_benchmark_task:
            raise RuntimeError(
                "CPU 仅允许承载 ATK 自动创建的 benchmark 任务："
                f"name={node_name!r} is_benchmark_task={is_benchmark_task}"
            )
        return "benchmark"
    if device == "gpu":
        return "golden"
    if device != "npu" or is_benchmark_task:
        raise RuntimeError(
            "无法识别 ATK 双标杆任务："
            f"device={device!r} name={node_name!r} "
            f"is_benchmark_task={is_benchmark_task}"
        )
    if node_name in DUT_NODE_NAMES:
        return "dut"
    raise RuntimeError(f"NPU node.name 必须为 {DUT_NODE_NAMES!r}，实际为 {node_name!r}")
