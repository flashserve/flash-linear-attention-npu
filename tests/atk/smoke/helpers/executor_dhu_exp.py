from __future__ import annotations
from typing import Any
import json, torch
from executor_chunk_gated_delta_rule_bwd_dhu import (
    OP_NAME,
    build_inputs,
    _bwd_dhu_reference,
    _marker_device,
    _int_tensor,
    _finite_tuple,
    _case_spec,
    register,
    BaseApi,
    TaskResult,
    InputDataset,
)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 高精度 golden。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _bwd_dhu_reference(
        inputs["q"],
        inputs["k"],
        inputs["w"],
        inputs["do"],
        inputs["dv"],
        g=inputs["g"],
        gK=inputs["gK"],
        h0=inputs["h0"],
        dht=inputs["dht"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        scale=inputs["scale"],
        chunk_size=inputs["chunk_size"],
        # 公共 CPU 节点始终传入 high_precision=True，使用 fp64 完成标杆计算。
        golden_mode="fp64" if high_precision else "npu",
        state_v_first=bool(spec.get("state_v_first", False)),
        nt_first=True,
        use_exp2=bool(spec["use_exp2"]),
    )


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.chunk_gated_delta_rule_bwd_dhu(
        inputs["q"],
        inputs["k"],
        inputs["w"],
        inputs["do"],
        inputs["dv"],
        inputs["scale"],
        inputs["chunk_size"],
        g=inputs["g"],
        gK=inputs["gK"],
        h0=inputs["h0"],
        dht=inputs["dht"],
        cu_seqlens=_int_tensor(inputs["cu_seqlens"], inputs["q"].device)
        if inputs["cu_seqlens"] is not None
        else None,
        chunk_indices=_int_tensor(inputs["chunk_indices"], inputs["q"].device)
        if inputs["chunk_indices"] is not None
        else None,
        transpose_state_layout=bool(spec.get("state_v_first", False)),
        use_exp2=bool(spec["use_exp2"]),
    )


@register("executor_dhu_exp")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # ATK removes non_param inputs before calling the executor. Read the
        # serialized spec from metadata so optional dht/layout/varlen fields survive.
        spec_config = self.task_result.case_config.get_input_data_config(
            name="case_spec"
        )
        spec = (
            json.loads(spec_config.range_values)
            if spec_config is not None and spec_config.range_values
            else _case_spec(input_data, OP_NAME)
        )
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(
                f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}"
            )
        return _finite_tuple(outputs, golden=self.device == "cpu")
