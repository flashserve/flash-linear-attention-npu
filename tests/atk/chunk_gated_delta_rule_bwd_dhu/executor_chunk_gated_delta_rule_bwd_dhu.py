"""chunk_gated_delta_rule_bwd_dhu 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。

CPU 标杆直接复用 `torch_custom/fla_npu/test/test_bwd_dhu.py` 里的
`chunk_gated_delta_rule_bwd_dhu_cpu`（真标杆，非 zeros stub）；输入全部为非零随机业务输入。
"""

from __future__ import annotations

import importlib.util
import json
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


OP_NAME = "chunk_gated_delta_rule_bwd_dhu"

_CPU_REFERENCE_FILE = (
    Path(__file__).resolve().parents[3]
    / "torch_custom"
    / "fla_npu"
    / "test"
    / "test_bwd_dhu.py"
)
_cpu_reference_spec = importlib.util.spec_from_file_location(
    "atk_chunk_gated_delta_rule_bwd_dhu_reference",
    _CPU_REFERENCE_FILE,
)
if _cpu_reference_spec is None or _cpu_reference_spec.loader is None:
    raise ImportError(f"Unable to load CPU reference: {_CPU_REFERENCE_FILE}")
_cpu_reference_module = importlib.util.module_from_spec(_cpu_reference_spec)
_cpu_reference_spec.loader.exec_module(_cpu_reference_module)
_bwd_dhu_reference = _cpu_reference_module.chunk_gated_delta_rule_bwd_dhu_cpu


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    state_v_first = bool(spec.get("state_v_first", False))
    state_tail = (V, K) if state_v_first else (K, V)
    cu_seqlens = spec.get("cu_seqlens")
    if cu_seqlens is not None and (B != 1 or cu_seqlens[-1] != T):
        raise ValueError("变长用例要求 B=1 且 cu_seqlens[-1]=T")
    state_count = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    gate_dtype = torch.float64 if high_precision else torch.float32
    use_gk = bool(spec.get("use_gk", False))
    return {
        "q": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 2),
        "w": _randn((B, HV, T, K), dtype_name, calc_dtype, device, seed + 3),
        "do": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 4),
        "dv": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 5),
        "g": None if use_gk else _gate((B, HV, T), gate_dtype, device, seed + 6),
        "gK": _kda_gate((B, HV, T, K), dtype_name, calc_dtype, device, seed + 6) if use_gk else None,
        "h0": (
            _randn((state_count, HV, *state_tail), dtype_name, calc_dtype, device, seed + 7)
            if bool(spec.get("with_h0", False))
            else None
        ),
        "dht": (
            _randn((state_count, HV, *state_tail), dtype_name, calc_dtype, device, seed + 8)
            if bool(spec.get("with_dht", False)) else None
        ),
        "cu_seqlens": cu_seqlens,
        "chunk_indices": (
            _cpu_reference_module.prepare_chunk_indices(cu_seqlens, chunk_size)
            if cu_seqlens is not None else None
        ),
        "chunk_size": chunk_size,
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
    }


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
        use_exp2=bool(spec.get("use_gk", False)),
    )


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.chunk_gated_delta_rule_bwd_dhu(
        inputs["q"], inputs["k"], inputs["w"], inputs["do"], inputs["dv"],
        inputs["scale"], inputs["chunk_size"], g=inputs["g"], gK=inputs["gK"],
        h0=inputs["h0"], dht=inputs["dht"],
        cu_seqlens=_int_tensor(inputs["cu_seqlens"], inputs["q"].device) if inputs["cu_seqlens"] is not None else None,
        chunk_indices=_int_tensor(inputs["chunk_indices"], inputs["q"].device) if inputs["chunk_indices"] is not None else None,
        transpose_state_layout=bool(spec.get("state_v_first", False)),
        use_exp2=bool(spec.get("use_gk", False)),
    )


@register("executor_chunk_gated_delta_rule_bwd_dhu")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # ATK removes non_param inputs before calling the executor. Read the
        # serialized spec from metadata so optional dht/layout/varlen fields survive.
        spec_config = self.task_result.case_config.get_input_data_config(name="case_spec")
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
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")
