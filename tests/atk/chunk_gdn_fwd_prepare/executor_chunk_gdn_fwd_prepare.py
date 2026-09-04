"""chunk_gdn_fwd_prepare 的 ATK executor。

输入生成、CPU 标杆（Gate_of_Babylon cpu_gdn_fwd_l2norm_to_recompute）、
run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
精度标准为 mixed_tolerance_bm：NPU DUT vs CPU 高精度 golden。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _calc_dtype,
    _case_spec,
    _finite_tuple,
    _marker_device,
    _randn,
)
from cpu_golden import cpu_gdn_fwd_l2norm_to_recompute

OP_NAME = "chunk_gdn_fwd_prepare"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec.get("chunk_size", 64))
    g_dtype = torch.float64 if high_precision else torch.float32
    return {
        "q": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 2),
        "v": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 3),
        "g": _randn((B, HV, T), "fp32", g_dtype, device, seed + 4, 0.2),
        "beta": _randn((B, HV, T), "fp32", g_dtype, device, seed + 5, 0.5),
        "chunk_size": chunk_size,
        "use_qk_l2norm_in_kernel": bool(spec.get("use_qk_l2norm_in_kernel", True)),
        "use_gate_in_kernel": bool(spec.get("use_gate_in_kernel", False)),
        "use_beta_sigmoid_in_kernel": bool(spec.get("use_beta_sigmoid_in_kernel", True)),
        "allow_neg_eigval": bool(spec.get("allow_neg_eigval", False)),
        "use_exp2": bool(spec.get("use_exp2", True)),
    }


def _forward_ref(inputs: dict[str, Any]):
    ref = cpu_gdn_fwd_l2norm_to_recompute(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        chunk_size=int(inputs["chunk_size"]),
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
        use_gate_in_kernel=inputs["use_gate_in_kernel"],
        use_beta_sigmoid_in_kernel=inputs["use_beta_sigmoid_in_kernel"],
        allow_neg_eigval=inputs["allow_neg_eigval"],
        layout="bnsd",
    )
    return (
        ref.q,
        ref.k,
        ref.q_rstd,
        ref.k_rstd,
        ref.beta,
        ref.g,
        ref.w,
        ref.u,
        ref.a,
    )


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _forward_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)

    outputs = ascendc.chunk_gdn_fwd_prepare(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        chunk_size=inputs["chunk_size"],
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
        use_gate_in_kernel=inputs["use_gate_in_kernel"],
        use_beta_sigmoid_in_kernel=inputs["use_beta_sigmoid_in_kernel"],
        allow_neg_eigval=inputs["allow_neg_eigval"],
        use_exp2=True,
    )
    torch.npu.synchronize()
    # Host isfinite/compare: avoid queuing extra NPU kernels on the op stream.
    return tuple(None if t is None else t.detach().cpu() for t in outputs)


@register("executor_chunk_gdn_fwd_prepare")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, high_precision=True)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")
