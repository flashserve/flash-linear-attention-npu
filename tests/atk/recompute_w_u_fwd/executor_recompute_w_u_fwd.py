"""recompute_w_u_fwd 的 ATK executor。

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


OP_NAME = "recompute_w_u_fwd"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    inputs = {
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "v": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 2),
        "beta": _rand((B, HV, T), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 3, 0.1, 0.9),
        "A": _randn((B, HV, T, chunk_size), dtype_name, calc_dtype, device, seed + 4),
        "g": _gate((B, HV, T), torch.float64 if high_precision else torch.float32, device, seed + 5),
        "chunk_size": chunk_size,
    }
    if OP_NAME != "recompute_w_u_fwd":
        inputs["dw"] = _randn((B, HV, T, K), dtype_name, calc_dtype, device, seed + 6)
        inputs["du"] = _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 7)
    if OP_NAME == "prepare_wy_repr_bwd_full":
        inputs["dA"] = _zeros((B, HV, T, chunk_size), dtype_name, calc_dtype, device)
    return inputs


def _recompute_w_u_ref(inputs):
    k, v, beta, A, g = inputs["k"], inputs["v"], inputs["beta"], inputs["A"], inputs["g"]
    B, HK, T, K = k.shape
    HV, V = v.shape[1], v.shape[3]
    chunk_size = int(inputs["chunk_size"])
    high = k.dtype == torch.float64
    calc = torch.float64 if high else torch.float32
    # 非高精度辅助调用会把 matmul 操作数量化到元素精度，
    # 累加保持 fp32；高精度路径全程 fp64（ATK 自动生成的高精度标杆）。
    elem = torch.float64 if high else k.dtype

    def quant(x: torch.Tensor) -> torch.Tensor:
        """把 matmul 操作数量化到元素精度，但保持 fp32 累加，与 NPU 对齐。"""
        return x.to(elem).float() if not high else x.to(calc)

    w = torch.zeros((B, HV, T, K), dtype=calc, device=k.device)
    u = torch.zeros((B, HV, T, V), dtype=calc, device=k.device)
    group = max(HV // HK, 1)
    for b in range(B):
        for hv in range(HV):
            hk = hv // group
            for start, end in _chunks(T, chunk_size):
                length = end - start
                a = quant(A[b, hv, start:end, :length])
                kbg = quant(k[b, hk, start:end].float() * (beta[b, hv, start:end].float() * torch.exp(g[b, hv, start:end].float())).unsqueeze(-1))
                vb = quant(v[b, hv, start:end].float() * beta[b, hv, start:end].float().unsqueeze(-1))
                w[b, hv, start:end] = torch.matmul(a, kbg)
                u[b, hv, start:end] = torch.matmul(a, vb)
    return w.to(k.dtype), u.to(v.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 高精度 golden。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _recompute_w_u_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.recompute_w_u_fwd(inputs["k"], inputs["v"], inputs["beta"], inputs["A"], inputs["chunk_size"], g=inputs["g"], gk=None, cu_seqlens=None, chunk_indices=None)


@register("executor_recompute_w_u_fwd")
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
