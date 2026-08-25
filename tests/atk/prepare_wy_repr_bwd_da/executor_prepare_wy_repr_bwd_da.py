"""ATK executor for prepare_wy_repr_bwd_da.

The CPU golden is embedded from fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/test/test_da.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from fla_npu.ops import ascendc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _case_spec,
    _chunks,
    _finite_tuple,
    _marker_device,
    _orig_dtype,
    _rand,
)


OP_NAME = "prepare_wy_repr_bwd_da"


def build_inputs(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    data_dtype = _orig_dtype(dtype_name)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    return {
        "k": _rand((B, HK, T, K), dtype_name, data_dtype, device, seed + 1, 0.0, 1.0),
        "v": _rand((B, HV, T, V), dtype_name, data_dtype, device, seed + 2, 0.0, 1.0),
        "beta": _rand((B, HV, T), "fp32", torch.float32, device, seed + 3, 0.0, 1.0),
        "A": _rand((B, HV, T, chunk_size), dtype_name, data_dtype, device, seed + 4, 0.0, 1.0),
        "g": -torch.arange(1, B * HV * T + 1, dtype=torch.float32)
        .reshape(B, HV, T)
        .to(device),
        "dw": _rand((B, HV, T, K), dtype_name, data_dtype, device, seed + 6, 0.0, 1.0),
        "du": _rand((B, HV, T, V), dtype_name, data_dtype, device, seed + 7, 0.0, 1.0),
        "chunk_size": chunk_size,
    }


def _compute_da_golden(inputs: dict[str, Any], high_precision: bool) -> torch.Tensor:
    """Port of test_da.py::compute_dA_cpu for the dense single-case route."""
    k, v = inputs["k"], inputs["v"]
    beta, A, g = inputs["beta"], inputs["A"], inputs["g"]
    dw, du = inputs["dw"], inputs["du"]
    B, HK, T, _ = k.shape
    HV = v.shape[1]
    chunk_size = int(inputs["chunk_size"])
    calc_dtype = torch.float64 if high_precision else torch.float32
    dA = torch.zeros(A.shape, dtype=calc_dtype, device=A.device)
    group_size = HV // HK

    for b in range(B):
        for hv in range(HV):
            hk = hv // group_size
            for start, end in _chunks(T, chunk_size):
                length = end - start
                a_chunk = A[b, hv, start:end, :length].to(calc_dtype)
                dw_chunk = dw[b, hv, start:end].to(calc_dtype)
                du_chunk = du[b, hv, start:end].to(calc_dtype)
                k_chunk = k[b, hk, start:end].to(calc_dtype)
                v_chunk = v[b, hv, start:end].to(calc_dtype)
                beta_chunk = beta[b, hv, start:end].to(calc_dtype)
                g_chunk = g[b, hv, start:end].to(calc_dtype)

                causal = torch.tril(
                    torch.ones((length, length), dtype=torch.bool, device=A.device),
                    diagonal=-1,
                )
                k_beta_g = k_chunk * (beta_chunk * torch.exp(g_chunk)).unsqueeze(-1)
                v_beta = v_chunk * beta_chunk.unsqueeze(-1)
                raw = torch.matmul(dw_chunk, k_beta_g.T) + torch.matmul(du_chunk, v_beta.T)
                masked = torch.where(causal, raw, torch.zeros_like(raw))
                transformed = torch.matmul(a_chunk.T, torch.matmul(masked, a_chunk.T))
                gate_ratio = torch.exp(g_chunk.unsqueeze(1) - g_chunk.unsqueeze(0))
                result = torch.where(causal, -transformed * gate_ratio, torch.zeros_like(transformed))
                dA[b, hv, start:end, :length] = result.T

    return dA if high_precision else dA.to(A.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    return _compute_da_golden(build_inputs(spec, torch.device("cpu")), high_precision)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data))

    return ascendc.prepare_wy_repr_bwd_da(
        inputs["k"],
        inputs["v"],
        inputs["beta"],
        inputs["A"],
        inputs["dw"],
        inputs["du"],
        inputs["g"],
        chunk_size=inputs["chunk_size"],
        cu_seqlens=None,
        chunk_indices=None,
    )


@register("executor_prepare_wy_repr_bwd_da")
class FunctionApi(BaseApi):
    """ATK execution entry."""

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
            raise RuntimeError(f"{OP_NAME} supports only NPU DUT and CPU benchmark nodes")
        return _finite_tuple(outputs)
