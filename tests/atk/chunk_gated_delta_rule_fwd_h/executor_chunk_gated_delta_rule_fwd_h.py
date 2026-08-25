"""chunk_gated_delta_rule_fwd_h 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。

w 语义（GVA）：w 与 u 同为 HV 个 head（`[B,HV,T,K]`），ACLNN 校验
`w.H == u.H`。k 为 HK 个 head，HV 与 HK 满足 `HV >= HK && HV % HK == 0`。
CPU 标杆逐 value-head 使用各自 `w[b,hv]` 与共享 `k[b, hk]，hk = hv // (HV/HK)`。
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


OP_NAME = "chunk_gated_delta_rule_fwd_h"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    return {
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "w": _randn((B, HV, T, K), dtype_name, calc_dtype, device, seed + 2),
        "u": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 3),
        "g": _gate((B, HV, T), torch.float64 if high_precision else torch.float32, device, seed + 4),
        "chunk_size": chunk_size,
    }


def _round_elem(x: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    """舍入到 elem_dtype（bf16/fp16）精度，仍留在 fp32 容器计算（对齐 Cube MMAD）。"""
    if elem_dtype == torch.float32:
        return x.to(torch.float32)
    return x.to(elem_dtype).to(torch.float32)


def _matmul_npu_aligned(a: torch.Tensor, b: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    """bf16/fp16 乘 + fp32 累加，与 NPU Cube MMAD 语义一致（同精度标杆关键）。"""
    return _round_elem(a, elem_dtype) @ _round_elem(b, elem_dtype)


def _forward_h_ref(inputs, golden_mode: str = "fp64"):
    """定长 CPU 标杆（w=HV, GVA 对齐 ACLNN / 内核）。

    golden_mode:
      "fp64" - 输入升 fp64、fp64 累加（升精度真值标杆，ATK 自动计算）。
      "npu"  - k/w/u 保持 bf16/fp16 乘、fp32 累加，逐 chunk 状态回写 elem_dtype，
               与 NPU 单算子同精度标杆一致（双标杆中的同精度参考）。
    """
    k, w, u, g = (inputs[name] for name in ("k", "w", "u", "g"))
    B, HK, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    chunk_size = int(inputs["chunk_size"])
    num_chunks = _num_chunks(T, chunk_size)
    group = HV // HK

    if golden_mode == "npu":
        elem_dtype = k.dtype
        k = k.to(elem_dtype)
        w = w.to(elem_dtype)
        u = u.to(elem_dtype)
        g = g.float()
        matmul = lambda a, b: _matmul_npu_aligned(a, b, elem_dtype)
        store = lambda x: _round_elem(x, elem_dtype)
    else:
        compute = torch.float64
        k = k.to(compute)
        w = w.to(compute)
        u = u.to(compute)
        g = g.to(compute)
        matmul = lambda a, b: a @ b
        store = lambda x: x

    h = torch.zeros((B, HV, num_chunks, K, V), dtype=k.dtype, device=k.device)
    v_new = torch.zeros((B, HV, T, V), dtype=u.dtype, device=u.device)
    for b in range(B):
        for hv in range(HV):
            hk = hv // group
            for chunk_idx, (start, end) in enumerate(_chunks(T, chunk_size)):
                k_chunk = k[b, hk, start:end]
                w_chunk = w[b, hv, start:end]
                u_chunk = u[b, hv, start:end]
                g_chunk = g[b, hv, start:end]
                state = h[b, hv, chunk_idx]
                current_v = u_chunk - matmul(w_chunk, state)
                v_new[b, hv, start:end] = current_v.to(u.dtype)
                if chunk_idx + 1 < num_chunks:
                    decay = torch.exp(g_chunk[-1] - g_chunk).unsqueeze(-1)
                    g_last = torch.exp(g_chunk[-1])
                    s_decayed = store(state) * g_last
                    s_update = matmul(k_chunk.transpose(-1, -2), current_v * decay)
                    h[b, hv, chunk_idx + 1] = store(s_decayed + s_update)
    return h, v_new


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 标杆：高精度用 fp64，其余用 npu 对齐（bf16/fp16 乘 + fp32 累加）。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _forward_h_ref(inputs, golden_mode="fp64" if high_precision else "npu")


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.chunk_gated_delta_rule_fwd_h(inputs["k"], inputs["w"], inputs["u"], inputs["g"], gk=None, initial_state=None, output_final_state=False, chunk_size=inputs["chunk_size"], cu_seqlens=None, chunk_indices=None, state_v_first=False)


@register("executor_chunk_gated_delta_rule_fwd_h")
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
        return _finite_tuple(outputs)
