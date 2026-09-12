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


def _forward_h_ref(inputs):
    """使用 FP64 输入和累加计算定长 CPU golden。"""
    k, w, u, g = (inputs[name] for name in ("k", "w", "u", "g"))
    B, HK, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    chunk_size = int(inputs["chunk_size"])
    num_chunks = _num_chunks(T, chunk_size)
    group = HV // HK

    compute = torch.float64
    k = k.to(compute)
    w = w.to(compute)
    u = u.to(compute)
    g = g.to(compute)

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
                current_v = u_chunk - w_chunk @ state
                v_new[b, hv, start:end] = current_v.to(u.dtype)
                if chunk_idx + 1 < num_chunks:
                    decay = torch.exp(g_chunk[-1] - g_chunk).unsqueeze(-1)
                    g_last = torch.exp(g_chunk[-1])
                    s_decayed = state * g_last
                    s_update = k_chunk.transpose(-1, -2) @ (current_v * decay)
                    h[b, hv, chunk_idx + 1] = s_decayed + s_update
    return h, v_new


def run_cpu(spec: dict[str, Any]):
    """运行 FP64 CPU golden。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=True)
    return _forward_h_ref(inputs)


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

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")
