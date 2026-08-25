"""chunk_gated_delta_rule_fwd_h 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。

w 语义（GVA）：w 与 u 同为 HV 个 head（`[B,HV,T,K]`），ACLNN 校验
`w.H == u.H`。k 为 HK 个 head，HV 与 HK 满足 `HV >= HK && HV % HK == 0`。
CPU 标杆逐 value-head 使用各自 `w[b,hv]` 与共享 `k[b, hk]，hk = hv // (HV/HK)`。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _calc_dtype,
    _case_spec,
    _chunks,
    _finite_tuple,
    _gate,
    _kda_gate,
    _marker_device,
    _num_chunks,
    _randn,
)


OP_NAME = "chunk_gated_delta_rule_fwd_h"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    gate_dtype = torch.float64 if high_precision else torch.float32
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    gate_mode = str(spec.get("gate_mode", "g")).lower()
    if gate_mode not in {"g", "gk"}:
        raise ValueError(f"gate_mode 仅支持 'g' 或 'gk'，当前值：{gate_mode!r}")

    g = None
    gk = None
    if gate_mode == "g":
        g = _gate((B, HV, T), gate_dtype, device, seed + 4)
    else:
        gk_chunks = [
            _kda_gate(
                (B, HV, end - start, K),
                "fp32",
                gate_dtype,
                device,
                seed + 4 + chunk_idx,
            )
            for chunk_idx, (start, end) in enumerate(_chunks(T, chunk_size))
        ]
        gk = torch.cat(gk_chunks, dim=2)

    initial_state_dtype = spec.get("initial_state_dtype")
    initial_state = None
    if initial_state_dtype is not None:
        initial_state_dtype = str(initial_state_dtype).lower()
        initial_state = _randn(
            (B, HV, K, V),
            initial_state_dtype,
            _calc_dtype(initial_state_dtype, high_precision),
            device,
            seed + 5,
        )

    return {
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "w": _randn((B, HV, T, K), dtype_name, calc_dtype, device, seed + 2),
        "u": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 3),
        "g": g,
        "gk": gk,
        "gate_mode": gate_mode,
        "use_exp2": bool(spec.get("use_exp2", False)),
        "output_final_state": bool(spec.get("output_final_state", False)),
        "initial_state": initial_state,
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
      "npu"  - k/w/u 保持 bf16/fp16 乘、fp32 累加；仅 FP32 state 且开启最终态输出时
               保持 FP32 递推，其余场景按 h 的输入 dtype 跨 chunk 回写。
    """
    k, w, u = (inputs[name] for name in ("k", "w", "u"))
    g, gk = inputs["g"], inputs["gk"]
    initial_state = inputs["initial_state"]
    gate_mode = inputs["gate_mode"]
    use_exp2 = inputs["use_exp2"]
    output_final_state = inputs["output_final_state"]
    B, HK, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    chunk_size = int(inputs["chunk_size"])
    num_chunks = _num_chunks(T, chunk_size)
    group = HV // HK

    if golden_mode == "npu":
        elem_dtype = k.dtype
        state_elem_dtype = initial_state.dtype if initial_state is not None else torch.float32
        k = k.to(elem_dtype)
        w = w.to(elem_dtype)
        u = u.to(elem_dtype)
        if g is not None:
            g = g.float()
        if gk is not None:
            gk = gk.float()
        if initial_state is not None:
            initial_state = initial_state.float()
        matmul = lambda a, b: _matmul_npu_aligned(a, b, elem_dtype)
        store_input = lambda x: _round_elem(x, elem_dtype)
        state_calc_dtype = torch.float32
    else:
        compute = torch.float64
        k = k.to(compute)
        w = w.to(compute)
        u = u.to(compute)
        if g is not None:
            g = g.to(compute)
        if gk is not None:
            gk = gk.to(compute)
        if initial_state is not None:
            initial_state = initial_state.to(compute)
        matmul = lambda a, b: a @ b
        store_input = lambda x: x
        state_elem_dtype = compute
        state_calc_dtype = compute

    gate_exp = torch.exp2 if use_exp2 else torch.exp
    use_fp32_recurrence = (
        golden_mode != "npu"
        or (output_final_state and state_elem_dtype == torch.float32)
    )

    h = torch.zeros((B, HV, num_chunks, K, V), dtype=k.dtype, device=k.device)
    v_new = torch.zeros((B, HV, T, V), dtype=u.dtype, device=u.device)
    final_state = torch.zeros((B, HV, K, V), dtype=state_elem_dtype, device=k.device)
    for b in range(B):
        for hv in range(HV):
            hk = hv // group
            if initial_state is None:
                rolling_state = torch.zeros((K, V), dtype=state_calc_dtype, device=k.device)
            else:
                rolling_state = initial_state[b, hv].to(state_calc_dtype)
            for chunk_idx, (start, end) in enumerate(_chunks(T, chunk_size)):
                k_chunk = k[b, hk, start:end]
                w_chunk = w[b, hv, start:end]
                u_chunk = u[b, hv, start:end]
                h_state = store_input(rolling_state)
                h[b, hv, chunk_idx] = h_state
                current_v = u_chunk - matmul(w_chunk, h_state)
                v_new[b, hv, start:end] = current_v.to(u.dtype)
                if gate_mode == "g":
                    g_chunk = g[b, hv, start:end]
                    decay = gate_exp(g_chunk[-1] - g_chunk).unsqueeze(-1)
                    state_scale = gate_exp(g_chunk[-1])
                    update_input = current_v * decay
                else:
                    gk_chunk = gk[b, hv, start:end]
                    state_scale = gate_exp(gk_chunk[-1]).unsqueeze(-1)
                    update_input = current_v
                state_base = rolling_state if use_fp32_recurrence else h_state
                next_state = state_base * state_scale + matmul(
                    k_chunk.transpose(-1, -2), update_input
                )
                rolling_state = next_state if use_fp32_recurrence else store_input(next_state)
            final_state[b, hv] = next_state
    if output_final_state:
        return h, v_new, final_state
    return h, v_new


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 标杆：高精度用 fp64，其余用 npu 对齐（bf16/fp16 乘 + fp32 累加）。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _forward_h_ref(inputs, golden_mode="fp64" if high_precision else "npu")


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    outputs = ascendc.chunk_gated_delta_rule_fwd_h(
        inputs["k"],
        inputs["w"],
        inputs["u"],
        inputs["g"],
        gk=inputs["gk"],
        initial_state=inputs["initial_state"],
        output_final_state=inputs["output_final_state"],
        chunk_size=inputs["chunk_size"],
        cu_seqlens=None,
        chunk_indices=None,
        use_exp2=inputs["use_exp2"],
        state_v_first=False,
    )
    if inputs["output_final_state"]:
        if outputs[2] is None:
            raise AssertionError("output_final_state=true 时 final_state 不得为 None")
        expected_dtype = (
            inputs["initial_state"].dtype
            if inputs["initial_state"] is not None
            else torch.float32
        )
        if outputs[2].dtype != expected_dtype:
            raise AssertionError(
                f"final_state dtype 应为 {expected_dtype}，实际为 {outputs[2].dtype}"
            )
    elif outputs[2] is not None:
        raise AssertionError("稳定 Python 入口在 output_final_state=false 时必须返回 None")
    return outputs


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
        visible_outputs = _finite_tuple(outputs)
        if not bool(spec.get("output_final_state", False)):
            # ATK 按首条用例固定输出数量；占位只用于让后续 final_state 进入精度比较。
            placeholder = torch.zeros((1,), dtype=torch.float32, device=visible_outputs[0].device)
            return visible_outputs + (placeholder,)
        return visible_outputs
