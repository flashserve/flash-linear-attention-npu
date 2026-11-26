"""chunk_gated_delta_rule_fwd_h of ATK executor.

Inputs, references, run_cpu/run_gpu/run_npu and FunctionApi live in this directory.

Supports fixed and variable length (B=1 + cu_seqlens/chunk_indices).

GPU dual benchmark (recommended):
- truth: GPU fp64 small-op stitching (`_forward_h_ref` golden_mode="fp64").
- control: GPU Triton first (`fla.ops.common.chunk_delta_h:chunk_gated_delta_rule_fwd_h`,
  seq-major layout [B,T,H,*], transposed to head-major [B,H,T,*] to match the DUT);
  on any Triton failure fall back to GPU torch same-precision small-op stitching
  (golden_mode="torch").

w (GVA): w/u share HV value heads; k has HK heads with HV>=HK and HV%HK==0.
"""

from __future__ import annotations

import importlib
import math
import os
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
    _as_bool,
    _calc_dtype,
    _canonical_chunk_indices,
    _case_spec,
    _ceil_div,
    _chunks,
    _finite_tuple,
    _gate,
    _gate_monotonic,
    _int_list,
    _int_tensor,
    _kda_gate,
    _marker_device,
    _num_chunks,
    _orig_dtype,
    _rand,
    _randn,
    _split_cu,
    _varlen_groups,
    _zeros,
)


OP_NAME = "chunk_gated_delta_rule_fwd_h"

# GPU fla Triton 实现（seq-major [B,T,H,*]）：<python_module>:<callable>。
_DEFAULT_TRITON_CALLABLE = "fla.ops.common.chunk_delta_h:chunk_gated_delta_rule_fwd_h"


def _cuda_device() -> torch.device:
    """GPU 参考节点使用的设备。远端 ATK GPU server 按 CUDA_VISIBLE_DEVICES 可见。"""
    return torch.device("cuda")


def _load_triton_callable():
    """加载 GPU fla Triton 标杆。不可用（未安装/无法导入/被禁用）时返回 None，
    由调用方回退到 GPU torch 同精度小算子拼接。"""
    if os.environ.get("GDN_ATK_TRITON", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    target = os.environ.get("GDN_ATK_TRITON_CALLABLE", _DEFAULT_TRITON_CALLABLE).strip()
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError("GDN_ATK_TRITON_CALLABLE 必须使用 '<python_module>:<callable>' 语法")
    try:
        module = importlib.import_module(module_name)
        callable_obj = getattr(module, attribute, None)
    except (ImportError, AttributeError):
        return None
    if not callable(callable_obj):
        return None
    return callable_obj


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "K", "V"))
    chunk_size = int(spec["chunk_size"])

    cu = _int_list(spec.get("cu_seqlens"))
    if cu is not None:
        B = 1
        T = int(cu[-1])
        chunk_indices = _canonical_chunk_indices(cu, chunk_size)
    else:
        T = int(spec["T"])
        chunk_indices = None

    return {
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "w": _randn((B, HV, T, K), dtype_name, calc_dtype, device, seed + 2),
        "u": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 3),
        "g": _gate_monotonic((B, HV, T), device, seed + 4,
                             elem_dtype=None,
                             calc_dtype=torch.float64 if high_precision else torch.float32),
        "cu_seqlens": cu,
        "chunk_indices": chunk_indices,
        "chunk_size": chunk_size,
    }


def _round_elem(x: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    if elem_dtype == torch.float32:
        return x.to(torch.float32)
    return x.to(elem_dtype).to(torch.float32)


def _matmul_npu_aligned(a: torch.Tensor, b: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    return _round_elem(a, elem_dtype) @ _round_elem(b, elem_dtype)


def _forward_h_ref(inputs, golden_mode: str = "fp64"):
    """定长/变长 CPU/GPU 标杆（w=HV，GVA 对齐 ACLNN / 内核）。

    golden_mode:
      "fp64" - 输入升 fp64、fp64 累加（升精度真值标杆）。
      "npu"  - k/w/u 保持 bf16/fp16 乘、fp32 累加，逐 chunk 状态回写 elem_dtype，
               与 NPU 单算子同精度标杆一致。
      "torch"- k/w/u 保持元素精度但 torch 默认 fp32 累加（GPU torch 同精度标杆）。
    """
    k, w, u, g = (inputs[name] for name in ("k", "w", "u", "g"))
    B, HK, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    chunk_size = int(inputs["chunk_size"])
    cu = inputs["cu_seqlens"]
    chunk_indices = inputs["chunk_indices"]
    group = HV // HK

    if cu is None:
        num_chunks = _num_chunks(T, chunk_size)
        # 单条“序列”（定长），块全局下标即 chunk_idx。
        seq_blocks = [[(start, end, chunk_idx) for chunk_idx, (start, end) in enumerate(_chunks(T, chunk_size))]]
    else:
        chunk_counts = [_ceil_div(e - s, chunk_size) for s, e in zip(cu, cu[1:])]
        starts = []
        acc = 0
        for cc in chunk_counts:
            starts.append(acc)
            acc += cc
        num_chunks = acc
        groups = _varlen_groups(cu, chunk_indices, chunk_size)
        seq_blocks = [
            [(start, end, starts[s] + i) for i, (start, end) in enumerate(groups[s])]
            for s in range(len(groups))
            if groups[s]
        ]

    if golden_mode == "npu":
        elem_dtype = k.dtype
        k = k.to(elem_dtype)
        w = w.to(elem_dtype)
        u = u.to(elem_dtype)
        g = g.float()
        matmul = lambda a, b: _matmul_npu_aligned(a, b, elem_dtype)
        store = lambda x: _round_elem(x, elem_dtype)
    elif golden_mode == "torch":
        elem_dtype = k.dtype
        k = k.to(elem_dtype)
        w = w.to(elem_dtype)
        u = u.to(elem_dtype)
        g = g.float()
        matmul = lambda a, b: (a.to(torch.float32) @ b.to(torch.float32))
        store = lambda x: x.to(elem_dtype)
    else:
        compute = torch.float64
        k = k.to(compute)
        w = w.to(compute)
        u = u.to(compute)
        g = g.to(compute)
        matmul = lambda a, b: a @ b
        store = lambda x: x

    h = torch.zeros((B, HV, num_chunks, K, V), dtype=compute if golden_mode not in ("npu", "torch") else k.dtype,
                    device=k.device)
    v_new = torch.zeros((B, HV, T, V), dtype=u.dtype, device=u.device)
    for b in range(B):
        for hv in range(HV):
            hk = hv // group
            for blocks in seq_blocks:
                for j, (start, end, global_idx) in enumerate(blocks):
                    k_chunk = k[b, hk, start:end]
                    w_chunk = w[b, hv, start:end]
                    u_chunk = u[b, hv, start:end]
                    g_chunk = g[b, hv, start:end]
                    state = h[b, hv, global_idx]
                    current_v = u_chunk - matmul(w_chunk, state)
                    v_new[b, hv, start:end] = current_v.to(u.dtype)
                    if j + 1 < len(blocks):
                        decay = torch.exp(g_chunk[-1] - g_chunk).unsqueeze(-1)
                        gate_last = torch.exp(g_chunk[-1])
                        s_decayed = store(state) * gate_last
                        s_update = matmul(k_chunk.transpose(-1, -2), current_v * decay)
                        h[b, hv, global_idx + 1] = store(s_decayed + s_update)
    return h, v_new


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _forward_h_ref(inputs, golden_mode="fp64" if high_precision else "npu")


def run_gpu_truth(spec: dict[str, Any]):
    inputs = build_inputs(spec, _cuda_device(), high_precision=True)
    return _forward_h_ref(inputs, golden_mode="fp64")


def _triton_same_precision(inputs: dict[str, Any], callable_obj):
    """GPU Triton 同精度标杆。fla Triton 是 seq-major [B,T,H,*]，DUT 是 head-major
    [B,H,T,*]，因此转置输入后再转回输出布局。"""
    k, w, u, g = inputs["k"], inputs["w"], inputs["u"], inputs["g"]
    dev = k.device
    kT = k.permute(0, 2, 1, 3).contiguous()
    wT = w.permute(0, 2, 1, 3).contiguous()
    uT = u.permute(0, 2, 1, 3).contiguous()
    gT = g.permute(0, 2, 1).contiguous()
    cu = _int_tensor(inputs["cu_seqlens"], dev)
    cu_cpu = None if inputs["cu_seqlens"] is None else torch.tensor(
        list(inputs["cu_seqlens"]), dtype=torch.int64, device="cpu")
    ci = _int_tensor(inputs["chunk_indices"], dev)
    h, v_new, final_state = callable_obj(
        kT, wT, uT, gT,
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=int(inputs["chunk_size"]),
        save_new_value=True,
        state_v_first=False,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        chunk_indices=ci,
    )
    del final_state
    # [B,NT,HV,K,V] -> [B,HV,NT,K,V]；[B,T,HV,V] -> [B,HV,T,V]
    h = h.permute(0, 2, 1, 3, 4).contiguous()
    v_new = v_new.permute(0, 2, 1, 3).contiguous()
    return h, v_new


def run_gpu_control(spec: dict[str, Any]):
    """GPU 双标杆标杆：优先 GPU Triton；不可用/失败时回退 GPU torch 同精度小算子拼接。"""
    inputs = build_inputs(spec, _cuda_device(), high_precision=False)
    triton_callable = _load_triton_callable()
    if triton_callable is not None:
        try:
            return _triton_same_precision(inputs, triton_callable)
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(f"GPU Triton 标杆失败，回落 GPU torch 同精度：{exc!r}")
    return _forward_h_ref(inputs, golden_mode="torch")


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    dev = _marker_device(input_data)
    from fla_npu.ops import ascendc

    return ascendc.chunk_gated_delta_rule_fwd_h(
        inputs["k"], inputs["w"], inputs["u"], inputs["g"],
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=inputs["chunk_size"],
        cu_seqlens=_int_tensor(inputs["cu_seqlens"], dev),
        chunk_indices=_int_tensor(inputs["chunk_indices"], dev),
        state_v_first=False,
    )


@register("executor_chunk_gated_delta_rule_fwd_h")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device in {"cpu", "gpu"} and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        elif self.device == "gpu":
            outputs = run_gpu_truth(spec) if self.high_precision else run_gpu_control(spec)
        else:
            raise RuntimeError(
                f"{OP_NAME} needs an NPU DUT and CPU/GPU reference, device={self.device!r}, "
                f"benchmark={self.is_benchmark_task}"
            )
        return _finite_tuple(outputs)
