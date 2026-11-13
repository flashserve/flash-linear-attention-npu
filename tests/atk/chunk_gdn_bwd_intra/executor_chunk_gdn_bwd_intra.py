"""chunk_gdn_bwd_intra 的 ATK executor。"""

from __future__ import annotations

import math
import os
import subprocess
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
    _calc_dtype,
    _case_spec,
    _chunks,
    _finite_tuple,
    _gate,
    _marker_device,
    _orig_dtype,
    _rand,
    _randn,
)


OP_NAME = "chunk_gdn_bwd_intra"
QK_DO_INPUT_STD = 0.20


def _param_dtype(name: str, high_precision: bool) -> torch.dtype:
    return torch.float64 if high_precision else _orig_dtype(name)


def _build_gate(
    shape: tuple[int, ...],
    dtype_name: str,
    device: torch.device,
    seed: int,
    high_precision: bool,
) -> torch.Tensor:
    data = _gate(shape, torch.float32, torch.device("cpu"), seed)
    return data.to(_orig_dtype(dtype_name)).to(_param_dtype(dtype_name, high_precision)).to(device)


def _canonical_chunk_indices(
    cu_seqlens: tuple[int, ...] | None,
    chunk_size: int,
) -> tuple[int, ...] | None:
    if cu_seqlens is None:
        return None
    indices = []
    for seq, (begin, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for local_chunk in range((end - begin + chunk_size - 1) // chunk_size):
            indices.extend((seq, local_chunk))
    return tuple(indices)


def _work_chunks(inputs: dict[str, Any]):
    """Yield the physical BNSD span for every fixed or varlen chunk."""

    q = inputs["q"]
    chunk_size = int(inputs["chunk_size"])
    cu_seqlens = inputs["cu_seqlens"]
    if cu_seqlens is None:
        for batch in range(q.shape[0]):
            for start, end in _chunks(q.shape[2], chunk_size):
                yield batch, start, end
        return
    for seq, local_chunk in zip(
        inputs["chunk_indices"][::2], inputs["chunk_indices"][1::2]
    ):
        start = cu_seqlens[seq] + local_chunk * chunk_size
        end = min(start + chunk_size, cu_seqlens[seq + 1])
        if start < end:
            yield 0, start, end


def build_inputs(
    spec: dict[str, Any],
    device: torch.device,
    high_precision: bool = False,
) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    g_dtype = str(spec.get("g_dtype", "fp32")).lower()
    beta_dtype = str(spec.get("beta_dtype", "fp32")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260904))
    B, HK, HV, T, K, V = (
        int(spec[name]) for name in ("B", "HK", "HV", "T", "K", "V")
    )
    chunk_size = int(spec["chunk_size"])
    raw_cu_seqlens = spec.get("cu_seqlens")
    cu_seqlens = (
        None
        if raw_cu_seqlens is None
        else tuple(int(value) for value in raw_cu_seqlens)
    )
    chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)
    return {
        "q": _randn(
            (B, HK, T, K), dtype_name, calc_dtype, device, seed + 1, QK_DO_INPUT_STD
        ),
        "k": _randn(
            (B, HK, T, K), dtype_name, calc_dtype, device, seed + 2, QK_DO_INPUT_STD
        ),
        "v": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 3),
        "g": _build_gate((B, HV, T), g_dtype, device, seed + 4, high_precision),
        "beta": _rand(
            (B, HV, T),
            beta_dtype,
            _param_dtype(beta_dtype, high_precision),
            device,
            seed + 5,
            0.1,
            0.9,
        ),
        "A": _randn((B, HV, T, chunk_size), dtype_name, calc_dtype, device, seed + 6),
        "d_o": _randn(
            (B, HV, T, V), dtype_name, calc_dtype, device, seed + 7, QK_DO_INPUT_STD
        ),
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
        "chunk_size": chunk_size,
        "use_exp2": bool(spec.get("use_exp2", True)),
        "main_dtype": _orig_dtype(dtype_name),
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
    }


def _compute_intermediates(inputs: dict[str, Any]):
    q, k, a = inputs["q"], inputs["k"], inputs["A"]
    g, beta = inputs["g"], inputs["beta"]
    B, HK, T, _ = q.shape
    HV = inputs["v"].shape[1]
    chunk_size = int(inputs["chunk_size"])
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    outputs = [
        torch.zeros((B, HV, T, chunk_size), dtype=calc, device=q.device)
        for _ in range(3)
    ]
    gate = torch.exp2 if inputs["use_exp2"] else torch.exp
    group = HV // HK
    for b, start, end in _work_chunks(inputs):
        for hv in range(HV):
            hk = hv // group
            length = end - start
            g_chunk = g[b, hv, start:end].to(calc)
            beta_chunk = beta[b, hv, start:end].to(calc)
            a_chunk = a[b, hv, start:end, :length].to(calc)
            bg = beta_chunk * gate(g_chunk)
            outputs[0][b, hv, start:end, :length] = a_chunk * bg.unsqueeze(0)
            outputs[1][b, hv, start:end, :length] = (
                a_chunk * beta_chunk.unsqueeze(0)
            )

            score = k[b, hk, start:end].to(calc) @ q[b, hk, start:end].to(calc).t()
            delta = g_chunk.unsqueeze(0) - g_chunk.unsqueeze(1)
            valid = torch.triu(
                torch.ones((length, length), dtype=torch.bool, device=q.device)
            )
            gate_delta = torch.zeros_like(delta)
            gate_delta[valid] = gate(delta[valid])
            outputs[2][b, hv, start:end, :length] = (
                float(inputs["scale"]) * score * gate_delta
            )
    return tuple(outputs)


def _cpu_ref(inputs: dict[str, Any], *, quantize_workspace: bool = True):
    """CPU reference; optionally model the device's main-dtype workspace writes."""
    q, k, v, d_o = inputs["q"], inputs["k"], inputs["v"], inputs["d_o"]
    B, HK, T, _ = q.shape
    HV = v.shape[1]
    chunk_size = int(inputs["chunk_size"])
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    main_dtype = inputs["main_dtype"]
    a_bg_all, a_beta_all, d_all = _compute_intermediates(inputs)
    outputs = [torch.zeros_like(v, dtype=calc) for _ in range(3)]
    group = HV // HK
    for b, start, end in _work_chunks(inputs):
        for hv in range(HV):
            hk = hv // group
            length = end - start
            a_bg = a_bg_all[b, hv, start:end, :length]
            a_beta = a_beta_all[b, hv, start:end, :length]
            d_local = d_all[b, hv, start:end, :length]
            if quantize_workspace:
                # Stage 1 writes main dtype workspace before the three Cube MMADs.
                a_bg = a_bg.to(main_dtype).to(calc)
                a_beta = a_beta.to(main_dtype).to(calc)
                d_local = d_local.to(main_dtype).to(calc)
            outputs[0][b, hv, start:end] = (
                a_bg @ k[b, hk, start:end].to(calc)
            )
            outputs[1][b, hv, start:end] = (
                a_beta @ v[b, hv, start:end].to(calc)
            )
            outputs[2][b, hv, start:end] = (
                d_local @ d_o[b, hv, start:end].to(calc)
            )
    return tuple(outputs)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    outputs = _cpu_ref(inputs)
    if high_precision:
        # 普通 CPU golden 保留设备输出的主 dtype；独立无量化 FP64
        # 数学标杆由 run_cpu_benchmark 提供，不经过这里的输出转换。
        main_dtype = inputs["main_dtype"]
        outputs = tuple(output.to(main_dtype) for output in outputs)
    return outputs


def run_cpu_benchmark(spec: dict[str, Any]):
    """FP64 mathematical benchmark without workspace or output quantization.

    Inputs still originate from the same main/auxiliary-dtype samples as the DUT.
    Only FunctionApi.cpu_benchmark normalizes the FP64 outputs to FP32 for ATK.
    This is separate from run_cpu's device-matching quantized workspace reference.
    """
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=True)
    return _cpu_ref(inputs, quantize_workspace=False)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)

    watchdog = subprocess.Popen(
        ["sh", "-c", 'sleep 60; kill -KILL "$1"', "operator-watchdog", str(os.getpid())]
    )
    try:
        outputs = ascendc.chunk_gdn_bwd_intra(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["A"],
            inputs["d_o"],
            inputs["scale"],
            inputs["chunk_size"],
            cu_seqlens=inputs["cu_seqlens"],
            chunk_indices=inputs["chunk_indices"],
            use_exp2=inputs["use_exp2"],
        )
        torch.npu.synchronize()
    finally:
        watchdog.terminate()
        watchdog.wait()
    return outputs


@register("executor_chunk_gdn_bwd_intra")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"
        self.qkv_type = None

    def cpu(self, input_data: InputDataset, with_output: bool = False):
        """设备匹配 CPU 参考：主 dtype 输入，FP32 累加与工作区量化。"""
        spec = _case_spec(input_data, OP_NAME)
        return _finite_tuple(run_cpu(spec, high_precision=False), golden=True)

    def cpu_benchmark(self, input_data: InputDataset, with_output: bool = False):
        """高精度 CPU 真值（fp64 计算，输出归一 fp32）。"""
        spec = _case_spec(input_data, OP_NAME)
        return _finite_tuple(run_cpu_benchmark(spec), golden=True)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device in {"npu", "pyaclnn"}:
            spec = _case_spec(input_data, OP_NAME)
            return _finite_tuple(run_npu(spec, input_data), golden=False)
        if self.device == "cpu":
            # ATK runs benchmark tasks through __call__ with this flag; BaseApi
            # does not automatically dispatch to a cpu_benchmark method.
            if self.task_result.is_benchmark_task:
                return self.cpu_benchmark(input_data, with_output)
            spec = _case_spec(input_data, OP_NAME)
            return _finite_tuple(run_cpu(spec, high_precision=True), golden=True)
        raise RuntimeError(
            f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}"
        )
