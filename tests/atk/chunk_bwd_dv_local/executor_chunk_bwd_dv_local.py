"""chunk_bwd_dv_local 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
"""

from __future__ import annotations

import math
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
    _int_tensor,
    _marker_device,
    _orig_dtype,
    _randn,
)


OP_NAME = "chunk_bwd_dv_local"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "var", "variable"}


def _case_id(spec: dict[str, Any]) -> int:
    return int(spec.get("case_id", spec.get("id", 0)))


def _is_var_len(spec: dict[str, Any]) -> bool:
    if "var_len" in spec:
        return _as_bool(spec.get("var_len"))
    if "variable_len" in spec:
        return _as_bool(spec.get("variable_len"))
    return _case_id(spec) % 4 >= 2


def _g_dtype_name(spec: dict[str, Any], q_dtype_name: str) -> str:
    explicit = spec.get("g_dtype") or spec.get("gtype")
    if explicit:
        return str(explicit).lower()
    return "fp32" if _case_id(spec) % 2 else q_dtype_name


def _make_cu_seqlens(total: int, spec: dict[str, Any], device: torch.device) -> torch.Tensor:
    total = int(total)
    requested_len = spec.get("cu_seqlens_len")
    if requested_len is not None:
        seq_count = max(1, min(total, int(requested_len) - 1))
    else:
        seq_count = max(1, min(total, int(spec.get("seq_count", min(4, total)))))
    base, rem = divmod(total, seq_count)
    cur = 0
    values = [0]
    for idx in range(seq_count):
        cur += base + (1 if idx < rem else 0)
        values.append(cur)
    return _int_tensor(values, device)


def _prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    values = [int(x) for x in cu_seqlens.detach().cpu().reshape(-1).tolist()]
    pairs: list[int] = []
    for seq_idx, (start, end) in enumerate(zip(values[:-1], values[1:])):
        length = end - start
        for chunk_idx in range((length + int(chunk_size) - 1) // int(chunk_size)):
            pairs.extend([seq_idx, chunk_idx])
    return _int_tensor(pairs, cu_seqlens.device)


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    g_dtype_name = _g_dtype_name(spec, dtype_name)
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    g_calc_dtype = torch.float64 if high_precision else _orig_dtype(g_dtype_name)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    if _is_var_len(spec):
        B = 1
    chunk_size = int(spec["chunk_size"])
    inputs = {
        "q": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 2),
        "g": _gate((B, HV, T), g_calc_dtype, device, seed + 4),
        "do": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 5),
        "chunk_size": chunk_size,
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
    }
    if _is_var_len(spec):
        inputs["cu_seqlens"] = _make_cu_seqlens(T, spec, device)
        inputs["chunk_indices"] = _prepare_chunk_indices(inputs["cu_seqlens"], chunk_size)
    else:
        inputs["cu_seqlens"] = None
        inputs["chunk_indices"] = None
    return inputs


def _tensor_from_kwargs(input_data: InputDataset, name: str) -> torch.Tensor:
    value = input_data.kwargs.get(name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{OP_NAME} 缺少输入 Tensor: {name}")
    return value


def _chunk_bwd_dv_local_ref(inputs: dict[str, Any]):
    q, k, do, g = inputs["q"], inputs["k"], inputs["do"], inputs["g"]
    B, HK, T, _ = q.shape
    HV = do.shape[1]
    chunk_size = int(inputs["chunk_size"])
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    out = torch.zeros_like(do, dtype=calc)
    group = max(HV // HK, 1)
    cu_seqlens = inputs.get("cu_seqlens")
    chunk_indices = inputs.get("chunk_indices")
    if cu_seqlens is None:
        chunks = [(b, start, end) for b in range(B) for start, end in _chunks(T, chunk_size)]
    else:
        if chunk_indices is None:
            chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)
        cu_values = [int(x) for x in cu_seqlens.detach().cpu().reshape(-1).tolist()]
        pair_values = [int(x) for x in chunk_indices.detach().cpu().reshape(-1).tolist()]
        chunks = []
        for seq_idx, chunk_idx in zip(pair_values[0::2], pair_values[1::2]):
            seq_len = cu_values[seq_idx + 1] - cu_values[seq_idx]
            local_start = chunk_idx * chunk_size
            local_end = min(local_start + chunk_size, seq_len)
            if local_start < local_end:
                chunks.append((0, cu_values[seq_idx] + local_start, cu_values[seq_idx] + local_end))
    for b, start, end in chunks:
        for hv in range(HV):
            hk = hv // group
            score = torch.matmul(k[b, hk, start:end].to(calc), q[b, hk, start:end].to(calc).t())
            score = score * float(inputs["scale"])
            g_chunk = g[b, hv, start:end].to(calc)
            gate = torch.exp(g_chunk[None, :] - g_chunk[:, None])
            mask = torch.triu(torch.ones_like(score))
            out[b, hv, start:end] = torch.matmul(score * gate * mask, do[b, hv, start:end].to(calc))
    return out.to(do.dtype)


def run_cpu(input_data: InputDataset, high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    q = _tensor_from_kwargs(input_data, "q").detach().cpu()
    k = _tensor_from_kwargs(input_data, "k").detach().cpu()
    do = _tensor_from_kwargs(input_data, "do").detach().cpu()
    g = _tensor_from_kwargs(input_data, "g").detach().cpu()
    if high_precision:
        q = q.to(torch.float64)
        k = k.to(torch.float64)
        do = do.to(torch.float64)
        g = g.to(torch.float64)
    inputs = {
        "q": q,
        "k": k,
        "do": do,
        "g": g,
        "chunk_size": int(input_data.kwargs["chunk_size"]),
        "scale": float(input_data.kwargs["scale"]),
        "cu_seqlens": input_data.kwargs.get("cu_seqlens"),
        "chunk_indices": input_data.kwargs.get("chunk_indices"),
    }
    return _chunk_bwd_dv_local_ref(inputs)


def run_npu(input_data: InputDataset):
    """运行 NPU DUT。"""
    from fla_npu.ops import ascendc

    return ascendc.chunk_bwd_dv_local(
        input_data.kwargs["q"],
        input_data.kwargs["k"],
        input_data.kwargs["do"],
        input_data.kwargs["g"],
        input_data.kwargs["scale"],
        input_data.kwargs["chunk_size"],
        g_gamma=None,
        A=None,
        cu_seqlens=input_data.kwargs.get("cu_seqlens"),
        chunk_indices=input_data.kwargs.get("chunk_indices"),
    )


@register("executor_chunk_bwd_dv_local")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if "q" not in input_data.kwargs:
            self.init_by_input_data(input_data)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(input_data)
        elif self.device == "cpu":
            outputs = run_cpu(input_data, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs)

    def init_by_input_data(self, input_data: InputDataset):
        spec = _case_spec(input_data, OP_NAME)
        dtype_name = str(spec.get("dtype", "bf16")).lower()
        g_dtype_name = _g_dtype_name(spec, dtype_name)
        device = _marker_device(input_data) if self.device in {"npu", "pyaclnn"} else torch.device("cpu")
        inputs = build_inputs(spec, device, high_precision=False)

        q = inputs["q"].to(_orig_dtype(dtype_name))
        k = inputs["k"].to(_orig_dtype(dtype_name))
        do = inputs["do"].to(_orig_dtype(dtype_name))
        g = inputs["g"].to(_orig_dtype(g_dtype_name))

        if self.device in {"npu", "pyaclnn"}:
            q = q.to(device)
            k = k.to(device)
            do = do.to(device)
            g = g.to(device)

        input_data.kwargs["q"] = q.contiguous()
        input_data.kwargs["k"] = k.contiguous()
        input_data.kwargs["do"] = do.contiguous()
        input_data.kwargs["g"] = g.contiguous()
        input_data.kwargs["scale"] = float(inputs["scale"])
        input_data.kwargs["chunk_size"] = int(inputs["chunk_size"])
        input_data.kwargs["g_gamma"] = None
        input_data.kwargs["A"] = None
        input_data.kwargs["cu_seqlens"] = inputs["cu_seqlens"]
        input_data.kwargs["chunk_indices"] = inputs["chunk_indices"]
