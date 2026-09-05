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
    _finite_tuple,
    _gate,
    _marker_device,
    _orig_dtype,
    _randn,
)


OP_NAME = "chunk_bwd_dv_local"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    local_indices = torch.cat(
        [torch.arange(_ceil_div(int(length.item()), chunk_size)) for length in lens]
    )
    sequence_indices = local_indices.eq(0).cumsum(0) - 1
    return torch.stack([sequence_indices, local_indices], dim=1).to(cu_seqlens)


def _build_cu_seqlens(spec: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    if not _as_bool(spec.get("var_len", False)):
        return None
    total_t = int(spec["T"])
    sequence_count = max(int(spec.get("cu_seqlens_len", spec.get("B", 1))), 1)
    base, remainder = divmod(total_t, sequence_count)
    lengths = [base + (1 if index < remainder else 0) for index in range(sequence_count)]
    if any(length <= 0 for length in lengths):
        raise RuntimeError(
            f"{OP_NAME} varlen sequence count ({sequence_count}) must not exceed T ({total_t})"
        )
    cu_seqlens = [0]
    for length in lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    return torch.tensor(cu_seqlens, dtype=torch.int64, device=device)


def build_inputs(
    spec: dict[str, Any], device: torch.device, high_precision: bool = False
) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    g_dtype_name = str(spec.get("g_dtype", dtype_name)).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    g_calc_dtype = torch.float64 if high_precision else _orig_dtype(g_dtype_name)
    seed = int(spec.get("seed", 20260817))

    batch, hk, hv, total_t, k_dim, v_dim = (
        int(spec[key]) for key in ("B", "HK", "HV", "T", "K", "V")
    )
    chunk_size = int(spec["chunk_size"])
    if hv % hk != 0:
        raise RuntimeError(f"HV({hv}) must be divisible by HK({hk})")

    cu_seqlens = _build_cu_seqlens(spec, device)
    physical_batch = 1 if cu_seqlens is not None else batch
    q_shape = (physical_batch, hk, total_t, k_dim)
    do_shape = (physical_batch, hv, total_t, v_dim)
    g_shape = (physical_batch, hv, total_t)
    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens.cpu(), chunk_size).to(device)

    return {
        "q": _randn(q_shape, dtype_name, calc_dtype, device, seed + 1),
        "k": _randn(q_shape, dtype_name, calc_dtype, device, seed + 2),
        "do": _randn(do_shape, dtype_name, calc_dtype, device, seed + 3),
        "g": _gate(g_shape, g_calc_dtype, device, seed + 4)
        .to(_orig_dtype(g_dtype_name))
        .to(g_calc_dtype),
        "chunk_size": chunk_size,
        "scale": float(spec.get("scale", 1.0 / math.sqrt(k_dim))),
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
    }


def _compute_chunk(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    *,
    batch_index: int,
    start: int,
    end: int,
    sequence_offset: int,
    sequence_length: int,
    chunk_size: int,
    scale: float,
    high_precision: bool,
) -> None:
    hk = q.shape[1]
    hv = do.shape[1]
    v_dim = do.shape[3]
    h_ratio = hv // hk
    chunk_len = end - start
    total_t = q.shape[2]
    block_t = min(chunk_size, max(16, 2 ** math.ceil(math.log2(max(total_t, 1)))))
    local_chunk = (start - sequence_offset) // chunk_size
    token_offsets = local_chunk * block_t + torch.arange(block_t, device=q.device)
    valid = token_offsets < sequence_length
    mask = (token_offsets[:, None] <= token_offsets[None, :]) & valid[:, None] & valid[None, :]
    calc_dtype = torch.float64 if high_precision else torch.float32

    for hv_index in range(hv):
        hk_index = hv_index // h_ratio
        q_chunk = q[batch_index, hk_index, start:end].to(calc_dtype)
        k_chunk = k[batch_index, hk_index, start:end].to(calc_dtype)
        g_chunk = g[batch_index, hv_index, start:end].to(calc_dtype)
        attention = torch.zeros((block_t, block_t), dtype=calc_dtype, device=q.device)
        attention[:chunk_len, :chunk_len] = torch.matmul(k_chunk, q_chunk.t())
        gate = torch.exp(
            torch.clamp(g_chunk[None, :] - g_chunk[:, None], max=0.0)
        )
        attention[:chunk_len, :chunk_len] *= gate * scale
        attention = torch.where(mask, attention, torch.zeros_like(attention))
        attention = attention.to(torch.float32 if high_precision else q.dtype)
        for v_start in range(0, v_dim, 128):
            v_end = min(v_start + 128, v_dim)
            do_chunk = do[batch_index, hv_index, start:end, v_start:v_end]
            if high_precision:
                do_chunk = do_chunk.to(torch.float32)
            out[batch_index, hv_index, start:end, v_start:v_end] += torch.matmul(
                attention[:chunk_len, :chunk_len], do_chunk
            ).to(calc_dtype)


def _chunk_bwd_dv_local_fixed_ref(
    inputs: dict[str, Any], high_precision: bool = False
) -> torch.Tensor:
    q, k, do, g = inputs["q"], inputs["k"], inputs["do"], inputs["g"]
    batch, total_t = q.shape[0], q.shape[2]
    chunk_size = int(inputs["chunk_size"])
    calc_dtype = torch.float64 if high_precision else torch.float32
    out = torch.zeros_like(do, dtype=calc_dtype)
    for batch_index in range(batch):
        for start in range(0, total_t, chunk_size):
            _compute_chunk(
                out,
                q,
                k,
                do,
                g,
                batch_index=batch_index,
                start=start,
                end=min(start + chunk_size, total_t),
                sequence_offset=0,
                sequence_length=total_t,
                chunk_size=chunk_size,
                scale=float(inputs["scale"]),
                high_precision=high_precision,
            )
    return out.to(do.dtype)


def _chunk_bwd_dv_local_varlen_ref(
    inputs: dict[str, Any], high_precision: bool = False
) -> torch.Tensor:
    q, k, do, g = inputs["q"], inputs["k"], inputs["do"], inputs["g"]
    cu_seqlens = inputs["cu_seqlens"]
    chunk_indices = inputs["chunk_indices"]
    if cu_seqlens is None or chunk_indices is None:
        raise RuntimeError("var_len reference requires cu_seqlens and chunk_indices")

    calc_dtype = torch.float64 if high_precision else torch.float32
    out = torch.zeros_like(do, dtype=calc_dtype)
    for sequence_index, local_chunk in chunk_indices.cpu().tolist():
        begin = int(cu_seqlens[sequence_index].item())
        sequence_end = int(cu_seqlens[sequence_index + 1].item())
        start = begin + int(local_chunk) * int(inputs["chunk_size"])
        end = min(start + int(inputs["chunk_size"]), sequence_end)
        if start >= end:
            continue
        _compute_chunk(
            out,
            q,
            k,
            do,
            g,
            batch_index=0,
            start=start,
            end=end,
            sequence_offset=begin,
            sequence_length=sequence_end - begin,
            chunk_size=int(inputs["chunk_size"]),
            scale=float(inputs["scale"]),
            high_precision=high_precision,
        )
    return out.to(do.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 高精度 golden。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    if inputs["cu_seqlens"] is not None:
        return _chunk_bwd_dv_local_varlen_ref(inputs, high_precision=high_precision)
    return _chunk_bwd_dv_local_fixed_ref(inputs, high_precision=high_precision)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    cu_seqlens = inputs["cu_seqlens"]
    chunk_indices = inputs["chunk_indices"]
    return ascendc.chunk_bwd_dv_local(
        inputs["q"],
        inputs["k"],
        inputs["do"],
        inputs["g"],
        inputs["scale"],
        inputs["chunk_size"],
        g_gamma=None,
        A=None,
        cu_seqlens=None if cu_seqlens is None else cu_seqlens.cpu().tolist(),
        chunk_indices=None if chunk_indices is None else chunk_indices.cpu().view(-1).tolist(),
    )


@register("executor_chunk_bwd_dv_local")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")
