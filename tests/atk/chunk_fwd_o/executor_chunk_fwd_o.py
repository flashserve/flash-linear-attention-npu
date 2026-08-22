"""chunk_fwd_o 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _calc_dtype,
    _finite_tuple,
    _marker_device,
    _num_chunks,
    _orig_dtype,
    _randn,
)
from gen_chunk_fwd_o import CASE_COUNT, OP_NAME, PROFILES


_REVIEWED_SPECS = PROFILES


def _reviewed_case_spec(input_data: InputDataset) -> dict[str, Any]:
    """Resolve the immutable reviewed spec from ATK's reliably returned case_id.

    ATK 26.7.8 does not preserve an ``attr/non_param`` JSON value in
    ``InputDataset.kwargs``.  ``case_id`` is preserved, so it is the canonical
    transport for this fixed, reviewed matrix; this also keeps the two
    benchmark nodes on exactly the same input specification.
    """
    raw_id = input_data.kwargs.get("case_id")
    if isinstance(raw_id, torch.Tensor):
        raw_id = raw_id.detach().cpu().item()
    try:
        case_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{OP_NAME} ATK case is missing a valid case_id") from exc
    if not 0 <= case_id < CASE_COUNT:
        raise RuntimeError(f"{OP_NAME} ATK case_id out of range: {case_id}")
    return _REVIEWED_SPECS[case_id]


def _sequence_ranges(spec: dict[str, Any]):
    if spec.get("mode") != "varlen":
        for batch in range(int(spec["B"])):
            yield batch, 0, int(spec["T"]), 0
        return
    offset = 0
    for sequence_length in spec["seqlens"]:
        end = offset + int(sequence_length)
        yield 0, offset, end, offset
        offset = end


def _chunk_indices(spec: dict[str, Any]) -> list[int] | None:
    if spec.get("mode") != "varlen":
        return None
    indices: list[int] = []
    for sequence_id, sequence_length in enumerate(spec["seqlens"]):
        for chunk_id in range(_num_chunks(int(sequence_length), int(spec["chunk_size"]))):
            indices.extend((sequence_id, chunk_id))
    return indices


def _num_states(spec: dict[str, Any]) -> int:
    if spec.get("mode") != "varlen":
        return _num_chunks(int(spec["T"]), int(spec["chunk_size"]))
    return sum(_num_chunks(length, int(spec["chunk_size"])) for length in spec["seqlens"])


def _build_gate(spec: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Generate stable, chunk-local gates while preserving the declared g dtype."""
    shape = (int(spec["B"]), int(spec["HV"]), int(spec["T"]))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(spec["seed"]) + 4)
    gate = torch.empty(shape, dtype=torch.float32)
    for batch, start, end, _ in _sequence_ranges(spec):
        for chunk_start in range(start, end, int(spec["chunk_size"])):
            chunk_end = min(chunk_start + int(spec["chunk_size"]), end)
            increments = torch.rand((shape[1], chunk_end - chunk_start), generator=generator) * 0.01 + 0.001
            gate[batch, :, chunk_start:chunk_end] = -torch.cumsum(increments, dim=-1)
    return gate.to(_orig_dtype(str(spec["g_dtype"]))).to(device)


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    return {
        "q": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 2),
        "v": _randn((B, HV, T, V), dtype_name, calc_dtype, device, seed + 3),
        "g": _build_gate(spec, device),
        "h": _randn((B, HV, _num_states(spec), K, V), dtype_name, calc_dtype, device, seed + 6),
        "chunk_size": chunk_size,
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
    }


def _chunk_fwd_o_ref(inputs, spec: dict[str, Any]):
    q, k, v, h, g = inputs["q"], inputs["k"], inputs["v"], inputs["h"], inputs["g"]
    B, HK, T, _ = q.shape
    HV, V = v.shape[1], v.shape[3]
    chunk_size = int(inputs["chunk_size"])
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    out = torch.zeros((B, HV, T, V), dtype=calc, device=q.device)
    group = HV // HK
    state_offset = 0
    for b, sequence_start, sequence_end, _ in _sequence_ranges(spec):
        sequence_chunks = _num_chunks(sequence_end - sequence_start, chunk_size)
        for hv in range(HV):
            hk = hv // group
            for chunk_id in range(sequence_chunks):
                start = sequence_start + chunk_id * chunk_size
                end = min(start + chunk_size, sequence_end)
                q_chunk = q[b, hk, start:end].to(calc)
                k_chunk = k[b, hk, start:end].to(calc)
                v_chunk = v[b, hv, start:end].to(calc)
                g_chunk = g[b, hv, start:end].to(calc)
                local = torch.matmul(q_chunk, k_chunk.t())
                gate = torch.exp(g_chunk[:, None] - g_chunk[None, :])
                mask = torch.tril(torch.ones_like(local))
                state_index = state_offset + chunk_id if spec.get("mode") == "varlen" else chunk_id
                state = torch.matmul(q_chunk * torch.exp(g_chunk)[:, None], h[b, hv, state_index].to(calc))
                out[b, hv, start:end] = (torch.matmul(local * gate * mask, v_chunk) + state) * float(inputs["scale"])
        state_offset += sequence_chunks
    return out.to(v.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _chunk_fwd_o_ref(inputs, spec)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    cu_seqlens = None
    if spec.get("mode") == "varlen":
        cu_seqlens = [0]
        for length in spec["seqlens"]:
            cu_seqlens.append(cu_seqlens[-1] + int(length))
    return ascendc.npu_chunk_fwd_o(inputs["q"], inputs["k"], inputs["v"], inputs["h"], inputs["scale"], g=inputs["g"], g_gamma=None, cu_seqlens=cu_seqlens, chunk_indices=_chunk_indices(spec), chunk_size=inputs["chunk_size"], transpose_state_layout=False)


@register("executor_chunk_fwd_o")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _reviewed_case_spec(input_data)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        visible_outputs = _finite_tuple(outputs)
        # A single ATK process runs all 200 cases.  Direct aclnn calls release
        # their input tensors after this method returns, but the NPU allocator
        # otherwise retains prior-case cache and can reserve the whole card.
        # Keep this test-only cleanup opt-in so performance measurements retain
        # their normal allocator behaviour.
        if self.device in {"npu", "pyaclnn"} and os.getenv("ATK_RELEASE_NPU_CACHE") == "1":
            torch.npu.synchronize()
            torch.npu.empty_cache()
        return visible_outputs
