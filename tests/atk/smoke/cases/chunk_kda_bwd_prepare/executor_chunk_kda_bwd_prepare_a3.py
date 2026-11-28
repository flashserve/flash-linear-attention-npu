"""ATK native L1 double-benchmark executor for ``ChunkKdaBwdPrepare``.

The CPU reference below is an independent expression of the three documented
outputs; it never calls the NPU operator or any implementation helper from the
KernelA source tree.
Input layouts are BNSD/BSND (dense) and NTD/TND (packed), while outputs stay
head-major BNSD/NTD.  State tensors are chunk-major, and dAqk_raw uses FP32
handoff as specified by docs/design_a3.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

# ATK runs several executor processes concurrently.  Leaving PyTorch at the
# host-wide thread default makes each FP64 Golden worker create hundreds of
# BLAS threads, which oversubscribes A5 validation hosts and can stall the
# task scheduler.  Keep this configurable, but conservative by default.
torch.set_num_threads(max(1, int(os.environ.get("KDA_PREPARE_ATK_CPU_THREADS", "1"))))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _finite_tuple, _marker_device, _num_chunks, _randn
from _kda_profiles import case_profiles
from gen_chunk_kda_bwd_prepare_a3 import CASE_COUNT, OP_NAME, PROFILES


def _case_spec(input_data: InputDataset, profiles=PROFILES) -> dict[str, Any]:
    """Use case_id, the stable transport preserved by ATK 26.7.8."""
    raw_id = input_data.kwargs.get("case_id")
    if isinstance(raw_id, torch.Tensor):
        raw_id = raw_id.detach().cpu().item()
    try:
        case_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{OP_NAME} ATK case is missing case_id") from exc
    if not 0 <= case_id < len(profiles):
        raise RuntimeError(f"{OP_NAME} ATK case_id out of range: {case_id}")
    spec = dict(profiles[case_id])
    spec["seed"] = int(input_data.kwargs.get("seed", spec["seed"]))
    return spec


def _chunk_metadata(seqlens: list[int], chunk_size: int) -> tuple[list[int], list[int]]:
    cu = [0]
    indices: list[int] = []
    for sequence_id, length in enumerate(seqlens):
        cu.append(cu[-1] + int(length))
        for chunk_id in range(_num_chunks(int(length), chunk_size)):
            indices.extend((sequence_id, chunk_id))
    return cu, indices


def _build_inputs(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Build BF16 DUT inputs deterministically on CPU, then move to device.

    CPU Golden evaluates each small tile in FP64 later.  Keeping the stored
    tensors BF16 avoids accidentally changing the operator input values and
    prevents a full-model H=96/T=16K golden from materialising a multi-GB FP64
    copy of ``h``.
    """
    mode = str(spec["mode"])
    batch, heads, tokens = (int(spec[key]) for key in ("B", "H", "T"))
    chunk_size = int(spec["chunk_size"])
    if mode == "dense":
        chunk_count = _num_chunks(tokens, chunk_size)
        aqk_shape = (batch, heads, tokens, 64)
        value_shape = (batch, heads, tokens, 128)
        h_shape = (batch, chunk_count, heads, 128, 128)
        cu_seqlens = chunk_indices = None
    else:
        seqlens = [int(value) for value in spec["seqlens"]]
        cu_seqlens, chunk_indices = _chunk_metadata(seqlens, chunk_size)
        chunk_count = len(chunk_indices) // 2
        aqk_shape = (heads, tokens, 64)
        value_shape = (heads, tokens, 128)
        h_shape = (chunk_count, heads, 128, 128)
    seed = int(spec["seed"])
    cpu_device = torch.device("cpu")
    inputs = {
        "aqk": _randn(aqk_shape, "bf16", torch.bfloat16, cpu_device, seed + 1),
        "v_new": _randn(value_shape, "bf16", torch.bfloat16, cpu_device, seed + 2),
        "d_o": _randn(value_shape, "bf16", torch.bfloat16, cpu_device, seed + 3),
        "h": _randn(h_shape, "bf16", torch.bfloat16, cpu_device, seed + 4),
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
    }
    # Construct continuous physical inputs on CPU before the DUT call.  No
    # layout conversion is inserted on NPU; output layout remains head-major.
    if spec["layout"] in ("BSND", "TND"):
        axes = (0, 2, 1, 3) if mode == "dense" else (1, 0, 2)
        for name in ("aqk", "v_new", "d_o"):
            inputs[name] = inputs[name].permute(axes).contiguous()
    for name in ("aqk", "v_new", "d_o", "h"):
        inputs[name] = inputs[name].to(device)
    return inputs


def _cpu_golden(inputs: dict[str, Any], spec: dict[str, Any], high_precision: bool) -> tuple[torch.Tensor, ...]:
    """Independent CPU formulation of dAqk, dv and dq_raw.

    For each valid in-chunk row range ``r``:
      dAqk = tril(dO @ v_new.T) * scale
      dv   = aqk[:, :r].T @ dO
      dq   = dO @ h          (or dO @ h.T for state_v_first=False)
    Matrix products and scale application use FP64 for Golden (the automatic
    ATK is_benchmark_task) and FP32 for Bench (the explicit ordinary CPU node).
    Both receive identical BF16 inputs, then convert outputs to the public
    FP32/BF16/FP32 dtypes required for the native ATK comparison.
    """
    aqk, v_new, d_o, h = (inputs[key].cpu() for key in ("aqk", "v_new", "d_o", "h"))
    if spec["layout"] in ("BSND", "TND"):
        axes = (0, 2, 1, 3) if spec["mode"] == "dense" else (1, 0, 2)
        aqk, v_new, d_o = (tensor.permute(axes) for tensor in (aqk, v_new, d_o))
    calc_dtype = torch.float64 if high_precision else torch.float32
    mode = str(spec["mode"])
    chunks = int(spec["chunk_size"])
    scale = float(spec["scale"])
    state_v_first = bool(spec["state_v_first"])
    d_aqk = torch.zeros_like(aqk, dtype=torch.float32, memory_format=torch.contiguous_format)
    dv = torch.empty_like(v_new, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
    dq_raw = torch.empty_like(d_o, dtype=torch.float32, memory_format=torch.contiguous_format)

    if mode == "dense":
        batch, heads, tokens, _ = aqk.shape
        sequence_ranges = [(batch_id, 0, tokens, 0) for batch_id in range(batch)]
        state_base = 0
        for batch_id, begin_seq, end_seq, _ in sequence_ranges:
            for head in range(heads):
                for local_chunk, begin in enumerate(range(begin_seq, end_seq, chunks)):
                    end = min(begin + chunks, end_seq)
                    rows = end - begin
                    _write_golden_tile(
                        d_aqk, dv, dq_raw, aqk, v_new, d_o, h,
                        batch_id, head, begin, end, rows, local_chunk,
                        scale, state_v_first, calc_dtype,
                        dense=True,
                    )
        return d_aqk, dv, dq_raw

    heads, tokens, _ = aqk.shape
    del tokens
    offset = 0
    state_index = 0
    for sequence_length in (int(value) for value in spec["seqlens"]):
        sequence_end = offset + sequence_length
        for begin in range(offset, sequence_end, chunks):
            end = min(begin + chunks, sequence_end)
            rows = end - begin
            for head in range(heads):
                _write_golden_tile(
                    d_aqk, dv, dq_raw, aqk, v_new, d_o, h,
                    0, head, begin, end, rows, state_index,
                    scale, state_v_first, calc_dtype,
                    dense=False,
                )
            state_index += 1
        offset = sequence_end
    return d_aqk, dv, dq_raw


def _write_golden_tile(
    d_aqk: torch.Tensor, dv: torch.Tensor, dq_raw: torch.Tensor,
    aqk: torch.Tensor, v_new: torch.Tensor, d_o: torch.Tensor, h: torch.Tensor,
    batch: int, head: int, begin: int, end: int, rows: int, state_index: int,
    scale: float, state_v_first: bool, calc_dtype: torch.dtype, *,
    dense: bool,
) -> None:
    if dense:
        aqk_tile = aqk[batch, head, begin:end]
        v_tile = v_new[batch, head, begin:end]
        do_tile = d_o[batch, head, begin:end]
        h_tile = h[batch, state_index, head]
    else:
        aqk_tile = aqk[head, begin:end]
        v_tile = v_new[head, begin:end]
        do_tile = d_o[head, begin:end]
        h_tile = h[state_index, head]
    aqk_calc, v_calc, do_calc, h_calc = (tile.to(calc_dtype) for tile in (aqk_tile, v_tile, do_tile, h_tile))
    tri = torch.tril(torch.ones((rows, rows), dtype=calc_dtype))
    a_raw = do_calc @ v_calc.transpose(0, 1)
    a_tile = a_raw * tri * scale
    h_right = h_calc if state_v_first else h_calc.transpose(0, 1)
    q_tile = do_calc @ h_right
    d_tile = aqk_calc[:, :rows].transpose(0, 1) @ do_calc
    if dense:
        d_aqk[batch, head, begin:end, :rows] = a_tile.float()
        dv[batch, head, begin:end] = d_tile.to(torch.bfloat16)
        dq_raw[batch, head, begin:end] = q_tile.float()
    else:
        d_aqk[head, begin:end, :rows] = a_tile.float()
        dv[head, begin:end] = d_tile.to(torch.bfloat16)
        dq_raw[head, begin:end] = q_tile.float()


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    return _cpu_golden(_build_inputs(spec, torch.device("cpu")), spec, high_precision)


def run_npu(spec: dict[str, Any], input_data: InputDataset, inputs=None):
    if inputs is None:
        inputs = _build_inputs(spec, _marker_device(input_data))
    from fla_npu.ops import ascendc

    operator = getattr(ascendc, "chunk_kda_bwd_prepare", None)
    if operator is None:
        operator = getattr(ascendc, "npu_chunk_kda_bwd_prepare", None)
    if operator is None:
        raise RuntimeError("fla_npu.ops.ascendc does not export chunk_kda_bwd_prepare with layout")

    return operator(
        inputs["aqk"], inputs["v_new"], inputs["d_o"], inputs["h"],
        scale=float(spec["scale"]), chunk_size=int(spec["chunk_size"]),
        state_v_first=bool(spec["state_v_first"]),
        layout=str(spec["layout"]),
        cu_seqlens=inputs["cu_seqlens"], chunk_indices=inputs["chunk_indices"],
    )


@register("executor_chunk_kda_bwd_prepare_a3")
class FunctionApi(BaseApi):
    """Three ATK roles: NPU DUT, ordinary CPU FP32 Bench, CPU FP64 Golden."""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        # Native double benchmark marks only its automatic Golden task True.
        # The explicit CPU node is Bench and must calculate in FP32.
        self.high_precision = self.device == "cpu" and bool(task_result.is_benchmark_task)
        if self.device == "cpu":
            role = "golden" if self.high_precision else "bench"
            precision = "FP64" if self.high_precision else "FP32"
            print(f"[kda-prepare-double-atk] role={role} calc_dtype={precision}", flush=True)
        self._profiling = "performance_device" in (task_result.task_type or [])
        self._memory_check = bool(task_result.mssanitizer)
        self._input_key = None
        self._inputs = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        profiles = case_profiles(self.task_result, Path(__file__).with_name("chunk_kda_bwd_prepare_a3_10000.json"), PROFILES)
        spec = _case_spec(input_data, profiles)
        if self.device in {"npu", "pyaclnn"}:
            # Repeated performance/DC calls keep the same reviewed values;
            # prepare tensors once so profiling contains only the DUT launch.
            key = (spec["case_id"], spec["seed"], str(_marker_device(input_data)))
            if key != self._input_key:
                self._inputs = None
                self._inputs = _build_inputs(spec, _marker_device(input_data))
                self._input_key = key
            outputs = run_npu(spec, input_data, self._inputs)
        elif self.device == "cpu":
            outputs = run_cpu(spec, high_precision=self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} only supports CPU and NPU ATK nodes, got {self.device!r}")
        # Memory instrumentation observes the DUT itself; finite-value checks
        # are performed in accuracy/DC and would launch unrelated NPU kernels.
        visible = tuple(outputs) if self._profiling or self._memory_check else _finite_tuple(outputs)
        if self.device in {"npu", "pyaclnn"} and os.getenv("ATK_RELEASE_NPU_CACHE") == "1":
            torch.npu.synchronize()
            torch.npu.empty_cache()
        return visible
