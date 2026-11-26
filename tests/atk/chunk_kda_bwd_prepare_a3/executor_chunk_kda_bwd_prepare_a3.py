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

from _prepare_input_utils import _finite_tuple, _marker_device, _num_chunks, _randn
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
    if not (case_id in profiles if isinstance(profiles, dict) else 0 <= case_id < len(profiles)):
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


def _user_build_inputs(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
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


def _build_inputs(spec, device):
    inputs = _user_build_inputs(spec, device)
    variable = spec["mode"] != "dense"
    rank3 = spec["layout"] in ("NTD", "TND")
    if variable and not rank3:
        for name in ("aqk", "v_new", "d_o", "h"):
            inputs[name] = inputs[name].unsqueeze(0)
    elif not variable and rank3:
        for name in ("aqk", "v_new", "d_o", "h"):
            inputs[name] = inputs[name].squeeze(0)
    return inputs

from reference import _cpu_golden


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
        layout=str(spec["layout"]), gva_ratio=int(spec.get("gva_ratio", 1)),
        cu_seqlens=inputs["cu_seqlens"], chunk_indices=inputs["chunk_indices"] if spec.get("explicit_chunk_indices", True) else None,
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
        import json
        root = Path(__file__).resolve().parent
        files = ["atk_chunk_kda_bwd_prepare_a3_perf.json"] if self._profiling else ["atk_chunk_kda_bwd_prepare_a3.json", "r1_regression.json"]
        if not hasattr(self, "_profiles"):
            self._profiles = {}
            for filename in files:
                for row in json.loads((root / filename).read_text()):
                    item = json.loads(next(x["range_values"] for x in row["inputs"] if x["name"] == "case_spec"))
                    self._profiles[item["case_id"]] = item
        spec = _case_spec(input_data, self._profiles)
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
