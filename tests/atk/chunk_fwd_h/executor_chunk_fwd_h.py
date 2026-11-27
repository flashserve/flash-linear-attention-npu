"""chunk_fwd_h ATK executor and independent CPU reference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fwd_h_cpu import CHUNK_SIZE, K_DIM, V_DIM, PreparedInputs, _reference, _to_physical_state
from _ascendc_common_executor import _case_spec, _finite_tuple, _marker_device


OP_NAME = "chunk_fwd_h"
DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_seqlens(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return None
    return [int(item) for item in text.split(",")]


def _cu_seqlens(seqlens: Optional[list[int]]) -> Optional[tuple[int, ...]]:
    if seqlens is None:
        return None
    values = [0]
    for length in seqlens:
        values.append(values[-1] + int(length))
    return tuple(values)


def _canonical_chunk_indices(
    seqlens: Optional[list[int]],
) -> Optional[tuple[int, ...]]:
    if seqlens is None:
        return None
    values: list[int] = []
    for sequence, length in enumerate(seqlens):
        for chunk in range((int(length) + CHUNK_SIZE - 1) // CHUNK_SIZE):
            values.extend((sequence, chunk))
    return tuple(values)


def _random_normal(
    shape: tuple[int, ...],
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
    scale: float,
) -> torch.Tensor:
    value = torch.randn(shape, generator=generator, dtype=torch.float32) * float(scale)
    return value.to(dtype).to(device)


def _gate_values(
    shape: tuple[int, ...],
    seqlens: Optional[list[int]],
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
    scale: float,
) -> torch.Tensor:
    steps = -torch.rand(shape, generator=generator, dtype=torch.float32) * float(scale)
    if seqlens is None:
        value = torch.cumsum(steps, dim=2)
    else:
        value = torch.empty_like(steps)
        begin = 0
        for length in seqlens:
            end = begin + int(length)
            value[:, :, begin:end] = torch.cumsum(steps[:, :, begin:end], dim=2)
            begin = end
    return value.to(dtype).to(device)


def build_inputs(spec: dict[str, Any], device: torch.device) -> PreparedInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(spec.get("seed", 20260817)))

    batch = int(spec["B"])
    k_heads = int(spec["HK"])
    v_heads = int(spec["HV"])
    total_tokens = int(spec["T"])
    seqlens = _parse_seqlens(spec.get("seqlens"))
    if seqlens is not None and sum(seqlens) != total_tokens:
        raise ValueError(
            f"{OP_NAME}: seqlens sum {sum(seqlens)} does not equal T={total_tokens}"
        )

    input_scale = float(spec.get("input_scale", 0.05))
    gate_step_scale = float(spec.get("gate_step_scale", 0.02))
    state_scale = float(spec.get("state_scale", 0.05))
    gate_dtype = DTYPES[str(spec.get("gate_dtype", "fp32"))]

    k = _random_normal(
        (batch, k_heads, total_tokens, K_DIM),
        generator,
        torch.bfloat16,
        device,
        input_scale,
    )
    w = _random_normal(
        (batch, v_heads, total_tokens, K_DIM),
        generator,
        torch.bfloat16,
        device,
        input_scale,
    )
    u_contiguous = _random_normal(
        (batch, v_heads, total_tokens, V_DIM),
        generator,
        torch.bfloat16,
        device,
        input_scale,
    )
    if _as_bool(spec.get("non_contiguous_u", False)):
        padded = torch.empty(
            (batch, v_heads, total_tokens, V_DIM * 2),
            dtype=torch.bfloat16,
            device=device,
        )
        padded[..., ::2].copy_(u_contiguous)
        u = padded[..., ::2]
        if u.is_contiguous():
            raise AssertionError("non_contiguous_u case unexpectedly produced a contiguous view")
    else:
        u = u_contiguous

    mode = str(spec.get("mode", "g"))
    g = None
    gk = None
    if mode == "g":
        g = _gate_values(
            (batch, v_heads, total_tokens),
            seqlens,
            generator,
            gate_dtype,
            device,
            gate_step_scale,
        )
    elif mode == "gk":
        gk = _gate_values(
            (batch, v_heads, total_tokens, K_DIM),
            seqlens,
            generator,
            gate_dtype,
            device,
            gate_step_scale,
        )
    else:
        raise ValueError(f"{OP_NAME}: unsupported gate mode {mode!r}")

    state_dtype_name = str(spec.get("state_dtype", "none"))
    initial_state = None
    if state_dtype_name != "none":
        sequences = len(seqlens) if seqlens is not None else batch
        logical_state = _random_normal(
            (sequences, v_heads, K_DIM, V_DIM),
            generator,
            DTYPES[state_dtype_name],
            device,
            state_scale,
        )
        initial_state = _to_physical_state(
            logical_state, _as_bool(spec.get("state_v_first", False))
        )

    cu = _cu_seqlens(seqlens)
    indices = (
        _canonical_chunk_indices(seqlens)
        if seqlens is not None and _as_bool(spec.get("explicit_chunk_indices", False))
        else None
    )
    return PreparedInputs(k, w, u, g, gk, initial_state, cu, indices, seqlens)


def run_cpu(spec: dict[str, Any], inputs: PreparedInputs, high_precision: bool = False):
    return _reference(
        inputs,
        high_precision=high_precision,
        output_final_state=_as_bool(spec.get("output_final_state", False)),
        use_exp2=_as_bool(spec.get("use_exp2", False)),
        state_v_first=_as_bool(spec.get("state_v_first", False)),
    )


def run_npu(spec: dict[str, Any], inputs: PreparedInputs):
    from fla_npu.ops.ascendc import chunk_fwd_h

    return chunk_fwd_h(
        inputs.k,
        inputs.w,
        inputs.u,
        g=inputs.g,
        gk=inputs.gk,
        initial_state=inputs.initial_state,
        output_final_state=_as_bool(spec.get("output_final_state", False)),
        chunk_size=CHUNK_SIZE,
        save_new_value=True,
        cu_seqlens=inputs.cu_seqlens,
        chunk_indices=inputs.chunk_indices,
        use_exp2=_as_bool(spec.get("use_exp2", False)),
        state_v_first=_as_bool(spec.get("state_v_first", False)),
    )


@register("executor_chunk_fwd_h")
class FunctionApi(BaseApi):
    """ATK FunctionApi using only fla_npu.ops.ascendc.chunk_fwd_h."""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.spec: Optional[dict[str, Any]] = None
        self.inputs: Optional[PreparedInputs] = None

    def init_by_input_data(self, input_data: InputDataset):
        self.spec = _case_spec(input_data, OP_NAME, self.task_result.case_config)
        self.inputs = build_inputs(self.spec, _marker_device(input_data))

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        if self.spec is None:
            self.init_by_input_data(input_data)
        assert self.spec is not None
        if self.inputs is None:
            self.inputs = build_inputs(self.spec, _marker_device(input_data))
        if self.device == "cpu":
            outputs = run_cpu(self.spec, self.inputs, bool(self.task_result.is_benchmark_task))
        elif self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(self.spec, self.inputs)
        else:
            raise RuntimeError(
                f"{OP_NAME} only supports CPU golden and NPU DUT nodes, got {self.device!r}"
            )
        return _finite_tuple(outputs)

    def export_custom_data(self, input_data: InputDataset):
        del input_data
        assert self.spec is not None
        return {
            "case_key": str(self.spec["case_key"]),
            "route": str(self.spec["route"]),
            "B": int(self.spec["B"]),
            "HK": int(self.spec["HK"]),
            "HV": int(self.spec["HV"]),
            "T": int(self.spec["T"]),
            "K": int(self.spec["K"]),
            "V": int(self.spec["V"]),
            "mode": str(self.spec["mode"]),
        }
