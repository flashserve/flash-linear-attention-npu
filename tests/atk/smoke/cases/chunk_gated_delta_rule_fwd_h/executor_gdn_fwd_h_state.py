"""Reproduce the gate-state and joint-gate profiles from their ATK case specs."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import json
from pathlib import Path
import sys

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_reference import state_reference

OP_NAME = "chunk_gated_delta_rule_fwd_h"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


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
    chunk_size: int,
) -> Optional[tuple[int, ...]]:
    if seqlens is None:
        return None
    values: list[int] = []
    for sequence, length in enumerate(seqlens):
        for chunk in range((int(length) + chunk_size - 1) // chunk_size):
            values.extend((sequence, chunk))
    return tuple(values)


def _to_physical_state(state: torch.Tensor, state_v_first: bool) -> torch.Tensor:
    return state.transpose(-1, -2).contiguous() if state_v_first else state.contiguous()


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


@dataclass
class PreparedInputs:
    k: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    g: Optional[torch.Tensor]
    gk: Optional[torch.Tensor]
    initial_state: Optional[torch.Tensor]
    cu_seqlens: Optional[tuple[int, ...]]
    chunk_indices: Optional[tuple[int, ...]]
    seqlens: Optional[list[int]]


def gate_state_inputs(spec: dict[str, Any], device: torch.device) -> PreparedInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(spec.get("seed", 20260817)))

    batch = int(spec["B"])
    k_heads = int(spec["HK"])
    v_heads = int(spec["HV"])
    total_tokens = int(spec["T"])
    key_dim, value_dim = int(spec["K"]), int(spec["V"])
    input_dtype = DTYPES[spec["dtype"]]
    seqlens = _parse_seqlens(spec.get("seqlens"))
    if seqlens is not None and sum(seqlens) != total_tokens:
        raise ValueError(f"{OP_NAME}: seqlens sum {sum(seqlens)} does not equal T={total_tokens}")

    input_scale = float(spec.get("input_scale", 0.05))
    gate_step_scale = float(spec.get("gate_step_scale", 0.02))
    state_scale = float(spec.get("state_scale", 0.05))
    gate_dtype = DTYPES[str(spec.get("gate_dtype", "fp32"))]

    k = _random_normal(
        (batch, k_heads, total_tokens, key_dim),
        generator,
        input_dtype,
        device,
        input_scale,
    )
    w = _random_normal(
        (batch, v_heads, total_tokens, key_dim),
        generator,
        input_dtype,
        device,
        input_scale,
    )
    u_contiguous = _random_normal(
        (batch, v_heads, total_tokens, value_dim),
        generator,
        input_dtype,
        device,
        input_scale,
    )
    if _as_bool(spec.get("non_contiguous_u", False)):
        padded = torch.empty(
            (batch, v_heads, total_tokens, value_dim * 2),
            dtype=input_dtype,
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
            (batch, v_heads, total_tokens, key_dim),
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
            (sequences, v_heads, key_dim, value_dim),
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
        _canonical_chunk_indices(seqlens, int(spec["chunk_size"]))
        if seqlens is not None and _as_bool(spec.get("explicit_chunk_indices", False))
        else None
    )
    return PreparedInputs(k, w, u, g, gk, initial_state, cu, indices, seqlens)


def joint_inputs(spec):
    generator = torch.Generator().manual_seed(spec["seed"])
    tokens, heads = spec["T"], spec["HV"]
    key_dim, value_dim = spec["K"], spec["V"]
    dtype = DTYPES[spec["dtype"]]
    bounds = [0, tokens // 2, tokens] if spec["packed"] else [0, tokens]
    gates = {}
    for name, shape in [("g", (1, heads, tokens)), ("gk", (1, heads, tokens, key_dim))]:
        if spec["mode"] in (name, "both"):
            steps = torch.rand(shape, generator=generator) * (-0.008)
            segments = [steps[:, :, start:end].cumsum(2) for start, end in zip(bounds, bounds[1:])]
            gates[name] = torch.cat(segments, dim=2).to(DTYPES[spec["gate_dtype"]])

    initial = None
    if spec["with_initial"]:
        size = (len(bounds) - 1, heads, key_dim, value_dim)
        initial = (torch.randn(size, generator=generator) * 0.01).to(DTYPES[spec["state_dtype"]])
        initial = _to_physical_state(initial, spec["state_v_first"])
    key_shape = (1, heads, tokens, key_dim)
    value_shape = (1, heads, tokens, value_dim)
    k = (torch.randn(key_shape, generator=generator) * 0.04).to(dtype)
    w = (torch.randn(key_shape, generator=generator) * 0.04).to(dtype)
    u = (torch.randn(value_shape, generator=generator) * 0.05).to(dtype)
    return k, w, u, gates, initial, bounds


@register("executor_gdn_fwd_h_state")
class StateApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        cfg = self.task_result.case_config.get_input_data_config(name="case_spec")
        spec = json.loads(cfg.range_values)
        if "packed" in spec:
            k, w, u, gates, initial, bounds = joint_inputs(spec)
            cu = bounds if spec["packed"] else None
            lengths = [r - l for l, r in zip(bounds, bounds[1:])]
            indices = (
                _canonical_chunk_indices(lengths, spec["chunk_size"])
                if spec["explicit_chunk_indices"]
                else None
            )
        else:
            x = gate_state_inputs(spec, torch.device("cpu"))
            k, w, u, initial, cu, indices = (
                x.k,
                x.w,
                x.u,
                x.initial_state,
                x.cu_seqlens,
                x.chunk_indices,
            )
            gates = {"g": x.g, "gk": x.gk}
        kwargs = dict(
            gates,
            initial_state=initial,
            chunk_size=spec["chunk_size"],
            cu_seqlens=cu,
            output_final_state=spec["output_final_state"],
            state_v_first=spec["state_v_first"],
        )
        if self.device == "cpu":
            outputs = state_reference(
                k, w, u, **kwargs, high_precision=bool(self.task_result.is_benchmark_task)
            )
        else:
            from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd_h

            device = next(
                v.device for v in input_data.kwargs.values() if isinstance(v, torch.Tensor)
            )
            kwargs = {
                n: v.to(device) if isinstance(v, torch.Tensor) else v for n, v in kwargs.items()
            }
            outputs = chunk_gated_delta_rule_fwd_h(
                k.to(device), w.to(device), u.to(device), chunk_indices=indices, **kwargs
            )
        visible = tuple(
            t.float() if t.dtype == torch.float64 else t for t in outputs if t is not None
        )
        if len(visible) != 2 + int(spec["output_final_state"]) or not all(
            torch.isfinite(t).all() for t in visible
        ):
            raise ValueError("Missing or nonfinite FwdH output")
        return visible
