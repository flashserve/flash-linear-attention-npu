"""ATK executor for the recurrent_gated_delta_rule Ascend C operator."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _case_spec,
    _finite_tuple,
    _int_tensor,
    _marker_device,
    _orig_dtype,
    _rand,
    _randn,
)
from reference_recurrent_gated_delta_rule import (
    recurrent_gated_delta_rule_reference,
)


OP_NAME = "recurrent_gated_delta_rule"


def _int_list(spec: dict[str, Any], name: str) -> list[int]:
    values = spec.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{OP_NAME}: {name} must be a non-empty integer list.")
    return [int(value) for value in values]


def _build_state(
    spec: dict[str, Any],
    shape: tuple[int, int, int, int],
    dtype_name: str,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    dense_state = _randn(
        shape,
        dtype_name,
        dtype,
        device,
        seed,
        scale=0.01,
    )
    state_layout = str(spec.get("state_layout", "contiguous"))
    if state_layout == "contiguous":
        return dense_state
    block_num, value_heads, value_dim, key_dim = shape
    if state_layout == "noncontiguous":
        storage = torch.empty(
            (value_heads, block_num, value_dim, key_dim),
            dtype=dense_state.dtype,
            device=device,
        )
        state = storage.permute(1, 0, 2, 3)
        state.copy_(dense_state)
        return state

    if state_layout not in {
        "head_padded",
        "block_padded",
        "head_block_padded",
    }:
        raise ValueError(f"{OP_NAME}: unsupported state_layout {state_layout!r}.")

    head_padding = key_dim if state_layout in {"head_padded", "head_block_padded"} else 0
    block_padding = (
        value_dim * key_dim
        if state_layout in {"block_padded", "head_block_padded"}
        else 0
    )
    head_stride = value_dim * key_dim + head_padding
    block_stride = value_heads * head_stride + block_padding
    storage_offset = key_dim + 1 if state_layout == "head_block_padded" else 0
    storage_size = (
        storage_offset
        + (block_num - 1) * block_stride
        + (value_heads - 1) * head_stride
        + value_dim * key_dim
    )
    storage = torch.empty(storage_size, dtype=dense_state.dtype, device=device)
    state = torch.as_strided(
        storage,
        shape,
        (block_stride, head_stride, key_dim, 1),
        storage_offset=storage_offset,
    )
    state.copy_(dense_state)
    return state


def _make_noncontiguous(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or tensor.ndim == 0:
        return tensor
    if tensor.ndim == 3:
        # Legal Q/K/V row padding: the innermost feature axis stays contiguous.
        storage = torch.empty((*tensor.shape[:-1], tensor.shape[-1] + 16),
                              dtype=tensor.dtype, device=tensor.device)
        view = storage[..., :tensor.shape[-1]]
        view.copy_(tensor)
        return view
    storage_shape = (*tensor.shape[:-1], tensor.shape[-1] * 2)
    storage = torch.empty(storage_shape, dtype=tensor.dtype, device=tensor.device)
    view = storage[..., ::2]
    view.copy_(tensor)
    return view


def _apply_input_layout(
    inputs: dict[str, Any], layout: str
) -> dict[str, Any]:
    layout_names = {
        "contiguous": (),
        "noncontiguous_qkv_beta_g": ("query", "key", "value", "beta", "g"),
        "noncontiguous_qkv": ("query", "key", "value"),
        "noncontiguous_metadata": (
            "actual_seq_lengths",
            "ssm_state_indices",
            "num_accepted_tokens",
        ),
        "noncontiguous_qk": ("query", "key"),
        "noncontiguous_v_beta_g": ("value", "beta", "g"),
        "noncontiguous_all": (
            "query",
            "key",
            "value",
            "beta",
            "g",
            "actual_seq_lengths",
            "ssm_state_indices",
            "num_accepted_tokens",
        ),
    }
    if layout not in layout_names:
        raise ValueError(f"{OP_NAME}: unsupported input_layout {layout!r}.")
    for name in layout_names[layout]:
        inputs[name] = _make_noncontiguous(inputs[name])
    return inputs


def _add_head_offsets(
    tensor: torch.Tensor, scale: float, dtype_name: str
) -> None:
    """Add zero-centered head offsets with magnitude bounded by scale."""
    storage_dtype = _orig_dtype(dtype_name)
    quantized = tensor.to(storage_dtype)
    head_count = tensor.shape[1]
    if head_count == 1:
        offsets = torch.zeros(1, device=tensor.device, dtype=storage_dtype)
    else:
        offsets = torch.linspace(
            -scale,
            scale,
            steps=head_count,
            dtype=torch.float32,
        ).to(storage_dtype).to(tensor.device)
    quantized.add_(
        offsets.view(1, -1, *([1] * (tensor.ndim - 2)))
    )
    tensor.copy_(quantized.to(tensor.dtype))


def _quantize_input_(tensor: torch.Tensor, dtype_name: str) -> None:
    tensor.copy_(tensor.to(_orig_dtype(dtype_name)).to(tensor.dtype))


def build_inputs(
    spec: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    seed = int(spec.get("seed", 20260817))
    seq_lengths = _int_list(spec, "seq_lengths")
    prefix_tokens = int(spec.get("prefix_tokens", 0))
    total_tokens = prefix_tokens + sum(seq_lengths)
    if total_tokens != int(spec["T"]):
        raise ValueError(
            f"{OP_NAME}: T must equal prefix_tokens + sum(seq_lengths), "
            f"got T={spec['T']} and computed {total_tokens}."
        )

    key_heads = int(spec["HK"])
    value_heads = int(spec["HV"])
    key_dim = int(spec["K"])
    value_dim = int(spec["V"])
    block_num = int(spec["block_num"])
    state_dtype_name = str(spec.get("state_dtype", "bf16"))
    state_dtype = _orig_dtype(state_dtype_name)
    low_precision_dtype = torch.bfloat16
    state_calc_dtype = state_dtype
    g_dtype_name = str(spec.get("g_dtype", "fp32"))
    g_calc_dtype = _orig_dtype(g_dtype_name)
    beta_dtype_name = str(spec.get("beta_dtype", "bf16"))
    beta_dtype = _orig_dtype(beta_dtype_name)

    state_indices_spec = spec.get("state_indices", list(range(total_tokens)))
    index_rank = int(spec.get("ssm_state_indices_rank", 1))
    if index_rank == 2:
        state_width = int(spec.get("state_index_width", max(seq_lengths, default=1)))
        if isinstance(state_indices_spec, list) and state_indices_spec and isinstance(state_indices_spec[0], list):
            state_indices = [[int(value) for value in row] for row in state_indices_spec]
        else:
            flat = [int(value) for value in state_indices_spec]
            if len(flat) != len(seq_lengths) * state_width:
                raise ValueError(f"{OP_NAME}: 2-D state_indices must contain B*W entries.")
            state_indices = [flat[i * state_width:(i + 1) * state_width] for i in range(len(seq_lengths))]
        if len(state_indices) != len(seq_lengths) or any(len(row) != state_width for row in state_indices):
            raise ValueError(f"{OP_NAME}: 2-D state_indices must have shape [B, W].")
    else:
        state_indices = [int(value) for value in state_indices_spec]
        if len(state_indices) != total_tokens:
            raise ValueError(
                f"{OP_NAME}: state_indices must contain T entries, "
                f"got {len(state_indices)} for T={total_tokens}."
            )

    gate_mode = str(spec.get("gate_mode", "g"))
    if gate_mode not in {"none", "g"}:
        raise ValueError(f"{OP_NAME}: unsupported gate_mode {gate_mode!r}.")

    gate_profile = str(spec.get("gate_profile", "random"))
    gate_min, gate_max = 0.001, 0.02
    if gate_profile == "near_zero":
        gate_min, gate_max = 0.000001, 0.0001
    elif gate_profile == "strong_decay":
        gate_min, gate_max = 0.5, 2.0

    g = None
    if gate_mode == "g":
        g = -_rand(
            (total_tokens, value_heads),
            g_dtype_name,
            g_calc_dtype,
            device,
            seed + 6,
            gate_min,
            gate_max,
        )
        if gate_profile == "per_head":
            values = torch.linspace(
                -0.001,
                -0.02,
                value_heads,
                dtype=torch.float32,
                device=device,
            ).to(g_calc_dtype)
            g.copy_(values.view(1, -1).expand_as(g))
            _quantize_input_(g, g_dtype_name)

    accepted_tokens = spec.get("accepted_tokens")
    if accepted_tokens is not None:
        accepted_tokens = [int(value) for value in accepted_tokens]
        if len(accepted_tokens) != len(seq_lengths):
            raise ValueError(
                f"{OP_NAME}: accepted_tokens must contain B entries, "
                f"got {len(accepted_tokens)} for B={len(seq_lengths)}."
            )

    inputs = {
        "query": _randn(
            (total_tokens, key_heads, key_dim),
            "bf16",
            low_precision_dtype,
            device,
            seed + 1,
        ),
        "key": _randn(
            (total_tokens, key_heads, key_dim),
            "bf16",
            low_precision_dtype,
            device,
            seed + 2,
        ),
        "value": _randn(
            (total_tokens, value_heads, value_dim),
            "bf16",
            low_precision_dtype,
            device,
            seed + 3,
        ),
        "state": _build_state(
            spec,
            (block_num, value_heads, value_dim, key_dim),
            state_dtype_name,
            state_calc_dtype,
            device,
            seed + 4,
        ),
        "beta": _rand(
            (total_tokens, value_heads),
            beta_dtype_name,
            beta_dtype,
            device,
            seed + 5,
            0.1,
            0.9,
        ),
        "g": g,
        "gk": None,
        "actual_seq_lengths": _int_tensor(
            [prefix_tokens, *seq_lengths], device, torch.int32
        ),
        "ssm_state_indices": _int_tensor(state_indices, device, torch.int32),
        "num_accepted_tokens": (
            None
            if accepted_tokens is None
            else _int_tensor(accepted_tokens, device, torch.int32)
        ),
        "scale": float(spec.get("scale", 1.0 / math.sqrt(key_dim))),
        "A_log": None,
        "dt_bias": None,
        "use_qk_l2norm_in_kernel": bool(spec.get("use_qk_l2norm_in_kernel", False)),
        "use_gate_in_kernel": bool(spec.get("use_gate_in_kernel", False)),
        "use_beta_sigmoid_in_kernel": bool(spec.get("use_beta_sigmoid_in_kernel", False)),
        "allow_neg_eigval": bool(spec.get("allow_neg_eigval", False)),
    }

    if inputs["use_gate_in_kernel"]:
        inputs["A_log"] = _randn((value_heads,), "fp32", torch.float32, device, seed + 8, scale=0.1)
        if bool(spec.get("with_dt_bias", True)):
            inputs["dt_bias"] = _randn((value_heads,), "fp32", torch.float32, device, seed + 9, scale=0.1)

    if str(spec.get("data_profile", "random")) == "traceable_gva":
        _add_head_offsets(inputs["query"], 0.01, "bf16")
        _add_head_offsets(inputs["key"], 0.02, "bf16")
        _add_head_offsets(inputs["value"], 0.005, "bf16")

    if str(spec.get("beta_profile", "random")) == "per_head":
        values = torch.linspace(
            0.1,
            0.9,
            value_heads,
            dtype=torch.float32,
            device=device,
        ).to(beta_dtype).to(inputs["beta"].dtype)
        inputs["beta"].copy_(values.view(1, -1).expand_as(inputs["beta"]))
        _quantize_input_(inputs["beta"], beta_dtype_name)

    state_profile = str(spec.get("state_profile", "random"))
    if state_profile.startswith("pulse_hv"):
        pulse_head = int(state_profile.removeprefix("pulse_hv"))
        inputs["state"].zero_()
        inputs["state"][:, pulse_head, 0, 0] = 0.25
    elif state_profile == "traceable":
        _add_head_offsets(inputs["state"], 0.0001, state_dtype_name)
    if state_profile != "random":
        _quantize_input_(inputs["state"], state_dtype_name)

    if spec.get("arch22"):
        inputs["beta"] = inputs["beta"].to(_orig_dtype(spec["beta_dtype"]))
        if inputs["g"] is not None:
            inputs["g"] = inputs["g"].to(_orig_dtype(spec["g_dtype"]))
        if spec.get("use_gate_in_kernel"):
            inputs["A_log"] = _randn((value_heads,), "fp32", torch.float32, device, seed + 8, scale=0.5)
            inputs["dt_bias"] = _randn((value_heads,), "fp32", torch.float32, device, seed + 9, scale=0.2)
    return _apply_input_layout(
        inputs, str(spec.get("input_layout", "contiguous"))
    )


def run_cpu(spec: dict[str, Any]):
    previous_threads = torch.get_num_threads()
    try:
        # ATK runs several CPU workers concurrently. One Torch thread per worker
        # avoids severe nested-parallelism overhead for the per-head reference.
        torch.set_num_threads(1)
        inputs = build_inputs(spec, torch.device("cpu"))
        outputs = None
        repeat_calls = int(spec.get("repeat_calls", 1))
        for call_index in range(repeat_calls):
            outputs = recurrent_gated_delta_rule_reference(
                query=inputs["query"],
                key=inputs["key"],
                value=inputs["value"],
                state=inputs["state"],
                beta=inputs["beta"],
                scale=inputs["scale"],
                actual_seq_lengths=inputs["actual_seq_lengths"],
                ssm_state_indices=inputs["ssm_state_indices"],
                num_accepted_tokens=inputs["num_accepted_tokens"],
                g=inputs["g"],
                gk=inputs["gk"],
                use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
                use_gate_in_kernel=inputs["use_gate_in_kernel"],
                A_log=inputs["A_log"],
                dt_bias=inputs["dt_bias"],
                use_beta_sigmoid_in_kernel=inputs["use_beta_sigmoid_in_kernel"],
                allow_neg_eigval=inputs["allow_neg_eigval"],
            )
            if call_index + 1 < repeat_calls:
                inputs["state"].copy_(outputs[1])
        if outputs is None:
            raise RuntimeError(f"{OP_NAME}: repeat_calls must be positive.")
        prefix_tokens = int(inputs["actual_seq_lengths"][0].item())
        return outputs[0][prefix_tokens:], outputs[1]
    finally:
        torch.set_num_threads(previous_threads)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data))
    from fla_npu.ops import ascendc

    out = None
    for _ in range(int(spec.get("repeat_calls", 1))):
        out = ascendc.recurrent_gated_delta_rule(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["state"],
            beta=inputs["beta"],
            scale=inputs["scale"],
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"],
            num_accepted_tokens=inputs["num_accepted_tokens"],
            g=inputs["g"],
            gk=inputs["gk"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
            use_gate_in_kernel=inputs["use_gate_in_kernel"],
            use_beta_sigmoid_in_kernel=inputs["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=inputs["allow_neg_eigval"],
        )
    if out is None:
        raise RuntimeError(f"{OP_NAME}: repeat_calls must be positive.")
    prefix_tokens = int(inputs["actual_seq_lengths"][0].item())
    return out[prefix_tokens:], inputs["state"]


@register("executor_recurrent_gated_delta_rule")
class FunctionApi(BaseApi):
    """ATK execution entry point."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec)
        else:
            raise RuntimeError(
                f"{OP_NAME} requires an NPU DUT node and a CPU reference node, "
                f"got {self.device!r}."
            )
        return _finite_tuple(outputs, golden=self.device == "cpu")

    def export_custom_data(self, input_data: InputDataset):
        spec = _case_spec(input_data, OP_NAME)
        return {
            "design_id": str(spec["design_id"]),
            "case_name": str(spec["name"]),
            "B": int(spec["B"]),
            "T": int(spec["T"]),
            "HK": int(spec["HK"]),
            "HV": int(spec["HV"]),
            "K": int(spec["K"]),
            "V": int(spec["V"]),
            "state_dtype": str(spec["state_dtype"]),
            "state_layout": str(spec.get("state_layout", "contiguous")),
            "input_layout": str(spec.get("input_layout", "contiguous")),
            "gate_mode": str(spec["gate_mode"]),
        }
