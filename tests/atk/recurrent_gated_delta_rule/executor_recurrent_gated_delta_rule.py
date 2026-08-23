"""ATK executor for the recurrent_gated_delta_rule Ascend C operator."""

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
    _case_spec,
    _finite_tuple,
    _int_tensor,
    _marker_device,
    _orig_dtype,
    _rand,
    _randn,
)


OP_NAME = "recurrent_gated_delta_rule"


def _int_list(spec: dict[str, Any], name: str) -> list[int]:
    values = spec.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{OP_NAME}: {name} must be a non-empty integer list.")
    return [int(value) for value in values]


def build_inputs(
    spec: dict[str, Any],
    device: torch.device,
    high_precision: bool = False,
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
    low_precision_dtype = torch.float64 if high_precision else torch.bfloat16
    state_calc_dtype = torch.float64 if high_precision else state_dtype
    gate_calc_dtype = torch.float64 if high_precision else torch.float32

    state_indices = [
        int(value)
        for value in spec.get("state_indices", list(range(total_tokens)))
    ]
    if len(state_indices) != total_tokens:
        raise ValueError(
            f"{OP_NAME}: state_indices must contain T entries, "
            f"got {len(state_indices)} for T={total_tokens}."
        )

    gate_mode = str(spec.get("gate_mode", "g"))
    if gate_mode not in {"g", "gk", "both"}:
        raise ValueError(f"{OP_NAME}: unsupported gate_mode {gate_mode!r}.")

    g = None
    if gate_mode in {"g", "both"}:
        g = -_rand(
            (total_tokens, value_heads),
            "fp32",
            gate_calc_dtype,
            device,
            seed + 6,
            0.001,
            0.02,
        )

    gk = None
    if gate_mode in {"gk", "both"}:
        gk = -_rand(
            (total_tokens, value_heads, key_dim),
            "fp32",
            gate_calc_dtype,
            device,
            seed + 7,
            0.001,
            0.02,
        )

    accepted_tokens = spec.get("accepted_tokens")
    if accepted_tokens is not None:
        accepted_tokens = [int(value) for value in accepted_tokens]
        if len(accepted_tokens) != len(seq_lengths):
            raise ValueError(
                f"{OP_NAME}: accepted_tokens must contain B entries, "
                f"got {len(accepted_tokens)} for B={len(seq_lengths)}."
            )

    return {
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
        "state": _randn(
            (block_num, value_heads, value_dim, key_dim),
            state_dtype_name,
            state_calc_dtype,
            device,
            seed + 4,
            scale=0.01,
        ),
        "beta": _rand(
            (total_tokens, value_heads),
            "bf16",
            low_precision_dtype,
            device,
            seed + 5,
            0.1,
            0.9,
        ),
        "g": g,
        "gk": gk,
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
    }


def recurrent_gated_delta_rule_reference(inputs: dict[str, Any]):
    query = inputs["query"]
    key = inputs["key"]
    value = inputs["value"]
    beta = inputs["beta"]
    g = inputs["g"]
    gk = inputs["gk"]
    actual_seq_lengths = inputs["actual_seq_lengths"].detach().cpu().tolist()
    state_indices = inputs["ssm_state_indices"].detach().cpu().tolist()
    accepted_tokens = inputs["num_accepted_tokens"]
    if accepted_tokens is not None:
        accepted_tokens = accepted_tokens.detach().cpu().tolist()

    calc_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    final_state = inputs["state"].to(calc_dtype).clone()
    total_tokens, key_heads, _ = query.shape
    value_heads, value_dim = value.shape[1:]
    head_group = value_heads // key_heads
    out = torch.zeros(
        (total_tokens, value_heads, value_dim),
        dtype=calc_dtype,
        device=query.device,
    )

    seq_start = int(actual_seq_lengths[0])
    for batch_index, seq_len_value in enumerate(actual_seq_lengths[1:]):
        seq_len = int(seq_len_value)
        if seq_len <= 0:
            continue
        seq_end = seq_start + seq_len
        state_token_index = seq_start
        if accepted_tokens is not None:
            state_token_index += int(accepted_tokens[batch_index]) - 1
        initial_state_slot = int(state_indices[state_token_index])

        for value_head in range(value_heads):
            key_head = value_head // head_group
            recurrent_state = final_state[initial_state_slot, value_head].clone()
            for token_index in range(seq_start, seq_end):
                if g is not None:
                    recurrent_state *= torch.exp(g[token_index, value_head].to(calc_dtype))
                if gk is not None:
                    recurrent_state *= torch.exp(
                        gk[token_index, value_head].to(calc_dtype)
                    ).unsqueeze(0)

                key_vector = key[token_index, key_head].to(calc_dtype)
                delta = value[token_index, value_head].to(calc_dtype)
                delta -= torch.matmul(recurrent_state, key_vector)
                delta *= beta[token_index, value_head].to(calc_dtype)
                recurrent_state += torch.outer(delta, key_vector)
                scaled_query = (
                    query[token_index, key_head].to(calc_dtype)
                    * float(inputs["scale"])
                )
                out[token_index, value_head] = torch.matmul(
                    recurrent_state, scaled_query
                )
                state_slot = int(state_indices[token_index])
                final_state[state_slot, value_head] = recurrent_state

        seq_start = seq_end

    return out.to(value.dtype), final_state.to(inputs["state"].dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return recurrent_gated_delta_rule_reference(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

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
    )
    prefix_tokens = int(inputs["actual_seq_lengths"][0].item())
    if prefix_tokens > 0:
        out[:prefix_tokens].zero_()
    return out, inputs["state"]


@register("executor_recurrent_gated_delta_rule")
class FunctionApi(BaseApi):
    """ATK execution entry point."""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(
                f"{OP_NAME} requires an NPU DUT node and a CPU reference node, "
                f"got {self.device!r}."
            )
        return _finite_tuple(outputs)

    def export_custom_data(self, input_data: InputDataset):
        spec = _case_spec(input_data, OP_NAME)
        return {
            "case_name": str(spec["name"]),
            "B": int(spec["B"]),
            "T": int(spec["T"]),
            "HK": int(spec["HK"]),
            "HV": int(spec["HV"]),
            "K": int(spec["K"]),
            "V": int(spec["V"]),
            "state_dtype": str(spec["state_dtype"]),
            "gate_mode": str(spec["gate_mode"]),
        }
