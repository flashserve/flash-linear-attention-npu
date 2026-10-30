"""Migrated chunk32 path of chunk_gated_delta_rule_fwd.

CPU golden is ``cpu_golden_fwd_emu.cpu_chunk_gated_delta_rule_fwd_emu``, the
step-by-step GPU emulation. NPU DUT is
``npu_chunk_gated_delta_rule_fwd(..., chunk_size=32)``. Both sides read the
same seeded inputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _case_spec, _finite_tuple, _marker_device
from cpu_golden_fwd_emu import cpu_chunk_gated_delta_rule_fwd_emu

OP_NAME = "chunk_gated_delta_rule_fwd_chunk32"


def _seqlens(spec: dict[str, Any]) -> Optional[list[int]]:
    raw = spec.get("seqlens")
    text = "" if raw is None else str(raw).strip()
    if not text:
        return None
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"seqlens must be positive lengths, got {values}")
    return values


def _state_count(spec: dict[str, Any], seqlens: Optional[list[int]]) -> int:
    return len(seqlens) if seqlens is not None else int(spec["B"])


def _flags(spec: dict[str, Any]) -> tuple[str, bool, bool, bool]:
    layout = str(spec.get("layout", "BNSD")).upper()
    if layout not in {"BNSD", "BSND", "NTD", "TND"}:
        raise ValueError(f"layout must be BNSD, BSND, NTD or TND, got {layout}")
    sequence_major = layout in {"BSND", "TND"}
    allow_neg = bool(spec.get("allow_neg_eigval", False))
    state_v_first = bool(spec.get("state_v_first", False))
    return layout, sequence_major, allow_neg, state_v_first


def build_inputs(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Quantize once on CPU, then move. CPU and NPU nodes share the seed."""

    if str(spec.get("dtype", "bf16")).lower() != "bf16":
        raise ValueError("q/k/v/beta/o are BF16 only")
    layout, sequence_major, allow_neg, state_v_first = _flags(spec)
    batch = int(spec["B"])
    k_heads = int(spec["HK"])
    v_heads = int(spec["HV"])
    tokens = int(spec["T"])
    key = int(spec["K"])
    value = int(spec["V"])
    if key != 128 or value != 128:
        raise ValueError(f"K and V must be 128, got K={key} V={value}")
    if k_heads <= 0 or v_heads % k_heads != 0:
        raise ValueError(f"Hv must be a positive multiple of Hk, got Hk={k_heads} Hv={v_heads}")
    seqlens = _seqlens(spec)
    if seqlens is None:
        if batch < 1 or tokens < 1:
            raise ValueError(f"B and T must be positive, got B={batch} T={tokens}")
    else:
        if batch != 1:
            raise ValueError("packed input requires B=1")
        if sum(seqlens) != tokens:
            raise ValueError(f"sum(seqlens)={sum(seqlens)} != T={tokens}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(spec.get("seed", 20261027)))
    if sequence_major:
        q = (torch.randn(batch, tokens, k_heads, key, generator=generator) * 0.05).to(torch.bfloat16)
        k = (torch.randn(batch, tokens, k_heads, key, generator=generator) * 0.05).to(torch.bfloat16)
        v = (torch.randn(batch, tokens, v_heads, value, generator=generator) * 0.05).to(torch.bfloat16)
        g = torch.randn(batch, tokens, v_heads, generator=generator) * 0.02
        beta = torch.randn(batch, tokens, v_heads, generator=generator).to(torch.bfloat16)
    else:
        q = (torch.randn(batch, k_heads, tokens, key, generator=generator) * 0.05).to(torch.bfloat16)
        k = (torch.randn(batch, k_heads, tokens, key, generator=generator) * 0.05).to(torch.bfloat16)
        v = (torch.randn(batch, v_heads, tokens, value, generator=generator) * 0.05).to(torch.bfloat16)
        g = torch.randn(batch, v_heads, tokens, generator=generator) * 0.02
        beta = torch.randn(batch, v_heads, tokens, generator=generator).to(torch.bfloat16)
    state_name = str(spec.get("state_dtype", "none")).lower()
    count = _state_count(spec, seqlens)
    state_shape = (count, v_heads, value, key) if state_v_first else (count, v_heads, key, value)
    if state_name in {"", "none"}:
        initial = None
    elif state_name == "bf16":
        initial = (torch.randn(state_shape, generator=generator) * 0.05).to(torch.bfloat16)
    elif state_name == "fp32":
        initial = torch.randn(state_shape, generator=generator) * 0.05
    elif state_name == "zero":
        initial = torch.zeros(state_shape, dtype=torch.float32)
    else:
        raise ValueError(f"state_dtype must be none/bf16/fp32/zero, got {state_name!r}")
    def move(tensor):
        if tensor is None or device.type == "cpu":
            return tensor
        return tensor.to(device)

    return {
        "q": move(q),
        "k": move(k),
        "v": move(v),
        "g": move(g),
        "beta": move(beta),
        "initial_state": move(initial),
        "seqlens": seqlens,
        "scale": float(spec.get("scale", 128 ** -0.5)),
        "layout": layout,
        "sequence_major": sequence_major,
        "allow_neg_eigval": allow_neg,
        "state_v_first": state_v_first,
    }


def _cu(seqlens: Optional[list[int]]) -> Optional[torch.Tensor]:
    if seqlens is None:
        return None
    values = [0]
    for length in seqlens:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int64)


def run_cpu(spec: dict[str, Any]):
    inputs = build_inputs(spec, torch.device("cpu"))
    # Public NTD/TND are rank-4 packed tensors, not the emulator's rank-3 names.
    golden_layout = "bsnd" if inputs["sequence_major"] else "bnsd"
    o, final_state = cpu_chunk_gated_delta_rule_fwd_emu(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        scale=inputs["scale"],
        initial_state=inputs["initial_state"],
        cu_seqlens=_cu(inputs["seqlens"]),
        chunk_size=32,
        layout=golden_layout,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=inputs["allow_neg_eigval"],
        state_v_first=inputs["state_v_first"],
    )
    if not inputs["sequence_major"]:
        o = o.transpose(1, 2).contiguous()
    return o, final_state


def _metadata(seqlens: Optional[list[int]]) -> tuple[Optional[list[int]], Optional[list[int]]]:
    if seqlens is None:
        return None, None
    cu = [0]
    chunks = []
    for sequence, length in enumerate(seqlens):
        cu.append(cu[-1] + int(length))
        for local_chunk in range((int(length) + 31) // 32):
            chunks.extend((sequence, local_chunk))
    return cu, chunks


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data))
    from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd

    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    cu, chunks = _metadata(inputs["seqlens"])
    if inputs["sequence_major"]:
        g, beta = inputs["g"], inputs["beta"]
    else:
        g = inputs["g"].transpose(1, 2).contiguous()
        beta = inputs["beta"].transpose(1, 2).contiguous()
    outputs = npu_chunk_gated_delta_rule_fwd(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g,
        beta,
        initial_state=inputs["initial_state"],
        output_final_state=True,
        chunk_size=32,
        cu_seqlens=cu,
        chunk_indices=chunks,
        scale=inputs["scale"],
        use_exp2=False,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=inputs["allow_neg_eigval"],
        disable_recompute=False,
        return_intermediate_states=False,
        state_v_first=inputs["state_v_first"],
        layout=inputs["layout"],
    )
    torch.npu.synchronize()
    return outputs[0].detach().cpu(), outputs[1].detach().cpu()


@register("executor_chunk_gated_delta_rule_fwd_chunk32")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(getattr(task_result, "is_benchmark_task", False))

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec)
        else:
            raise RuntimeError(
                f"{OP_NAME} only runs an NPU DUT and a CPU golden, got device {self.device!r}"
            )
        return _finite_tuple(outputs)
