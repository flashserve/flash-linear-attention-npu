"""chunk_gated_delta_rule_fwd_32 ATK executor.

CPU golden is the prepare + chunk_fwd_h + chunk_fwd_o chain in ``scripts/reference.py``.
NPU DUT is ``npu_chunk_gated_delta_rule_fwd_32``. Both sides read the same seeded inputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.post_process.single_benchmark_compare import SingleBenchmarkAccuracyCompare
from atk.tasks.task_plugins_register import ACCURACY_REGISTRY

from _ascendc_common_executor import _case_spec, _finite_tuple, _marker_device
from reference import default_scale, forward_reference

# Published ATK 26.9.24 does not register this name. Compare workers import
# every module under atk.tasks.post_process, so the copy installed there is
# what the post-process process actually sees. Registering here covers a
# worker that imports this executor before the lookup.
if "mixed_tolerance_bm" not in ACCURACY_REGISTRY:
    ACCURACY_REGISTRY.register_with_key(
        "mixed_tolerance_bm",
        SingleBenchmarkAccuracyCompare,
    )


OP_NAME = "chunk_gated_delta_rule_fwd_32"


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


def build_inputs(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Quantize once on CPU, then move. CPU and NPU nodes share the seed."""

    if str(spec.get("dtype", "bf16")).lower() != "bf16":
        raise ValueError("q/k/v/beta/o are BF16 only")
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
            raise ValueError("NTD packed input requires B=1")
        if sum(seqlens) != tokens:
            raise ValueError(f"sum(seqlens)={sum(seqlens)} != T={tokens}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(spec.get("seed", 20261027)))
    q = (torch.randn(batch, k_heads, tokens, key, generator=generator) * 0.05).to(torch.bfloat16)
    k = (torch.randn(batch, k_heads, tokens, key, generator=generator) * 0.05).to(torch.bfloat16)
    v = (torch.randn(batch, v_heads, tokens, value, generator=generator) * 0.05).to(torch.bfloat16)
    g = torch.randn(batch, v_heads, tokens, generator=generator) * 0.02
    beta = torch.randn(batch, v_heads, tokens, generator=generator).to(torch.bfloat16)
    state_name = str(spec.get("state_dtype", "none")).lower()
    count = _state_count(spec, seqlens)
    state_shape = (count, v_heads, key, value)
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
        "scale": float(spec.get("scale", default_scale())),
    }


def run_cpu(spec: dict[str, Any]):
    inputs = build_inputs(spec, torch.device("cpu"))
    return forward_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["initial_state"],
        inputs["seqlens"],
        inputs["scale"],
    )


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data))
    from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd_32

    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    cu = None if inputs["seqlens"] is None else [0, *torch.tensor(inputs["seqlens"]).cumsum(0).tolist()]
    # Python fills canonical chunk_indices at block 32 when only cu_seqlens is set.
    outputs = npu_chunk_gated_delta_rule_fwd_32(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        initial_state=inputs["initial_state"],
        cu_seqlens=cu,
        scale=inputs["scale"],
    )
    torch.npu.synchronize()
    return tuple(tensor.detach().cpu() for tensor in outputs)


@register("executor_chunk_gated_delta_rule_fwd_32")
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
