"""Migrated chunk32 path of chunk_gated_delta_rule_fwd.

Mixed tolerance compares the NPU DUT with the CPU emulation in
``cpu_golden_fwd_emu``. Double benchmark keeps that CPU emulation as the
golden and adds the GPU Triton forward in ``gpu_triton_fwd`` as the
same-precision benchmark. Both modes rebuild inputs from the case seed.
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
from cpu_golden import RCP_LN2
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


def _flags(spec: dict[str, Any]) -> tuple[str, bool, bool, bool, bool, bool, bool]:
    layout = str(spec.get("layout", "BNSD")).upper()
    if layout not in {"BNSD", "BSND", "NTD", "TND"}:
        raise ValueError(f"layout must be BNSD, BSND, NTD or TND, got {layout}")
    sequence_major = layout in {"BSND", "TND"}
    allow_neg = bool(spec.get("allow_neg_eigval", False))
    state_v_first = bool(spec.get("state_v_first", False))
    out_l2 = bool(spec.get("out_l2", False))
    disable_recompute = bool(spec.get("disable_recompute", False))
    out_h = bool(spec.get("out_h", False))
    return layout, sequence_major, allow_neg, state_v_first, out_l2, disable_recompute, out_h


def build_inputs(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Quantize once on CPU, then move. CPU and NPU nodes share the seed."""

    if str(spec.get("dtype", "bf16")).lower() != "bf16":
        raise ValueError("q/k/v/beta/o are BF16 only")
    layout, sequence_major, allow_neg, state_v_first, out_l2, disable_recompute, out_h = _flags(spec)
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
        "out_l2": out_l2,
        "disable_recompute": disable_recompute,
        "out_h": out_h,
    }


def _cu(seqlens: Optional[list[int]]) -> Optional[torch.Tensor]:
    if seqlens is None:
        return None
    values = [0]
    for length in seqlens:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int64)


def _cpu(tensor: Optional[torch.Tensor], dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.detach().to(device="cpu").contiguous()


def _state_dtype(spec: dict[str, Any]) -> torch.dtype:
    state_name = str(spec.get("state_dtype", "none")).lower()
    return torch.bfloat16 if state_name == "bf16" else torch.float32


def _l2_public(q: torch.Tensor, k: torch.Tensor, layout: str):
    """Hats stay in the input layout. rstd is always ``[B, Hk, T]``."""

    def one(x: torch.Tensor):
        values = x.float()
        rstd = torch.rsqrt(values.square().sum(dim=-1) + 1e-6)
        hat = (values * rstd.unsqueeze(-1)).to(torch.bfloat16)
        return hat, rstd

    q_hat, q_rstd = one(q)
    k_hat, k_rstd = one(k)
    if layout in {"BSND", "TND"}:
        q_rstd = q_rstd.permute(0, 2, 1)
        k_rstd = k_rstd.permute(0, 2, 1)
    return q_hat.contiguous(), k_hat.contiguous(), q_rstd.contiguous(), k_rstd.contiguous()


def _beta_public(beta: torch.Tensor, layout: str, allow_neg: bool) -> torch.Tensor:
    beta_bth = beta.transpose(1, 2) if layout in {"BNSD", "NTD"} else beta
    scale = 2.0 if allow_neg else 1.0
    return (scale * torch.sigmoid(beta_bth.float())).contiguous()


def _pack(spec: dict[str, Any], o, state, g_cumsum=None, a=None, beta_eff=None,
          h=None, q_hat=None, k_hat=None, q_rstd=None, k_rstd=None):
    """Same slot order as the ten-tuple, with disabled exports left as None."""

    out_l2 = bool(spec.get("out_l2", False))
    export_ga = bool(spec.get("disable_recompute", False))
    out_h = bool(spec.get("out_h", False))
    state_dtype = _state_dtype(spec)
    return (
        _cpu(o, torch.bfloat16),
        _cpu(state, state_dtype),
        _cpu(g_cumsum, torch.float32) if export_ga else None,
        _cpu(a, torch.bfloat16) if export_ga else None,
        _cpu(beta_eff, torch.float32) if out_l2 else None,
        _cpu(h, torch.bfloat16) if out_h else None,
        _cpu(q_hat, torch.bfloat16) if out_l2 else None,
        _cpu(k_hat, torch.bfloat16) if out_l2 else None,
        _cpu(q_rstd, torch.float32) if out_l2 else None,
        _cpu(k_rstd, torch.float32) if out_l2 else None,
    )


def _exports_from_stages(inputs: dict[str, Any], stages: dict[str, torch.Tensor]):
    """Public optional tensors from the Triton-equivalent emulation.

    ``g_cumsum`` is the natural chunk-local sum. The Triton recurrence stores
    that sum times ``RCP_LN2`` and then applies ``exp2``; the kernel writes the
    sum itself and applies ``exp``, which is the same value.     ``h`` uses the same ``[V, K]`` layout as ``final_state`` when
    ``state_v_first`` is set.
    """

    g_cumsum = (stages["g_cumsum"].float() / RCP_LN2).contiguous()
    a = stages["A"].permute(0, 2, 1, 3).contiguous()
    # The emulation already stores h as [V, K] when state_v_first, matching the kernel.
    h = stages["h"]
    q_hat, k_hat, q_rstd, k_rstd = _l2_public(inputs["q"], inputs["k"], inputs["layout"])
    beta_eff = _beta_public(inputs["beta"], inputs["layout"], inputs["allow_neg_eigval"])
    return g_cumsum, a, beta_eff, h.contiguous(), q_hat, k_hat, q_rstd, k_rstd


def run_cpu(spec: dict[str, Any]):
    inputs = build_inputs(spec, torch.device("cpu"))
    # Public NTD/TND are rank-4 packed tensors, not the emulator's rank-3 names.
    golden_layout = "bsnd" if inputs["sequence_major"] else "bnsd"
    o, final_state, stages = cpu_chunk_gated_delta_rule_fwd_emu(
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
        return_stages=True,
    )
    if not inputs["sequence_major"]:
        o = o.transpose(1, 2).contiguous()
    g_cumsum, a, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd = _exports_from_stages(inputs, stages)
    return _pack(
        spec, o, final_state, g_cumsum, a, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd,
    )


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
        disable_recompute=inputs["disable_recompute"],
        return_intermediate_states=inputs["out_h"],
        state_v_first=inputs["state_v_first"],
        layout=inputs["layout"],
        is_out_l2norm=inputs["out_l2"],
    )
    torch.npu.synchronize()
    return _pack(spec, *outputs)


def run_gpu(spec: dict[str, Any]):
    """Same-precision benchmark: FLA Triton chunk forward at chunk size 32."""

    if not torch.cuda.is_available():
        raise RuntimeError("GPU Triton benchmark requires CUDA")
    from gpu_triton_fwd import gpu_chunk_gated_delta_rule_fwd_stages

    inputs = build_inputs(spec, torch.device("cuda"))
    # Public NTD/TND are rank-4 packed tensors, same as the CPU emulation.
    golden_layout = "bsnd" if inputs["sequence_major"] else "bnsd"
    cu = _cu(inputs["seqlens"])
    stages = gpu_chunk_gated_delta_rule_fwd_stages(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        scale=inputs["scale"],
        initial_state=inputs["initial_state"],
        cu_seqlens=None if cu is None else cu.to(device="cuda"),
        chunk_size=32,
        layout=golden_layout,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=inputs["allow_neg_eigval"],
        state_v_first=inputs["state_v_first"],
    )
    torch.cuda.synchronize()
    output = stages["o"]
    if not inputs["sequence_major"]:
        output = output.transpose(1, 2).contiguous()
    # The stage dict is on CUDA. The public conversion below runs on CPU.
    stages = {key: value.detach().to("cpu") for key, value in stages.items()}
    inputs = {key: (value.detach().to("cpu") if torch.is_tensor(value) else value) for key, value in inputs.items()}
    g_cumsum, a, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd = _exports_from_stages(inputs, stages)
    return _pack(
        spec, output, stages["final_state"], g_cumsum, a, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd,
    )


@register("executor_chunk_gated_delta_rule_fwd_chunk32")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(getattr(task_result, "is_benchmark_task", False))

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        case_id = spec.get("case_id", "?")
        if self.device in {"npu", "pyaclnn"}:
            print(
                f"[chunk32-atk] case={case_id} role=dut target=npu_chunk32",
                flush=True,
            )
            outputs = run_npu(spec, input_data)
        elif self.device == "gpu":
            print(
                f"[chunk32-atk] case={case_id} role=benchmark target=gpu_triton_chunk32",
                flush=True,
            )
            outputs = run_gpu(spec)
        elif self.device == "cpu":
            role = "golden" if self.is_benchmark_task else "mixed"
            print(
                f"[chunk32-atk] case={case_id} role={role} target=cpu_emu_chunk32",
                flush=True,
            )
            outputs = run_cpu(spec)
        else:
            raise RuntimeError(
                f"{OP_NAME} supports NPU DUT, CPU emulation and GPU Triton, got device {self.device!r}"
            )
        return _finite_tuple(outputs)
