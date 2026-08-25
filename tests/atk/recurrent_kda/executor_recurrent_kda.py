"""recurrent_kda 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _case_spec,
    _finite_tuple,
    _int_tensor,
    _kda_gate,
    _marker_device,
    _rand,
    _randn,
    _zeros,
)


OP_NAME = "recurrent_kda"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    calc_dtype = torch.float64 if high_precision else torch.bfloat16
    seed = int(spec.get("seed", 20260817))
    B, T, H, HV, K, V = (int(spec[x]) for x in ("B", "T", "H", "HV", "K", "V"))
    return {
        "q": _randn((B, T, H, K), "bf16", calc_dtype, device, seed + 1),
        "k": _randn((B, T, H, K), "bf16", calc_dtype, device, seed + 2),
        "v": _randn((B, T, HV, V), "bf16", calc_dtype, device, seed + 3),
        "g": _kda_gate((B, T, HV, K), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 4),
        "beta": _rand((B, T, HV), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 5, 0.1, 0.9),
        "initial_state": _zeros((B, HV, V, K), "fp32", torch.float64 if high_precision else torch.float32, device),
        "cu_seqlens": _int_tensor([i * T for i in range(B + 1)], device, torch.int64),
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
        "layout": str(spec.get("layout", "BSND")),
    }


# CPU reference copied from the operator PTA reference so this executor is self-contained.
def _flatten_bsnd(x: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return x
    if layout != "BSND":
        raise ValueError("layout must be BSND or TND")
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def _restore_layout(x: torch.Tensor, ref: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return x
    return x.reshape(ref.shape)


def _seq_ranges(total_tokens: int, cu_seqlens: Sequence[int]):
    if len(cu_seqlens) < 2:
        raise ValueError("cu_seqlens must contain at least two cumulative offsets")
    offsets = [int(offset) for offset in cu_seqlens]
    if offsets[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if any(end < start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    if offsets[-1] != total_tokens:
        raise ValueError("the last cu_seqlens offset must equal the packed token count")
    return list(zip(offsets, offsets[1:]))


def _state_slot(ssm_state_indices: torch.Tensor, seq_idx: int, start: int, token: int) -> int:
    if ssm_state_indices.ndim == 1:
        return int(ssm_state_indices[token].item())
    if ssm_state_indices.ndim == 2:
        return int(ssm_state_indices[seq_idx, token - start].item())
    raise ValueError("ssm_state_indices must be packed [T] or speculative [seq_num,max_step]")


def recurrent_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    *,
    cu_seqlens: Optional[Sequence[int]] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    layout: str = "BSND",
    scale: Optional[float] = None,
    output_final_state: bool = True,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
    state_v_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del output_final_state, inplace_final_state

    q_flat = _flatten_bsnd(q, layout).float()
    k_flat = _flatten_bsnd(k, layout).float()
    v_flat = _flatten_bsnd(v, layout).float()
    g_flat = _flatten_bsnd(g, layout).float()
    beta_flat = _flatten_bsnd(beta, layout).float()
    total_tokens, h, dk = q_flat.shape
    _, hv, dv = v_flat.shape
    scale = (dk ** -0.5) if scale is None else scale

    if use_qk_l2norm_in_kernel:
        q_flat = F.normalize(q_flat, p=2, dim=-1)
        k_flat = F.normalize(k_flat, p=2, dim=-1)
    q_flat = q_flat * scale

    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True")
        gate = g_flat
        if dt_bias is not None:
            gate = gate + dt_bias.float().reshape(hv, dk).unsqueeze(0)
        exp_a = torch.exp(A_log.float()).reshape(1, hv, 1)
        if safe_gate:
            gate = lower_bound * torch.sigmoid(exp_a * gate)
        else:
            gate = -exp_a * F.softplus(gate)
    else:
        gate = g_flat
    gate_decay = torch.exp(gate.float())

    beta_eff = beta_flat
    if use_beta_sigmoid_in_kernel:
        beta_eff = torch.sigmoid(beta_eff)
        if allow_neg_eigval:
            beta_eff = beta_eff * 2.0

    if cu_seqlens is None:
        if layout.upper() == "BSND":
            dense_seq_len = q.shape[1]
            cu_seqlens = [i * dense_seq_len for i in range(q.shape[0] + 1)]
        else:
            cu_seqlens = [0, total_tokens]
    ranges = _seq_ranges(total_tokens, cu_seqlens)
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    if initial_state is None:
        state = torch.zeros((len(ranges), hv, dv, dk), dtype=torch.float32, device=q.device)
    else:
        state = initial_state.float().clone()
        if not state_v_first:
            state = state.transpose(-1, -2).contiguous()
    out_flat = torch.zeros_like(v_flat, dtype=torch.float32)

    for seq_idx, (start, end) in enumerate(ranges):
        if start == end:
            continue
        state_slot = seq_idx
        if ssm_state_indices is not None:
            token = start
            if num_accepted_tokens is not None:
                token = start + int(num_accepted_tokens[seq_idx].item()) - 1
            state_slot = _state_slot(ssm_state_indices, seq_idx, start, token)
        for hv_idx in range(hv):
            h_idx = hv_idx // (hv // h)
            state_cur = state[state_slot, hv_idx].clone()
            for token in range(start, end):
                state_cur = state_cur * gate_decay[token, hv_idx].unsqueeze(0)
                delta = v_flat[token, hv_idx] - torch.mv(state_cur, k_flat[token, h_idx])
                delta = delta * beta_eff[token, hv_idx]
                state_cur = state_cur + torch.outer(delta, k_flat[token, h_idx])
                out_flat[token, hv_idx] = torch.mv(state_cur, q_flat[token, h_idx])
                out_slot = _state_slot(ssm_state_indices, seq_idx, start, token) if ssm_state_indices is not None else seq_idx
                state[out_slot, hv_idx] = state_cur

    if not state_v_first:
        state = state.transpose(-1, -2).contiguous()
    return _restore_layout(out_flat.to(q.dtype), v, layout), state.to(state_dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """Run the CPU reference at original or fp64 precision."""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return recurrent_kda_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        layout=inputs["layout"],
        scale=inputs["scale"],
        output_final_state=True,
        inplace_final_state=False,
        state_v_first=bool(spec.get("state_v_first", True)),
    )


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.recurrent_kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=None,
        layout=inputs["layout"],
        scale=inputs["scale"],
        output_final_state=True,
        inplace_final_state=False,
        state_v_first=bool(spec.get("state_v_first", True)),
    )


@register("executor_recurrent_kda")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs)
