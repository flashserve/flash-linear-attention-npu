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
    calc = torch.float64 if high_precision else torch.bfloat16
    aux = torch.float64 if high_precision else torch.float32
    seed = int(spec.get("seed", 20260817))
    B, T, H, HV, K, V = (int(spec[x]) for x in ("B", "T", "H", "HV", "K", "V"))
    lengths = spec.get("lengths", [T] * B)
    capacity = sum(lengths) + int(spec.get("padding", 0))
    # The operator infers the storage mode from cu_seqlens presence.  Keep
    # layout in the case files only as a compatibility argument.
    varlen = bool(spec.get("cu", True))
    prefix = (capacity,) if varlen else (B, T)
    g_dtype, beta_dtype = spec.get("g_dtype", "fp32"), spec.get("beta_dtype", "fp32")
    gd = aux if high_precision or g_dtype == "fp32" else (
        torch.float16 if g_dtype == "fp16" else torch.bfloat16)
    bd = aux if high_precision or beta_dtype == "fp32" else (
        torch.float16 if beta_dtype == "fp16" else torch.bfloat16)
    q = _randn((*prefix, H, K), "bf16", calc, device, seed + 1)
    k = _randn((*prefix, H, K), "bf16", calc, device, seed + 2)
    v = _randn((*prefix, HV, V), "bf16", calc, device, seed + 3)
    if spec.get("pattern") in {"zero", "small", "signal"}:
        q.zero_(); k.zero_()
        if spec["pattern"] == "small":
            q[..., 0] = 2.0 ** -12; k[..., 0] = 2.0 ** -12
        elif spec["pattern"] == "signal":
            q[..., 0] = 1.0; k[..., 0] = 1.0; v.fill_(1.0)
    has_a_log = bool(spec.get("a_log", spec.get("use_gate_in_kernel", False)))
    if has_a_log:
        g = _randn((*prefix, HV, K), g_dtype, gd, device, seed + 4, scale=0.3)
        if spec.get("large_gate", False):
            g[..., ::2] = 100.0; g[..., 1::2] = -100.0
    else:
        g = _kda_gate((*prefix, HV, K), g_dtype, gd, device, seed + 4)
    beta = (_randn((*prefix, HV), beta_dtype, bd, device, seed + 5, scale=0.6)
            if spec.get("use_beta_sigmoid_in_kernel", False) else
            _rand((*prefix, HV), beta_dtype, bd, device, seed + 5, 0.1, 0.9))
    if spec.get("pattern") == "signal":
        g.zero_(); beta.fill_(0.5)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    metadata_dtype = torch.int32 if spec.get("metadata_dtype") == "int32" else torch.int64
    indices, accepted = None, None
    state_capacity = len(lengths)
    if spec.get("indices"):
        indices = torch.full((len(lengths), max(lengths) + 1), -1, dtype=metadata_dtype, device=device)
        for seq, length in enumerate(lengths):
            for step in range(length):
                indices[seq, step] = seq * 3 + step % 3
            if length > 1:
                indices[seq, length - 2:length] = seq * 3
        state_capacity = len(lengths) * 3 + 2
        if spec["indices"] == "packed":
            indices = torch.cat([indices[i, :length] for i, length in enumerate(lengths)])
            # Packed metadata covers the physical token capacity, including
            # the unused tail.  Only cu_seqlens' valid prefix is consumed.
            if capacity > indices.numel():
                indices = torch.cat([indices, torch.full(
                    (capacity - indices.numel(),), -1, dtype=metadata_dtype, device=device)])
        accepted = _int_tensor([min(int(spec.get("accepted", 1)), n) if n else 0 for n in lengths],
                               device, metadata_dtype)
    tail = (V, K) if spec.get("state_v_first", True) else (K, V)
    state_dtype = spec.get("state_dtype", "fp32")
    sd = aux if high_precision or state_dtype == "fp32" else torch.bfloat16
    state = _randn((state_capacity, HV, *tail), state_dtype, sd, device, seed + 6, scale=0.02)
    if spec.get("state_zero", False):
        state.zero_()
    view_kind = spec.get("qkv_view", "contiguous")
    tensors = {"q": q, "k": k, "v": v}
    if view_kind == "fused":
        # dim-last causal_conv1d output split into Q/K/V, including nonzero offsets.
        widths = [H * K, H * K, HV * V]
        storage = torch.zeros((*prefix, sum(widths) + 16), dtype=calc, device=device)
        cursor = 8
        for name, width in zip(tensors, widths):
            view = storage[..., cursor:cursor + width].view(*tensors[name].shape)
            view.copy_(tensors[name]); tensors[name] = view; cursor += width
    elif view_kind in {"head_gap", "feature_gap"}:
        for name, tensor in tensors.items():
            storage = torch.zeros((*tensor.shape[:-1], tensor.shape[-1] * 2), dtype=calc, device=device)
            view = storage[..., ::2] if view_kind == "feature_gap" else storage[..., :tensor.shape[-1]]
            view.copy_(tensor); tensors[name] = view
    elif view_kind == "batch_gap":
        for index, (name, tensor) in enumerate(tensors.items()):
            storage = torch.zeros((B, T + index + 1, tensor.shape[-2], tensor.shape[-1]),
                                  dtype=calc, device=device)
            view = storage[:, :T]
            view.copy_(tensor); tensors[name] = view
    if spec.get("strided_state", False):
        storage = torch.zeros((state_capacity, HV + 1, *tail), dtype=sd, device=device)
        view = storage[:, :HV]; view.copy_(state); state = view
    inputs = dict(**tensors, g=g, beta=beta, initial_state=None if spec.get("initial_none") else state,
                  cu_seqlens=_int_tensor(offsets, device, metadata_dtype) if spec.get("cu", True) else None,
                  ssm_state_indices=indices, num_accepted_tokens=accepted)
    if has_a_log:
        # Apply the shift before promotion so DUT and FP64 golden see exactly
        # the same FP32 input bits (rather than two different rounding paths).
        a_log = _randn((HV,), "fp32", torch.float32, torch.device("cpu"), seed + 7, scale=0.1) - 2.0
        inputs["A_log"] = a_log.to(aux).to(device)
        inputs["dt_bias"] = (None if spec.get("dt_bias_none", False) else
                             _randn((HV, K), "fp32", aux, device, seed + 8, scale=0.05))
    elif spec.get("ignored_gate_parameters"):
        # Deliberately invalid raw-gate dtype/shape/device: inactive inputs are ignored.
        inputs["dt_bias"] = torch.zeros(3, dtype=torch.int32)
    return inputs


def call_options(spec):
    return dict(layout=spec.get("layout", "BSND"), scale=spec.get("scale", 128 ** -0.5),
                output_final_state=True, inplace_final_state=bool(spec.get("inplace", False)),
                state_v_first=bool(spec.get("state_v_first", True)),
                use_qk_l2norm_in_kernel=bool(spec.get("use_qk_l2norm_in_kernel", False)),
                use_gate_in_kernel=bool(spec.get("use_gate_in_kernel", False)),
                use_beta_sigmoid_in_kernel=bool(spec.get("use_beta_sigmoid_in_kernel", False)),
                allow_neg_eigval=bool(spec.get("allow_neg_eigval", False)),
                safe_gate=bool(spec.get("safe_gate", False)),
                lower_bound=spec.get("lower_bound", -5.0))


# CPU reference copied from the operator PTA reference so this executor is self-contained.
def _flatten_bsnd(x: torch.Tensor, layout: str) -> torch.Tensor:
    del layout
    if x.ndim == 3:
        return x
    if x.ndim == 4:
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
    raise ValueError("q/k/v/g must be rank 3 with cu_seqlens or rank 4 without it")


def _restore_layout(x: torch.Tensor, ref: torch.Tensor, layout: str) -> torch.Tensor:
    del layout
    return x if ref.ndim == 3 else x.reshape(ref.shape)


def _seq_ranges(total_tokens: int, cu_seqlens: Sequence[int]):
    if len(cu_seqlens) < 2:
        raise ValueError("cu_seqlens must contain at least two cumulative offsets")
    offsets = [int(offset) for offset in cu_seqlens]
    if offsets[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if any(end < start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    if offsets[-1] > total_tokens:
        raise ValueError("the last cu_seqlens offset must not exceed token capacity")
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
    state_storage_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del output_final_state, inplace_final_state

    work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q_flat = _flatten_bsnd(q, layout).to(work_dtype)
    k_flat = _flatten_bsnd(k, layout).to(work_dtype)
    v_flat = _flatten_bsnd(v, layout).to(work_dtype)
    g_flat = _flatten_bsnd(g, layout).to(work_dtype)
    beta_flat = beta.reshape(-1, beta.shape[-1]).to(work_dtype)
    total_tokens, h, dk = q_flat.shape
    _, hv, dv = v_flat.shape
    scale = (dk ** -0.5) if scale is None else scale

    if use_qk_l2norm_in_kernel:
        q_flat = q_flat / torch.sqrt((q_flat * q_flat).sum(dim=-1, keepdim=True) + 1e-6)
        k_flat = k_flat / torch.sqrt((k_flat * k_flat).sum(dim=-1, keepdim=True) + 1e-6)
    q_flat = q_flat * scale

    if A_log is not None:
        gate = g_flat
        if dt_bias is not None:
            gate = gate + dt_bias.to(work_dtype).reshape(hv, dk).unsqueeze(0)
        exp_a = torch.exp(A_log.to(work_dtype)).reshape(1, hv, 1)
        if safe_gate:
            gate = lower_bound * torch.sigmoid(exp_a * gate)
        else:
            gate = -exp_a * F.softplus(gate)
    else:
        gate = g_flat
    gate_decay = torch.exp(gate.to(work_dtype))

    beta_eff = beta_flat
    if use_beta_sigmoid_in_kernel:
        beta_eff = torch.sigmoid(beta_eff)
        if allow_neg_eigval:
            beta_eff = beta_eff * 2.0

    if cu_seqlens is None:
        dense_seq_len = q.shape[1]
        cu_seqlens = [i * dense_seq_len for i in range(q.shape[0] + 1)]
    ranges = _seq_ranges(total_tokens, cu_seqlens)
    state_dtype = initial_state.dtype if initial_state is not None else work_dtype
    if initial_state is None:
        state = torch.zeros((len(ranges), hv, dv, dk), dtype=work_dtype, device=q.device)
    else:
        state = initial_state.to(work_dtype).clone()
        if not state_v_first:
            state = state.transpose(-1, -2).contiguous()
    out_flat = torch.zeros_like(v_flat, dtype=work_dtype)

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
                # A5 stores each recurrent update in the interface State dtype.
                # Keep out based on the unrounded FP32/FP64 register result.
                if state_storage_dtype == torch.bfloat16:
                    state_cur = state_cur.to(torch.bfloat16).to(work_dtype)
                out_slot = _state_slot(ssm_state_indices, seq_idx, start, token) if ssm_state_indices is not None else seq_idx
                state[out_slot, hv_idx] = state_cur

    if not state_v_first:
        state = state.transpose(-1, -2).contiguous()
    return _restore_layout(out_flat.to(q.dtype), v, layout), state.to(state_dtype)


def _visible_outputs(outputs, spec):
    if not spec.get("padding"):
        return outputs
    valid = sum(spec["lengths"])
    out, state = outputs
    return (out[:valid] if spec.get("cu", True) else out[:, :valid]), state


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    torch.set_num_threads(1)
    inputs = build_inputs(spec, torch.device("cpu"), high_precision)
    storage_dtype = (torch.bfloat16 if spec.get("soc", "ascend950") == "ascend950"
                     and spec.get("state_dtype", "fp32") == "bf16" and not spec.get("initial_none") else None)
    return _visible_outputs(recurrent_kda_reference(**inputs, **call_options(spec),
                                                   state_storage_dtype=storage_dtype), spec)


def run_npu(spec: dict[str, Any], input_data: InputDataset, inputs=None):
    if inputs is None:
        inputs = build_inputs(spec, _marker_device(input_data))
    from fla_npu.ops import ascendc
    before = (inputs["initial_state"].clone() if not spec.get("inplace", False)
              and inputs["initial_state"] is not None else None)
    outputs = ascendc.recurrent_kda(**inputs, **call_options(spec))
    if not spec.get("inplace", False) and before is not None:
        if not torch.equal(inputs["initial_state"], before):
            raise AssertionError("non-inplace recurrent_kda modified initial_state")
    return _visible_outputs(outputs, spec)


@register("executor_recurrent_kda")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"
        self._performance_inputs = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            if not with_output:
                # Performance loops reuse bound inputs, excluding random input
                # construction and output-validation kernels from device timing.
                # Accuracy/determinism always rebuild the original state.
                if self._performance_inputs is None:
                    self._performance_inputs = build_inputs(spec, _marker_device(input_data))
                return run_npu(spec, input_data, self._performance_inputs)
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs, golden=self.device == "cpu")
