"""recurrent_kda 的 ATK executor。"""

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
    _marker_device,
    _orig_dtype,
    _rand,
    _randn,
)
from gen_recurrent_kda import _spec as _generated_spec


OP_NAME = "recurrent_kda"


def _resolved_case_spec(input_data: InputDataset) -> dict[str, Any]:
    """解析完整 spec；ATK 丢弃 non_param attr 时按 case_id 恢复。"""
    spec = _case_spec(input_data, OP_NAME)
    required = {
        "gate_dtype",
        "beta_dtype",
        "state_dtype",
        "state_capacity",
        "cu_mode",
        "ssm_mode",
        "output_final_state",
        "inplace_final_state",
    }
    if required.issubset(spec):
        return spec

    if "case_id" not in spec:
        missing = sorted(required - set(spec))
        raise KeyError(f"case_spec is incomplete and case_id is unavailable: {missing}")
    restored = _generated_spec(int(spec["case_id"]))
    for name in ("B", "T", "H", "HV", "K", "V", "layout", "state_v_first"):
        if name in spec and spec[name] != restored[name]:
            raise ValueError(
                f"restored case_spec mismatch for {name}: input={spec[name]!r}, "
                f"generated={restored[name]!r}"
            )
    return restored


def _metadata_dtype(name: str) -> torch.dtype:
    return torch.int32 if str(name).lower() == "int32" else torch.int64


def _capacity(spec: dict[str, Any]) -> int:
    return int(spec["B"]) * int(spec["T"])


def _logical_lengths(spec: dict[str, Any]) -> list[int]:
    lengths = spec.get("seq_lengths")
    if lengths is not None:
        return [int(length) for length in lengths]
    if str(spec.get("layout", "BSND")) == "BSND":
        return [int(spec["T"])] * int(spec["B"])
    return [_capacity(spec)]


def _cu_values(lengths: Sequence[int]) -> list[int]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return offsets


def _tensor_shape(spec: dict[str, Any], tail: tuple[int, ...]) -> tuple[int, ...]:
    if str(spec.get("layout", "BSND")) == "TND":
        return (_capacity(spec), *tail)
    return (int(spec["B"]), int(spec["T"]), *tail)


def _randn_input(
    shape: tuple[int, ...],
    dtype_name: str,
    calc_dtype: torch.dtype,
    device: torch.device,
    seed: int,
    *,
    scale: float = 0.05,
    noncontiguous: bool = False,
) -> torch.Tensor:
    if not noncontiguous:
        return _randn(shape, dtype_name, calc_dtype, device, seed, scale=scale)
    base_shape = (*shape[:-1], shape[-1] * 2)
    base = _randn(base_shape, dtype_name, calc_dtype, device, seed, scale=scale)
    return base[..., ::2]


def _rand_input(
    shape: tuple[int, ...],
    dtype_name: str,
    calc_dtype: torch.dtype,
    device: torch.device,
    seed: int,
    low: float,
    high: float,
    *,
    noncontiguous: bool = False,
) -> torch.Tensor:
    if not noncontiguous:
        return _rand(shape, dtype_name, calc_dtype, device, seed, low, high)
    base_shape = (*shape[:-1], shape[-1] * 2)
    base = _rand(base_shape, dtype_name, calc_dtype, device, seed, low, high)
    return base[..., ::2]


def _build_state(
    spec: dict[str, Any],
    device: torch.device,
    calc_dtype: torch.dtype,
    seed: int,
) -> Optional[torch.Tensor]:
    if bool(spec.get("initial_state_none", False)):
        return None
    state_capacity = int(spec["state_capacity"])
    HV, K, V = (int(spec[name]) for name in ("HV", "K", "V"))
    tail = (HV, V, K) if bool(spec.get("state_v_first", False)) else (HV, K, V)
    shape = (state_capacity, *tail)
    state_dtype = str(spec.get("state_dtype", "fp32"))
    if not bool(spec.get("state_noncontiguous", False)):
        return _randn(shape, state_dtype, calc_dtype, device, seed, scale=0.01)

    # 仅在 slot/head 两个外层维制造间隔，最后二维 state matrix 保持稠密。
    base_shape = (state_capacity + 1, HV + 1, *tail[1:])
    base = _randn(base_shape, state_dtype, calc_dtype, device, seed, scale=0.01)
    state = base[:state_capacity, :HV]
    if state.stride(-1) != 1 or state.stride(-2) != state.shape[-1]:
        raise RuntimeError("non-contiguous state must keep its inner matrix dense")
    return state


def _build_metadata(
    spec: dict[str, Any],
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    lengths = _logical_lengths(spec)
    offsets = _cu_values(lengths)
    cu_seqlens = None
    if str(spec.get("cu_mode", "none")) != "none":
        cu_seqlens = _int_tensor(
            offsets,
            device,
            _metadata_dtype(str(spec.get("cu_dtype", "int64"))),
        )

    ssm_mode = str(spec.get("ssm_mode", "none"))
    if ssm_mode == "none":
        return cu_seqlens, None, None

    capacity = _capacity(spec)
    state_capacity = int(spec["state_capacity"])
    seed = int(spec.get("seed", 20260817))
    slot_offset = seed % state_capacity
    slots = [(slot_offset + seq_index) % state_capacity for seq_index in range(len(lengths))]
    ssm_dtype = _metadata_dtype(str(spec.get("ssm_dtype", "int64")))

    if ssm_mode == "packed":
        values = [0] * capacity
        for seq_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            values[start:end] = [slots[seq_index]] * (end - start)
        ssm_state_indices = _int_tensor(values, device, ssm_dtype)
    elif ssm_mode == "speculative":
        max_step = max(1, max(lengths))
        values = [[slots[seq_index]] * max_step for seq_index in range(len(lengths))]
        ssm_state_indices = _int_tensor(values, device, ssm_dtype)
    else:
        raise ValueError(f"unsupported ssm_mode: {ssm_mode}")

    num_accepted_tokens = None
    if bool(spec.get("accepted_tokens", False)):
        accepted = [max(1, (length + 1) // 2) for length in lengths]
        num_accepted_tokens = _int_tensor(
            accepted,
            device,
            _metadata_dtype(str(spec.get("accepted_dtype", "int64"))),
        )
    return cu_seqlens, ssm_state_indices, num_accepted_tokens


def build_inputs(
    spec: dict[str, Any],
    device: torch.device,
    high_precision: bool = False,
) -> dict[str, Any]:
    seed = int(spec.get("seed", 20260817))
    H, HV, K, V = (int(spec[name]) for name in ("H", "HV", "K", "V"))
    gate_dtype_name = str(spec.get("gate_dtype", "fp32"))
    beta_dtype_name = str(spec.get("beta_dtype", "fp32"))
    state_dtype_name = str(spec.get("state_dtype", "fp32"))
    data_calc_dtype = torch.float64 if high_precision else torch.bfloat16
    gate_calc_dtype = torch.float64 if high_precision else _orig_dtype(gate_dtype_name)
    beta_calc_dtype = torch.float64 if high_precision else _orig_dtype(beta_dtype_name)
    state_calc_dtype = torch.float64 if high_precision else _orig_dtype(state_dtype_name)
    noncontiguous = bool(spec.get("input_noncontiguous", False))

    q_shape = _tensor_shape(spec, (H, K))
    v_shape = _tensor_shape(spec, (HV, V))
    g_shape = _tensor_shape(spec, (HV, K))
    beta_shape = _tensor_shape(spec, (HV,))

    q = _randn_input(q_shape, "bf16", data_calc_dtype, device, seed + 1, noncontiguous=noncontiguous)
    k = _randn_input(q_shape, "bf16", data_calc_dtype, device, seed + 2, noncontiguous=noncontiguous)
    v = _randn_input(v_shape, "bf16", data_calc_dtype, device, seed + 3, noncontiguous=noncontiguous)

    if bool(spec.get("use_gate_in_kernel", False)):
        g = _randn_input(
            g_shape,
            gate_dtype_name,
            gate_calc_dtype,
            device,
            seed + 4,
            scale=0.2,
            noncontiguous=noncontiguous,
        )
    else:
        g = -_rand_input(
            g_shape,
            gate_dtype_name,
            gate_calc_dtype,
            device,
            seed + 4,
            0.001,
            0.02,
            noncontiguous=noncontiguous,
        )

    if bool(spec.get("use_beta_sigmoid_in_kernel", False)):
        beta = _randn_input(
            beta_shape,
            beta_dtype_name,
            beta_calc_dtype,
            device,
            seed + 5,
            scale=0.5,
            noncontiguous=noncontiguous,
        )
    else:
        beta = _rand_input(
            beta_shape,
            beta_dtype_name,
            beta_calc_dtype,
            device,
            seed + 5,
            0.1,
            0.9,
            noncontiguous=noncontiguous,
        )

    initial_state = _build_state(spec, device, state_calc_dtype, seed + 6)
    cu_seqlens, ssm_state_indices, num_accepted_tokens = _build_metadata(spec, device)

    A_log = None
    dt_bias = None
    if bool(spec.get("use_gate_in_kernel", False)):
        A_log = -_rand((HV,), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 7, 0.2, 1.2)
        dt_bias_mode = str(spec.get("dt_bias_mode", "none"))
        if dt_bias_mode == "flat":
            dt_bias = _randn(
                (HV * K,), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 8, scale=0.05
            )
        elif dt_bias_mode == "matrix":
            dt_bias = _randn(
                (HV, K), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 8, scale=0.05
            )
        elif dt_bias_mode != "none":
            raise ValueError(f"unsupported dt_bias_mode: {dt_bias_mode}")

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "num_accepted_tokens": num_accepted_tokens,
        "layout": str(spec.get("layout", "BSND")),
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
        "valid_tokens": sum(_logical_lengths(spec)),
    }


# CPU reference is kept local so NPU, same-precision CPU and FP64 benchmark use
# the exact same case_spec and input quantization.
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
    if offsets[-1] > total_tokens:
        raise ValueError("the last cu_seqlens offset must not exceed token capacity")
    if any(end - start > 8 for start, end in zip(offsets, offsets[1:])):
        raise ValueError("each recurrent sequence length must be <= 8")
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

    work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q_flat = _flatten_bsnd(q, layout).to(work_dtype)
    k_flat = _flatten_bsnd(k, layout).to(work_dtype)
    v_flat = _flatten_bsnd(v, layout).to(work_dtype)
    g_flat = _flatten_bsnd(g, layout).to(work_dtype)
    beta_flat = _flatten_bsnd(beta, layout).to(work_dtype)
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
            gate = gate + dt_bias.to(work_dtype).reshape(hv, dk).unsqueeze(0)
        exp_a = torch.exp(A_log.to(work_dtype)).reshape(1, hv, 1)
        if safe_gate:
            gate = lower_bound * torch.sigmoid(exp_a * gate)
        else:
            gate = -exp_a * F.softplus(gate)
    else:
        gate = g_flat
    gate_decay = torch.exp(gate)

    beta_eff = beta_flat
    if use_beta_sigmoid_in_kernel:
        beta_eff = torch.sigmoid(beta_eff)
        if allow_neg_eigval:
            beta_eff = beta_eff * 2.0

    if cu_seqlens is None:
        if layout == "BSND":
            dense_seq_len = q.shape[1]
            cu_seqlens = [i * dense_seq_len for i in range(q.shape[0] + 1)]
        else:
            cu_seqlens = [0, total_tokens]
    ranges = _seq_ranges(total_tokens, cu_seqlens)
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
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
                out_slot = (
                    _state_slot(ssm_state_indices, seq_idx, start, token)
                    if ssm_state_indices is not None
                    else seq_idx
                )
                state[out_slot, hv_idx] = state_cur

    if not state_v_first:
        state = state.transpose(-1, -2).contiguous()
    return _restore_layout(out_flat.to(q.dtype), v, layout), state.to(state_dtype)


def _visible_outputs(
    outputs: tuple[torch.Tensor, Optional[torch.Tensor]],
    spec: dict[str, Any],
    valid_tokens: int,
    ssm_state_indices: Optional[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    out, final_state = outputs
    capacity = _capacity(spec)
    if valid_tokens < capacity:
        flat = out.reshape(capacity, *out.shape[-2:]).clone()
        flat[valid_tokens:].zero_()
        out = flat.reshape(out.shape)
    visible = [out]
    if bool(spec.get("output_final_state", True)) and final_state is not None:
        lengths = _logical_lengths(spec)
        if ssm_state_indices is None:
            referenced_slots = set(range(len(lengths)))
        else:
            indices = ssm_state_indices.detach().cpu()
            referenced_slots = set()
            if indices.ndim == 1:
                offsets = _cu_values(lengths)
                for start, end in zip(offsets[:-1], offsets[1:]):
                    referenced_slots.update(int(value) for value in indices[start:end].tolist())
            elif indices.ndim == 2:
                for seq_index, length in enumerate(lengths):
                    referenced_slots.update(
                        int(value) for value in indices[seq_index, :length].tolist()
                    )
            else:
                raise ValueError("ssm_state_indices must be packed [T] or speculative [seq_num,max_step]")

        # The kernel writes final_state only for slots actually selected by tokens.
        # Out-of-place unused pool slots therefore have no output semantics and may
        # retain arbitrary GM data; canonicalize only those slots before ATK checks.
        hidden_slots = set(range(final_state.shape[0])) - referenced_slots
        if hidden_slots:
            final_state = final_state.clone()
            for slot in hidden_slots:
                final_state[slot].zero_()
        visible.append(final_state)
    return tuple(visible)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    outputs = recurrent_kda_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
        layout=inputs["layout"],
        scale=inputs["scale"],
        output_final_state=bool(spec.get("output_final_state", True)),
        inplace_final_state=bool(spec.get("inplace_final_state", False)),
        use_qk_l2norm_in_kernel=bool(spec.get("use_qk_l2norm_in_kernel", False)),
        use_gate_in_kernel=bool(spec.get("use_gate_in_kernel", False)),
        use_beta_sigmoid_in_kernel=bool(spec.get("use_beta_sigmoid_in_kernel", False)),
        allow_neg_eigval=bool(spec.get("allow_neg_eigval", False)),
        safe_gate=bool(spec.get("safe_gate", False)),
        lower_bound=float(spec.get("lower_bound", -5.0)),
        state_v_first=bool(spec.get("state_v_first", False)),
    )
    return _visible_outputs(outputs, spec, inputs["valid_tokens"], inputs["ssm_state_indices"])


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    outputs = ascendc.recurrent_kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
        layout=inputs["layout"],
        scale=inputs["scale"],
        output_final_state=bool(spec.get("output_final_state", True)),
        inplace_final_state=bool(spec.get("inplace_final_state", False)),
        use_qk_l2norm_in_kernel=bool(spec.get("use_qk_l2norm_in_kernel", False)),
        use_gate_in_kernel=bool(spec.get("use_gate_in_kernel", False)),
        use_beta_sigmoid_in_kernel=bool(spec.get("use_beta_sigmoid_in_kernel", False)),
        allow_neg_eigval=bool(spec.get("allow_neg_eigval", False)),
        safe_gate=bool(spec.get("safe_gate", False)),
        lower_bound=float(spec.get("lower_bound", -5.0)),
        state_v_first=bool(spec.get("state_v_first", False)),
    )
    return _visible_outputs(outputs, spec, inputs["valid_tokens"], inputs["ssm_state_indices"])


@register("executor_recurrent_kda")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _resolved_case_spec(input_data)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(
                f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}"
            )
        return _finite_tuple(outputs)
