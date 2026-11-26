"""recurrent_kda 的 ATK executor。"""

from __future__ import annotations

import importlib
import json
import math
import os
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
_DEFAULT_TRITON_CALLABLE = (
    "fla.ops.kda.fused_recurrent:fused_recurrent_kda_fwd"
)
_TRITON_CALLABLE_ENV = "RECURRENT_KDA_ATK_TRITON_CALLABLE"


def _resolved_case_spec(input_data: InputDataset, case_config=None) -> dict[str, Any]:
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
    entries = case_config.get("inputs", ()) if isinstance(case_config, dict) else getattr(case_config, "inputs", ())
    restored = None
    for entry in entries or ():
        name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
        if name == "case_spec":
            raw = entry.get("range_values") if isinstance(entry, dict) else entry.range_values
            restored = json.loads(raw) if isinstance(raw, str) else raw
            break
    if restored is None:
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
    # A2/A3 removed the historical "each sequence length <= 8" requirement, so the
    # reference no longer rejects long sequences; only ordering/capacity is checked.
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
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    del inplace_final_state

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
        q_flat = q_flat / torch.sqrt((q_flat * q_flat).sum(dim=-1, keepdim=True) + 1e-6)
        k_flat = k_flat / torch.sqrt((k_flat * k_flat).sum(dim=-1, keepdim=True) + 1e-6)
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

    out = _restore_layout(out_flat.to(q.dtype), v, layout)
    if not output_final_state:
        return out, None
    if not state_v_first:
        state = state.transpose(-1, -2).contiguous()
    return out, state.to(state_dtype)


def _visible_outputs(
    outputs: tuple[torch.Tensor, Optional[torch.Tensor]],
    spec: dict[str, Any],
    valid_tokens: int,
    ssm_state_indices: Optional[torch.Tensor],
    *,
    triton_final_state: bool = False,
) -> tuple[torch.Tensor, ...]:
    out, final_state = outputs
    inplace_final_state = bool(spec.get("inplace_final_state", False))
    capacity = _capacity(spec)
    if valid_tokens < capacity:
        flat = out.reshape(capacity, *out.shape[-2:]).clone()
        flat[valid_tokens:].zero_()
        out = flat.reshape(out.shape)
    if not bool(spec.get("output_final_state", True)):
        return (out,)

    visible = [out]
    if final_state is not None:
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
        state_dtype = str(spec.get("state_dtype", "fp32")).lower()
        # Triton allocates out-of-place final_state as FP32. Match the NPU BF16
        # state contract before normalizing every compared value to FP32.
        if triton_final_state and state_dtype == "bf16":
            final_state = final_state.to(torch.bfloat16)
        if state_dtype == "bf16" or not inplace_final_state:
            final_state = final_state.to(torch.float32)
        visible.append(final_state)
    return tuple(visible)


def run_torch_reference(
    spec: dict[str, Any],
    device: torch.device,
    high_precision: bool = False,
):
    inputs = build_inputs(spec, device, high_precision=high_precision)
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


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    return run_torch_reference(spec, torch.device("cpu"), high_precision)


def _load_triton_callable():
    target = os.environ.get(
        _TRITON_CALLABLE_ENV,
        _DEFAULT_TRITON_CALLABLE,
    ).strip()
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(
            f"{_TRITON_CALLABLE_ENV} must use '<python_module>:<callable>' syntax"
        )
    module = importlib.import_module(module_name)
    callable_obj = getattr(module, attribute, None)
    if not callable(callable_obj):
        raise RuntimeError(f"configured Triton target is not callable: {target}")
    return target, callable_obj


def _cuda_long(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if tensor is None else tensor.to(dtype=torch.int64)


def _triton_tensor_layout(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return tensor.unsqueeze(0)
    if layout == "BSND":
        return tensor
    raise RuntimeError(f"unsupported recurrent_kda GPU layout: {layout}")


def _triton_control_supported(spec: dict[str, Any]) -> bool:
    """The upstream allocator cannot size an indexed out-of-place state pool."""
    if str(spec.get("ssm_mode", "none")) == "none":
        return True
    if bool(spec.get("inplace_final_state", False)):
        return True
    return int(spec["state_capacity"]) == len(_logical_lengths(spec))


def run_gpu_control(spec: dict[str, Any], input_data: InputDataset):
    device = _marker_device(input_data)
    if device.type != "cuda":
        raise RuntimeError("Triton control must run on an ATK GPU node")
    if not _triton_control_supported(spec):
        return run_torch_reference(spec, device, high_precision=False)

    inputs = build_inputs(spec, device, high_precision=False)
    target, triton_callable = _load_triton_callable()
    layout = str(inputs["layout"])
    outputs = triton_callable(
        q=_triton_tensor_layout(inputs["q"], layout),
        k=_triton_tensor_layout(inputs["k"], layout),
        v=_triton_tensor_layout(inputs["v"], layout),
        g=_triton_tensor_layout(inputs["g"], layout),
        beta=_triton_tensor_layout(inputs["beta"], layout),
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        initial_state=inputs["initial_state"],
        scale=inputs["scale"],
        output_final_state=bool(spec.get("output_final_state", True)),
        inplace_final_state=bool(spec.get("inplace_final_state", False)),
        state_v_first=bool(spec.get("state_v_first", False)),
        cu_seqlens=_cuda_long(inputs["cu_seqlens"]),
        ssm_state_indices=_cuda_long(inputs["ssm_state_indices"]),
        num_accepted_tokens=_cuda_long(inputs["num_accepted_tokens"]),
        use_qk_l2norm_in_kernel=bool(spec.get("use_qk_l2norm_in_kernel", False)),
        use_gate_in_kernel=bool(spec.get("use_gate_in_kernel", False)),
        use_beta_sigmoid_in_kernel=bool(
            spec.get("use_beta_sigmoid_in_kernel", False)
        ),
        allow_neg_eigval=bool(spec.get("allow_neg_eigval", False)),
        lower_bound=(
            float(spec.get("lower_bound", -5.0))
            if bool(spec.get("safe_gate", False))
            else None
        ),
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise RuntimeError(
            f"Triton callable {target} must return (output, final_state)"
        )
    output, final_state = outputs
    if layout == "TND":
        output = output.squeeze(0)
    return _visible_outputs(
        (output, final_state),
        spec,
        inputs["valid_tokens"],
        inputs["ssm_state_indices"],
        triton_final_state=True,
    )


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    # Preserve the reference's implicit zero-state semantics on newer wrappers.
    if inputs["initial_state"] is None:
        hv, k, v = (int(spec[name]) for name in ("HV", "K", "V"))
        tail = (hv, v, k) if bool(spec.get("state_v_first", False)) else (hv, k, v)
        inputs["initial_state"] = torch.zeros(
            (len(_logical_lengths(spec)), *tail),
            dtype=_orig_dtype(str(spec.get("state_dtype", "fp32"))),
            device=inputs["q"].device,
        )
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
        self.case_config = getattr(task_result, "case_config", None)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = (
            self.device in {"cpu", "gpu"} and self.is_benchmark_task
        )
        self.cpu_control = self.device == "cpu" and not self.is_benchmark_task
        self.triton_control = self.device == "gpu" and not self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _resolved_case_spec(input_data, self.case_config)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.high_precision:
            outputs = run_torch_reference(
                spec,
                _marker_device(input_data),
                high_precision=True,
            )
        elif self.cpu_control:
            outputs = run_cpu(spec, high_precision=False)
        elif self.triton_control:
            outputs = run_gpu_control(spec, input_data)
        else:
            raise RuntimeError(
                f"{OP_NAME} requires an NPU DUT and a CPU or GPU reference node"
            )
        visible_outputs = _finite_tuple(outputs)
        if not bool(spec.get("output_final_state", True)):
            return visible_outputs[:1]
        return visible_outputs
