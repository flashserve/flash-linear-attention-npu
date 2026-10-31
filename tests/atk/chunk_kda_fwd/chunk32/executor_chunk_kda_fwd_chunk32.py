"""ATK executor and dual references for chunk_kda_fwd."""

from __future__ import annotations

import importlib
import json
import math
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import numpy as np

    torch.serialization.add_safe_globals(
        [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float32)),
            type(np.dtype(np.float64)),
            type(np.dtype(np.int32)),
            type(np.dtype(np.int64)),
        ]
    )
except (AttributeError, ImportError):
    pass

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_OUTPUT_NAMES = (
    "attn_out",
    "final_state",
    "gk",
    "Aqk",
    "Akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state_out",
)
_REFERENCE_WORKERS = int(os.environ.get("KDA_ATK_REFERENCE_WORKERS", "48"))
_LEGACY_REFERENCE_CACHE: OrderedDict[str, tuple] = OrderedDict()
_LEGACY_REFERENCE_CACHE_ENTRIES = int(
    os.environ.get("KDA_ATK_REFERENCE_CACHE_ENTRIES", "1")
)
_DEFAULT_TRITON_CALLABLE = "fla.ops.kda.chunk_fwd:chunk_kda_fwd"


def _selected_output_names() -> Optional[set[str]]:
    value = os.environ.get("KDA_ATK_VISIBLE_OUTPUTS", "").strip()
    if not value:
        return None
    selected = {name.strip() for name in value.split(",") if name.strip()}
    unknown = selected.difference(_OUTPUT_NAMES)
    if unknown:
        raise ValueError(f"unknown KDA_ATK_VISIBLE_OUTPUTS entries: {sorted(unknown)}")
    return selected


@dataclass
class _PreparedInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    A_log: Optional[torch.Tensor]
    dt_bias: Optional[torch.Tensor]
    initial_state: Optional[torch.Tensor]
    cu_seqlens: Optional[list[int]]
    chunk_indices: Optional[list[int]]
    legacy_seed: Optional[int] = None


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _optional_tensor(value):
    if value is None:
        return None
    if torch.is_tensor(value) and value.numel() == 0:
        return None
    return value


def _normalize_int_list(value, name: str) -> Optional[list[int]]:
    if value is None:
        return None
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and value[0] in (None, "null")
    ):
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an integer list or null")
    normalized = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{name}[{index}] must be an integer")
        try:
            converted = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be an integer") from exc
        if isinstance(item, float) and not item.is_integer():
            raise ValueError(f"{name}[{index}] must be an integer")
        normalized.append(converted)
    return normalized


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.float64: "fp64",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported floating dtype: {dtype}") from exc


def _published_output(output: torch.Tensor) -> torch.Tensor:
    if output.is_floating_point():
        return output.to(torch.float32)
    return output


def _derive_spec(
    values: dict,
    q_dtype: str,
    g_dtype: str,
    beta_dtype: str,
    case_id: int,
) -> tuple[dict, Optional[list[int]], Optional[list[int]]]:
    layout = str(values.get("layout", "BSND"))
    if layout not in {"BSND", "BNSD", "TND", "NTD"}:
        raise ValueError(f"unsupported layout: {layout!r}")

    q = values["q"]
    k = values["k"]
    v = values["v"]
    g = values["g"]
    beta = values["beta"]
    if not all(torch.is_tensor(item) for item in (q, k, v, g, beta)):
        raise TypeError("q, k, v, g, and beta must be tensors")
    if q.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError(f"unsupported q dtype: {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must use the same dtype")
    if q_dtype not in {"bf16", "fp16"}:
        raise ValueError(f"q/k/v config dtype must be bf16 or fp16, got {q_dtype!r}")
    if g_dtype not in {"bf16", "fp32"}:
        raise ValueError(f"g config dtype must be bf16 or fp32, got {g_dtype!r}")
    if beta_dtype not in {"bf16", "fp32"}:
        raise ValueError(f"beta config dtype must be bf16 or fp32, got {beta_dtype!r}")

    q_shape = tuple(q.shape)
    v_shape = tuple(v.shape)
    if q_shape != tuple(k.shape):
        raise ValueError("q and k must have identical shapes")
    if layout == "BSND":
        if len(q_shape) != 4:
            raise ValueError("BSND q/k must be rank 4")
        batch, total_t, h_num, k_dim = q_shape
        if len(v_shape) != 4:
            raise ValueError("BSND v/g must be rank 4")
        hv_num, v_dim = v_shape[2], v_shape[3]
        expected_v = (batch, total_t, hv_num, v_dim)
        expected_g = (batch, total_t, hv_num, k_dim)
        expected_beta = (batch, total_t, hv_num)
    elif layout == "BNSD":
        if len(q_shape) != 4:
            raise ValueError("BNSD q/k must be rank 4")
        batch, h_num, total_t, k_dim = q_shape
        if len(v_shape) != 4:
            raise ValueError("BNSD v/g must be rank 4")
        hv_num, v_dim = v_shape[1], v_shape[3]
        expected_v = (batch, hv_num, total_t, v_dim)
        expected_g = (batch, hv_num, total_t, k_dim)
        expected_beta = (batch, hv_num, total_t)
    elif layout == "TND":
        if len(q_shape) != 3:
            raise ValueError("TND q/k must be rank 3")
        batch, total_t, h_num, k_dim = 1, *q_shape
        if len(v_shape) != 3:
            raise ValueError("TND v/g must be rank 3")
        hv_num, v_dim = v_shape[1], v_shape[2]
        expected_v = (total_t, hv_num, v_dim)
        expected_g = (total_t, hv_num, k_dim)
        expected_beta = (total_t, hv_num)
    else:
        if len(q_shape) != 3:
            raise ValueError("NTD q/k must be rank 3")
        h_num, total_t, k_dim = q_shape
        batch = 1
        if len(v_shape) != 3:
            raise ValueError("NTD v/g must be rank 3")
        hv_num, v_dim = v_shape[0], v_shape[2]
        expected_v = (hv_num, total_t, v_dim)
        expected_g = (hv_num, total_t, k_dim)
        expected_beta = (hv_num, total_t)

    if v_shape != expected_v or tuple(g.shape) != expected_g or tuple(beta.shape) != expected_beta:
        raise ValueError("v, g, or beta shape does not match the selected layout")
    if not (0 < h_num <= hv_num <= 128 and hv_num % h_num == 0):
        raise ValueError("H/HV must satisfy 0 < H <= HV <= 128 and HV % H == 0")
    if not (16 <= k_dim <= 256 and k_dim % 16 == 0):
        raise ValueError("K must be a multiple of 16 in [16, 256]")
    if not (16 <= v_dim <= 256 and v_dim % 16 == 0):
        raise ValueError("V must be a multiple of 16 in [16, 256]")

    chunk_size = int(values.get("chunk_size", 64))
    if chunk_size not in {32, 64, 128}:
        raise ValueError("chunk_size must be 64 or 128")
    cu_seqlens = _normalize_int_list(values.get("cu_seqlens"), "cu_seqlens")
    if layout in {"TND", "NTD"} and cu_seqlens is None:
        raise ValueError(f"{layout} requires cu_seqlens")
    if cu_seqlens is not None:
        if len(cu_seqlens) < 2:
            raise ValueError("cu_seqlens must contain at least [0, total_tokens]")
        if cu_seqlens[0] != 0 or cu_seqlens[-1] != total_t:
            raise ValueError("cu_seqlens must start at 0 and end at total_tokens")
        if any(left > right for left, right in zip(cu_seqlens, cu_seqlens[1:])):
            raise ValueError("cu_seqlens must be nondecreasing")
        if len(cu_seqlens) - 1 > 1024:
            raise ValueError("cu_seqlens supports at most 1024 logical sequences")
        if layout in {"BSND", "BNSD"} and batch != 1:
            raise ValueError("rank-4 varlen input requires B=1")

    chunk_indices = _normalize_int_list(values.get("chunk_indices"), "chunk_indices")
    canonical_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)
    if chunk_size == 32:
        # fused kernel 从 cu 推导执行顺序，不读取索引表；但 direct case_spec 仍须先
        # 校验显式列表，不能把非法输入静默归一化为 None。
        if chunk_indices is not None and chunk_indices != canonical_indices:
            raise ValueError("chunk_indices must use canonical sequence-major order")
        chunk_indices = None
        values["disable_recompute"] = False
    elif chunk_indices is not None and chunk_indices != canonical_indices:
        raise ValueError("chunk_indices must use canonical sequence-major order")

    seq_num = len(cu_seqlens) - 1 if cu_seqlens is not None else batch
    state_v_first = _as_bool(values.get("state_v_first", False))
    initial_state = _optional_tensor(values.get("initial_state"))
    if initial_state is not None:
        state_shape = (
            (seq_num, hv_num, v_dim, k_dim)
            if state_v_first
            else (seq_num, hv_num, k_dim, v_dim)
        )
        if tuple(initial_state.shape) != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}")

    use_gate = _as_bool(values.get("use_gate_in_kernel", False))
    safe_gate = _as_bool(values.get("safe_gate", False))
    a_log = _optional_tensor(values.get("A_log"))
    dt_bias = _optional_tensor(values.get("dt_bias"))
    if use_gate and a_log is None:
        raise ValueError("A_log is required when use_gate_in_kernel is true")
    if a_log is not None and tuple(a_log.shape) != (hv_num,):
        raise ValueError(f"A_log must have shape {(hv_num,)}")
    if dt_bias is not None and tuple(dt_bias.shape) != (hv_num * k_dim,):
        raise ValueError(f"dt_bias must have shape {(hv_num * k_dim,)}")
    lower_bound = float(values.get("lower_bound", -5.0))
    if use_gate and safe_gate and not (-5.0 <= lower_bound < 0.0):
        raise ValueError("lower_bound must be in [-5, 0) for safe gate")

    spec = {
        "case_key": f"chunk_kda_fwd_{case_id:04d}",
        "B": int(batch),
        "H": int(h_num),
        "HV": int(hv_num),
        "T": int(total_t),
        "K": int(k_dim),
        "V": int(v_dim),
        "q_dtype": q_dtype,
        "g_dtype": g_dtype,
        "beta_dtype": beta_dtype,
        "layout": layout,
        "scale": float(values["scale"]),
        "chunk_size": chunk_size,
        "output_final_state": _as_bool(values.get("output_final_state", False)),
        "safe_gate": safe_gate,
        "lower_bound": lower_bound,
        "use_gate_in_kernel": use_gate,
        "disable_recompute": _as_bool(values.get("disable_recompute", False)),
        "return_intermediate_states": _as_bool(values.get("return_intermediate_states", False)),
        "state_v_first": state_v_first,
    }
    return spec, cu_seqlens, chunk_indices


def _direct_inputs(values: dict, spec: dict, cu_seqlens, chunk_indices, high_precision: bool) -> _PreparedInputs:
    def convert(value):
        value = _optional_tensor(value)
        if value is not None and high_precision and value.is_floating_point():
            return value.to(torch.float64)
        return value

    return _PreparedInputs(
        q=convert(values["q"]),
        k=convert(values["k"]),
        v=convert(values["v"]),
        g=convert(values["g"]),
        beta=convert(values["beta"]),
        A_log=convert(values.get("A_log")),
        dt_bias=convert(values.get("dt_bias")),
        initial_state=convert(values.get("initial_state")),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def _canonical_chunk_indices(cu_seqlens: Optional[list[int]], chunk_size: int) -> Optional[list[int]]:
    if cu_seqlens is None:
        return None
    indices = []
    for seq_id, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for chunk_id in range((end - start + chunk_size - 1) // chunk_size):
            indices.extend((seq_id, chunk_id))
    return indices


def _legacy_cu_seqlens(value) -> Optional[list[int]]:
    value = str(value or "").strip()
    if not value:
        return None
    return [int(item) for item in value.split(",")]


def _legacy_target_dtype(original: str, high_precision: bool) -> torch.dtype:
    return torch.float64 if high_precision else _DTYPES[original]


def _legacy_random_quantized(
    shape,
    generator: torch.Generator,
    original_dtype: torch.dtype,
    target_dtype: torch.dtype,
    device,
    *,
    low: float,
    high: float,
) -> torch.Tensor:
    value = torch.rand(shape, generator=generator, dtype=torch.float32)
    value = value.mul(high - low).add(low).to(original_dtype).to(target_dtype)
    return value.to(device)


def _legacy_normal_quantized(
    shape,
    generator: torch.Generator,
    original_dtype: torch.dtype,
    target_dtype: torch.dtype,
    device,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    sigmoid: bool = False,
    l2_normalize: bool = False,
) -> torch.Tensor:
    value = torch.randn(shape, generator=generator, dtype=torch.float32).mul(std).add(mean)
    if sigmoid:
        value = torch.sigmoid(value)
    if l2_normalize:
        value = value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + 1e-6)
    return value.to(original_dtype).to(target_dtype).to(device)


def _legacy_layout_from_bsnd(
    tensor: torch.Tensor, layout: str, *, beta: bool = False
) -> torch.Tensor:
    if layout == "BSND":
        return tensor.contiguous()
    if layout == "BNSD":
        return tensor.permute(0, 2, 1) if beta else tensor.permute(0, 2, 1, 3)
    if layout == "TND":
        return tensor.squeeze(0).contiguous()
    if layout == "NTD":
        squeezed = tensor.squeeze(0)
        return squeezed.permute(1, 0) if beta else squeezed.permute(1, 0, 2)
    raise ValueError(f"unsupported layout: {layout}")


def _legacy_inputs(
    spec: dict,
    low_marker: torch.Tensor,
    *,
    high_precision: bool,
    seed: int,
) -> _PreparedInputs:
    """Materialize the repository's existing marker/case_spec JSON cases."""

    device = low_marker.device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    batch = int(spec["B"])
    total_t = int(spec["T"])
    h_num = int(spec["H"])
    hv_num = int(spec["HV"])
    k_dim = int(spec["K"])
    v_dim = int(spec["V"])
    layout = str(spec["layout"])
    data_profile = str(spec.get("data_profile", "uniform"))

    q_original = _DTYPES[str(spec["q_dtype"])]
    q_target = _legacy_target_dtype(str(spec["q_dtype"]), high_precision)
    g_original = _DTYPES[str(spec["g_dtype"])]
    g_target = _legacy_target_dtype(str(spec["g_dtype"]), high_precision)
    beta_original = _DTYPES[str(spec["beta_dtype"])]
    beta_target = _legacy_target_dtype(str(spec["beta_dtype"]), high_precision)
    fp32_target = torch.float64 if high_precision else torch.float32

    if data_profile == "normal":
        q_bsnd = _legacy_normal_quantized(
            (batch, total_t, h_num, k_dim), generator, q_original, q_target, device,
            std=float(spec["qk_scale"]), l2_normalize=False,
        )
        k_bsnd = _legacy_normal_quantized(
            (batch, total_t, h_num, k_dim), generator, q_original, q_target, device,
            std=float(spec["qk_scale"]), l2_normalize=False,
        )
        v_bsnd = _legacy_normal_quantized(
            (batch, total_t, hv_num, v_dim), generator, q_original, q_target, device,
            std=float(spec["v_scale"]),
        )
        g_bsnd = _legacy_normal_quantized(
            (batch, total_t, hv_num, k_dim), generator, g_original, g_target, device,
            std=float(spec["gate_scale"]),
        )
        beta_bsnd = _legacy_normal_quantized(
            (batch, total_t, hv_num), generator, beta_original, beta_target, device,
            mean=float(spec["beta_bias"]), std=float(spec["beta_scale"]), sigmoid=False,
        )
    else:
        data_scale = float(spec["data_scale"])
        q_bsnd = _legacy_random_quantized(
            (batch, total_t, h_num, k_dim), generator, q_original, q_target, device,
            low=-data_scale, high=data_scale,
        )
        k_bsnd = _legacy_random_quantized(
            (batch, total_t, h_num, k_dim), generator, q_original, q_target, device,
            low=-data_scale, high=data_scale,
        )
        v_bsnd = _legacy_random_quantized(
            (batch, total_t, hv_num, v_dim), generator, q_original, q_target, device,
            low=-data_scale, high=data_scale,
        )
        gate_scale = float(spec["gate_scale"])
        gate_low, gate_high = (
            (-gate_scale, gate_scale)
            if _as_bool(spec["use_gate_in_kernel"])
            else (-0.02 * gate_scale, -0.002 * gate_scale)
        )
        g_bsnd = _legacy_random_quantized(
            (batch, total_t, hv_num, k_dim), generator, g_original, g_target, device,
            low=gate_low, high=gate_high,
        )
        beta_bsnd = _legacy_random_quantized(
            (batch, total_t, hv_num), generator, beta_original, beta_target, device,
            low=0.0, high=1.0,
        )

    a_log = None
    if _as_bool(spec["use_gate_in_kernel"]):
        if data_profile == "normal":
            a_log = _legacy_normal_quantized(
                (hv_num,), generator, torch.float32, fp32_target, device,
                std=float(spec["a_log_scale"]),
            )
        else:
            a_log = _legacy_random_quantized(
                (hv_num,), generator, torch.float32, fp32_target, device,
                low=-6.0, high=-2.0,
            )

    dt_bias = None
    if _as_bool(spec["dt_bias"]):
        if data_profile == "normal":
            dt_bias = _legacy_normal_quantized(
                (hv_num * k_dim,), generator, torch.float32, fp32_target, device,
                mean=float(spec["dt_bias_mean"]), std=float(spec["dt_bias_scale"]),
            )
        else:
            dt_bias = _legacy_random_quantized(
                (hv_num * k_dim,), generator, torch.float32, fp32_target, device,
                low=-2.0, high=2.0,
            )

    cu_seqlens = _legacy_cu_seqlens(spec.get("cu_seqlens"))
    seq_num = len(cu_seqlens) - 1 if cu_seqlens is not None else batch
    initial_state = None
    if _as_bool(spec["initial_state"]):
        state_shape = (
            (seq_num, hv_num, v_dim, k_dim)
            if _as_bool(spec["state_v_first"])
            else (seq_num, hv_num, k_dim, v_dim)
        )
        initial_state = _legacy_random_quantized(
            state_shape, generator, torch.float32, fp32_target, device,
            low=-0.02, high=0.02,
        )

    chunk_indices = None
    if _as_bool(spec["explicit_chunk_indices"]):
        chunk_indices = _canonical_chunk_indices(cu_seqlens, int(spec["chunk_size"]))
    return _PreparedInputs(
        q=_legacy_layout_from_bsnd(q_bsnd, layout),
        k=_legacy_layout_from_bsnd(k_bsnd, layout),
        v=_legacy_layout_from_bsnd(v_bsnd, layout),
        g=_legacy_layout_from_bsnd(g_bsnd, layout),
        beta=_legacy_layout_from_bsnd(beta_bsnd, layout, beta=True),
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        legacy_seed=seed,
    )


def _layout_to_bsnd(tensor: torch.Tensor, layout: str, *, beta: bool = False) -> torch.Tensor:
    if layout == "BSND":
        return tensor
    if layout == "BNSD":
        return tensor.permute(0, 2, 1) if beta else tensor.permute(0, 2, 1, 3)
    if layout == "TND":
        return tensor.unsqueeze(0)
    if layout == "NTD":
        return (tensor.permute(1, 0) if beta else tensor.permute(1, 0, 2)).unsqueeze(0)
    raise ValueError(f"unsupported layout: {layout}")


def _gate_cumsum(inputs: _PreparedInputs, spec: dict) -> torch.Tensor:
    layout = str(spec["layout"])
    g = _layout_to_bsnd(inputs.g, layout).to(
        torch.float64 if inputs.g.dtype == torch.float64 else torch.float32
    )
    compute_dtype = g.dtype
    if _as_bool(spec["use_gate_in_kernel"]):
        eig = torch.exp(inputs.A_log.to(compute_dtype)).view(1, 1, int(spec["HV"]), 1)
        raw = g
        if inputs.dt_bias is not None:
            raw = raw + inputs.dt_bias.to(compute_dtype).view(1, 1, int(spec["HV"]), int(spec["K"]))
        if _as_bool(spec["safe_gate"]):
            gate = float(spec["lower_bound"]) * torch.sigmoid(eig * raw)
        else:
            gate = -eig * torch.nn.functional.softplus(raw)
    else:
        gate = g

    gk = torch.empty_like(gate)
    cu = inputs.cu_seqlens
    chunk_size = int(spec["chunk_size"])
    spans = [(0, int(spec["T"]))] if cu is None else list(zip(cu, cu[1:]))
    for seq_start, seq_end in spans:
        for start in range(seq_start, seq_end, chunk_size):
            end = min(start + chunk_size, seq_end)
            gk[:, start:end] = torch.cumsum(gate[:, start:end], dim=1) / math.log(2.0)
    return gk


def _spans(batch: int, total_t: int, chunk_size: int, cu: Optional[list[int]]):
    if cu is None:
        return [
            (batch_id, batch_id, chunk_id, start, min(start + chunk_size, total_t))
            for batch_id in range(batch)
            for chunk_id, start in enumerate(range(0, total_t, chunk_size))
        ]
    spans = []
    global_chunk = 0
    for seq_id, (seq_start, seq_end) in enumerate(zip(cu, cu[1:])):
        for start in range(seq_start, seq_end, chunk_size):
            spans.append((0, seq_id, global_chunk, start, min(start + chunk_size, seq_end)))
            global_chunk += 1
    return spans


def _stable_causal_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    operand_dtype: Optional[torch.dtype] = None,
    scale_in_q_operands: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_num, length, _ = q.shape
    qk = torch.zeros((head_num, length, length), dtype=q.dtype, device=q.device)
    kk = torch.zeros_like(qk)
    column_ids = torch.arange(length, device=q.device)

    # A safe gate step is bounded by 5. Shifting each 16-row query tile by
    # its final cumulative gate keeps both factors finite in float32 while
    # preserving 2**(g_i - g_j) exactly up to normal rounding.
    for row_start in range(0, length, 16):
        row_end = min(row_start + 16, length)
        shift = g[:, row_end - 1 : row_end]
        left_factor = torch.exp2(g[:, row_start:row_end] - shift)
        right_factor = torch.exp2(shift - g[:, :row_end])
        right = k[:, :row_end] * right_factor
        q_left = (q * scale if scale_in_q_operands else q)[:, row_start:row_end] * left_factor
        k_left = k[:, row_start:row_end] * left_factor
        if operand_dtype is not None:
            # The kernels keep the shifted factors in bfloat16 (qPlus/kPlus in
            # chunk_kda_fwd_prepare_vec.h).  bfloat16 shares fp32 exponent
            # range, so quantising there neither overflows nor over-states the
            # precision the operator actually uses.
            factor_dtype = torch.bfloat16
            right = right.to(factor_dtype).to(q.dtype)
            q_left = q_left.to(factor_dtype).to(q.dtype)
            k_left = k_left.to(factor_dtype).to(q.dtype)
        q_scores = torch.bmm(
            q_left,
            right.transpose(1, 2),
        ).mul(1.0 if scale_in_q_operands else float(scale))
        k_scores = torch.bmm(
            k_left,
            right.transpose(1, 2),
        )
        causal = column_ids[:row_end].unsqueeze(0) <= column_ids[row_start:row_end].unsqueeze(1)
        qk[:, row_start:row_end, :row_end] = q_scores.masked_fill(~causal, 0.0)
        kk[:, row_start:row_end, :row_end] = k_scores.masked_fill(~causal, 0.0)
    return qk, kk


def _reference_impl(inputs: _PreparedInputs, spec: dict):
    layout = str(spec["layout"])
    q = _layout_to_bsnd(inputs.q, layout)
    k = _layout_to_bsnd(inputs.k, layout)
    v = _layout_to_bsnd(inputs.v, layout)
    beta = _layout_to_bsnd(inputs.beta, layout, beta=True)
    if _as_bool(spec.get("use_beta_sigmoid_in_kernel")):
        beta = torch.sigmoid(beta)   # true 语义:raw beta 进,kernel(与 golden)内激活
    if _as_bool(spec.get("use_qk_l2norm_in_kernel")):
        # chunk32 融合 kernel 核内对 q/k 逐 token 做 L2 归一(issue #829 仅支持 true);
        # golden 必须复刻,否则未归一输入下与 kernel 系统性发散(2026-10-14 归因)
        q = torch.nn.functional.normalize(q, p=2, dim=-1, eps=float(spec.get("epsilon", 1e-6)))
        k = torch.nn.functional.normalize(k, p=2, dim=-1, eps=float(spec.get("epsilon", 1e-6)))
    gk = _gate_cumsum(inputs, spec)
    compute_dtype = torch.float64 if any(
        tensor is not None and tensor.dtype == torch.float64
        for tensor in (q, k, v, beta, inputs.A_log, inputs.dt_bias, inputs.initial_state)
    ) else torch.float32

    batch, total_t, h_num, k_dim = q.shape
    hv_num, v_dim = v.shape[2], v.shape[3]
    chunk_size = int(spec["chunk_size"])
    cu = inputs.cu_seqlens
    seq_num = len(cu) - 1 if cu is not None else batch
    chunks_per_seq = (total_t + chunk_size - 1) // chunk_size
    total_chunks = sum(
        (end - start + chunk_size - 1) // chunk_size for start, end in zip(cu, cu[1:])
    ) if cu is not None else chunks_per_seq

    output_dtype = q.dtype
    o = torch.zeros((batch, total_t, hv_num, v_dim), dtype=output_dtype, device=q.device)
    aqk = torch.zeros((batch, hv_num, total_t, chunk_size), dtype=output_dtype, device=q.device)
    akk = torch.zeros_like(aqk)
    expose_gk = not _as_bool(spec["use_gate_in_kernel"]) or _as_bool(spec["disable_recompute"])
    export_full = _as_bool(spec["disable_recompute"])
    export_h = export_full or _as_bool(spec["return_intermediate_states"])
    w_out = torch.zeros((batch, hv_num, total_t, k_dim), dtype=output_dtype, device=q.device) if export_full else None
    u_out = torch.zeros((batch, hv_num, total_t, v_dim), dtype=output_dtype, device=q.device) if export_full else None
    qg_out = torch.zeros_like(w_out) if export_full else None
    kg_out = torch.zeros_like(w_out) if export_full else None
    v_new_out = torch.zeros_like(u_out) if export_full else None
    h_shape = (batch, total_chunks, hv_num, k_dim, v_dim)
    h_out = torch.zeros(h_shape, dtype=output_dtype, device=q.device) if export_h else None

    if inputs.initial_state is None:
        state = torch.zeros((seq_num, hv_num, k_dim, v_dim), dtype=compute_dtype, device=q.device)
    else:
        state = inputs.initial_state.to(compute_dtype)
        if _as_bool(spec["state_v_first"]):
            state = state.transpose(-1, -2)
        state = state.clone()

    group = hv_num // h_num
    eye_cache = {}
    for batch_id, seq_id, chunk_id, start, end in _spans(batch, total_t, chunk_size, cu):
        length = end - start
        q_block = q[batch_id, start:end].permute(1, 0, 2).to(compute_dtype)
        k_block = k[batch_id, start:end].permute(1, 0, 2).to(compute_dtype)
        q_block = q_block.repeat_interleave(group, dim=0)
        k_block = k_block.repeat_interleave(group, dim=0)
        v_block = v[batch_id, start:end].permute(1, 0, 2).to(compute_dtype)
        beta_block = beta[batch_id, start:end].transpose(0, 1).to(compute_dtype)
        g_block = gk[batch_id, start:end].permute(1, 0, 2).to(compute_dtype)

        exp_g = torch.exp2(g_block)
        qg_block = q_block * exp_g
        k_exp = k_block * exp_g
        qk, kk = _stable_causal_scores(q_block, k_block, g_block, float(spec["scale"]))
        strict = torch.ones((length, length), dtype=torch.bool, device=q.device).tril(-1)
        lhs = kk.mul(beta_block.unsqueeze(-1)).masked_fill(~strict, 0.0)
        eye = eye_cache.get(length)
        if eye is None:
            eye = torch.eye(length, dtype=compute_dtype, device=q.device)
            eye_cache[length] = eye
        inverse = torch.linalg.solve_triangular(lhs + eye, eye.expand(hv_num, -1, -1), upper=False)

        w_block = torch.bmm(inverse, k_block * beta_block.unsqueeze(-1) * exp_g)
        u_block = torch.bmm(inverse, v_block * beta_block.unsqueeze(-1))
        last_g = g_block[:, -1]
        kg_block = k_block * torch.exp2(last_g.unsqueeze(1) - g_block)
        previous = state[seq_id if cu is not None else batch_id].clone()
        v_new_block = u_block - torch.bmm(w_block, previous)
        state[seq_id if cu is not None else batch_id] = (
            torch.exp2(last_g).unsqueeze(-1) * previous
            + torch.bmm(kg_block.transpose(1, 2), v_new_block)
        )
        out_block = (
            torch.bmm(qg_block, previous) * float(spec["scale"])
            + torch.bmm(qk, v_new_block)
        )

        o[batch_id, start:end] = out_block.permute(1, 0, 2).to(output_dtype)
        aqk[batch_id, :, start:end, :length] = qk.to(output_dtype)
        akk[batch_id, :, start:end, :length] = inverse.to(output_dtype)
        if export_full:
            w_out[batch_id, :, start:end] = w_block.to(output_dtype)
            u_out[batch_id, :, start:end] = u_block.to(output_dtype)
            qg_out[batch_id, :, start:end] = qg_block.to(output_dtype)
            kg_out[batch_id, :, start:end] = kg_block.to(output_dtype)
            v_new_out[batch_id, :, start:end] = v_new_block.to(output_dtype)
        if export_h:
            h_out[batch_id, chunk_id] = previous.to(output_dtype)

    final_state = state if _as_bool(spec["output_final_state"]) else None
    if final_state is not None and _as_bool(spec["state_v_first"]):
        final_state = final_state.transpose(-1, -2)
    if h_out is not None and _as_bool(spec["state_v_first"]):
        h_out = h_out.transpose(-1, -2)

    rank3 = layout in {"TND", "NTD"}
    gk_out = gk.permute(0, 2, 1, 3)
    if rank3:
        o = o.squeeze(0)
        aqk = aqk.squeeze(0)
        akk = akk.squeeze(0)
        gk_out = gk_out.squeeze(0)
        w_out = None if w_out is None else w_out.squeeze(0)
        u_out = None if u_out is None else u_out.squeeze(0)
        qg_out = None if qg_out is None else qg_out.squeeze(0)
        kg_out = None if kg_out is None else kg_out.squeeze(0)
        v_new_out = None if v_new_out is None else v_new_out.squeeze(0)
        h_out = None if h_out is None else h_out.squeeze(0)
    return (
        o,
        final_state,
        gk_out if expose_gk else None,
        aqk,
        akk,
        w_out,
        u_out,
        qg_out,
        kg_out,
        v_new_out,
        h_out,
        inputs.initial_state,
    )


def _reference_model_parallel(inputs: _PreparedInputs, spec: dict):
    layout = str(spec["layout"])
    q = _layout_to_bsnd(inputs.q, layout)
    k = _layout_to_bsnd(inputs.k, layout)
    v = _layout_to_bsnd(inputs.v, layout)
    beta = _layout_to_bsnd(inputs.beta, layout, beta=True)
    if _as_bool(spec.get("use_beta_sigmoid_in_kernel")):
        beta = torch.sigmoid(beta)   # true 语义:raw beta 进,kernel(与 golden)内激活
    if _as_bool(spec.get("use_qk_l2norm_in_kernel")):
        # chunk32 融合 kernel 核内对 q/k 逐 token 做 L2 归一(issue #829 仅支持 true);
        # golden 必须复刻,否则未归一输入下与 kernel 系统性发散(2026-10-14 归因)
        q = torch.nn.functional.normalize(q, p=2, dim=-1, eps=float(spec.get("epsilon", 1e-6)))
        k = torch.nn.functional.normalize(k, p=2, dim=-1, eps=float(spec.get("epsilon", 1e-6)))
    gk = _gate_cumsum(inputs, spec)
    compute_dtype = torch.float64 if any(
        tensor is not None and tensor.dtype == torch.float64
        for tensor in (
            q,
            k,
            v,
            beta,
            inputs.A_log,
            inputs.dt_bias,
            inputs.initial_state,
        )
    ) else torch.float32

    batch, total_t, h_num, k_dim = q.shape
    hv_num, v_dim = v.shape[2], v.shape[3]
    chunk_size = int(spec["chunk_size"])
    cu = inputs.cu_seqlens
    seq_num = len(cu) - 1 if cu is not None else batch
    chunks_per_seq = (total_t + chunk_size - 1) // chunk_size
    total_chunks = sum(
        (end - start + chunk_size - 1) // chunk_size for start, end in zip(cu, cu[1:])
    ) if cu is not None else chunks_per_seq
    spans = _spans(batch, total_t, chunk_size, cu)

    output_dtype = q.dtype
    o = torch.zeros((batch, total_t, hv_num, v_dim), dtype=output_dtype, device=q.device)
    aqk = torch.zeros((batch, hv_num, total_t, chunk_size), dtype=output_dtype, device=q.device)
    akk = torch.zeros_like(aqk)
    w_out = torch.zeros((batch, hv_num, total_t, k_dim), dtype=output_dtype, device=q.device)
    u_out = torch.zeros((batch, hv_num, total_t, v_dim), dtype=output_dtype, device=q.device)
    qg_out = torch.zeros_like(w_out)
    kg_out = torch.zeros_like(w_out)
    v_new_out = torch.zeros_like(u_out)
    h_out = torch.zeros(
        (batch, total_chunks, hv_num, k_dim, v_dim),
        dtype=output_dtype,
        device=q.device,
    )
    final_states = torch.zeros(
        (seq_num, hv_num, k_dim, v_dim), dtype=compute_dtype, device=q.device
    )

    lengths = {end - start for _, _, _, start, end in spans}
    strict_masks = {
        length: torch.ones((length, length), dtype=torch.bool, device=q.device).tril(-1)
        for length in lengths
    }
    eyes = {
        length: torch.eye(length, dtype=compute_dtype, device=q.device)
        for length in lengths
    }
    group = hv_num // h_num
    operand_dtype = output_dtype if output_dtype in {torch.bfloat16, torch.float16} else None

    def quantize_intermediate(tensor: torch.Tensor) -> torch.Tensor:
        if operand_dtype is None:
            return tensor
        return tensor.to(operand_dtype).to(compute_dtype)

    def run_head(hv_index: int):
        q_head = hv_index // group
        if inputs.initial_state is None:
            state = torch.zeros(
                (seq_num, k_dim, v_dim), dtype=compute_dtype, device=q.device
            )
        else:
            state = inputs.initial_state[:, hv_index].to(compute_dtype)
            if _as_bool(spec["state_v_first"]):
                state = state.transpose(-1, -2)
            state = state.clone()
        for batch_id, seq_id, chunk_id, start, end in spans:
            length = end - start
            q_block = q[batch_id, start:end, q_head].to(compute_dtype)
            k_block = k[batch_id, start:end, q_head].to(compute_dtype)
            v_block = v[batch_id, start:end, hv_index].to(compute_dtype)
            beta_block = beta[batch_id, start:end, hv_index].to(compute_dtype)
            g_block = gk[batch_id, start:end, hv_index].to(compute_dtype)
            strict = strict_masks[length]

            qk, kk = _stable_causal_scores(
                q_block.unsqueeze(0),
                k_block.unsqueeze(0),
                g_block.unsqueeze(0),
                float(spec["scale"]),
                operand_dtype=operand_dtype,
            )
            qk = qk.squeeze(0)
            kk = kk.squeeze(0)
            lhs = kk.mul(beta_block[:, None]).masked_fill(~strict, 0.0)
            inverse = torch.linalg.solve_triangular(
                lhs + eyes[length], eyes[length], upper=False
            )

            exp_g = torch.exp2(g_block)
            inverse_typed = quantize_intermediate(inverse)
            qk_typed = quantize_intermediate(qk)
            k_weighted = quantize_intermediate(
                k_block * beta_block[:, None] * exp_g
            )
            v_weighted = quantize_intermediate(v_block * beta_block[:, None])
            w_block = inverse_typed @ k_weighted
            u_block = inverse_typed @ v_weighted
            last_g = g_block[-1]
            qg_block = quantize_intermediate(q_block * exp_g)
            kg_block = quantize_intermediate(
                k_block * torch.exp2(last_g[None, :] - g_block)
            )
            state_index = seq_id if cu is not None else batch_id
            previous = state[state_index].clone()
            previous_typed = quantize_intermediate(previous)
            w_typed = quantize_intermediate(w_block)
            u_typed = quantize_intermediate(u_block)
            v_new_block = u_typed - w_typed @ previous_typed
            v_new_typed = quantize_intermediate(v_new_block)
            state[state_index] = (
                torch.exp2(last_g)[:, None] * previous
                + kg_block.T @ v_new_typed
            )
            out_block = (
                qg_block @ previous_typed * float(spec["scale"])
                + qk_typed @ v_new_typed
            )

            o[batch_id, start:end, hv_index] = out_block.to(output_dtype)
            aqk[batch_id, hv_index, start:end, :length] = qk.to(output_dtype)
            akk[batch_id, hv_index, start:end, :length] = inverse.to(output_dtype)
            w_out[batch_id, hv_index, start:end] = w_block.to(output_dtype)
            u_out[batch_id, hv_index, start:end] = u_block.to(output_dtype)
            qg_out[batch_id, hv_index, start:end] = qg_block.to(output_dtype)
            kg_out[batch_id, hv_index, start:end] = kg_block.to(output_dtype)
            v_new_out[batch_id, hv_index, start:end] = v_new_block.to(output_dtype)
            h_out[batch_id, chunk_id, hv_index] = previous.to(output_dtype)
        final_states[:, hv_index] = state

    with ThreadPoolExecutor(max_workers=min(_REFERENCE_WORKERS, hv_num)) as pool:
        list(pool.map(run_head, range(hv_num)))

    final_state = final_states if _as_bool(spec["output_final_state"]) else None
    if _as_bool(spec["state_v_first"]):
        final_state = None if final_state is None else final_state.transpose(-1, -2)
        h_out = h_out.transpose(-1, -2)

    rank3 = layout in {"TND", "NTD"}
    gk_out = gk.permute(0, 2, 1, 3)
    if rank3:
        o = o.squeeze(0)
        aqk = aqk.squeeze(0)
        akk = akk.squeeze(0)
        gk_out = gk_out.squeeze(0)
        w_out = w_out.squeeze(0)
        u_out = u_out.squeeze(0)
        qg_out = qg_out.squeeze(0)
        kg_out = kg_out.squeeze(0)
        v_new_out = v_new_out.squeeze(0)
        h_out = h_out.squeeze(0)
    return (
        o,
        final_state,
        gk_out,
        aqk,
        akk,
        w_out,
        u_out,
        qg_out,
        kg_out,
        v_new_out,
        h_out,
        inputs.initial_state,
    )


def _apply_output_policy(outputs: tuple, spec: dict) -> tuple:
    selected = list(outputs)
    if not _as_bool(spec["output_final_state"]):
        selected[1] = None
    if _as_bool(spec["use_gate_in_kernel"]) and not _as_bool(spec["disable_recompute"]):
        selected[2] = None
    if not _as_bool(spec["disable_recompute"]):
        selected[5:10] = [None] * 5
    if not (
        _as_bool(spec["disable_recompute"])
        or _as_bool(spec["return_intermediate_states"])
    ):
        selected[10] = None
    return tuple(selected)


def _reference_cache_key(inputs: _PreparedInputs, spec: dict, implementation: str) -> str:
    ignored = {
        "case_key",
        "disable_recompute",
        "optional_spec",
        "output_final_state",
        "profile",
        "return_intermediate_states",
        "route",
        "shape_spec",
        "soc",
        "tags",
    }
    payload = {key: value for key, value in spec.items() if key not in ignored}
    payload["reference_implementation"] = implementation
    payload["reference_q_dtype"] = str(inputs.q.dtype)
    payload["runtime_seed"] = inputs.legacy_seed
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _full_reference(inputs: _PreparedInputs, spec: dict, runner):
    # Always derive the reference from ATK's materialized tensors so saved-input
    # replays and accuracy_lt loops compare the values that were actually run.
    cache_key = None
    if inputs.legacy_seed is not None:
        cache_key = _reference_cache_key(inputs, spec, runner.__name__)
        outputs = _LEGACY_REFERENCE_CACHE.get(cache_key)
        if outputs is not None:
            _LEGACY_REFERENCE_CACHE.move_to_end(cache_key)
            return _apply_output_policy(outputs, spec)

    full_spec = dict(spec)
    full_spec["disable_recompute"] = True
    full_spec["output_final_state"] = True
    full_spec["return_intermediate_states"] = True
    outputs = runner(inputs, full_spec)
    if cache_key is not None:
        _LEGACY_REFERENCE_CACHE[cache_key] = outputs
        _LEGACY_REFERENCE_CACHE.move_to_end(cache_key)
        while len(_LEGACY_REFERENCE_CACHE) > max(0, _LEGACY_REFERENCE_CACHE_ENTRIES):
            _LEGACY_REFERENCE_CACHE.popitem(last=False)
    return _apply_output_policy(outputs, spec)


def _torch_fp64_golden(inputs: _PreparedInputs, spec: dict):
    if inputs.q.device.type not in {"cpu", "cuda"}:
        raise RuntimeError("Torch FP64 golden must run on an ATK CPU or GPU node")
    if any(
        tensor is not None and tensor.is_floating_point() and tensor.dtype != torch.float64
        for tensor in (
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            inputs.A_log,
            inputs.dt_bias,
            inputs.initial_state,
        )
    ):
        raise RuntimeError("Torch FP64 golden received a non-FP64 floating input")
    if inputs.q.device.type == "cpu":
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            return _full_reference(
                inputs,
                spec,
                _reference_model_parallel,
            )
        finally:
            torch.set_num_threads(previous_threads)
    return _full_reference(
        inputs,
        spec,
        _reference_impl,
    )


def _torch_same_precision(inputs: _PreparedInputs, spec: dict):
    if inputs.q.device.type != "cpu":
        raise RuntimeError("Torch same-precision control must run on an ATK CPU node")
    if inputs.q.dtype != _DTYPES[str(spec["q_dtype"])]:
        raise RuntimeError(
            "Torch same-precision control received an unexpected q dtype: "
            f"expected={_DTYPES[str(spec['q_dtype'])]}, actual={inputs.q.dtype}"
        )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _full_reference(
            inputs,
            spec,
            _reference_model_parallel,
        )
    finally:
        torch.set_num_threads(previous_threads)


def _load_triton_callable():
    module_name, _, attribute = _DEFAULT_TRITON_CALLABLE.partition(":")
    module = importlib.import_module(module_name)
    callable_obj = getattr(module, attribute, None)
    if not callable(callable_obj):
        raise RuntimeError(f"configured Triton target is not callable: {_DEFAULT_TRITON_CALLABLE}")
    return _DEFAULT_TRITON_CALLABLE, callable_obj


def _cuda_long(values: Optional[list[int]], device, *, pairs: bool = False):
    if values is None:
        return None
    tensor = torch.tensor(values, dtype=torch.int64, device=device)
    return tensor.reshape(-1, 2) if pairs else tensor


def _triton_gate_cumsum(
    inputs: _PreparedInputs,
    spec: dict,
    g: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    chunk_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    if _as_bool(spec["use_gate_in_kernel"]):
        from fla.ops.kda.gate import kda_gate_chunk_cumsum

        return kda_gate_chunk_cumsum(
            g=g,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            scale=1.0 / math.log(2.0),
            chunk_size=int(spec["chunk_size"]),
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=(
                float(spec["lower_bound"]) if _as_bool(spec["safe_gate"]) else None
            ),
        )

    from fla.ops.utils import chunk_local_cumsum

    return chunk_local_cumsum(
        g,
        scale=1.0 / math.log(2.0),
        chunk_size=int(spec["chunk_size"]),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def _sequence_major_to_head_major(tensor: Optional[torch.Tensor], spec: dict):
    if tensor is None or tensor.dim() != 4:
        return tensor
    total_t, hv_num = int(spec["T"]), int(spec["HV"])
    if tensor.shape[1] == total_t and tensor.shape[2] == hv_num:
        return tensor.permute(0, 2, 1, 3).contiguous()
    if tensor.shape[1] == hv_num and tensor.shape[2] == total_t:
        return tensor
    raise RuntimeError(
        "Triton intermediate has an unsupported layout: "
        f"shape={tuple(tensor.shape)}, expected BT(HV)D or B(HV)TD"
    )


def _normalize_triton_outputs(outputs: tuple, inputs: _PreparedInputs, spec: dict) -> tuple:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != len(_OUTPUT_NAMES):
        raise RuntimeError(
            f"Triton callable must return {len(_OUTPUT_NAMES)} values in chunk_kda_fwd order"
        )
    normalized = list(outputs)
    for index in range(2, 10):
        normalized[index] = _sequence_major_to_head_major(normalized[index], spec)
    normalized[11] = inputs.initial_state
    if str(spec["layout"]) in {"TND", "NTD"}:
        for index in (0, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            tensor = normalized[index]
            if isinstance(tensor, torch.Tensor) and tensor.dim() >= 1 and tensor.shape[0] == 1:
                normalized[index] = tensor.squeeze(0)
    return tuple(normalized)


def _zero_undefined_triton_aqk_regions(
    outputs: tuple, inputs: _PreparedInputs, spec: dict
) -> tuple:
    """Zero Aqk elements that the causal Triton kernel intentionally leaves unwritten."""
    normalized = list(outputs)
    total_t = int(spec["T"])
    chunk_size = int(spec["chunk_size"])
    if total_t < 0 or chunk_size <= 0:
        raise RuntimeError(
            f"invalid Aqk dimensions: T={total_t}, chunk_size={chunk_size}"
        )

    aqk = normalized[3]
    if not isinstance(aqk, torch.Tensor):
        return tuple(normalized)
    if aqk.dim() not in (3, 4):
        raise RuntimeError(f"Aqk has unsupported rank {aqk.dim()}")
    if tuple(aqk.shape[-2:]) != (total_t, chunk_size):
        raise RuntimeError(
            f"Aqk has unsupported shape {tuple(aqk.shape)}; expected trailing "
            f"dimensions ({total_t}, {chunk_size})"
        )

    spans = (
        [(0, total_t)]
        if inputs.cu_seqlens is None
        else [
            (int(start), int(end))
            for start, end in zip(inputs.cu_seqlens, inputs.cu_seqlens[1:])
        ]
    )
    if (
        not spans
        or spans[0][0] != 0
        or spans[-1][1] != total_t
        or any(
            start < 0
            or end < 0
            or start > end
            or start > total_t
            or end > total_t
            for start, end in spans
        )
        or any(left[1] != right[0] for left, right in zip(spans, spans[1:]))
    ):
        raise RuntimeError(f"invalid sequence spans for T={total_t}: {spans}")

    # Aqk stores one causal row per token and one column per position in its
    # current chunk. Packed sequences restart the chunk offset at each span.
    local_rows = torch.empty(total_t, dtype=torch.int64, device=aqk.device)
    for start, end in spans:
        local_rows[start:end] = torch.arange(
            end - start, dtype=torch.int64, device=aqk.device
        ).remainder(chunk_size)
    columns = torch.arange(chunk_size, dtype=torch.int64, device=aqk.device)
    valid = columns.unsqueeze(0) <= local_rows.unsqueeze(1)
    valid = valid.reshape((1,) * (aqk.dim() - 2) + valid.shape)

    with torch.no_grad():
        aqk.masked_fill_(~valid, 0)
    return tuple(normalized)


def _triton_same_precision_impl(inputs: _PreparedInputs, spec: dict):
    if inputs.q.device.type != "cuda":
        raise RuntimeError("Triton same-precision control must run on an ATK GPU node")
    target, triton_callable = _load_triton_callable()
    chunk_size = int(spec["chunk_size"])
    if chunk_size != 64:
        raise RuntimeError(
            "the GPU dual-benchmark matrix is scoped to chunk_size=64; "
            f"got chunk_size={chunk_size} from {target}"
        )

    layout = str(spec["layout"])
    q = _layout_to_bsnd(inputs.q, layout).contiguous()
    k = _layout_to_bsnd(inputs.k, layout).contiguous()
    v = _layout_to_bsnd(inputs.v, layout).contiguous()
    g = _layout_to_bsnd(inputs.g, layout).contiguous()
    beta = _layout_to_bsnd(inputs.beta, layout, beta=True).contiguous()
    cu_seqlens = _cuda_long(inputs.cu_seqlens, q.device)
    cu_seqlens_cpu = (
        None if inputs.cu_seqlens is None
        else torch.tensor(inputs.cu_seqlens, dtype=torch.int64, device="cpu")
    )
    chunk_indices = _cuda_long(inputs.chunk_indices, q.device, pairs=True)

    outputs = list(
        triton_callable(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=float(spec["scale"]),
            initial_state=inputs.initial_state,
            output_final_state=True,
            state_v_first=_as_bool(spec["state_v_first"]),
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            safe_gate=_as_bool(spec["safe_gate"]),
            lower_bound=(
                float(spec["lower_bound"]) if _as_bool(spec["safe_gate"]) else None
            ),
            use_gate_in_kernel=_as_bool(spec["use_gate_in_kernel"]),
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            disable_recompute=True,
            return_intermediate_states=True,
        )
    )
    if len(outputs) == len(_OUTPUT_NAMES) and outputs[2] is None:
        outputs[2] = _triton_gate_cumsum(
            inputs, spec, g, cu_seqlens, chunk_indices
        )
    outputs = _normalize_triton_outputs(tuple(outputs), inputs, spec)
    return _zero_undefined_triton_aqk_regions(outputs, inputs, spec)


def _triton_same_precision(inputs: _PreparedInputs, spec: dict):
    return _full_reference(inputs, spec, _triton_same_precision_impl)


def _public_kwargs(inputs: _PreparedInputs, spec: dict) -> dict:
    return {
        "layout": str(spec["layout"]),
        "initial_state": inputs.initial_state,
        "output_final_state": _as_bool(spec["output_final_state"]),
        "cu_seqlens": inputs.cu_seqlens,
        "chunk_indices": inputs.chunk_indices,
        "safe_gate": _as_bool(spec["safe_gate"]),
        "lower_bound": float(spec["lower_bound"]),
        "use_gate_in_kernel": _as_bool(spec["use_gate_in_kernel"]),
        "A_log": inputs.A_log,
        "dt_bias": inputs.dt_bias,
        "disable_recompute": _as_bool(spec["disable_recompute"]),
        "return_intermediate_states": _as_bool(spec["return_intermediate_states"]),
        "state_v_first": _as_bool(spec["state_v_first"]),
        "epsilon": float(spec.get("epsilon", 1e-6)),
        "use_qk_l2norm_in_kernel": _as_bool(spec["use_qk_l2norm_in_kernel"]),
        "use_beta_sigmoid_in_kernel": _as_bool(spec["use_beta_sigmoid_in_kernel"]),
        "allow_neg_eigval": _as_bool(spec.get("allow_neg_eigval", False)),
        "use_exp2": _as_bool(spec.get("use_exp2", True)),
    }


def _run_positive_npu(inputs: _PreparedInputs, spec: dict):
    from fla_npu.ops.ascendc import chunk_kda_fwd

    args = (
        inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta,
        float(spec["scale"]), int(spec["chunk_size"]),
    )
    return chunk_kda_fwd(*args, **_public_kwargs(inputs, spec))


@register("executor_chunk_kda_fwd")
class ChunkKdaFwdApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.task_result = task_result
        self.spec = None
        self.inputs = None
        case_config = getattr(task_result, "case_config", None)
        case_id = case_config.get("id") if isinstance(case_config, dict) else getattr(case_config, "id", None)
        self.runtime_case_id = None if case_id is None else int(case_id)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device in {"cpu", "gpu"} and self.is_benchmark_task
        self.cpu_control = self.device == "cpu" and not self.high_precision
        self.triton_control = self.device == "gpu" and not self.high_precision
        task_names = {
            str(getattr(task_type, "value", task_type))
            for task_type in (task_result.task_type or [])
        }
        self.randomize_legacy_values = bool(
            getattr(task_result, "disable_id_seed", False)
            and "accuracy_lt" in task_names
        )

    def init_by_input_data(self, input_data: InputDataset):
        values = input_data.kwargs
        case_id = 0 if self.runtime_case_id is None else self.runtime_case_id
        legacy_names = {"case_spec", "low_precision_marker", "fp32_marker"}
        if legacy_names.issubset(values):
            raw_spec = values["case_spec"]
            if not isinstance(raw_spec, str):
                raise TypeError("legacy case_spec must be a JSON string")
            self.spec = json.loads(raw_spec)
            if not isinstance(self.spec, dict):
                raise ValueError("legacy case_spec must decode to a JSON object")
            runtime_seed = int(self.spec["seed"])
            if self.randomize_legacy_values:
                if self.runtime_case_id is None:
                    raise RuntimeError(
                        "accuracy_lt with --disable_id_seed requires an ATK runtime case id"
                    )
                runtime_seed = (
                    runtime_seed * 0x9E3779B185EBCA87 + self.runtime_case_id
                ) % (2**63 - 1)
            self.inputs = _legacy_inputs(
                self.spec,
                values["low_precision_marker"],
                high_precision=self.high_precision,
                seed=runtime_seed,
            )
            input_schema = "legacy"
        elif "q" in values:
            q_config = self.task_result.case_config.get_input_data_config(name="q")
            g_config = self.task_result.case_config.get_input_data_config(name="g")
            beta_config = self.task_result.case_config.get_input_data_config(name="beta")
            q_dtype = str(getattr(q_config, "dtype", _dtype_name(values["q"].dtype)))
            g_dtype = str(getattr(g_config, "dtype", _dtype_name(values["g"].dtype)))
            beta_dtype = str(
                getattr(beta_config, "dtype", _dtype_name(values["beta"].dtype))
            )
            self.spec, cu_seqlens, chunk_indices = _derive_spec(
                values, q_dtype, g_dtype, beta_dtype, case_id
            )
            self.inputs = _direct_inputs(
                values,
                self.spec,
                cu_seqlens,
                chunk_indices,
                self.high_precision,
            )
            runtime_seed = case_id
            input_schema = "direct"
        else:
            raise ValueError("chunk_kda_fwd case must use direct inputs or legacy case_spec")
        if os.environ.get("KDA_ATK_TRACE_SEED") == "1":
            print(
                "KDA_ATK_RUNTIME_SEED",
                self.device,
                "benchmark=" + str(self.is_benchmark_task),
                "high_precision=" + str(self.high_precision),
                "cpu_control=" + str(self.cpu_control),
                "triton_control=" + str(self.triton_control),
                "schema=" + input_schema,
                "case_id=" + str(self.runtime_case_id),
                "seed=" + str(runtime_seed),
                flush=True,
            )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del input_data, with_output
        if self.high_precision:
            outputs = _torch_fp64_golden(self.inputs, self.spec)
        elif self.cpu_control:
            outputs = _torch_same_precision(self.inputs, self.spec)
        elif self.triton_control:
            outputs = _triton_same_precision(self.inputs, self.spec)
        elif self.device == "npu":
            outputs = _run_positive_npu(self.inputs, self.spec)
        else:
            raise RuntimeError(
                "positive chunk_kda_fwd cases require one NPU node and one CPU or GPU reference node; "
                f"got device={self.device!r}, benchmark={self.is_benchmark_task}"
            )
        selected_names = _selected_output_names()
        named_visible = tuple(
            (name, _published_output(output))
            for name, output in zip(_OUTPUT_NAMES, outputs)
            if isinstance(output, torch.Tensor)
            and (selected_names is None or name in selected_names)
        )
        visible = tuple(output for _, output in named_visible)
        if not visible:
            raise RuntimeError("chunk_kda_fwd returned no visible tensor outputs")
        for name, output in named_visible:
            if not torch.isfinite(output).all().item():
                if os.environ.get("KDA_ATK_ALLOW_NONFINITE_OUTPUT") == "1":
                    nonfinite_count = int((~torch.isfinite(output)).sum().item())
                    print(
                        "KDA_ATK_NONFINITE_OUTPUT",
                        f"case_id={self.runtime_case_id}",
                        f"device={self.device}",
                        f"output={name}",
                        f"count={nonfinite_count}",
                        flush=True,
                    )
                    continue
                raise RuntimeError(f"{name} contains NaN or Inf")
        return visible

    def export_custom_data(self, input_data: InputDataset):
        del input_data
        return {
            "case_key": str(self.spec["case_key"]),
            "layout": str(self.spec["layout"]),
            "shape": {
                "B": int(self.spec["B"]),
                "H": int(self.spec["H"]),
                "HV": int(self.spec["HV"]),
                "T": int(self.spec["T"]),
                "K": int(self.spec["K"]),
                "V": int(self.spec["V"]),
            },
            "chunk_size": int(self.spec["chunk_size"]),
        }
