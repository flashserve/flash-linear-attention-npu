"""ATK executor, dual references, and route adapters for chunk_kda_fwd."""

from __future__ import annotations

import ctypes
import importlib
import json
import math
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

try:
    from persistent_reference_cache import (
        CacheReader,
        CacheWriter,
        OUTPUT_NAMES,
        PinnedCatalog,
        ReferenceCacheError,
        build_chunk_kda_metadata,
        default_catalog_reference,
        default_cache_dir,
    )
except ModuleNotFoundError:
    from test.chunk_kda_fwd.persistent_reference_cache import (
        CacheReader,
        CacheWriter,
        OUTPUT_NAMES,
        PinnedCatalog,
        ReferenceCacheError,
        build_chunk_kda_metadata,
        default_catalog_reference,
        default_cache_dir,
    )


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_OUTPUT_NAMES = OUTPUT_NAMES
_REFERENCE_WORKERS = int(os.environ.get("KDA_ATK_REFERENCE_WORKERS", "48"))
_REFERENCE_CACHE: OrderedDict[str, tuple] = OrderedDict()
_PERSISTENT_CATALOGS: dict[tuple[str, str], PinnedCatalog] = {}
_REFERENCE_CACHE_ENTRIES = int(os.environ.get("KDA_ATK_REFERENCE_CACHE_ENTRIES", "0"))
_DEFAULT_TRITON_CALLABLE = "fla.ops.kda.chunk_fwd:chunk_kda_fwd"
_EXECUTOR_PATH = Path(__file__).resolve()
_PREPARED_TENSOR_NAMES = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "A_log",
    "dt_bias",
    "initial_state",
)


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
    seed: int


def _persistent_cache_mode() -> str:
    mode = os.environ.get("KDA_ATK_PERSISTENT_CACHE_MODE", "off").strip().lower()
    if mode not in {"off", "readonly"}:
        raise ReferenceCacheError(
            "KDA_ATK_PERSISTENT_CACHE_MODE must be 'off' or 'readonly'; "
            "build caches with build_reference_cache.py"
        )
    return mode


def _validate_persistent_cache_task(mode: str, task_names: set[str]) -> None:
    if "accuracy_lt" in task_names and mode != "off":
        raise ReferenceCacheError(
            "accuracy_lt must randomize numeric inputs and cannot reuse a fixed persistent cache; "
            "set KDA_ATK_PERSISTENT_CACHE_MODE=off"
        )


def _spec_requires_references(spec: dict) -> bool:
    tags = {tag.strip() for tag in str(spec.get("tags", "")).split(",")}
    if not tags or tags == {""}:
        return True
    return "accuracy" in tags and "negative" not in tags


def _persistent_cache_metadata(spec: dict, seed: int) -> dict:
    return build_chunk_kda_metadata(
        spec,
        seed,
        _EXECUTOR_PATH,
        include_references=_spec_requires_references(spec),
    )


def _persistent_cache_reader(spec: dict, seed: int) -> CacheReader:
    cache_dir = default_cache_dir()
    reference = default_catalog_reference()
    key = (str(cache_dir.expanduser().absolute()), str(reference))
    catalog = _PERSISTENT_CATALOGS.get(key)
    if catalog is None:
        catalog = PinnedCatalog(cache_dir, reference)
        _PERSISTENT_CATALOGS[key] = catalog
    return catalog.reader_for(
        spec,
        seed,
        _EXECUTOR_PATH,
        include_references=_spec_requires_references(spec),
    )


def _prepared_inputs_to_cpu(inputs: _PreparedInputs) -> dict:
    payload = {
        name: (
            None
            if getattr(inputs, name) is None
            else getattr(inputs, name).detach().to(device="cpu").contiguous()
        )
        for name in _PREPARED_TENSOR_NAMES
    }
    payload.update(
        {
            "cu_seqlens": None if inputs.cu_seqlens is None else list(inputs.cu_seqlens),
            "chunk_indices": None if inputs.chunk_indices is None else list(inputs.chunk_indices),
            "seed": int(inputs.seed),
        }
    )
    return payload


def _prepared_inputs_from_cpu(
    payload: dict,
    device,
    *,
    high_precision: bool = False,
) -> _PreparedInputs:
    required = set(_PREPARED_TENSOR_NAMES) | {"cu_seqlens", "chunk_indices", "seed"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ReferenceCacheError("cached input shard has an invalid field set")

    tensors = {}
    for name in _PREPARED_TENSOR_NAMES:
        value = payload[name]
        if value is not None and not isinstance(value, torch.Tensor):
            raise ReferenceCacheError(f"cached input {name} is not a tensor")
        if value is not None and value.device.type != "cpu":
            raise ReferenceCacheError(f"cached input {name} is not on CPU")
        if value is not None and high_precision and value.is_floating_point():
            value = value.to(torch.float64)
        tensors[name] = None if value is None else value.to(device=device)
    return _PreparedInputs(
        **tensors,
        cu_seqlens=(
            None if payload["cu_seqlens"] is None else list(payload["cu_seqlens"])
        ),
        chunk_indices=(
            None if payload["chunk_indices"] is None else list(payload["chunk_indices"])
        ),
        seed=int(payload["seed"]),
    )


def _select_cached_input_payload(payload: dict, spec: dict) -> dict:
    if payload.get("schema") != "chunk_kda_fwd.canonical_input_variants.v1":
        return payload
    aliases = payload.get("aliases")
    variants = payload.get("variants")
    if not isinstance(aliases, dict) or not isinstance(variants, dict):
        raise ReferenceCacheError("cached canonical input bundle is malformed")
    requested = str(spec.get("materialized_variant", ""))
    primary = aliases.get(requested)
    if primary is None or primary not in variants:
        raise ReferenceCacheError(
            f"cached canonical input bundle has no variant {requested!r}"
        )
    return variants[primary]


def _outputs_to_cpu(outputs: tuple) -> tuple:
    if not isinstance(outputs, tuple) or len(outputs) != len(_OUTPUT_NAMES):
        raise ReferenceCacheError(
            f"reference output must be a {len(_OUTPUT_NAMES)}-element tuple"
        )
    return tuple(
        None if output is None else output.detach().to(device="cpu").contiguous()
        for output in outputs
    )


def _outputs_from_cpu(outputs: tuple, device) -> tuple:
    if not isinstance(outputs, tuple) or len(outputs) != len(_OUTPUT_NAMES):
        raise ReferenceCacheError(
            f"cached reference must be a {len(_OUTPUT_NAMES)}-element tuple"
        )
    if any(
        output is not None
        and (not isinstance(output, torch.Tensor) or output.device.type != "cpu")
        for output in outputs
    ):
        raise ReferenceCacheError("cached reference contains a non-CPU tensor value")
    return tuple(None if output is None else output.to(device=device) for output in outputs)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_cu(value) -> Optional[list[int]]:
    value = str(value or "").strip()
    if not value:
        return None
    return [int(item) for item in value.split(",")]


def _canonical_chunk_indices(cu_seqlens: Optional[list[int]], chunk_size: int) -> Optional[list[int]]:
    if cu_seqlens is None:
        return None
    indices = []
    for seq_id, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for chunk_id in range((end - start + chunk_size - 1) // chunk_size):
            indices.extend((seq_id, chunk_id))
    return indices


def _target_dtype(original: str, high_precision: bool) -> torch.dtype:
    if not high_precision:
        return _DTYPES[original]
    return torch.float64


def _random_quantized(
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


def _normal_quantized(
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


def _same_shape_noncontiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a tensor into a strided view without changing its logical values."""
    storage = torch.empty((*tensor.shape, 2), dtype=tensor.dtype, device=tensor.device)
    view = storage[..., 0]
    view.copy_(tensor)
    if tensor.numel() > 1 and view.is_contiguous():
        raise RuntimeError("failed to construct a non-contiguous input view")
    return view


def _apply_input_storage(inputs: _PreparedInputs, spec: dict) -> _PreparedInputs:
    selected = spec.get("input_storage", [])
    if isinstance(selected, str):
        selected = [name.strip() for name in selected.split(",") if name.strip()]
    selected = list(selected)
    unknown = set(selected).difference({"q", "k", "v", "g", "beta"})
    if unknown:
        raise ValueError(f"unsupported non-contiguous inputs: {sorted(unknown)}")
    for name in selected:
        setattr(inputs, name, _same_shape_noncontiguous(getattr(inputs, name)))
    return inputs


def _layout_from_bsnd(tensor: torch.Tensor, layout: str, *, beta: bool = False) -> torch.Tensor:
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


def _inject_gva_traceable_values(
    q_bsnd: torch.Tensor,
    k_bsnd: torch.Tensor,
    v_bsnd: torch.Tensor,
    g_bsnd: torch.Tensor,
    beta_bsnd: torch.Tensor,
    cu_seqlens: Optional[list[int]],
    chunk_size: int,
) -> None:
    h_num = q_bsnd.shape[2]
    hv_num = v_bsnd.shape[2]
    q_codes = torch.linspace(-0.03, 0.03, h_num, dtype=torch.float32)
    hv_codes = torch.linspace(-0.04, 0.04, hv_num, dtype=torch.float32)

    if cu_seqlens is None:
        spans = [(batch_id, 0, q_bsnd.shape[1]) for batch_id in range(q_bsnd.shape[0])]
    else:
        spans = [(0, start, end) for start, end in zip(cu_seqlens, cu_seqlens[1:])]

    ordinal = 0
    for batch_id, start, end in spans:
        for token in range(start, end, chunk_size):
            factor = 1.0 + 0.0625 * (ordinal % 7)
            q_value = (q_codes * factor).to(device=q_bsnd.device, dtype=q_bsnd.dtype)
            hv_value = (hv_codes * factor).to(device=v_bsnd.device, dtype=v_bsnd.dtype)
            q_bsnd[batch_id, token, :, 0] = q_value
            k_bsnd[batch_id, token, :, 0] = q_value.flip(0).to(k_bsnd.dtype)
            v_bsnd[batch_id, token, :, 0] = hv_value
            g_bsnd[batch_id, token, :, 0] = hv_value.to(g_bsnd.dtype)
            beta_bsnd[batch_id, token, :] = torch.sigmoid(
                hv_codes.to(device=beta_bsnd.device) * factor
            ).to(beta_bsnd.dtype)
            ordinal += 1


def _prepare_inputs(
    spec: dict,
    low_marker: torch.Tensor,
    fp32_marker: torch.Tensor,
    *,
    high_precision: bool = False,
    seed: Optional[int] = None,
) -> _PreparedInputs:
    device = low_marker.device
    del fp32_marker
    generator = torch.Generator(device="cpu")
    runtime_seed = int(spec["seed"] if seed is None else seed)
    generator.manual_seed(runtime_seed)

    batch = int(spec["B"])
    total_t = int(spec["T"])
    h_num = int(spec["H"])
    hv_num = int(spec["HV"])
    k_dim = int(spec["K"])
    v_dim = int(spec["V"])
    layout = str(spec["layout"])
    data_profile = str(spec.get("data_profile", "uniform"))

    q_original = _DTYPES[str(spec["q_dtype"])]
    q_target = _target_dtype(str(spec["q_dtype"]), high_precision)
    g_original = _DTYPES[str(spec["g_dtype"])]
    g_target = _target_dtype(str(spec["g_dtype"]), high_precision)
    beta_original = _DTYPES[str(spec["beta_dtype"])]
    beta_target = _target_dtype(str(spec["beta_dtype"]), high_precision)
    a_log_original = _DTYPES[str(spec.get("a_log_dtype", "fp32"))]
    a_log_target = _target_dtype(str(spec.get("a_log_dtype", "fp32")), high_precision)
    dt_bias_original = _DTYPES[str(spec.get("dt_bias_dtype", "fp32"))]
    dt_bias_target = _target_dtype(str(spec.get("dt_bias_dtype", "fp32")), high_precision)
    fp32_target = torch.float64 if high_precision else torch.float32

    if data_profile == "model_h96":
        q_bsnd = _normal_quantized(
            (batch, total_t, h_num, k_dim), generator, q_original, q_target, device,
            std=float(spec["qk_scale"]), l2_normalize=True,
        )
        k_bsnd = _normal_quantized(
            (batch, total_t, h_num, k_dim), generator, q_original, q_target, device,
            std=float(spec["qk_scale"]), l2_normalize=True,
        )
        v_bsnd = _normal_quantized(
            (batch, total_t, hv_num, v_dim), generator, q_original, q_target, device,
            std=float(spec["v_scale"]),
        )
        g_bsnd = _normal_quantized(
            (batch, total_t, hv_num, k_dim), generator, g_original, g_target, device,
            std=float(spec["gate_scale"]),
        )
        beta_bsnd = _normal_quantized(
            (batch, total_t, hv_num), generator, beta_original, beta_target, device,
            mean=float(spec["beta_bias"]), std=float(spec["beta_scale"]), sigmoid=True,
        )
    else:
        data_scale = float(spec["data_scale"])
        q_bsnd = _random_quantized(
            (batch, total_t, h_num, k_dim), generator, q_original, q_target, device,
            low=-data_scale, high=data_scale,
        )
        k_bsnd = _random_quantized(
            (batch, total_t, h_num, k_dim), generator, q_original, q_target, device,
            low=-data_scale, high=data_scale,
        )
        v_bsnd = _random_quantized(
            (batch, total_t, hv_num, v_dim), generator, q_original, q_target, device,
            low=-data_scale, high=data_scale,
        )

        gate_scale = float(spec["gate_scale"])
        if _as_bool(spec["use_gate_in_kernel"]):
            gate_low, gate_high = -gate_scale, gate_scale
        else:
            gate_low, gate_high = -0.02 * gate_scale, -0.002 * gate_scale
        g_bsnd = _random_quantized(
            (batch, total_t, hv_num, k_dim), generator, g_original, g_target, device,
            low=gate_low, high=gate_high,
        )
        beta_bsnd = _random_quantized(
            (batch, total_t, hv_num), generator, beta_original, beta_target, device,
            low=0.0, high=1.0,
        )

    A_log = None
    if _as_bool(spec["use_gate_in_kernel"]):
        if data_profile == "model_h96":
            A_log = _normal_quantized(
                (hv_num,), generator, a_log_original, a_log_target, device,
                std=float(spec["a_log_scale"]),
            )
        else:
            A_log = _random_quantized(
                (hv_num,), generator, a_log_original, a_log_target, device,
                low=-6.0, high=-2.0,
            )
    dt_bias = None
    if _as_bool(spec["dt_bias"]):
        if data_profile == "model_h96":
            dt_bias = _normal_quantized(
                (hv_num * k_dim,), generator, dt_bias_original, dt_bias_target, device,
                mean=float(spec["dt_bias_mean"]), std=float(spec["dt_bias_scale"]),
            )
        else:
            dt_bias = _random_quantized(
                (hv_num * k_dim,), generator, dt_bias_original, dt_bias_target, device,
                low=-2.0, high=2.0,
            )

    cu_seqlens = _parse_cu(spec.get("cu_seqlens"))
    seq_num = len(cu_seqlens) - 1 if cu_seqlens is not None else batch
    initial_state = None
    if _as_bool(spec["initial_state"]):
        state_shape = (
            (seq_num, hv_num, v_dim, k_dim)
            if _as_bool(spec["state_v_first"])
            else (seq_num, hv_num, k_dim, v_dim)
        )
        initial_state = _random_quantized(
            state_shape, generator, torch.float32, fp32_target, device, low=-0.02, high=0.02,
        )

    data_variant = str(spec.get("data_variant", "random"))
    traceable_head_mapping = _as_bool(spec.get("traceable_head_mapping", False))
    if data_variant == "head_distinct_a_log":
        values = torch.linspace(-6.0, -2.0, hv_num, dtype=torch.float32)
        A_log = values.to(a_log_original).to(a_log_target).to(device)
    elif data_variant == "head_distinct_dt_bias":
        head_values = torch.linspace(-1.5, 1.5, hv_num, dtype=torch.float32).unsqueeze(1)
        dim_values = torch.linspace(-0.2, 0.2, k_dim, dtype=torch.float32).unsqueeze(0)
        values = (head_values + dim_values).reshape(-1)
        dt_bias = values.to(dt_bias_original).to(dt_bias_target).to(device)
    elif data_variant.startswith("initial_state_pulse_hv_"):
        if initial_state is None:
            raise ValueError(f"{data_variant} requires initial_state")
        pulse_head = int(data_variant.rsplit("_", 1)[-1])
        if pulse_head < 0 or pulse_head >= hv_num:
            raise ValueError(f"{data_variant} is outside HV={hv_num}")
        initial_state.zero_()
        initial_state[:, pulse_head, 0, 0] = 1.0
    elif data_variant == "gva_head_traceable":
        traceable_head_mapping = True
    elif data_variant != "random":
        raise ValueError(f"unsupported data_variant: {data_variant}")

    if traceable_head_mapping:
        _inject_gva_traceable_values(
            q_bsnd,
            k_bsnd,
            v_bsnd,
            g_bsnd,
            beta_bsnd,
            cu_seqlens,
            int(spec["chunk_size"]),
        )

    chunk_indices = None
    if _as_bool(spec["explicit_chunk_indices"]):
        chunk_indices = _canonical_chunk_indices(cu_seqlens, int(spec["chunk_size"]))
    return _apply_input_storage(_PreparedInputs(
        q=_layout_from_bsnd(q_bsnd, layout),
        k=_layout_from_bsnd(k_bsnd, layout),
        v=_layout_from_bsnd(v_bsnd, layout),
        g=_layout_from_bsnd(g_bsnd, layout),
        beta=_layout_from_bsnd(beta_bsnd, layout, beta=True),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        seed=runtime_seed,
    ), spec)


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
        q_left = q[:, row_start:row_end] * left_factor
        k_left = k[:, row_start:row_end] * left_factor
        if operand_dtype is not None:
            right = right.to(operand_dtype).to(q.dtype)
            q_left = q_left.to(operand_dtype).to(q.dtype)
            k_left = k_left.to(operand_dtype).to(q.dtype)
        q_scores = torch.bmm(
            q_left,
            right.transpose(1, 2),
        ).mul(float(scale))
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
    elif cu is not None:
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
    gk = _gate_cumsum(inputs, spec)
    compute_dtype = torch.float64 if any(
        tensor is not None and tensor.dtype == torch.float64
        for tensor in (q, k, v, beta, inputs.A_log, inputs.dt_bias)
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
    expose_gk = not _as_bool(spec["use_gate_in_kernel"]) or _as_bool(spec["disable_recompute"])
    export_full = _as_bool(spec["disable_recompute"])
    export_h = export_full or _as_bool(spec["return_intermediate_states"])
    w_out = torch.zeros((batch, hv_num, total_t, k_dim), dtype=output_dtype, device=q.device) if export_full else None
    u_out = torch.zeros((batch, hv_num, total_t, v_dim), dtype=output_dtype, device=q.device) if export_full else None
    qg_out = torch.zeros_like(w_out) if export_full else None
    kg_out = torch.zeros_like(w_out) if export_full else None
    v_new_out = torch.zeros_like(u_out) if export_full else None
    h_out = (
        torch.zeros(
            (batch, total_chunks, hv_num, k_dim, v_dim),
            dtype=output_dtype,
            device=q.device,
        )
        if export_h
        else None
    )
    if inputs.initial_state is None:
        initial_states = torch.zeros(
            (seq_num, hv_num, k_dim, v_dim), dtype=compute_dtype, device=q.device
        )
    else:
        initial_states = inputs.initial_state.to(compute_dtype)
        if _as_bool(spec["state_v_first"]):
            initial_states = initial_states.transpose(-1, -2)
        initial_states = initial_states.clone()
    final_states = torch.empty_like(initial_states)

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
        state = initial_states[:, hv_index].clone()
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
            if export_full:
                w_out[batch_id, hv_index, start:end] = w_block.to(output_dtype)
                u_out[batch_id, hv_index, start:end] = u_block.to(output_dtype)
                qg_out[batch_id, hv_index, start:end] = qg_block.to(output_dtype)
                kg_out[batch_id, hv_index, start:end] = kg_block.to(output_dtype)
                v_new_out[batch_id, hv_index, start:end] = v_new_block.to(output_dtype)
            if export_h:
                h_out[batch_id, chunk_id, hv_index] = previous.to(output_dtype)
        final_states[:, hv_index] = state

    if q.device.type == "cuda":
        for hv_index in range(hv_num):
            run_head(hv_index)
    else:
        with ThreadPoolExecutor(max_workers=min(_REFERENCE_WORKERS, hv_num)) as pool:
            list(pool.map(run_head, range(hv_num)))

    final_state = final_states if _as_bool(spec["output_final_state"]) else None
    if _as_bool(spec["state_v_first"]):
        final_state = None if final_state is None else final_state.transpose(-1, -2)
        h_out = None if h_out is None else h_out.transpose(-1, -2)

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
    elif cu is not None:
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
    payload["runtime_seed"] = inputs.seed
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _cached_full_reference(inputs: _PreparedInputs, spec: dict, implementation: str, runner):
    cache_key = _reference_cache_key(inputs, spec, implementation)
    outputs = _REFERENCE_CACHE.get(cache_key)
    if outputs is None:
        while len(_REFERENCE_CACHE) >= max(0, _REFERENCE_CACHE_ENTRIES):
            if not _REFERENCE_CACHE:
                break
            _REFERENCE_CACHE.popitem(last=False)
        full_spec = dict(spec)
        full_spec["disable_recompute"] = True
        full_spec["output_final_state"] = True
        full_spec["return_intermediate_states"] = True
        outputs = runner(inputs, full_spec)
        if _REFERENCE_CACHE_ENTRIES > 0:
            _REFERENCE_CACHE[cache_key] = outputs
            _REFERENCE_CACHE.move_to_end(cache_key)
    else:
        _REFERENCE_CACHE.move_to_end(cache_key)
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
    return _cached_full_reference(
        inputs,
        spec,
        f"torch_{inputs.q.device.type}_fp64",
        _reference_impl,
    )


def _torch_same_precision(inputs: _PreparedInputs, spec: dict):
    if inputs.q.device.type not in {"cpu", "cuda"}:
        raise RuntimeError(
            "Torch same-precision reference must run on an ATK CPU or GPU node"
        )
    if inputs.q.dtype == torch.float64:
        raise RuntimeError("Torch same-precision reference received FP64 q input")
    return _cached_full_reference(
        inputs,
        spec,
        f"torch_{inputs.q.device.type}_same_precision",
        _reference_model_parallel,
    )


def build_persistent_reference_cache(
    spec: dict,
    cache_dir: Path,
    *,
    seed: Optional[int] = None,
    overwrite: bool = False,
    include_references: Optional[bool] = None,
) -> Path:
    """Build one deterministic input entry, optionally with both CPU references."""
    runtime_seed = int(spec["seed"] if seed is None else seed)
    if include_references is None:
        include_references = _spec_requires_references(spec)
    metadata = build_chunk_kda_metadata(
        spec,
        runtime_seed,
        _EXECUTOR_PATH,
        include_references=include_references,
    )
    marker = torch.empty(0, device="cpu")

    with CacheWriter(cache_dir, metadata, overwrite=overwrite) as writer:
        if include_references:
            base_inputs = _prepare_inputs(spec, marker, marker, seed=runtime_seed)
            serialized_inputs = _prepared_inputs_to_cpu(base_inputs)
        else:
            try:
                from canonical_execution_adapter import materialize_input_variants
            except ModuleNotFoundError:
                from test.chunk_kda_fwd.canonical_execution_adapter import (
                    materialize_input_variants,
                )
            input_plan = materialize_input_variants(spec)
            serialized_inputs = {
                "schema": input_plan["schema"],
                "aliases": input_plan["aliases"],
                "variants": {
                    variant: _prepared_inputs_to_cpu(
                        _prepare_inputs(
                            variant_spec,
                            marker,
                            marker,
                            seed=int(variant_spec["seed"]),
                        )
                    )
                    for variant, variant_spec in input_plan["variant_specs"].items()
                },
            }
        writer.write_shard("inputs", serialized_inputs)

        if not include_references:
            return writer.commit()

        fp64_inputs = _prepared_inputs_from_cpu(
            serialized_inputs, torch.device("cpu"), high_precision=True
        )
        fp64_outputs = _reference_impl(fp64_inputs, spec)
        writer.write_shard("cpu_fp64", _outputs_to_cpu(fp64_outputs))
        del fp64_outputs, fp64_inputs

        same_precision_outputs = _reference_model_parallel(base_inputs, spec)
        writer.write_shard(
            "cpu_same_precision", _outputs_to_cpu(same_precision_outputs)
        )
        del same_precision_outputs
        return writer.commit()


def _load_triton_callable():
    target = os.environ.get("KDA_ATK_TRITON_CALLABLE", _DEFAULT_TRITON_CALLABLE).strip()
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(
            "KDA_ATK_TRITON_CALLABLE must use '<python_module>:<callable>' syntax"
        )
    module = importlib.import_module(module_name)
    callable_obj = getattr(module, attribute, None)
    if not callable(callable_obj):
        raise RuntimeError(f"configured Triton target is not callable: {target}")
    return target, callable_obj


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


def _zero_undefined_triton_triangular_regions(
    outputs: tuple, inputs: _PreparedInputs, spec: dict
) -> tuple:
    """Normalize storage that the upstream Triton kernel intentionally leaves unwritten."""
    normalized = list(outputs)
    total_t = int(spec["T"])
    chunk_size = int(spec["chunk_size"])
    spans = (
        [(0, total_t)]
        if inputs.cu_seqlens is None
        else list(zip(inputs.cu_seqlens, inputs.cu_seqlens[1:]))
    )
    if (
        not spans
        or spans[0][0] != 0
        or spans[-1][1] != total_t
        or any(start < 0 or start > end for start, end in spans)
        or any(left[1] != right[0] for left, right in zip(spans, spans[1:]))
    ):
        raise RuntimeError(
            f"invalid sequence spans for T={total_t}: {spans}"
        )

    local_rows = torch.empty(total_t, dtype=torch.int64)
    for start, end in spans:
        local_rows[start:end] = torch.arange(end - start).remainder(chunk_size)
    columns = torch.arange(chunk_size, dtype=torch.int64)
    valid_cpu = columns.unsqueeze(0) <= local_rows.unsqueeze(1)

    for index in (3, 4):
        tensor = normalized[index]
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() not in (3, 4):
            raise RuntimeError(
                f"{_OUTPUT_NAMES[index]} has unsupported rank {tensor.dim()}"
            )
        if tensor.shape[-2:] != (total_t, chunk_size):
            raise RuntimeError(
                f"{_OUTPUT_NAMES[index]} has unsupported shape {tuple(tensor.shape)}; "
                f"expected trailing dimensions ({total_t}, {chunk_size})"
            )

        valid = valid_cpu.to(device=tensor.device)
        valid = valid.reshape((1,) * (tensor.dim() - 2) + valid.shape)
        tensor.masked_fill_(~valid, 0)
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
    return _zero_undefined_triton_triangular_regions(outputs, inputs, spec)


def _triton_same_precision(inputs: _PreparedInputs, spec: dict):
    return _cached_full_reference(
        inputs, spec, "cuda_triton_same_precision", _triton_same_precision_impl
    )


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
    }


def _run_positive_npu(inputs: _PreparedInputs, spec: dict):
    route = str(spec["route"])
    args = (
        inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta,
        float(spec["scale"]), int(spec["chunk_size"]),
    )
    if route == "ascendc":
        from fla_npu.ops.ascendc import chunk_kda_fwd

        return chunk_kda_fwd(*args, **_public_kwargs(inputs, spec))
    if route == "aclnn":
        from fla_npu.ops.ascendc._aclnn_ctypes import npu_chunk_kda_fwd

        return npu_chunk_kda_fwd(*args, **_public_kwargs(inputs, spec))
    if route == "direct_launch":
        import ascend_ops  # noqa: F401

        gk = _gate_cumsum(inputs, spec).permute(0, 2, 1, 3).contiguous()
        beta = inputs.beta.to(torch.float32).contiguous()
        outputs = list(torch.ops.ascend_ops.chunk_kda_fwd_direct(
            inputs.q,
            inputs.k,
            inputs.v,
            gk,
            beta,
            float(spec["scale"]),
            int(spec["chunk_size"]),
            initial_state=inputs.initial_state,
            output_final_state=_as_bool(spec["output_final_state"]),
            safe_gate=_as_bool(spec["safe_gate"]),
        ))
        outputs[10] = outputs[10].permute(0, 2, 1, 3, 4).contiguous()
        return _apply_output_policy(tuple(outputs) + (inputs.initial_state,), spec)
    raise ValueError(f"unsupported route: {route}")


def _recent_aclnn_error() -> str:
    try:
        library = ctypes.CDLL("libascendcl.so")
        function = library.aclGetRecentErrMsg
        function.argtypes = []
        function.restype = ctypes.c_char_p
        value = function()
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"unable to query aclGetRecentErrMsg: {exc}") from exc
    return "" if value is None else value.decode("utf-8", errors="replace")


def _head_axis(layout: str) -> int:
    return 2 if layout == "BSND" else (1 if layout in {"BNSD", "TND"} else 0)


def _resize_axis(tensor: torch.Tensor, axis: int, size: int) -> torch.Tensor:
    if size <= tensor.shape[axis]:
        return tensor.narrow(axis, 0, size).contiguous()
    repeats = [1] * tensor.dim()
    repeats[axis] = (size + tensor.shape[axis] - 1) // tensor.shape[axis]
    return tensor.repeat(*repeats).narrow(axis, 0, size).contiguous()


def _mutate_raw(inputs: _PreparedInputs, spec: dict, outputs: list):
    mutation = str(spec["mutation"])
    layout = str(spec["layout"])
    values = {
        "q": inputs.q,
        "k": inputs.k,
        "v": inputs.v,
        "g": inputs.g,
        "beta": inputs.beta,
        "A_log": inputs.A_log,
        "dt_bias": inputs.dt_bias,
        "initial_state": inputs.initial_state,
        "cu": inputs.cu_seqlens,
        "indices": inputs.chunk_indices,
        "layout": layout,
        "chunk_size": int(spec["chunk_size"]),
        "lower_bound": float(spec["lower_bound"]),
    }
    null_inputs = {"null_q": "q", "null_k": "k", "null_v": "v", "null_g": "g", "null_beta": "beta"}
    if mutation in null_inputs:
        values[null_inputs[mutation]] = None
    elif mutation == "null_attn":
        outputs[0] = None
    elif mutation == "null_aqk":
        outputs[3] = None
    elif mutation == "null_akk":
        outputs[4] = None
    elif mutation == "layout_null":
        values["layout"] = None
    elif mutation == "missing_alog":
        values["A_log"] = None
    elif mutation.startswith("chunk_"):
        values["chunk_size"] = {
            "chunk_invalid": 32,
            "chunk_zero": 0,
            "chunk_32": 32,
            "chunk_96": 96,
        }[mutation]
    elif mutation == "layout_lower":
        values["layout"] = layout.lower()
    elif mutation == "layout_invalid":
        values["layout"] = "INVALID"
    elif mutation == "rank_invalid":
        values["q"] = inputs.q.unsqueeze(0) if inputs.q.dim() == 3 else inputs.q.squeeze(0)
    elif mutation == "rank_bsnd_inputs_rank3":
        for name in ("q", "k", "v", "g", "beta"):
            values[name] = getattr(inputs, name).squeeze(0).contiguous()
    elif mutation == "rank_tnd_inputs_rank4":
        for name in ("q", "k", "v", "g", "beta"):
            values[name] = getattr(inputs, name).unsqueeze(0).contiguous()
    elif mutation == "beta_rank_invalid":
        values["beta"] = inputs.beta.unsqueeze(0).contiguous()
    elif mutation == "qk_shape":
        values["k"] = inputs.k[..., :-16].contiguous()
    elif mutation == "v_shape":
        values["v"] = inputs.v[..., :-16].contiguous()
    elif mutation == "g_shape":
        values["g"] = inputs.g[..., :-16].contiguous()
    elif mutation == "beta_shape":
        values["beta"] = inputs.beta.narrow(_head_axis(layout), 0, max(1, int(spec["HV"]) - 1)).contiguous()
    elif mutation in {
        "v_batch_mismatch", "v_token_mismatch", "v_head_mismatch",
        "g_batch_mismatch", "g_token_mismatch", "g_head_mismatch",
        "g_key_dim_mismatch", "beta_batch_mismatch", "beta_token_mismatch",
        "beta_head_mismatch",
    }:
        axis_sizes = {
            "v_batch_mismatch": ("v", 0, 2),
            "v_token_mismatch": ("v", 1, int(spec["T"]) - 1),
            "v_head_mismatch": ("v", 2, int(spec["HV"]) - 1),
            "g_batch_mismatch": ("g", 0, 2),
            "g_token_mismatch": ("g", 1, int(spec["T"]) - 1),
            "g_head_mismatch": ("g", 2, int(spec["HV"]) - 1),
            "g_key_dim_mismatch": ("g", 3, int(spec["K"]) - 16),
            "beta_batch_mismatch": ("beta", 0, 2),
            "beta_token_mismatch": ("beta", 1, int(spec["T"]) - 1),
            "beta_head_mismatch": ("beta", 2, int(spec["HV"]) - 1),
        }
        name, axis, size = axis_sizes[mutation]
        values[name] = _resize_axis(getattr(inputs, name), axis, size)
    elif mutation == "tnd_token_mismatch":
        values["v"] = _resize_axis(inputs.v, 0, int(spec["T"]) - 1)
    elif mutation == "ntd_head_mismatch":
        values["v"] = _resize_axis(inputs.v, 0, int(spec["HV"]) - 1)
    elif mutation == "bnsd_shape_mismatch":
        values["beta"] = _resize_axis(inputs.beta, 2, int(spec["T"]) - 1)
    elif mutation in {
        "h_zero", "hv_zero", "hv_lt_h", "hv_lt_h_4_2", "hv_not_divisible",
        "hv_not_divisible_3_8",
        "h_gt_128", "hv_gt_128",
    }:
        head_sizes = {
            "h_zero": (0, 4),
            "hv_zero": (4, 0),
            "hv_lt_h": (2, 1),
            "hv_lt_h_4_2": (4, 2),
            "hv_not_divisible": (3, 4),
            "hv_not_divisible_3_8": (3, 8),
            "h_gt_128": (129, 129),
            "hv_gt_128": (1, 129),
        }
        h_num, hv_num = head_sizes[mutation]
        head_axis = _head_axis(layout)
        values["q"] = _resize_axis(inputs.q, head_axis, h_num)
        values["k"] = _resize_axis(inputs.k, head_axis, h_num)
        values["v"] = _resize_axis(inputs.v, head_axis, hv_num)
        values["g"] = _resize_axis(inputs.g, head_axis, hv_num)
        values["beta"] = _resize_axis(inputs.beta, head_axis, hv_num)
    elif mutation in {
        "k_lt_16", "k_gt_256", "k_unaligned",
        "k_zero", "k_15", "k_17", "k_272",
    }:
        k_size = {
            "k_lt_16": 8, "k_gt_256": 272, "k_unaligned": 24,
            "k_zero": 0, "k_15": 15, "k_17": 17, "k_272": 272,
        }[mutation]
        values["q"] = _resize_axis(inputs.q, inputs.q.dim() - 1, k_size)
        values["k"] = _resize_axis(inputs.k, inputs.k.dim() - 1, k_size)
        values["g"] = _resize_axis(inputs.g, inputs.g.dim() - 1, k_size)
    elif mutation in {
        "v_lt_16", "v_gt_256", "v_unaligned",
        "v_zero", "v_15", "v_17", "v_272",
    }:
        v_size = {
            "v_lt_16": 8, "v_gt_256": 272, "v_unaligned": 24,
            "v_zero": 0, "v_15": 15, "v_17": 17, "v_272": 272,
        }[mutation]
        values["v"] = _resize_axis(inputs.v, inputs.v.dim() - 1, v_size)
    elif mutation == "q_fp32":
        values["q"] = inputs.q.float()
    elif mutation == "qkv_fp32":
        for name in ("q", "k", "v"):
            values[name] = getattr(inputs, name).float()
    elif mutation == "k_dtype":
        values["k"] = inputs.k.half()
    elif mutation == "v_dtype":
        values["v"] = inputs.v.half()
    elif mutation == "g_fp16":
        values["g"] = inputs.g.half()
    elif mutation == "g_int32":
        values["g"] = inputs.g.to(torch.int32)
    elif mutation == "beta_fp16":
        values["beta"] = inputs.beta.half()
    elif mutation == "beta_int32":
        values["beta"] = inputs.beta.to(torch.int32)
    elif mutation == "alog_dtype":
        values["A_log"] = inputs.A_log.half()
    elif mutation == "dtbias_dtype":
        values["dt_bias"] = inputs.dt_bias.half()
    elif mutation == "state_dtype":
        values["initial_state"] = inputs.initial_state.to(torch.bfloat16)
    elif mutation in {
        "attn_dtype", "gk_dtype", "aqk_dtype", "akk_dtype", "w_dtype",
        "u_dtype", "final_dtype", "h_dtype",
    }:
        output_index = {
            "attn_dtype": 0,
            "final_dtype": 1,
            "gk_dtype": 2,
            "aqk_dtype": 3,
            "akk_dtype": 4,
            "w_dtype": 5,
            "u_dtype": 6,
            "h_dtype": 10,
        }[mutation]
        if mutation in {"gk_dtype", "final_dtype"}:
            outputs[output_index] = outputs[output_index].to(inputs.q.dtype)
        else:
            outputs[output_index] = outputs[output_index].float()
    elif mutation == "cu_short":
        values["cu"] = [0]
    elif mutation == "cu_start":
        values["cu"] = [1, int(spec["T"])]
    elif mutation == "cu_end":
        values["cu"] = [0, int(spec["T"]) - 1]
    elif mutation == "cu_order":
        values["cu"] = [0, 65, 64, int(spec["T"])]
    elif mutation == "varlen_b2":
        values["layout"] = "BSND"
        for name in ("q", "k", "v", "g", "beta"):
            values[name] = inputs.__dict__[name].repeat(2, *([1] * (inputs.__dict__[name].dim() - 1)))
        values["cu"] = [0, int(spec["T"])]
    elif mutation == "seq_gt_1024":
        values["cu"] = [0] * 1025 + [int(spec["T"])]
    elif mutation == "indices_without_cu":
        values["cu"] = None
        values["indices"] = [0, 0]
    elif mutation in {"indices_count", "indices_missing_pair"}:
        values["cu"] = [0, 64, int(spec["T"])]
        values["indices"] = [0, 0]
    elif mutation == "indices_extra_pair":
        values["cu"] = [0, 64, int(spec["T"])]
        values["indices"] = [0, 0, 1, 0, 1, 1]
    elif mutation == "indices_order":
        values["cu"] = [0, 64, int(spec["T"])]
        values["indices"] = [1, 0, 0, 0]
    elif mutation in {"state_shape_kv", "state_shape_vk"}:
        values["initial_state"] = inputs.initial_state.transpose(-1, -2).contiguous()
    elif mutation == "initial_state_key_heads":
        values["initial_state"] = inputs.initial_state.narrow(1, 0, int(spec["H"])).contiguous()
    elif mutation == "g_beta_key_heads":
        head_axis = _head_axis(layout)
        values["g"] = _resize_axis(inputs.g, head_axis, int(spec["H"]))
        values["beta"] = _resize_axis(inputs.beta, head_axis, int(spec["H"]))
    elif mutation in {"alog_shape", "alog_shape_plus_one"}:
        values["A_log"] = _resize_axis(inputs.A_log, 0, int(spec["HV"]) + 1)
    elif mutation in {"dtbias_shape", "dtbias_shape_minus_one"}:
        values["dt_bias"] = inputs.dt_bias[:-1].contiguous()
    elif mutation == "lower_low":
        values["lower_bound"] = -5.001
    elif mutation == "lower_high":
        values["lower_bound"] = 0.0
    elif mutation == "final_shape":
        outputs[1] = outputs[1].transpose(-1, -2).contiguous()
    elif mutation == "gk_sequence_major":
        outputs[2] = outputs[2].transpose(1, 2).contiguous()
    elif mutation == "aqk_last_dim":
        outputs[3] = outputs[3][..., :-1].contiguous()
    elif mutation == "w_last_dim_v":
        outputs[5] = _resize_axis(outputs[5], outputs[5].dim() - 1, int(spec["V"]))
    elif mutation == "u_last_dim_k":
        outputs[6] = _resize_axis(outputs[6], outputs[6].dim() - 1, int(spec["K"]))
    elif mutation == "h_layout_or_state":
        outputs[10] = outputs[10].transpose(1, 2).contiguous()
    else:
        raise ValueError(f"unsupported negative mutation: {mutation}")
    return values


def _raw_output_tensors(inputs: _PreparedInputs, spec: dict) -> list:
    layout = str(spec["layout"])
    rank3 = layout in {"TND", "NTD"}
    batch, total_t, hv_num = int(spec["B"]), int(spec["T"]), int(spec["HV"])
    k_dim, v_dim, chunk_size = int(spec["K"]), int(spec["V"]), int(spec["chunk_size"])
    total_chunks = (total_t + chunk_size - 1) // chunk_size
    state_shape = (
        (batch, hv_num, v_dim, k_dim) if _as_bool(spec["state_v_first"])
        else (batch, hv_num, k_dim, v_dim)
    )
    attn_shape = (total_t, hv_num, v_dim) if rank3 else (batch, total_t, hv_num, v_dim)
    matrix_shape = (hv_num, total_t, chunk_size) if rank3 else (batch, hv_num, total_t, chunk_size)
    k_shape = (hv_num, total_t, k_dim) if rank3 else (batch, hv_num, total_t, k_dim)
    v_shape = (hv_num, total_t, v_dim) if rank3 else (batch, hv_num, total_t, v_dim)
    h_shape = (
        (total_chunks, hv_num, v_dim, k_dim) if rank3 and _as_bool(spec["state_v_first"])
        else (total_chunks, hv_num, k_dim, v_dim) if rank3
        else (batch, total_chunks, hv_num, v_dim, k_dim) if _as_bool(spec["state_v_first"])
        else (batch, total_chunks, hv_num, k_dim, v_dim)
    )
    new = lambda shape, dtype=inputs.q.dtype: torch.empty(shape, dtype=dtype, device=inputs.q.device)
    return [
        new(attn_shape),
        new(state_shape, torch.float32),
        new(k_shape, torch.float32),
        new(matrix_shape),
        new(matrix_shape),
        new(k_shape),
        new(v_shape),
        new(k_shape),
        new(k_shape),
        new(v_shape),
        new(h_shape),
    ]


def _run_negative_aclnn(inputs: _PreparedInputs, spec: dict):
    from fla_npu.ops.ascendc._aclnn_ctypes import _GET_WORKSPACE_ARGTYPES, _call_aclnn

    outputs = _raw_output_tensors(inputs, spec)
    values = _mutate_raw(inputs, spec, outputs)
    layout_buffer = None
    layout_arg = ctypes.c_char_p()
    if values["layout"] is not None:
        layout_buffer = ctypes.create_string_buffer(str(values["layout"]).encode("utf-8"))
        layout_arg = ctypes.cast(layout_buffer, ctypes.c_char_p)
    try:
        _call_aclnn(
            "aclnnChunkKdaFwd",
            lambda ctx: [
                ctx.tensor(values["q"], "q"),
                ctx.tensor(values["k"], "k"),
                ctx.tensor(values["v"], "v"),
                ctx.tensor(values["g"], "g"),
                ctx.tensor(values["beta"], "beta"),
                ctx.tensor(values["A_log"], "A_log"),
                ctx.tensor(values["dt_bias"], "dt_bias"),
                ctx.tensor(values["initial_state"], "initial_state"),
                ctx.int_array(values["cu"]),
                ctx.int_array(values["indices"]),
                layout_arg,
                ctypes.c_double(float(spec["scale"])),
                ctypes.c_int64(values["chunk_size"]),
                ctypes.c_bool(_as_bool(spec["safe_gate"])),
                ctypes.c_double(values["lower_bound"]),
                ctypes.c_bool(_as_bool(spec["use_gate_in_kernel"])),
                ctypes.c_bool(_as_bool(spec["state_v_first"])),
                *[ctx.tensor(output, _OUTPUT_NAMES[index]) for index, output in enumerate(outputs)],
            ],
            tuple(output for output in outputs if output is not None),
        )
    except RuntimeError as exc:
        match = re.search(r"aclnnStatus=(\d+)", str(exc))
        actual_code = int(match.group(1)) if match else None
        expected_code = int(spec["expected_return_code"])
        if actual_code != expected_code:
            raise RuntimeError(
                f"negative interception returned {actual_code}, expected {expected_code}: {exc}"
            ) from exc
        recent_error = _recent_aclnn_error()
        expected_message = str(spec["expected_message"])
        if expected_message not in recent_error:
            raise RuntimeError(
                f"negative interception code matched but message did not: expected {expected_message!r}, "
                f"actual {recent_error!r}"
            ) from exc
        raise RuntimeError(
            f"{spec['expected_code_name']}({expected_code}): {expected_message}; recent_error={recent_error}"
        ) from exc
    raise RuntimeError("negative interception unexpectedly returned ACLNN_SUCCESS")


def _run_negative_ascendc(inputs: _PreparedInputs, spec: dict):
    """Validate the public Python route's exception type and message."""
    from fla_npu.ops.ascendc import chunk_kda_fwd

    outputs = _raw_output_tensors(inputs, spec)
    values = _mutate_raw(inputs, spec, outputs)
    try:
        chunk_kda_fwd(
            values["q"],
            values["k"],
            values["v"],
            values["g"],
            values["beta"],
            float(spec["scale"]),
            values["chunk_size"],
            layout=values["layout"],
            initial_state=values["initial_state"],
            output_final_state=_as_bool(spec["output_final_state"]),
            cu_seqlens=values["cu"],
            chunk_indices=values["indices"],
            safe_gate=_as_bool(spec["safe_gate"]),
            lower_bound=values["lower_bound"],
            use_gate_in_kernel=_as_bool(spec["use_gate_in_kernel"]),
            A_log=values["A_log"],
            dt_bias=values["dt_bias"],
            disable_recompute=_as_bool(spec["disable_recompute"]),
            return_intermediate_states=_as_bool(spec["return_intermediate_states"]),
            state_v_first=_as_bool(spec["state_v_first"]),
        )
    except RuntimeError as exc:
        actual_code = type(exc).__name__
        expected_code = str(spec["expected_return_code"])
        if actual_code != expected_code:
            raise RuntimeError(
                f"negative public interception returned {actual_code}, "
                f"expected {expected_code}: {exc}"
            ) from exc
        expected_message = str(spec["expected_message"])
        if expected_message not in str(exc):
            raise RuntimeError(
                "negative public interception exception matched but message did not: "
                f"expected {expected_message!r}, actual {str(exc)!r}"
            ) from exc
        raise RuntimeError(
            f"{spec['expected_code_name']}({expected_code}): "
            f"{expected_message}; actual_error={exc}"
        ) from exc
    raise RuntimeError("negative public interception unexpectedly succeeded")


def _reference_role(device: str, is_benchmark_task: bool) -> str:
    if device == "cpu":
        return "cpu_fp64" if is_benchmark_task else "cpu_same_precision"
    if device == "gpu":
        return "gpu_fp64" if is_benchmark_task else "gpu_triton_same_precision"
    if device == "npu":
        return "npu_dut"
    return "unsupported"


@register("executor_chunk_kda_fwd")
class ChunkKdaFwdApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.spec = None
        self.inputs = None
        case_config = getattr(task_result, "case_config", None)
        case_id = case_config.get("id") if isinstance(case_config, dict) else getattr(case_config, "id", None)
        self.runtime_case_id = None if case_id is None else int(case_id)
        task_names = {
            str(getattr(task_type, "value", task_type)).lower().rsplit(".", 1)[-1]
            for task_type in (task_result.task_type or [])
        }
        self.task_names = task_names
        self.randomize_values = bool(
            task_result.disable_id_seed and "accuracy_lt" in task_names
        )
        self.persistent_cache_mode = _persistent_cache_mode()
        _validate_persistent_cache_task(self.persistent_cache_mode, task_names)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        # The benchmark node supplies FP64; the regular reference node supplies
        # Triton unless this case explicitly requires a Torch control.
        self.high_precision = (
            self.device in {"cpu", "gpu"} and self.is_benchmark_task
        )
        self.cpu_control = self.device == "cpu" and not self.is_benchmark_task
        self.triton_control = self.device == "gpu" and not self.is_benchmark_task
        self.gpu_torch_control = False
        self.cache_reader = None
        self.cache_validation_receipt = None
        self.execution_device = None

    def init_by_input_data(self, input_data: InputDataset):
        self.spec = json.loads(str(input_data.kwargs["case_spec"]))
        gpu_control_reference = str(
            self.spec.get("gpu_control_reference", "triton_same_precision")
        )
        if gpu_control_reference not in {
            "triton_same_precision",
            "torch_same_precision",
        }:
            raise RuntimeError(
                "gpu_control_reference must be 'triton_same_precision' or "
                "'torch_same_precision'"
            )
        if self.device == "gpu" and not self.is_benchmark_task:
            self.gpu_torch_control = gpu_control_reference == "torch_same_precision"
            self.triton_control = not self.gpu_torch_control
        runtime_seed = int(self.spec["seed"])
        if self.randomize_values:
            if self.runtime_case_id is None:
                raise RuntimeError("accuracy_lt with --disable_id_seed requires an ATK runtime case id")
            runtime_seed = (
                runtime_seed * 0x9E3779B185EBCA87 + self.runtime_case_id
            ) % (2**63 - 1)
        self.execution_device = input_data.kwargs["low_precision_marker"].device
        use_persistent_cache = (
            self.persistent_cache_mode == "readonly" and not self.randomize_values
        )
        if use_persistent_cache:
            self.cache_reader = _persistent_cache_reader(self.spec, runtime_seed)
            self.cache_validation_receipt = dict(
                self.cache_reader.validation_receipt
            )
            role = (
                "gpu_torch_same_precision"
                if self.gpu_torch_control
                else _reference_role(self.device, self.is_benchmark_task)
            )
            if role in {
                "cpu_fp64",
                "cpu_same_precision",
                "gpu_fp64",
            }:
                # Reference values are useful only if the deterministic input shard
                # they were built from is still present and payload-valid.
                self.cache_reader.load_shard("inputs")
            else:
                cached_inputs = self.cache_reader.load_shard("inputs")
                self.inputs = _prepared_inputs_from_cpu(
                    _select_cached_input_payload(cached_inputs, self.spec),
                    self.execution_device,
                    high_precision=False,
                )
                self.inputs = _apply_input_storage(self.inputs, self.spec)
                if self.inputs.seed != runtime_seed:
                    raise ReferenceCacheError(
                        f"cached input seed {self.inputs.seed} does not match {runtime_seed}"
                    )
        else:
            self.inputs = _prepare_inputs(
                self.spec,
                input_data.kwargs["low_precision_marker"],
                input_data.kwargs["fp32_marker"],
                high_precision=self.high_precision,
                seed=runtime_seed,
            )
        if os.environ.get("KDA_ATK_TRACE_SEED") == "1":
            print(
                "KDA_ATK_RUNTIME_SEED",
                self.device,
                "benchmark=" + str(self.is_benchmark_task),
                "high_precision=" + str(self.high_precision),
                "cpu_control=" + str(self.cpu_control),
                "triton_control=" + str(self.triton_control),
                "gpu_torch_control=" + str(self.gpu_torch_control),
                "persistent_cache=" + str(self.cache_reader is not None),
                "producer_torch="
                + str(
                    None
                    if self.cache_validation_receipt is None
                    else self.cache_validation_receipt["producer_torch_version"]
                ),
                "consumer_torch="
                + str(
                    None
                    if self.cache_validation_receipt is None
                    else self.cache_validation_receipt["consumer_torch_version"]
                ),
                "case_id=" + str(self.runtime_case_id),
                "seed=" + str(runtime_seed),
                flush=True,
            )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del input_data, with_output
        tags = {tag.strip() for tag in str(self.spec["tags"]).split(",")}
        if "negative" in tags:
            if self.device != "npu":
                raise RuntimeError("negative cases must run on an NPU route")
            route = str(self.spec["route"])
            if route == "aclnn":
                return _run_negative_aclnn(self.inputs, self.spec)
            if route == "ascendc":
                return _run_negative_ascendc(self.inputs, self.spec)
            raise RuntimeError(f"negative cases do not support route={route!r}")
        role = (
            "gpu_torch_same_precision"
            if self.gpu_torch_control
            else _reference_role(self.device, self.is_benchmark_task)
        )
        if self.cache_reader is not None and role in {
            "cpu_fp64",
            "cpu_same_precision",
            "gpu_fp64",
        }:
            shard = (
                "cpu_fp64"
                if role in {"cpu_fp64", "gpu_fp64"}
                else "cpu_same_precision"
            )
            outputs = _outputs_from_cpu(
                self.cache_reader.load_shard(shard), self.execution_device
            )
        elif self.high_precision:
            outputs = _torch_fp64_golden(self.inputs, self.spec)
        elif self.cpu_control:
            outputs = _torch_same_precision(self.inputs, self.spec)
        elif self.gpu_torch_control:
            outputs = _torch_same_precision(self.inputs, self.spec)
        elif self.triton_control:
            outputs = _triton_same_precision(self.inputs, self.spec)
        elif self.device == "npu":
            outputs = _run_positive_npu(self.inputs, self.spec)
        else:
            raise RuntimeError(
                "positive chunk_kda_fwd cases require one NPU node and a supported "
                "CPU/GPU reference node; "
                f"got device={self.device!r}, benchmark={self.is_benchmark_task}"
            )
        selected_names = _selected_output_names()
        named_visible = tuple(
            (name, output.to(torch.float32) if output.is_floating_point() else output)
            for name, output in zip(_OUTPUT_NAMES, outputs)
            if isinstance(output, torch.Tensor)
            and (selected_names is None or name in selected_names)
        )
        visible = tuple(output for _, output in named_visible)
        if not visible:
            raise RuntimeError("chunk_kda_fwd returned no visible tensor outputs")
        for name, output in named_visible:
            if not torch.isfinite(output).all().item():
                raise RuntimeError(f"{name} contains NaN or Inf")
        return visible

    def export_custom_data(self, input_data: InputDataset):
        del input_data
        return {
            "case_key": str(self.spec["case_key"]),
            "soc": str(self.spec["soc"]),
            "route": str(self.spec["route"]),
            "B": int(self.spec["B"]),
            "H": int(self.spec["H"]),
            "HV": int(self.spec["HV"]),
            "T": int(self.spec["T"]),
            "K": int(self.spec["K"]),
            "V": int(self.spec["V"]),
            "chunk_size": int(self.spec["chunk_size"]),
            "gpu_control_reference": str(
                self.spec.get("gpu_control_reference", "triton_same_precision")
            ),
        }
