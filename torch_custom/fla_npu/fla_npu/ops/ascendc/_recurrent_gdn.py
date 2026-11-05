"""Validate the RecurrentGDN V2 contract before submitting a mutable launch.

Checks preserve Q/K/V and State views. Metadata ownership checks are cached
only while the original tensors and their mutation versions remain unchanged.
"""
from __future__ import annotations

import math
import numbers
import weakref
from collections import OrderedDict
from itertools import product

_METADATA_CACHE = OrderedDict()

def _require_shape(
    name: str,
    tensor: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"{name} shape must be {expected}, got {actual}.")


def _require_dtype(name: str, tensor: torch.Tensor, allowed: tuple) -> None:
    if tensor.dtype not in allowed:
        raise TypeError(f"{name} dtype must be one of {allowed}, got {tensor.dtype}.")


def _storage_bounds(name: str, tensor: torch.Tensor) -> None:
    import torch
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must use strided (ND) layout.")
    if any(stride < 0 for stride in tensor.stride()):
        raise ValueError(f"{name} strides must be non-negative.")
    if tensor.numel() == 0:
        return
    last = tensor.storage_offset() + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    end = (last + 1) * tensor.element_size()
    if tensor.storage_offset() < 0 or end > min(tensor.untyped_storage().nbytes(), (1 << 63) - 1):
        raise ValueError(f"{name} view exceeds storage or 64-bit address bounds.")


def _byte_intervals(tensor: torch.Tensor) -> list[tuple[int, int]]:
    """Exact occupied byte intervals, including holes and storage offsets.

    Used only for ambiguous State overlap or shared-storage alias checks.
    Dense trailing axes are coalesced to avoid enumerating every element.
    """
    if tensor.numel() == 0:
        return []
    axes = sorted((stride, size) for size, stride in zip(tensor.shape, tensor.stride()) if size > 1)
    span, count = 1, 0
    for stride, size in axes:
        if stride != span:
            break
        span *= size
        count += 1
    rest = axes[count:]
    itemsize = tensor.element_size()
    base = tensor.data_ptr()  # Already includes storage_offset.
    starts = sorted(
        base + sum(index * stride for index, (stride, _) in zip(indices, rest)) * itemsize
        for indices in product(*(range(size) for _, size in rest))
    )
    return [(start, start + span * itemsize) for start in starts]


def _validate_state_layout(state: torch.Tensor, others: dict) -> None:
    if any(stride <= 0 for stride in state.stride()):
        raise ValueError("State strides must be positive.")
    if state.stride(-1) != 1 or state.stride(-2) < 128:
        raise ValueError("State requires stride_K=1 and stride_V>=128.")
    if (state.stride(-2) - 128) * state.element_size() > (1 << 32) - 1:
        raise ValueError("State DMA row gap exceeds UINT32_MAX bytes.")
    # This quick proof covers ordinary padding, axis permutations and offsets.
    span, proven_disjoint = 1, True
    for stride, size in sorted((s, n) for s, n in zip(state.stride(), state.shape) if n > 1):
        if stride < span:
            proven_disjoint = False
            break
        span += (size - 1) * stride
    state_intervals = None
    if not proven_disjoint:
        state_intervals = _byte_intervals(state)
        if any(b > c for (_, b), (c, _) in zip(state_intervals, state_intervals[1:])):
            raise ValueError("State view must not overlap itself.")

    def extent(tensor):
        start = tensor.data_ptr()
        end = start + (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))) * tensor.element_size()
        return start, end

    state_start, state_end = extent(state)
    for name, tensor in others.items():
        if tensor is None or tensor.numel() == 0:
            continue
        other_start, other_end = extent(tensor)
        if state_end <= other_start or other_end <= state_start:
            continue
        if state_intervals is None:
            state_intervals = _byte_intervals(state)
        a, b, i, j = state_intervals, _byte_intervals(tensor), 0, 0
        while i < len(a) and j < len(b):
            if max(a[i][0], b[j][0]) < min(a[i][1], b[j][1]):
                raise ValueError(f"State must not overlap input {name}.")
            if a[i][1] <= b[j][1]:
                i += 1
            else:
                j += 1


def _validate_state_ownership(lengths, state_indices, accepted, table_mode):
    writers, readers = {}, []
    start = lengths[0]
    for batch, length in enumerate(lengths[1:]):
        if length:
            row = state_indices[batch] if table_mode else state_indices[start:start + length]
            initial_position = 0 if accepted is None else accepted[batch] - 1
            readers.append((row[initial_position], batch))
            for slot in row[:length]:
                if writers.setdefault(slot, batch) != batch:
                    raise ValueError("State writes overlap across requests.")
        start += length
    for slot, batch in readers:
        if slot in writers and writers[slot] != batch:
            raise ValueError("Initial State is written by another request.")


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    use_gate_in_kernel: bool,
    use_qk_l2norm_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    _allow_cpu: bool = False,
) -> tuple[list[int], list[int] | list[list[int]], list[int] | None]:
    """Validate the current design contract before any numerical/state update."""
    import torch

    flags = (use_qk_l2norm_in_kernel, use_gate_in_kernel,
             use_beta_sigmoid_in_kernel, allow_neg_eigval)
    if any(not isinstance(flag, bool) for flag in flags):
        raise TypeError("All operator switches must be bool.")
    if not isinstance(scale, numbers.Real) or isinstance(scale, bool) or not math.isfinite(float(scale)):
        raise ValueError("scale must be a finite real number (not bool).")
    if gk is not None:
        raise ValueError("gk is reserved; only scalar gate g is supported.")
    named_tensors = {
        "query": query,
        "key": key,
        "value": value,
        "state": state,
        "beta": beta,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "g": g,
        "gk": gk,
        "A_log": A_log,
        "dt_bias": dt_bias,
    }
    optional = {"num_accepted_tokens", "g", "gk", "A_log", "dt_bias"}
    for name, tensor in named_tensors.items():
        if tensor is None and name in optional:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.device.type != ("cpu" if _allow_cpu else "npu"):
            raise ValueError(f"{name} must be on {'CPU' if _allow_cpu else 'NPU'}.")
        if tensor.device != query.device:
            raise ValueError("All RecurrentGDN tensors must be on the same device.")
        _storage_bounds(name, tensor)
    for name in ("query", "key", "value"):
        _require_dtype(name, named_tensors[name], (torch.bfloat16,))
    for name in ("state", "beta", "g"):
        if named_tensors[name] is not None:
            _require_dtype(name, named_tensors[name], (torch.bfloat16, torch.float32))
    for name in ("A_log", "dt_bias"):
        if named_tensors[name] is not None:
            _require_dtype(name, named_tensors[name], (torch.float32,))

    if query.ndim != 3:
        raise ValueError(f"query must be rank 3, got shape={tuple(query.shape)}.")
    total_tokens, key_heads, key_dim = query.shape
    _require_shape("key", key, (total_tokens, key_heads, key_dim))

    if value.ndim != 3:
        raise ValueError(f"value must be rank 3, got shape={tuple(value.shape)}.")
    value_tokens, value_heads, value_dim = value.shape
    if value_tokens != total_tokens:
        raise ValueError(
            f"query/value token counts must match, got {total_tokens} and "
            f"{value_tokens}."
        )
    if key_heads <= 0 or value_heads <= 0 or value_heads % key_heads != 0:
        raise ValueError(
            "value head count must be a positive integer multiple of key head "
            f"count, got Nk={key_heads}, Nv={value_heads}."
        )

    if total_tokens <= 0 or key_dim != 128 or value_dim != 128:
        raise ValueError("T must be positive and K=V=128.")
    if value_heads // key_heads > 16:
        raise ValueError("HV/HK must be an integer in [1,16].")
    for name in ("query", "key", "value"):
        tensor = named_tensors[name]
        if any(s <= 0 for s in tensor.stride()) or tensor.stride(-1) != 1 or tensor.stride(0) < 128:
            raise ValueError(f"{name} requires stride_D=1 and token stride>=128.")
        if (tensor.stride(0) - 128) * tensor.element_size() > (1 << 32) - 1:
            raise ValueError(f"{name} DMA row gap exceeds UINT32_MAX bytes.")

    if state.ndim != 4:
        raise ValueError(f"state must be rank 4, got shape={tuple(state.shape)}.")
    state_blocks = state.shape[0]
    _require_shape(
        "state",
        state,
        (state_blocks, value_heads, value_dim, key_dim),
    )
    if state_blocks <= 0:
        raise ValueError("state must contain at least one State pool slot.")
    _require_shape("beta", beta, (total_tokens, value_heads))
    if g is not None:
        _require_shape("g", g, (total_tokens, value_heads))
    for name, tensor in (("A_log", A_log), ("dt_bias", dt_bias)):
        if tensor is not None:
            _require_shape(name, tensor, (value_heads,))
    _validate_state_layout(state, {k: v for k, v in named_tensors.items() if k != "state"})
    if use_gate_in_kernel:
        if g is None:
            raise ValueError(
                "g is required when use_gate_in_kernel=True because g is "
                "the raw scalar gate input."
            )
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True.")

    if actual_seq_lengths.ndim != 1 or actual_seq_lengths.numel() < 2:
        raise ValueError(
            "actual_seq_lengths must be [prefix, length_0, ..., length_B-1]."
        )
    if actual_seq_lengths.dtype != torch.int32:
        raise TypeError(
            "actual_seq_lengths must use INT32, got "
            f"{actual_seq_lengths.dtype}."
        )
    tensors = (actual_seq_lengths, ssm_state_indices, num_accepted_tokens)
    versions = []
    for tensor in tensors:
        try:
            versions.append(None if tensor is None else tensor._version)
        except RuntimeError:
            versions.append(None)
    cache_key = (tuple(id(t) for t in tensors), tuple(versions), total_tokens, state_blocks)
    cached = _METADATA_CACHE.get(cache_key)
    if cached is not None and all(
        (ref is None and tensor is None) or (ref is not None and ref() is tensor)
        for ref, tensor in zip(cached, tensors)
    ):
        _METADATA_CACHE.move_to_end(cache_key)
        return None

    lengths = [int(item) for item in actual_seq_lengths.detach().cpu().tolist()]
    prefix_tokens = lengths[0]
    seq_lengths = lengths[1:]
    if prefix_tokens < 0 or any(length < 0 for length in seq_lengths):
        raise ValueError(f"sequence lengths must be non-negative, got {lengths}.")
    if sum(lengths) != total_tokens:
        raise ValueError(
            "actual_seq_lengths must sum to T, got "
            f"sum={sum(lengths)} and T={total_tokens}."
        )

    if ssm_state_indices.dtype != torch.int32:
        raise TypeError(
            "ssm_state_indices must use INT32, got "
            f"{ssm_state_indices.dtype}."
        )
    batch_size = len(seq_lengths)
    if ssm_state_indices.ndim == 1:
        _require_shape("ssm_state_indices", ssm_state_indices, (total_tokens,))
        state_widths = seq_lengths
    elif ssm_state_indices.ndim == 2:
        if ssm_state_indices.shape[0] != batch_size:
            raise ValueError(
                "2-D ssm_state_indices row count must equal B, got "
                f"shape={tuple(ssm_state_indices.shape)} and B={batch_size}."
            )
        state_width = int(ssm_state_indices.shape[1])
        if state_width <= 0:
            raise ValueError("2-D ssm_state_indices must have a positive W.")
        if any(length > state_width for length in seq_lengths):
            raise ValueError(
                f"every sequence length must be <= W={state_width}, got "
                f"{seq_lengths}."
            )
        state_widths = [state_width] * batch_size
    else:
        raise ValueError(
            "ssm_state_indices must be rank 1 [T] or rank 2 [B, W], got "
            f"shape={tuple(ssm_state_indices.shape)}."
        )

    flat_indices = ssm_state_indices.detach().cpu().reshape(-1).to(torch.int64)
    if torch.any(flat_indices < 0) or torch.any(flat_indices >= state_blocks):
        raise ValueError(
            "ssm_state_indices values must be in [0, BlockNum), got "
            f"BlockNum={state_blocks}."
        )

    accepted: list[int] | None = None
    if num_accepted_tokens is not None:
        if num_accepted_tokens.dtype != torch.int32:
            raise TypeError(
                "num_accepted_tokens must use INT32, got "
                f"{num_accepted_tokens.dtype}."
            )
        _require_shape(
            "num_accepted_tokens",
            num_accepted_tokens,
            (batch_size,),
        )
        accepted = [int(item) for item in num_accepted_tokens.detach().cpu().tolist()]
        for batch_index, seq_len in enumerate(seq_lengths):
            # Empty rows are skipped before the kernel reads accepted[b].
            if seq_len == 0:
                continue
            if not 1 <= accepted[batch_index] <= state_widths[batch_index]:
                raise ValueError(
                    f"num_accepted_tokens[{batch_index}] must be in "
                    f"[1, {state_widths[batch_index]}], got "
                    f"{accepted[batch_index]}."
                )

    state_indices = ssm_state_indices.detach().cpu().tolist()
    _validate_state_ownership(lengths, state_indices, accepted, ssm_state_indices.ndim == 2)
    if all(t is None or v is not None for t, v in zip(tensors, versions)):
        _METADATA_CACHE[cache_key] = tuple(None if t is None else weakref.ref(t) for t in tensors)
        if len(_METADATA_CACHE) > 64:
            _METADATA_CACHE.popitem(last=False)
    return lengths, state_indices, accepted


def validate_recurrent_gdn(query, key, value, state, *, beta, scale,
                           actual_seq_lengths, ssm_state_indices, num_accepted_tokens=None,
                           g=None, gk=None, A_log=None, dt_bias=None,
                           use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                           use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                           _allow_cpu=False):
    _validate_inputs(query, key, value, state, beta, scale, actual_seq_lengths,
        ssm_state_indices, num_accepted_tokens, g, gk, A_log, dt_bias,
        use_gate_in_kernel, use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel,
        allow_neg_eigval, _allow_cpu)
    return float(scale)
