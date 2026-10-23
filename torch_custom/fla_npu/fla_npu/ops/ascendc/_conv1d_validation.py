"""Opt-in Conv1d metadata validation, independent of the execution backend.

The checks preserve the public contract from the former Python wrappers. Device
metadata is copied to CPU only when the caller explicitly requests validation.
No tensor data or mutation version is changed by these helpers.
"""

from __future__ import annotations

from collections.abc import Sequence


def _cpu_metadata_values(value):
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(
                f"query_start_loc_cpu must be a CPU Tensor, got {value.device}")
        if value.dim() != 1:
            raise ValueError(
                f"query_start_loc_cpu must be rank 1, got shape {tuple(value.shape)}")
        if value.dtype not in {torch.bool, torch.int32, torch.int64}:
            raise TypeError(
                f"query_start_loc_cpu must use bool, int32, or int64, got {value.dtype}")
        return [int(item) for item in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    raise TypeError(
        "query_start_loc_cpu must be a CPU Tensor, an integer sequence, or None")


def validate_query_start_loc(x, device_value, cpu_value, *, max_query_len=None):
    """Reject invalid boundaries and undersized update bounds before launch."""
    if device_value is None and cpu_value is None:
        return
    import torch

    values = _cpu_metadata_values(cpu_value)
    if device_value is not None:
        if values is not None:
            raise ValueError(
                "query_start_loc and query_start_loc_cpu are mutually exclusive")
        if not isinstance(device_value, torch.Tensor):
            raise TypeError("query_start_loc must be a device Tensor or None")
        if device_value.dim() != 1:
            raise ValueError(
                f"query_start_loc must be rank 1, got shape {tuple(device_value.shape)}")
        if device_value.dtype != torch.int32:
            raise TypeError(f"query_start_loc must use int32, got {device_value.dtype}")
        if device_value.device != x.device:
            raise ValueError(
                f"query_start_loc must be on {x.device}, got {device_value.device}")
        values = [int(item) for item in device_value.detach().cpu().tolist()]
    total_tokens = x.shape[0]
    if len(values) < 2:
        raise ValueError("query_start_loc must contain at least [0, total_tokens]")
    if values[0] != 0 or values[-1] != total_tokens:
        raise ValueError(
            "query_start_loc must start at 0 and end at "
            f"{total_tokens}, got ({values[0]}, {values[-1]})")
    if any(left > right for left, right in zip(values, values[1:])):
        raise ValueError(f"query_start_loc must be non-decreasing, got {values}")
    if max_query_len is not None:
        observed_max = max(
            (right - left for left, right in zip(values, values[1:])), default=0)
        if max_query_len >= 0 and max_query_len < observed_max:
            raise ValueError(
                f"max_query_len={max_query_len} is smaller than the observed "
                f"segment length {observed_max}")
