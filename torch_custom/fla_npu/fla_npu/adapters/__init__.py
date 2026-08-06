"""Compatibility adapters for third-party model/operator packages."""

from .triton_ascend_kda import (
    install_triton_ascend_kda_adapter,
    is_triton_ascend_kda_adapter_installed,
    remove_triton_ascend_kda_adapter,
    triton_ascend_chunk_kda_fwd,
)

__all__ = [
    "install_triton_ascend_kda_adapter",
    "is_triton_ascend_kda_adapter_installed",
    "remove_triton_ascend_kda_adapter",
    "triton_ascend_chunk_kda_fwd",
]
