"""Opt-in experimental operators; configure before importing torch_npu/acl."""

from ._loader import configure as _configure

_configure()

from ._rwkv8 import fused_recurrent_rwkv8

__all__ = ["fused_recurrent_rwkv8"]
