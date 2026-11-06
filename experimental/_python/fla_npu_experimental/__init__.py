"""Opt-in experimental operators; configure before importing torch_npu/acl."""

from ._loader import configure as _configure

_configure()


__all__ = []
