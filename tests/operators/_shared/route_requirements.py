"""Explicit environment gates for optional operator routes."""

from __future__ import annotations

import os

import pytest


def _require_flag(flag: str, route: str) -> None:
    if os.environ.get(flag) != "1":
        pytest.skip(
            f"{route} tests require {flag}=1 and their dedicated extension build",
            allow_module_level=True,
        )


def require_legacy_route() -> None:
    """Require the wheel built with the optional torch.ops.npu extension."""
    _require_flag("FLA_NPU_RUN_LEGACY_TESTS", "legacy torch.ops.npu route")


def require_fast_kernel_route() -> None:
    """Require the dedicated fast-kernel ascend_ops extension."""
    _require_flag("FLA_NPU_RUN_FAST_KERNEL_TESTS", "fast-kernel route")
