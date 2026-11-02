"""Validate retired backend selectors without importing torch or CANN."""

from __future__ import annotations

import os


def validate_backend_environment() -> None:
    mode = os.environ.get("FLA_NPU_STABLE_ABI", "").strip().lower()
    if mode not in {"", "stable", "1", "true", "yes", "on"}:
        raise RuntimeError(
            f"FLA_NPU_STABLE_ABI={mode!r} is unsupported: Stable ABI is the "
            "only Ascend C backend. Unset FLA_NPU_STABLE_ABI or set it to stable."
        )
    validation = os.environ.get("FLA_NPU_STABLE_VALIDATE", "").strip().lower()
    if validation not in {"", "0", "false", "no", "off"}:
        raise RuntimeError(
            "FLA_NPU_STABLE_VALIDATE no longer selects Python validation: the "
            "ctypes execution backend has been removed. Unset this variable; "
            "input checks run in the Stable wrappers, adapters and ACLNN."
        )
