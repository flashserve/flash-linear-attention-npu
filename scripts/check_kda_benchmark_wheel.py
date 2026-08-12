#!/usr/bin/env python3
"""Validate the scoped wheel required by the KDA benchmark."""

from __future__ import annotations

import os
from pathlib import Path


REQUIRED_EXPORTS = (
    "chunk_kda_fwd",
    "npu_chunk_kda_fwd",
    "kda_gate_cumsum",
    "npu_kda_gate_cumsum",
    "chunk_gated_delta_rule_fwd_h",
    "npu_chunk_gated_delta_rule_fwd_h",
)
REQUIRED_CONFIGS = (
    "chunk_kda_fwd.json",
    "kda_gate_cumsum.json",
    "chunk_gated_delta_rule_fwd_h.json",
)


def main() -> int:
    import fla_npu
    from fla_npu.ops import ascendc

    missing_exports = [name for name in REQUIRED_EXPORTS if not hasattr(ascendc, name)]
    if missing_exports:
        raise AssertionError("scoped wheel exports are missing: " + ", ".join(missing_exports))

    package_root = Path(fla_npu.__file__).resolve().parent
    config_dirs = list(
        (package_root / "opp" / "vendors").glob(
            "*/op_impl/ai_core/tbe/kernel/config/*"
        )
    )
    missing_configs = [
        name
        for name in REQUIRED_CONFIGS
        if not any((directory / name).is_file() for directory in config_dirs)
    ]
    if missing_configs:
        raise AssertionError(
            "scoped wheel OPP configs are missing: " + ", ".join(missing_configs)
        )

    libraries = list(
        (package_root / "opp" / "vendors").glob("*/op_api/lib/libcust_opapi.so")
    )
    if not libraries:
        raise AssertionError("scoped wheel libcust_opapi.so is missing")
    configured = os.environ.get("FLA_NPU_OP_API_LIB")
    if not configured or Path(configured).resolve() not in {
        library.resolve() for library in libraries
    }:
        raise AssertionError("FLA_NPU_OP_API_LIB does not select the scoped wheel library")

    print("KDA benchmark scoped wheel check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
