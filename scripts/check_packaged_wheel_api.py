# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate the installed flash-linear-attention-npu wheel API surface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ASCENDC_NAMES = (
    "causal_conv1d",
    "causal_conv1d_bwd",
    "chunk_bwd_dqkwg",
    "chunk_bwd_dv_local",
    "chunk_fwd_o",
    "chunk_gated_delta_rule_bwd_dhu",
    "chunk_gated_delta_rule_fwd_h",
    "prepare_wy_repr_bwd_da",
    "prepare_wy_repr_bwd_full",
    "recompute_w_u_fwd",
    "solve_tri",
)

TRITON_NAMES = (
    "autocast_custom_bwd",
    "autocast_custom_fwd",
    "causal_conv1d",
    "causal_conv1d_triton",
    "chunk_local_cumsum",
    "chunk_scaled_dot_kkt_fwd",
    "input_guard",
    "l2norm",
    "solve_tril",
)
REQUIRED_ASCENDC_CONFIGS = (
    "recompute_wu_fwd.json",
    "recompute_w_u_fwd.json",
)


def _require_attr(obj, name: str, owner: str) -> None:
    if not hasattr(obj, name):
        raise AssertionError(f"{owner}.{name} is missing")


def _require_packaged_opp_configs(fla_npu_module) -> None:
    package_root = Path(fla_npu_module.__file__).resolve().parent
    config_dirs = list(
        (package_root / "opp" / "vendors").glob(
            "*/op_impl/ai_core/tbe/kernel/config/*"
        )
    )
    if not config_dirs:
        raise AssertionError("packaged OPP kernel config directory is missing")

    missing = []
    for config_name in REQUIRED_ASCENDC_CONFIGS:
        if not any((config_dir / config_name).exists() for config_dir in config_dirs):
            missing.append(config_name)
    if missing:
        raise AssertionError(
            "packaged OPP kernel configs are missing: " + ", ".join(missing)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-triton",
        action="store_true",
        help="Deprecated compatibility flag. Triton API validation is opt-in by default.",
    )
    parser.add_argument(
        "--check-triton",
        action="store_true",
        help="Also validate fla_npu.ops.triton exports. This may initialize the triton backend.",
    )
    parser.add_argument(
        "--check-legacy-torch-npu",
        action="store_true",
        help="Also validate optional torch_npu.ops compatibility exports.",
    )
    args = parser.parse_args()

    import fla_npu
    from fla_npu.ops import ascendc

    if fla_npu.is_legacy_torch_ops_loaded():
        raise AssertionError("packaged wheel API check should not load legacy torch.ops.npu extension")
    if "torch_npu" in sys.modules and not args.check_legacy_torch_npu:
        raise AssertionError("packaged wheel API check should not import torch_npu by default")

    for name in ASCENDC_NAMES:
        _require_attr(ascendc, name, "fla_npu.ops.ascendc")
        _require_attr(ascendc, f"npu_{name}", "fla_npu.ops.ascendc")

    if args.check_legacy_torch_npu:
        import torch_npu

        ascendc.install_torch_npu_ops_compat()
        for name in ASCENDC_NAMES:
            _require_attr(torch_npu.ops, name, "torch_npu.ops")
            _require_attr(torch_npu.ops, f"npu_{name}", "torch_npu.ops")

    _require_packaged_opp_configs(fla_npu)

    if ascendc.BACKWARD_OPS.get("causal_conv1d") != "causal_conv1d_bwd":
        raise AssertionError("causal_conv1d backward binding metadata is missing")

    if args.check_triton and not args.skip_triton:
        from fla_npu.ops import triton

        for name in TRITON_NAMES:
            _require_attr(triton, name, "fla_npu.ops.triton")

    print("Packaged wheel API check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
