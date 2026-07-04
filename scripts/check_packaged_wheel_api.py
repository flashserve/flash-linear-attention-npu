"""Validate the installed flash-linear-attention-npu wheel API surface."""

from __future__ import annotations

import argparse
from pathlib import Path


ASCENDC_NAMES = (
    "fast_gelu_custom",
    "fast_gelu_custom_backward",
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
)

TRITON_NAMES = (
    "causal_conv1d",
    "causal_conv1d_triton",
    "chunk_local_cumsum",
    "chunk_scaled_dot_kkt_fwd",
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
    parser.add_argument("--skip-triton", action="store_true", help="Only validate Ascend C APIs.")
    args = parser.parse_args()

    import fla_npu
    import torch_npu
    from fla_npu.ops import ascendc

    for name in ASCENDC_NAMES:
        _require_attr(ascendc, name, "fla_npu.ops.ascendc")
        _require_attr(ascendc, f"npu_{name}", "fla_npu.ops.ascendc")
        _require_attr(torch_npu.ops, name, "torch_npu.ops")
        _require_attr(torch_npu.ops, f"npu_{name}", "torch_npu.ops")

    _require_packaged_opp_configs(fla_npu)

    if ascendc.BACKWARD_OPS.get("causal_conv1d") != "causal_conv1d_bwd":
        raise AssertionError("causal_conv1d backward binding metadata is missing")
    if ascendc.BACKWARD_OPS.get("fast_gelu_custom") != "fast_gelu_custom_backward":
        raise AssertionError("fast_gelu_custom backward binding metadata is missing")

    if not args.skip_triton:
        from fla_npu.ops import triton

        for name in TRITON_NAMES:
            _require_attr(triton, name, "fla_npu.ops.triton")

    print("Packaged wheel API check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
