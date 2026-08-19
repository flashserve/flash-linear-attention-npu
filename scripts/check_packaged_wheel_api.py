"""Validate the installed flash-linear-attention-npu wheel API surface."""

from __future__ import annotations

import argparse
import os
import subprocess
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


def _require_safe_packaged_opapi(fla_npu_module) -> None:
    package_root = Path(fla_npu_module.__file__).resolve().parent
    vendor_dirs = list((package_root / "opp" / "vendors").glob("*"))
    if not vendor_dirs:
        raise AssertionError("packaged OPP vendor directory is missing")

    custom_libraries = [
        vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
        for vendor_dir in vendor_dirs
    ]
    if not any(path.is_file() for path in custom_libraries):
        raise AssertionError("packaged libcust_opapi.so is missing")

    conflicting_libraries = []
    for vendor_dir in vendor_dirs:
        candidate = vendor_dir / "op_api" / "lib" / "libopapi.so"
        if candidate.exists() or candidate.is_symlink():
            conflicting_libraries.append(candidate)
    if conflicting_libraries:
        raise AssertionError(
            "packaged custom OPP must not contain CANN library alias libopapi.so: "
            + ", ".join(str(path) for path in conflicting_libraries)
        )

    configured_library = Path(os.environ.get("FLA_NPU_OP_API_LIB", "")).resolve()
    packaged_libraries = {
        path.resolve() for path in custom_libraries if path.is_file()
    }
    if configured_library not in packaged_libraries:
        raise AssertionError(
            "FLA_NPU_OP_API_LIB must point to the packaged libcust_opapi.so"
        )

    packaged_library = next(path for path in custom_libraries if path.is_file())
    dynamic = subprocess.check_output(
        ["readelf", "-d", str(packaged_library)],
        encoding="utf-8",
        errors="replace",
    )
    if "(SONAME)" in dynamic:
        raise AssertionError(
            "packaged libcust_opapi.so must not define a SONAME: "
            f"{packaged_library}"
        )

    opapi_dirs = {str(path.parent.resolve()) for path in custom_libraries}
    ld_library_dirs = {
        str(Path(path).resolve())
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
    }
    if opapi_dirs & ld_library_dirs:
        raise AssertionError("packaged FLA op_api directory leaked into LD_LIBRARY_PATH")

    extensions = list(package_root.glob("custom_aclnn_extension_lib*.so"))
    if not extensions:
        raise AssertionError("packaged legacy custom_aclnn_extension_lib is missing")
    extension_dynamic = subprocess.check_output(
        ["readelf", "-d", str(extensions[0])],
        encoding="utf-8",
        errors="replace",
    )
    if "/op_api/lib" in extension_dynamic:
        raise AssertionError("legacy extension still contains the FLA op_api RUNPATH")


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
    _require_safe_packaged_opapi(fla_npu)

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
