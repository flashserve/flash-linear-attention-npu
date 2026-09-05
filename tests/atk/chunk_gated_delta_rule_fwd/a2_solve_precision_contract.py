#!/usr/bin/env python3
"""校验 A2 融合 GDN 的 BT64 FP32 SolveTri 对齐合同。"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FUSED_ROOT = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd"
)
FUSED_CMAKE = FUSED_ROOT / "op_host/CMakeLists.txt"
ARCH22_TILING = FUSED_ROOT / (
    "op_host/op_tiling/arch22/chunk_gated_delta_rule_fwd_arch22_tiling.cpp"
)
ARCH22_KERNEL = FUSED_ROOT / (
    "op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp"
)
COMBINED_SOLVE = FUSED_ROOT / (
    "op_kernel/internal/arch22/operators/chunk_kkt_solve_tri/op_kernel/"
    "chunk_cumsum_kkt_solve_tri.cpp"
)
STAGING = FUSED_ROOT / (
    "op_kernel/internal/arch22/operators/chunk_kkt_solve_tri/op_kernel/"
    "solve_layout_staging.h"
)


def _read_text(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        resolved = Path("\\\\?\\" + str(resolved))
    return resolved.read_text(encoding="utf-8")


def check_contracts() -> dict[str, object]:
    cmake = _read_text(FUSED_CMAKE)
    tiling = _read_text(ARCH22_TILING)
    kernel = _read_text(ARCH22_KERNEL)
    combined = _read_text(COMBINED_SOLVE)
    staging = _read_text(STAGING)
    checks = {
        "public_fp32_header_is_reused_read_only": (
            "-I${CMAKE_CURRENT_SOURCE_DIR}/../../solve_tri/op_kernel" in cmake
            and '#include "solve_tri_fp32.h"' in combined
        ),
        "bt64_uses_public_fp32_classes": (
            "if constexpr (MATRIX_SIZE == 64)" in combined
            and "NsSolveTri::SolveTriCubeFp32<T>" in combined
            and "NsSolveTri::SolveTriVectorFp32<T>" in combined
        ),
        "bt64_workspace_is_fp32_sized": (
            "FP32_SOLVE_WORKSPACE_SLOTS * abc.BT * abc.BT * sizeof(float)"
            in tiling
        ),
        "bt128_low_precision_workspace_is_preserved": (
            "LOW_PRECISION_SOLVE_WORKSPACE_SLOTS * abc.BT * abc.BT * sizeof(uint16_t)"
            in tiling
        ),
        "varlen_bt64_uses_tnd_staging": (
            "abc.BT == 64 && abc.isVarlen != 0" in kernel
            and "solveTiling.layoutMode = 2" in kernel
            and kernel.count("TransposeBhtTnd<InputT>") == 2
        ),
        "dense_bt64_retires_kkt_before_fp32_solve": (
            "The public SolveTri runs after a kernel boundary" in kernel
            and "AscendC::SyncAll<false>();\n        RunSolvePhase<InputT, 64>(aWorkspace"
            in kernel
        ),
        "staging_is_arch22_private": (
            "namespace NsPhase6SolveLayoutStaging" in staging
            and "GDN_PHASE6_SOLVE_LAYOUT_STAGING_H" in staging
        ),
    }
    return {
        "schema": "gdn-a2-solve-precision-contract/v1",
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    result = check_contracts()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
