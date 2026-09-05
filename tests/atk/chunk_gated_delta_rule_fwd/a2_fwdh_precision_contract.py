#!/usr/bin/env python3
"""校验 A2 融合 GDN 私有 FwdH 的精度修复合同。"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARCH22_OPERATORS = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_kernel/internal/arch22/operators"
)
FWD_H_KERNEL = ARCH22_OPERATORS / (
    "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
)
FWD_H_VNEW_EPILOGUE = ARCH22_OPERATORS / (
    "chunk_gated_delta_rule_fwd_h/op_kernel/epilogue/block/"
    "block_epilogue_gdn_fwdh_vnew.hpp"
)
FUSED_HO_KERNEL = ARCH22_OPERATORS / (
    "chunk_recompute_wu_fwd_ho/op_kernel/chunk_recompute_wu_fwd_ho.cpp"
)


def _read_text(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        resolved = Path("\\\\?\\" + str(resolved))
    return resolved.read_text(encoding="utf-8")


def _partial_call(source: str, marker: str) -> str:
    call = source[source.index(marker) :]
    return call[: call.index("} else {")]


def check_contracts() -> dict[str, object]:
    kernel = _read_text(FWD_H_KERNEL)
    epilogue = _read_text(FWD_H_VNEW_EPILOGUE)
    fused = _read_text(FUSED_HO_KERNEL)
    tail_h = kernel.split("void ComputeTailHWorkspace", maxsplit=1)[1].split(
        "void PresetVectorPipelineEvents", maxsplit=1
    )[0]
    c1 = _partial_call(kernel, "if (cube1Offsets.blockTokens < chunkSize)")
    c2 = _partial_call(kernel, "if (cube2Offsets.blockTokens < chunkSize)")
    ordered_writeback = (
        "gmHWorkspace[offsets.hWorkOffset + kRow * offsets.vBlockDim],\n"
        "                accumUb, offsets.vBlockDim);\n"
        "            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);\n"
        "            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);\n"
        "            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);\n"
        "            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);"
    )
    checks = {
        "tail_c2_mte3_retired": ordered_writeback in tail_h,
        "partial_c1_uses_exact_shape": (
            "cube1Shape);" in c1 and "EmptyClass{}, true" not in c1
        ),
        "partial_c2_uses_exact_shape": (
            "cube2Shape);" in c2 and "EmptyClass{}, true" not in c2
        ),
        "embedded_schedule_preserved": (
            "InputT, GT, StateT, float, TileShapes, kGated, true, false, true>"
            in fused
        ),
        "no_unbalanced_fwdh_subblock_barrier": (
            "CrossCoreBarrier<0x1, PIPE_MTE3>" not in kernel
            and "CrossCoreBarrier<0x1, PIPE_MTE3>" not in epilogue
        ),
    }
    return {
        "schema": "gdn-a2-fwdh-precision-contract/v2",
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    result = check_contracts()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
