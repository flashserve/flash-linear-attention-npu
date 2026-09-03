from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE6 = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp"
)
KKT_CUBE = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_kernel/internal/coefficient_generation/chunk_gated_delta_rule_kkt_cube.h"
)


def test_cumsum_is_published_globally_before_coefficient_epilogue():
    source = PHASE6.read_text(encoding="utf-8")
    handoff = source.split(
        "if ASCEND_IS_AIV {\n"
        "        RunPhase6Cumsum(rawG, cuSeqlens, chunkIndices, gCumsumBht, coefficient);\n"
        "    }",
        maxsplit=1,
    )[1].split("kkt.ProcessEpilogueForSolve", maxsplit=1)[0]

    assert "AscendC::SyncAll<false>();" in handoff
    assert "CrossCoreWaitFlag(PHASE6_SCORE_READY_FLAG)" not in handoff


def test_a5_catlass_score_tail_keeps_physical_bt_column_stride():
    source = KKT_CUBE.read_text(encoding="utf-8")
    catlass_path = source.split("ProcessAscend950Catlass", maxsplit=1)[1]

    assert (
        "Catlass::GemmCoord shape{static_cast<uint32_t>(valid), BT_VALUE, K_DIM};"
        in catlass_path
    )
    assert (
        "Catlass::GemmCoord shape{static_cast<uint32_t>(valid), "
        "static_cast<uint32_t>(valid), K_DIM};"
        not in catlass_path
    )
