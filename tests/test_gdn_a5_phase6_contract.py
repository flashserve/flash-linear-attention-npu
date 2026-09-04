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
COEFFICIENT = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_kernel/internal/coefficient_generation/"
    "chunk_gated_delta_rule_coefficient_generation.cpp"
)
EMBEDDED_FWD_O = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_kernel/internal/operators/chunk_fwd_o/op_kernel/gemm/kernel/"
    "gdn_fwd_o_kernel.hpp"
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


def test_a5_kkt_epilogue_joins_both_aiv_subblocks_before_solve():
    source = COEFFICIENT.read_text(encoding="utf-8")
    a5_path = source.split("#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310", maxsplit=1)[1]
    handoff = a5_path.split("if ASCEND_IS_AIV", maxsplit=1)[1].split("// Phase6 passes", maxsplit=1)[0]

    join = "Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();"
    publish = "CrossCoreSetFlag<0x2, PIPE_MTE3>(KKT_READY_FLAG);"
    assert join in handoff
    assert publish in handoff
    assert handoff.index(join) < handoff.index(publish)


def test_solved_a_joins_fix_and_both_aiv_mte3_writes_before_recompute():
    source = PHASE6.read_text(encoding="utf-8")
    handoff = source.split(
        "// SolveTri may publish A through AIC FIX or AIV MTE3.", maxsplit=1
    )[1].split("GM_ADDR w", maxsplit=1)[0]

    set_fix_mte2 = (
        "AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>"
        "(PHASE6_SOLVE_FIX_TO_MTE2_EVENT);"
    )
    wait_fix_mte2 = (
        "AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>"
        "(PHASE6_SOLVE_FIX_TO_MTE2_EVENT);"
    )
    join_aivs = "Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();"
    publish_aivs = (
        "AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>"
        "(PHASE6_SOLVE_AIV_DONE_FLAG);"
    )
    wait_aivs = "AscendC::CrossCoreWaitFlag(PHASE6_SOLVE_AIV_DONE_FLAG);"
    publish = "AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(PHASE6_SOLVE_DONE_FLAG);"
    assert join_aivs in handoff
    assert publish_aivs in handoff
    assert set_fix_mte2 in handoff
    assert wait_fix_mte2 in handoff
    assert wait_aivs in handoff
    assert publish in handoff
    assert handoff.index(join_aivs) < handoff.index(publish_aivs)
    assert (
        handoff.index(set_fix_mte2)
        < handoff.index(wait_fix_mte2)
        < handoff.index(wait_aivs)
        < handoff.index(publish)
    )
    assert "AscendC::SyncAll<false>();" not in handoff


def test_embedded_fwdo_joins_aiv_subblocks_before_shared_publications():
    source = EMBEDDED_FWD_O.read_text(encoding="utf-8")
    join = "Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();"

    for publish in (
        "Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done[streamId]);",
        "Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);",
    ):
        before = source[: source.index(publish)]
        assert before.rfind(join) > before.rfind("epilogueGDNFwdO")
