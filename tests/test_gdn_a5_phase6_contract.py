from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE6 = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp"
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
