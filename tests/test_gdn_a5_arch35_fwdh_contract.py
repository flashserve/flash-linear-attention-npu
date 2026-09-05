from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FWD_H = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/"
    "arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
)
STANDALONE_FWD_H = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/"
    "op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
)
STANDALONE_FWD_O = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/"
    "gdn_fwd_o_kernel.hpp"
)
FWD_H_VNEW = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/"
    "arch35/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
)
STANDALONE_FWD_H_VNEW = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/"
    "op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
)
FUSED_STATE_OUTPUT = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_kernel/internal/gated_delta_rule_state_update_output/"
    "chunk_gated_delta_rule_state_update_output.cpp"
)


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def test_short_tail_h_retires_mte3_before_reusing_accumulator():
    source = FWD_H.read_text(encoding="utf-8")
    tail_h = _between(source, "void ComputeTailHWorkspace", "void Process()")
    ordered_writeback = (
        "gmHWorkspace[offsets.hWorkOffset + kRow * offsets.vBlockDim],\n"
        "                accumUb, offsets.vBlockDim);\n"
        "            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(tailEventId);\n"
        "            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);\n"
        "            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(tailEventId);\n"
        "            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);"
    )

    assert ordered_writeback in tail_h


def test_partial_tiles_rely_on_actual_shape_without_full_l1_clear():
    for path in (FWD_H, STANDALONE_FWD_H):
        source = path.read_text(encoding="utf-8")
        c1 = _between(
            source,
            "if (cube1Offsets.blockTokens < chunkSize)",
            "} else {",
        )
        c2 = _between(
            source,
            "if (cube2Offsets.blockTokens < chunkSize)",
            "} else {",
        )

        assert "blockMmadWHTail(" in c1
        assert "cube1Shape);" in c1
        assert "EmptyClass{}, true" not in c1
        assert "blockMmadKVTail(" in c2
        assert "cube2Shape);" in c2
        assert "EmptyClass{}, true" not in c2


def test_embedded_fwdh_matches_standalone_compile_time_mode():
    source = FUSED_STATE_OUTPUT.read_text(encoding="utf-8")

    assert (
        "InputT, GT, StateT, float, TileShapes, kGated, true, false, false>"
        in source
    )
    assert (
        "InputT, GT, StateT, float, TileShapes, kGated, true, false, true>"
        not in source
    )


def test_a5_fwdh_v2_joins_both_aiv_subblocks_before_cube_reuse():
    join = "Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();"
    publish = "Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);"

    for path in (FWD_H, STANDALONE_FWD_H):
        source = path.read_text(encoding="utf-8")
        handoff = source[: source.index(publish)]
        assert handoff.rfind(join) > handoff.rfind("cube2Done[streamId]")


def test_standalone_fwdo_joins_aiv_subblocks_before_shared_publications():
    source = STANDALONE_FWD_O.read_text(encoding="utf-8")
    join = "Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();"

    for publish in (
        "Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done[streamId]);",
        "Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);",
    ):
        before = source[: source.index(publish)]
        assert before.rfind(join) > before.rfind("epilogueGDNFwdO")


def test_a5_vnew_publication_drains_gm_write_and_joins_aiv_subblocks():
    direct_publish = "Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);"
    output_write = "CopyUbToGm(vnewOutputThisTile, vNewOutputUbTensor"
    wait_write = "AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>"

    for path in (FWD_H_VNEW, STANDALONE_FWD_H_VNEW):
        source = path.read_text(encoding="utf-8")
        helper = _between(source, "void PublishVec1Done", "private:")
        tiled = _between(source, "for (uint32_t rowStart = rowBegin", "if constexpr (kGated)")

        assert source.count(direct_publish) == 1
        assert "Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();" in helper
        assert direct_publish in helper
        assert output_write in tiled
        assert wait_write in tiled[tiled.index(output_write) :]
        assert "PublishVec1Done(vec1Done);" in tiled[tiled.index(output_write) :]
