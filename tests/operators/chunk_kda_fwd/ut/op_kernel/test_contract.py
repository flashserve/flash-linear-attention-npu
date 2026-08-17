"""Static kernel/tiling contract for chunk_kda_fwd."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OP_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd"
LEGACY_COMMON_KERNEL = ROOT / "fla/ops/ascendc/common/kda/chunk_kda_fwd_kernel.hpp"
CASE_MANIFEST = ROOT / "tests/op_cases/chunk_kda_fwd.json"
DIRECT_SOURCE = (
    ROOT
    / "examples/fast_kernel_launch_example/csrc/chunk_kda_fwd/chunk_kda_fwd_direct.cpp"
)
KERNEL_ENTRY = OP_ROOT / "op_kernel/chunk_kda_fwd.cpp"
TILING_ENTRY = OP_ROOT / "op_host/chunk_kda_fwd_tiling.cpp"
GENERIC_STAGE_IMPLEMENTATIONS = {
    "prepare": OP_ROOT / "op_kernel/chunk_kda_fwd_prepare.h",
    "post_wu": OP_ROOT / "op_kernel/chunk_kda_fwd_post_wu.h",
    "output": OP_ROOT / "op_kernel/chunk_kda_fwd_finalize.h",
}
ARCH35_STAGE_IMPLEMENTATIONS = {
    "prepare": OP_ROOT / "op_kernel/arch35/chunk_kda_fwd_prepare.h",
    "post_wu": OP_ROOT / "op_kernel/arch35/chunk_kda_fwd_post_wu.h",
    "output": OP_ROOT / "op_kernel/arch35/chunk_kda_fwd_finalize.h",
}
STAGE_IMPLEMENTATIONS = ARCH35_STAGE_IMPLEMENTATIONS
ARCH35_KERNEL = OP_ROOT / "op_kernel/arch35/chunk_kda_fwd_impl.h"
ARCH35_FWD_H = OP_ROOT / "op_kernel/arch35/chunk_kda_fwd_fwd_h.h"
KERNEL_COMMON = OP_ROOT / "op_kernel/chunk_kda_fwd_common.h"
COMPACT_PLAN = OP_ROOT / "op_kernel/chunk_kda_fwd_plan.h"
ARCH35_TILING = OP_ROOT / "op_host/arch35/chunk_kda_fwd_tiling_impl.h"
ARCH35_GDN_SCHEDULER = (
    ROOT
    / "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
    "chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/block/"
    "block_scheduler_gdn_fwd_h.hpp"
)
ARCH35_GDN_KERNEL = (
    ROOT
    / "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
    "chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/"
    "gdn_fwd_h_kernel.hpp"
)
GENERIC_GDN_KERNEL = (
    ROOT
    / "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
    "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/"
    "gdn_fwd_h_kernel.hpp"
)
REMOVED_FUSED_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd_fused_a5"
REMOVED_STAGE_ROOTS = (
    ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd_prepare",
    ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd_post_wu",
    ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd_finalize",
)


def test_direct_launch_keeps_registered_public_entry():
    text = DIRECT_SOURCE.read_text(encoding="utf-8")
    assert "chunk_kda_fwd_direct" in text
    registration = text.split("TORCH_LIBRARY_FRAGMENT", 1)[1]
    assert 'm.def("chunk_kda_fwd_direct(' in registration
    assert registration.count('m.impl("chunk_kda_fwd_direct"') == 2


def test_direct_launch_stage_calls_follow_compact_plan_signatures():
    text = DIRECT_SOURCE.read_text(encoding="utf-8")
    prepare = " ".join(
        text.split("void ChunkKdaPrepareDirectKernel(", 1)[1]
        .split("void ChunkKdaPostWuDirectKernel(", 1)[0]
        .split()
    )
    post_wu = " ".join(
        text.split("void ChunkKdaPostWuDirectKernel(", 1)[1]
        .split("void RunChunkKdaFwdHDirect(", 1)[0]
        .split()
    )
    output = " ".join(
        text.split("void ChunkKdaOutputDirectKernel(", 1)[1]
        .split("void LaunchStages(", 1)[0]
        .split()
    )

    assert (
        "KdaPrepare::RunChunkKdaPrepare<SAFE_GATE, T, float, float, "
        "float, float, DirectKdaTilingData, 0, 0, 0>(" in prepare
    )
    assert (
        "q, k, v, gk, nullptr, nullptr, nullptr, beta, initialState, "
        "nullptr, nullptr, nullptr, aqk, akk, qg," in prepare
    )
    assert (
        "userWorkspace + tiling.prepareUSeedOffset, kg, userWorkspace, "
        "tiling, pipe, true);"
        in prepare
    )
    assert (
        "q, k, v, gk, beta, initialState, nullptr, nullptr, nullptr, "
        "userWorkspace + tiling.postWuWSeedOffset" in post_wu
    )
    assert (
        "q, k, v, gk, beta, initialState, nullptr, nullptr, nullptr, "
        "userWorkspace + tiling.outputQgScaledOffset" in output
    )
    release_helper = text.split(
        "__aicore__ inline void ReleaseAicPipeReservedMmadEvents(", 1
    )[1].split("template <bool SAFE_GATE", 1)[0]
    assert "__NPU_ARCH__ == 2201" in release_helper
    assert "pipe.DestroyWithoutPipeAll();" in release_helper
    assert "__NPU_ARCH__ == 3510" in release_helper
    assert "pipe.Destroy();" in release_helper
    assert 'op_kernel/arch35/chunk_kda_fwd_prepare.h"' in text
    assert 'op_kernel/chunk_kda_fwd_prepare.h"' in text
    assert text.count("#define KDA_ENABLE_COMPACT_PLAN_VIEW 1") == 1
    assert text.count("#undef KDA_ENABLE_COMPACT_PLAN_VIEW") == 1
    assert text.index("#define KDA_ENABLE_COMPACT_PLAN_VIEW 1") < text.index(
        'op_kernel/chunk_kda_fwd_plan.h"'
    )
    assert text.index("#undef KDA_ENABLE_COMPACT_PLAN_VIEW") > text.index(
        'op_kernel/chunk_kda_fwd_prepare.h"'
    )
    assert text.count("ReleaseAicPipeReservedMmadEvents(pipe);") == 3
    assert "DirectFwdHTilingView stateTiling{};" in text
    assert "stateTiling.useCompactSequencePlan = false;" in text
    assert "stateTiling.compactPlan = nullptr;" in text
    assert "tiling.hasInitialState = initialState.has_value();" in text

    plan = COMPACT_PLAN.read_text(encoding="utf-8")
    assert plan.count(
        "#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)"
    ) == 2
    assert plan.count("defined(KDA_ENABLE_COMPACT_PLAN_VIEW)") == 1


def test_a2_block_mmad_alignment_checks_use_target_cann_namespace():
    block_root = ROOT / "fla/ops/ascendc/common/kernel_utils/block"
    for name in (
        "block_mmad_pingpong_tla.hpp",
        "block_mmad_pingpong_tla_multi.hpp",
        "block_mmad_pingpong_tla_preloadA_l1B.hpp",
    ):
        text = (block_root / name).read_text(encoding="utf-8")
        alignment_checks = text.split("static constexpr uint32_t _32B", 1)[1].split(
            "#endif", 1
        )[0]
        assert alignment_checks.count("AscendC::SizeOfBits<") == 5
        assert not re.search(r"(?<!AscendC::)SizeOfBits<", alignment_checks)


def test_a5_reuses_public_physical_entry_with_internal_stage_implementations():
    assert not LEGACY_COMMON_KERNEL.exists()
    assert KERNEL_ENTRY.exists()
    assert TILING_ENTRY.exists()
    assert all(path.exists() for path in GENERIC_STAGE_IMPLEMENTATIONS.values())
    assert all(path.exists() for path in ARCH35_STAGE_IMPLEMENTATIONS.values())
    assert not REMOVED_FUSED_ROOT.exists()
    assert all(not path.exists() for path in REMOVED_STAGE_ROOTS)
    assert ARCH35_KERNEL.exists() and ARCH35_FWD_H.exists() and ARCH35_TILING.exists()
    assert KERNEL_COMMON.exists()

    entry = KERNEL_ENTRY.read_text(encoding="utf-8")
    assert entry.count('extern "C" __global__ __aicore__ void chunk_kda_fwd(') == 1
    assert '#include "arch35/chunk_kda_fwd_impl.h"' in entry
    assert not (OP_ROOT / "op_kernel/chunk_kda_fwd_impl.h").exists()

    common = KERNEL_COMMON.read_text(encoding="utf-8")
    prepare = ARCH35_STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    assert '#include "chunk_kda_fwd_prepare.h"' in common
    assert '#include "arch35/chunk_kda_fwd_prepare.h"' in common
    assert '#include "chunk_kda_fwd_finalize.h"' in common
    assert '#include "arch35/chunk_kda_fwd_finalize.h"' in common
    assert '#include "chunk_kda_fwd_post_wu.h"' in prepare
    assert "common/kda/chunk_kda_fwd_kernel.hpp" not in common

    for path in GENERIC_STAGE_IMPLEMENTATIONS.values():
        generic = path.read_text(encoding="utf-8")
        assert "__CCE_AICORE__" not in generic
        assert "A5" not in generic
        assert "Arch35" not in generic


def test_cross_core_sync_uses_group_mode2_without_mode4_subblock_offsets():
    all_sources = [
        KERNEL_ENTRY,
        KERNEL_COMMON,
        *GENERIC_STAGE_IMPLEMENTATIONS.values(),
        ARCH35_KERNEL,
        ARCH35_FWD_H,
        *ARCH35_STAGE_IMPLEMENTATIONS.values(),
    ]
    for path in all_sources:
        text = path.read_text(encoding="utf-8")
        assert "CrossCoreSetFlag<0x4" not in text
        assert "CrossCoreWaitFlag<0x4" not in text

    prepare = ARCH35_STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    fwd_h = ARCH35_FWD_H.read_text(encoding="utf-8")
    assert "KDA_DIRECT_SCORE_SUBBLOCK_FLAG_STRIDE" not in prepare
    assert "KDA_FWD_H_SUBBLOCK_FLAG_OFFSET" not in fwd_h
    assert "CrossCoreBarrier<0x1, PIPE_MTE3>();" in prepare
    assert "CrossCoreSetFlag<0x2, PIPE_MTE3>(syncReadyFlag_);" in prepare
    assert "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);" in prepare
    assert "CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FWD_H_DIRECT_READY_FLAG);" in fwd_h


def test_prepare_score_mmad_reuses_no_unit_double_l0c_across_aqk_and_akk():
    stage_sources = (
        *GENERIC_STAGE_IMPLEMENTATIONS.values(),
        *ARCH35_STAGE_IMPLEMENTATIONS.values(),
    )
    prepare_sources = (
        GENERIC_STAGE_IMPLEMENTATIONS["prepare"],
        ARCH35_STAGE_IMPLEMENTATIONS["prepare"],
    )
    score_policy = (
        "Catlass::Gemm::MmadPingpongTlaMulti<KdaArchTag, "
        "false, false, 2, true, 2, 1, 2, 2>"
    )

    for path in stage_sources:
        text = path.read_text(encoding="utf-8")
        assert "PipeBarrier<PIPE_ALL>" not in text
    for path in prepare_sources:
        text = path.read_text(encoding="utf-8")
        score = text.split(
            "__aicore__ inline void ComputeRawAqkAkkCubeBlock", 1
        )[1].split("__aicore__ inline bool UseAkkCubeSolve", 1)[0]
        assert score_policy in text
        assert "BlockMmadTla<KdaScoreDispatchPolicy" in score
        assert score.count("blockMmad.preSetFlags();") == 1
        assert score.count("blockMmad.finalWaitFlags();") == 1
        aqk = score.index("blockMmad(blockQPos, blockKNeg, blockAqk, shape);")
        akk = score.index("blockMmad(blockKPos, blockKNeg, blockAkk, shape);")
        assert score.index("blockMmad.preSetFlags();") < aqk < akk < score.index(
            "blockMmad.finalWaitFlags();"
        )
    for stage_map in (GENERIC_STAGE_IMPLEMENTATIONS, ARCH35_STAGE_IMPLEMENTATIONS):
        for stage_name in ("post_wu", "output"):
            assert "KdaScoreDispatchPolicy" not in stage_map[stage_name].read_text(
                encoding="utf-8"
            )


def test_post_wu_limits_stage2_policy_to_runtime_full_head_windows():
    generic = GENERIC_STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    arch35 = ARCH35_STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    single = generic.split(
        "__attribute__((noinline)) __aicore__ void ComputePostWuCube", 1
    )[1].split(
        "__attribute__((noinline)) __aicore__ void ComputeCompactPostWuCubeHeadWindow", 1
    )[0]
    full_window = generic.split(
        "__attribute__((noinline)) __aicore__ void ComputeCompactPostWuCubeHeadWindow", 1
    )[1].split(
        "__attribute__((noinline)) __aicore__ void CopyScratchWAndFinalizeKg", 1
    )[0]
    full_dispatch = generic.split(
        "__aicore__ inline void ProcessCompactPostAicHeadWindow", 1
    )[1].split("__aicore__ inline void ProcessCompactPostAicHeadRange", 1)[0]
    arch35_single = arch35.split(
        "__aicore__ inline void ComputePostWuCube(uint64_t b", 1
    )[1].split("__aicore__ inline bool UseTypicalPostWuGate", 1)[0]
    single_flat = " ".join(single.split())
    arch35_single_flat = " ".join(arch35_single.split())

    assert (
        "using KdaDispatchPolicy = "
        "Common::MmadPingpong<KdaArchTag, false, false, 2>;"
        in generic
    )
    assert (
        "using KdaSingleDispatchPolicy = "
        "Common::MmadPingpong<KdaArchTag, false, false, 1>;"
        in generic
    )
    assert (
        "using KdaWideDispatchPolicy = "
        "Common::MmadPingpong<KdaArchTag, false, false, 1>;"
        in generic
    )
    assert "单项路径没有跨调用的 BlockMmad 生命周期" in generic
    assert "K/V > 128 的 256 列 tile 会占满 A2 L0C" in generic
    assert (
        "using WBlockMmad128 = Common::BlockMmadTla<KdaSingleDispatchPolicy, "
        "PostL1TileShape128, PostL0TileShape128, ElementA, ElementB, float, void, "
        "WTileCopy>;"
        in single_flat
    )
    assert (
        "using WBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy, "
        "PostL1TileShape256, PostL0TileShape256, ElementA, ElementB, float, void, "
        "WTileCopy>;"
        in single_flat
    )
    assert "using UBlockMmad128 = Common::BlockMmadTla<KdaSingleDispatchPolicy" in single
    assert "using UBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy" in single
    assert single.count("if (K_ <= 128)") == 1
    w_dispatch = single.split("Catlass::Arch::Resource<KdaArchTag> wResource;", 1)[1].split(
        "// 单项路径", 1
    )[0]
    narrow_w, wide_w = w_dispatch.split("} else {", 1)
    assert (
        "WBlockMmad128 wBlockMmad(wResource); ComputePostWuW(wBlockMmad, b, hv, "
        "chunkIdx, start, curT);"
        in " ".join(narrow_w.split())
    )
    assert (
        "WBlockMmad256 wBlockMmad(wResource); ComputePostWuW(wBlockMmad, b, hv, "
        "chunkIdx, start, curT);"
        in " ".join(wide_w.split())
    )

    assert "using WBlockMmad = Common::BlockMmadTla<KdaDispatchPolicy" in full_window
    assert "using UBlockMmad = Common::BlockMmadTla<KdaDispatchPolicy" in full_window
    assert full_window.count("WBlockMmad wBlockMmad(wResource);") == 1
    assert full_window.count("UBlockMmad uBlockMmad(uResource);") == 1
    assert full_window.count(
        "for (uint32_t lane = 0; lane < headCnt; ++lane)"
    ) == 3
    w_owner = full_window.index("WBlockMmad wBlockMmad(wResource);")
    w_loop = full_window.index(
        "for (uint32_t lane = 0; lane < headCnt; ++lane)", w_owner
    )
    u_owner = full_window.index("UBlockMmad uBlockMmad(uResource);", w_loop)
    u_loop = full_window.index(
        "for (uint32_t lane = 0; lane < headCnt; ++lane)", u_owner
    )
    publish = full_window.index(
        "CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);", u_loop
    )
    assert w_owner < w_loop < u_owner < u_loop < publish
    assert "BT_ == 64 && K_ == 128 && V_ == 128" in full_dispatch
    assert "ComputeCompactPostWuCubeHeadWindow" in full_dispatch
    assert "headBase, headCnt" in full_dispatch

    assert (
        "using KdaDispatchPolicy = "
        "Common::MmadPingpong<KdaArchTag, false, false, 2>;"
        in arch35
    )
    assert (
        "using KdaWideDispatchPolicy = "
        "Common::MmadPingpong<KdaArchTag, false, false, 1>;"
        in arch35
    )
    for output_type in ("T", "float"):
        assert arch35_single_flat.count(
            "using WBlockMmad128 = Common::BlockMmadTla<KdaDispatchPolicy, "
            "PostL1TileShape128, PostL0TileShape128, ElementA, ElementB, "
            f"{output_type}, void, WTileCopy>;"
        ) == 1
        assert arch35_single_flat.count(
            "using WBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy, "
            "PostL1TileShape256, PostL0TileShape256, ElementA, ElementB, "
            f"{output_type}, void, WTileCopy>;"
        ) == 1
    assert "using UBlockMmad128 = Common::BlockMmadTla<KdaDispatchPolicy" in arch35
    assert "using UBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy" in arch35
    assert arch35_single.count("if (K_ <= 128)") == 1
    arch35_w_dispatch = arch35_single.split(
        "Catlass::Arch::Resource<KdaArchTag> wResource;", 1
    )[1].split("// 离开当前作用域", 1)[0]
    arch35_narrow_w, arch35_wide_w = arch35_w_dispatch.split("} else {", 1)
    expected_arch35_call = "wBlockMmad(blockA, blockB, blockC, shape);"
    assert "WBlockMmad128 wBlockMmad(wResource);" in arch35_narrow_w
    assert expected_arch35_call in arch35_narrow_w
    assert "WBlockMmad256 wBlockMmad(wResource);" in arch35_wide_w
    assert expected_arch35_call in arch35_wide_w

    for source in (single, arch35_single):
        assert "using PostL1TileShape256 = tla::Shape<KdaInt128, tla::_256, tla::_256>;" in source
        assert "using PostL0TileShape256 = tla::Shape<KdaInt128, tla::_256, KdaInt64>;" in source

    opapi = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    assert "constexpr int64_t MAX_KDA_K_DIM = 256;" in opapi
    assert "info.kDim >= 16 && info.kDim <= MAX_KDA_K_DIM && info.kDim % 16 == 0" in opapi
    element_bytes = 2
    wide_l1_bytes = (
        128 * 256 * element_bytes * 2 + 256 * 256 * element_bytes * 2
    )
    wide_l0a_bytes = 128 * 64 * element_bytes * 2
    wide_l0b_bytes = 64 * 256 * element_bytes * 2
    wide_l0c_bytes = 128 * 256 * 4
    assert wide_l1_bytes <= 512 * 1024
    assert wide_l0a_bytes <= 64 * 1024
    assert wide_l0b_bytes <= 64 * 1024
    assert wide_l0c_bytes <= 128 * 1024


def test_arch35_prepare_does_not_keep_dead_manual_unit_flag_paths():
    prepare = ARCH35_STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")

    for forbidden in (
        "KDA_ARCH35_ENABLE_MANUAL_SCORE_PIPELINE",
        "ComputeRawAqkAkkCubeFullArch35",
        "ComputeAkkMergeCubeWorkspaceArch35",
        "LoadSolveTile",
        "SetMMLayoutTransform",
        "0b11",
    ):
        assert forbidden not in prepare
    assert "__aicore__ inline void ComputeRawAqkAkkCubeBlock" in prepare
    assert "Common::BlockMmadTla<KdaSolveDispatchPolicy" in prepare


def test_prepare_solve_uses_single_l0c_and_closes_each_fix_to_mte2_raw():
    prepare_sources = (
        GENERIC_STAGE_IMPLEMENTATIONS["prepare"],
        ARCH35_STAGE_IMPLEMENTATIONS["prepare"],
    )

    for path in prepare_sources:
        text = path.read_text(encoding="utf-8")
        solve = text.split("__aicore__ inline void CubeGemmSolveSub", 1)[1].split(
            "__aicore__ inline void AddSolveTmpToX", 1
        )[0]
        assert (
            "using KdaSolveDispatchPolicy = "
            "Common::MmadPingpong<KdaArchTag, false, false, 1>;"
            in text
        )
        assert "KdaSolveDispatchPolicy = Common::MmadPingpong<KdaArchTag, false, false, 2>" not in text
        assert "Common::BlockMmadTla<KdaSolveDispatchPolicy" in solve
        assert solve.count(
            "SetFlag<HardEvent::FIX_MTE2>(solveFixToMte2Event_);"
        ) == 1
        assert solve.count(
            "WaitFlag<HardEvent::FIX_MTE2>(solveFixToMte2Event_);"
        ) == 1
        assert solve.index("blockMmad(blockA, blockB, blockC, shape);") < solve.index(
            "SetFlag<HardEvent::FIX_MTE2>(solveFixToMte2Event_);"
        )
        assert solve.index(
            "SetFlag<HardEvent::FIX_MTE2>(solveFixToMte2Event_);"
        ) < solve.index("WaitFlag<HardEvent::FIX_MTE2>(solveFixToMte2Event_);")
        assert "pipe_->AllocEventID<HardEvent::FIX_MTE2>()" in text
        assert "pipe_->ReleaseEventID<HardEvent::FIX_MTE2>" in text


def test_generic_gdn_fwd_h_scopes_l0c_lifecycle_over_runtime_stream_batch():
    kernel = GENERIC_GDN_KERNEL.read_text(encoding="utf-8")
    process = kernel.split("__aicore__ inline void Process()", 1)[1]
    process_aic = process.split("if ASCEND_IS_AIC", 1)[1].split(
        "if ASCEND_IS_AIV", 1
    )[0]

    assert "std::is_same<TileShapes, GDNFwdHTileShapes128>::value ? 2 : 1" in kernel
    assert (
        "Gemm::MmadPingpongTlaMulti<ArchTag, false, false, L0C_STAGES>"
        in kernel
    )
    for forbidden in (
        "DispatchPolicyTlaTail",
        "BlockMmadWHTail",
        "BlockMmadKVTail",
        "PipeBarrier<PIPE_ALL>",
    ):
        assert forbidden not in kernel

    assert process_aic.count("blockMmadWH.preSetFlags();") == 1
    assert process_aic.count("blockMmadWH.finalWaitFlags();") == 1
    assert process_aic.count("blockMmadKV.preSetFlags();") == 1
    assert process_aic.count("blockMmadKV.finalWaitFlags();") == 1
    wh_seed = process_aic.index("blockMmadWH.preSetFlags();")
    wh_loop = process_aic.index(
        "for (uint32_t i = 0; i < PING_PONG_STAGES; ++i)", wh_seed
    )
    wh_drain = process_aic.index("blockMmadWH.finalWaitFlags();", wh_loop)
    kv_seed = process_aic.index("blockMmadKV.preSetFlags();", wh_drain)
    kv_loop = process_aic.index(
        "for (uint32_t i = 0; i < PING_PONG_STAGES; ++i)", kv_seed
    )
    kv_drain = process_aic.index("blockMmadKV.finalWaitFlags();", kv_loop)
    assert wh_seed < wh_loop < wh_drain < kv_seed < kv_loop < kv_drain


def test_gdn_fwd_h_splits_cube2_rows_to_the_static_l0c_tile():
    generic = GENERIC_GDN_KERNEL.read_text(encoding="utf-8")
    arch35 = ARCH35_GDN_KERNEL.read_text(encoding="utf-8")

    for kernel in (generic, arch35):
        helper = kernel.split(
            "__aicore__ inline void ComputeCube2RowTiles", 1
        )[1].split("// vec 1", 1)[0]
        assert (
            "static constexpr uint32_t CUBE2_ROW_TILE_M = "
            "tla::get<0>(L0TileShapeVTla{});"
            in kernel
        )
        assert "rowOffset < kHeadDim" in helper
        assert "rowOffset += CUBE2_ROW_TILE_M" in helper
        assert "Min(CUBE2_ROW_TILE_M, kHeadDim - rowOffset)" in helper
        k_tile = helper.split("auto tensorBlockK = GetTile(", 1)[1].split(
            "auto tensorBlockHwork = GetTile(", 1
        )[0]
        hwork_tile = helper.split("auto tensorBlockHwork = GetTile(", 1)[1].split(
            "blockMmad(", 1
        )[0]
        assert "tensorK, tla::MakeCoord(rowOffset, 0)" in k_tile
        assert "tla::MakeShape(rowCount, blockTokens)" in k_tile
        assert "tensorHwork, tla::MakeCoord(rowOffset, 0)" in hwork_tile
        assert "tla::MakeShape(rowCount, vBlockDim)" in hwork_tile
        assert "cube2Shape, EmptyClass{}, clearL1Padding" in helper

    assert "GemmCoord cube2Shape{kHeadDim" not in generic

    generic_process = generic.split("__aicore__ inline void Process()", 1)[1]
    assert generic_process.count("ComputeCube2RowTiles(") == 1
    assert "cube2Offsets.blockTokens < chunkSize" in generic_process

    arch35_process = arch35.split("__aicore__ inline void Process()", 1)[1]
    assert arch35_process.count("ComputeCube2RowTiles(") == 3
    assert "blockMmadKVDirectUb(" in arch35_process
    assert arch35_process.count("GemmCoord cube2Shape{kHeadDim") == 1
    assert arch35.count("kHeadDim == 128 && vHeadDim == 128") == 2


def test_prepare_post_wu_finalize_share_one_runtime_head_window_protocol():
    stage_sources = (
        *ARCH35_STAGE_IMPLEMENTATIONS.values(),
        *GENERIC_STAGE_IMPLEMENTATIONS.values(),
    )
    forbidden_protocols = (
        "PAIR_HEADS",
        "HeadPair",
        "PairAligned",
        "activeHeadPairMode_",
    )
    fixed_window_step = re.compile(
        r"\b(?:head|headBase)\s*\+=\s*KDA_HEADS_PER_TASK\b"
    )
    runtime_window_step = re.compile(
        r"\b(?:head|headBase)\s*\+=\s*headCnt\b"
    )
    decode_with_runtime_heads = re.compile(
        r"DecodeChunkHeadGroupTask\(\s*"
        r"static_cast<uint32_t>\(task\),\s*"
        r"static_cast<uint32_t>\(H_\),\s*"
        r"static_cast<uint32_t>\(HV_\),\s*"
        r"chunkOrdinal,\s*begin,\s*end\)",
        re.MULTILINE,
    )

    assert "constexpr uint32_t KDA_HEADS_PER_TASK = 4;" in COMPACT_PLAN.read_text(
        encoding="utf-8"
    )
    for path in stage_sources:
        text = path.read_text(encoding="utf-8")
        for forbidden in forbidden_protocols:
            assert forbidden not in text
        assert not re.search(
            r"template\s*<[^>]*(?:HEAD_COUNT|HEAD_RATIO|HEADS_PER_TASK|"
            r"headCount|headRatio)",
            text,
        )
        assert "headsPerWindow" not in text
        assert "windowsPerQuery" not in text
        assert not fixed_window_step.search(text)
        assert not re.search(
            r"Process\w*Head(?:Window|Range|s)?\s*<[^>]*"
            r"KDA_HEADS_PER_TASK[^>]*>",
            text,
        )

        helper_count = text.count("KdaForward::HeadWindowHeadCount(")
        step_count = len(runtime_window_step.findall(text))
        assert helper_count > 0
        assert step_count == helper_count

        decode_count = text.count("DecodeChunkHeadGroupTask(")
        assert decode_count == 2
        assert len(decode_with_runtime_heads.findall(text)) == decode_count

    for count in (1, 2, 3, 4):
        explicit_count = re.compile(
            rf"Process\w*Head(?:Window|Range|s)?\s*<[^>]*,?\s*{count}\s*>"
        )
        for path in stage_sources:
            assert not explicit_count.search(path.read_text(encoding="utf-8"))


def test_runtime_head_window_properties_cover_all_supported_integer_gva_ratios():
    plan = COMPACT_PLAN.read_text(encoding="utf-8")

    def runtime_windows(query_heads, value_heads):
        ratio = value_heads // query_heads
        if ratio <= 4:
            width = (4 // ratio) * ratio
            starts = range(0, value_heads, width)
            return [(start, min(width, value_heads - start)) for start in starts]
        windows = []
        for query_head in range(query_heads):
            query_begin = query_head * ratio
            for local_begin in range(0, ratio, 4):
                windows.append((query_begin + local_begin, min(4, ratio - local_begin)))
        return windows

    def resident_loads(query_heads, value_heads):
        ratio = value_heads // query_heads
        loads = []
        for begin, count in runtime_windows(query_heads, value_heads):
            window_loads = []
            for value_head in range(begin, begin + count):
                query_head = value_head // ratio
                if not window_loads or window_loads[-1] != query_head:
                    window_loads.append(query_head)
            loads.append(window_loads)
        return loads

    ratios_seen = set()
    multi_query_ratios_seen = set()
    # 穷举能力范围内所有合法H/HV组合，并校验窗口和resident读取的共同性质。
    for query_heads in range(1, 129):
        for ratio in range(1, 128 // query_heads + 1):
            value_heads = query_heads * ratio
            shape = (query_heads, value_heads, ratio)
            ratios_seen.add(ratio)
            if query_heads > 1:
                multi_query_ratios_seen.add(ratio)

            windows = runtime_windows(query_heads, value_heads)
            assert windows[0][0] == 0, shape
            assert windows[-1][0] + windows[-1][1] == value_heads, shape
            assert all(0 < count <= 4 for _, count in windows), shape
            assert all(
                begin + count == next_begin
                for (begin, count), (next_begin, _) in zip(windows, windows[1:])
            ), shape
            assert [
                value_head
                for begin, count in windows
                for value_head in range(begin, begin + count)
            ] == list(range(value_heads)), shape

            if ratio <= 4:
                width = (4 // ratio) * ratio
                assert all(
                    begin % ratio == 0
                    and count % ratio == 0
                    and count == min(width, value_heads - begin)
                    for begin, count in windows
                ), shape
            else:
                assert all(
                    begin // ratio == (begin + count - 1) // ratio
                    and begin % ratio % 4 == 0
                    and count
                    == min(4, (begin // ratio + 1) * ratio - begin)
                    for begin, count in windows
                ), shape

            flattened_loads = [
                query_head
                for window_loads in resident_loads(query_heads, value_heads)
                for query_head in window_loads
            ]
            loads_per_query = 1 if ratio <= 4 else (ratio + 3) // 4
            assert len(flattened_loads) == query_heads * loads_per_query, shape
            assert flattened_loads == [
                query_head
                for query_head in range(query_heads)
                for _ in range(loads_per_query)
            ], shape

    assert ratios_seen == set(range(1, 129))
    assert set(range(1, 65)).issubset(multi_query_ratios_seen)

    # 非4整除比例单列，防止余数窗口在后续改动中被虚拟head补齐。
    assert runtime_windows(1, 7) == [(0, 4), (4, 3)]
    assert resident_loads(1, 7) == [[0], [0]]
    assert runtime_windows(1, 19) == [
        (0, 4),
        (4, 4),
        (8, 4),
        (12, 4),
        (16, 3),
    ]
    assert resident_loads(1, 19) == [[0], [0], [0], [0], [0]]

    # 长流会把headGroupCount压成1，ratio=3仍必须按3推进，不能退化成4步。
    assert runtime_windows(32, 96) == [(head, 3) for head in range(0, 96, 3)]
    long_stream = plan.split(
        "KDA_PLAN_INLINE uint32_t ComputeChunkHeadGroupCount", 1
    )[1].split("KDA_PLAN_INLINE uint32_t HeadGroupBegin", 1)[0]
    assert "groupingChunkCount >= physicalCoreCount" in long_stream
    assert "return 1;" in long_stream

    assert "(KDA_HEADS_PER_TASK / headRatio) * headRatio" in plan
    assert "queryHead * headRatio + localWindow * KDA_HEADS_PER_TASK" in plan
    assert "headsUntilQueryEnd < KDA_HEADS_PER_TASK" in plan
    owner = plan.split(
        "KDA_PLAN_INLINE uint32_t HeadGroupBegin", 1
    )[1].split("KDA_PLAN_INLINE uint32_t HeadGroupEnd", 1)[0]
    assert "static_cast<uint64_t>(group) * windowCount / groupCount" in owner
    assert "return HeadWindowBegin(" in owner


def test_generic_target_shape_keeps_raw_qk_resident_per_runtime_window():
    prepare = GENERIC_STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    post_wu = GENERIC_STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")

    assert "constexpr uint32_t KDA_AIV_VEC_LOCAL_BYTES = 184 * 1024;" in prepare
    assert "KDA_AIV_UB_BUDGET_BYTES" not in prepare

    gate_rows_body = prepare.split(
        "__aicore__ inline uint64_t GatePipelineRows", 1
    )[1].split("__aicore__ inline uint64_t GateInputSlotBytes", 1)[0]
    assert gate_rows_body.count("KDA_AIV_VEC_LOCAL_BYTES") == 2
    assert "fixedBytes + residentBytes" in gate_rows_body
    assert "KDA_AIV_VEC_LOCAL_BYTES - fixedBytes - residentBytes" in gate_rows_body
    assert "availableBytes / bytesPerRow" in gate_rows_body
    assert "rows < KDA_GATE_TILE_ROWS ? rows : KDA_GATE_TILE_ROWS" in gate_rows_body
    assert (
        "static_cast<uint32_t>(gatePipelineBytes + ResidentRawQKBytes())"
        in prepare
    )

    vec_bytes = 32768 * 4
    exp2_bytes = 256 * (4 + 2)
    resident_bytes = 2 * 64 * 128 * 2
    bytes_per_gate_row = 128 * 3 * (3 * 2 + 4)
    gate_rows = min(
        (184 * 1024 - vec_bytes - exp2_bytes - resident_bytes)
        // bytes_per_gate_row,
        16,
    )
    gate_pipeline_bytes = gate_rows * bytes_per_gate_row
    assert gate_rows == 6
    assert vec_bytes + exp2_bytes + gate_pipeline_bytes + resident_bytes == 184 * 1024

    def align32(value):
        return (value + 31) // 32 * 32

    for k_dim in range(16, 257, 16):
        for gate_element_bytes in (2, 4):
            for use_resident in (False, True):
                if use_resident and k_dim != 128:
                    continue
                current_resident_bytes = 2 * 64 * k_dim * 2 if use_resident else 0
                available_bytes = (
                    184 * 1024 - vec_bytes - exp2_bytes - current_resident_bytes
                )
                current_bytes_per_row = (
                    k_dim * 3 * (3 * 2 + gate_element_bytes)
                )
                current_rows = min(
                    available_bytes // current_bytes_per_row,
                    16,
                )
                current_gate_bytes = current_rows * current_bytes_per_row
                total_bytes = (
                    align32(exp2_bytes)
                    + align32(vec_bytes)
                    + align32(current_gate_bytes + current_resident_bytes)
                )
                assert current_rows > 0
                assert total_bytes <= 184 * 1024

    use_qk = prepare.split("__aicore__ inline bool UseRawQKResident", 1)[1].split(
        "__aicore__ inline uint64_t ResidentRawQKBytes", 1
    )[0]
    load_qk = prepare.split("__aicore__ inline void LoadResidentRawQK", 1)[1].split(
        "__aicore__ inline void PrefetchQKGate", 1
    )[0]
    prefetch_qk = prepare.split("__aicore__ inline void PrefetchQKGate", 1)[1].split(
        "__aicore__ inline void PrefetchKGate", 1
    )[0]
    prepare_window = prepare.split(
        "__aicore__ inline void ProcessCompactPreAivHeadWindow", 1
    )[1].split("__aicore__ inline void ProcessCompactPreAivHeadRange", 1)[0]

    assert "IsSameType<T, float>::value" in use_qk
    assert "BT_ == 64 && K_ == 128 && V_ == 128" in use_qk
    assert "const uint64_t rowBegin = curT * subBlockIdx / subBlockNum;" in load_qk
    assert "const uint64_t rowEnd = curT * (subBlockIdx + 1) / subBlockNum;" in load_qk
    assert "qResident[rowBegin * K_]" in load_qk
    assert "kResident[rowBegin * K_]" in load_qk
    assert "CopyRowsIn(qOwned, q_," in load_qk
    assert "CopyRowsIn(kOwned, k_," in load_qk
    for transformed in ("qg_", "kg_", "preparedQG_", "preparedKG_"):
        assert transformed not in load_qk
    assert "if (!residentRawQKActive_)" in prefetch_qk
    assert "CopyRowsIn(qTyped, q_," in prefetch_qk
    assert "CopyRowsIn(kTyped, k_," in prefetch_qk
    assert "uint64_t residentQHead = H_;" in prepare_window
    assert "residentRawQKActive_ = false;" in prepare_window
    assert "qHead != residentQHead" in prepare_window
    assert "LoadResidentRawQK(" in prepare_window
    assert "residentQHead = qHead;" in prepare_window

    use_k = post_wu.split("__aicore__ inline bool UseRawKResident", 1)[1].split(
        "__aicore__ inline LocalTensor<T> ResidentKTyped", 1
    )[0]
    load_k = post_wu.split("__aicore__ inline void LoadResidentRawK", 1)[1].split(
        "__aicore__ inline bool UsePostWuCube", 1
    )[0]
    post_window = post_wu.split(
        "__aicore__ inline void ProcessCompactPostAivHeadWindow", 1
    )[1].split("__aicore__ inline void ProcessCompactPostAivHeadRange", 1)[0]

    assert "BT_ == 64 && K_ == 128 && V_ == 128" in use_k
    assert "const uint64_t rowBegin = curT * subBlockIdx / subBlockNum;" in load_k
    assert "const uint64_t rowEnd = curT * (subBlockIdx + 1) / subBlockNum;" in load_k
    assert "kResident[rowBegin * K_]" in load_k
    assert "CopyRowsIn(kOwned, k_," in load_k
    for transformed in ("qg_", "kg_", "preparedQG_", "preparedKG_"):
        assert transformed not in load_k
    assert "uint64_t residentQHead = H_;" in post_window
    assert "residentRawKActive_ = false;" in post_window
    assert "qHead != residentQHead" in post_window
    assert "LoadResidentRawK(" in post_window
    assert "residentQHead = qHead;" in post_window


def test_a5_target_shape_keeps_raw_qk_resident_per_runtime_window():
    prepare = ARCH35_STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    post_wu = ARCH35_STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")

    begin_resident = prepare.split(
        "__aicore__ inline void BeginRawQkResidentGroupArch35", 1
    )[1].split("__aicore__ inline bool RawQkResidentContainsArch35", 1)[0]
    prepare_window = prepare.split(
        "__aicore__ inline void ProcessOwnedChunkAivHeadWindow", 1
    )[1].split("__aicore__ inline void ProcessOwnedChunkAivHeads", 1)[0]

    assert "COMPILE_BT == 64 && COMPILE_K == 128" in begin_resident
    assert "COMPILE_V == 128" in begin_resident
    assert "subBlockNum == KDA_SCORE_LANES" in begin_resident
    assert "const uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;" in begin_resident
    assert "const uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;" in begin_resident
    assert "LocalTensor<T> qResident = RawQResidentArch35();" in begin_resident
    assert "LocalTensor<T> kResident = RawKResidentArch35();" in begin_resident
    assert "CopyRowsIn(qResident, q_," in begin_resident
    assert "CopyRowsIn(kResident, k_," in begin_resident
    for transformed in ("qg_", "kg_", "preparedQG_", "preparedKG_"):
        assert transformed not in begin_resident
    assert "groupHv / (HV_ / H_)" in prepare_window
    assert "nextHv / (HV_ / H_) != groupH" in prepare_window
    assert prepare_window.index("BeginRawQkResidentGroupArch35(") < prepare_window.index(
        "ProcessOwnedChunkAivQueryHeadGroup("
    )

    prefetch = post_wu.split(
        "__aicore__ inline void PrefetchTypicalKgPipeline", 1
    )[1].split("__aicore__ inline void StageTypicalKgResidentK", 1)[0]
    post_window = post_wu.split(
        "__aicore__ inline void ProcessTypicalFullPostAivHeadWindow", 1
    )[1].split("__aicore__ inline void ProcessTypicalFullPostAicHeadWindow", 1)[0]

    assert "LocalTensor<T> residentK = TypicalGatePipelineResidentK();" in prefetch
    assert "CopyRowsIn(\n                        residentK, k_," in prefetch
    assert "if (reloadResidentK)" in prefetch
    assert "CopyRowsIn(kStage, k_," in prefetch
    for transformed in ("qg_", "kg_", "preparedQG_", "preparedKG_"):
        assert transformed not in prefetch
    assert "sizeof(T) == sizeof(uint16_t) && BT_ == 64 && K_ == 128" in post_window
    assert "V_ == 128 && subBlockNum == 2" in post_window
    assert "const uint64_t rowBegin = (BT_ * subBlockIdx) / subBlockNum;" in post_window
    assert "const uint64_t rowEnd = (BT_ * (subBlockIdx + 1)) / subBlockNum;" in post_window
    assert "useResidentK, useResidentK" in post_window
    assert "const bool reloadResidentK = useResidentK && nextH != h;" in post_window
    assert "StageTypicalKgResidentK(slot, rowEnd - rowBegin);" in post_window

    fallback_begin = post_wu.split(
        "__aicore__ inline void BeginFallbackKResidentGroupArch35", 1
    )[1].split("__aicore__ inline bool StageFallbackKFromResidentArch35", 1)[0]
    fallback_stage = post_wu.split(
        "__aicore__ inline bool StageFallbackKFromResidentArch35", 1
    )[1].split("__aicore__ inline uint64_t TypicalGateStageElems", 1)[0]
    fallback_window = post_wu.split(
        "__aicore__ inline void ProcessCompactPostHeadWindow", 1
    )[1].split("__aicore__ inline void ProcessCompactPostHeadRange", 1)[0]
    tail_window = post_wu.split(
        "__aicore__ inline void ProcessTailAuxHeadWindow", 1
    )[1].split("__aicore__ inline void ProcessTailAuxHeadRange", 1)[0]

    assert "BT_ != 64 || K_ != 128 || V_ != 128" in fallback_begin
    assert "subBlockNum != 2 || subBlockIdx >= subBlockNum" in fallback_begin
    assert "const uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;" in fallback_begin
    assert "(curT * (subBlockIdx + 1)) / subBlockNum" in fallback_begin
    assert "LocalTensor<T> residentK = FallbackKResidentArch35();" in fallback_begin
    assert "CopyRowsIn(\n                residentK, k_," in fallback_begin
    for transformed in ("qg_", "kg_", "preparedQG_", "preparedKG_"):
        assert transformed not in fallback_begin
    assert "Adds(dst, FallbackKResidentArch35()[residentOffset]" in fallback_stage
    assert "fallbackKResidentHasVectorReader_ = true;" in fallback_stage
    assert "bool residentQHeadValid = false;" in fallback_window
    assert "!residentQHeadValid || qHead != residentQHead" in fallback_window
    assert "BeginFallbackKResidentGroupArch35(" in fallback_window
    assert "residentQHead = qHead;" in fallback_window
    assert "bool residentQHeadValid = false;" in tail_window
    assert "!residentQHeadValid || h != residentQHead" in tail_window
    assert "BeginFallbackKResidentGroupArch35(" in tail_window


def test_tiling_key_selects_shape_family_independently_of_platform():
    tiling = TILING_ENTRY.read_text(encoding="utf-8")
    arch35 = ARCH35_TILING.read_text(encoding="utf-8")
    l0 = (OP_ROOT / "op_host/op_api/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    op_def = (OP_ROOT / "op_host/chunk_kda_fwd_def.cpp").read_text(encoding="utf-8")

    for field in (
        "prepareScratchOffset",
        "postWuScratchOffset",
        "outputScratchOffset",
        "vWorkspaceOffset",
    ):
        assert field in tiling
    assert "const bool useChunk64K128V128Template" in tiling
    scenario_condition = tiling.split(
        "const bool useChunk64K128V128Template", 1
    )[1].split(";", 1)[0]
    assert "chunkSize == 64" in scenario_condition
    assert "shape.kDim == 128" in scenario_condition
    assert "shape.vDim == 128" in scenario_condition
    assert "isAscend950" not in scenario_condition
    assert "chunkSize == 64 && kDim == 128 && vDim == 128" in arch35
    shape_condition = arch35.split("const bool shapeSupported", 1)[1].split(";", 1)[0]
    assert "isAscend950" in shape_condition
    assert "!isVarLen" not in shape_condition
    assert "seqlen % chunkSize" not in shape_condition
    assert "const bool denseScheduled = !isVarLen;" in arch35
    gate_selection = arch35.split("options.computeGateInPrepare =", 1)[1].split(";", 1)[0]
    assert "denseAligned" not in gate_selection
    assert "isVarLen" not in gate_selection
    assert "enabled" not in arch35
    for gate_condition in (
        "qIsBf16",
        "rawGIsFp32",
        "hasALog",
        "gateParamsAreFp32",
        "useGateInKernel",
        "safeGate",
    ):
        assert gate_condition in arch35
        assert gate_condition in gate_selection
    assert "const bool gateParamsAreFp32" in tiling
    assert "!hasALog || aLogDesc->GetDataType() == ge::DT_FLOAT" in tiling
    assert "!hasDtBias || dtBiasDesc->GetDataType() == ge::DT_FLOAT" in tiling
    for private_attr in (
        "logical_batch",
        "logical_seqlen",
        "logical_q_heads",
        "defer_gate_cumsum",
        "enablePrivateA5Path",
        "store_qg",
    ):
        assert private_attr not in op_def
    for private_attr in (
        "logical_batch",
        "logical_seqlen",
        "logical_q_heads",
        "defer_gate_cumsum",
        "store_qg",
    ):
        assert private_attr not in l0
    assert 'AddConfig("ascend950", config)' in op_def
    assert 'AddConfig("ascend910b", config)' in op_def
    assert 'AddConfig("ascend910_93", config)' in op_def


def test_shared_kernel_driver_matches_public_tensor_address_count():
    entry = KERNEL_ENTRY.read_text(encoding="utf-8")
    signature = entry.split("__aicore__ inline void RunDispatched(", 1)[1].split(
        ")", 1
    )[0]
    assert signature.count("GM_ADDR") == 23
    assert "compactPlan" in entry


def test_outer_tpipe_releases_reserved_aic_events_before_manual_pipelines():
    entry = KERNEL_ENTRY.read_text(encoding="utf-8")
    arch35 = ARCH35_KERNEL.read_text(encoding="utf-8")
    common = KERNEL_COMMON.read_text(encoding="utf-8")
    driver = entry.split("__aicore__ inline void RunDispatched(", 1)[1].split(
        "} // namespace KdaForward", 1
    )[0]

    release_helper = common.split(
        "__aicore__ inline void ReleaseAicPipeReservedMmadEvents(", 1
    )[1].split("struct GateRuntimeTiling", 1)[0]
    assert "__NPU_ARCH__ == 2201" in release_helper
    assert "__CCE_AICORE__ == 2201" in release_helper
    assert "__NPU_ARCH__ == 3510" in release_helper
    assert "__CCE_AICORE__ == 310" in release_helper
    assert "if ASCEND_IS_AIC" in release_helper
    assert release_helper.count("pipe.DestroyWithoutPipeAll();") == 1
    assert release_helper.count("pipe.Destroy();") == 1

    arch35_setup = driver.split(
        "#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310", 1
    )[1].split("#if KDA_COMPILE_ARCH35_FAST_PATH", 1)[0]
    setup = arch35_setup.split("AscendC::TPipe pipe;", 1)[1].split(
        "RunPostGateFrontEndStages<", 1
    )[0]
    assert setup.count("ReleaseAicPipeReservedMmadEvents(pipe);") == 1

    generic_setup = driver.split(
        'static_assert(!USE_ARCH35, "arch35 backend is unavailable on this architecture");',
        1,
    )[1].split("RunFrontEndStages<", 1)[0]
    assert generic_setup.count("AscendC::TPipe pipe;") == 1
    assert generic_setup.count("ReleaseAicPipeReservedMmadEvents(pipe);") == 1

    generic_isolation = driver.split(
        "if (!tiling.isVarLen && tiling.seqlen % tiling.chunkSize == 0)", 1
    )[1].split("RunGenericBackEnd<", 1)[0]
    assert "if ASCEND_IS_AIV" in generic_isolation
    assert generic_isolation.count("pipe.Destroy();") == 1

    arch35_isolation = arch35.split("if (isolateGenericBackEnd)", 1)[1].split(
        "RunGenericBackEnd<", 1
    )[0]
    assert "if ASCEND_IS_AIV" in arch35_isolation
    assert arch35_isolation.count("pipe.Destroy();") == 1

    generic_tail = common.split("__aicore__ inline void RunGenericTailBackEnd(", 1)[
        1
    ].split("template <typename T, typename BETA_T, typename TilingData>", 1)[0]
    assert generic_tail.count("TPipe pipe;") == 1
    assert generic_tail.count("ReleaseAicPipeReservedMmadEvents(pipe);") == 1

    generic_backend = common.split("__aicore__ inline void RunGenericBackEnd(", 1)[
        1
    ].split("template <typename T, typename BETA_T, typename TilingData>", 1)[0]
    assert generic_backend.count("TPipe pipe;") == 1
    assert generic_backend.count("ReleaseAicPipeReservedMmadEvents(pipe);") == 1


def test_a2_a3_frontend_gate_reuses_outer_tpipe_and_resets_aiv_before_prepare():
    entry = KERNEL_ENTRY.read_text(encoding="utf-8")
    common = KERNEL_COMMON.read_text(encoding="utf-8")
    gate_wrapper = common.split(
        "__aicore__ inline void RunGateCumsum(", 1
    )[1].split("template <bool SAFE_GATE, typename T", 1)[0]
    gate_dispatch = entry.split(
        "__aicore__ inline void DispatchGateMode(", 1
    )[1].split("template <typename G_T, typename TilingData>", 1)[0]
    gate_stage = entry.split(
        "__aicore__ inline void RunGateStage(", 1
    )[1].split("template <bool SAFE_GATE, typename T", 1)[0]
    front_stages = entry.split(
        "__aicore__ inline void RunFrontEndStages(", 1
    )[1].split("template <bool USE_ARCH35", 1)[0]

    assert "TPipe gatePipe;" not in gate_wrapper
    assert "const TilingData &tiling, TPipe &pipe" in gate_wrapper
    assert "gk, gateTiling, &pipe);" in gate_wrapper
    assert "const TilingData &tiling, AscendC::TPipe &pipe" in gate_dispatch
    assert gate_dispatch.count("gk, tiling, pipe);") == 5
    assert gate_stage.count("gk, tiling, pipe);") == 2
    assert front_stages.count("addresses.gk, tiling, pipe);") == 2
    assert "DispatchGateMode<true" in front_stages
    assert "DispatchGateMode<false" in front_stages
    assert "RunGateStage<" not in front_stages
    assert "RunPostGateFrontEndStages<" not in front_stages

    gate_call = max(
        front_stages.index("DispatchGateMode<true"),
        front_stages.index("DispatchGateMode<false"),
    )
    gate_done = front_stages.index("if (!tiling.computeGateInPrepare)")
    prepare_start = front_stages.index("DispatchPrepareSafeGate<")
    transition = front_stages[gate_done:prepare_start]
    assert gate_call < gate_done < prepare_start
    assert "if ASCEND_IS_AIV" in transition
    assert transition.count("pipe.Reset();") == 1
    assert transition.count("SyncAll<false>();") == 1
    assert transition.index("pipe.Reset();") < transition.index("SyncAll<false>();")


def test_a5_gate_tpipe_is_destroyed_before_sync_and_post_gate_tpipe():
    entry = KERNEL_ENTRY.read_text(encoding="utf-8")
    driver = entry.split("__aicore__ inline void RunDispatched(", 1)[1].split(
        "} // namespace KdaForward", 1
    )[0]
    arch35_setup = driver.split(
        "#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310", 1
    )[1].split("#if KDA_COMPILE_ARCH35_FAST_PATH", 1)[0]

    guard = arch35_setup.index("if (!tiling.computeGateInPrepare)")
    aiv_guard = arch35_setup.index("if ASCEND_IS_AIV", guard)
    gate_ctor = arch35_setup.index("AscendC::TPipe gatePipe;", aiv_guard)
    gate_call = arch35_setup.index("RunGateStage<G_T>(", gate_ctor)
    gate_call_end = arch35_setup.index("gatePipe);", gate_call) + len("gatePipe);")
    sync = arch35_setup.index("SyncAll<false>();", gate_call_end)
    outer_ctor = arch35_setup.index("AscendC::TPipe pipe;", sync)
    release = arch35_setup.index("ReleaseAicPipeReservedMmadEvents(pipe);", outer_ctor)
    prepare = arch35_setup.index("RunPostGateFrontEndStages<", release)

    assert guard < aiv_guard < gate_ctor < gate_call < sync
    assert sync < outer_ctor < release < prepare
    assert " ".join(arch35_setup[gate_call_end:outer_ctor].split()) == (
        "} } SyncAll<false>(); }"
    )
    assert arch35_setup.count("AscendC::TPipe gatePipe;") == 1
    assert arch35_setup.count("AscendC::TPipe pipe;") == 1
    assert arch35_setup.count("SyncAll<false>();") == 1
    assert "pipe.Reset();" not in arch35_setup
    assert "ReleaseAicPipeReservedMmadEvents(gatePipe);" not in arch35_setup
    assert "g_tPipePtr" not in arch35_setup
    assert "RunFrontEndStages<" not in arch35_setup


def test_optional_output_workspace_uses_ir_instantiation_state():
    tiling = TILING_ENTRY.read_text(encoding="utf-8")
    has_output = tiling.split(
        "bool HasOutput(gert::TilingContext *context, size_t index)", 1
    )[1].split("struct ShapeInfo", 1)[0]

    assert "GetIrOutputInstanceInfo(index)" in has_output
    assert "GetInstanceNum() == 0" in has_output
    assert "GetOutputShape(instanceInfo->GetInstanceStart())" in has_output
    assert "GetShapeSize() != 1" in has_output
    assert "GetOutputDesc(index)" not in has_output

    kernel = KERNEL_COMMON.read_text(encoding="utf-8")
    assert "return storeOutput ? output : userWorkspace + offset;" in kernel

    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    assert "const op::Shape placeholderShape = MakeShape({1});" in aclnn
    assert "usePrivateA5Path" not in aclnn
    for export in ("wExport", "uExport", "qgExport", "kgExport", "vNewExport"):
        assert f"{export} == nullptr" in aclnn
        assert "AllocTensor(executorPtr, placeholderShape" in aclnn
    assert "outputFinalState ? stateShape4 : placeholderShape" in aclnn
    assert "hExport == nullptr ? placeholderShape : hShape5" in aclnn


def test_generic_post_wu_scratch_covers_varlen_chunk_padding():
    tiling = TILING_ENTRY.read_text(encoding="utf-8")
    allocation = tiling.split(
        "uint64_t postWuScratchOffset", 1
    )[1].split("uint64_t fwdHWorkspaceBaseOffset", 1)[0]

    assert "hChunkCount, static_cast<uint64_t>(shape.vHeads)" in allocation
    assert "static_cast<uint64_t>(chunkSize)" in allocation
    assert "static_cast<uint64_t>(shape.kDim), sizeof(float)" in allocation
    assert "CheckedProduct" in allocation
    assert "postWuScratchOffset, postWuScratchBytes, cursor" in allocation

    lengths = (81, 159, 127, 145, 95, 177, 113, 127)
    total_chunks = sum((length + 63) // 64 for length in lengths)
    assert total_chunks * 64 > sum(lengths)


def test_generic_tail_w_uses_immutable_seed_snapshot():
    post_wu = GENERIC_STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    common = KERNEL_COMMON.read_text(encoding="utf-8")
    seed_copy = post_wu.split(
        "void CopyTailSeedRows", 1
    )[1].split("bool ResolveFlatChunk", 1)[0]
    tail_cube = post_wu.split(
        "void ProcessVarlenTailAic", 1
    )[1].split("void ProcessVarlenTailAiv", 1)[0]

    assert "CopyVectorIn(typed, preparedQG_" in seed_copy
    assert "CopyVectorOut(w_" in seed_copy
    assert "CopyVectorOut(u_" not in seed_copy
    assert "ComputePostWuCube" in tail_cube
    assert "outputScratchOffset + uSeedBytes" in post_wu
    snapshot_condition = post_wu.split(
        "bool UseVarlenTailCubeSnapshot", 1
    )[1].split(";", 1)[0]
    assert "BT_ == 64" in snapshot_condition
    assert "K_ == 128" in snapshot_condition
    assert "V_ == 128" in snapshot_condition
    assert "RunChunkKdaPostWuTailSeedCopy" in common
    assert "RunChunkKdaPostWuTail" in common
    assert "tiling.hasVarlenTail && tiling.chunkSize == 64" in common
    assert "tiling.kHeadDim == 128 && tiling.vHeadDim == 128" in common


def test_varlen_and_tail_use_the_same_physical_l0():
    l0 = (OP_ROOT / "op_host/op_api/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    assert l0.count("l0op::ChunkKdaFwd(") == 0
    assert l0.count("OP_TYPE_REGISTER(ChunkKdaFwd);") == 1
    assert l0.count("ADD_TO_LAUNCHER_LIST_AICORE(") == 1
    for stage in ("ChunkKdaFwdPrepare", "ChunkKdaFwdPostWu", "ChunkKdaFwdFinalize"):
        assert stage not in l0


def test_l0_keeps_the_public_result_contract():
    l0 = (OP_ROOT / "op_host/op_api/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    for output in (
        "attnOut",
        "finalStateOut",
        "gk",
        "aqkOut",
        "akkOut",
        "wOut",
        "uOut",
        "qgOut",
        "kgOut",
        "vNewOut",
        "hOut",
    ):
        assert output in l0
    assert "params.gkOut" in aclnn and "gkCompute" in aclnn
    assert "OP_TYPE_REGISTER(ChunkKdaFwd);" in l0
    for stage in ("ChunkKdaFwdPrepare", "ChunkKdaFwdPostWu", "ChunkKdaFwdFinalize"):
        assert f"OP_TYPE_REGISTER({stage});" not in l0
        assert f"ADD_TO_LAUNCHER_LIST_AICORE({stage}" not in l0
    assert "ChunkKdaFwdFusedArch35" not in l0


def test_prepare_post_wu_fusion_stays_inside_chunk_kda_fwd():
    common = KERNEL_COMMON.read_text(encoding="utf-8")
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    arch35 = ARCH35_TILING.read_text(encoding="utf-8")

    assert "op.ProcessAicFused(postWu);" in prepare
    assert "if (!tiling.fusePostWu && !tiling.fusePostWuIntoFwdH)" in common
    assert "const bool canFusePreparePostWu =" in arch35
    can_fuse_selection = arch35.split(
        "const bool canFusePreparePostWu =", 1
    )[1].split(";", 1)[0]
    assert "sequenceAwareVarlen" in can_fuse_selection
    assert "hasVarlenTail" not in can_fuse_selection
    assert "const bool denseScheduled = !isVarLen;" in arch35
    assert "const bool sequenceAwareVarlen = isVarLen && seqNum > 0;" in arch35
    fwd_h_selection = arch35.split(
        "options.useDenseFwdH =", 1
    )[1].split(";", 1)[0]
    assert "denseScheduled" in fwd_h_selection
    assert "sequenceAwareVarlen" in fwd_h_selection
    fuse_into_selection = arch35.split(
        "options.fusePostWuIntoFwdH =", 1
    )[1].split(";", 1)[0]
    assert fuse_into_selection.strip() == "false"
    fuse_selection = arch35.split("options.fusePostWu =", 1)[1].split(";", 1)[0]
    assert "canFusePreparePostWu" in fuse_selection
    assert "denseScheduled" not in fuse_selection
    assert "options.fusePostWu" in arch35
    assert "options.fusePostWuIntoFwdH" in arch35
    assert "batchChunkIdx" not in prepare
    assert "ProcessPreparedFullHeadBatchArch35" in post_wu
    assert "ProcessPreparedTailSingleArch35" in post_wu
    assert "postWu.ProcessPreparedTailSingleArch35" in prepare
    arch35_post_wu = common.split(
        "__aicore__ inline void RunPostWuStage", 1
    )[1].split("template <typename T, typename TileShapes", 1)[0]
    arch35_only = arch35_post_wu.split(
        "#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310", 1
    )[0]
    assert "RunChunkKdaPostWuTailSeedCopy" not in arch35_only
    assert "RunChunkKdaPostWuTail" not in arch35_only


def test_a5_varlen_fwd_h_uses_sequence_aware_full_chunks_and_splits_mixed_tail():
    tiling = TILING_ENTRY.read_text(encoding="utf-8")
    arch35 = ARCH35_TILING.read_text(encoding="utf-8")
    impl = ARCH35_KERNEL.read_text(encoding="utf-8")
    fwd_h = ARCH35_FWD_H.read_text(encoding="utf-8")
    common = KERNEL_COMMON.read_text(encoding="utf-8")
    finalize = ARCH35_STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")

    plan = COMPACT_PLAN.read_text(encoding="utf-8")

    assert "info.hasVarlenTail = info.hasVarlenTail || hasTail;" in tiling
    assert "tiling.set_hasVarlenTail(hasVarlenTail);" in tiling
    assert "const bool sequenceAwareVarlen = isVarLen && seqNum > 0;" in arch35
    assert "if (tiling.useDenseFwdH)" in impl
    assert "RunGenericTailBackEnd" not in impl
    assert "LoadFwdHeadRange" not in plan
    assert "struct FwdHeadRange" not in plan
    assert "LoadChunkCoreCursor" in plan
    assert "plan.AlignedSequenceCount()" in fwd_h
    assert "plan.TailedSequenceCount()" in fwd_h
    assert "ProcessSelectedSequenceAic<false>" in fwd_h
    assert "ProcessSelectedSequenceAic<true>" in fwd_h
    assert "sequenceChunks_ = sequenceTokens / KDA_FWD_H_CHUNK;" in fwd_h
    assert "sequenceTailTokens_ = sequenceTokens % KDA_FWD_H_CHUNK;" in fwd_h
    assert "sequenceTotalChunks_ = sequenceChunks_ + (sequenceTailTokens_ != 0);" in fwd_h
    assert "ProcessOwnedChunksAic<false>" in finalize
    assert "ProcessOwnedChunksAic<true>" in finalize
    assert "LoadChunkCoreCursor" in finalize
    assert "LoadFwdHeadRange" not in finalize
    assert "LoadChunkCoreCursor" not in fwd_h
    assert "headBegin_ = coreIdx * heads_ / fwdCoreNum_;" in fwd_h
    assert "headEnd_ = (coreIdx + 1) * heads_ / fwdCoreNum_;" in fwd_h


def test_a5_varlen_tail_scheduler_preserves_sequence_state_and_full_chunk_fast_path():
    scheduler = ARCH35_GDN_SCHEDULER.read_text(encoding="utf-8")
    kernel = ARCH35_GDN_KERNEL.read_text(encoding="utf-8")

    assert "tailOnly = isVariedLen > 1;" in scheduler
    assert "batchTokens % chunkSize != 0" in scheduler
    assert "stream.batchIdx = b - 1;" in scheduler
    assert "newStream.chunkIdx = tailOnly ? newStream.fullChunks : 0;" in scheduler
    assert "stream.active = stream.chunkIdx < stream.batchChunks;" in scheduler
    assert "stream.chunkOffset + (tailOnly ? stream.fullChunks : 0)" in scheduler
    assert "GetVarlenStateBatchIdx" in scheduler
    assert "cachedVarlenSequenceValid" in scheduler
    assert "stream.chunkIdx < stream.fullChunks" in scheduler
    assert "useChunkAwareMmad" in kernel
    assert "cube1Offsets.blockTokens != chunkSize" in kernel
    assert "cube2Offsets.blockTokens != chunkSize" in kernel
    assert "GetVarlenStateBatchIdx(batchIdx)" in kernel
    assert "(stateBatchIdx * vNumHead + vHeadIdx) * stateBlockSize" in kernel


def test_a5_varlen_chunk_plan_covers_requested_lengths_and_distributions():
    chunk_size = 64

    def distributions(total):
        quotient, remainder = divmod(total, 8)
        balanced = [quotient + (index < remainder) for index in range(8)]
        base = total // 8
        mixed = [
            base - 47,
            base + 31,
            base - 1,
            base + 17,
            base - 33,
            base + 49,
            base - 15,
        ]
        mixed.append(total - sum(mixed))
        short = [chunk_size] * (total // chunk_size)
        if total % chunk_size:
            short.append(total % chunk_size)
        return {
            "single": [total],
            "balanced8": balanced,
            "mixed_tail": mixed,
            "short64": short,
        }

    for total in (1024, 1536, 2048, 4096, 8192, 16384):
        for name, lengths in distributions(total).items():
            assert sum(lengths) == total
            chunk_prefix = 0
            tails = []
            for state_index, length in enumerate(lengths):
                full_chunks, tail_tokens = divmod(length, chunk_size)
                if tail_tokens:
                    tails.append(
                        {
                            "state_index": state_index,
                            "flat_chunk": chunk_prefix + full_chunks,
                            "tokens": tail_tokens,
                        }
                    )
                chunk_prefix += full_chunks + bool(tail_tokens)

            assert chunk_prefix == sum(
                (length + chunk_size - 1) // chunk_size for length in lengths
            )
            if name in {"single", "balanced8", "short64"}:
                assert not tails
            else:
                assert tails
                assert all(0 < tail["tokens"] < chunk_size for tail in tails)
                assert [tail["state_index"] for tail in tails] == [
                    index for index, length in enumerate(lengths)
                    if length % chunk_size
                ]


def test_u_seed_does_not_alias_post_wu_output():
    common = KERNEL_COMMON.read_text(encoding="utf-8")
    assert "GM_ADDR uSeed;" in common
    assert "userWorkspace + tiling.outputScratchOffset" in common
    assert "tiling.fusePostWu || tiling.fusePostWuIntoFwdH" in common
    assert "addresses.w, akk, uSeed" in common
    assert "addresses.w, addresses.u, addresses.kg" in common


def test_generic_prepare_reads_sequence_major_inputs_with_head_stride():
    prepare = GENERIC_STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    assert "inputSequenceMajor_ = tiling.inputSequenceMajor;" in prepare
    assert "return ((b * T_ + t) * H_ + h) * K_ + d;" in prepare
    assert "return ((b * T_ + t) * HV_ + hv) * V_ + d;" in prepare
    assert "CopyRowsIn(qTyped, q_, QOffset" in prepare
    assert "inputSequenceMajor_ ? H_ * K_ : K_" in prepare
    assert "sourceSequenceMajor ? HV_ * dim : dim" in prepare
    assert "matrixLocal, inputSequenceMajor_" in prepare


def test_intermediate_outputs_keep_canonical_bnsd_graph_views_between_stages():
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    required_compute_names = {
        "aqkOut": "aqkCompute",
        "akkOut": "akkCompute",
    }
    for output, compute in required_compute_names.items():
        assert f"const aclTensor *{compute} = params.{output};" in aclnn
        assert f"AsRank4({compute}," in aclnn
        assert f"Transpose(params.{output}" not in aclnn
    optional_names = {
        "wOut": "wExport",
        "uOut": "uExport",
        "qgOut": "qgExport",
        "kgOut": "kgExport",
        "vNewOut": "vNewExport",
        "hOut": "hExport",
    }
    for output, export in optional_names.items():
        assert f"const aclTensor *{export} = params.{output};" in aclnn
        assert f"Transpose(params.{output}" not in aclnn
    assert "const aclTensor *hCompute = AllocTensor(" in aclnn
    for compute in ("wCompute", "uCompute", "qgCompute", "kgCompute", "vNewCompute"):
        assert f"const aclTensor *{compute}" in aclnn
    assert "MakeShape({info.batch, info.hvNum, info.seqlen" in aclnn
    assert "MakeShape({info.batch, info.hvNum, info.totalChunks" in aclnn
    assert "MakeShape({info.batch, info.totalChunks, info.hvNum" in aclnn
    assert "std::vector<int64_t>{0, 2, 1, 3, 4}" in aclnn
    assert "std::vector<int64_t>{0, 2, 1, 4, 3}" in aclnn
    assert "use fixed sequence-major layout" in aclnn


def test_safe_gate_is_supported_across_public_and_direct_routes():
    aclnn_runtime = (
        ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"
    ).read_text(encoding="utf-8")
    legacy_runtime = (
        ROOT / "torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp"
    ).read_text(encoding="utf-8")
    direct = DIRECT_SOURCE.read_text(encoding="utf-8")
    prepare = KERNEL_ENTRY.read_text(encoding="utf-8")

    assert "safe_gate is reserved" not in aclnn_runtime
    assert "safe_gate=true is not supported" not in legacy_runtime
    assert "ctypes.c_bool(safe_gate)" in aclnn_runtime
    assert "bool safe_gate=False" in direct
    assert "RunPrepareVariant<true" in prepare
    assert "RunPrepareVariant<false" in prepare
    assert prepare.count("arch35::RunBackEnd<") == 1
    assert "RunBackEnd<USE_GATE_IN_KERNEL" not in prepare
    assert "ScoreRefBlockSize" in STAGE_IMPLEMENTATIONS["prepare"].read_text(
        encoding="utf-8"
    )


def test_typical_chunk64_k128_v128_uses_internal_prepare_specialization():
    prepare_entry = KERNEL_ENTRY.read_text(encoding="utf-8")
    prepare_impl = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    prepare_tiling = TILING_ENTRY.read_text(encoding="utf-8")

    assert "TILING_KEY_IS(2)" in prepare_entry
    assert "if (TILING_KEY_IS(1))" in prepare_entry
    assert "else if (TILING_KEY_IS(2))" in prepare_entry
    assert "TILING_KEY_VAR == 1UL" in prepare_entry
    assert "TILING_KEY_VAR == 2UL" in prepare_entry
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);" in prepare_entry
    assert "KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);" in prepare_entry
    assert "KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);" in prepare_entry
    assert "DispatchPrepareSafeGate" in prepare_entry
    assert "RunFrontEndStages" in prepare_entry
    assert "RunDispatched<false" in prepare_entry
    assert "ChunkKdaFwdTilingData, 64, 128, 128" in prepare_entry
    assert "ConfigureChunkKdaFwdArch35" in prepare_tiling
    assert "SetTilingKey(useChunk64K128V128Template ? 2 : 1)" in prepare_tiling
    key2_branch = prepare_entry.split("else if (TILING_KEY_IS(2))", 1)[1]
    assert "RunDispatched<true" in key2_branch
    assert "RunDispatched<false" in key2_branch
    assert "uint32_t COMPILE_BT = 0" in prepare_impl
    assert "COMPILE_K == 0 ? tiling.kHeadDim : COMPILE_K" in prepare_impl


def test_fixed_tiling_key_translation_unit_only_instantiates_its_key():
    entry = KERNEL_ENTRY.read_text(encoding="utf-8")
    fixed_key1 = entry.split(
        "#if defined(TILING_KEY_VAR) && TILING_KEY_VAR == 1UL", 1
    )[1].split(
        "#elif defined(TILING_KEY_VAR) && TILING_KEY_VAR == 2UL", 1
    )[0]
    fixed_key2 = entry.split(
        "#elif defined(TILING_KEY_VAR) && TILING_KEY_VAR == 2UL", 1
    )[1].split("#else\n    if (TILING_KEY_IS(1))", 1)[0]

    assert fixed_key1.count("RunDispatched<false") == 1
    assert "RunDispatched<true" not in fixed_key1
    assert "ChunkKdaFwdTilingData, 0, 0, 0" in fixed_key1
    assert "#if KDA_COMPILE_ARCH35_FAST_PATH" in fixed_key2
    assert fixed_key2.count("RunDispatched<true") == 1
    assert fixed_key2.count("RunDispatched<false") == 1
    assert "ChunkKdaFwdTilingData, 0, 0, 0" not in fixed_key2


def test_compact_plan_head_group_helpers_are_inline_on_device():
    plan = COMPACT_PLAN.read_text(encoding="utf-8")
    assert "#define KDA_PLAN_INLINE __aicore__ inline" in plan
    assert "#define KDA_PLAN_INLINE constexpr" in plan
    for helper in (
        "HeadWindowCount",
        "HeadWindowBegin",
        "HeadWindowHeadCount",
        "ComputeChunkHeadGroupCount",
        "HeadGroupBegin",
        "HeadGroupEnd",
    ):
        assert f"KDA_PLAN_INLINE uint32_t {helper}(" in plan
    assert "static_cast<uint64_t>(group) * windowCount / groupCount" in plan
    assert "HeadWindowBegin(" in plan
    assert "group + 1, groupCount, queryHeadCount, valueHeadCount" in plan


def test_arch35_large_unrolled_workhorses_keep_explicit_compiler_boundaries():
    prepare = ARCH35_STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    post_wu = ARCH35_STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    finalize = ARCH35_STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")

    for function_name in (
        "ProcessChunkPreAivFp32",
        "ProcessChunkPreAic",
    ):
        assert (
            f"__attribute__((noinline)) __aicore__ void {function_name}("
            in prepare
        )
    for function_name in (
        "ProcessPreparedTailSingleArch35",
        "ProcessCompactPostHead",
    ):
        assert (
            f"__attribute__((noinline)) __aicore__ void {function_name}("
            in post_wu
        )
    for function_name in ("ProcessChunkOutAiv", "ProcessChunkOutAic"):
        assert (
            f"__attribute__((noinline)) __aicore__ void {function_name}("
            in finalize
        )

    assert "__aicore__ inline void RunChunkKdaPrepare(" in prepare
    assert prepare.count("__attribute__((noinline))") == 2
    assert post_wu.count("__attribute__((noinline))") == 2
    assert finalize.count("__attribute__((noinline))") == 2
    for inline_leaf, source in (
        ("PrepareGateProducts", prepare),
        ("ComputeRawAqkAkkCubeBlock", prepare),
        ("ComputePostWuCube", post_wu),
        ("FinalizeOutputRows", finalize),
    ):
        assert f"__aicore__ inline void {inline_leaf}(" in source


def test_generic_large_unrolled_workhorses_keep_explicit_compiler_boundaries():
    prepare = GENERIC_STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    post_wu = GENERIC_STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    finalize = GENERIC_STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    common = KERNEL_COMMON.read_text(encoding="utf-8")

    for function_name in (
        "ProcessChunkPreAivFp32",
        "FinishDeferredSafeChunk",
        "ProcessChunkPreAic",
    ):
        assert (
            f"__attribute__((noinline)) __aicore__ void {function_name}("
            in prepare
        )
    for function_name in (
        "ComputePostWuCube",
        "ComputeCompactPostWuCubeHeadWindow",
        "CopyScratchWAndFinalizeKg",
        "ComputeTailWuVector",
        "CopyTailSeedRows",
    ):
        assert (
            f"__attribute__((noinline)) __aicore__ void {function_name}("
            in post_wu
        )
    for function_name in (
        "ProcessChunkOutAiv",
        "ProcessChunkOutAic",
    ):
        assert (
            f"__attribute__((noinline)) __aicore__ void {function_name}("
            in finalize
        )
    assert "__aicore__ inline void RunFwdH(" in common

    assert "__aicore__ inline void RunChunkKdaPrepare(" in prepare
    assert prepare.count("__attribute__((noinline))") == 3
    assert post_wu.count("__attribute__((noinline))") == 5
    assert finalize.count("__attribute__((noinline))") == 2
    assert common.count("__attribute__((noinline))") == 0
    for inline_wrapper, source in (
        ("RunChunkKdaPostWu", post_wu),
        ("RunChunkKdaPostWuTailSeedCopy", post_wu),
        ("RunChunkKdaPostWuTail", post_wu),
        ("RunChunkKdaOutput", finalize),
    ):
        assert f"__aicore__ inline void {inline_wrapper}(" in source
    for inline_leaf, source in (
        ("ProcessChunkPreAicFp32", prepare),
        ("PrepareGateProducts", prepare),
        ("ComputeTailWuRow", post_wu),
        ("FinalizeOutputRows", finalize),
    ):
        assert f"__aicore__ inline void {inline_leaf}(" in source


def test_gate_dispatch_does_not_reinstantiate_fwd_h_or_finalize_back_end():
    entry = KERNEL_ENTRY.read_text(encoding="utf-8")
    gate_dispatch = entry.split(
        "__aicore__ inline void DispatchGateMode(", 1
    )[1].split("template <typename G_T, typename TilingData>", 1)[0]
    gate_stage = entry.split(
        "__aicore__ inline void RunGateStage(", 1
    )[1].split("template <bool SAFE_GATE, typename T, typename BETA_T", 1)[0]
    prepare_variant = entry.split(
        "__aicore__ inline void RunPrepareVariant(", 1
    )[1].split("template <typename T, typename BETA_T", 1)[0]
    post_gate_front_stages = entry.split(
        "__aicore__ inline void RunPostGateFrontEndStages(", 1
    )[1].split("template <typename T, typename G_T, typename BETA_T", 1)[0]
    front_stages = entry.split(
        "__aicore__ inline void RunFrontEndStages(", 1
    )[1].split("template <bool USE_ARCH35", 1)[0]
    driver = entry.split("__aicore__ inline void RunDispatched(", 1)[1].split(
        "} // namespace KdaForward", 1
    )[0]

    assert "RunGateCumsum<" in gate_dispatch
    assert "RunPrepareStage<" not in gate_dispatch
    assert "RunGenericBackEnd<" not in gate_dispatch
    assert "arch35::RunBackEnd<" not in gate_dispatch
    assert "DispatchGateMode<true" in gate_stage
    assert "DispatchGateMode<false" in gate_stage
    assert "RunPrepareStage<" not in gate_stage
    assert "RunPrepareStage<SAFE_GATE" in prepare_variant
    assert "RunPostWuStage<" not in prepare_variant
    assert "RunGateStage<" not in post_gate_front_stages
    assert "DispatchGateMode<" not in post_gate_front_stages
    assert "DispatchPrepareSafeGate<" in post_gate_front_stages
    assert post_gate_front_stages.count("RunPostWuStage<") == 1
    assert "DispatchGateMode<true" in front_stages
    assert "DispatchGateMode<false" in front_stages
    assert "DispatchPrepareSafeGate<" in front_stages
    assert front_stages.count("RunPostWuStage<") == 1
    assert "RunGateStage<" not in front_stages
    assert "RunPostGateFrontEndStages<" not in front_stages
    assert driver.index("RunGateStage<G_T>(") < driver.index(
        "RunPostGateFrontEndStages<"
    ) < driver.index("arch35::RunBackEnd<")
    assert driver.count("arch35::RunBackEnd<") == 1


def test_a5_regbase_triangular_state_update_orders_dependent_ub_rows():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    state_update = prepare.split(
        "static __simd_vf__ inline void ForwardSubDiag16Regbase", 1
    )[1].split("static __simd_vf__ inline void ApplyKdaRowScaleRegbase", 1)[0]

    assert state_update.count("sourceRow < row") >= 2
    assert state_update.count(
        "LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>()"
    ) >= 2


def test_a5_fwd_h_reads_predecayed_vector_gate_k_from_workspace_without_redecay():
    kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/"
        "op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")
    epilogue = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/"
        "op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
    ).read_text(encoding="utf-8")

    assert kernel.count("cube2OffsetK = kGated ? cube2Offsets.kDecayWorkOffset") == 2
    assert kernel.count("auto tensorK = kGated") >= 2
    assert epilogue.count("if constexpr (kGated)") >= 2
    assert "KDA passes kg = k * exp2(g_last - gk). Keep that decay exactly once." in epilogue
    assert epilogue.count("Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done)") >= 4


def test_fp16_score_pipeline_does_not_fall_back_to_two_row_cube_tiles():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    score_ref_block = prepare.split(
        "__aicore__ inline uint64_t ScoreRefBlockSize() const", 1
    )[1].split("__aicore__ inline uint64_t ScoreRowBlockCount", 1)[0]
    assert "KDA_SCORE_REF_BC = 32" in prepare
    assert "KDA_SAFE_SCORE_REF_BC = 32" in prepare
    assert "return KDA_SAFE_SCORE_REF_BC;" in score_ref_block
    assert "return KDA_SCORE_REF_BC;" in score_ref_block
    assert "return 2;" not in score_ref_block


def test_safe_fp16_score_pipeline_uses_bf16_on_all_supported_soc():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    score_type = prepare.split("using SCORE_T =", 1)[1].split(";", 1)[0]

    assert "SAFE_GATE && IsSameType<T, half>::value" in score_type
    assert "bfloat16_t" in score_type
    assert "__CCE_AICORE__" not in score_type


def test_fwd_h_varlen_metadata_is_resolved_locally_without_shared_writes():
    schedulers = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/block/"
        "block_scheduler_gdn_fwd_h.hpp",
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
        "chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/block/"
        "block_scheduler_gdn_fwd_h.hpp",
    )
    for path in schedulers:
        source = path.read_text(encoding="utf-8")
        init = source.split("void InitRuntime(", 1)[1].split(
            "void ResolveVarlenSequence(", 1
        )[0]
        resolve = source.split("void ResolveVarlenSequence(", 1)[1].split(
            "void InitNewStream(", 1
        )[0]

        assert "writeMetadata" not in init
        assert "chunkPrefix += batchChunk" in init
        assert "totalChunks = chunkPrefix" in init
        assert (
            "totalTokens = prevSeq" in init
            or "totalTokens = static_cast<uint64_t>(prevSeq)" in init
        )
        assert "inputTokenBatch = tokenBatch" in init
        assert "gmNumChunks.SetValue" not in source
        assert "gmNumSeq.SetValue" not in source
        assert "gmNumChunks.GetValue" not in source
        assert "gmNumSeq.GetValue" not in source
        assert "gmSeqlen.GetValue" in resolve
        assert "stream.chunkOffset" in resolve
        assert "stream.tokenOffset" in resolve

    generic_kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/"
        "gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")
    assert "ResolveWaveStream(waveIdx, initialStream)" in generic_kernel
    assert "const uint32_t chunkOffset = initialStream.chunkOffset;" in generic_kernel
    assert "GetVarlenChunkOffset(batchIdx)" not in generic_kernel
    assert "gmNumChunks.GetValue" not in generic_kernel
    assert "gmNumSeq.GetValue" not in generic_kernel

    arch35_kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
        "chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/"
        "gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")
    assert "GetVarlenChunkOffset(batchIdx)" in arch35_kernel
    assert "gmNumChunks.GetValue" not in arch35_kernel
    assert "gmNumSeq.GetValue" not in arch35_kernel


def test_prepare_aiv_overlaps_gate_mte2_with_vec_using_two_ub_slots():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    gate_bulk = prepare.split(
        "__aicore__ inline void PrepareGateProductsBulk", 1
    )[1].split("__aicore__ inline void PrepareGateProducts(", 1)[0]
    signal_output = prepare.split(
        "__aicore__ inline void SignalGateOutputDone()", 1
    )[1].split("template <typename CopyT>", 1)[0]
    pipelined_loop = gate_bulk.split(
        "for (uint64_t tileRow = rowBegin", 1
    )[1]

    assert "KDA_GATE_TILE_ROWS = 16" in prepare
    assert "KDA_GATE_PIPELINE_DEPTH = 3" in prepare
    assert "GatePipelineRows() * K_" in prepare
    assert "PrefetchQKGate(gateSlot" in gate_bulk
    assert "nextGateSlot = (gateSlot + 1) % KDA_GATE_PIPELINE_DEPTH" in gate_bulk
    assert "PrefetchQKGate(nextGateSlot" in gate_bulk
    assert pipelined_loop.index("WaitGateInputReady();") < pipelined_loop.index(
        "PrefetchQKGate(nextGateSlot"
    ) < pipelined_loop.index("if (useRef) {")
    assert pipelined_loop.index("WaitGateOutputForMte2(nextGateSlot);") < pipelined_loop.index(
        "PrefetchQKGate(nextGateSlot"
    )
    assert "SetFlag<HardEvent::MTE3_MTE2>" in signal_output
    assert "SetFlag<HardEvent::MTE3_V>" in signal_output
    assert "WaitFlag" not in signal_output


def test_prepare_uses_a5_regbase_gate_math_with_a2_a3_fallback():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    assert '#include "kernel_utils/vector/regbase.hpp"' in prepare
    assert "static __simd_vf__ inline void PrepareKdaGateQwRegbase" in prepare
    assert "static __simd_vf__ inline void PrepareKdaGateKgRegbase" in prepare
    assert "PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, true," in prepare
    assert "PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, false, false>" in prepare
    assert "PrepareKdaGateQwKgRegbase<T, T, GK_T, true, false, false>" in prepare
    assert "PrepareKdaGateQwKgRegbase<T, T, GK_T, false, false, false>" in prepare
    assert "ClampKdaGateRegbaseOutput" in prepare
    assert "KDA_EXP_INPUT_MAX" in prepare
    assert "KDA_EXP_INPUT_MIN" in prepare
    assert "row >= validRows" in prepare
    assert "#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310" in prepare
    assert "Cast(qTyped, outFp32, RoundMode::CAST_RINT" in prepare


def test_a5_fused_prepare_exports_final_kg_with_post_wu_semantics():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    score_factors = prepare.split(
        "__aicore__ inline void PrepareScoreFactorsBulk", 1
    )[1].split("__aicore__ inline void PrepareGateProductsBulk", 1)[0]
    fused_call = score_factors.split(
        "PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, true, true, true>", 1
    )[1].split("} else {", 1)[0]

    assert "reinterpret_cast<uint64_t>(vTyped.GetPhyAddr())" in fused_call
    assert "reinterpret_cast<uint64_t>(finalRefFp32.GetPhyAddr())" in fused_call
    assert "KdaPostWu::ComputePostKdaKgRegbase<T, GK_T>" not in fused_call


def test_a5_typical_post_wu_uses_regbase_kg_double_buffer():
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    post_aiv = post_wu.split(
        "__aicore__ inline void ProcessTypicalFullPostAivHeadWindow", 1
    )[1].split(
        "__aicore__ inline void ProcessTypicalFullPostAicHeadWindow", 1
    )[0]

    assert '#include "kernel_utils/vector/regbase.hpp"' in post_wu
    assert "KDA_TYPICAL_GATE_TILE_ROWS = 16" in post_wu
    assert "KDA_TYPICAL_GATE_PIPELINE_ROWS = 32" in post_wu
    assert "KDA_TYPICAL_GATE_PIPELINE_STAGES = 3" in post_wu
    assert "ComputePostKdaKgRegbase" in post_wu
    assert "PrefetchTypicalKg(slot ^ 1" in post_wu
    assert "ProcessTypicalFullPostAivHeadWindow" in post_wu
    assert "PrefetchTypicalKgPipeline" in post_wu
    assert "TypicalGatePipelineRef" in post_wu
    assert "CanPipelineTypicalKg" in post_wu
    assert "for (uint32_t lane = 0; lane < headCnt; ++lane)" in post_aiv
    assert "ComputeTypicalKgPipelineRegs" in post_aiv
    assert "StoreTypicalKgPipeline" in post_aiv
    assert "CrossCoreWaitFlagWithReverse" not in post_aiv


def test_a5_post_wu_keeps_full_chunks_fused_and_handles_tail_in_prepare():
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    full_pipeline_dispatch = post_wu.split(
        "__aicore__ inline bool UseFullPostWuPipelineArch35", 1
    )[1].split("__aicore__ inline void ComputePostWuCubeFusedArch35", 1)[0]
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    common = KERNEL_COMMON.read_text(encoding="utf-8")

    assert "curT == 64" in full_pipeline_dispatch
    assert "ProcessPreparedFullHeadBatchArch35" in post_wu
    assert "ProcessPreparedTailSingleArch35" in post_wu
    assert "postWu.ProcessPreparedTailSingleArch35" in prepare
    assert "struct FusedTailHeadWindowState" in prepare
    tail_pipeline = prepare.split(
        "__aicore__ inline void ProcessOwnedChunkAicHeadWindowFusedTail", 1
    )[1].split(
        "__aicore__ inline void ProcessOwnedChunkAicHeadsFusedTail", 1
    )[0]
    assert "for (uint32_t headOffset = 0; headOffset < headCnt;" in tail_pipeline
    score_pos = tail_pipeline.index("ProcessChunkPreAic(")
    consume_previous_pos = tail_pipeline.index("DrainTailSinglePostWu")
    publish_current_pos = tail_pipeline.index("pending.valid = true;")
    assert consume_previous_pos < score_pos < publish_current_pos
    tail_drain = prepare.split(
        "__aicore__ inline void DrainFusedTailHeadWindowState", 1
    )[1].split(
        "__aicore__ inline void ProcessOwnedChunkAicHeadWindowTaskFusedTail", 1
    )[0]
    assert "DrainTailSinglePostWu(postWu, state.singlePending);" in tail_drain
    assert "state.singlePending.valid = false;" in tail_drain
    arch35_post_wu = common.split(
        "__aicore__ inline void RunPostWuStage", 1
    )[1].split("template <typename T, typename TileShapes", 1)[0]
    arch35_only = arch35_post_wu.split(
        "#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310", 1
    )[0]
    assert "RunChunkKdaPostWuTailSeedCopy" not in arch35_only
    assert "RunChunkKdaPostWuTail" not in arch35_only


def test_a5_prepare_closes_mode2_score_stream_before_deferred_solve():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    producer = prepare.split(
        "__attribute__((noinline)) __aicore__ void ProcessChunkPreAivFp32", 1
    )[1].split(
        "__aicore__ inline void FinishDeferredSafeChunk", 1
    )[0]
    consumer = prepare.split(
        "__aicore__ inline void ProcessChunkPreAicFp32", 1
    )[1].split("struct OwnedChunkDesc", 1)[0]

    assert "KDA_DIRECT_SCORE_SUBBLOCK_FLAG_STRIDE" not in prepare
    assert "CrossCoreSetFlag<0x4" not in prepare
    assert "CrossCoreWaitFlag<0x4" not in prepare
    assert "JoinAivMte3();" in producer
    assert "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);" in producer
    assert "CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);" in producer
    assert producer.index("JoinAivMte3();") < producer.index(
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);"
    )
    assert "CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);" in consumer
    assert "CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);" in consumer
    assert consumer.index(
        "CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);"
    ) < consumer.index("ComputeRawAqkAkkCubeBlock(") < consumer.index(
        "CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);"
    )


def test_a5_fwd_h_uses_real_two_slot_mode2_queues_and_drains_credits():
    fwd_h = ARCH35_FWD_H.read_text(encoding="utf-8")
    acquire_direct = fwd_h.split(
        "__aicore__ inline LocalTensor<float> AcquireDirectBufferAiv", 1
    )[1].split("__aicore__ inline void SetDirectFreeAiv", 1)[0]
    initialize_direct = fwd_h.split(
        "__aicore__ inline void InitializeDirectFreeCreditsAiv", 1
    )[1].split("__aicore__ inline void DrainDirectFreeCreditsAic", 1)[0]
    drain_direct = fwd_h.split(
        "__aicore__ inline void DrainDirectFreeCreditsAic", 1
    )[1].split("__aicore__ inline void DrainL1FreeCreditsAiv", 1)[0]
    drain_l1 = fwd_h.split(
        "__aicore__ inline void DrainL1FreeCreditsAiv", 1
    )[1].split("__aicore__ inline void SelectSequence", 1)[0]
    init_l0c = fwd_h.split(
        "__aicore__ inline void InitL0CPipelineAic", 1
    )[1].split("__aicore__ inline uint32_t AcquireL0CSlotAic", 1)[0]
    acquire_l0c = fwd_h.split(
        "__aicore__ inline uint32_t AcquireL0CSlotAic", 1
    )[1].split("__aicore__ inline void DrainL0CPipelineAic", 1)[0]
    drain_l0c = fwd_h.split(
        "__aicore__ inline void DrainL0CPipelineAic", 1
    )[1].split("__aicore__ inline void SelectSequence", 1)[0]
    publish_direct = fwd_h.split(
        "__aicore__ inline void PublishDirectTile", 1
    )[1].split("__aicore__ inline void PublishDirect(", 1)[0]
    state_product = fwd_h.split(
        "__aicore__ inline void ComputeStateProductsAic", 1
    )[1].split("__aicore__ inline void ComputeVnewProductsAic", 1)[0]
    vnew_product = fwd_h.split(
        "__aicore__ inline void ComputeVnewProductsAic", 1
    )[1].split("__aicore__ inline void ProcessAic()", 1)[0]
    process_aic = fwd_h.split(
        "__aicore__ inline void ProcessAic()", 1
    )[1].split("template <bool HAS_TAIL>", 1)[0]
    process_aiv = fwd_h.split(
        "__aicore__ inline void ProcessAiv()", 1
    )[1].split("template <bool HAS_TAIL>", 1)[0]
    process_chunk_aiv = fwd_h.split(
        "__aicore__ inline void ProcessChunkAiv", 1
    )[1].split("__aicore__ inline void ProcessAiv()", 1)[0]

    def kib_offset(name):
        match = re.search(rf"{name} = (\d+) \* 1024;", fwd_h)
        assert match is not None
        return int(match.group(1))

    def flag_id(name):
        match = re.search(rf"constexpr uint64_t {name} = (\d+);", fwd_h)
        assert match is not None
        return int(match.group(1))

    def local_event_id(name):
        match = re.search(rf"constexpr TEventID {name} = (\d+);", fwd_h)
        assert match is not None
        return int(match.group(1))

    expected_flag_ids = {
        "KDA_FWD_H_DIRECT_FREE_FLAG": 0,
        "KDA_FWD_H_DIRECT_READY_FLAG": 1,
        "KDA_FWD_H_STATE_FREE_FLAG": 2,
        "KDA_FWD_H_STATE_READY_FLAG": 3,
        "KDA_FWD_H_VNEW_FREE_FLAG": 4,
        "KDA_FWD_H_VNEW_READY_FLAG": 5,
    }
    actual_flag_ids = {
        name: flag_id(name)
        for name in expected_flag_ids
    }
    assert actual_flag_ids == expected_flag_ids
    assert set(actual_flag_ids.values()) == set(range(6))

    # flag 0..5 六个方向必须独占，禁止退回保留 ID、同 ID 双向或 +16 分流。
    source_without_comments = re.sub(
        r"//.*?$|/\*.*?\*/", "", fwd_h,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert max(actual_flag_ids.values()) <= 7
    assert not ({8, 9, 10} & set(actual_flag_ids.values()))
    assert "KDA_FWD_H_L1_FREE_FLAG" not in fwd_h
    assert "KDA_FWD_H_L1_READY_FLAG" not in fwd_h
    assert not re.search(r"\+\s*16\b", source_without_comments)
    assert "CrossCoreFlagWithReverse" not in fwd_h
    assert "CrossCoreSetFlagWithReverse" not in fwd_h
    assert "CrossCoreWaitFlagWithReverse" not in fwd_h
    assert local_event_id("KDA_FWD_H_IO_REUSE_EVENT") == 0
    assert local_event_id("KDA_FWD_H_IO_REUSE_EVENT") <= 5
    assert "static_assert(KDA_FWD_H_IO_REUSE_EVENT <= 5" in fwd_h

    assert "constexpr uint32_t KDA_FWD_H_DIRECT_BUFFER_DEPTH = 2;" in fwd_h
    direct_begin = kib_offset("KDA_FWD_H_UB_DIRECT_OFFSET")
    direct_slot_size = kib_offset("KDA_FWD_H_UB_DIRECT_SLOT_BYTES")
    vnew_begin = kib_offset("KDA_FWD_H_UB_VNEW_OFFSET")
    assert direct_begin + 2 * direct_slot_size <= vnew_begin
    assert "directPublishIndex_ % KDA_FWD_H_DIRECT_BUFFER_DEPTH" in publish_direct
    assert "directConsumeIndex_ % KDA_FWD_H_DIRECT_BUFFER_DEPTH" in acquire_direct
    for block in (publish_direct, acquire_direct):
        assert "slot * KDA_FWD_H_UB_DIRECT_SLOT_BYTES" in block or (
            "directSlot * KDA_FWD_H_UB_DIRECT_SLOT_BYTES" in block
        )

    assert "constexpr uint32_t KDA_FWD_H_L1_BUFFER_DEPTH = 2;" in fwd_h
    state_ranges = [
        (kib_offset("KDA_FWD_H_L1_H_OFFSET"), 32),
        (kib_offset("KDA_FWD_H_L1_H1_OFFSET"), 32),
    ]
    vnew_ranges = [
        (kib_offset("KDA_FWD_H_L1_V_OFFSET"), 16),
        (kib_offset("KDA_FWD_H_L1_V1_OFFSET"), 16),
    ]
    all_ranges = state_ranges + vnew_ranges
    for index, (begin, size) in enumerate(all_ranges):
        for other_begin, other_size in all_ranges[index + 1 :]:
            assert begin + size <= other_begin or other_begin + other_size <= begin
    assert "statePublishCount_[subBlockIdx] % KDA_FWD_H_L1_BUFFER_DEPTH" in fwd_h
    assert "stateConsumeIndex_ % KDA_FWD_H_L1_BUFFER_DEPTH" in fwd_h
    assert "vnewPublishCount_[subBlockIdx] % KDA_FWD_H_L1_BUFFER_DEPTH" in fwd_h
    assert "vnewConsumeIndex_ % KDA_FWD_H_L1_BUFFER_DEPTH" in fwd_h
    assert "return slot == 0 ? KDA_FWD_H_L1_H_OFFSET : KDA_FWD_H_L1_H1_OFFSET;" in fwd_h
    assert "return slot == 0 ? KDA_FWD_H_L1_V_OFFSET : KDA_FWD_H_L1_V1_OFFSET;" in fwd_h

    assert "constexpr uint32_t KDA_FWD_H_L0C_BUFFER_DEPTH = 2;" in fwd_h
    assert "constexpr uint32_t KDA_FWD_H_L0C_SLOT_BYTES = 128 * 128 * sizeof(float);" in fwd_h
    assert (
        "KDA_FWD_H_L0C_BUFFER_DEPTH * KDA_FWD_H_L0C_SLOT_BYTES <= ArchTag::L0C_SIZE"
        in fwd_h
    )
    assert "slot < KDA_FWD_H_L0C_BUFFER_DEPTH" in init_l0c
    assert init_l0c.count("SetFlag<HardEvent::FIX_M>(L0CEvent(slot));") == 1
    assert "WaitFlag<HardEvent::FIX_M>" not in init_l0c
    assert "l0cProductIndex_ % KDA_FWD_H_L0C_BUFFER_DEPTH" in acquire_l0c
    assert "WaitFlag<HardEvent::FIX_M>(L0CEvent(slot));" in acquire_l0c
    assert "slot < KDA_FWD_H_L0C_BUFFER_DEPTH" in drain_l0c
    assert drain_l0c.count("WaitFlag<HardEvent::FIX_M>(L0CEvent(slot));") == 1
    for product in (state_product, vnew_product):
        assert "AcquireL0CSlotAic();" in product
        assert "l0cSlot * KDA_FWD_H_L0C_SLOT_BYTES" in product
        assert "true, 0);" in product
        assert "0b11" not in product
        assert "l0cSlot);" in product
    assert "copyL0CToDst(tensorUb, tensorL0C);" in publish_direct
    assert "++l0cProductIndex_;" in publish_direct
    assert "SetFlag<HardEvent::FIX_M>(l0cEvent);" in publish_direct
    assert "0b11" not in fwd_h
    assert "SetMMLayoutTransform" not in fwd_h
    assert fwd_h.count("InitL0CPipelineAic();") == 1
    assert fwd_h.count("DrainL0CPipelineAic();") == 1
    assert process_aic.index("InitL0CPipelineAic();") < process_aic.index(
        "KdaForward::CompactSequencePlanView plan(compactPlanAddr_);"
    )

    assert "CrossCoreSetFlag<0x4" not in fwd_h
    assert "CrossCoreWaitFlag<0x4" not in fwd_h
    assert "KDA_FWD_H_SUBBLOCK_FLAG_OFFSET" not in fwd_h
    set_modes = re.findall(r"CrossCoreSetFlag<([^,>]+),", fwd_h)
    assert set_modes and set(set_modes) == {"0x2"}
    assert "CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FWD_H_DIRECT_READY_FLAG);" in fwd_h
    assert "CrossCoreSetFlag<0x2, PIPE_V>(KDA_FWD_H_DIRECT_FREE_FLAG);" in fwd_h
    assert fwd_h.count("CrossCoreSetFlag<0x2, PIPE>(flag);") == 2
    assert "SetL1SlotFlagAivToAic<PIPE_MTE3>(KDA_FWD_H_STATE_READY_FLAG);" in fwd_h
    assert "SetL1SlotFlagAivToAic<PIPE_MTE3>(KDA_FWD_H_VNEW_READY_FLAG);" in fwd_h
    assert "SetL1SlotFlagAicToAiv<PIPE_FIX>(KDA_FWD_H_STATE_FREE_FLAG);" in fwd_h
    assert "SetL1SlotFlagAicToAiv<PIPE_FIX>(KDA_FWD_H_VNEW_FREE_FLAG);" in fwd_h
    assert "slot < KDA_FWD_H_DIRECT_BUFFER_DEPTH" in initialize_direct
    assert "SetDirectFreeAiv();" in initialize_direct
    assert "slot < KDA_FWD_H_DIRECT_BUFFER_DEPTH" in drain_direct
    assert "WaitDirectFreeAic();" in drain_direct
    assert "stateCredits" in drain_l1 and "KDA_FWD_H_STATE_FREE_FLAG" in drain_l1
    assert "vnewCredits" in drain_l1 and "KDA_FWD_H_VNEW_FREE_FLAG" in drain_l1
    assert "GetSubBlockIdx()" in process_aiv
    assert "GetSubBlockNum()" in process_aiv
    assert "GetBlockIdx()) / subBlockNum" in process_aiv
    assert process_aic.index("if (!SelectHeadRange(coreIdx))") < process_aic.index(
        "InitL0CPipelineAic();"
    )
    assert process_aiv.index("if (!SelectHeadRange(coreIdx))") < process_aiv.index(
        "InitializeDirectFreeCreditsAiv();"
    )
    assert process_aiv.index("InitializeDirectFreeCreditsAiv();") < process_aiv.index(
        "KdaForward::CompactSequencePlanView plan(compactPlanAddr_);"
    )

    # 两个 AIV 走同一 ProcessChunk：每份 direct 都在读取完成后归还一次。
    direct_events = re.findall(
        r"(?:AcquireDirectBufferAiv|SetDirectFreeAiv)\(\)", process_chunk_aiv
    )
    assert direct_events == [
        "AcquireDirectBufferAiv()",
        "SetDirectFreeAiv()",
        "AcquireDirectBufferAiv()",
        "SetDirectFreeAiv()",
    ]
    assert process_aic.index("DrainDirectFreeCreditsAic();") < process_aic.index(
        "WaitFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);"
    )
    assert process_aic.index("DrainDirectFreeCreditsAic();") < process_aic.index(
        "DrainL0CPipelineAic();"
    )
    assert process_aic.index("DrainL0CPipelineAic();") < process_aic.index(
        "WaitFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);"
    )
    assert process_aiv.index(
        "WaitFlag<HardEvent::MTE3_MTE2>(aivMte3ToMte2Event_);"
    ) < process_aiv.index("DrainL1FreeCreditsAiv(subBlockIdx);")


def test_a5_fwd_h_dispatches_staging_events_as_literals_and_closes_each_slot():
    fwd_h = ARCH35_FWD_H.read_text(encoding="utf-8")
    wait_helper = fwd_h.split(
        "__aicore__ inline void WaitL1StagingSlotFreeAic", 1
    )[1].split("__aicore__ inline void SetL1StagingSlotFreeAic", 1)[0]
    set_helper = fwd_h.split(
        "__aicore__ inline void SetL1StagingSlotFreeAic", 1
    )[1].split("template <bool IS_TAIL = false>", 1)[0]
    prefetch = fwd_h.split(
        "__aicore__ inline void PrefetchIndependentProductsAic", 1
    )[1].split("__aicore__ inline void ComputeStateProductsAic", 1)[0]
    vnew = fwd_h.split(
        "__aicore__ inline void ComputeVnewProductsAic", 1
    )[1].split("__aicore__ inline void ProcessAic()", 1)[0]
    process_aic = fwd_h.split(
        "__aicore__ inline void ProcessAic()", 1
    )[1].split("template <bool HAS_TAIL>", 1)[0]

    assert "aicL1ReuseEvents_" not in fwd_h
    for helper, operation in (
        (wait_helper, "WaitFlag"),
        (set_helper, "SetFlag"),
    ):
        for slot in range(4):
            case_body = helper.split(f"case {slot}:", 1)[1].split("return;", 1)[0]
            assert case_body.count(
                f"{operation}<HardEvent::MTE1_MTE2>({slot});"
            ) == 1
            assert re.findall(
                rf"{operation}<HardEvent::MTE1_MTE2>\((\d)\);", case_body
            ) == [str(slot)]
    assert fwd_h.count("WaitFlag<HardEvent::MTE1_MTE2>(") == 4
    assert fwd_h.count("SetFlag<HardEvent::MTE1_MTE2>(") == 4
    assert "static_assert(KDA_FWD_H_L1_STAGING_DEPTH == 4" in fwd_h
    assert "chunk & 3" in prefetch
    assert "chunk & 3" in vnew
    assert prefetch.count("WaitL1StagingSlotFreeAic(slot);") == 1
    assert vnew.count("SetL1StagingSlotFreeAic(slot);") == 1
    assert prefetch.index("WaitL1StagingSlotFreeAic(slot);") < prefetch.index(
        "CopyGmToL1ARmW{}(tensorL1W, blockW);"
    )
    assert vnew.index("copyL1ToL0AK(tensorL0Kg, tensorL1Kg);") < vnew.index(
        "SetL1StagingSlotFreeAic(slot);"
    )
    assert vnew.index("copyL1ToL0B(tensorL0V, tensorL1V);") < vnew.index(
        "SetL1StagingSlotFreeAic(slot);"
    )
    assert vnew.index("PublishDirect<DirectTileCopyCM>(") < vnew.index(
        "SetL1StagingSlotFreeAic(slot);"
    )
    for slot in range(4):
        assert process_aic.count(f"SetL1StagingSlotFreeAic({slot});") == 1
        assert process_aic.count(f"WaitL1StagingSlotFreeAic({slot});") == 1
    assert process_aic.index("SetL1StagingSlotFreeAic(3);") < process_aic.index(
        "KdaForward::CompactSequencePlanView plan(compactPlanAddr_);"
    )
    assert process_aic.index("for (uint32_t ordinal = 0;") < process_aic.index(
        "WaitL1StagingSlotFreeAic(0);"
    )
    assert process_aic.rindex("ProcessSelectedSequenceAic<true>(b);") < process_aic.index(
        "WaitL1StagingSlotFreeAic(0);"
    )

    # 每槽入口只有一份free credit；每个payload先消费再归还，退出时清零。
    for sequence_chunks in ((0,), (1,), (3, 5), (257, 0, 4)):
        credits = [1] * 4
        for chunk_count in sequence_chunks:
            for _head in range(4):
                for chunk in range(chunk_count):
                    slot = chunk % 4
                    assert credits[slot] == 1
                    credits[slot] -= 1
                    credits[slot] += 1
        for slot in range(4):
            assert credits[slot] == 1
            credits[slot] -= 1
        assert credits == [0, 0, 0, 0]


def test_a5_fwd_h_raw_mode2_protocol_model_closes_every_channel():
    depth = 2
    aiv_count = 2

    def run_direct_channel(transaction_count):
        free_tokens = 0
        ready_tokens = [0] * aiv_count
        aic_ready_sets = 0
        aiv_ready_waits = [0] * aiv_count
        aiv_free_sets = [0] * aiv_count
        aic_free_waits = 0
        producer_slots = []
        consumer_slots = [[] for _ in range(aiv_count)]

        # 两个 AIV 各 set 一次才聚合成 AIC 可见的一个 mode2 free token。
        for _ in range(depth):
            for aiv in range(aiv_count):
                aiv_free_sets[aiv] += 1
            free_tokens += 1

        for ordinal in range(transaction_count):
            assert free_tokens > 0
            free_tokens -= 1
            aic_free_waits += 1

            slot = ordinal % depth
            producer_slots.append(slot)
            aic_ready_sets += 1
            for aiv in range(aiv_count):
                ready_tokens[aiv] += 1
                assert ready_tokens[aiv] > 0
                ready_tokens[aiv] -= 1
                aiv_ready_waits[aiv] += 1
                consumer_slots[aiv].append(slot)
                aiv_free_sets[aiv] += 1
            free_tokens += 1

        # 阶段出口排空入口 seed 对应的两份 credit。
        for _ in range(depth):
            assert free_tokens > 0
            free_tokens -= 1
            aic_free_waits += 1

        assert free_tokens == 0
        assert ready_tokens == [0] * aiv_count
        assert aiv_ready_waits == [aic_ready_sets] * aiv_count
        assert aiv_free_sets == [aic_free_waits] * aiv_count
        assert all(slots == producer_slots for slots in consumer_slots)
        assert producer_slots == [ordinal % depth for ordinal in range(transaction_count)]

    def run_l1_channel(transaction_count):
        ready_tokens = 0
        free_tokens = [0] * aiv_count
        aiv_ready_sets = [0] * aiv_count
        aic_ready_waits = 0
        aic_free_sets = 0
        aiv_free_waits = [0] * aiv_count
        producer_slots = [[] for _ in range(aiv_count)]
        consumer_slots = []

        for ordinal in range(transaction_count):
            slot = ordinal % depth
            for aiv in range(aiv_count):
                if ordinal >= depth:
                    assert free_tokens[aiv] > 0
                    free_tokens[aiv] -= 1
                    aiv_free_waits[aiv] += 1
                producer_slots[aiv].append(slot)
                aiv_ready_sets[aiv] += 1

            # 两个 AIV 的 ready set 聚合为 AIC 的一次 wait。
            ready_tokens += 1
            assert ready_tokens > 0
            ready_tokens -= 1
            aic_ready_waits += 1
            consumer_slots.append(slot)

            # AIC 的一次 free set 同时发给两个 AIV。
            aic_free_sets += 1
            for aiv in range(aiv_count):
                free_tokens[aiv] += 1

        drain_count = min(transaction_count, depth)
        for aiv in range(aiv_count):
            for _ in range(drain_count):
                assert free_tokens[aiv] > 0
                free_tokens[aiv] -= 1
                aiv_free_waits[aiv] += 1

        assert ready_tokens == 0
        assert free_tokens == [0] * aiv_count
        assert aiv_ready_sets == [aic_ready_waits] * aiv_count
        assert aiv_free_waits == [aic_free_sets] * aiv_count
        assert all(slots == consumer_slots for slots in producer_slots)
        assert consumer_slots == [ordinal % depth for ordinal in range(transaction_count)]

    cases = [
        ("empty", 0, 0, False),
        *((f"head_count_{head_count}", head_count, 1, False)
          for head_count in range(1, 5)),
        ("odd_tail", 1, 2, True),
        ("even_tail", 2, 1, True),
        ("a5_96_heads_256_chunks", 96, 256, False),
    ]
    for _case_name, head_count, full_chunks, has_tail in cases:
        payload_count = head_count * (full_chunks + int(has_tail))
        direct_count = payload_count * 2
        run_direct_channel(direct_count)
        run_l1_channel(payload_count)
        run_l1_channel(payload_count)

    # 单通道额外覆盖 0/1/2/3 和大 N，直接验证奇数尾不会遗留 credit。
    for transaction_count in (0, 1, 2, 3, 96 * 256):
        run_direct_channel(transaction_count)
        run_l1_channel(transaction_count)


def test_a5_post_wu_initializes_only_slots_consumed_by_each_full_run():
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    aic_pipeline = post_wu.split(
        "__aicore__ inline void ProcessTypicalFullPostAicHeadWindow", 1
    )[1].split("#endif", 1)[0]

    assert "InitializePostWuPipelineSlot(slot);" in aic_pipeline
    assert "InitializePostWuPipelineEvents();" not in aic_pipeline
    assert "if (!reuseSlot) {" in aic_pipeline
    assert "InitializePostWuPipelineSlot(nextSlot);" in aic_pipeline
    next_slot_init = aic_pipeline.index("InitializePostWuPipelineSlot(nextSlot);")
    assert next_slot_init < aic_pipeline.index(
        "PrefetchPostWuPipelineArch35(", next_slot_init
    )


def test_a5_post_wu_local_pipeline_events_avoid_reserved_ids_and_overlap():
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")

    def constant(name):
        match = re.search(rf"constexpr uint(?:16|32)_t {name} = (\d+);", post_wu)
        assert match is not None
        return int(match.group(1))

    depth = constant("KDA_POST_PIPELINE_STAGE_COUNT")
    main_begin = constant("KDA_POST_EVENT")
    u_begin = constant("KDA_POST_PIPELINE_U_EVENT")
    main_events = set(range(main_begin, main_begin + depth))
    u_events = set(range(u_begin, u_begin + depth))

    assert depth == 2
    assert main_events == {3, 4}
    assert u_events == {0, 1}
    assert max(main_events | u_events) <= 5
    assert main_events.isdisjoint(u_events)
    assert post_wu.count(
        "KDA_POST_PIPELINE_U_EVENT + KDA_POST_PIPELINE_STAGE_COUNT - 1 <= 5"
    ) == 1
    assert "PostWU main and U pipelines must use disjoint event IDs" in post_wu


def test_a5_fused_full_post_wu_batches_runtime_heads_into_two_slots():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    full_state = prepare.split("struct FusedFullHeadWindowState", 1)[1].split(
        "struct FusedTailHeadWindowState", 1
    )[0]
    flush = prepare.split(
        "__aicore__ inline void FlushFusedFullPostWuBatch", 1
    )[1].split("__aicore__ inline void EnqueueFusedFullPostWuTask", 1)[0]
    enqueue = prepare.split(
        "__aicore__ inline void EnqueueFusedFullPostWuTask", 1
    )[1].split(
        "__aicore__ inline void ProcessOwnedChunkAicHeadWindowFusedFull", 1
    )[0]
    full_window = prepare.split(
        "__aicore__ inline void ProcessOwnedChunkAicHeadWindowFusedFull", 1
    )[1].split(
        "__aicore__ inline void ProcessOwnedChunkAicHeadsFusedFull", 1
    )[0]
    full_drain = prepare.split(
        "__aicore__ inline void DrainFusedFullHeadWindowState", 1
    )[1].split(
        "__aicore__ inline void ProcessOwnedChunkAicHeadWindowTaskFusedFull", 1
    )[0]
    full_dispatch = prepare.split(
        "__aicore__ inline void ProcessOwnedChunkAicHeadWindowTaskFusedFull", 1
    )[1].split("__aicore__ inline void DrainFusedTailHeadWindowState", 1)[0]
    tail_dispatch = prepare.split(
        "__aicore__ inline void ProcessOwnedChunkAicHeadWindowTaskFusedTail", 1
    )[1].split("__aicore__ inline void ProcessPreAicHeadWindowsFused", 1)[0]
    fused_stage = prepare.split(
        "__aicore__ inline void ProcessPreAicHeadWindowsFused", 1
    )[1].split("__aicore__ inline void ProcessPreAiv", 1)[0]

    single_wrapper = post_wu.split(
        "__aicore__ inline void ProcessPreparedFullHeadBatchArch35", 1
    )[1].split(
        "__attribute__((noinline)) __aicore__ void ProcessPreparedTailSingleArch35", 1
    )[0]
    full_items = post_wu.split(
        "__aicore__ inline void ProcessPreparedFullHeadBatchItemsArch35", 1
    )[1].split("#endif", 1)[0]
    tail_single = post_wu.split(
        "__attribute__((noinline)) __aicore__ void ProcessPreparedTailSingleArch35", 1
    )[1].split("#endif", 1)[0]

    assert "KDA_POST_QUEUE_STORAGE" not in prepare
    assert "batchB[KDA_POST_QUEUE_DEPTH]" in full_state
    assert "batchHv[KDA_POST_QUEUE_DEPTH]" in full_state
    assert "batchStart[KDA_POST_QUEUE_DEPTH]" in full_state
    assert "uint16_t batchCount = 0;" in full_state
    assert "singlePending" not in full_state

    assert "for (uint16_t i = 0; i < batchCount; ++i)" in flush
    assert "CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>" in flush
    assert "postWuReadyFlag_" in flush
    assert "ProcessPreparedFullHeadBatchArch35" in flush
    enqueue_store = enqueue.index("state.batchB[state.batchCount] = b;")
    enqueue_increment = enqueue.index("++state.batchCount;")
    enqueue_boundary = enqueue.index(
        "state.batchCount != KDA_POST_QUEUE_DEPTH"
    )
    enqueue_flush = enqueue.index("FlushFusedFullPostWuBatch(")
    enqueue_reset = enqueue.index("state.batchCount = 0;")
    assert enqueue_store < enqueue_increment < enqueue_boundary < enqueue_flush
    assert enqueue_flush < enqueue_reset
    assert "state.batchCount = 1;" not in enqueue
    assert "state.batchB[0] =" not in enqueue
    assert "state.batchHv[0] =" not in enqueue
    assert "state.batchStart[0] =" not in enqueue
    assert "for (uint32_t headOffset = 0; headOffset < headCnt;" in full_window
    assert "EnqueueFusedFullPostWuTask(" in full_window
    for full_path in (full_window, full_drain, full_dispatch):
        assert "ProcessPreparedTailSingleArch35" not in full_path
    assert "FlushFusedFullPostWuBatch(" in full_drain
    assert "state.batchCount = 0;" in full_drain
    assert "ProcessOwnedChunkAicHeadsFusedFull" in full_dispatch
    assert "ProcessOwnedChunkAicHeadsFusedTail" in tail_dispatch
    assert fused_stage.count(
        "DrainFusedTailHeadWindowState(postWu, tailState);"
    ) == 2

    assert "ProcessPreparedFullHeadBatchItemsArch35" in single_wrapper
    assert "batchB, batchHv, batchStart, taskCount" in single_wrapper
    assert "const uint16_t itemCount = taskCount;" in full_items
    assert full_items.count("InitializePostWuPipelineSlot(slot);") == 1
    assert "if (!reuseSlot)" in full_items
    assert "InitializePostWuPipelineSlot(nextSlot);" in full_items
    assert "nextItem >= KDA_POST_PIPELINE_STAGE_COUNT" in full_items
    assert "slot ^= 1;" in full_items
    assert "FinalizePostWuPipelineEvents(usedSlotCount);" in full_items
    assert "InitializePostWuPipelineEvents" not in post_wu

    assert "curT" in tail_single
    assert tail_single.count("InitializePostWuPipelineSlot(0);") == 1
    assert "PrefetchPostWuPipelineArch35(resource, 0, b, hv, start, curT, false);" in tail_single
    assert "FinalizePostWuPipelineEvents(1);" in tail_single


def test_a5_nonfused_post_wu_consumes_host_chunk_cursor_with_runtime_head_windows():
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    scheduler = post_wu.split("struct OwnedChunkDesc", 1)[1].split(
        "private:\n    GlobalTensor<T> q_", 1
    )[0]
    stage = scheduler.split(
        "__aicore__ inline void ProcessCompactPostStage", 1
    )[1].split("__aicore__ inline void ProcessPostAiv", 1)[0]
    g1 = scheduler.split(
        "__aicore__ inline void ProcessG1PostStage", 1
    )[1].split("__aicore__ inline void ProcessGroupedPostStage", 1)[0]
    grouped = scheduler.split(
        "__aicore__ inline void ProcessGroupedPostStage", 1
    )[1].split("__aicore__ inline void ProcessCompactPostStage", 1)[0]
    head_range = scheduler.split(
        "__aicore__ inline void ProcessCompactPostHeadRange", 1
    )[1].split("__aicore__ inline void ProcessG1PostStage", 1)[0]
    aux = scheduler.split(
        "__aicore__ inline void ProcessTailAuxStageFromPlan", 1
    )[1].split("__aicore__ inline void ProcessVarlenTailSeedCopyAiv", 1)[0]

    for forbidden in (
        "tasksPerCore",
        "taskNum",
        "GetHeadMajorTaskRange",
        "ResolveHeadMajorChunk",
        "ProcessPostAicPipelineArch35",
        "ProcessPostAivPipelineArch35",
    ):
        assert forbidden not in scheduler

    assert stage.index("LoadChunkCoreCursor") < stage.index(
        "if (plan.HeadGroupCount() == 1)"
    )
    assert stage.count("if (plan.HeadGroupCount() == 1)") == 1
    assert "ProcessG1PostStage<IS_AIC>" in stage
    assert "ProcessGroupedPostStage<IS_AIC>" in stage

    assert "LoadOwnedFullChunk" in g1
    assert "LoadOwnedTailChunk" in g1
    assert "ProcessCompactPostHeadRange<IS_AIC, false>" in g1
    assert "ProcessCompactPostHeadRange<IS_AIC, true>" in g1
    assert "LoadGroupedFullTask" in grouped
    assert "LoadGroupedTailTask" in grouped
    assert "ProcessCompactPostHeadRange<IS_AIC, false>" in grouped
    assert "ProcessCompactPostHeadRange<IS_AIC, true>" in grouped

    assert "KdaForward::HeadWindowHeadCount(" in head_range
    assert "static_cast<uint32_t>(H_)" in head_range
    assert "static_cast<uint32_t>(HV_)" in head_range
    assert "head += headCnt" in head_range
    assert "head += KDA_HEADS_PER_TASK" not in head_range
    assert "ProcessCompactPostHeadWindow<IS_AIC, IS_TAIL>(" in head_range
    assert "chunk, head, subBlockIdx, subBlockNum, headCnt);" in head_range
    assert "ProcessCompactPostHeadWindow<IS_AIC, IS_TAIL," not in head_range
    assert "ProcessTailAuxHeadWindow<COPY_SEED, CUBE_AIC>(" in scheduler
    assert "ProcessTailAuxHeadWindow<COPY_SEED, CUBE_AIC," not in scheduler
    assert "if constexpr (IS_TAIL)" in scheduler
    assert "curT < BT_" not in scheduler

    assert aux.index("LoadChunkCoreCursor") < aux.index(
        "if (plan.HeadGroupCount() == 1)"
    )
    assert "LoadOwnedTailChunk" in aux
    assert "LoadGroupedTailTask" in aux
    assert "ProcessTailAuxStageFromPlan<true, false>();" in scheduler
    assert "ProcessTailAuxStageFromPlan<false, true>();" in scheduler
    assert "ProcessTailAuxStageFromPlan<false, false>();" in scheduler
    assert "ProcessCompactPostStage<false>();" in scheduler
    assert "ProcessCompactPostStage<true>();" in scheduler


def test_a5_finalize_allocates_each_tail_scalar_event_from_its_own_pool():
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    assert "vToSEvent_ = pipe_->AllocEventID<HardEvent::V_S>();" in finalize
    assert "sToVEvent_ = pipe_->AllocEventID<HardEvent::S_V>();" in finalize
    assert "sToMte2Event_ = pipe_->AllocEventID<HardEvent::S_MTE2>();" in finalize
    assert "pipe_->ReleaseEventID<HardEvent::V_S>(vToSEvent_);" in finalize
    assert "pipe_->ReleaseEventID<HardEvent::S_V>(sToVEvent_);" in finalize
    assert "pipe_->ReleaseEventID<HardEvent::S_MTE2>(sToMte2Event_);" in finalize
    tail = finalize.split("__aicore__ inline void ComputeTailLocalRows", 1)[1].split(
        "template <typename CopyT>", 1
    )[0]
    assert "SetFlag<HardEvent::V_S>(mte2ToVEvent_)" not in tail
    assert tail.count("SetFlag<HardEvent::V_S>(vToSEvent_)") == 2
    assert tail.count("SetFlag<HardEvent::S_V>(sToVEvent_)") == 2
    assert tail.count("SetFlag<HardEvent::S_MTE2>(sToMte2Event_)") == 2


def test_a5_finalize_manual_full_chunk_paths_use_real_double_l0c():
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    dispatch = finalize.split(
        "__aicore__ inline void ComputeOutputCube(", 1
    )[1].split("using ElementA = T;", 1)[0]
    staged = finalize.split(
        "__aicore__ inline void ComputeOutputCubeStagedArch35", 1
    )[1].split("__aicore__ inline void PrefetchOutputTileArch35", 1)[0]
    prefetched = finalize.split(
        "__aicore__ inline void ComputePrefetchedOutputTileArch35", 1
    )[1].split("__aicore__ inline void DrainOutputInputPipelineEvents", 1)[0]
    initialize_l0c = finalize.split(
        "__aicore__ inline void InitOutputL0CPipelineState", 1
    )[1].split("__aicore__ inline void DrainOutputL0CPipelineState", 1)[0]
    drain_l0c = finalize.split(
        "__aicore__ inline void DrainOutputL0CPipelineState", 1
    )[1].split("__aicore__ inline uint64_t QOffset", 1)[0]
    writeback = finalize.split(
        "__aicore__ inline void FinalizeOutputRows(", 1
    )[1].split("__aicore__ inline bool ResolveFlatChunk(", 1)[0]

    assert "constexpr uint32_t KDA_OUTPUT_L0C_SLOT_DEPTH = 2;" in finalize
    assert "constexpr uint32_t KDA_OUTPUT_L0C_SLOT_BYTES = 64 * 128 * sizeof(float);" in finalize
    assert "TEventID mToFixEvents[KDA_OUTPUT_L0C_SLOT_DEPTH]{};" in finalize
    assert "TEventID fixToMEvents[KDA_OUTPUT_L0C_SLOT_DEPTH]{};" in finalize
    assert "slot < KDA_OUTPUT_L0C_SLOT_DEPTH" in initialize_l0c
    assert "pipe_->AllocEventID<HardEvent::M_FIX>()" in initialize_l0c
    assert "pipe_->AllocEventID<HardEvent::FIX_M>()" in initialize_l0c
    assert initialize_l0c.count(
        "SetFlag<HardEvent::FIX_M>(state.fixToMEvents[slot]);"
    ) == 1
    assert "slot < KDA_OUTPUT_L0C_SLOT_DEPTH" in drain_l0c
    assert "WaitFlag<HardEvent::FIX_M>(state.fixToMEvents[slot]);" in drain_l0c
    assert "pipe_->ReleaseEventID<HardEvent::M_FIX>" in drain_l0c
    assert "pipe_->ReleaseEventID<HardEvent::FIX_M>" in drain_l0c

    assert "BT_ == 64 && K_ == 128 && V_ == 128 && curT == BT_" in dispatch
    assert "l0cState != nullptr" in dispatch
    assert "ComputeOutputCubeStagedArch35" in dispatch
    for manual_path in (staged, prefetched):
        assert manual_path.count("l0cState.nextSlot ^= 1U;") == 2
        assert "qhL0CSlot * KDA_OUTPUT_L0C_SLOT_BYTES" in manual_path
        assert "aqkVL0CSlot * KDA_OUTPUT_L0C_SLOT_BYTES" in manual_path
        assert manual_path.count("true, 0);") == 2
        assert "0b11" not in manual_path
        assert "SetMMLayoutTransform" not in manual_path
        assert "copyL0CToDst(blockO, tileQhL0C);" in manual_path
        assert "copyL0CToDst(blockLocal, tileAqkVL0C);" in manual_path
        assert manual_path.count("WaitFlag<HardEvent::FIX_M>") == 2
        assert manual_path.count("SetFlag<HardEvent::M_FIX>") == 2
        assert manual_path.count("WaitFlag<HardEvent::M_FIX>") == 2
        assert manual_path.count("SetFlag<HardEvent::FIX_M>") == 2
    assert "fusedA5Output" not in writeback
    assert "CopyVectorIn(localLocal, u_" in writeback
    assert "Add(outLocal, stateLocal, localLocal" in writeback


def test_a5_finalize_keeps_l0c_events_at_full_chunk_owner_scope():
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    prefetched = finalize.split(
        "__aicore__ inline void ComputePrefetchedOutputTileArch35", 1
    )[1].split("__aicore__ inline void DrainOutputInputPipelineEvents", 1)[0]
    owned_pipeline = finalize.split(
        "__aicore__ inline void ProcessOwnedFullHeadWindowAicPipelinedArch35", 1
    )[1].split(
        "__aicore__ inline void ProcessOwnedFullChunksAicPipelinedArch35", 1
    )[0]
    owned_full_chunks = finalize.split(
        "__aicore__ inline void ProcessOwnedFullChunksAicPipelinedArch35", 1
    )[1].split("__aicore__ inline void ProcessChunkOutAiv", 1)[0]
    owned_dispatch = finalize.split(
        "__aicore__ inline void ProcessOwnedChunksAic", 1
    )[1].split("__aicore__ inline void ProcessOutAiv", 1)[0]

    assert "Fixpipe<" not in prefetched
    assert "OutputL0CPipelineState &l0cState" in prefetched
    assert "tileIndex + 1 >= 2" in owned_pipeline
    assert owned_pipeline.count("AcquireOutputProducerSlot(producerState);") == 2
    assert owned_pipeline.count("PublishOutputProducerSlot(producerState);") == 1
    last_v_tile = owned_pipeline.split(
        "if (currentNOffset + 128 >= V_) {", 1
    )[1].split("++tileIndex;", 1)[0]
    assert "PublishOutputProducerSlot(producerState);" in last_v_tile
    assert "KdaForward::HeadWindowHeadCount(" in owned_full_chunks
    assert "static_cast<uint32_t>(H_)" in owned_full_chunks
    assert "static_cast<uint32_t>(HV_)" in owned_full_chunks
    assert "head += headCnt" in owned_full_chunks
    assert "head += KDA_HEADS_PER_TASK" not in owned_full_chunks
    assert "ProcessOwnedFullHeadWindowAicPipelinedArch35(" in owned_full_chunks
    assert "l0cState, headCnt);" in owned_full_chunks
    assert "ProcessOwnedFullHeadWindowAicPipelinedArch35<" not in owned_full_chunks
    assert "DrainOutputInputPipelineEvents(tileIndex);" in owned_full_chunks
    assert owned_full_chunks.count("InitOutputL0CPipelineState(l0cState);") == 1
    assert "l0cState, headCnt);" in owned_full_chunks
    assert owned_full_chunks.count("DrainOutputL0CPipelineState(l0cState);") == 1
    assert "DrainOutputProducerState(producerState);" in owned_full_chunks
    assert "ProcessOwnedFullChunksAicPipelinedArch35(plan, cursor);" in owned_dispatch
    assert "OutputL0CPipelineState *l0cStatePtr = nullptr;" in owned_dispatch
    assert owned_dispatch.count("InitOutputL0CPipelineState(l0cState);") == 1
    assert "l0cStatePtr = &l0cState;" in owned_dispatch
    assert "producerState, l0cStatePtr" in owned_dispatch
    assert "if (l0cStatePtr != nullptr)" in owned_dispatch
    assert owned_dispatch.count("DrainOutputL0CPipelineState(l0cState);") == 1
    assert "ProcessOutAicPipelinedArch35" not in finalize


def test_a5_finalize_drains_last_input_slot_events_before_leaving_full_pipeline():
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    drain = finalize.split(
        "__aicore__ inline void DrainOutputInputPipelineEvents", 1
    )[1].split("#endif", 1)[0]
    owned_full_chunks = finalize.split(
        "__aicore__ inline void ProcessOwnedFullChunksAicPipelinedArch35", 1
    )[1].split("__aicore__ inline void ProcessChunkOutAiv", 1)[0]

    assert "if (tileCount > 0)" in drain
    assert "WaitFlag<HardEvent::MTE1_MTE2>(0);" in drain
    assert "if (tileCount > 1)" in drain
    assert "WaitFlag<HardEvent::MTE1_MTE2>(1);" in drain
    assert owned_full_chunks.index(
        "DrainOutputInputPipelineEvents(tileIndex);"
    ) < owned_full_chunks.index("DrainOutputL0CPipelineState(l0cState);")
    assert owned_full_chunks.index(
        "DrainOutputL0CPipelineState(l0cState);"
    ) < owned_full_chunks.index("DrainOutputProducerState(producerState);")


def test_a5_finalize_reuses_two_scratch_slots_with_single_mode2_credit_stream():
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    producer_acquire = finalize.split(
        "__aicore__ inline void AcquireOutputProducerSlot", 1
    )[1].split("__aicore__ inline void PublishOutputProducerSlot", 1)[0]
    producer_publish = finalize.split(
        "__aicore__ inline void PublishOutputProducerSlot", 1
    )[1].split("__aicore__ inline void DrainOutputProducerState", 1)[0]
    producer_drain = finalize.split(
        "__aicore__ inline void DrainOutputProducerState", 1
    )[1].split("__aicore__ inline void AcquireOutputConsumerSlot", 1)[0]
    consumer_acquire = finalize.split(
        "__aicore__ inline void AcquireOutputConsumerSlot", 1
    )[1].split("__aicore__ inline void ReleaseOutputConsumerSlot", 1)[0]
    consumer_release = finalize.split(
        "__aicore__ inline void ReleaseOutputConsumerSlot", 1
    )[1].split("struct FullChunkIterator", 1)[0]
    aiv_task = finalize.split(
        "__attribute__((noinline)) __aicore__ void ProcessChunkOutAiv", 1
    )[1].split(
        "__attribute__((noinline)) __aicore__ void ProcessChunkOutAic", 1
    )[0]
    aic_task = finalize.split(
        "__attribute__((noinline)) __aicore__ void ProcessChunkOutAic", 1
    )[1].split("__aicore__ inline void ProcessChunkOutAivHeadWindow", 1)[0]

    assert "constexpr uint32_t KDA_OUTPUT_SLOT_DEPTH = 2;" in finalize
    assert "constexpr uint8_t KDA_OUTPUT_DONE_FLAG = 2;" in finalize
    assert "constexpr uint8_t KDA_OUTPUT_COMPLETION_FLAG = 4;" in finalize
    for forbidden in (
        "KDA_SCORE_DONE_FLAG0",
        "KDA_SCORE_DONE_FLAG1",
        "KDA_SCORE_READY_FLAG0",
        "KDA_SCORE_READY_FLAG1",
        "KDA_OUTPUT_DONE_FLAG0",
        "KDA_OUTPUT_DONE_FLAG1",
        "KDA_OUTPUT_COMPLETION_FLAG0",
        "KDA_OUTPUT_COMPLETION_FLAG1",
        "outputDoneFlag0_",
        "outputDoneFlag1_",
        "outputCompletionFlag0_",
        "outputCompletionFlag1_",
    ):
        assert forbidden not in finalize

    assert "state.descriptorIndex % KDA_OUTPUT_SLOT_DEPTH" in producer_acquire
    assert "if (state.outstandingCount >= KDA_OUTPUT_SLOT_DEPTH)" in producer_acquire
    assert producer_acquire.index(
        "if (state.outstandingCount >= KDA_OUTPUT_SLOT_DEPTH)"
    ) < producer_acquire.index(
        "WaitOutputCompletion();"
    )
    assert producer_acquire.index("WaitOutputCompletion();") < producer_acquire.index(
        "--state.outstandingCount;"
    )
    assert "SetOutputDone();" in producer_publish
    assert "++state.outstandingCount;" in producer_publish
    assert producer_publish.index("SetOutputDone();") < producer_publish.index(
        "++state.descriptorIndex;"
    )
    assert "while (state.outstandingCount != 0)" in producer_drain
    assert "WaitOutputCompletion();" in producer_drain
    assert "--state.outstandingCount;" in producer_drain
    assert "state.descriptorIndex = 0;" in producer_drain

    assert "state.descriptorIndex % KDA_OUTPUT_SLOT_DEPTH" in consumer_acquire
    assert "WaitOutputDone();" in consumer_acquire
    assert "SetOutputCompletion();" in consumer_release
    assert consumer_release.index("SetOutputCompletion();") < consumer_release.index(
        "++state.descriptorIndex;"
    )

    finalize_rows = (
        "FinalizeOutputRows(b, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);"
    )
    assert aiv_task.count("AcquireOutputConsumerSlot(consumerState)") == 1
    assert aiv_task.count(finalize_rows) == 1
    assert aiv_task.count("ReleaseOutputConsumerSlot(consumerState)") == 1
    assert aiv_task.index("AcquireOutputConsumerSlot(consumerState)") < aiv_task.index(
        finalize_rows
    )
    assert aiv_task.index(finalize_rows) < aiv_task.index(
        "ReleaseOutputConsumerSlot(consumerState)"
    )

    compute_output = "ComputeOutputCube(b, hv, chunkIdx, start, curT, l0cState);"
    assert aic_task.index("AcquireOutputProducerSlot(producerState)") < aic_task.index(
        compute_output
    )
    assert aic_task.index(compute_output) < aic_task.index(
        "PublishOutputProducerSlot(producerState);"
    )
    assert "CrossCoreSetFlag<0x2, PIPE_FIX>(outputDoneFlag_);" in finalize
    assert "CrossCoreSetFlag<0x2, PIPE_MTE3>(outputCompletionFlag_);" in finalize
    assert "CrossCoreFlag outputDoneFlag_{KDA_OUTPUT_DONE_FLAG};" in finalize
    assert "CrossCoreFlag outputCompletionFlag_{KDA_OUTPUT_COMPLETION_FLAG};" in finalize
    assert "WaitOutputCompletion(slot" not in finalize
    assert "WaitOutputDone(slot" not in finalize
    assert "SetOutputDone(slot" not in finalize
    assert "SetOutputCompletion(slot" not in finalize
    assert "CrossCoreFlagWithReverse" not in finalize


def test_finalize_two_slot_protocol_model_drains_each_phase():
    def run_phase(descriptor_count):
        outstanding_count = 0
        issued_slots = []
        reuse_wait_count = 0
        max_outstanding = 0
        for descriptor_index in range(descriptor_count):
            slot = descriptor_index % 2
            if outstanding_count >= 2:
                reuse_wait_count += 1
                outstanding_count -= 1
            issued_slots.append(slot)
            outstanding_count += 1
            max_outstanding = max(max_outstanding, outstanding_count)
        drain_wait_count = outstanding_count
        outstanding_count = 0
        return (
            issued_slots,
            reuse_wait_count,
            drain_wait_count,
            max_outstanding,
            outstanding_count,
        )

    for full_descriptors, tail_descriptors in ((0, 0), (1, 1), (2, 3), (31, 32), (96, 7)):
        full = run_phase(full_descriptors)
        tail = run_phase(tail_descriptors)
        for descriptor_count, phase in (
            (full_descriptors, full),
            (tail_descriptors, tail),
        ):
            issued, reuse_wait_count, drain_wait_count, max_outstanding, final_state = phase
            assert issued == [index % 2 for index in range(descriptor_count)]
            assert reuse_wait_count == max(0, descriptor_count - 2)
            assert drain_wait_count == min(2, descriptor_count)
            assert max_outstanding <= 2
            assert final_state == 0
        if tail_descriptors:
            assert tail[0][0] == 0


def test_generic_finalize_uses_two_slots_with_single_mode2_credit_stream():
    finalize = GENERIC_STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    producer_acquire = finalize.split(
        "__aicore__ inline void AcquireOutputProducerSlot", 1
    )[1].split("__aicore__ inline void PublishOutputProducerSlot", 1)[0]
    producer_publish = finalize.split(
        "__aicore__ inline void PublishOutputProducerSlot", 1
    )[1].split("__aicore__ inline void DrainOutputProducerState", 1)[0]
    producer_drain = finalize.split(
        "__aicore__ inline void DrainOutputProducerState", 1
    )[1].split("__aicore__ inline void AcquireOutputConsumerSlot", 1)[0]
    consumer_acquire = finalize.split(
        "__aicore__ inline void AcquireOutputConsumerSlot", 1
    )[1].split("__aicore__ inline void ReleaseOutputConsumerSlot", 1)[0]
    consumer_release = finalize.split(
        "__aicore__ inline void ReleaseOutputConsumerSlot", 1
    )[1].split("__aicore__ inline void ResetOutputConsumerState", 1)[0]
    aiv_task = finalize.split(
        "__attribute__((noinline)) __aicore__ void ProcessChunkOutAiv", 1
    )[1].split("__attribute__((noinline)) __aicore__ void ProcessChunkOutAic", 1)[0]
    aic_task = finalize.split(
        "__attribute__((noinline)) __aicore__ void ProcessChunkOutAic", 1
    )[1].split("template <bool IS_TAIL>", 1)[0]
    finalize_rows_body = finalize.split(
        "__aicore__ inline void FinalizeOutputRows", 1
    )[1].split("__aicore__ inline bool ResolveFlatChunk", 1)[0]

    assert "constexpr uint32_t KDA_OUTPUT_SLOT_DEPTH = 2;" in finalize
    assert "constexpr uint8_t KDA_OUTPUT_DONE_FLAG = 2;" in finalize
    assert "constexpr uint8_t KDA_OUTPUT_COMPLETION_FLAG = 4;" in finalize
    for forbidden in (
        "KDA_SCORE_DONE_FLAG0",
        "KDA_SCORE_DONE_FLAG1",
        "KDA_SCORE_READY_FLAG0",
        "KDA_SCORE_READY_FLAG1",
        "KDA_OUTPUT_DONE_FLAG0",
        "KDA_OUTPUT_DONE_FLAG1",
        "KDA_OUTPUT_COMPLETION_FLAG0",
        "KDA_OUTPUT_COMPLETION_FLAG1",
        "outputDoneFlag0_",
        "outputDoneFlag1_",
        "outputCompletionFlag0_",
        "outputCompletionFlag1_",
    ):
        assert forbidden not in finalize
    assert "CrossCoreFlagWithReverse" not in finalize
    assert "CrossCoreSetFlagWithReverse" not in finalize
    assert "CrossCoreWaitFlagWithReverse" not in finalize

    assert "state.descriptorIndex % KDA_OUTPUT_SLOT_DEPTH" in producer_acquire
    assert "if (state.outstandingCount >= KDA_OUTPUT_SLOT_DEPTH)" in producer_acquire
    assert producer_acquire.index(
        "if (state.outstandingCount >= KDA_OUTPUT_SLOT_DEPTH)"
    ) < producer_acquire.index(
        "WaitOutputCompletion();"
    )
    assert "--state.outstandingCount;" in producer_acquire
    assert "SetOutputDone();" in producer_publish
    assert "++state.outstandingCount;" in producer_publish
    assert "while (state.outstandingCount != 0)" in producer_drain
    assert "WaitOutputCompletion();" in producer_drain
    assert "--state.outstandingCount;" in producer_drain
    assert "state.descriptorIndex = 0;" in producer_drain
    assert "state.descriptorIndex % KDA_OUTPUT_SLOT_DEPTH" in consumer_acquire
    assert "WaitOutputDone();" in consumer_acquire
    assert "SetOutputCompletion();" in consumer_release
    assert consumer_release.index("SetOutputCompletion();") < consumer_release.index(
        "++state.descriptorIndex;"
    )

    finalize_rows = "FinalizeOutputRows(b, hv, start, curT, subBlockIdx, subBlockNum);"
    assert aiv_task.index("AcquireOutputConsumerSlot(consumerState)") < aiv_task.index(
        finalize_rows
    )
    assert aiv_task.index(finalize_rows) < aiv_task.index(
        "ReleaseOutputConsumerSlot(consumerState)"
    )
    assert finalize_rows_body.index("CopyRowsOut(vNew_") < finalize_rows_body.index(
        "WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);"
    )
    assert finalize_rows_body.index("CopyRowsOut(vNew_") < finalize_rows_body.index(
        "WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);"
    )
    assert aic_task.index("AcquireOutputProducerSlot(producerState)") < aic_task.index(
        "ComputeOutputCube(b, hv, chunkIdx, start, curT);"
    )
    assert aic_task.index("ComputeOutputCube(b, hv, chunkIdx, start, curT);") < aic_task.index(
        "PublishOutputProducerSlot(producerState);"
    )
    assert "CrossCoreSetFlag<0x2, PIPE_FIX>(outputDoneFlag_);" in finalize
    assert "CrossCoreSetFlag<0x2, PIPE_MTE3>(outputCompletionFlag_);" in finalize
    assert "CrossCoreWaitFlag<0x2, PIPE_MTE2>(outputDoneFlag_.id);" in finalize
    assert "CrossCoreWaitFlag<0x2, PIPE_MTE2>(outputCompletionFlag_.id);" in finalize
    assert "CrossCoreFlag outputDoneFlag_{KDA_OUTPUT_DONE_FLAG};" in finalize
    assert "CrossCoreFlag outputCompletionFlag_{KDA_OUTPUT_COMPLETION_FLAG};" in finalize
    assert "WaitOutputCompletion(slot" not in finalize
    assert "WaitOutputDone(slot" not in finalize
    assert "SetOutputDone(slot" not in finalize
    assert "SetOutputCompletion(slot" not in finalize


def test_generic_finalize_uses_runtime_head_windows_and_drains_each_phase():
    finalize = GENERIC_STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    aiv_range = finalize.split(
        "__aicore__ inline void ProcessCompactOutAivHeadRange", 1
    )[1].split("__aicore__ inline void ProcessCompactOutAicHeadWindow", 1)[0]
    aic_range = finalize.split(
        "__aicore__ inline void ProcessCompactOutAicHeadRange", 1
    )[1].split("__aicore__ inline void ProcessG1FullOutAivPhase", 1)[0]
    for dispatcher, stem in (
        (aiv_range, "ProcessCompactOutAivHeadWindow"),
        (aic_range, "ProcessCompactOutAicHeadWindow"),
    ):
        assert "KdaForward::HeadWindowHeadCount(" in dispatcher
        assert "static_cast<uint32_t>(H_)" in dispatcher
        assert "static_cast<uint32_t>(HV_)" in dispatcher
        assert "headBase += headCnt" in dispatcher
        assert "headBase += KDA_HEADS_PER_TASK" not in dispatcher
        assert f"{stem}<IS_TAIL>(" in dispatcher
        assert f"{stem}<IS_TAIL," not in dispatcher

    process_aiv = finalize.split(
        "__aicore__ inline void ProcessOutAiv", 1
    )[1].split("__aicore__ inline void ProcessOutAic", 1)[0]
    process_aic = finalize.split(
        "__aicore__ inline void ProcessOutAic", 1
    )[1].split("private:", 1)[0]
    load_aiv_cursor = "if (!plan.LoadChunkCoreCursor("
    load_aic_cursor = "if (!plan.LoadChunkCoreCursor("
    assert process_aiv.index(load_aiv_cursor) < process_aiv.index(
        "OutputConsumerState fullState{};"
    )
    assert process_aic.index(load_aic_cursor) < process_aic.index(
        "OutputProducerState fullState{};"
    )
    assert process_aiv.count("ResetOutputConsumerState(fullState);") == 2
    assert process_aiv.count("ResetOutputConsumerState(tailState);") == 1
    assert process_aic.count("DrainOutputProducerState(fullState);") == 2
    assert process_aic.count("DrainOutputProducerState(tailState);") == 1
    for process in (process_aiv, process_aic):
        assert "ProcessG1FullOut" in process
        assert "ProcessG1TailOut" in process
        assert "ProcessGroupedFullOut" in process
        assert "ProcessGroupedTailOut" in process
    for stage, state in ((process_aiv, "consumerState"), (process_aic, "producerState")):
        assert "ResolveFlatChunk" in stage
        assert state in stage


def test_generic_finalize_uses_dynamic_per_core_two_slot_scratch():
    finalize = GENERIC_STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    runner = finalize.split(
        "__aicore__ inline void RunChunkKdaOutput(", 1
    )[1]
    assert "2 * KDA_OUTPUT_SLOT_DEPTH * solveCoreIdx_ * outputTileElements_" in finalize
    assert "activeOutputSlot_ * 2 * outputTileElements_" in finalize
    assert "OutputScratchOffset(mOffset, nOffset)" in finalize
    assert "OutputScratchOffset(tileRow, 0)" in finalize
    assert "const uint64_t outputElements = B_ * HV_ * T_ * V_;" not in finalize
    assert "GM_ADDR stateScratch = outputScratch;" in runner
    assert "GM_ADDR localScratch = outputScratch;" in runner
    assert "outputScratch + outputElements" not in runner
    assert "32 * KDA_OUTPUT_SLOT_DEPTH" not in finalize


def test_manifest_registers_a5_bf16_full_chunk_finalize_regression():
    manifest = json.loads(CASE_MANIFEST.read_text(encoding="utf-8"))
    case = next(
        item
        for item in manifest["cases"]
        if item["id"] == "chunk_kda_fwd_a5_bf16_full_chunk_finalize"
    )
    coverage = manifest["coverage_requirements"]

    assert case["id"] in coverage["accuracy_case_ids"]
    assert case["id"] in coverage["generalization_case_ids"]
    assert case["dtype"]["q_k_v"] == "bfloat16"
    assert case["layout"] == "BSND"
    assert case["shape"]["chunk_size"] == 64
    assert case["shape"]["T"] % case["shape"]["chunk_size"] == 0
    assert case["attrs"]["safe_gate"] is True
    assert case["attrs"]["state_v_first"] is True
    assert case["soc"] == ["ascend950"]


def test_fwd_h_uses_fixed_scalar_exp_and_keywise_exp2_on_a2_and_a5():
    fwd_h_root = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
    )
    dispatch = (fwd_h_root / "op_kernel/chunk_gated_delta_rule_fwd_h.cpp").read_text(
        encoding="utf-8"
    )
    assert dispatch.count("TileShapes, false>(") == 2
    assert "tilingData->useExp2" not in dispatch
    for arch in ("", "arch35/"):
        epilogue = fwd_h_root / f"op_kernel/{arch}epilogue/block"
        update = (epilogue / "block_epilogue_gdn_fwdh_update.hpp").read_text(
            encoding="utf-8"
        )
        vnew = (epilogue / "block_epilogue_gdn_fwdh_vnew.hpp").read_text(
            encoding="utf-8"
        )
        assert "LN2" in update and "LN2" in vnew
        assert "AscendC::Exp" in update and "AscendC::Exp" in vnew


def test_a2_fwd_h_keeps_fp32_state_updates():
    fwd_h_kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel"
    )
    update = (
        fwd_h_kernel / "epilogue/block/block_epilogue_gdn_fwdh_update.hpp"
    ).read_text(encoding="utf-8")
    vnew = (
        fwd_h_kernel / "epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
    ).read_text(encoding="utf-8")
    kernel = (fwd_h_kernel / "gemm/kernel/gdn_fwd_h_kernel.hpp").read_text(
        encoding="utf-8"
    )

    assert "bool useFp32StateUpdate" in update
    assert "AscendC::GlobalTensor<FinalStateElement> initialState" in update
    assert "initialState[rowStart * outputStride]" in update
    assert "CopyGmToUb(calcUbTensor, finalStateThisTile" in update
    assert "CopyUbToGm(finalStateThisTile, hUpdateUbTensor" in update
    assert "CopyUbToGm(hOutputThisTile, hUbTensor" in update
    assert "gmInitialState[vec2Offsets.initialStateOffset]" in kernel
    assert "useInitialState, " in kernel
    assert "event0FromMte3[streamId] = true;" in kernel
    empty_subblock = vnew.split("if (rowBegin >= mActual) {", 1)[1].split(
        "return;", 1
    )[0]
    assert "if (waitWsFromMte3)" in empty_subblock
    assert "WaitFlag<AscendC::HardEvent::MTE3_MTE2>" in empty_subblock
    assert "SetFlag<AscendC::HardEvent::V_MTE2>" in empty_subblock
    assert empty_subblock.index("CrossCoreWaitFlag") < empty_subblock.index(
        "if (waitWsFromMte3)"
    )
    assert empty_subblock.index("if (waitWsFromMte3)") < empty_subblock.index(
        "CrossCoreSetFlag<0x2"
    )


def test_a2_fwd_h_tail_waits_for_recurrent_state_before_reading_h():
    fwd_h_kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel"
    )
    kernel = (fwd_h_kernel / "gemm/kernel/gdn_fwd_h_kernel.hpp").read_text(
        encoding="utf-8"
    )
    vnew = (
        fwd_h_kernel / "epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
    ).read_text(encoding="utf-8")

    vec1_dispatch = kernel.split(
        "const GDNFwdHOffsets& vec1Offsets", 1
    )[1].split("if (storeFinalState", 1)[0]
    ready_wait = "Arch::CrossCoreWaitFlag("
    tail_compute = "ComputeTailVWorkspace(vec1Offsets);"
    assert "bool tailVectorPath = vec1Offsets.blockTokens < 16;" in vec1_dispatch
    assert vec1_dispatch.index(ready_wait) < vec1_dispatch.index(tail_compute)
    assert "waitWsFromMte3, (streamId == 0), tailVectorPath" in kernel
    assert "bool cube1AlreadyWaited" in vnew
    assert vnew.count("if (!cube1AlreadyWaited)") == 3


def test_a2_fwd_h_tail_stages_coefficients_before_scalar_vector_use():
    kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")
    tail_v_workspace = kernel.split("ComputeTailVWorkspace", 1)[1].split(
        "ComputeTailHWorkspace", 1
    )[0]
    tail_h_workspace = kernel.split("ComputeTailHWorkspace", 1)[1].split(
        "PresetVectorPipelineEvents", 1
    )[0]

    assert "LoadScalarAsFloat" not in kernel
    assert "gmW.GetValue" not in tail_v_workspace
    assert "gmKDecayWorkspace.GetValue" not in tail_h_workspace
    assert "gmW[offsets.wOffset + tokenRow * kHeadDim]" in tail_v_workspace
    assert (
        "gmKDecayWorkspace[offsets.kDecayWorkOffset + tokenRow * kHeadDim]"
        in tail_h_workspace
    )
    for body in (tail_v_workspace, tail_h_workspace):
        assert "constexpr uint32_t TAIL_EVENT_ID = EVENT_ID3;" in body
        assert "EVENT_ID7" not in body
        assert "TAIL_WEIGHT_INPUT_OFFSET = 169 * 1024" in body
        assert "TAIL_WEIGHT_FLOAT_OFFSET = 170 * 1024" in body
        assert "weightFloatUb.GetValue" in body
        assert "WaitFlag<AscendC::HardEvent::V_MTE2>(TAIL_EVENT_ID)" in body
        assert "SetFlag<AscendC::HardEvent::V_MTE2>(TAIL_EVENT_ID)" in body
        assert "SetFlag<AscendC::HardEvent::MTE2_V>(TAIL_EVENT_ID)" in body
        assert "WaitFlag<AscendC::HardEvent::MTE2_V>(TAIL_EVENT_ID)" in body
        assert "SetFlag<AscendC::HardEvent::V_S>(TAIL_EVENT_ID)" in body
        assert "WaitFlag<AscendC::HardEvent::V_S>(TAIL_EVENT_ID)" in body
        assert "SetFlag<AscendC::HardEvent::S_V>(TAIL_EVENT_ID)" in body
        assert "WaitFlag<AscendC::HardEvent::S_V>(TAIL_EVENT_ID)" in body
        assert "SetFlag<AscendC::HardEvent::S_MTE2>(TAIL_EVENT_ID)" in body
        assert "WaitFlag<AscendC::HardEvent::S_MTE2>(TAIL_EVENT_ID)" in body
        normalized_body = " ".join(body.split())
        weight_copy = normalized_body.index("AscendC::DataCopy( weightInputUb,")
        ordered_markers = (
            "SetFlag<AscendC::HardEvent::MTE2_V>(TAIL_EVENT_ID)",
            "AscendC::Cast( weightFloatUb",
            "SetFlag<AscendC::HardEvent::V_S>(TAIL_EVENT_ID)",
            "float weight = weightFloatUb.GetValue",
            "SetFlag<AscendC::HardEvent::S_V>(TAIL_EVENT_ID)",
            "AscendC::Muls(",
            "SetFlag<AscendC::HardEvent::S_MTE2>(TAIL_EVENT_ID)",
        )
        positions = [weight_copy]
        cursor = weight_copy
        for marker in ordered_markers:
            cursor = normalized_body.index(marker, cursor)
            positions.append(cursor)
        assert positions == sorted(positions)
        assert normalized_body.index(
            "WaitFlag<AscendC::HardEvent::V_MTE2>(TAIL_EVENT_ID)"
        ) < weight_copy
        assert normalized_body.rindex(
            "SetFlag<AscendC::HardEvent::V_MTE2>(TAIL_EVENT_ID)"
        ) > normalized_body.rindex(
            "WaitFlag<AscendC::HardEvent::MTE3_V>(TAIL_EVENT_ID)"
        )


def test_a2_fwd_h_contiguous_update_respects_calc_ub_capacity():
    update = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/epilogue/block/block_epilogue_gdn_fwdh_update.hpp"
    ).read_text(encoding="utf-8")

    assert "CONTIGUOUS_CALC_BUF_BYTES = 32 * 1024" in update
    assert (
        "CONTIGUOUS_CALC_MAX_ELEMENTS =\n"
        "        CONTIGUOUS_CALC_BUF_BYTES / sizeof(float)" in update
    )
    assert "PING_BUF_0_OFFSET = CONTIGUOUS_CALC_BUF_BYTES" in update
    fast_guard = " ".join(
        update.split("uint32_t mActualThisSubBlock = rowEnd - rowBegin;", 1)[1]
        .split("AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock", 1)[0]
        .split()
    )
    assert update.count("uint32_t mActualThisSubBlock = rowEnd - rowBegin;") == 1
    assert (
        "uint64_t contiguousElementCountWide = "
        "static_cast<uint64_t>(mActualThisSubBlock) * nActual;" in fast_guard
    )
    assert (
        "if (nActual <= 128 && nActual == outputStride && "
        "contiguousElementCountWide <= CONTIGUOUS_CALC_MAX_ELEMENTS)"
        in fast_guard
    )
    assert (
        "uint32_t contiguousElementCount = "
        "static_cast<uint32_t>(contiguousElementCountWide);" in fast_guard
    )

    max_elements = 32 * 1024 // 4

    def uses_contiguous_path(k_head_dim, v_block_dim):
        rows_per_aiv = (k_head_dim + 1) // 2
        return v_block_dim <= 128 and rows_per_aiv * v_block_dim <= max_elements

    assert uses_contiguous_path(128, 128)
    assert not uses_contiguous_path(144, 128)
    assert not uses_contiguous_path(256, 128)
    assert uses_contiguous_path(256, 64)


def test_a5_fwd_h_uses_canonical_h_state_update_and_fp32_final_state():
    fwd_h_kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/arch35"
    )
    update = (
        fwd_h_kernel / "epilogue/block/block_epilogue_gdn_fwdh_update.hpp"
    ).read_text(encoding="utf-8")
    vnew = (
        fwd_h_kernel / "epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
    ).read_text(encoding="utf-8")
    kernel = (fwd_h_kernel / "gemm/kernel/gdn_fwd_h_kernel.hpp").read_text(
        encoding="utf-8"
    )

    assert "CopyGmToUb(hUbTensor, hInputThisTile" in update
    assert "AscendC::Cast(calcUbTensor, hUbTensor" in update
    assert "bool useFp32StateUpdate" in update
    assert "initialState[rowStart * outputStride]" in update
    assert "CopyGmToUb(calcUbTensor, finalStateThisTile" in update
    assert "ApplyRowScale(calcUbTensor, gkLastUbTensor" in update
    assert "AscendC::Add<float>(" in update
    assert "hUpdateUbTensorThisTile, calcUbTensor, hUpdateUbTensorThisTile" in update
    assert "CopyUbToGm(finalStateThisTile, hUpdateUbTensor" in update
    assert "CopyUbToGm(hOutputThisTile, hUbTensor" in update
    assert "uint32_t updateReadyEvent = EVENT_ID3 + pingpongFlag;" in update
    assert "WaitFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent)" in update
    assert "gmInitialState[vec2Offsets.initialStateOffset]" in kernel
    assert "event0FromMte3[streamId] = true;" in kernel
    empty_subblock = vnew.split("if (rowBegin >= mActual) {", 1)[1].split(
        "return;", 1
    )[0]
    assert "if (waitWsFromMte3)" in empty_subblock
    assert "WaitFlag<AscendC::HardEvent::MTE3_MTE2>" in empty_subblock
    assert "SetFlag<AscendC::HardEvent::V_MTE2>" in empty_subblock
    assert empty_subblock.index("CrossCoreWaitFlag") < empty_subblock.index(
        "if (waitWsFromMte3)"
    )
    assert empty_subblock.index("if (waitWsFromMte3)") < empty_subblock.index(
        "CrossCoreSetFlag<0x2"
    )


def test_a5_fwd_h_routes_sub_16_token_tail_away_from_cube_mmad():
    fwd_h_kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/arch35"
    )
    kernel = (fwd_h_kernel / "gemm/kernel/gdn_fwd_h_kernel.hpp").read_text(
        encoding="utf-8"
    )
    update = (
        fwd_h_kernel / "epilogue/block/block_epilogue_gdn_fwdh_update.hpp"
    ).read_text(encoding="utf-8")
    vnew = (
        fwd_h_kernel / "epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
    ).read_text(encoding="utf-8")

    assert kernel.count("blockTokens < 16") >= 6
    assert "ComputeTailVWorkspace(" in kernel
    assert "ComputeTailHWorkspace(" in kernel
    assert "vec1Offsets, EVENT_ID3 + (i == 0 ? 0 : pongBaseEvent)" in kernel
    assert "vec2Offsets, EVENT_ID3 + (i == 0 ? 0 : pongBaseEvent)" in kernel
    assert "cubeBlockScheduler.cube1Done[streamId]" in kernel
    assert "cubeBlockScheduler.cube2Done[streamId]" in kernel
    assert "bool useDirectForTask = useDirectFp32Ub && !tailVectorPath;" in kernel
    assert "bool cube1AlreadyWaited" in vnew
    assert "else if (!cube1AlreadyWaited)" in vnew
    assert "bool cube2AlreadyWaited" in update
    assert "else if (!cube2AlreadyWaited)" in update


def test_a5_fwd_h_uses_regular_cube_for_full_chunks_and_bounded_cube_for_tails():
    kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")

    assert "MmadPingpongTlaMulti<ArchTag, true, false, 1>" in kernel
    assert "BlockMmadWHTail" in kernel
    assert "BlockMmadKVTail" in kernel
    assert kernel.count("EmptyClass{}, true") == 1
    kv_tail = kernel.split("blockMmadKVTail.preSetFlags();", 1)[1].split(
        "blockMmadKVTail.finalWaitFlags();", 1
    )[0]
    assert "ComputeCube2RowTiles(" in kv_tail
    assert (
        "cube2Offsets.vBlockDim, cube2Offsets.blockTokens, true);"
        in " ".join(kv_tail.split())
    )
    assert "bool useChunkAwareMmad = isVariedLen || (seqlen % chunkSize != 0);" in kernel
    assert "cube1Offsets.blockTokens != chunkSize" in kernel
    assert "cube2Offsets.blockTokens != chunkSize" in kernel
    assert "blockMmadWH(" in kernel
    assert "blockMmadKV(" in kernel
    assert "blockMmadWHTail(" in kernel
    assert "blockMmadKVTail(" in kernel
    assert kernel.count("seqlen % chunkSize == 0") == 2


def test_a5_direct_ub_fwd_h_requires_enough_dense_tasks_for_started_cores():
    kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")

    assert kernel.count(
        "denseTaskCount >= AscendC::GetBlockNum()"
    ) == 2


def test_a5_generic_fwd_h_synchronizes_all_cores_before_stage_handshake():
    kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")
    process_prefix = kernel.split("__aicore__ inline void Process()", 1)[1].split(
        "if ASCEND_IS_AIC", 1
    )[0]

    assert "AscendC::SyncAll<false>();" in process_prefix
    assert "if (isVariedLen)" not in process_prefix


def test_a5_finalize_uses_fp32_vector_accumulation_for_sub_16_reduction():
    finalize = (
        OP_ROOT / "op_kernel/arch35/chunk_kda_fwd_finalize.h"
    ).read_text(encoding="utf-8")

    assert "constexpr uint32_t KDA_CUBE_MIN_REDUCTION = 16;" in finalize
    assert "ComputeTailLocalRows" in finalize
    assert "ComputeTailStateRows" in finalize
    assert "preparedQG_" in finalize
    assert "propagatedH_" in finalize
    assert "coefficientTyped" in finalize
    assert "coefficients.GetValue(j)" in finalize
    assert "LoadScalarAsFloat" not in finalize
    assert "HardEvent::V_S" in finalize
    assert "HardEvent::S_V" in finalize
    assert finalize.count("curT < KDA_CUBE_MIN_REDUCTION") >= 2


def _assert_tail_output_reuse_fence(body: str, event_id: str) -> None:
    copy_out = body.index("AscendC::DataCopy(\n                gm")
    markers = (
        f"SetFlag<AscendC::HardEvent::MTE3_MTE2>({event_id})",
        f"WaitFlag<AscendC::HardEvent::MTE3_MTE2>({event_id})",
        f"SetFlag<AscendC::HardEvent::MTE3_V>({event_id})",
        f"WaitFlag<AscendC::HardEvent::MTE3_V>({event_id})",
    )
    assert all(body.count(marker) == 1 for marker in markers)
    positions = [body.index(marker, copy_out) for marker in markers]
    assert [copy_out, *positions] == sorted([copy_out, *positions])
    after_mte3_to_v = body[positions[-1] + len(markers[-1]) :]
    assert "AscendC::Duplicate(" not in after_mte3_to_v
    assert "AscendC::Add(" not in after_mte3_to_v


def test_a5_fwd_h_tail_stages_w_coefficients_in_ub_before_scalar_read():
    kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")

    assert "TAIL_WEIGHT_INPUT_OFFSET" in kernel
    assert "TAIL_WEIGHT_FLOAT_OFFSET" in kernel
    assert "weightFloatUb.GetValue(kIdx)" in kernel
    tail_v_workspace = kernel.split("ComputeTailVWorkspace", 1)[1].split(
        "ComputeTailHWorkspace", 1
    )[0]
    tail_h_workspace = kernel.split("ComputeTailHWorkspace", 1)[1].split(
        "__aicore__ inline void Process()", 1
    )[0]
    assert "AscendC::ResetMask();" in tail_v_workspace
    assert "AscendC::ResetMask();" in tail_h_workspace
    assert "LoadScalarAsFloat" not in tail_v_workspace
    assert "weightFloatUb.GetValue(kRow)" in tail_h_workspace
    assert "LoadScalarAsFloat" not in tail_h_workspace
    assert "HardEvent::V_S" in kernel
    assert "HardEvent::S_V" in kernel
    assert "HardEvent::V_MTE2" in tail_v_workspace
    assert "HardEvent::V_MTE2" in tail_h_workspace
    assert "WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId)" in tail_v_workspace
    assert "SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId)" in tail_v_workspace
    assert "WaitFlag<AscendC::HardEvent::V_MTE2>(tailEventId)" in tail_h_workspace
    assert "SetFlag<AscendC::HardEvent::V_MTE2>(tailEventId)" in tail_h_workspace
    assert "HardEvent::MTE3_MTE2>(tailEventId)" in tail_v_workspace
    assert "HardEvent::MTE3_MTE2>(tailEventId)" in tail_h_workspace
    assert "SetFlag<AscendC::HardEvent::MTE3_V>(tailEventId)" in tail_v_workspace
    assert "WaitFlag<AscendC::HardEvent::MTE3_V>(tailEventId)" in tail_v_workspace
    assert "SetFlag<AscendC::HardEvent::MTE3_V>(tailEventId)" in tail_h_workspace
    assert "WaitFlag<AscendC::HardEvent::MTE3_V>(tailEventId)" in tail_h_workspace
    _assert_tail_output_reuse_fence(tail_v_workspace, "tailEventId")
    _assert_tail_output_reuse_fence(tail_h_workspace, "tailEventId")
    assert "EVENT_ID3 + (i == 0 ? 0 : pongBaseEvent)" in kernel


def test_generic_fwd_h_tail_waits_for_mte3_before_reusing_accumulator():
    kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")
    tail_v_workspace = kernel.split("ComputeTailVWorkspace", 1)[1].split(
        "ComputeTailHWorkspace", 1
    )[0]
    tail_h_workspace = kernel.split("ComputeTailHWorkspace", 1)[1].split(
        "PresetVectorPipelineEvents", 1
    )[0]

    for body in (tail_v_workspace, tail_h_workspace):
        _assert_tail_output_reuse_fence(body, "TAIL_EVENT_ID")


def test_kda_keeps_fp32_state_update_when_final_state_is_not_returned():
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    l0 = (OP_ROOT / "op_host/op_api/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    tiling = TILING_ENTRY.read_text(encoding="utf-8")
    kernel = KERNEL_COMMON.read_text(encoding="utf-8")
    assert "const bool outputFinalState = params.finalStateOut != nullptr;" in aclnn
    assert "outputFinalState ? stateShape4 : placeholderShape" in aclnn
    assert "MakeShape({info.seqNum, info.hvNum, info.kDim, info.vDim})" in aclnn
    assert "qHead, kHead, vHead, gHead, betaHead" in aclnn
    assert "params.lowerBound, attnCompute, finalStateCompute, gkCompute" in aclnn
    assert "static_cast<uint64_t>(seqNum)" in tiling
    assert "static_cast<uint64_t>(shape.vHeads)" in tiling
    assert "static_cast<uint64_t>(shape.kDim)" in tiling
    assert "static_cast<uint64_t>(shape.vDim), sizeof(float)" in tiling
    assert "stateBytes" in tiling
    assert "allocateHidden(storeFinalState, stateBytes" in tiling
    assert "tiling.finalStateStorageOffset," in kernel
    assert "tiling.storeFinalState" in kernel
    assert "OP_OUTPUT(attnOut, finalStateOut, gkOut, aqkOut" in l0


def test_aclnn_l2_optional_outputs_are_publicly_pointer_driven():
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    header = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.h").read_text(
        encoding="utf-8"
    )
    op_def = (OP_ROOT / "op_host/chunk_kda_fwd_def.cpp").read_text(encoding="utf-8")

    for policy in (
        "outputFinalState",
        "disableRecompute",
        "returnIntermediateStates",
    ):
        assert policy not in header
    for policy in (
        '"output_final_state"',
        '"disable_recompute"',
        '"return_intermediate_states"',
    ):
        assert policy not in op_def
    assert "const bool outputFinalState = params.finalStateOut != nullptr;" in aclnn
    assert "params.useGateInKernel || params.gkOut != nullptr" not in aclnn
    assert "const aclTensor *gkCompute = params.gkOut;" in aclnn
    assert "if (gkCompute == nullptr)" in aclnn
    assert "AllocTensor(executorPtr, placeholderShape, DataType::DT_FLOAT)" in aclnn
    for compute_name in ("wCompute", "uCompute", "qgCompute", "kgCompute", "vNewCompute"):
        assert f"const aclTensor *{compute_name}" in aclnn
    for export_name in ("wExport", "uExport", "qgExport", "kgExport", "vNewExport"):
        assert export_name in aclnn
    tiling = TILING_ENTRY.read_text(encoding="utf-8")
    for store_name in ("storeGk", "storeW", "storeU", "storeQG", "storeKg", "storeVNew", "storeH"):
        assert f"const bool {store_name} = HasOutput" in tiling
    assert "if (hExport != nullptr)" in aclnn
    assert "const aclTensor *hResult = Transpose(result[10], hPerm, executorPtr);" in aclnn
    assert "l0op::ViewCopy(hResult, hExport" in aclnn


def test_state_v_first_defaults_false_and_legacy_returns_are_optional():
    runtime = (
        ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"
    ).read_text(encoding="utf-8")
    schema = (ROOT / "torch_custom/fla_npu/npu_custom.yaml").read_text(
        encoding="utf-8"
    )
    op_def = (OP_ROOT / "op_host/chunk_kda_fwd_def.cpp").read_text(
        encoding="utf-8"
    )
    assert "state_v_first=False" in runtime
    assert "bool? state_v_first=False" in schema
    assert (
        "-> (Tensor, Tensor?, Tensor?, Tensor, Tensor, Tensor?, Tensor?, Tensor?, "
        "Tensor?, Tensor?, Tensor?, Tensor?)"
    ) in schema
    assert 'Attr("state_v_first").AttrType(OPTIONAL).Bool(false)' in op_def


def test_fwd_h_dispatches_optional_gate_and_state_dtype_without_full_runtime_expansion():
    dispatch = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/chunk_gated_delta_rule_fwd_h.cpp"
    ).read_text(encoding="utf-8")
    assert "ChunkGatedDeltaRuleFwdHDispatchGate<DTYPE_K, float, TileShapes, false>" in dispatch
    assert "ChunkGatedDeltaRuleFwdHDispatchGate<DTYPE_K, DTYPE_K, TileShapes, false>" in dispatch
    assert "ChunkGatedDeltaRuleFwdHLaunchTyped<DataT, float, StateT" in dispatch
    assert "ChunkGatedDeltaRuleFwdHLaunchTyped<DataT, bfloat16_t, StateT" in dispatch
    assert "ChunkGatedDeltaRuleFwdHLaunchTyped<DataT, half, StateT" in dispatch
    assert "tilingData->gDataType" in dispatch
    assert "tilingData->stateDataType" in dispatch
    assert "DTYPE_GK" not in dispatch
    assert "->dataType" not in dispatch


def test_a5_fwd_h_kda_hot_path_uses_fused_dual_issue_regbase():
    block_root = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/arch35/epilogue/block"
    )
    regbase = (block_root / "block_epilogue_gdn_fwdh_regbase.hpp").read_text(
        encoding="utf-8"
    )
    assert "__simd_vf__" in regbase
    assert "RegTensor<float> matrixReg0" in regbase
    assert "RegTensor<float> matrixReg1" in regbase
    assert "LoadAlign" in regbase
    assert "StoreAlign" in regbase
    assert "MaskReg mask0" in regbase and "MaskReg mask1" in regbase
    assert "row + 1" in regbase
    assert "ComputeVNewRegbaseDualIssue" in regbase
    assert "PrepareKGateRegbase" in regbase

    update = (block_root / "block_epilogue_gdn_fwdh_update.hpp").read_text(
        encoding="utf-8"
    )
    vnew = (block_root / "block_epilogue_gdn_fwdh_vnew.hpp").read_text(
        encoding="utf-8"
    )
    assert "VF_CALL<detail::PrepareKGateRegbase<GElementInput, true>>" in update
    assert "VF_CALL<detail::ApplyRowScaleDualIssue>" in update
    assert "PrepareKGate(gkLastUbTensor, gkInputUbTensor" in update
    assert "VF_CALL<detail::ComputeVNewRegbaseDualIssue" in vnew
    assert "VF_CALL<detail::ApplyRowScaleDualIssue>" in vnew
    assert "AscendC::LocalTensor<float> decayInput = scalarGated ?" in vnew


def test_aqk_akk_score_path_avoids_global_pipe_barrier():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    score_block = prepare.split(
        "__aicore__ inline void ComputeRawAqkAkkCubeBlock(uint64_t b", 1
    )[1].split("__aicore__ inline bool UseAkkCubeSolve", 1)[0]
    assert "PipeBarrier<PIPE_ALL>()" not in score_block


def test_a5_prepare_joins_both_aiv_subcores_before_shared_ready_signal():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    join = prepare.split(
        "__aicore__ inline void JoinAivMte3()", 1
    )[1].split("__aicore__ inline void RunAicAfterBothAivReady", 1)[0]
    assert "CrossCoreBarrier<0x1, PIPE_MTE3>();" in join
    assert "PipeBarrier<PIPE_MTE3>();" in join
    for forbidden in (
        "PAIR_HEADS",
        "HeadPair",
        "PairAligned",
        "activeHeadPairMode_",
    ):
        assert forbidden not in prepare
    assert "ProcessPreAivHeadWindows();" in prepare
    run_after_join = prepare.split(
        "__aicore__ inline void RunAicAfterBothAivReady", 1
    )[1].split("__aicore__ inline void SignalAicSolveReady", 1)[0]
    signal_solve = prepare.split(
        "__aicore__ inline void SignalAicSolveReady", 1
    )[1].split("__aicore__ inline void WaitAicSolveDone", 1)[0]
    score_loop = prepare.split(
        "__attribute__((noinline)) __aicore__ void ProcessChunkPreAivFp32", 1
    )[1].split("Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);", 1)[0]
    assert run_after_join.index("JoinAivMte3();") < run_after_join.index(
        "CrossCoreSetFlag<0x2, PIPE_MTE3>(syncReadyFlag_);"
    )
    assert run_after_join.index("JoinAivMte3();") < run_after_join.index(
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(mchSyncReadyFlag_);"
    )
    assert signal_solve.index("JoinAivMte3();") < signal_solve.index(
        "CrossCoreSetFlag<0x2, PIPE_MTE3>(syncReadyFlag_);"
    )
    assert score_loop.rindex("JoinAivMte3();") < score_loop.rindex(
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);"
    )


def test_a5_fused_gate_runtime_protocol_has_one_gm_writer_per_head():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    materialize = prepare.split(
        "template <bool WRITE_GATE_TO_GM>", 1
    )[1].split("__aicore__ inline LocalTensor<T> GateDirectQ", 1)[0]
    dispatch = prepare.split(
        "__attribute__((noinline)) __aicore__ void ProcessChunkPreAivFp32", 1
    )[1].split("bool usePostWuCube", 1)[0]

    assert materialize.count("if constexpr (WRITE_GATE_TO_GM)") == 2
    assert materialize.index("AccumulateRawSafeGateChunk128Regbase") < materialize.index(
        "if constexpr (WRITE_GATE_TO_GM)"
    )
    assert materialize.count("SetFlag<HardEvent::V_MTE3>(vToMte3Event_);") == 1
    assert materialize.count("WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);") == 1
    assert materialize.count("SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);") == 1
    assert materialize.count("WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);") == 1
    assert materialize.count("CopyVectorOut(gk_") == 1
    assert "if (subBlockIdx == 0)" in dispatch
    assert "MaterializeRawGateChunkArch35<true>" in dispatch
    assert "MaterializeRawGateChunkArch35<false>" in dispatch

    def gate_writers(sub_block_count=2):
        return [
            sub_block_idx
            for sub_block_idx in range(sub_block_count)
            if sub_block_idx == 0
        ]

    assert gate_writers() == [0]


def test_a5_prepare_and_finalize_use_unified_runtime_head_ranges():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")

    for forbidden in (
        "PAIR_HEADS",
        "HeadPair",
        "PairAligned",
        "activeHeadPairMode_",
    ):
        assert forbidden not in prepare

    prepare_ranges = (
        (
            "ProcessOwnedChunkAivHeads",
            "FlushDeferredSingleHead",
            "ProcessOwnedChunkAivHeadWindow",
        ),
        (
            "ProcessOwnedChunkAicHeads",
            "ProcessOwnedChunkAicHeadWindowTask",
            "ProcessOwnedChunkAicHeadWindow",
        ),
        (
            "ProcessOwnedChunkAicHeadsFusedTail",
            "FlushFusedFullPostWuBatch",
            "ProcessOwnedChunkAicHeadWindowFusedTail",
        ),
        (
            "ProcessOwnedChunkAicHeadsFusedFull",
            "DrainFusedFullHeadWindowState",
            "ProcessOwnedChunkAicHeadWindowFusedFull",
        ),
    )
    for range_name, next_name, window_name in prepare_ranges:
        dispatcher = prepare.split(
            f"__aicore__ inline void {range_name}", 1
        )[1].split(f"__aicore__ inline void {next_name}", 1)[0]
        assert "KdaForward::HeadWindowHeadCount(" in dispatcher
        assert "static_cast<uint32_t>(H_)" in dispatcher
        assert "static_cast<uint32_t>(HV_)" in dispatcher
        assert "head += headCnt" in dispatcher
        assert "head += KDA_HEADS_PER_TASK" not in dispatcher
        assert f"{window_name}(" in dispatcher

    prepare_aiv = prepare.split(
        "__aicore__ inline void ProcessPreAivHeadWindows", 1
    )[1].split("__aicore__ inline void ProcessPreAicHeadWindows", 1)[0]
    prepare_aic = prepare.split(
        "__aicore__ inline void ProcessPreAicHeadWindows", 1
    )[1].split("template <typename PostWuOp>", 1)[0]
    for stage in (prepare_aiv, prepare_aic):
        assert "if (plan.HeadGroupCount() == 1)" in stage
        assert "LoadOwnedFullChunk" in stage
        assert "LoadOwnedTailChunk" in stage
        assert "LoadGroupedFullTask" in stage
        assert "LoadGroupedTailTask" in stage
        assert "fullState" in stage and "tailState" in stage

    fused = prepare.split(
        "__aicore__ inline void ProcessPreAicHeadWindowsFused", 1
    )[1].split("__aicore__ inline void ProcessPreAiv", 1)[0]
    assert "if (plan.HeadGroupCount() == 1)" in fused
    assert "DrainFusedFullHeadWindowState" in fused
    assert "DrainFusedTailHeadWindowState" in fused

    aiv_finalize = finalize.split(
        "__aicore__ inline void ProcessChunkOutAivHeads", 1
    )[1].split("__aicore__ inline void ProcessChunkOutAicHeadWindow", 1)[0]
    aic_finalize = finalize.split(
        "__aicore__ inline void ProcessChunkOutAicHeads", 1
    )[1].split("template <bool IS_TAIL>", 1)[0]
    for dispatcher, stem in (
        (aiv_finalize, "ProcessChunkOutAivHeadWindow"),
        (aic_finalize, "ProcessChunkOutAicHeadWindow"),
    ):
        assert "KdaForward::HeadWindowHeadCount(" in dispatcher
        assert "static_cast<uint32_t>(H_)" in dispatcher
        assert "static_cast<uint32_t>(HV_)" in dispatcher
        assert "head += headCnt" in dispatcher
        assert "head += KDA_HEADS_PER_TASK" not in dispatcher
        assert f"{stem}(" in dispatcher
        assert f"{stem}<" not in dispatcher
        assert "headCnt" in dispatcher

    pipelined = finalize.split(
        "__aicore__ inline void ProcessOwnedFullChunksAicPipelinedArch35", 1
    )[1].split("__aicore__ inline void ProcessChunkOutAiv", 1)[0]
    assert "KdaForward::HeadWindowHeadCount(" in pipelined
    assert "static_cast<uint32_t>(H_)" in pipelined
    assert "static_cast<uint32_t>(HV_)" in pipelined
    assert "head += headCnt" in pipelined
    assert "head += KDA_HEADS_PER_TASK" not in pipelined
    assert "ProcessOwnedFullHeadWindowAicPipelinedArch35(" in pipelined
    assert "ProcessOwnedFullHeadWindowAicPipelinedArch35<" not in pipelined
    owned_aiv = finalize.split(
        "__aicore__ inline void ProcessOwnedChunksAiv", 1
    )[1].split("__aicore__ inline void ProcessOwnedChunksAic", 1)[0]
    owned_aic = finalize.split(
        "__aicore__ inline void ProcessOwnedChunksAic", 1
    )[1].split("__aicore__ inline void ProcessOutAiv", 1)[0]
    for stage, process_heads in (
        (owned_aiv, "ProcessChunkOutAivHeads"),
        (owned_aic, "ProcessChunkOutAicHeads"),
    ):
        assert "LoadOwnedFullChunk" in stage
        assert "LoadGroupedFullTask" in stage
        assert "LoadOwnedTailChunk" in stage
        assert "LoadGroupedTailTask" in stage
        assert stage.count(f"{process_heads}(") == 4
    process_aiv = finalize.split(
        "__aicore__ inline void ProcessOutAiv", 1
    )[1].split("__aicore__ inline void ProcessOutAic", 1)[0]
    process_aic = finalize.split(
        "__aicore__ inline void ProcessOutAic", 1
    )[1].split("private:", 1)[0]
    assert process_aiv.index("ProcessOwnedChunksAiv<false>") < process_aiv.index(
        "ProcessOwnedChunksAiv<true>"
    )
    assert process_aic.index("ProcessOwnedChunksAic<false>") < process_aic.index(
        "ProcessOwnedChunksAic<true>"
    )
    assert "for (uint64_t hv = headBegin; hv < headEnd; ++hv)" not in finalize


def test_a5_prepare_exports_solved_akk_without_redundant_fp32_round_trip():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    finalize = prepare.split(
        "__aicore__ inline void FinalizePrepareIntermediates", 1
    )[1].split("__aicore__ inline bool ResolveFlatChunk", 1)[0]
    finish_deferred = prepare.split(
        "__aicore__ inline void FinishDeferredSafeChunk", 1
    )[1].split("__attribute__((noinline)) __aicore__ void ProcessChunkPreAic", 1)[0]
    assert "uint64_t chunkIdx" in finalize
    assert "SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)" in finalize
    assert "CopyVectorIn(akkLocal, solveWorkspace_, xBase, matrixElems);" in finalize
    assert "COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128" in finalize
    assert "if constexpr (!(SAFE_GATE && COMPILE_BT == 64" in finish_deferred


def test_a5_prepare_exports_masked_aqk_before_finalize_round_trip():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    solve_rows = prepare.split(
        "__aicore__ inline void PrepareAqkAkkSolveInputRows", 1
    )[1].split("__aicore__ inline void CubeGemmSolveSub", 1)[0]
    finalize = prepare.split(
        "__aicore__ inline void FinalizePrepareIntermediates", 1
    )[1].split("__aicore__ inline bool ResolveFlatChunk", 1)[0]
    assert "LocalTensor<T> aqkTyped = GateQTyped(0);" in solve_rows
    assert "Muls(aqkMat, aqkMat, scale_" in solve_rows
    assert "CopyVectorOut(o_, AOffset(b, hv, token, 0), aqkTyped" in solve_rows
    assert "return;" in solve_rows
    assert "if constexpr (!(SAFE_GATE && COMPILE_BT == 64" in finalize


def test_a5_prepare_fuses_beta_w_and_v_into_score_factor_staging():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    regbase = prepare.split(
        "static __simd_vf__ inline void PrepareKdaGateQwKgRegbase", 1
    )[1].split("static __simd_vf__ inline void ForwardSubDiag16Regbase", 1)[0]
    score_factors = prepare.split(
        "__aicore__ inline void PrepareScoreFactorsBulk", 1
    )[1].split("__aicore__ inline void PrepareGateProductsBulk", 1)[0]
    finish_deferred = prepare.split(
        "__aicore__ inline void FinishDeferredSafeChunk", 1
    )[1].split("__aicore__ inline void FinishDeferredSafeChunkPair", 1)[0]
    assert "return 2;" in prepare.split(
        "__aicore__ inline constexpr uint64_t GateBufferDepth", 1
    )[1].split("__aicore__ inline uint64_t GateInputSlotBytes", 1)[0]
    assert "CastHalf2Float<InputT>" in regbase
    assert "LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);" in regbase
    assert "vDirect + offset" in regbase
    assert "CopyVectorOut(vNew_" in score_factors
    assert "pairHeads" not in finish_deferred
    assert "PrepareWuCubeInputs(" in finish_deferred
    assert "uint64_t subBlockNum)" in finish_deferred
    assert prepare.count("FinishDeferredSafeChunk(") > 1


def test_fwd_h_gk_only_path_skips_scalar_gate_scaling():
    block_root = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel"
    )
    for arch in ("", "arch35/"):
        text = (
            block_root
            / f"{arch}epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
        ).read_text(encoding="utf-8")
        assert text.count("if constexpr (scalarGated)") >= 4
        assert "WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag)" in text
        if arch:
            assert text.count("ApplyRowScale(calcUbTensor, gUbTensor") == 2
            assert text.count("ComputeVNew(wsUbTensor") == 2
        else:
            assert text.count("Adds<float>(calcUbTensor, wsUbTensor, 0.0f") == 2


def test_arch22_fwd_h_direct_init_does_not_use_a5_local_buffers():
    arch22 = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")
    for a5_only_buffer in (
        "ubHUpdatePing",
        "ubHUpdatePong",
        "ubVWorkPing",
        "ubVWorkPong",
        "l1VUpdatePing",
        "l1VUpdatePong",
    ):
        assert a5_only_buffer not in arch22


def test_output_layout_conversion_stays_in_kernel_copy_out():
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    runtime = (
        ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"
    ).read_text(encoding="utf-8")
    generalization = (
        ROOT / "tests/operators/_shared/npu_generalization.py"
    ).read_text(encoding="utf-8")
    assert "OutputOffset" in finalize
    assert "outputSequenceMajor" not in finalize
    assert "return ((b * T_ + t) * HV_ + hv) * V_ + d;" in finalize
    assert "const uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;" in finalize
    assert "CopyRowsOut(vNew_, OutputOffset(b, hv, ti, 0), outTyped, tileRows, V_, HV_ * V_);" in finalize
    assert "LoopModeParams loopParams" in finalize
    assert "SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);" in finalize
    assert finalize.count("ResetLoopModePara(DataCopyMVType::UB_TO_OUT);") == 2
    assert "KdaLayoutSwap12" not in aclnn
    assert "KdaFwdCopyAfter" not in aclnn
    assert "attn_shape" in runtime and "matrix_shape" in runtime and "h_shape" in runtime
    assert "(_chunk_count(case), Hv, *state_tail)" in generalization
    assert "(B, _chunk_count(case), Hv, *state_tail)" in generalization


def test_finalize_keeps_fp32_cube_outputs_in_per_core_workspace_tiles():
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    output_runner = finalize.split("__aicore__ inline void RunChunkKdaOutput(", 1)[1]
    tiling = (OP_ROOT / "op_host/chunk_kda_fwd_tiling.cpp").read_text(
        encoding="utf-8"
    )
    assert "GM_ADDR stateScratch = outputScratch;" in output_runner
    assert "GM_ADDR localScratch = outputScratch;" in output_runner
    assert "propagatedVNew, propagatedH, stateScratch" in output_runner
    assert "userWorkspace, localScratch" in output_runner
    assert "userWorkspace, userWorkspace, o, propagatedH" in output_runner
    assert "2 * KDA_OUTPUT_SLOT_DEPTH * solveCoreIdx_ * outputTileElements_" in finalize
    assert "activeOutputSlot_ * 2 * outputTileElements_" in finalize
    assert "KDA_OUTPUT_SLOT_DEPTH = 2" in finalize
    assert "OutputScratchOffset(mOffset, nOffset)" in finalize
    assert "OutputScratchOffset(tileRow, 0)" in finalize
    assert "KDA_OUTPUT_SLOT_DEPTH = 2" in tiling
    assert "KDA_OUTPUT_SCRATCH_PLANES = 2" in tiling
    assert "static_cast<uint64_t>(blockDim), KDA_OUTPUT_SLOT_DEPTH" in tiling
    assert "KDA_OUTPUT_SCRATCH_PLANES, static_cast<uint64_t>(chunkSize)" in tiling
    assert "static_cast<uint64_t>(chunkSize)" in tiling
    assert "static_cast<uint64_t>(shape.vDim)" in tiling
    assert "outputScratchElements, sizeof(float), finalizeScratchBytes" in tiling
    assert "postWuStagingBytes = vTensorBytes" in tiling
    assert "postWuStagingBytes, kTensorBytes, postWuStagingBytes" in tiling
    assert "!isAscend950 && isVarLen && hasVarlenTail" in tiling
    assert "std::max(finalizeScratchBytes, postWuStagingBytes)" in tiling
    assert "cursor, outputScratchBytes" in tiling
    output_scratch = tiling.split("uint64_t outputScratchElements = 0;", 1)[1].split(
        "uint64_t outputScratchOffset", 1
    )[0]
    assert "blockDim" in output_scratch
    assert "finalizeScratchBytes" in output_scratch
    assert "postWuStagingBytes" in output_scratch
    assert "arch35Options.useDenseFwdH" not in output_scratch
    assert "32" not in output_scratch


def test_workspace_and_compact_plan_arithmetic_fail_closed_on_overflow():
    tiling = TILING_ENTRY.read_text(encoding="utf-8")
    helpers = tiling.split("bool CheckedAdd", 1)[1].split("bool HasOutput", 1)[0]
    plan_append = tiling.split("const size_t fixedTilingBytes", 1)[1].split(
        "return ge::GRAPH_SUCCESS;", 1
    )[0]

    assert "std::numeric_limits<uint64_t>::max() - lhs" in helpers
    assert "std::numeric_limits<uint64_t>::max() / lhs" in helpers
    assert "CheckedAdd(bytes, KDA_ALIGN - 1, rounded)" in helpers
    assert "CheckedAlign(cursor, offset) && CheckedAdd(offset, bytes, cursor)" in helpers
    assert "fixedTilingBytes > std::numeric_limits<uint32_t>::max()" in plan_append
    assert "compactPlan.size() > std::numeric_limits<uint32_t>::max()" in plan_append
    assert "CheckedAdd(fixedTilingBytes, compactPlan.size(), totalTilingBytes)" in plan_append
    assert "totalTilingBytes > rawTiling->GetCapacity()" in plan_append
    assert "static_cast<size_t>(totalTilingBytes)" in plan_append


def test_manifest_registers_positive_tnd_output_layout_case():
    manifest = json.loads(CASE_MANIFEST.read_text(encoding="utf-8"))
    case = next(item for item in manifest["cases"] if item["id"] == "chunk_kda_fwd_tnd_layout")
    coverage = manifest["coverage_requirements"]
    assert case["id"] in coverage["accuracy_case_ids"]
    assert case["id"] in coverage["generalization_case_ids"]
    assert case["layout"] == "TND"
    assert case["shape"]["H_k"] == 1 and case["shape"]["H_v"] == 2
    assert case["attrs"]["output_final_state"] is True
    assert case["attrs"]["return_intermediate_states"] is True
    assert set(case["soc"]) == {"ascend910b", "ascend910_93", "ascend950"}


def test_manifest_registers_state_v_first_sequence_major_h_case():
    manifest = json.loads(CASE_MANIFEST.read_text(encoding="utf-8"))
    case = next(item for item in manifest["cases"] if item["id"] == "chunk_kda_fwd_state_v_first")
    coverage = manifest["coverage_requirements"]
    assert case["id"] in coverage["accuracy_case_ids"]
    assert case["id"] in coverage["generalization_case_ids"]
    assert case["attrs"]["state_v_first"] is True
    assert case["attrs"]["return_intermediate_states"] is True
    assert case["shape"]["K"] == 128 and case["shape"]["V"] == 256
    assert set(case["soc"]) == {"ascend910b", "ascend910_93", "ascend950"}


def test_tiling_key_has_design_rationale():
    design = (OP_ROOT / "docs/design.md").read_text(encoding="utf-8")
    assert "### 9.1 tiling key" in design
    assert "编译期" in design and "独立" in design


def test_a5_gate_chunk_bulk_regbase_is_arch_guarded():
    gate = (
        ROOT
        / "fla/ops/ascendc/kda/kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h"
    ).read_text(encoding="utf-8")
    marker = "__aicore__ inline void ProcessChunkBulkFp32("
    arch_regions = gate.split(
        "#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310"
    )[1:]
    guarded = next(region.split("#endif", 1)[0] for region in arch_regions if marker in region)
    assert "AccumulateGateChunk128Regbase" in guarded.split(marker, 1)[1]
