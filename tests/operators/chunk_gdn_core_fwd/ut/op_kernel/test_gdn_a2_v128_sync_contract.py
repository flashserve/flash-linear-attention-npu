"""Static contracts for the A2 fused GDN determinism fix."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
FWD_H_SCHEDULER = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/"
    "op_kernel/gemm/block/block_scheduler_gdn_fwd_h.hpp"
)
FWD_H_KERNEL = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/"
    "op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
)
FWD_H_VNEW_EPILOGUE = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/"
    "op_kernel/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
)
FWD_O_KERNEL = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/"
    "op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp"
)
FUSED_HO_KERNEL = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_recompute_wu_fwd_ho/"
    "op_kernel/chunk_recompute_wu_fwd_ho.cpp"
)


def test_fwd_h_splits_v256_into_v128_logical_tasks():
    scheduler = FWD_H_SCHEDULER.read_text(encoding="utf-8")
    kernel = FWD_H_KERNEL.read_text(encoding="utf-8")

    assert "vBlockSize = Min(vHeadDim, static_cast<uint32_t>(128));" in scheduler
    assert "vBlockCount = CeilDiv(vHeadDim, vBlockSize);" in scheduler
    assert "taskNum = vBlockCount * batch * vNumHead;" in scheduler
    assert "taskCount = logicalHeadTasks * vecBlockScheduler.vBlockCount;" in kernel
    assert "vBlockOffset = vBlockIdx * vecBlockScheduler.vBlockSize;" in kernel
    assert "vBlockDim == vHeadDim ? rowsPerTile : 1" in kernel


def test_fwd_o_joins_aiv_subblocks_before_completion_publication():
    kernel = FWD_O_KERNEL.read_text(encoding="utf-8")

    vec1_completion = (
        "Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();\n"
        "                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>("
        "vecBlockScheduler.vec1Done[streamId]);"
    )
    vec2_completion = (
        "Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();\n"
        "                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>("
        "vecBlockScheduler.vec2Done[streamId]);"
    )
    obsolete_fence = (
        "blockMmadAttenVNEW128.finalWaitFlags();\n"
        "                    AscendC::PipeBarrier<PIPE_FIX>();"
    )

    assert vec1_completion in kernel
    assert vec2_completion in kernel
    assert kernel.count("CrossCoreBarrier<0x1, PIPE_MTE3>();") == 2
    assert obsolete_fence not in kernel


def test_fwd_h_v1_joins_aiv_subblocks_before_c2_consumes_workspace():
    epilogue = FWD_H_VNEW_EPILOGUE.read_text(encoding="utf-8")

    publish_helper = (
        "void PublishVec1Done(Arch::CrossCoreFlag &vec1Done)\n"
        "    {\n"
        "        // V1 partitions the token rows across the two AIV subblocks while C2\n"
        "        // consumes the complete workspace tile. Publish only after both MTE3\n"
        "        // producers have closed the same workspace generation.\n"
        "        Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();\n"
        "        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);"
    )

    assert publish_helper in epilogue
    assert epilogue.count("PublishVec1Done(vec1Done);") == 5
    assert epilogue.count("CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);") == 1


def test_fwd_h_v2_joins_aiv_subblocks_before_releasing_workspace():
    kernel = FWD_H_KERNEL.read_text(encoding="utf-8")

    v2_completion = (
        "Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();\n"
        "                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>("
        "vecBlockScheduler.vec2Done[streamId]);"
    )

    assert v2_completion in kernel


def test_fwd_h_tail_c2_retires_mte3_before_reusing_accumulator():
    kernel = FWD_H_KERNEL.read_text(encoding="utf-8")
    tail_h = kernel.split("void ComputeTailHWorkspace", maxsplit=1)[1].split(
        "void PresetVectorPipelineEvents", maxsplit=1
    )[0]

    ordered_writeback = (
        "gmHWorkspace[offsets.hWorkOffset + kRow * offsets.vBlockDim],\n"
        "                accumUb, offsets.vBlockDim);\n"
        "            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);\n"
        "            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);\n"
        "            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);\n"
        "            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);"
    )

    assert ordered_writeback in tail_h


def test_embedded_fwd_h_uses_standalone_compile_mode():
    fused = FUSED_HO_KERNEL.read_text(encoding="utf-8")

    assert (
        "InputT, GT, StateT, float, TileShapes, kGated, true, false, false>"
        in fused
    )
    assert "GDNFwdOKernel<InputT, GT, float, true>" in fused
