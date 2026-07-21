"""Static kernel/tiling contract for chunk_kda_fwd."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OP_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd"
COMMON_KERNEL = ROOT / "fla/ops/ascendc/common/kda/chunk_kda_fwd_kernel.hpp"
CASE_MANIFEST = ROOT / "tests/op_cases/chunk_kda_fwd.json"
DIRECT_SOURCE = (
    ROOT
    / "examples/fast_kernel_launch_example/csrc/chunk_kda_fwd/chunk_kda_fwd_direct.cpp"
)
STAGE_KERNELS = {
    "prepare": ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/chunk_kda_fwd_prepare.cpp",
    "post_wu": ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd_post_wu/op_kernel/chunk_kda_fwd_post_wu.cpp",
    "output": ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd_finalize/op_kernel/chunk_kda_fwd_finalize.cpp",
}
STAGE_TILINGS = {
    "prepare": STAGE_KERNELS["prepare"].parent.parent
    / "op_host/chunk_kda_fwd_prepare_tiling.h",
    "post_wu": STAGE_KERNELS["post_wu"].parent.parent
    / "op_host/chunk_kda_fwd_post_wu_tiling.h",
    "output": STAGE_KERNELS["output"].parent.parent
    / "op_host/chunk_kda_fwd_finalize_tiling.h",
}


def test_direct_launch_uses_four_real_stage_kernels():
    text = DIRECT_SOURCE.read_text(encoding="utf-8")
    assert "chunk_kda_fwd_direct" in text
    assert "stage" not in text.split("TORCH_LIBRARY_FRAGMENT", 1)[1]
    assert text.count("<<<blockDim") == 4
    for name in (
        "ChunkKdaPrepareDirectKernel",
        "ChunkKdaPostWuDirectKernel",
        "ChunkKdaFwdHDirectKernel",
        "ChunkKdaOutputDirectKernel",
    ):
        assert name in text
    assert "RunChunkKdaFused" not in text
    assert "SyncAll" not in text


def test_each_device_kernel_owns_exactly_one_stage():
    common = COMMON_KERNEL.read_text(encoding="utf-8")
    assert "template <KdaPhase PHASE, bool SAFE_GATE, typename T" in common
    assert "RunChunkKdaFused" not in common
    assert "SyncAll" not in common
    assert not (OP_ROOT / "op_kernel/chunk_kda_fwd.cpp").exists()
    assert not (OP_ROOT / "op_host/chunk_kda_fwd_tiling.cpp").exists()

    expected = {
        "prepare": "RunChunkKdaPrepare",
        "post_wu": "RunChunkKdaPostWu",
        "output": "RunChunkKdaOutput",
    }
    for stage, path in STAGE_KERNELS.items():
        text = path.read_text(encoding="utf-8")
        assert text.count(expected[stage]) >= 1
        assert "SyncAll" not in text
        for other_stage, runner in expected.items():
            if other_stage != stage:
                assert runner not in text


def test_each_stage_declares_generated_matmul_workspace_dependency():
    for path in STAGE_KERNELS.values():
        text = path.read_text(encoding="utf-8")
        assert '#include "lib/matmul_intf.h"' in text


def test_each_stage_tiling_owns_only_its_workspace():
    expected = {
        "prepare": "prepareScratchOffset",
        "post_wu": "postWuScratchOffset",
        "output": "outputScratchOffset",
    }
    for stage, path in STAGE_TILINGS.items():
        text = path.read_text(encoding="utf-8")
        assert expected[stage] in text
        assert "fwdH" not in text
        for other_stage, field in expected.items():
            if other_stage != stage:
                assert field not in text


def test_l0_queues_standalone_fwd_h_between_kda_stages():
    l0 = (OP_ROOT / "op_host/op_api/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    launches = (
        l0.index("ADD_TO_LAUNCHER_LIST_AICORE(\n        ChunkKdaFwdPrepare"),
        l0.index("ADD_TO_LAUNCHER_LIST_AICORE(\n        ChunkKdaFwdPostWu"),
        l0.index("auto hResult = ChunkGatedDeltaRuleFwdH("),
        l0.index("ADD_TO_LAUNCHER_LIST_AICORE(\n        ChunkKdaFwdFinalize"),
    )
    assert launches == tuple(sorted(launches))
    assert "chunkSize, true, hOut" in l0
    assert "kgOut, wOut, uOut, nullptr, gk" in l0
    assert "ZerosLike" not in l0
    assert "RunChunkKdaFused" not in l0


def test_intermediate_outputs_keep_canonical_bnsd_graph_views_between_stages():
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    for output in ("aqkOut", "akkOut", "wOut", "uOut", "qgOut", "kgOut", "vNewOut", "hOut"):
        assert f"l0op::Reshape(params.{output}" in aclnn
    intermediate_block = aclnn.split("if (returnIntermediates) {", 1)[1].split(
        "const bool internalIntermediateOutputsReady", 1
    )[0]
    assert "if (isTnd)" not in intermediate_block
    assert "KdaFwdMakeShape({batch, hvNum" in intermediate_block


def test_safe_gate_is_supported_across_public_and_direct_routes():
    aclnn_runtime = (
        ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"
    ).read_text(encoding="utf-8")
    legacy_runtime = (
        ROOT / "torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp"
    ).read_text(encoding="utf-8")
    direct = DIRECT_SOURCE.read_text(encoding="utf-8")
    prepare = STAGE_KERNELS["prepare"].read_text(encoding="utf-8")

    assert "safe_gate is reserved" not in aclnn_runtime
    assert "safe_gate=true is not supported" not in legacy_runtime
    assert "ctypes.c_bool(safe_gate)" in aclnn_runtime
    assert "bool safe_gate=False" in direct
    assert "RunChunkKdaPrepare<true" in prepare
    assert "RunChunkKdaPrepare<false" in prepare
    assert "ScoreRefBlockSize" in COMMON_KERNEL.read_text(encoding="utf-8")


def test_fp16_score_pipeline_does_not_fall_back_to_two_row_cube_tiles():
    common = COMMON_KERNEL.read_text(encoding="utf-8")
    score_ref_block = common.split(
        "__aicore__ inline uint64_t ScoreRefBlockSize() const", 1
    )[1].split("__aicore__ inline uint64_t ScoreRowBlockCount", 1)[0]
    assert "return KDA_SCORE_REF_BC;" in score_ref_block
    assert "return 2;" not in score_ref_block


def test_fwd_h_supports_exp_and_exp2_on_a2_and_a5():
    fwd_h_root = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
    )
    dispatch = (fwd_h_root / "op_kernel/chunk_gated_delta_rule_fwd_h.cpp").read_text(
        encoding="utf-8"
    )
    assert "ChunkGatedDeltaRuleFwdHDispatchExp<TileShapes, true>" in dispatch
    assert "ChunkGatedDeltaRuleFwdHDispatchExp<TileShapes, false>" in dispatch
    assert "true, false, useExp2" in dispatch
    for arch in ("", "arch35/"):
        epilogue = fwd_h_root / f"op_kernel/{arch}epilogue/block"
        update = (epilogue / "block_epilogue_gdn_fwdh_update.hpp").read_text(
            encoding="utf-8"
        )
        vnew = (epilogue / "block_epilogue_gdn_fwdh_vnew.hpp").read_text(
            encoding="utf-8"
        )
        assert "if constexpr (useExp2)" in update
        assert "if constexpr (useExp2)" in vnew
        assert "LN2" in update and "LN2" in vnew


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
    assert "ApplyKGateUpdateRegbaseDualIssue" in regbase

    update = (block_root / "block_epilogue_gdn_fwdh_update.hpp").read_text(
        encoding="utf-8"
    )
    vnew = (block_root / "block_epilogue_gdn_fwdh_vnew.hpp").read_text(
        encoding="utf-8"
    )
    assert "VF_CALL<detail::PrepareKGateRegbase" in update
    assert "VF_CALL<detail::ApplyKGateUpdateRegbaseDualIssue" in update
    assert "if constexpr (kGated && !scalarGated)" in update
    assert "VF_CALL<detail::ComputeVNewRegbaseDualIssue" in vnew
    assert "AscendC::LocalTensor<float> decayInput = scalarGated ?" in vnew


def test_aqk_akk_share_one_l1_resident_right_matrix_slot():
    common = COMMON_KERNEL.read_text(encoding="utf-8")
    resident_mmad = (
        ROOT
        / "fla/ops/ascendc/common/kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
    ).read_text(encoding="utf-8")
    assert "using KdaScoreDispatchPolicy" in common
    assert "MmadPingpongTlaMulti<KdaArchTag, true, false, 1, true, 2, 1, 2, 2>" in common
    assert "KdaScoreDispatchPolicy::ENABLE_L1_RESIDENT" in common
    assert "KdaScoreDispatchPolicy::L1B_STAGES == 1" in common
    score_block = common.split(
        "__aicore__ inline void ComputeRawAqkAkkCubeBlock", 1
    )[1].split("__aicore__ inline bool UseAkkCubeSolve", 1)[0]
    assert "BlockMmadTla<KdaScoreDispatchPolicy" in score_block
    assert score_block.count("blockMmad(block") == 2
    assert "blockMmad.preSetFlags();" in score_block
    assert "blockMmad.finalWaitFlags();" in score_block
    assert "PipeBarrier<PIPE_ALL>()" not in score_block
    assert resident_mmad.count("static_cast<uint32_t>(tla::get<0>(tensorTile") == 4
    assert resident_mmad.count("static_cast<uint32_t>(tla::get<1>(tensorTile") == 4


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
    common = COMMON_KERNEL.read_text(encoding="utf-8")
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    runtime = (
        ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"
    ).read_text(encoding="utf-8")
    assert "OutputOffset" in common
    assert "outputSequenceMajor_" in common
    assert "const uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;" in common
    assert "CopyVectorOut(vNew_, OutputOffset(b, hv, ti, 0), outTyped, elems);" in common
    assert "!isInternalLayout" in aclnn
    assert "KdaFwdCopyAfter" not in aclnn
    assert "bnsd_k_shape" in runtime and "bnsd_v_shape" in runtime


def test_finalize_keeps_fp32_cube_outputs_in_workspace():
    common = COMMON_KERNEL.read_text(encoding="utf-8")
    output_runner = common.split("__aicore__ inline void RunChunkKdaOutput(", 1)[1]
    assert "GM_ADDR stateScratch = outputScratch;" in output_runner
    assert "GM_ADDR localScratch = outputScratch + outputElements * sizeof(float);" in output_runner
    assert "propagatedVNew, propagatedH, stateScratch" in output_runner
    assert "userWorkspace, localScratch" in output_runner
    assert "userWorkspace, userWorkspace, o, propagatedH" in output_runner


def test_manifest_registers_positive_tnd_output_layout_case():
    manifest = json.loads(CASE_MANIFEST.read_text(encoding="utf-8"))
    case = next(item for item in manifest["cases"] if item["id"] == "chunk_kda_fwd_tnd_layout")
    coverage = manifest["coverage_requirements"]
    assert case["id"] in coverage["accuracy_case_ids"]
    assert case["id"] in coverage["generalization_case_ids"]
    assert case["layout"] == "TND"
    assert case["shape"]["H_k"] == 1 and case["shape"]["H_v"] == 2
    assert case["attrs"]["output_final_state"] is True
    assert case["attrs"]["return_intermediate"] is True
    assert set(case["soc"]) == {"ascend910b", "ascend910_93", "ascend950"}


def test_tiling_key_has_design_rationale():
    design = (OP_ROOT / "docs/design.md").read_text(encoding="utf-8")
    assert "模板化方案与 tiling key" in design
    assert "编译期" in design and "独立" in design
