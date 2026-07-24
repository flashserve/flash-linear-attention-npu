"""Static kernel/tiling contract for chunk_kda_fwd."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OP_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd"
LEGACY_COMMON_KERNEL = ROOT / "fla/ops/ascendc/common/kda/chunk_kda_fwd_kernel.hpp"
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
STAGE_IMPLEMENTATIONS = {
    stage: path.with_name(f"chunk_kda_fwd_{stage}_kernel.hpp")
    for stage, path in STAGE_KERNELS.items()
}
STAGE_IMPLEMENTATIONS["output"] = STAGE_KERNELS["output"].with_name(
    "chunk_kda_fwd_finalize_kernel.hpp"
)
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
    assert not LEGACY_COMMON_KERNEL.exists()
    assert not (OP_ROOT / "op_kernel/chunk_kda_fwd.cpp").exists()
    assert not (OP_ROOT / "op_host/chunk_kda_fwd_tiling.cpp").exists()

    expected = {
        "prepare": ("RunChunkKdaPrepare", "ChunkKdaFwdPrepareKernel"),
        "post_wu": ("RunChunkKdaPostWu", "ChunkKdaFwdPostWuKernel"),
        "output": ("RunChunkKdaOutput", "ChunkKdaFwdFinalizeKernel"),
    }
    for stage, path in STAGE_KERNELS.items():
        entry = path.read_text(encoding="utf-8")
        implementation_path = STAGE_IMPLEMENTATIONS[stage]
        implementation = implementation_path.read_text(encoding="utf-8")
        runner, kernel_class = expected[stage]
        assert f'#include "{implementation_path.name}"' in entry
        assert runner in entry and runner in implementation
        assert kernel_class in implementation
        assert "KdaPhase" not in implementation
        assert "RunChunkKdaFused" not in implementation
        assert "SyncAll" not in entry and "SyncAll" not in implementation
        for other_stage, (other_runner, other_class) in expected.items():
            if other_stage != stage:
                assert other_runner not in entry and other_runner not in implementation
                assert other_class not in implementation

    direct = DIRECT_SOURCE.read_text(encoding="utf-8")
    for implementation_path in STAGE_IMPLEMENTATIONS.values():
        assert implementation_path.as_posix().split("fla/", 1)[1] in direct
    assert "common/kda/chunk_kda_fwd_kernel.hpp" not in direct


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
    assert "chunkIndicesOptional, true, chunkSize, false, hOut, vNewOut" in l0
    assert "kgOut, wOut, uOut, neutralGForH, gk" in l0
    assert "const aclTensor *neutralGForH = ZerosLike(beta, executor);" in l0
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
    assert "ScoreRefBlockSize" in STAGE_IMPLEMENTATIONS["prepare"].read_text(
        encoding="utf-8"
    )


def test_fp16_score_pipeline_does_not_fall_back_to_two_row_cube_tiles():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    score_ref_block = prepare.split(
        "__aicore__ inline uint64_t ScoreRefBlockSize() const", 1
    )[1].split("__aicore__ inline uint64_t ScoreRowBlockCount", 1)[0]
    assert "KDA_SCORE_REF_BC = 32" in prepare
    assert "KDA_SAFE_SCORE_REF_BC = 16" in prepare
    assert "return KDA_SAFE_SCORE_REF_BC;" in score_ref_block
    assert "return KDA_SCORE_REF_BC;" in score_ref_block
    assert "return 2;" not in score_ref_block


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
    assert "KDA_GATE_PIPELINE_DEPTH = 2" in prepare
    assert "GatePipelineRows() * K_" in prepare
    assert "PrefetchQKGate(gateSlot" in gate_bulk
    assert "PrefetchQKGate(gateSlot ^ 1" in gate_bulk
    assert pipelined_loop.index("WaitGateInputReady();") < pipelined_loop.index(
        "PrefetchQKGate(gateSlot ^ 1"
    ) < pipelined_loop.index("if (useRef) {")
    assert pipelined_loop.index("WaitGateOutputForMte2();") < pipelined_loop.index(
        "PrefetchQKGate(gateSlot ^ 1"
    )
    assert "SetFlag<HardEvent::MTE3_MTE2>" in signal_output
    assert "SetFlag<HardEvent::MTE3_V>" in signal_output
    assert "WaitFlag" not in signal_output


def test_prepare_uses_a5_regbase_gate_math_with_a2_a3_fallback():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    assert '#include "kernel_utils/vector/regbase.hpp"' in prepare
    assert "static __simd_vf__ inline void PrepareKdaGateQwRegbase" in prepare
    assert "static __simd_vf__ inline void PrepareKdaGateKgRegbase" in prepare
    assert "PrepareKdaGateQwRegbase<T, GK_T, true>" in prepare
    assert "PrepareKdaGateQwRegbase<T, GK_T, false>" in prepare
    assert "PrepareKdaGateKgRegbase<T, GK_T, true>" in prepare
    assert "PrepareKdaGateKgRegbase<T, GK_T, false>" in prepare
    assert "ClampKdaGateRegbaseOutput" in prepare
    assert "KDA_EXP_INPUT_MAX" in prepare
    assert "KDA_EXP_INPUT_MIN" in prepare
    assert "row >= validRows" in prepare
    assert "#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310" in prepare
    assert "Cast(qTyped, outFp32, RoundMode::CAST_RINT" in prepare


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


def test_a2_fwd_h_keeps_fp32_recurrence_state():
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

    assert "bool useFp32Recurrence" in update
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


def test_a5_fwd_h_uses_canonical_h_recurrence_and_fp32_final_state():
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
    assert "ApplyRowScale(calcUbTensor, gkLastUbTensor" in update
    assert "AscendC::Add<float>(hUpdateUbTensor, calcUbTensor" in update
    assert "CopyUbToGm(finalStateThisTile, hUpdateUbTensor" in update
    assert "CopyUbToGm(hOutputThisTile, hUbTensor" in update
    assert "gmInitialState[vec2Offsets.initialStateOffset]" in kernel
    assert "event0FromMte3[streamId] = vec2Offsets.isFinalState;" in kernel
    empty_subblock = vnew.split("if (rowBegin >= mActual) {", 1)[1].split(
        "return;", 1
    )[0]
    assert "if (waitWsFromMte3)" in empty_subblock
    assert "WaitFlag<AscendC::HardEvent::MTE3_MTE2>" in empty_subblock
    assert "SetFlag<AscendC::HardEvent::V_MTE2>" in empty_subblock


def test_kda_keeps_fp32_recurrence_when_final_state_is_not_returned():
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    l0 = (OP_ROOT / "op_host/op_api/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    assert "const aclTensor *finalStateCompute = params.finalStateOut;" in aclnn
    assert "if (!params.outputFinalState)" in aclnn
    assert "KdaFwdMakeShape({seqNum, hvNum, kDim, vDim})" in aclnn
    assert "DataType::DT_FLOAT, Format::FORMAT_ND" in aclnn
    assert "oBnsd, finalStateCompute," in aclnn
    assert "chunkIndicesOptional, true, chunkSize, false, hOut, vNewOut," in l0


def test_fwd_h_dispatches_optional_gate_dtype_without_full_runtime_expansion():
    dispatch = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/chunk_gated_delta_rule_fwd_h.cpp"
    ).read_text(encoding="utf-8")
    assert (
        "ChunkGatedDeltaRuleFwdHDispatchGate<DTYPE_K, DTYPE_INITAL_STATE, "
        "TileShapes, useExp2>" in dispatch
    )
    assert "ChunkGatedDeltaRuleFwdHLaunchTyped<DataT, float, StateT" in dispatch
    assert "ChunkGatedDeltaRuleFwdHLaunchTyped<DataT, bfloat16_t, StateT" in dispatch
    assert "ChunkGatedDeltaRuleFwdHLaunchTyped<DataT, half, StateT" in dispatch
    assert "gdnFwdHTilingData->gDataType" in dispatch
    assert "DTYPE_GK" not in dispatch
    for runtime_dtype in ("->dataType", "->stateDataType"):
        assert runtime_dtype not in dispatch


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


def test_aqk_akk_share_one_l1_resident_right_matrix_slot():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    resident_mmad = (
        ROOT
        / "fla/ops/ascendc/common/kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
    ).read_text(encoding="utf-8")
    assert "using KdaScoreDispatchPolicy" in prepare
    assert "MmadPingpongTlaMulti<KdaArchTag, true, false, 1, true, 2, 1, 2, 2>" in prepare
    assert "KdaScoreDispatchPolicy::ENABLE_L1_RESIDENT" in prepare
    assert "KdaScoreDispatchPolicy::L1B_STAGES == 1" in prepare
    score_block = prepare.split(
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
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(
        encoding="utf-8"
    )
    runtime = (
        ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"
    ).read_text(encoding="utf-8")
    assert "OutputOffset" in finalize
    assert "outputSequenceMajor_" in finalize
    assert "const uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;" in finalize
    assert "CopyVectorOut(vNew_, OutputOffset(b, hv, ti, 0), outTyped, elems);" in finalize
    assert "!isInternalLayout" in aclnn
    assert "KdaFwdCopyAfter" not in aclnn
    assert "bnsd_k_shape" in runtime and "bnsd_v_shape" in runtime


def test_finalize_keeps_fp32_cube_outputs_in_workspace():
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    output_runner = finalize.split("__aicore__ inline void RunChunkKdaOutput(", 1)[1]
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
