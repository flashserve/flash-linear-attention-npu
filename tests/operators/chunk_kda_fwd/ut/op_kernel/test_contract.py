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
    assert "kgOut, wOut, uOut, nullptr, gk, initialStateOptional" in l0
    assert "chunkIndicesOptional, outputFinalState, chunkSize, hOut, vNewOut" in l0
    assert "neutralGForH" not in l0
    assert "RunChunkKdaFused" not in l0


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
        "wOut": ("wExport", "wCompute"),
        "uOut": ("uExport", "uCompute"),
        "qgOut": ("qgExport", "qgCompute"),
        "kgOut": ("kgExport", "kgCompute"),
        "vNewOut": ("vNewExport", "vNewCompute"),
        "hOut": ("hExport", "hCompute"),
    }
    for output, (export, compute) in optional_names.items():
        assert f"const aclTensor *{export} = params.{output};" in aclnn
        assert f"const aclTensor *{compute} = AllocTensor(" in aclnn
        assert f"Transpose(params.{output}" not in aclnn
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
        assert "totalTokens = prevSeq" in init
        assert "inputTokenBatch = tokenBatch" in init
        assert "gmNumChunks.SetValue" not in source
        assert "gmNumSeq.SetValue" not in source
        assert "gmNumChunks.GetValue" not in source
        assert "gmNumSeq.GetValue" not in source
        assert "gmSeqlen.GetValue" in resolve
        assert "stream.chunkOffset" in resolve
        assert "stream.tokenOffset" in resolve

    kernels = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/"
        "gdn_fwd_h_kernel.hpp",
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
        "chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/"
        "gdn_fwd_h_kernel.hpp",
    )
    for path in kernels:
        source = path.read_text(encoding="utf-8")
        assert "GetVarlenChunkOffset(batchIdx)" in source
        assert "gmNumChunks.GetValue" not in source
        assert "gmNumSeq.GetValue" not in source


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
    assert "PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, true>" in prepare
    assert "PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, false>" in prepare
    assert "PrepareKdaGateQwKgRegbase<T, T, GK_T, true, false>" in prepare
    assert "PrepareKdaGateQwKgRegbase<T, T, GK_T, false, false>" in prepare
    assert "ClampKdaGateRegbaseOutput" in prepare
    assert "KDA_EXP_INPUT_MAX" in prepare
    assert "KDA_EXP_INPUT_MIN" in prepare
    assert "row >= validRows" in prepare
    assert "#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310" in prepare
    assert "Cast(qTyped, outFp32, RoundMode::CAST_RINT" in prepare


def test_a5_fused_post_wu_protects_l0c_reuse_with_fix_to_cube_events():
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    fused = post_wu.split(
        "__aicore__ inline void ComputePostWuCubeFusedA5", 1
    )[1].split("__aicore__ inline void ComputePostWuCube(", 1)[0]

    assert fused.count("SetFlag<HardEvent::FIX_M>(KDA_POST_EVENT_FIX)") == 2
    assert fused.count("WaitFlag<HardEvent::FIX_M>(KDA_POST_EVENT_FIX)") == 2
    assert "FIX_MTE2>(KDA_POST_EVENT_FIX)" not in fused


def test_a5_finalize_stages_full_chunk_mmads_without_l0c_accumulation():
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    dispatch = finalize.split(
        "__aicore__ inline void ComputeOutputCube(", 1
    )[1].split("using ElementA = T;", 1)[0]
    staged = finalize.split(
        "__aicore__ inline void ComputeOutputCubeStagedA5", 1
    )[1].split("__aicore__ inline void PrefetchOutputTileA5", 1)[0]
    writeback = finalize.split(
        "__aicore__ inline void FinalizeOutputRows(", 1
    )[1].split("__aicore__ inline bool ResolveFlatChunk(", 1)[0]

    assert "BT_ == 64 && curT == BT_" in dispatch
    assert "ComputeOutputCubeStagedA5" in dispatch
    assert staged.count("true, 0b11") == 2
    assert "false, 0b11" not in staged
    assert "copyL0CToDst(blockO, tileL0C, 0b11);" in staged
    assert "copyL0CToDst(blockLocal, tileL0C, 0b11);" in staged
    assert "fusedA5Output" not in writeback
    assert "CopyVectorIn(localLocal, u_" in writeback
    assert "Add(outLocal, stateLocal, localLocal" in writeback


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
    assert "AscendC::Add<float>(" in update
    assert "hUpdateUbTensorThisTile, calcUbTensor, hUpdateUbTensorThisTile" in update
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
    assert "const bool outputFinalState = params.finalStateOut != nullptr;" in aclnn
    assert "const aclTensor *finalStateCompute = AllocTensor(" in aclnn
    assert "MakeShape({info.seqNum, info.hvNum, info.kDim, info.vDim})" in aclnn
    assert "AllocTensor(executorPtr, stateShape4, DataType::DT_FLOAT)" in aclnn
    assert "finalStateCompute, aqkCompute," in aclnn
    assert "chunkIndicesOptional, outputFinalState, chunkSize, hOut, vNewOut," in l0


def test_aclnn_l2_optional_outputs_are_pointer_driven():
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
    for compute_name in (
        "wCompute",
        "uCompute",
        "qgCompute",
        "kgCompute",
        "vNewCompute",
        "hCompute",
    ):
        assert f"const aclTensor *{compute_name} = AllocTensor(" in aclnn
    for result_index, export_name in (
        (4, "wExport"),
        (5, "uExport"),
        (6, "qgExport"),
        (7, "kgExport"),
        (8, "vNewExport"),
    ):
        assert f"if ({export_name} != nullptr)" in aclnn
        assert f"l0op::ViewCopy(result[{result_index}]" in aclnn
    assert "if (hExport != nullptr)" in aclnn
    assert "const aclTensor *hResult = Transpose(result[9], hPerm, executorPtr);" in aclnn
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


def test_a5_prepare_joins_both_aiv_subcores_before_shared_ready_signal():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    join = prepare.split(
        "__aicore__ inline void JoinA5AivMte3()", 1
    )[1].split("__aicore__ inline void RunAicAfterBothAivReady", 1)[0]
    assert "CrossCoreBarrier<0x1, PIPE_MTE3>();" in join
    assert "PipeBarrier<PIPE_MTE3>();" in join
    run_after_join = prepare.split(
        "__aicore__ inline void RunAicAfterBothAivReady", 1
    )[1].split("__aicore__ inline void SignalAicSolveReady", 1)[0]
    signal_solve = prepare.split(
        "__aicore__ inline void SignalAicSolveReady", 1
    )[1].split("__aicore__ inline void WaitAicSolveDone", 1)[0]
    score_loop = prepare.split(
        "__aicore__ inline void ProcessChunkPreAivFp32", 1
    )[1].split("Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);", 1)[0]
    assert run_after_join.index("JoinA5AivMte3();") < run_after_join.index(
        "CrossCoreSetFlag<0x2, PIPE_MTE3>(syncReadyFlag_);"
    )
    assert run_after_join.index("JoinA5AivMte3();") < run_after_join.index(
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(mchSyncReadyFlag_);"
    )
    assert signal_solve.index("JoinA5AivMte3();") < signal_solve.index(
        "CrossCoreSetFlag<0x2, PIPE_MTE3>(syncReadyFlag_);"
    )
    assert score_loop.rindex("JoinA5AivMte3();") < score_loop.rindex(
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);"
    )


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
    assert "模板化方案与 tiling key" in design
    assert "编译期" in design and "独立" in design
