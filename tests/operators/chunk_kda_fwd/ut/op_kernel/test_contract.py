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
ARCH35_TILING = OP_ROOT / "op_host/arch35/chunk_kda_fwd_tiling_impl.h"
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


def test_a5_reuses_chunk_kda_fwd_kernel_launch_entry_with_internal_arch35_stages():
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


def test_mode4_cross_core_sync_is_isolated_to_arch35():
    generic_sources = [
        KERNEL_ENTRY,
        KERNEL_COMMON,
        *GENERIC_STAGE_IMPLEMENTATIONS.values(),
    ]
    for path in generic_sources:
        text = path.read_text(encoding="utf-8")
        assert "CrossCoreSetFlag<0x4" not in text
        assert "CrossCoreWaitFlag<0x4" not in text

    arch35_sources = [
        ARCH35_KERNEL,
        ARCH35_FWD_H,
        *ARCH35_STAGE_IMPLEMENTATIONS.values(),
    ]
    arch35_text = "\n".join(path.read_text(encoding="utf-8") for path in arch35_sources)
    assert "CrossCoreSetFlag<0x4" in arch35_text
    assert "CrossCoreWaitFlag<0x4" in arch35_text


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
    assert "const bool denseAligned = !isVarLen && seqlen % chunkSize == 0;" in arch35
    gate_selection = arch35.split("options.computeGateInPrepare =", 1)[1].split(";", 1)[0]
    assert "denseAligned" not in gate_selection
    assert "isVarLen" not in gate_selection
    assert "enabled" not in arch35
    for gate_condition in (
        "qIsBf16",
        "rawGIsFp32",
        "hasALog",
        "useGateInKernel",
        "safeGate",
    ):
        assert gate_condition in arch35
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


def test_arch35_key1_stub_matches_public_kernel_tensor_address_count():
    entry = KERNEL_ENTRY.read_text(encoding="utf-8")
    stub = entry.split(
        "#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 310", 1
    )[1].split("#endif", 1)[0]
    signature = stub.split("DispatchArch35SafeGate(", 1)[1].split(")", 1)[0]
    assert signature.count("GM_ADDR") == 22


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


def test_varlen_and_tail_reuse_chunk_kda_fwd_l0_registration_and_launcher():
    l0 = (OP_ROOT / "op_host/op_api/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    assert l0.count("l0op::ChunkKdaFwd(") == 0
    assert l0.count("OP_TYPE_REGISTER(ChunkKdaFwd);") == 1
    assert l0.count("ADD_TO_LAUNCHER_LIST_AICORE(") == 1
    for stage in ("ChunkKdaFwdPrepare", "ChunkKdaFwdPostWu", "ChunkKdaFwdFinalize"):
        assert stage not in l0


def test_a5_bf16_sub16_tail_uses_bounded_regbase_templates():
    finalize = ARCH35_STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    post_wu = ARCH35_STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    manifest = json.loads(CASE_MANIFEST.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["cases"]}

    assert "ComputeKdaTailOutputRegbase" in finalize
    assert "ComputeTailOutputRegbaseRows" in finalize
    output_dispatch = finalize.rsplit(
        "ComputeTailOutputRegbaseRows(", 1
    )[0][-500:]
    assert "BT_ == 64 && K_ == 128 && V_ == 128" in output_dispatch
    assert "curT < KDA_CUBE_MIN_REDUCTION" in finalize

    assert "ComputeKdaTailWuRegbase" in post_wu
    assert "ComputeTailWuRegbaseArch35" in post_wu
    assert "subBlockIdx != 0" in post_wu
    assert "curT > maxTailRows" in post_wu
    assert "BT_ == 64 && K_ == 128 && V_ == 128" in post_wu
    assert "curT < KDA_POST_REGBASE_TAIL_LIMIT" in post_wu

    h96 = cases["chunk_kda_fwd_a5_bf16_sub16_varlen_h96"]
    assert h96["shape"] == {
        "N": 4,
        "H_k": 96,
        "H_v": 96,
        "T": 281,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "N_c": 8,
    }
    cu = h96["optional_inputs"]["cu_seqlens"]
    assert [length % 64 for length in (b - a for a, b in zip(cu, cu[1:]))] == [1, 2, 7, 15]
    assert h96["expect"]["binary_deterministic_runs"] == 50


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


def test_a5_dense_aligned_prepare_post_wu_fusion_stays_inside_chunk_kda_fwd():
    common = KERNEL_COMMON.read_text(encoding="utf-8")
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    arch35 = ARCH35_TILING.read_text(encoding="utf-8")

    assert "op.ProcessAicFused(postWu);" in prepare
    assert "if (!tiling.fusePostWu && !tiling.fusePostWuIntoFwdH)" in common
    assert "const bool canFusePreparePostWu =" in arch35
    assert "denseAligned && qIsBf16 && safeGate && vHeads % 2 == 0;" in arch35
    assert "options.useDenseFwdH = denseAligned && qIsBf16;" in arch35
    fuse_into_selection = arch35.split(
        "options.fusePostWuIntoFwdH =", 1
    )[1].split(";", 1)[0]
    assert "canFusePreparePostWu" in fuse_into_selection
    fuse_selection = arch35.split("options.fusePostWu =", 1)[1].split(";", 1)[0]
    assert "canFusePreparePostWu" in fuse_selection
    assert "denseAligned" not in fuse_selection
    assert "options.fusePostWu" in arch35
    assert "options.fusePostWuIntoFwdH" in arch35
    assert "batchChunkIdx" not in prepare and "batchEnd" in prepare
    assert "ProcessPreparedFullHeadPairBatchArch35" in post_wu
    assert "batchEnd[task] - batchStart[task] == BT_" in post_wu
    assert "ProcessPreparedTailHeadPairArch35" in post_wu
    assert "if (curT < BT_)" in post_wu
    assert "ComputeTailWuVector(" in post_wu


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
    assert "arch35::Run<true" in prepare
    assert "arch35::Run<false" in prepare
    assert "DispatchGeneric<true" in prepare
    assert "DispatchGeneric<false" in prepare
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
    assert "TILING_KEY_VAR == 2UL" in prepare_entry
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);" in prepare_entry
    assert "KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);" in prepare_entry
    assert "KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);" in prepare_entry
    assert "DispatchArch35SafeGate" in prepare_entry
    assert "DispatchGenericSafeGate" in prepare_entry
    assert "ChunkKdaFwdTilingData, 64, 128, 128" in prepare_entry
    assert "ConfigureChunkKdaFwdArch35" in prepare_tiling
    assert "SetTilingKey(useChunk64K128V128Template ? 2 : 1)" in prepare_tiling
    key2_branch = prepare_entry.split("else if (TILING_KEY_IS(2))", 1)[1]
    assert "DispatchArch35SafeGate" in key2_branch
    assert "DispatchGenericSafeGate" in key2_branch
    assert "uint32_t COMPILE_BT = 0" in prepare_impl
    assert "COMPILE_K == 0 ? tiling.kHeadDim : COMPILE_K" in prepare_impl


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
    assert "T, SCORE_T, GK_T, true, true, exportFinalKg, true>" in prepare
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


def test_a5_typical_post_wu_uses_regbase_kg_double_buffer():
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    post_aiv = post_wu.split(
        "__aicore__ inline void ProcessChunkPostAiv", 1
    )[1].split("__aicore__ inline void ProcessChunkPostAic", 1)[0]

    assert '#include "kernel_utils/vector/regbase.hpp"' in post_wu
    assert "KDA_TYPICAL_GATE_TILE_ROWS = 16" in post_wu
    assert "KDA_TYPICAL_GATE_PIPELINE_ROWS = 32" in post_wu
    assert "KDA_TYPICAL_GATE_PIPELINE_STAGES = 3" in post_wu
    assert "ComputePostKdaKgRegbase" in post_wu
    assert "PrefetchTypicalKg(slot ^ 1" in post_wu
    assert "ProcessPostAivPipelineArch35" in post_wu
    assert "PrefetchTypicalKgPipeline" in post_wu
    assert "TypicalGatePipelineRef" in post_wu
    assert "CanPipelineTypicalKg" in post_wu
    assert "if (UseTypicalPostWuGate(curT))" in post_aiv
    assert "ComputeTypicalKg" in post_aiv
    assert "CrossCoreWaitFlagWithReverse" not in post_aiv.split("#else", 1)[0]


def test_a5_post_wu_tail_uses_two_slot_arch35_pipeline_with_bounded_tiles():
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    full_pipeline_dispatch = post_wu.split(
        "__aicore__ inline bool UseFullPostWuPipelineArch35", 1
    )[1].split("__aicore__ inline void ComputePostWuCubeFusedArch35", 1)[0]
    tail_pipeline = post_wu.split(
        "__aicore__ inline void ProcessPreparedTailHeadPairArch35", 1
    )[1].split("__aicore__ inline void ProcessPreparedHeadPairBatchArch35", 1)[0]
    prefetch = post_wu.split(
        "__aicore__ inline void PrefetchPostWuPipelineArch35", 1
    )[1].split("__aicore__ inline void PrefetchPostWuPipelineU", 1)[0]
    bounded_mmad = post_wu.split(
        "__aicore__ inline void ComputePrefetchedPostWuPipelineArch35", 1
    )[1].split("__aicore__ inline void ComputePostWuCube", 1)[0]

    assert "curT == 64" in full_pipeline_dispatch
    assert "InitializePostWuPipelineEvents();" in tail_pipeline
    assert "lane < KDA_POST_HEAD_PAIR_LANES" in tail_pipeline
    assert "start, curT, false" in tail_pipeline
    assert "FinalizePostWuPipelineEvents(KDA_POST_HEAD_PAIR_LANES);" in tail_pipeline
    assert "const uint32_t m = static_cast<uint32_t>(curT);" in prefetch
    assert "const uint32_t k = static_cast<uint32_t>(curT);" in prefetch
    assert "const uint32_t m = static_cast<uint32_t>(curT);" in bounded_mmad
    assert "const uint32_t k = static_cast<uint32_t>(curT);" in bounded_mmad
    assert "copyL0CToDst(blockWOut, tileL0CW);" in bounded_mmad
    assert "copyL0CToDst(blockUOut, tileL0CU);" in bounded_mmad


def test_a5_post_wu_initializes_only_slots_consumed_by_each_full_run():
    post_wu = STAGE_IMPLEMENTATIONS["post_wu"].read_text(encoding="utf-8")
    aic_pipeline = post_wu.split(
        "__aicore__ inline void ProcessPostAicPipelineArch35", 1
    )[1].split("#endif", 1)[0]

    assert "InitializePostWuPipelineSlot(slot);" in aic_pipeline
    assert "InitializePostWuPipelineEvents();" not in aic_pipeline
    assert "if (!reuseSlot) {" in aic_pipeline
    assert "InitializePostWuPipelineSlot(nextSlot);" in aic_pipeline
    assert aic_pipeline.index("InitializePostWuPipelineSlot(nextSlot);") < aic_pipeline.index(
        "PrefetchPostWuPipelineArch35(\n                        resource, nextSlot"
    )


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


def test_a5_finalize_stages_full_chunk_mmads_without_l0c_accumulation():
    finalize = STAGE_IMPLEMENTATIONS["output"].read_text(encoding="utf-8")
    dispatch = finalize.split(
        "__aicore__ inline void ComputeOutputCube(", 1
    )[1].split("using ElementA = T;", 1)[0]
    staged = finalize.split(
        "__aicore__ inline void ComputeOutputCubeStagedArch35", 1
    )[1].split("__aicore__ inline void PrefetchOutputTileArch35", 1)[0]
    writeback = finalize.split(
        "__aicore__ inline void FinalizeOutputRows(", 1
    )[1].split("__aicore__ inline bool ResolveFlatChunk(", 1)[0]

    assert "BT_ == 64 && curT == BT_" in dispatch
    assert "ComputeOutputCubeStagedArch35" in dispatch
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


def test_a5_fwd_h_uses_bounded_single_stage_cube_for_aligned_tail_rows():
    kernel = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h"
        / "op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
    ).read_text(encoding="utf-8")

    assert "MmadPingpongTlaMulti<ArchTag, true, false, 1>" in kernel
    assert "BlockMmadWHTail" in kernel
    assert "BlockMmadKVTail" in kernel
    assert kernel.count("EmptyClass{}, true") == 2
    assert "bool useBoundedMmad = isVariedLen || (seqlen % chunkSize != 0);" in kernel
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
    assert "EVENT_ID3 + (i == 0 ? 0 : pongBaseEvent)" in kernel


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
    assert "finalStateStorageOffset = storeFinalState ? 0" in tiling
    assert "stateElements * sizeof(float)" in tiling
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
    assert "if (!headPairMode_)" in join
    assert "PipeBarrier<PIPE_MTE3>();" in join
    assert "headPairMode_ = KDA_ARCH35_ENABLE_HEAD_PAIR" in prepare
    assert "if (headPairMode_ && !isAivOnly_)" in prepare
    run_after_join = prepare.split(
        "__aicore__ inline void RunAicAfterBothAivReady", 1
    )[1].split("__aicore__ inline void SignalAicSolveReady", 1)[0]
    signal_solve = prepare.split(
        "__aicore__ inline void SignalAicSolveReady", 1
    )[1].split("__aicore__ inline void WaitAicSolveDone", 1)[0]
    score_loop = prepare.split(
        "__aicore__ inline void ProcessChunkPreAivFp32", 1
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


def test_a5_prepare_exports_solved_akk_without_redundant_fp32_round_trip():
    prepare = STAGE_IMPLEMENTATIONS["prepare"].read_text(encoding="utf-8")
    finalize = prepare.split(
        "__aicore__ inline void FinalizePrepareIntermediates", 1
    )[1].split("__aicore__ inline bool ResolveFlatChunk", 1)[0]
    finish_deferred = prepare.split(
        "__aicore__ inline void FinishDeferredSafeChunk", 1
    )[1].split("__aicore__ inline void FinishDeferredSafeChunkPair", 1)[0]
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
    assert "HV_ % KDA_SCORE_LANES != 0" in finish_deferred


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


def test_design_documents_all_internal_template_families_by_interface_layer():
    design = (OP_ROOT / "docs/design.md").read_text(encoding="utf-8")
    for interface_layer in (
        "fla_npu.ops.ascendc.chunk_kda_fwd",
        "aclnnChunkKdaFwd",
        "ChunkKdaFwd L0",
        "chunk_kda_fwd device kernel launch 入口",
    ):
        assert interface_layer in design
    for template_family in (
        "key=1` 通用 shape 模板",
        "key=2` chunk64/K128/V128 模板",
        "standalone Gate",
        "Prepare 内联 Gate",
        "Prepare/Post-WU 阶段内流水融合",
        "Post-WU/FwdH 阶段内流水融合",
        "独立内部 Post-WU 阶段",
        "Prepare arch35 safe-gate head-pair",
        "Post-WU arch35 BF16 sub-16",
        "GDNFwdHTileShapes128",
        "GDNFwdHTileShapes256",
        "ChunkKdaFwdFwdH",
        "Finalize arch35 dense-aligned pipeline",
        "Finalize arch35 BF16 sub-16",
        "ComputeTailWuRegbaseArch35",
        "ComputeTailOutputRegbaseRows",
    ):
        assert template_family in design
    assert "9,600 B" in design
    assert "40,928 B" in design and "40,960 B" in design
    assert "物理入口" not in design
    assert "物理 L0" not in design


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
