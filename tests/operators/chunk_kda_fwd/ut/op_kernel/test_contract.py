"""Static kernel/tiling contract for chunk_kda_fwd."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OP_ROOT = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd"
DIRECT_SOURCE = (
    ROOT
    / "examples/fast_kernel_launch_example/csrc/chunk_kda_fwd/chunk_kda_fwd_direct.cpp"
)


def test_direct_launch_matches_a_real_kernel_entry():
    kernel_sources = list((OP_ROOT / "op_kernel").glob("*.cpp"))
    assert DIRECT_SOURCE.is_file() and kernel_sources
    text = DIRECT_SOURCE.read_text(encoding="utf-8")
    assert "<<<blockDim" in text and "workspace" in text and "tiling" in text
    assert "chunk_kda_fwd_direct" in text
    assert "stage" not in text.split("TORCH_LIBRARY_FRAGMENT", 1)[1]
    assert text.count("<<<blockDim") == 1
    assert "ChunkKdaFusedDirectKernel" in text
    assert "RunChunkKdaFused" in text
    assert any("__global__" in source.read_text(encoding="utf-8") for source in kernel_sources)


def test_phase_and_dtype_are_compile_time_kernel_parameters():
    kernel = (OP_ROOT / "op_kernel/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    tiling = (OP_ROOT / "op_host/chunk_kda_fwd_tiling.h").read_text(encoding="utf-8")
    assert "template <KdaPhase PHASE, bool SAFE_GATE, typename T" in kernel
    assert "RunChunkKdaFused<true, DTYPE_Q" in kernel
    assert "RunChunkKdaFused<false, DTYPE_Q" in kernel
    assert "ChunkKdaFwdKernel<KdaPhase::PREPARE, SAFE_GATE" in kernel
    assert "ChunkKdaFwdKernel<KdaPhase::POST_WU, false" in kernel
    assert "GDNFwdHKernel" in kernel and "InitFromData" in kernel
    assert "GDNFwdHTileShapes128, true, false" in kernel
    assert "GDNFwdHTileShapes256, true, false" in kernel
    assert "ChunkKdaFwdKernel<KdaPhase::OUTPUT, false" in kernel
    assert kernel.count("AscendC::SyncAll<false>();") == 3
    assert kernel.count("pipe.Reset();") == 3
    assert kernel.count("ReleaseEventID<HardEvent::V_MTE3>(vToMte3Event_)") == 1
    boundaries = kernel.split("pipe.Reset();")[1:]
    assert len(boundaries) == 3
    assert all(part.lstrip().startswith("AscendC::SyncAll<false>();") for part in boundaries)
    assert "tilingData.safeGate" in kernel
    for runtime_dispatch_field in ("stage", "dataType", "gateDataType"):
        assert runtime_dispatch_field not in tiling


def test_safe_gate_is_supported_across_public_and_direct_routes():
    aclnn_runtime = (
        ROOT / "torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"
    ).read_text(encoding="utf-8")
    legacy_runtime = (
        ROOT / "torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp"
    ).read_text(encoding="utf-8")
    direct = DIRECT_SOURCE.read_text(encoding="utf-8")
    op_def = (OP_ROOT / "op_host/chunk_kda_fwd_def.cpp").read_text(encoding="utf-8")
    tiling = (OP_ROOT / "op_host/chunk_kda_fwd_tiling.cpp").read_text(encoding="utf-8")

    assert "safe_gate is reserved" not in aclnn_runtime
    assert "safe_gate=true is not supported" not in legacy_runtime
    assert "ctypes.c_bool(safe_gate)" in aclnn_runtime
    chunk_kda_abi = aclnn_runtime.split('"aclnnChunkKdaFwd": [', 1)[1].split("],", 1)[0]
    assert "ctypes.c_bool,\n        ctypes.c_bool,\n        ctypes.c_int64" in chunk_kda_abi
    assert "bool safe_gate=False" in direct
    assert 'Attr("safe_gate").AttrType(REQUIRED).Bool(false)' in op_def
    assert "tiling.set_safeGate(safeGate);" in tiling


def test_tiling_key_has_design_rationale():
    design = (OP_ROOT / "docs/design.md").read_text(encoding="utf-8")
    assert "模板化方案与 tiling key" in design
    assert "编译期" in design and "语义阶段" in design


def test_host_uses_batch_schedule_for_cross_phase_barriers():
    tiling = (OP_ROOT / "op_host/chunk_kda_fwd_tiling.cpp").read_text(encoding="utf-8")
    assert "SetScheduleMode(KDA_BATCH_MODE)" in tiling


def test_embedded_fwd_h_uses_a_stage_local_flag_protocol():
    schedulers = [
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel"
        / relative
        for relative in (
            "gemm/block/block_scheduler_gdn_fwd_h.hpp",
            "arch35/gemm/block/block_scheduler_gdn_fwd_h.hpp",
        )
    ]
    for scheduler in schedulers:
        text = scheduler.read_text(encoding="utf-8")
        assert "UseEmbeddedStageFlags" in text
        assert "cube2Done[0].id = 0" in text and "cube2Done[1].id = 1" in text
        assert "vec1Done[0].id = 6" in text and "vec1Done[1].id = 7" in text
        assert text.count("UseEmbeddedStageFlags();") == 2


def test_internal_layout_reuses_requested_intermediate_outputs():
    aclnn = (OP_ROOT / "op_host/op_api/aclnn_chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    assert "outputs must be all present or all empty" in aclnn
    assert aclnn.count("result = l0op::KdaChunkForward(") == 1
    assert "l0op::ChunkGatedDeltaRuleFwdH" not in aclnn
    allocation = aclnn.split("if (isInternalLayout) {", 2)[2].split(
        "result = l0op::KdaChunkForward", 1
    )[0]
    assert "if (!returnIntermediates)" in allocation
    assert "} else {\n        aqkComputeBnst = executorPtr->AllocTensor" in allocation
    assert "returnIntermediates && result[2] != aqkBnst" in aclnn


def test_embedded_fwd_h_compiles_out_scalar_gate_math():
    epilogue_root = (
        ROOT
        / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel"
    )
    for arch in ("", "arch35/"):
        vnew = (epilogue_root / f"{arch}epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp").read_text(
            encoding="utf-8"
        )
        update = (epilogue_root / f"{arch}epilogue/block/block_epilogue_gdn_fwdh_update.hpp").read_text(
            encoding="utf-8"
        )
        assert "if constexpr (!scalarGated)" in vnew
        assert "Duplicate<float>(gUbTensor, 1.0f, mActual)" in vnew
        assert "if constexpr (scalarGated)" in update
        assert "float muls = 1.0f" in update
