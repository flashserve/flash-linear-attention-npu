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
    for phase in ("PREPARE", "POST_WU", "OUTPUT"):
        assert f"LaunchKdaPhase<KdaPhase::{phase}" in text
    assert "LaunchState" in text
    assert any("__global__" in source.read_text(encoding="utf-8") for source in kernel_sources)


def test_phase_and_dtype_are_compile_time_kernel_parameters():
    kernel = (OP_ROOT / "op_kernel/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    tiling = (OP_ROOT / "op_host/chunk_kda_fwd_tiling.h").read_text(encoding="utf-8")
    assert "template <KdaPhase PHASE, bool SAFE_GATE, typename T" in kernel
    assert "ChunkKdaFwdKernel<KdaPhase::PREPARE, true, DTYPE_Q" in kernel
    assert "ChunkKdaFwdKernel<KdaPhase::PREPARE, false, DTYPE_Q" in kernel
    assert "ChunkKdaFwdKernel<KdaPhase::POST_WU, false, DTYPE_Q" in kernel
    assert "ChunkKdaFwdKernel<KdaPhase::OUTPUT, false, DTYPE_Q" in kernel
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
