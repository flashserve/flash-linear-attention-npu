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
    assert "template <KdaPhase PHASE, typename T" in kernel
    assert "ChunkKdaFwdKernel<KdaPhase::PREPARE, DTYPE_Q" in kernel
    assert "ChunkKdaFwdKernel<KdaPhase::POST_WU, DTYPE_Q" in kernel
    assert "ChunkKdaFwdKernel<KdaPhase::OUTPUT, DTYPE_Q" in kernel
    for runtime_dispatch_field in ("stage", "dataType", "gateDataType"):
        assert runtime_dispatch_field not in tiling


def test_tiling_key_has_design_rationale():
    design = (OP_ROOT / "docs/design.md").read_text(encoding="utf-8")
    assert "模板化方案与 tiling key" in design
    assert "编译期" in design and "语义阶段" in design
