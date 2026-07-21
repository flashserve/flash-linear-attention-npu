"""Static kernel/tiling contract for prepare_wy_repr_bwd_da."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OP_ROOT = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da"


def test_direct_launch_matches_a_real_kernel_entry():
    direct = ROOT / "tests/operators/prepare_wy_repr_bwd_da/routes/test_direct_prepare_wy_repr_bwd_da.cpp"
    kernel_sources = list((OP_ROOT / "op_kernel").glob("*.cpp"))
    assert direct.is_file() and kernel_sources
    text = direct.read_text(encoding="utf-8")
    assert "<<<blockDim" in text and "workspace" in text and "tiling" in text
    assert any("__global__" in source.read_text(encoding="utf-8") for source in kernel_sources)


def test_tiling_key_has_design_rationale():
    design = (OP_ROOT / "docs/design.md").read_text(encoding="utf-8")
    assert "模板化方案与 tiling key" in design
    assert "组合" in design or "不使用 tiling key" in design
