"""Static kernel/tiling contract for recurrent_kda."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OP_ROOT = ROOT / "fla/ops/ascendc/kda/recurrent_kda"


def test_direct_launch_matches_a_real_kernel_entry():
    direct = ROOT / "tests/operators/recurrent_kda/routes/test_direct_recurrent_kda.cpp"
    kernel = OP_ROOT / "op_kernel/recurrent_kda.cpp"
    assert direct.is_file() and kernel.is_file()
    direct_text = direct.read_text(encoding="utf-8")
    kernel_text = kernel.read_text(encoding="utf-8")
    assert "recurrent_kda<<<blockDim" in direct_text
    assert "__global__ __aicore__ void recurrent_kda" in kernel_text
    assert "numAcceptedTokens" in direct_text and "finalState" in direct_text


def test_tiling_key_has_design_rationale():
    design = (OP_ROOT / "docs/design.md").read_text(encoding="utf-8")
    assert "模板化方案与 tiling key" in design
    assert "不引入额外 tiling key" in design


def test_kernel_is_not_wired_to_gdn_recurrent_interface():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            OP_ROOT / "op_kernel/recurrent_kda.cpp",
            OP_ROOT / "op_host/op_api/aclnn_recurrent_kda.cpp",
        ]
    )
    assert "RecurrentGatedDeltaRule" not in text
    assert "aclnnRecurrentGatedDeltaRule" not in text
