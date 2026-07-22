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
    assert "__global__ __aicore__ void" in kernel_text and "recurrent_kda(" in kernel_text
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


def test_kernel_validates_device_metadata_before_state_access():
    text = (OP_ROOT / "op_kernel/recurrent_kda.h").read_text(encoding="utf-8")

    empty_skip = text.index("if (seqLen64 == 0)")
    slot_validation = text.index("ValidateStateSlots(batch_i, seq0, seqLen)")
    state_prefetch = text.index("PrefetchState(nextStateOffset, nextSingleV)")
    assert empty_skip < slot_validation < state_prefetch
    assert "batchIdx * ssmStateStride_" in text
    assert "stateSlot >= static_cast<int64_t>(stateCapacity_)" in text
