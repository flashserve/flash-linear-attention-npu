"""Static L2 cache-hint contract for ChunkKdaFwd stage handoffs."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
PREPARE_KERNEL = (
    ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_prepare.h"
)
GATE_KERNEL = (
    ROOT
    / "fla/ops/ascendc/kda/kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h"
)


def test_gate_input_bypasses_l2_only_on_a5_while_gk_handoff_stays_normal():
    source = GATE_KERNEL.read_text(encoding="utf-8")
    init = source.split("__aicore__ inline void Init", 1)[1].split(
        "__aicore__ inline void Process", 1
    )[0]

    assert "defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)" in init
    assert "g_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)" in init
    assert "SetL2CacheHint<CacheRwMode" not in init
    assert "CacheMode::CACHE_MODE_DISABLE" in init
    assert "gk_.SetL2CacheHint" not in init
    assert "gk_.template SetL2CacheHint" not in init


def test_prepare_bypasses_only_proven_single_read_a5_inputs():
    source = PREPARE_KERNEL.read_text(encoding="utf-8")
    init = source.split("__aicore__ inline void Init", 1)[1].split(
        "__aicore__ inline void ProcessAivOnly", 1
    )[0]

    assert "defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)" in init
    assert "if (computeGateInPrepare_)" in init
    assert "rawG_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)" in init
    non_gva = init.split("if (H_ == HV_)", 1)[1].split("#endif", 1)[0]
    assert "q_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)" in non_gva
    assert "SetL2CacheHint<CacheRwMode" not in init
    for reused in (
        "v_",
        "beta_",
        "k_",
        "gk_",
        "preparedQG_",
        "preparedAqk_",
        "w_",
        "u_",
        "kg_",
    ):
        assert f"{reused}.SetL2CacheHint" not in init
        assert f"{reused}.template SetL2CacheHint" not in init
