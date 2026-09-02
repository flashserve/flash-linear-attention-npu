from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
FWD_H = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/"
    "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
)


def _partial_call(source: str, marker: str) -> str:
    call = source[source.index(marker) :]
    return call[: call.index("} else {")]


def test_partial_fwdh_cube_calls_rely_on_actual_shape_not_full_l1_clear():
    source = FWD_H.read_text(encoding="utf-8")
    c1 = _partial_call(source, "if (cube1Offsets.blockTokens < chunkSize)")
    c2 = _partial_call(source, "if (cube2Offsets.blockTokens < chunkSize)")

    assert "blockMmadWHTail(" in c1
    assert "cube1Shape);" in c1
    assert "EmptyClass{}, true" not in c1
    assert "blockMmadKVTail(" in c2
    assert "cube2Shape);" in c2
    assert "EmptyClass{}, true" not in c2
