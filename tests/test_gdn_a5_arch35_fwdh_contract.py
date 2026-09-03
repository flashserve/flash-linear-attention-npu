from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FWD_H = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
    "op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/"
    "arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
)


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def test_short_tail_h_retires_mte3_before_reusing_accumulator():
    source = FWD_H.read_text(encoding="utf-8")
    tail_h = _between(source, "void ComputeTailHWorkspace", "void Process()")
    ordered_writeback = (
        "gmHWorkspace[offsets.hWorkOffset + kRow * offsets.vBlockDim],\n"
        "                accumUb, offsets.vBlockDim);\n"
        "            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(tailEventId);\n"
        "            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);\n"
        "            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(tailEventId);\n"
        "            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tailEventId);"
    )

    assert ordered_writeback in tail_h
