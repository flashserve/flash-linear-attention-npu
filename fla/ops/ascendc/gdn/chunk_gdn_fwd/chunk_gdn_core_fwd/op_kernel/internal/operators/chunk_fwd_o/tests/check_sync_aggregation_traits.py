#!/usr/bin/env python3
"""静态检查 FwdO A5 mode2 聚合同步开关的依赖闭环。"""

from __future__ import annotations

import re
from pathlib import Path


KERNEL = (
    Path(__file__).resolve().parents[1]
    / "op_kernel"
    / "gemm"
    / "kernel"
    / "gdn_fwd_o_kernel.hpp"
)
OUTPUT_EPILOGUE = (
    Path(__file__).resolve().parents[1]
    / "op_kernel"
    / "arch35"
    / "epilogue"
    / "block"
    / "block_epilogue_gdn_fwdo_output.hpp"
)


def compact(source: str) -> str:
    """去除空白以便检查多行 C++ 表达式的结构。"""
    return re.sub(r"\s+", "", source)


def check_trait_interface(kernel: str) -> None:
    signature = compact(kernel[kernel.index("template<") : kernel.index(">\nclass GDNFwdOKernel")])
    assert "boolkChunkPipeline=false" in signature
    assert "boolkFwdOAggregateQkMaskBarrier=false" in signature
    assert "boolkFwdOAggregateOutputBarrier=false" in signature
    assert signature.index("kFwdOAggregateQkMaskBarrier") < signature.index(
        "kFwdOAggregateOutputBarrier"
    )


def check_qkmask_mode2_path(kernel: str) -> None:
    begin = kernel.index("EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask")
    end = kernel.index("// AscendC::PipeBarrier<PIPE_ALL>();", begin)
    block = kernel[begin:end]
    assert "if constexpr (!kFwdOAggregateQkMaskBarrier)" in block
    assert block.count("CrossCoreBarrier<0x1, PIPE_MTE3>()") == 2, (
        "A5 默认路径和非 A5 路径应各保留一个 barrier"
    )
    publish = "CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done[streamId])"
    assert block.count(publish) == 1
    assert block.index(publish) > block.index(
        "if constexpr (!kFwdOAggregateQkMaskBarrier)"
    ), "mode2 两个 AIV 的 MTE3 发布不能被开关一起删除"


def check_output_mode2_path(kernel: str) -> None:
    begin = kernel.index("EpilogueGDNFwdOOutput epilogueGDNFwdOOutput")
    end = kernel.index("needRun = true;", begin)
    block = compact(kernel[begin:end])
    assert (
        "(isVariedLen==0||kFwdOAggregateOutputBarrier)"
        "?&vecBlockScheduler.vec2Done[streamId]:nullptr"
    ) in block
    assert "ifconstexpr(!kFwdOAggregateOutputBarrier)" in block
    assert block.count("CrossCoreBarrier<0x1,PIPE_MTE3>()") == 2
    assert block.count(
        "CrossCoreSetFlag<0x2,PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId])"
    ) == 2, "A5 默认与非 A5 外层发布必须保留，但聚合路径不得再发布"


def check_output_epilogue_publish_points(epilogue: str) -> None:
    compacted = compact(epilogue)
    publish = "CrossCoreSetFlag<0x2,PIPE_MTE2>(*setFlag)"
    assert compacted.count(publish) == 4, (
        "wide 零行、wide 最后 tile、narrow 单 stage 和 narrow 双 stage "
        "都必须发布 vec2Done"
    )
    assert (
        "if(rowBegin>=mActual){"
        "if(waitFlag)Arch::CrossCoreWaitFlag(*waitFlag);"
        "if(setFlag)Arch::" + publish
    ) in compacted, "零行 AIV 也必须先消费 cube3Done，再参与 mode2 聚合"
    assert (
        "DataCopy(aUbTensor,attnInputThisTile,rowsThisTile*nActual);"
        "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2+pingpongFlag);"
        "if(setFlag&&rowStart+rowsThisTile>=rowEnd){Arch::" + publish
    ) in compacted
    assert (
        "DataCopy(aUbTensor,attnInputThisSubBlock,mActualThisSubBlock*nActual);"
        "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2+pingpongFlag);"
        "if(setFlag)Arch::" + publish
    ) in compacted
    assert (
        "DataCopy(aUbTensor,attnInputThisSubBlock,mActualThisStage*nActual);"
        "AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2+pingpongFlag);"
        "if(setFlag&&stage==1){Arch::" + publish
    ) in compacted


def main() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    epilogue = OUTPUT_EPILOGUE.read_text(encoding="utf-8")
    check_trait_interface(kernel)
    check_qkmask_mode2_path(kernel)
    check_output_mode2_path(kernel)
    check_output_epilogue_publish_points(epilogue)
    print("FwdO sync aggregation trait check: PASS")


if __name__ == "__main__":
    main()
