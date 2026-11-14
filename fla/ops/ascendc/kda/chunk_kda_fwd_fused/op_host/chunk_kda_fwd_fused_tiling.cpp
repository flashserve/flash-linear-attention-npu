#include "chunk_kda_fwd_fused_tiling.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <register/op_impl_registry.h>
// 旧版 opbase 的 pkg_inc 里有一份残缺的 op_common/log/log.h（缺 OP_LOGE_FOR_INVALID_*
// 家族且 OP_LOGE_WITH_INVALID_INPUT 有拼写缺陷），与 toolkit 版共用同一 include guard；
// 短路径 "log/log.h" 经 -I.../include/op_common 稳定命中 toolkit 完整版。
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_V_IDX = 2;
constexpr size_t INPUT_G_IDX = 3;
constexpr size_t INPUT_BETA_IDX = 4;
constexpr size_t INPUT_A_LOG_IDX = 5;
constexpr size_t INPUT_INITIAL_STATE_IDX = 7;
constexpr size_t INPUT_CU_SEQLENS_IDX = 8;

constexpr size_t OUTPUT_FINAL_STATE_IDX = 1;

constexpr size_t ATTR_LAYOUT_IDX = 0;
constexpr size_t ATTR_SCALE_IDX = 1;
constexpr size_t ATTR_LOWER_BOUND_IDX = 2;

constexpr uint64_t KDA_FD32_MAX_VARLEN_SEQ = 1024;
constexpr uint64_t KDA_FD32_CHUNK_TOKENS = 128;
constexpr uint64_t KDA_FD32_HEAD_DIM = 128;
constexpr float KDA_FD32_EPSILON = 1.0e-6F;

uint64_t Fp32BitsToBf16Pattern(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint64_t>(bits >> 16);
}

uint64_t Fp32Bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint64_t>(bits);
}
} // namespace

ge::graphStatus Tiling4ChunkKdaFwdFused(gert::TilingContext *context)
{
    const auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    const auto attrs = context->GetAttrs();
    if (qDesc == nullptr || attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), qDesc == nullptr ? "q" : "attrs");
        return ge::GRAPH_FAILED;
    }
    const float scale = static_cast<float>(*attrs->GetAttrPointer<double>(ATTR_SCALE_IDX));
    const float lowerBound = *attrs->GetAttrPointer<float>(ATTR_LOWER_BOUND_IDX);
    const char *layout = attrs->GetAttrPointer<char>(ATTR_LAYOUT_IDX);

    const auto cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
    const bool isVarlen = cuTensor != nullptr;
    const auto qShape = context->GetInputShape(INPUT_Q_IDX)->GetStorageShape();
    const auto vShape = context->GetInputShape(INPUT_V_IDX)->GetStorageShape();
    const auto aLogShape = context->GetInputShape(INPUT_A_LOG_IDX)->GetStorageShape();
    if (qShape.GetDimNum() != 3 || vShape.GetDimNum() != 3 || aLogShape.GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), "q/v/a_log rank",
            std::to_string(qShape.GetDimNum()) + "/" + std::to_string(vShape.GetDimNum()) +
                "/" + std::to_string(aLogShape.GetDimNum()),
            "3/3/1");
        return ge::GRAPH_FAILED;
    }
    const bool headMajorVarlen = isVarlen && layout != nullptr && std::strcmp(layout, "NTD") == 0;
    const uint64_t headNum = headMajorVarlen ? static_cast<uint64_t>(qShape.GetDim(0))
                                             : isVarlen ? static_cast<uint64_t>(qShape.GetDim(1))
                                                        : static_cast<uint64_t>(qShape.GetDim(0));
    const uint64_t seqLen = headMajorVarlen ? static_cast<uint64_t>(qShape.GetDim(1))
                                            : isVarlen ? static_cast<uint64_t>(qShape.GetDim(0))
                                                       : static_cast<uint64_t>(qShape.GetDim(1));
    const uint64_t kDim = static_cast<uint64_t>(qShape.GetDim(2));
    const uint64_t hvNum = headMajorVarlen ? static_cast<uint64_t>(vShape.GetDim(0))
                                           : isVarlen ? static_cast<uint64_t>(vShape.GetDim(1))
                                                      : static_cast<uint64_t>(vShape.GetDim(0));
    const uint64_t hvReal = static_cast<uint64_t>(aLogShape.GetDim(0));
    if (kDim != KDA_FD32_HEAD_DIM) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "K", std::to_string(kDim), "128");
        return ge::GRAPH_FAILED;
    }
    if (seqLen == 0) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "T", "0", ">0");
        return ge::GRAPH_FAILED;
    }
    if (headNum == 0 || hvNum < headNum || hvNum % headNum != 0) {
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), "H/HV", std::to_string(headNum) + "/" + std::to_string(hvNum),
            "H>0 and HV a positive multiple of H");
        return ge::GRAPH_FAILED;
    }
    if (hvReal == 0 || hvNum % hvReal != 0) {
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), "HV/aLogHV", std::to_string(hvNum) + "/" + std::to_string(hvReal),
            "aLogHV>0 and HV%aLogHV=0");
        return ge::GRAPH_FAILED;
    }

    uint64_t nSeq = 0;
    uint64_t cuData[KDA_FD32_MAX_VARLEN_SEQ + 1] = {};
    if (isVarlen) {
        nSeq = static_cast<uint64_t>(cuTensor->GetStorageShape().GetDim(0)) - 1;
        const int64_t *cu = cuTensor->GetData<int64_t>();
        if (cu == nullptr || nSeq == 0 || nSeq > KDA_FD32_MAX_VARLEN_SEQ || cu[0] != 0 ||
            static_cast<uint64_t>(cu[nSeq]) != seqLen) {
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "cu_seqlens", "nSeq=" + std::to_string(nSeq),
                "expected 1..1024 sequences with cu[0]=0 and cu[nSeq]=T=" +
                    std::to_string(seqLen));
            return ge::GRAPH_FAILED;
        }
        for (uint64_t i = 0; i <= nSeq; ++i) {
            if (i > 0 && cu[i] < cu[i - 1]) {
                OP_LOGE_FOR_INVALID_VALUE(
                    context->GetNodeName(), "cu_seqlens",
                    "decreasing at index " + std::to_string(i), "must be nondecreasing");
                return ge::GRAPH_FAILED;
            }
            cuData[i] = static_cast<uint64_t>(cu[i]);
        }
    }
    const uint64_t stateHeads = isVarlen ? nSeq * hvNum : hvNum;
    const uint64_t taskCount = stateHeads;

    const bool hasInitialState = context->GetOptionalInputDesc(INPUT_INITIAL_STATE_IDX) != nullptr;
    if (hasInitialState) {
        const auto isShape = context->GetOptionalInputShape(INPUT_INITIAL_STATE_IDX)->GetStorageShape();
        if (isShape.GetDimNum() != 3 ||
            static_cast<uint64_t>(isShape.GetDim(0)) != stateHeads ||
            static_cast<uint64_t>(isShape.GetDim(1)) != KDA_FD32_HEAD_DIM ||
            static_cast<uint64_t>(isShape.GetDim(2)) != KDA_FD32_HEAD_DIM) {
            OP_LOGE_FOR_INVALID_SHAPE(
                context->GetNodeName(), "initial_state",
                "rank=" + std::to_string(isShape.GetDimNum()) + ",dims=[" +
                    std::to_string(isShape.GetDim(0)) + "," + std::to_string(isShape.GetDim(1)) +
                    "," + std::to_string(isShape.GetDim(2)) + "]",
                "[" + std::to_string(stateHeads) + ",128,128]");
            return ge::GRAPH_FAILED;
        }
    }

    bool outputFinalState = false;
    const auto fsInfo = context->GetIrOutputInstanceInfo(OUTPUT_FINAL_STATE_IDX);
    if (fsInfo != nullptr && fsInfo->GetInstanceNum() != 0) {
        const auto fsShape = context->GetOutputShape(fsInfo->GetInstanceStart());
        if (fsShape != nullptr && fsShape->GetStorageShape().GetShapeSize() != 1) {
            const auto &fs = fsShape->GetStorageShape();
            if (fs.GetDimNum() != 3 ||
                static_cast<uint64_t>(fs.GetDim(0)) != stateHeads ||
                static_cast<uint64_t>(fs.GetDim(1)) != KDA_FD32_HEAD_DIM ||
                static_cast<uint64_t>(fs.GetDim(2)) != KDA_FD32_HEAD_DIM) {
                OP_LOGE_FOR_INVALID_SHAPE(
                    context->GetNodeName(), "final_state",
                    "rank=" + std::to_string(fs.GetDimNum()) + ",dims=[" +
                        std::to_string(fs.GetDim(0)) + "," + std::to_string(fs.GetDim(1)) +
                        "," + std::to_string(fs.GetDim(2)) + "]",
                    "[" + std::to_string(stateHeads) + ",128,128]");
                return ge::GRAPH_FAILED;
            }
            outputFinalState = true;
        }
    }

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t aicCoreNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAic());
    const uint64_t blockNum = taskCount < aicCoreNum ? taskCount : aicCoreNum;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0;
    }

    ChunkKdaFwdFusedTilingData tiling;
    tiling.set_nChunks((seqLen + KDA_FD32_CHUNK_TOKENS - 1) / KDA_FD32_CHUNK_TOKENS);
    tiling.set_scaleBf16(Fp32BitsToBf16Pattern(scale));
    tiling.set_gateScaleBits(Fp32Bits(lowerBound));
    tiling.set_headNum(headNum);
    tiling.set_seqLen(seqLen);
    tiling.set_hasInitialState(hasInitialState ? 1 : 0);
    tiling.set_outputFinalState(outputFinalState ? 1 : 0);
    tiling.set_qkGroup(hvNum / headNum);
    tiling.set_taskCount(taskCount);
    tiling.set_hvNumReal(hvReal);
    tiling.set_blockNum(blockNum);
    tiling.set_varlenMode(headMajorVarlen ? 2 : (isVarlen ? 1 : 0));
    tiling.set_epsilonBits(Fp32Bits(KDA_FD32_EPSILON));
    const uint64_t gateFlags = (context->GetInputDesc(INPUT_G_IDX)->GetDataType() == ge::DT_FLOAT ? 1 : 0) |
                               (context->GetInputDesc(INPUT_BETA_IDX)->GetDataType() == ge::DT_FLOAT ? 2 : 0);
    tiling.set_gateFlags(gateFlags);
    tiling.set_cuData(cuData);
    if (isVarlen) {
        tiling.set_nChunks(0);
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaFwdFused(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwdFused)
    .Tiling(Tiling4ChunkKdaFwdFused)
    .TilingParse<ChunkKdaFwdFusedCompileInfo>(TilingPrepare4ChunkKdaFwdFused);
} // namespace optiling
