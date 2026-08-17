#include "chunk_kda_fwd_tiling.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <vector>
#include <register/op_impl_registry.h>
#include "arch35/chunk_kda_fwd_tiling_impl.h"
#include "../op_kernel/chunk_kda_fwd_plan.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_V_IDX = 2;
constexpr size_t INPUT_G_IDX = 3;
constexpr size_t INPUT_A_LOG_IDX = 5;
constexpr size_t INPUT_DT_BIAS_IDX = 6;
constexpr size_t INPUT_INITIAL_STATE_IDX = 7;
constexpr size_t INPUT_CU_SEQLENS_IDX = 8;
constexpr size_t INPUT_CHUNK_INDICES_IDX = 9;

constexpr size_t OUTPUT_FINAL_STATE_IDX = 1;
constexpr size_t OUTPUT_GK_IDX = 2;
constexpr size_t OUTPUT_W_IDX = 5;
constexpr size_t OUTPUT_U_IDX = 6;
constexpr size_t OUTPUT_QG_IDX = 7;
constexpr size_t OUTPUT_KG_IDX = 8;
constexpr size_t OUTPUT_V_NEW_IDX = 9;
constexpr size_t OUTPUT_H_IDX = 10;

constexpr size_t ATTR_LAYOUT_IDX = 0;
constexpr size_t ATTR_SCALE_IDX = 1;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 2;
constexpr size_t ATTR_SAFE_GATE_IDX = 3;
constexpr size_t ATTR_LOWER_BOUND_IDX = 4;
constexpr size_t ATTR_USE_GATE_IDX = 5;

constexpr uint64_t KDA_ALIGN = 512;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SOLVE_PIPELINE_DEPTH = 4;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 4;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint64_t KDA_GDN_PIPELINE_DEPTH = 2;
// 每个输出 descriptor 含 state/local 两个平面；kernel 逐 head 发布并以
// mode2 completion 回收两个 descriptor 槽，因此总共分配 2 x 2 个矩阵。
constexpr uint64_t KDA_OUTPUT_SLOT_DEPTH = 2;
constexpr uint64_t KDA_OUTPUT_SCRATCH_PLANES = 2;
constexpr uint32_t KDA_BATCH_MODE = 1;
constexpr int64_t KDA_PARAM_DTYPE_BF16 = 1;
constexpr int64_t KDA_PARAM_DTYPE_FP32 = 2;

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedProduct(std::initializer_list<uint64_t> factors, uint64_t &result)
{
    result = 1;
    for (uint64_t factor : factors) {
        if (!CheckedMul(result, factor, result)) {
            return false;
        }
    }
    return true;
}

bool CheckedAlign(uint64_t bytes, uint64_t &aligned)
{
    uint64_t rounded = 0;
    if (!CheckedAdd(bytes, KDA_ALIGN - 1, rounded)) {
        return false;
    }
    aligned = rounded / KDA_ALIGN * KDA_ALIGN;
    return true;
}

bool AllocateWorkspace(uint64_t &cursor, uint64_t bytes, uint64_t &offset)
{
    return CheckedAlign(cursor, offset) && CheckedAdd(offset, bytes, cursor);
}

bool HasOutput(gert::TilingContext *context, size_t index)
{
    const auto instanceInfo = context->GetIrOutputInstanceInfo(index);
    if (instanceInfo == nullptr || instanceInfo->GetInstanceNum() == 0) {
        return false;
    }
    const auto outputShape = context->GetOutputShape(instanceInfo->GetInstanceStart());
    return outputShape != nullptr &&
        outputShape->GetStorageShape().GetShapeSize() != 1;
}

struct ShapeInfo {
    int64_t rank = 0;
    int64_t batch = 0;
    int64_t seqlen = 0;
    int64_t qHeads = 0;
    int64_t vHeads = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
    bool sequenceMajor = false;
};

struct SequencePlanInfo {
    bool isVarLen = false;
    int64_t seqNum = 0;
    int64_t totalChunks = 0;
    bool hasVarlenTail = false;
    uint32_t chunkSize = 0;
    uint32_t denseTailTokens = 0;
    uint32_t totalFullChunks = 0;
    uint32_t totalTailChunks = 0;
    std::vector<uint32_t> seqChunkOffsets;
    std::vector<uint32_t> fullChunkCounts;
    std::vector<uint16_t> alignedSequenceIds;
    std::vector<uint16_t> tailedSequenceIds;
};

bool AlignPlanOffset(uint64_t offset, uint64_t alignment, uint64_t &aligned)
{
    if (alignment == 0) {
        return false;
    }
    uint64_t rounded = 0;
    if (!CheckedAdd(offset, alignment - 1, rounded)) {
        return false;
    }
    aligned = rounded / alignment * alignment;
    return true;
}

template <typename T>
bool AppendPlanVector(std::vector<uint8_t> &payload, const std::vector<T> &values,
                      uint32_t &encodedOffset)
{
    uint64_t offset = 0;
    uint64_t vectorBytes = 0;
    uint64_t payloadEnd = 0;
    if (!AlignPlanOffset(static_cast<uint64_t>(payload.size()), alignof(T), offset) ||
        !CheckedMul(static_cast<uint64_t>(values.size()), sizeof(T), vectorBytes) ||
        !CheckedAdd(offset, vectorBytes, payloadEnd) ||
        offset > std::numeric_limits<uint32_t>::max() ||
        payloadEnd > std::numeric_limits<uint32_t>::max() ||
        payloadEnd > static_cast<uint64_t>(payload.max_size())) {
        return false;
    }
    encodedOffset = static_cast<uint32_t>(offset);
    payload.resize(static_cast<size_t>(payloadEnd));
    if (!values.empty()) {
        std::memcpy(payload.data() + encodedOffset, values.data(),
                    static_cast<size_t>(vectorBytes));
    }
    return true;
}

bool BuildCompactSequencePlan(const SequencePlanInfo &sequenceInfo,
                              uint32_t blockDim, uint32_t queryHeads,
                              uint32_t valueHeads,
                              std::vector<uint8_t> &payload)
{
    const uint64_t logicalChunkCount =
        static_cast<uint64_t>(sequenceInfo.totalFullChunks) +
        sequenceInfo.totalTailChunks;
    if (sequenceInfo.seqNum <= 0 ||
        sequenceInfo.seqNum > std::numeric_limits<uint32_t>::max() ||
        (sequenceInfo.isVarLen && sequenceInfo.seqNum > 1024) ||
        sequenceInfo.totalChunks <= 0 ||
        sequenceInfo.totalChunks > std::numeric_limits<uint32_t>::max() ||
        logicalChunkCount == 0 ||
        logicalChunkCount > std::numeric_limits<uint32_t>::max() ||
        blockDim == 0 || queryHeads == 0 || valueHeads < queryHeads ||
        valueHeads % queryHeads != 0 ||
        valueHeads > std::numeric_limits<uint16_t>::max()) {
        return false;
    }
    const uint32_t headGroupCount = KdaForward::ComputeChunkHeadGroupCount(
        sequenceInfo.totalFullChunks, sequenceInfo.totalTailChunks, blockDim,
        queryHeads, valueHeads);
    if (headGroupCount == 0) {
        return false;
    }

    KdaForward::CompactSequencePlanHeader header{};
    header.magic = KdaForward::KDA_COMPACT_PLAN_MAGIC;
    header.version = KdaForward::KDA_COMPACT_PLAN_VERSION;
    header.kind = static_cast<uint32_t>(sequenceInfo.isVarLen
        ? KdaForward::CompactPlanKind::VARLEN_INDEXED
        : KdaForward::CompactPlanKind::DENSE_AFFINE);
    header.sequenceCount = static_cast<uint32_t>(sequenceInfo.seqNum);
    header.totalChunks = sequenceInfo.isVarLen
        ? static_cast<uint32_t>(sequenceInfo.totalChunks)
        : sequenceInfo.totalFullChunks + sequenceInfo.totalTailChunks;
    header.totalFullChunks = sequenceInfo.totalFullChunks;
    header.totalTailChunks = sequenceInfo.totalTailChunks;
    header.chunkSize = sequenceInfo.chunkSize;
    header.alignedSequenceCount = sequenceInfo.isVarLen
        ? static_cast<uint32_t>(sequenceInfo.alignedSequenceIds.size())
        : (sequenceInfo.denseTailTokens == 0
            ? static_cast<uint32_t>(sequenceInfo.seqNum) : 0);
    header.tailedSequenceCount = sequenceInfo.isVarLen
        ? static_cast<uint32_t>(sequenceInfo.tailedSequenceIds.size())
        : (sequenceInfo.denseTailTokens != 0
            ? static_cast<uint32_t>(sequenceInfo.seqNum) : 0);
    const uint64_t groupedTaskCount = logicalChunkCount * headGroupCount;
    if (groupedTaskCount > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    header.chunkUsedCoreNum = static_cast<uint32_t>(
        std::min<uint64_t>(blockDim, groupedTaskCount));
    header.headGroupCount = headGroupCount;
    // 版本6保留该字段以维持21-word布局；runtime窗口由H/HV统一推导，
    // host不再发布pair/single模板协议。
    header.chunkStageFlags = 0;
    header.fwdUsedCoreNum = std::min(blockDim, valueHeads);
    if (!sequenceInfo.isVarLen) {
        header.denseFullChunksPerSequence = sequenceInfo.fullChunkCounts.empty()
            ? 0
            : sequenceInfo.fullChunkCounts.front();
        header.denseTailTokens = sequenceInfo.denseTailTokens;
    }

    payload.assign(sizeof(header), 0);
    if (sequenceInfo.isVarLen) {
        if (!AppendPlanVector(payload, sequenceInfo.seqChunkOffsets,
                              header.seqChunkOffsetsOffset) ||
            !AppendPlanVector(payload, sequenceInfo.alignedSequenceIds,
                              header.alignedSequenceIdsOffset) ||
            !AppendPlanVector(payload, sequenceInfo.tailedSequenceIds,
                              header.tailedSequenceIdsOffset)) {
            return false;
        }
    }

    const uint64_t fullTaskCount =
        static_cast<uint64_t>(header.totalFullChunks) * headGroupCount;
    const uint64_t tailTaskCount =
        static_cast<uint64_t>(header.totalTailChunks) * headGroupCount;
    std::vector<KdaForward::ChunkCoreCursor> cursors(
        header.chunkUsedCoreNum);
    for (uint32_t core = 0; core < header.chunkUsedCoreNum; ++core) {
        auto &cursor = cursors[core];
        const uint64_t taskBegin =
            groupedTaskCount * core / header.chunkUsedCoreNum;
        const uint64_t taskEnd =
            groupedTaskCount * (core + 1) / header.chunkUsedCoreNum;
        cursor.fullBegin = static_cast<uint32_t>(
            std::min(taskBegin, fullTaskCount));
        cursor.fullEnd = static_cast<uint32_t>(
            std::min(taskEnd, fullTaskCount));
        cursor.tailBegin = static_cast<uint32_t>(
            taskBegin > fullTaskCount ? taskBegin - fullTaskCount : 0);
        cursor.tailEnd = static_cast<uint32_t>(
            taskEnd > fullTaskCount ? taskEnd - fullTaskCount : 0);
        cursor.tailBegin = std::min(
            cursor.tailBegin, static_cast<uint32_t>(tailTaskCount));
        cursor.tailEnd = std::min(
            cursor.tailEnd, static_cast<uint32_t>(tailTaskCount));

        const uint32_t fullChunkBegin = cursor.fullBegin / headGroupCount;
        uint32_t sequence = 0;
        uint32_t localChunk = 0;
        if (!sequenceInfo.isVarLen) {
            const uint32_t fullChunksPerSequence =
                sequenceInfo.fullChunkCounts.empty()
                    ? 0 : sequenceInfo.fullChunkCounts.front();
            if (fullChunksPerSequence != 0) {
                sequence = fullChunkBegin / fullChunksPerSequence;
                localChunk = fullChunkBegin % fullChunksPerSequence;
            } else {
                sequence = static_cast<uint32_t>(sequenceInfo.seqNum);
            }
        } else {
            uint32_t prefix = 0;
            while (sequence < sequenceInfo.fullChunkCounts.size() &&
                   prefix + sequenceInfo.fullChunkCounts[sequence] <=
                       fullChunkBegin) {
                prefix += sequenceInfo.fullChunkCounts[sequence];
                ++sequence;
            }
            localChunk = fullChunkBegin - prefix;
        }
        cursor.fullStartSequence = sequence;
        cursor.fullStartLocalChunk = localChunk;
    }
    if (!AppendPlanVector(payload, cursors, header.chunkCoreCursorsOffset) ||
        payload.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    header.payloadBytes = static_cast<uint32_t>(payload.size());
    std::memcpy(payload.data(), &header, sizeof(header));
    return true;
}

bool ResolveShape(gert::TilingContext *context, const char *layout, ShapeInfo &info)
{
    const auto qShapePtr = context->GetInputShape(INPUT_Q_IDX);
    const auto vShapePtr = context->GetInputShape(INPUT_V_IDX);
    if (qShapePtr == nullptr || vShapePtr == nullptr || layout == nullptr) {
        return false;
    }
    const auto &qShape = qShapePtr->GetStorageShape();
    const auto &vShape = vShapePtr->GetStorageShape();
    info.rank = qShape.GetDimNum();
    if (info.rank != vShape.GetDimNum() || (info.rank != 3 && info.rank != 4)) {
        return false;
    }

    info.sequenceMajor = std::strcmp(layout, "BSND") == 0 || std::strcmp(layout, "TND") == 0;
    if (info.rank == 4) {
        info.batch = qShape.GetDim(0);
        if (info.sequenceMajor) {
            info.seqlen = qShape.GetDim(1);
            info.qHeads = qShape.GetDim(2);
            info.vHeads = vShape.GetDim(2);
        } else {
            info.qHeads = qShape.GetDim(1);
            info.vHeads = vShape.GetDim(1);
            info.seqlen = qShape.GetDim(2);
        }
        info.kDim = qShape.GetDim(3);
        info.vDim = vShape.GetDim(3);
    } else {
        info.batch = 1;
        if (info.sequenceMajor) {
            info.seqlen = qShape.GetDim(0);
            info.qHeads = qShape.GetDim(1);
            info.vHeads = vShape.GetDim(1);
        } else {
            info.qHeads = qShape.GetDim(0);
            info.vHeads = vShape.GetDim(0);
            info.seqlen = qShape.GetDim(1);
        }
        info.kDim = qShape.GetDim(2);
        info.vDim = vShape.GetDim(2);
    }
    return info.batch > 0 && info.seqlen > 0 && info.qHeads > 0 && info.vHeads > 0 &&
           info.kDim > 0 && info.vDim > 0 && info.vHeads % info.qHeads == 0;
}

bool ResolveSequenceInfo(gert::TilingContext *context, int64_t seqlen,
                         int64_t chunkSize, int64_t batch,
                         SequencePlanInfo &info)
{
    if (chunkSize <= 0 ||
        chunkSize > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    info.chunkSize = static_cast<uint32_t>(chunkSize);
    const auto cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
    info.isVarLen = cuTensor != nullptr;
    info.seqNum = batch;
    info.totalChunks = seqlen / chunkSize + (seqlen % chunkSize != 0);
    info.hasVarlenTail = false;
    if (!info.isVarLen) {
        if (batch <= 0 ||
            batch > static_cast<int64_t>(
                std::numeric_limits<uint32_t>::max())) {
            return false;
        }
        const uint64_t fullChunks64 = static_cast<uint64_t>(seqlen / chunkSize);
        const uint32_t tailChunks = static_cast<uint32_t>(seqlen % chunkSize != 0);
        uint64_t totalFullChunks64 = 0;
        if (fullChunks64 > std::numeric_limits<uint32_t>::max() ||
            !CheckedMul(static_cast<uint64_t>(batch), fullChunks64,
                        totalFullChunks64) ||
            totalFullChunks64 > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const uint32_t fullChunks = static_cast<uint32_t>(fullChunks64);
        info.denseTailTokens = static_cast<uint32_t>(seqlen % chunkSize);
        info.totalFullChunks = static_cast<uint32_t>(totalFullChunks64);
        info.totalTailChunks = static_cast<uint32_t>(batch) * tailChunks;
        info.fullChunkCounts.assign(1, fullChunks);
        return info.totalChunks > 0;
    }

    info.seqNum = cuTensor->GetStorageShape().GetDim(0) - 1;
    const int64_t *cu = cuTensor->GetData<int64_t>();
    if (info.seqNum <= 0 || info.seqNum > 1024 || cu == nullptr ||
        cu[0] != 0 || cu[info.seqNum] != seqlen) {
        return false;
    }
    info.totalChunks = 0;
    info.hasVarlenTail = false;
    info.seqChunkOffsets.assign(info.seqNum + 1, 0);
    info.fullChunkCounts.assign(info.seqNum, 0);
    for (int64_t seq = 0; seq < info.seqNum; ++seq) {
        if (cu[seq] < 0 || cu[seq + 1] < cu[seq]) {
            return false;
        }
        const int64_t seqLen = cu[seq + 1] - cu[seq];
        const uint64_t fullChunks64 =
            static_cast<uint64_t>(seqLen / chunkSize);
        const bool hasTail = seqLen % chunkSize != 0;
        const uint64_t nextFullChunks =
            static_cast<uint64_t>(info.totalFullChunks) + fullChunks64;
        const uint64_t nextTotalChunks =
            static_cast<uint64_t>(info.totalChunks) + fullChunks64 +
            static_cast<uint64_t>(hasTail);
        if (fullChunks64 > std::numeric_limits<uint32_t>::max() ||
            nextFullChunks > std::numeric_limits<uint32_t>::max() ||
            nextTotalChunks > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const uint32_t fullChunks = static_cast<uint32_t>(fullChunks64);
        info.fullChunkCounts[seq] = fullChunks;
        info.totalFullChunks = static_cast<uint32_t>(nextFullChunks);
        info.totalTailChunks += static_cast<uint32_t>(hasTail);
        info.hasVarlenTail = info.hasVarlenTail || hasTail;
        info.totalChunks = static_cast<int64_t>(nextTotalChunks);
        info.seqChunkOffsets[seq + 1] = static_cast<uint32_t>(nextTotalChunks);
        if (hasTail) {
            info.tailedSequenceIds.push_back(static_cast<uint16_t>(seq));
        } else {
            info.alignedSequenceIds.push_back(static_cast<uint16_t>(seq));
        }
    }

    const auto chunkShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
    if (chunkShape != nullptr &&
        chunkShape->GetStorageShape().GetShapeSize() != info.totalChunks * 2) {
        return false;
    }
    return info.totalChunks > 0;
}
} // namespace

ge::graphStatus Tiling4ChunkKdaFwd(gert::TilingContext *context)
{
    const auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    const auto gDesc = context->GetInputDesc(INPUT_G_IDX);
    const auto attrs = context->GetAttrs();
    if (qDesc == nullptr || gDesc == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const char *layout = attrs->GetStr(ATTR_LAYOUT_IDX);
    const float scale = static_cast<float>(*attrs->GetAttrPointer<double>(ATTR_SCALE_IDX));
    const int64_t chunkSize = *attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX);
    const bool safeGate = *attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE_IDX);
    const float lowerBound = *attrs->GetAttrPointer<float>(ATTR_LOWER_BOUND_IDX);
    const bool useGateInKernel = *attrs->GetAttrPointer<bool>(ATTR_USE_GATE_IDX);
    if (chunkSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    ShapeInfo shape;
    if (!ResolveShape(context, layout, shape)) {
        return ge::GRAPH_FAILED;
    }
    SequencePlanInfo sequenceInfo;
    if (!ResolveSequenceInfo(context, shape.seqlen, chunkSize, shape.batch,
                             sequenceInfo)) {
        return ge::GRAPH_FAILED;
    }
    const bool isVarLen = sequenceInfo.isVarLen;
    const int64_t seqNum = sequenceInfo.seqNum;
    const int64_t totalChunks = sequenceInfo.totalChunks;
    const bool hasVarlenTail = sequenceInfo.hasVarlenTail;

    const auto aLogDesc = context->GetOptionalInputDesc(INPUT_A_LOG_IDX);
    const auto dtBiasDesc = context->GetOptionalInputDesc(INPUT_DT_BIAS_IDX);
    const auto initialStateDesc = context->GetOptionalInputDesc(INPUT_INITIAL_STATE_IDX);
    const bool hasALog = aLogDesc != nullptr;
    const bool hasDtBias = dtBiasDesc != nullptr;
    const bool hasInitialState = initialStateDesc != nullptr;
    if ((hasALog && aLogDesc->GetDataType() != ge::DT_FLOAT &&
         aLogDesc->GetDataType() != ge::DT_BF16) ||
        (hasDtBias && dtBiasDesc->GetDataType() != ge::DT_FLOAT &&
         dtBiasDesc->GetDataType() != ge::DT_BF16) ||
        (hasInitialState && initialStateDesc->GetDataType() != ge::DT_FLOAT)) {
        return ge::GRAPH_FAILED;
    }
    const bool gateParamsAreFp32 =
        (!hasALog || aLogDesc->GetDataType() == ge::DT_FLOAT) &&
        (!hasDtBias || dtBiasDesc->GetDataType() == ge::DT_FLOAT);
    const bool storeFinalState = HasOutput(context, OUTPUT_FINAL_STATE_IDX);
    const bool storeGk = HasOutput(context, OUTPUT_GK_IDX);
    const bool storeW = HasOutput(context, OUTPUT_W_IDX);
    const bool storeU = HasOutput(context, OUTPUT_U_IDX);
    const bool storeQG = HasOutput(context, OUTPUT_QG_IDX);
    const bool storeKg = HasOutput(context, OUTPUT_KG_IDX);
    const bool storeVNew = HasOutput(context, OUTPUT_V_NEW_IDX);
    const bool storeH = HasOutput(context, OUTPUT_H_IDX);

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t blockDim = std::max<uint32_t>(platform.GetCoreNumAic(), 1);
    std::vector<uint8_t> compactPlan;
    if (!BuildCompactSequencePlan(
            sequenceInfo, blockDim,
            static_cast<uint32_t>(shape.qHeads),
            static_cast<uint32_t>(shape.vHeads), compactPlan)) {
        return ge::GRAPH_FAILED;
    }
    const bool isAscend950 =
        platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    const bool useChunk64K128V128Template =
        chunkSize == 64 && shape.kDim == 128 && shape.vDim == 128;
    const auto arch35Options = arch35::ConfigureChunkKdaFwdArch35(
        isAscend950, qDesc->GetDataType() == ge::DT_BF16,
        gDesc->GetDataType() == ge::DT_FLOAT, hasALog, gateParamsAreFp32,
        useGateInKernel, safeGate, isVarLen, hasVarlenTail, seqNum, shape.seqlen,
        shape.vHeads, chunkSize, shape.kDim, shape.vDim, storeQG, storeVNew,
        storeH);

    const uint64_t dataBytes =
        qDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    uint64_t tokenHeads = 0;
    uint64_t kTensorBytes = 0;
    uint64_t vTensorBytes = 0;
    uint64_t gkBytes = 0;
    uint64_t stateBytes = 0;
    uint64_t hChunkCount = 0;
    uint64_t hBytes = 0;
    if (!CheckedProduct(
            {static_cast<uint64_t>(shape.batch),
             static_cast<uint64_t>(shape.vHeads),
             static_cast<uint64_t>(shape.seqlen)}, tokenHeads) ||
        !CheckedProduct(
            {tokenHeads, static_cast<uint64_t>(shape.kDim), dataBytes},
            kTensorBytes) ||
        !CheckedProduct(
            {tokenHeads, static_cast<uint64_t>(shape.vDim), dataBytes},
            vTensorBytes) ||
        !CheckedProduct(
            {tokenHeads, static_cast<uint64_t>(shape.kDim), sizeof(float)},
            gkBytes) ||
        !CheckedProduct(
            {static_cast<uint64_t>(seqNum),
             static_cast<uint64_t>(shape.vHeads),
             static_cast<uint64_t>(shape.kDim),
             static_cast<uint64_t>(shape.vDim), sizeof(float)}, stateBytes) ||
        !CheckedAdd(sequenceInfo.totalFullChunks,
                    sequenceInfo.totalTailChunks, hChunkCount) ||
        !CheckedProduct(
            {hChunkCount, static_cast<uint64_t>(shape.vHeads),
             static_cast<uint64_t>(shape.kDim),
             static_cast<uint64_t>(shape.vDim), dataBytes}, hBytes)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t cursor = 0;
    auto allocateHidden = [&cursor](bool storeOutput, uint64_t bytes,
                                    uint64_t &offset) {
        offset = 0;
        return storeOutput || AllocateWorkspace(cursor, bytes, offset);
    };
    uint64_t gkStorageOffset = 0;
    uint64_t finalStateStorageOffset = 0;
    uint64_t wStorageOffset = 0;
    uint64_t uStorageOffset = 0;
    uint64_t qgStorageOffset = 0;
    uint64_t kgStorageOffset = 0;
    uint64_t vNewStorageOffset = 0;
    uint64_t hStorageOffset = 0;
    uint64_t qgScaledOffset = 0;
    if (!allocateHidden(storeGk, gkBytes, gkStorageOffset) ||
        !allocateHidden(storeFinalState, stateBytes,
                        finalStateStorageOffset) ||
        !allocateHidden(storeW, kTensorBytes, wStorageOffset) ||
        !allocateHidden(storeU, vTensorBytes, uStorageOffset) ||
        !allocateHidden(storeQG, kTensorBytes, qgStorageOffset) ||
        !allocateHidden(storeKg, kTensorBytes, kgStorageOffset) ||
        !allocateHidden(storeVNew, vTensorBytes, vNewStorageOffset) ||
        !allocateHidden(storeH, hBytes, hStorageOffset) ||
        !AllocateWorkspace(cursor, kTensorBytes, qgScaledOffset)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t matrixBytes = 0;
    uint64_t prepareAqkFp32Offset = 0;
    uint64_t prepareAkkFp32Offset = 0;
    uint64_t prepareScratchOffset = 0;
    if (!CheckedProduct(
            {tokenHeads, static_cast<uint64_t>(chunkSize), sizeof(float)},
            matrixBytes) ||
        !AllocateWorkspace(cursor, matrixBytes, prepareAqkFp32Offset) ||
        !AllocateWorkspace(cursor, matrixBytes, prepareAkkFp32Offset) ||
        !CheckedAlign(cursor, prepareScratchOffset)) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t solveDepth = safeGate ? KDA_SOLVE_PIPELINE_DEPTH : 1;
    uint64_t solveBytes = 0;
    uint64_t alignedSolveBytes = 0;
    uint64_t scoreBytes = 0;
    if (!CheckedProduct(
            {static_cast<uint64_t>(blockDim), solveDepth,
             KDA_SOLVE_SCRATCH_SLOTS, static_cast<uint64_t>(chunkSize),
             static_cast<uint64_t>(chunkSize), sizeof(float)}, solveBytes) ||
        !CheckedAlign(solveBytes, alignedSolveBytes) ||
        !CheckedProduct(
            {static_cast<uint64_t>(blockDim), KDA_SCORE_QUEUE_SLOTS,
             KDA_SCORE_SCRATCH_PLANES, static_cast<uint64_t>(chunkSize),
             static_cast<uint64_t>(shape.kDim), dataBytes}, scoreBytes) ||
        !CheckedAdd(prepareScratchOffset, alignedSolveBytes, cursor) ||
        !CheckedAdd(cursor, scoreBytes, cursor)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t postWuScratchOffset = 0;
    if (!CheckedAlign(cursor, postWuScratchOffset)) {
        return ge::GRAPH_FAILED;
    }
    if (!arch35Options.fusePostWu && !arch35Options.fusePostWuIntoFwdH) {
        uint64_t postWuScratchBytes = 0;
        if (!CheckedProduct(
                {hChunkCount, static_cast<uint64_t>(shape.vHeads),
                 static_cast<uint64_t>(chunkSize),
                 static_cast<uint64_t>(shape.kDim), sizeof(float)},
                postWuScratchBytes) ||
            !CheckedAdd(postWuScratchOffset, postWuScratchBytes, cursor)) {
            return ge::GRAPH_FAILED;
        }
    }

    uint64_t fwdHWorkspaceBaseOffset = 0;
    if (!CheckedAlign(cursor, fwdHWorkspaceBaseOffset)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t fwdHCursor = 0;
    uint64_t vWorkspaceBytes = 0;
    uint64_t kDecayWorkspaceBytes = 0;
    uint64_t hWorkspaceBytes = 0;
    uint64_t vWorkspaceOffset = 0;
    uint64_t vUpdateWorkspaceOffset = 0;
    uint64_t kDecayWorkspaceOffset = 0;
    uint64_t hWorkspaceOffset = 0;
    if (!CheckedProduct(
            {static_cast<uint64_t>(blockDim),
             static_cast<uint64_t>(chunkSize),
             static_cast<uint64_t>(shape.vDim), sizeof(float),
             KDA_GDN_PIPELINE_DEPTH}, vWorkspaceBytes) ||
        !CheckedProduct(
            {static_cast<uint64_t>(blockDim),
             static_cast<uint64_t>(chunkSize),
             static_cast<uint64_t>(shape.kDim), sizeof(float),
             KDA_GDN_PIPELINE_DEPTH}, kDecayWorkspaceBytes) ||
        !CheckedProduct(
            {static_cast<uint64_t>(blockDim),
             static_cast<uint64_t>(shape.kDim),
             static_cast<uint64_t>(shape.vDim), sizeof(float),
             KDA_GDN_PIPELINE_DEPTH}, hWorkspaceBytes) ||
        !AllocateWorkspace(fwdHCursor, vWorkspaceBytes, vWorkspaceOffset) ||
        !AllocateWorkspace(
            fwdHCursor, vWorkspaceBytes, vUpdateWorkspaceOffset) ||
        !AllocateWorkspace(
            fwdHCursor, kDecayWorkspaceBytes, kDecayWorkspaceOffset) ||
        !AllocateWorkspace(fwdHCursor, hWorkspaceBytes, hWorkspaceOffset)) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t tokenBatch = isVarLen ? static_cast<uint64_t>(seqNum) : 1;
    uint64_t metadataCount = 0;
    uint64_t metadataBytes = 0;
    uint64_t numSeqWorkspaceOffset = 0;
    uint64_t numChunksWorkspaceOffset = 0;
    uint64_t alignedFwdHBytes = 0;
    if (!CheckedAdd(tokenBatch, 1, metadataCount) ||
        !CheckedMul(metadataCount, sizeof(int64_t), metadataBytes) ||
        !AllocateWorkspace(
            fwdHCursor, metadataBytes, numSeqWorkspaceOffset) ||
        !AllocateWorkspace(
            fwdHCursor, metadataBytes, numChunksWorkspaceOffset) ||
        !CheckedAlign(fwdHCursor, alignedFwdHBytes) ||
        !CheckedAdd(fwdHWorkspaceBaseOffset, alignedFwdHBytes, cursor)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outputScratchElements = 0;
    uint64_t finalizeScratchBytes = 0;
    if (!CheckedProduct(
            {static_cast<uint64_t>(blockDim), KDA_OUTPUT_SLOT_DEPTH,
             KDA_OUTPUT_SCRATCH_PLANES, static_cast<uint64_t>(chunkSize),
             static_cast<uint64_t>(shape.vDim)}, outputScratchElements) ||
        !CheckedMul(
            outputScratchElements, sizeof(float), finalizeScratchBytes)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t postWuStagingBytes = 0;
    if (!arch35Options.fusePostWu && !arch35Options.fusePostWuIntoFwdH) {
        // Prepare 先在这里写入完整的 u 初值，随后才进入独立 PostWU 阶段。
        // generic C64/K128/V128 的变长尾块还会紧接着保存 w；Finalize 更晚
        // 执行，因此可以复用同一块空间作为逐核双槽 scratch。
        postWuStagingBytes = vTensorBytes;
        const bool needsGenericTailSnapshot =
            !isAscend950 && isVarLen && hasVarlenTail &&
            chunkSize == 64 && shape.kDim == 128 && shape.vDim == 128;
        if (needsGenericTailSnapshot &&
            !CheckedAdd(
                postWuStagingBytes, kTensorBytes, postWuStagingBytes)) {
            return ge::GRAPH_FAILED;
        }
    }
    const uint64_t outputScratchBytes =
        std::max(finalizeScratchBytes, postWuStagingBytes);
    uint64_t outputScratchOffset = 0;
    uint64_t totalWorkspace = 0;
    uint64_t workspaceWithLibrary = 0;
    if (!AllocateWorkspace(
            cursor, outputScratchBytes, outputScratchOffset) ||
        !CheckedAlign(cursor, totalWorkspace) ||
        !CheckedAdd(platform.GetLibApiWorkSpaceSize(), totalWorkspace,
                    workspaceWithLibrary)) {
        return ge::GRAPH_FAILED;
    }

    constexpr uint64_t maxTilingOffset =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    for (uint64_t offset : {
             gkStorageOffset, finalStateStorageOffset, wStorageOffset,
             uStorageOffset, qgStorageOffset, kgStorageOffset,
             vNewStorageOffset, hStorageOffset, qgScaledOffset,
             prepareAqkFp32Offset, prepareAkkFp32Offset,
             prepareScratchOffset, postWuScratchOffset,
             outputScratchOffset, fwdHWorkspaceBaseOffset,
             vWorkspaceOffset, vUpdateWorkspaceOffset,
             kDecayWorkspaceOffset, hWorkspaceOffset,
             numSeqWorkspaceOffset, numChunksWorkspaceOffset}) {
        if (offset > maxTilingOffset) {
            return ge::GRAPH_FAILED;
        }
    }

    context->SetBlockDim(blockDim);
    context->SetTilingKey(useChunk64K128V128Template ? 2 : 1);
    context->SetScheduleMode(KDA_BATCH_MODE);
    context->GetWorkspaceSizes(1)[0] = workspaceWithLibrary;

    ChunkKdaFwdTilingData tiling;
    tiling.set_batch(shape.batch);
    tiling.set_seqNum(seqNum);
    tiling.set_qHeadNum(shape.qHeads);
    tiling.set_vHeadNum(shape.vHeads);
    tiling.set_seqlen(shape.seqlen);
    tiling.set_kHeadDim(shape.kDim);
    tiling.set_vHeadDim(shape.vDim);
    tiling.set_chunkSize(chunkSize);
    tiling.set_totalChunks(totalChunks);
    tiling.set_inputRank(shape.rank);
    tiling.set_scale(scale);
    tiling.set_lowerBound(lowerBound);
    tiling.set_hasInitialState(hasInitialState);
    tiling.set_isVarLen(isVarLen);
    tiling.set_safeGate(safeGate);
    tiling.set_inputSequenceMajor(shape.sequenceMajor);
    tiling.set_useGateInKernel(useGateInKernel);
    tiling.set_hasALog(hasALog);
    tiling.set_hasDtBias(hasDtBias);
    tiling.set_computeGateInPrepare(arch35Options.computeGateInPrepare);
    tiling.set_fusePostWu(arch35Options.fusePostWu);
    tiling.set_fusePostWuIntoFwdH(arch35Options.fusePostWuIntoFwdH);
    tiling.set_useDenseFwdH(arch35Options.useDenseFwdH);
    tiling.set_hasVarlenTail(hasVarlenTail);
    tiling.set_storeFinalState(storeFinalState);
    tiling.set_storeGk(storeGk);
    tiling.set_storeW(storeW);
    tiling.set_storeU(storeU);
    tiling.set_storeQG(storeQG);
    tiling.set_storeKg(storeKg);
    tiling.set_storeVNew(storeVNew);
    tiling.set_storeH(storeH);
    tiling.set_gateDataType(gDesc->GetDataType() == ge::DT_FLOAT ? 2 :
        (gDesc->GetDataType() == ge::DT_BF16 ? 1 : 0));
    tiling.set_aLogDataType(hasALog && aLogDesc->GetDataType() == ge::DT_BF16
        ? KDA_PARAM_DTYPE_BF16 : KDA_PARAM_DTYPE_FP32);
    tiling.set_dtBiasDataType(hasDtBias && dtBiasDesc->GetDataType() == ge::DT_BF16
        ? KDA_PARAM_DTYPE_BF16 : KDA_PARAM_DTYPE_FP32);
    tiling.set_gateUsedCoreNum(static_cast<int64_t>(blockDim) * 2);
    tiling.set_prepareUsedCoreNum(blockDim);
    tiling.set_postWuUsedCoreNum(blockDim);
    tiling.set_outputUsedCoreNum(blockDim);
    tiling.set_gkStorageOffset(gkStorageOffset);
    tiling.set_finalStateStorageOffset(finalStateStorageOffset);
    tiling.set_wStorageOffset(wStorageOffset);
    tiling.set_uStorageOffset(uStorageOffset);
    tiling.set_qgStorageOffset(qgStorageOffset);
    tiling.set_kgStorageOffset(kgStorageOffset);
    tiling.set_vNewStorageOffset(vNewStorageOffset);
    tiling.set_hStorageOffset(hStorageOffset);
    tiling.set_qgScaledOffset(qgScaledOffset);
    tiling.set_prepareAqkFp32Offset(prepareAqkFp32Offset);
    tiling.set_prepareAkkFp32Offset(prepareAkkFp32Offset);
    tiling.set_prepareScratchOffset(prepareScratchOffset);
    tiling.set_postWuScratchOffset(postWuScratchOffset);
    tiling.set_outputScratchOffset(outputScratchOffset);
    tiling.set_fwdHWorkspaceBaseOffset(fwdHWorkspaceBaseOffset);
    tiling.set_vWorkspaceOffset(vWorkspaceOffset);
    tiling.set_vUpdateWorkspaceOffset(vUpdateWorkspaceOffset);
    tiling.set_kDecayWorkspaceOffset(kDecayWorkspaceOffset);
    tiling.set_hWorkspaceOffset(hWorkspaceOffset);
    tiling.set_numSeqWorkspaceOffset(numSeqWorkspaceOffset);
    tiling.set_numChunksWorkspaceOffset(numChunksWorkspaceOffset);
    const size_t fixedTilingBytes = tiling.GetDataSize();
    if (fixedTilingBytes > std::numeric_limits<uint32_t>::max() ||
        compactPlan.size() > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t compactPlanOffset =
        static_cast<uint32_t>(fixedTilingBytes);
    tiling.set_compactPlanOffset(compactPlanOffset);
    tiling.set_compactPlanBytes(static_cast<uint32_t>(compactPlan.size()));
    auto *rawTiling = context->GetRawTilingData();
    uint64_t totalTilingBytes = 0;
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr ||
        !CheckedAdd(fixedTilingBytes, compactPlan.size(), totalTilingBytes) ||
        totalTilingBytes > std::numeric_limits<uint32_t>::max() ||
        totalTilingBytes > rawTiling->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    auto *rawTilingBytes = static_cast<uint8_t *>(rawTiling->GetData());
    std::memcpy(rawTilingBytes + compactPlanOffset,
                compactPlan.data(), compactPlan.size());
    rawTiling->SetDataSize(static_cast<size_t>(totalTilingBytes));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaFwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwd)
    .Tiling(Tiling4ChunkKdaFwd)
    .TilingParse<ChunkKdaFwdCompileInfo>(TilingPrepare4ChunkKdaFwd);
} // namespace optiling
