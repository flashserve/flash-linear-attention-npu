#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_TILING_ARCH22_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_TILING_ARCH22_H

#include "../op_tiling/chunk_gated_delta_rule_bwd_finalize_tiling.h"
#include "chunk_gated_delta_rule_bwd_finalize_validation_arch22.h"
#include "chunk_gated_delta_rule_bwd_finalize_error_arch22.h"
#include "../../op_kernel/arch22/chunk_gated_delta_rule_bwd_finalize_workspace.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {
namespace gdn_finalize_arch22 {

using namespace GDN;
using GDN::FinalizeHostArch22::ShapeString;
using GDN::FinalizeHostArch22::ParameterKind;

// A2/A3 的 shape、属性、调度和 workspace 检查不进入 A5 基线路径。
enum InputIndex : size_t {
    INPUT_Q_IDX = 0,
    INPUT_K_IDX,
    INPUT_V_IDX,
    INPUT_V_NEW_IDX,
    INPUT_DO_IDX,
    INPUT_DU_IDX,
    INPUT_G_IDX,
    INPUT_BETA_IDX,
    INPUT_H_IDX,
    INPUT_DH_IDX,
    INPUT_A_IDX,
    INPUT_Q_RSTD_IDX,
    INPUT_K_RSTD_IDX,
    INPUT_BETA_RAW_IDX,
    INPUT_CU_SEQLENS_IDX,
    INPUT_CHUNK_INDICES_IDX,
};

constexpr int64_t CHUNK_SIZE_64 = 64;
constexpr int64_t CHUNK_SIZE_128 = 128;
constexpr int64_t K_SIZE_128 = 128;
constexpr int64_t V_SIZE_128 = 128;
constexpr int64_t V_SIZE_256 = 256;
int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

bool FitsProduct(std::initializer_list<int64_t> values)
{
    int64_t product = 1;
    for (int64_t value : values) {
        if (value <= 0 || product > std::numeric_limits<int64_t>::max() / value) {
            return false;
        }
        product *= value;
    }
    return true;
}

int DTypeKey(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return TPL_BF16;
    }
    if (dtype == ge::DT_FLOAT) {
        return TPL_FP32;
    }
    return -1;
}

bool CheckRank(const gert::StorageShape *shape, size_t rank)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() == rank;
}

ge::graphStatus InvalidParameter(gert::TilingContext *context, const std::string &parameter,
                                 const std::string &actual, const std::string &expected,
                                 ParameterKind kind = ParameterKind::VALUE, size_t inputIndex = 0)
{
    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "A2/A3 parameter[%s]: actual=%s; expected=%s.",
            parameter.c_str(), actual.c_str(), expected.c_str());
    if (!GDN::FinalizeHostArch22::ReportParameterError(parameter, actual, expected, kind, inputIndex)) {
        OP_LOGE(context->GetNodeName(), "A2/A3 parameter[%s]: actual=%s; expected=%s.",
                parameter.c_str(), actual.c_str(), expected.c_str());
    }
    return ge::GRAPH_FAILED;
}

std::string ShapeString(std::initializer_list<int64_t> dimensions)
{
    std::ostringstream stream;
    stream << "[";
    size_t index = 0;
    for (const int64_t dimension : dimensions) {
        stream << (index++ == 0 ? "" : ", ") << dimension;
    }
    stream << "]";
    return stream.str();
}

ge::graphStatus ValidateShape(gert::TilingContext *context, const char *parameter,
                              const gert::Shape &actual, std::initializer_list<int64_t> expected, size_t inputIndex)
{
    size_t index = 0;
    bool matches = actual.GetDimNum() == expected.size();
    if (matches) {
        for (const int64_t dimension : expected) {
            matches = matches && actual.GetDim(index++) == dimension;
        }
    }
    if (!matches) {
        return InvalidParameter(context, parameter, ShapeString(actual), ShapeString(expected),
                                ParameterKind::SHAPE, inputIndex);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateShapes(gert::TilingContext *context,
                               GDN::ChunkGatedDeltaRuleBwdFinalizeTilingData &tiling,
                               bool stateVFirst)
{
    // 先检查真实输入 rank，再提取 shape；错误报告保留具体参数和 actual/expected。
    const gert::StorageShape *shapes[] = {
        context->GetRequiredInputShape(INPUT_Q_IDX), context->GetRequiredInputShape(INPUT_K_IDX),
        context->GetRequiredInputShape(INPUT_V_IDX), context->GetRequiredInputShape(INPUT_V_NEW_IDX),
        context->GetRequiredInputShape(INPUT_DO_IDX), context->GetRequiredInputShape(INPUT_DU_IDX),
        context->GetRequiredInputShape(INPUT_G_IDX), context->GetRequiredInputShape(INPUT_BETA_IDX),
        context->GetRequiredInputShape(INPUT_H_IDX), context->GetRequiredInputShape(INPUT_DH_IDX),
        context->GetRequiredInputShape(INPUT_A_IDX)};
    const char *names[] = {"q", "k", "v", "v_new", "do", "du", "g", "beta", "h", "dh", "a"};
    const size_t ranks[] = {4, 4, 4, 4, 4, 4, 3, 3, 5, 5, 4};
    for (size_t index = 0; index < 11; ++index) {
        if (!CheckRank(shapes[index], ranks[index])) {
            const std::string actual = shapes[index] == nullptr ? "absent" :
                "rank=" + std::to_string(shapes[index]->GetStorageShape().GetDimNum()) +
                ", shape=" + ShapeString(shapes[index]->GetStorageShape());
            return InvalidParameter(context, std::string(names[index]) + ".rank", actual,
                                    std::to_string(ranks[index]), ParameterKind::SHAPE, index);
        }
    }
    const gert::Shape &q = shapes[INPUT_Q_IDX]->GetStorageShape();
    const gert::Shape &v = shapes[INPUT_V_IDX]->GetStorageShape();
    const gert::Shape &a = shapes[INPUT_A_IDX]->GetStorageShape();
    const gert::Shape &h = shapes[INPUT_H_IDX]->GetStorageShape();
    tiling.B = q.GetDim(0);
    tiling.HK = q.GetDim(1);
    tiling.T = q.GetDim(2);
    tiling.K = q.GetDim(3);
    tiling.HV = v.GetDim(1);
    tiling.V = v.GetDim(3);
    tiling.chunkSize = a.GetDim(3);
    if (tiling.B <= 0 || tiling.HK <= 0 || tiling.HV <= 0 || tiling.T <= 0) {
        return InvalidParameter(context, "B/HK/HV/T",
                                ShapeString({tiling.B, tiling.HK, tiling.HV, tiling.T}), "all dimensions > 0");
    }
    if (tiling.K != K_SIZE_128 || (tiling.V != V_SIZE_128 && tiling.V != V_SIZE_256)) {
        return InvalidParameter(context, "K/V", ShapeString({tiling.K, tiling.V}), "K=128; V=128 or 256");
    }
    if (tiling.chunkSize != CHUNK_SIZE_64 && tiling.chunkSize != CHUNK_SIZE_128) {
        return InvalidParameter(context, "a.BT", std::to_string(tiling.chunkSize), "64 or 128 on A2/A3");
    }
    if (tiling.HV % tiling.HK != 0) {
        return InvalidParameter(context, "HV/HK", ShapeString({tiling.HV, tiling.HK}), "HV % HK == 0");
    }
    tiling.headRatio = tiling.HV / tiling.HK;
    if (tiling.headRatio < 1) {
        return InvalidParameter(context, "HV/HK", std::to_string(tiling.headRatio), ">= 1");
    }
    if (!FitsProduct({tiling.B, tiling.HV, tiling.T, std::max({tiling.K, tiling.V, tiling.chunkSize}), 128})) {
        return InvalidParameter(context, "B/HV/T/max(K,V,BT)/128",
                                ShapeString({tiling.B, tiling.HV, tiling.T,
                                             std::max({tiling.K, tiling.V, tiling.chunkSize}), 128}),
                                "positive factors with product <= INT64_MAX for A2/A3 address arithmetic");
    }
    if (ValidateShape(context, "k.shape", shapes[INPUT_K_IDX]->GetStorageShape(),
                      {tiling.B, tiling.HK, tiling.T, tiling.K}, INPUT_K_IDX) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const size_t valueIndices[] = {INPUT_V_IDX, INPUT_V_NEW_IDX, INPUT_DO_IDX, INPUT_DU_IDX};
    for (const size_t index : valueIndices) {
        if (ValidateShape(context, (std::string(names[index]) + ".shape").c_str(),
                          shapes[index]->GetStorageShape(), {tiling.B, tiling.HV, tiling.T, tiling.V}, index) !=
                ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    const size_t gateIndices[] = {INPUT_G_IDX, INPUT_BETA_IDX};
    for (const size_t index : gateIndices) {
        if (ValidateShape(context, (std::string(names[index]) + ".shape").c_str(),
                          shapes[index]->GetStorageShape(), {tiling.B, tiling.HV, tiling.T}, index)
                != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (ValidateShape(context, "a.shape", a, {tiling.B, tiling.HV, tiling.T, tiling.chunkSize}, INPUT_A_IDX) !=
            ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t stateDim3 = stateVFirst ? tiling.V : tiling.K;
    const int64_t stateDim4 = stateVFirst ? tiling.K : tiling.V;
    const size_t stateIndices[] = {INPUT_H_IDX, INPUT_DH_IDX};
    for (const size_t index : stateIndices) {
        if (ValidateShape(context, (std::string(names[index]) + ".shape (state_v_first=" +
                                   (stateVFirst ? "true)" : "false)")).c_str(),
                          shapes[index]->GetStorageShape(),
                          {tiling.B, h.GetDim(1), tiling.HV, stateDim3, stateDim4}, index) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus Tiling(gert::TilingContext *context,
                       const platform_ascendc::PlatformAscendC &platform)
{
    auto *tiling = context->GetTilingData<GDN::ChunkGatedDeltaRuleBwdFinalizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *stateVFirst = attrs->GetAttrPointer<bool>(5);
    const bool stateVFirstValue = stateVFirst == nullptr ? false : *stateVFirst;
    OP_CHECK_IF(ValidateShapes(context, *tiling, stateVFirstValue) != ge::GRAPH_SUCCESS, ,
                return ge::GRAPH_FAILED);

    const double *scale = attrs->GetAttrPointer<double>(0);
    const int64_t *chunkSize = attrs->GetAttrPointer<int64_t>(1);
    const bool *useL2Norm = attrs->GetAttrPointer<bool>(2);
    const bool *useBetaSigmoid = attrs->GetAttrPointer<bool>(3);
    const bool *useGate = attrs->GetAttrPointer<bool>(4);
    const bool *useExp2 = attrs->GetAttrPointer<bool>(6);
    const bool useL2NormValue = useL2Norm == nullptr ? false : *useL2Norm;
    const bool useBetaSigmoidValue = useBetaSigmoid == nullptr ? false : *useBetaSigmoid;
    const bool useGateValue = useGate == nullptr ? false : *useGate;
    const bool useExp2Value = useExp2 == nullptr ? true : *useExp2;
    const int64_t actualChunkSize = chunkSize == nullptr ? tiling->chunkSize : *chunkSize;
    if (actualChunkSize != tiling->chunkSize) {
        return InvalidParameter(context, "chunk_size", std::to_string(actualChunkSize),
                                "a.BT=" + std::to_string(tiling->chunkSize));
    }
    if (scale != nullptr && (!std::isfinite(*scale) || !std::isfinite(static_cast<float>(*scale)))) {
        return InvalidParameter(context, "scale", std::to_string(*scale), "finite after conversion to FP32");
    }
    if (useGateValue) {
        return InvalidParameter(context, "use_gate_in_kernel", "true", "false on A2/A3");
    }
    const auto cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS_IDX);
    const auto chunkIndicesShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
    if ((cuShape == nullptr) != (chunkIndicesShape == nullptr)) {
        return InvalidParameter(context, "cu_seqlens/chunk_indices",
                                std::string(cuShape == nullptr ? "absent/" : "present/") +
                                (chunkIndicesShape == nullptr ? "absent" : "present"),
                                "both present or both absent");
    }
    tiling->isVariable = cuShape != nullptr ? 1 : 0;
    if (tiling->isVariable != 0) {
        if (tiling->B != 1) {
            return InvalidParameter(context, "B", std::to_string(tiling->B), "1 in varlen mode");
        }
        const gert::StorageShape *metadataShapes[] = {cuShape, chunkIndicesShape};
        const char *metadataNames[] = {"cu_seqlens.rank", "chunk_indices.rank"};
        for (size_t index = 0; index < 2; ++index) {
            if (!CheckRank(metadataShapes[index], 1)) {
                return InvalidParameter(context, metadataNames[index],
                                        ShapeString(metadataShapes[index]->GetStorageShape()), "rank 1",
                                        ParameterKind::SHAPE, INPUT_CU_SEQLENS_IDX + index);
            }
        }
        const gert::Shape &cu = cuShape->GetStorageShape();
        const gert::Shape &chunks = chunkIndicesShape->GetStorageShape();
        // aclIntArray 转成的 chunk_indices 是一维扁平数组，按
        // [seqIdx0, chunkIdx0, seqIdx1, chunkIdx1, ...] 两两分组。
        if (cu.GetDim(0) < 2 || chunks.GetDim(0) <= 0 || chunks.GetDim(0) % 2 != 0) {
            return InvalidParameter(context, "cu_seqlens.size/chunk_indices.size",
                                    ShapeString({cu.GetDim(0), chunks.GetDim(0)}),
                                    "cu_seqlens.size >= 2; positive even chunk_indices.size");
        }
        tiling->seqNum = cu.GetDim(0) - 1;
        tiling->totalChunkNum = chunks.GetDim(0) / 2;
        tiling->chunkNumForT = 0;
    } else {
        const uint64_t lastStart = static_cast<uint64_t>((tiling->T - 1) / tiling->chunkSize * tiling->chunkSize);
        if (!GDN::FinalizeHostArch22::ChunkEndFitsUint32(lastStart, static_cast<uint64_t>(tiling->chunkSize))) {
            return InvalidParameter(context, "T/BT", "T=" + std::to_string(tiling->T) +
                                    ", last chunk start=" + std::to_string(lastStart) +
                                    ", BT=" + std::to_string(tiling->chunkSize),
                                    "last chunk start + BT <= UINT32_MAX on A2/A3");
        }
        tiling->seqNum = tiling->B;
        tiling->chunkNumForT = CeilDiv(tiling->T, tiling->chunkSize);
        tiling->totalChunkNum = tiling->B * tiling->chunkNumForT;
    }
    const auto hShape = context->GetRequiredInputShape(INPUT_H_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hShape);
    const int64_t stateChunkNum = tiling->isVariable != 0 ? tiling->totalChunkNum : tiling->chunkNumForT;
    if (hShape->GetStorageShape().GetDim(1) != stateChunkNum) {
        return InvalidParameter(context, "h/dh.NT", std::to_string(hShape->GetStorageShape().GetDim(1)),
                                "chunk schedule NT=" + std::to_string(stateChunkNum));
    }
    if (std::max({tiling->B, tiling->HK, tiling->HV, tiling->T, tiling->totalChunkNum, tiling->seqNum}) >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return InvalidParameter(context, "B/HK/HV/T/totalChunkNum/seqNum",
                                ShapeString({tiling->B, tiling->HK, tiling->HV, tiling->T,
                                             tiling->totalChunkNum, tiling->seqNum}),
                                "each dimension <= UINT32_MAX on A2/A3");
    }

    const uint32_t aicCoreNum = std::max<uint32_t>(1, platform.GetCoreNumAic());
    // A2 单次混合 kernel 中处理完整 chunk；HK/HV 归约不跨逻辑任务。
    tiling->taskGroupSize = 1;
    tiling->headGroupNum = 1;
    tiling->taskNum = tiling->totalChunkNum;

    const auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    const auto kDesc = context->GetInputDesc(INPUT_K_IDX);
    const auto vDesc = context->GetInputDesc(INPUT_V_IDX);
    const auto gDesc = context->GetInputDesc(INPUT_G_IDX);
    const auto betaDesc = context->GetInputDesc(INPUT_BETA_IDX);
    const auto vNewDesc = context->GetInputDesc(INPUT_V_NEW_IDX);
    const auto dODesc = context->GetInputDesc(INPUT_DO_IDX);
    const auto duDesc = context->GetInputDesc(INPUT_DU_IDX);
    const auto hDesc = context->GetInputDesc(INPUT_H_IDX);
    const auto dhDesc = context->GetInputDesc(INPUT_DH_IDX);
    const auto aDesc = context->GetInputDesc(INPUT_A_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vNewDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dODesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, duDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, hDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dhDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, aDesc);
    const int qKey = DTypeKey(qDesc->GetDataType());
    const int gKey = DTypeKey(gDesc->GetDataType());
    decltype(qDesc) mainDescs[] = {qDesc, kDesc, vDesc, vNewDesc, dODesc, duDesc,
                                                      hDesc, dhDesc, aDesc};
    const char *mainNames[] = {"q", "k", "v", "v_new", "do", "du", "h", "dh", "a"};
    for (size_t index = 0; index < 9; ++index) {
        if (mainDescs[index]->GetDataType() != ge::DT_BF16) {
            return InvalidParameter(context, std::string(mainNames[index]) + ".dtype",
                                    std::to_string(static_cast<int>(mainDescs[index]->GetDataType())), "BF16",
                                    ParameterKind::DTYPE);
        }
    }
    if (gKey != TPL_BF16 && gKey != TPL_FP32) {
        return InvalidParameter(context, "g.dtype", std::to_string(static_cast<int>(gDesc->GetDataType())),
                                "BF16 or FP32", ParameterKind::DTYPE);
    }
    if (gDesc->GetDataType() != betaDesc->GetDataType()) {
        return InvalidParameter(context, "beta.dtype", std::to_string(static_cast<int>(betaDesc->GetDataType())),
                                "g.dtype=" + std::to_string(static_cast<int>(gDesc->GetDataType())),
                                ParameterKind::DTYPE);
    }

    // 可选输入按实际启用的 Stage 15/Stage 12 功能校验。
    const auto qRstdShape = context->GetOptionalInputShape(INPUT_Q_RSTD_IDX);
    const auto kRstdShape = context->GetOptionalInputShape(INPUT_K_RSTD_IDX);
    const auto betaRawShape = context->GetOptionalInputShape(INPUT_BETA_RAW_IDX);
    const auto qRstdDesc = context->GetOptionalInputDesc(INPUT_Q_RSTD_IDX);
    const auto kRstdDesc = context->GetOptionalInputDesc(INPUT_K_RSTD_IDX);
    const auto betaRawDesc = context->GetOptionalInputDesc(INPUT_BETA_RAW_IDX);
    if (useL2NormValue) {
        const gert::StorageShape *rstdShapes[] = {qRstdShape, kRstdShape};
        decltype(qRstdDesc) rstdDescs[] = {qRstdDesc, kRstdDesc};
        const char *rstdNames[] = {"q_rstd", "k_rstd"};
        for (size_t index = 0; index < 2; ++index) {
            if (rstdShapes[index] == nullptr || rstdDescs[index] == nullptr) {
                return InvalidParameter(context, rstdNames[index], "absent shape or descriptor",
                                        "present when use_qk_l2_norm_in_kernel=true");
            }
            if (ValidateShape(context, rstdNames[index], rstdShapes[index]->GetStorageShape(),
                              {tiling->B, tiling->HK, tiling->T}, INPUT_Q_RSTD_IDX + index) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (rstdDescs[index]->GetDataType() != ge::DT_FLOAT) {
                return InvalidParameter(context, std::string(rstdNames[index]) + ".dtype",
                                        std::to_string(static_cast<int>(rstdDescs[index]->GetDataType())), "FP32",
                                        ParameterKind::DTYPE);
            }
        }
    }
    if (useBetaSigmoidValue) {
        if (betaRawShape == nullptr || betaRawDesc == nullptr) {
            return InvalidParameter(context, "beta_raw", "absent shape or descriptor",
                                    "present when use_beta_sigmoid_in_kernel=true");
        }
        if (ValidateShape(context, "beta_raw.shape", betaRawShape->GetStorageShape(),
                          {tiling->B, tiling->HV, tiling->T}, INPUT_BETA_RAW_IDX) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (betaRawDesc->GetDataType() != betaDesc->GetDataType()) {
            return InvalidParameter(context, "beta_raw.dtype",
                                    std::to_string(static_cast<int>(betaRawDesc->GetDataType())),
                                    "beta.dtype=" + std::to_string(static_cast<int>(betaDesc->GetDataType())),
                                    ParameterKind::DTYPE);
        }
    }

    tiling->scale = scale == nullptr ? 1.0f : static_cast<float>(*scale);
    tiling->stateVFirst = stateVFirstValue ? 1 : 0;

    const uint32_t blockDim = std::max<uint32_t>(1, std::min<uint64_t>(aicCoreNum, tiling->taskNum));
    const int64_t maxDimension = std::max({tiling->K, tiling->V, tiling->chunkSize});
    if (!FitsProduct({tiling->totalChunkNum, tiling->HV, tiling->K, tiling->V, 128}) ||
        !FitsProduct({blockDim, tiling->HV, tiling->chunkSize, maxDimension, 256})) {
        return InvalidParameter(context, "state/workspace factors",
                                "state=" + ShapeString({tiling->totalChunkNum, tiling->HV, tiling->K, tiling->V, 128}) +
                                ", workspace=" + ShapeString({blockDim, tiling->HV,
                                                              tiling->chunkSize, maxDimension, 256}),
                                "positive factors with product <= INT64_MAX on A2/A3");
    }
    const GDN::A2FinalizeWorkspaceLayout layout(*tiling, blockDim);
    const uint64_t systemWorkspace = platform.GetLibApiWorkSpaceSize();
    if (layout.totalBytes > std::numeric_limits<size_t>::max() - systemWorkspace) {
        return InvalidParameter(context, "workspace bytes",
                                "operator=" + std::to_string(layout.totalBytes) +
                                ", system=" + std::to_string(systemWorkspace),
                                "operator + system <= SIZE_MAX=" + std::to_string(std::numeric_limits<size_t>::max()));
    }
    context->GetWorkspaceSizes(1)[0] = systemWorkspace + layout.totalBytes;
    const uint64_t baseTilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(qKey), static_cast<uint64_t>(gKey),
        static_cast<uint64_t>(useL2NormValue), static_cast<uint64_t>(useBetaSigmoidValue),
        static_cast<uint64_t>(useExp2Value));
    OP_CHECK_IF(baseTilingKey >= GDN::A2_FINALIZE_MHA_OWNER_KEY_MASK,
                OP_LOGE(context->GetNodeName(), "A2 Finalize base TilingKey overlaps the private MHA-owner bit."),
                return ge::GRAPH_FAILED);
    const bool enableMhaOwner = qKey == TPL_BF16 && tiling->K == K_SIZE_128 &&
        tiling->chunkSize == CHUNK_SIZE_64 && tiling->V == V_SIZE_128 && tiling->HK == tiling->HV;
    const uint64_t actualTilingKey = baseTilingKey |
        (enableMhaOwner ? GDN::A2_FINALIZE_MHA_OWNER_KEY_MASK : uint64_t{0});
    context->SetTilingKey(actualTilingKey);
    OP_LOGD(context->GetNodeName(),
            "GDN_FINALIZE_ARCH22_TILING actualKey=%" PRIu64 " owner=%u gKey=%d norm=%u sigmoid=%u exp2=%u",
            actualTilingKey, static_cast<unsigned>(enableMhaOwner), gKey,
            static_cast<unsigned>(useL2NormValue), static_cast<unsigned>(useBetaSigmoidValue),
            static_cast<unsigned>(useExp2Value));
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

} // namespace gdn_finalize_arch22
} // namespace
} // namespace optiling

#endif
