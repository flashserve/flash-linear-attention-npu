#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_TILING_ARCH22_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_TILING_ARCH22_H

#include "../op_tiling/chunk_gated_delta_rule_bwd_finalize_tiling.h"
#include "chunk_gated_delta_rule_bwd_finalize_validation_arch22.h"
#include "../../op_kernel/arch22/chunk_gated_delta_rule_bwd_finalize_workspace.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"
#include "platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include "op_common/log/log.h"
#include "../../../../../common/host_utils/log.h"

namespace optiling {
namespace {
namespace gdn_finalize_arch22 {

using namespace GDN;
using GDN::FinalizeHostArch22::ShapeString;

constexpr const char *OP_NAME = "ChunkGatedDeltaRuleBwdFinalize";

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

ge::graphStatus ValidateShape(gert::TilingContext *, const char *parameter,
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
        const std::string actualShape = std::string(parameter) + "=" + ShapeString(actual);
        const std::string expectedShape = ShapeString(expected);
        OP_LOGE_WITH_INVALID_INPUT_SHAPE(OP_NAME, inputIndex, actualShape.c_str(), expectedShape.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateShapes(gert::TilingContext *context,
                               GDN::ChunkGatedDeltaRuleBwdFinalizeTilingData &tiling,
                               bool stateVFirst)
{

    // 先从真实输入 shape 提取运行时规格，再校验目标 SoC 支持的档位；
    // 禁止用编译期常量代替 shape 推导结果写入 tiling data。
    const auto qShape = context->GetRequiredInputShape(INPUT_Q_IDX);
    const auto kShape = context->GetRequiredInputShape(INPUT_K_IDX);
    const auto vShape = context->GetRequiredInputShape(INPUT_V_IDX);
    const auto vNewShape = context->GetRequiredInputShape(INPUT_V_NEW_IDX);
    const auto dOShape = context->GetRequiredInputShape(INPUT_DO_IDX);
    const auto duShape = context->GetRequiredInputShape(INPUT_DU_IDX);
    const auto gShape = context->GetRequiredInputShape(INPUT_G_IDX);
    const auto betaShape = context->GetRequiredInputShape(INPUT_BETA_IDX);
    const auto hShape = context->GetRequiredInputShape(INPUT_H_IDX);
    const auto dhShape = context->GetRequiredInputShape(INPUT_DH_IDX);
    const auto aShape = context->GetRequiredInputShape(INPUT_A_IDX);
    if (!CheckRank(qShape, 4) || !CheckRank(kShape, 4) || !CheckRank(vShape, 4) ||
        !CheckRank(vNewShape, 4) || !CheckRank(dOShape, 4) || !CheckRank(duShape, 4) ||
        !CheckRank(gShape, 3) || !CheckRank(betaShape, 3) || !CheckRank(hShape, 5) ||
        !CheckRank(dhShape, 5) || !CheckRank(aShape, 4)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(context->GetNodeName(), "q,k,v,vNew,dO,du,g,beta,h,dh,A",
            "rank mismatch", "q/k/v/vNew/dO/du/A rank-4, g/beta rank-3, h/dh rank-5");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape q = qShape->GetStorageShape();
    const gert::Shape k = kShape->GetStorageShape();
    const gert::Shape v = vShape->GetStorageShape();
    const gert::Shape vNew = vNewShape->GetStorageShape();
    const gert::Shape dO = dOShape->GetStorageShape();
    const gert::Shape du = duShape->GetStorageShape();
    const gert::Shape a = aShape->GetStorageShape();
    const gert::Shape h = hShape->GetStorageShape();
    const gert::Shape dh = dhShape->GetStorageShape();

    tiling.B = q.GetDim(0);
    tiling.HK = q.GetDim(1);
    tiling.T = q.GetDim(2);
    tiling.K = q.GetDim(3);
    tiling.HV = v.GetDim(1);
    tiling.V = v.GetDim(3);
    tiling.chunkSize = a.GetDim(3);
    if (tiling.B <= 0 || tiling.HK <= 0 || tiling.HV <= 0 || tiling.T <= 0) {

        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "B,HK,HV,T", "non-positive", "positive");
        return ge::GRAPH_FAILED;
    }
    if (tiling.K != K_SIZE_128 ||
        (tiling.V != V_SIZE_128 && tiling.V != V_SIZE_256)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "K,V", "unsupported", "K=128, V=128/256 on A2/A3");
        return ge::GRAPH_FAILED;
    }
    if (tiling.chunkSize != CHUNK_SIZE_64 &&
        tiling.chunkSize != CHUNK_SIZE_128) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "chunkSize", std::to_string(tiling.chunkSize), "64 or 128 on A2/A3");
        return ge::GRAPH_FAILED;
    }
    if (tiling.HV % tiling.HK != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "HV,HK", "not divisible", "HV % HK == 0");

        return ge::GRAPH_FAILED;
    }
    tiling.headRatio = tiling.HV / tiling.HK;
    if (tiling.headRatio < 1) {

        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "HV/HK", std::to_string(tiling.headRatio), "positive");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(!FitsProduct({tiling.B, tiling.HV, tiling.T,
                              std::max({tiling.K, tiling.V, tiling.chunkSize}), 128}),
                OP_LOGE(context->GetNodeName(), "Input dimensions overflow the internal address range."),
                return ge::GRAPH_FAILED);

    if (k.GetDim(0) != tiling.B || k.GetDim(1) != tiling.HK ||
        k.GetDim(2) != tiling.T || k.GetDim(3) != tiling.K) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(context->GetNodeName(), "q,k", "mismatch", "same shape");
        return ge::GRAPH_FAILED;
    }
    if (v.GetDim(0) != tiling.B || v.GetDim(2) != tiling.T ||
        vNew.GetDim(0) != tiling.B || vNew.GetDim(1) != tiling.HV ||
        vNew.GetDim(2) != tiling.T || vNew.GetDim(3) != tiling.V ||
        dO.GetDim(0) != tiling.B || dO.GetDim(1) != tiling.HV ||
        dO.GetDim(2) != tiling.T || dO.GetDim(3) != tiling.V ||
        du.GetDim(0) != tiling.B || du.GetDim(1) != tiling.HV ||
        du.GetDim(2) != tiling.T || du.GetDim(3) != tiling.V) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(context->GetNodeName(), "v,vNew,dO,du", "mismatch", "[B, HV, T, V]");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape g = gShape->GetStorageShape();
    const gert::Shape beta = betaShape->GetStorageShape();
    if (g.GetDim(0) != tiling.B || g.GetDim(1) != tiling.HV || g.GetDim(2) != tiling.T ||
        beta.GetDim(0) != tiling.B || beta.GetDim(1) != tiling.HV || beta.GetDim(2) != tiling.T) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(context->GetNodeName(), "g,beta", "mismatch", "[B, HV, T]");
        return ge::GRAPH_FAILED;
    }
    if (a.GetDim(0) != tiling.B || a.GetDim(1) != tiling.HV || a.GetDim(2) != tiling.T) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "A", "mismatch", "[B, HV, T, BT]");

        return ge::GRAPH_FAILED;
    }
    const int64_t stateDim3 = stateVFirst ? tiling.V : tiling.K;
    const int64_t stateDim4 = stateVFirst ? tiling.K : tiling.V;

    if (h.GetDim(0) != tiling.B || h.GetDim(2) != tiling.HV ||
        h.GetDim(3) != stateDim3 || h.GetDim(4) != stateDim4 ||
        dh.GetDim(0) != tiling.B || dh.GetDim(2) != tiling.HV ||
        dh.GetDim(1) != h.GetDim(1) || dh.GetDim(3) != stateDim3 ||
        dh.GetDim(4) != stateDim4) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(context->GetNodeName(), "h,dh", "state dim mismatch", "match state_v_first layout");
        return ge::GRAPH_FAILED;

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

        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "chunkSize", std::to_string(actualChunkSize), "match A last dim");
        return ge::GRAPH_FAILED;
    }
    if (scale != nullptr &&
        (!std::isfinite(*scale) || !std::isfinite(static_cast<float>(*scale)))) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "scale", "non-finite", "finite in FP32");
        return ge::GRAPH_FAILED;
    }
    if (useGateValue) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "useGateInKernel", "true", "false");

        return ge::GRAPH_FAILED;
    }
    const auto cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS_IDX);
    const auto chunkIndicesShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
    if ((cuShape == nullptr) != (chunkIndicesShape == nullptr)) {

        OP_LOGE_LIBOPAPI_REPORT_COMPAT(context->GetNodeName(), "cu_seqlens,chunk_indices");

        return ge::GRAPH_FAILED;
    }
    tiling->isVariable = cuShape != nullptr ? 1 : 0;
    if (tiling->isVariable != 0) {

        if (!CheckRank(cuShape, 1) || !CheckRank(chunkIndicesShape, 1) || tiling->B != 1) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "B,cu_seqlens,chunk_indices",
                "invalid", "varlen B=1, rank-1 metadata");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape cu = cuShape->GetStorageShape();
        const gert::Shape chunks = chunkIndicesShape->GetStorageShape();
        // aclIntArray 转成的 chunk_indices 是一维扁平数组，按
        // [seqIdx0, chunkIdx0, seqIdx1, chunkIdx1, ...] 两两分组。
        if (cu.GetDim(0) < 2 || chunks.GetDim(0) <= 0 || chunks.GetDim(0) % 2 != 0) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "cu_seqlens,chunk_indices", "invalid", "valid varlen index shapes");

            return ge::GRAPH_FAILED;
        }
        tiling->seqNum = cu.GetDim(0) - 1;
        tiling->totalChunkNum = chunks.GetDim(0) / 2;
        tiling->chunkNumForT = 0;
    } else {
        const uint64_t lastStart = static_cast<uint64_t>((tiling->T - 1) / tiling->chunkSize * tiling->chunkSize);
        if (!GDN::FinalizeHostArch22::ChunkEndFitsUint32(lastStart, static_cast<uint64_t>(tiling->chunkSize))) {
            const std::string actual = "T=" + std::to_string(tiling->T) +
                ", last chunk start=" + std::to_string(lastStart) + ", BT=" + std::to_string(tiling->chunkSize);
            OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "T/BT", actual.c_str(),
                                      "last chunk start + BT <= UINT32_MAX on A2/A3");
            return ge::GRAPH_FAILED;
        }
        tiling->seqNum = tiling->B;
        tiling->chunkNumForT = CeilDiv(tiling->T, tiling->chunkSize);
        tiling->totalChunkNum = tiling->B * tiling->chunkNumForT;
    }
    const auto hShape = context->GetRequiredInputShape(INPUT_H_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hShape);
    const int64_t stateChunkNum = tiling->isVariable != 0 ? tiling->totalChunkNum : tiling->chunkNumForT;
    if (hShape->GetStorageShape().GetDim(1) != stateChunkNum) {

        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "h,dh", "NT mismatch", "match chunk schedule");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(std::max({tiling->B, tiling->HK, tiling->HV, tiling->T,
                              tiling->totalChunkNum, tiling->seqNum}) >
                        static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                OP_LOGE(context->GetNodeName(), "A2 dimensions exceed the uint32 scheduling range."),
                return ge::GRAPH_FAILED);


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

    const int betaKey = DTypeKey(betaDesc->GetDataType());
    if (qKey != TPL_BF16 ||
        (gKey != TPL_BF16 && gKey != TPL_FP32) ||
        (betaKey != TPL_BF16 && betaKey != TPL_FP32)) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(context->GetNodeName(), "q,g,beta", "unsupported", "q BF16, g/beta BF16 or FP32");
        return ge::GRAPH_FAILED;
    }
    if (gDesc->GetDataType() != betaDesc->GetDataType()) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(context->GetNodeName(), "g,beta", "dtype mismatch", "same dtype");
        return ge::GRAPH_FAILED;
    }
    if (kDesc->GetDataType() != qDesc->GetDataType() ||
        vDesc->GetDataType() != qDesc->GetDataType() ||
        vNewDesc->GetDataType() != qDesc->GetDataType() ||
        dODesc->GetDataType() != qDesc->GetDataType() ||
        duDesc->GetDataType() != qDesc->GetDataType() ||
        hDesc->GetDataType() != qDesc->GetDataType() ||
        dhDesc->GetDataType() != qDesc->GetDataType() ||
        aDesc->GetDataType() != qDesc->GetDataType()) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(context->GetNodeName(), "q,k,v,vNew,dO,du,h,dh,A", "dtype mismatch", "BF16");

        return ge::GRAPH_FAILED;
    }

    // 可选输入按实际启用的 Stage 15/Stage 12 功能校验。
    const auto qRstdShape = context->GetOptionalInputShape(INPUT_Q_RSTD_IDX);
    const auto kRstdShape = context->GetOptionalInputShape(INPUT_K_RSTD_IDX);
    const auto betaRawShape = context->GetOptionalInputShape(INPUT_BETA_RAW_IDX);
    const auto qRstdDesc = context->GetOptionalInputDesc(INPUT_Q_RSTD_IDX);
    const auto kRstdDesc = context->GetOptionalInputDesc(INPUT_K_RSTD_IDX);
    const auto betaRawDesc = context->GetOptionalInputDesc(INPUT_BETA_RAW_IDX);
    if (useL2NormValue) {

        if (!CheckRank(qRstdShape, 3) || !CheckRank(kRstdShape, 3) ||
            qRstdDesc == nullptr || kRstdDesc == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(context->GetNodeName(), "q_rstd,k_rstd");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape qRstd = qRstdShape->GetStorageShape();
        const gert::Shape kRstd = kRstdShape->GetStorageShape();
        OP_CHECK_IF(qRstd.GetDim(0) != tiling->B || qRstd.GetDim(1) != tiling->HK ||
                        qRstd.GetDim(2) != tiling->T ||
                        kRstd.GetDim(0) != tiling->B || kRstd.GetDim(1) != tiling->HK ||
                        kRstd.GetDim(2) != tiling->T ||
                        qRstdDesc->GetDataType() != ge::DT_FLOAT ||
                        kRstdDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context->GetNodeName(),
                            "q_rstd and k_rstd must be fp32 [B, HK, T]."),
                    return ge::GRAPH_FAILED);
    }
    if (useBetaSigmoidValue) {
        if (!CheckRank(betaRawShape, 3) || betaRawDesc == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(context->GetNodeName(), "beta_raw");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape betaRaw = betaRawShape->GetStorageShape();
        OP_CHECK_IF(betaRaw.GetDim(0) != tiling->B || betaRaw.GetDim(1) != tiling->HV ||
                        betaRaw.GetDim(2) != tiling->T ||
                        betaRawDesc->GetDataType() != betaDesc->GetDataType(),
                    OP_LOGE(context->GetNodeName(),
                            "beta_raw must match beta shape and dtype."),
                    return ge::GRAPH_FAILED);

    }

    tiling->scale = scale == nullptr ? 1.0f : static_cast<float>(*scale);
    tiling->stateVFirst = stateVFirstValue ? 1 : 0;

    const uint32_t blockDim = std::max<uint32_t>(1, std::min<uint64_t>(aicCoreNum, tiling->taskNum));
    const int64_t maxDimension = std::max({tiling->K, tiling->V, tiling->chunkSize});
    if (!FitsProduct({tiling->totalChunkNum, tiling->HV, tiling->K, tiling->V, 128}) ||
        !FitsProduct({blockDim, tiling->HV, tiling->chunkSize, maxDimension, 256})) {
        const std::string actual =
            "state=" + ShapeString({tiling->totalChunkNum, tiling->HV, tiling->K, tiling->V, 128}) +
            ", workspace=" + ShapeString({blockDim, tiling->HV, tiling->chunkSize, maxDimension, 256});
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "state/workspace factors", actual.c_str(),
                                  "positive factors with product <= INT64_MAX on A2/A3");
        return ge::GRAPH_FAILED;
    }
    const GDN::A2FinalizeWorkspaceLayout layout(*tiling, blockDim);
    const uint64_t systemWorkspace = platform.GetLibApiWorkSpaceSize();
    if (layout.totalBytes > std::numeric_limits<size_t>::max() - systemWorkspace) {
        const std::string actual = "operator=" + std::to_string(layout.totalBytes) +
            ", system=" + std::to_string(systemWorkspace);
        const std::string expected =
            "operator + system <= SIZE_MAX=" + std::to_string(std::numeric_limits<size_t>::max());
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "workspace bytes", actual.c_str(), expected.c_str());
        return ge::GRAPH_FAILED;
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
