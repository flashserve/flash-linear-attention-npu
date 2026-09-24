/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "merge_fwd_bwd_kernel_tiling.h"
#include "../../op_kernel/merge_fwd_bwd_kernel_struct.h"
#include "../../op_kernel/merge_fwd_bwd_kernel_tiling_key.h"

#include <algorithm>
#include <cstdlib>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

constexpr int64_t kK = 128;
constexpr int64_t kV = 128;
constexpr int64_t kVK = kV + kK;
constexpr size_t ATTR_FORWARD = 0;
constexpr size_t ATTR_RANK = 1;
constexpr size_t ATTR_N = 2;
constexpr size_t INPUT_AG_HM = 0;

ge::graphStatus Tiling4MergeFwdBwdKernel(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<MergeFwBwd::MergeFwdBwdKernelTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const auto *agDesc = context->GetInputDesc(INPUT_AG_HM);
    const auto *agShapePtr = context->GetInputShape(INPUT_AG_HM);
    const auto *hDesc = context->GetOutputDesc(0);
    const auto *hShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, agDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, agShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, hDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, hShapePtr);

    const ge::DataType dtype = agDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_BF16,
                OP_LOGE(context->GetNodeName(), "ag_hm dtype must be FP32 or BF16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hDesc->GetDataType() != dtype,
                OP_LOGE(context->GetNodeName(), "h dtype must match ag_hm."),
                return ge::GRAPH_FAILED);

    const gert::Shape ag = agShapePtr->GetStorageShape();
    OP_CHECK_IF(ag.GetDimNum() != 4,
                OP_LOGE(context->GetNodeName(), "ag_hm must be rank-4 [S,HV,K,V+K]."),
                return ge::GRAPH_FAILED);
    const int64_t S = ag.GetDim(0);
    const int64_t Hv = ag.GetDim(1);
    const int64_t K = ag.GetDim(2);
    const int64_t VK = ag.GetDim(3);
    OP_CHECK_IF(S < 1 || S > 1024 || Hv < 1 || Hv > 256,
                OP_LOGE(context->GetNodeName(), "S/HV out of range."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(K != kK || VK != kVK,
                OP_LOGE(context->GetNodeName(), "K must be 128 and last dim must be 256."),
                return ge::GRAPH_FAILED);

    const gert::Shape h = hShapePtr->GetStorageShape();
    OP_CHECK_IF(h.GetDimNum() != 3 || h.GetDim(0) != Hv || h.GetDim(1) != kK || h.GetDim(2) != kV,
                OP_LOGE(context->GetNodeName(), "h must be [HV,128,128]."),
                return ge::GRAPH_FAILED);

    auto *attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);
    const bool forward = *(attrPtr->GetAttrPointer<bool>(ATTR_FORWARD));
    const int64_t rank = *(attrPtr->GetAttrPointer<int64_t>(ATTR_RANK));
    const int64_t N = *(attrPtr->GetAttrPointer<int64_t>(ATTR_N));
    OP_CHECK_IF(N < 1 || N > S,
                OP_LOGE(context->GetNodeName(), "preOrPostNumRanks must be in [1, S]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rank < 0 || rank >= S,
                OP_LOGE(context->GetNodeName(), "rank must be in [0, S)."),
                return ge::GRAPH_FAILED);
    const bool unitWorld = (S == 1 && N == 1 && rank == 0);
    if (!unitWorld) {
        if (forward) {
            OP_CHECK_IF(rank < N,
                        OP_LOGE(context->GetNodeName(), "forward merge requires rank >= N."),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(rank + N >= S,
                        OP_LOGE(context->GetNodeName(), "backward merge requires rank + N < S."),
                        return ge::GRAPH_FAILED);
        }
    }
    tiling->S = S;
    tiling->Hv = Hv;
    tiling->K = kK;
    tiling->V = kV;
    tiling->rank = rank;
    tiling->N = N;
    tiling->forward = forward ? 1 : 0;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t coreNum = ascendcPlatform.GetCoreNum();
    const auto *compileInfo = context->GetCompileInfo<MergeFwdBwdKernelCompileInfo>();
    if (aicNum <= 2U && compileInfo != nullptr && compileInfo->aicNum > 2U) {
        aicNum = compileInfo->aicNum;
    }
    if (aivNum < 8U && compileInfo != nullptr && compileInfo->aivNum >= 8U) {
        aivNum = compileInfo->aivNum;
    }
    if (aicNum <= 2U) {
        if (aivNum >= 8U) {
            aicNum = aivNum / 2U;
        } else if (coreNum >= 8U) {
            aicNum = coreNum;
        } else {
            aicNum = 28U;
        }
    }
    if (aivNum < aicNum * 2U) {
        aivNum = aicNum * 2U;
    }
    // Fixpipe→UB: each AIV has one C slot. Default keep ≤4 heads/Mix core
    // (≤2/AIV). MERGE_FWD_BWD_USED_AIC pin is honored as-is for stress.
    uint32_t usedAic = static_cast<uint32_t>(std::min<int64_t>(
        static_cast<int64_t>(std::max(1U, aicNum)), std::max<int64_t>(1, Hv)));
    const uint32_t minAicForUb = static_cast<uint32_t>((std::max<int64_t>(1, Hv) + 3) / 4);
    if (usedAic < minAicForUb) {
        usedAic = std::min(minAicForUb, std::max(1U, aicNum));
    }
    const char *usedAicEnv = std::getenv("MERGE_FWD_BWD_USED_AIC");
    if (usedAicEnv != nullptr) {
        const int64_t pin = std::strtoll(usedAicEnv, nullptr, 10);
        if (pin > 0) {
            usedAic = static_cast<uint32_t>(std::min<int64_t>(
                static_cast<int64_t>(std::max(1U, aicNum)), pin));
        }
    }
    tiling->usedAic = static_cast<int64_t>(usedAic);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(usedAic * 2U, aicNum, aivNum);
    if (blockDim < usedAic) {
        blockDim = usedAic;
    }
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);

    using namespace MergeFwBwd;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        dtype == ge::DT_FLOAT ? MERGE_FWD_BWD_KERNEL_TPL_FP32 : MERGE_FWD_BWD_KERNEL_TPL_BF16);
    context->SetTilingKey(tilingKey);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    const uint64_t sysWs = static_cast<uint64_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    tiling->sysWorkspaceSize = static_cast<int64_t>(sysWs);
    // Internal FP32 scratch [3,HV,128,128] for H + M ping-pong (not an Op input).
    const uint64_t scratchWs = static_cast<uint64_t>(MergeFwBwd::kScratchBanks) *
                               static_cast<uint64_t>(std::max<int64_t>(1, Hv)) *
                               static_cast<uint64_t>(MergeFwBwd::kKvElems) * 4ULL;
    currentWorkspace[0] = sysWs + scratchWs;

    auto *rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    rawTiling->SetDataSize(sizeof(MergeFwBwd::MergeFwdBwdKernelTilingData));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParse4MergeFwdBwdKernel(gert::TilingParseContext *context)
{
    auto *compileInfo = context->GetCompiledInfo<MergeFwdBwdKernelCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->aicNum = 24;
    compileInfo->aivNum = 48;
    auto *platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        const uint32_t aicNum = platform.GetCoreNumAic();
        const uint32_t aivNum = platform.GetCoreNumAiv();
        if (aicNum > 2U) {
            compileInfo->aicNum = aicNum;
        }
        if (aivNum >= 8U) {
            compileInfo->aivNum = aivNum;
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MergeFwdBwdKernel)
    .Tiling(Tiling4MergeFwdBwdKernel)
    .TilingParse<MergeFwdBwdKernelCompileInfo>(TilingParse4MergeFwdBwdKernel);

} // namespace optiling
