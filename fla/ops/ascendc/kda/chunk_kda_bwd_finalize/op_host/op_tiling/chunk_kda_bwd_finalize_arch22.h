#ifndef KDA_FINALIZE_ARCH22_HOST_H
#define KDA_FINALIZE_ARCH22_HOST_H

#include <cstring>
#include <limits>
#include "../../op_kernel/arch22/chunk_kda_bwd_finalize_common.h"

namespace optiling {
inline ge::graphStatus TilingFinalizeArch22(
    gert::TilingContext *context, const ChunkKdaBwdFinalizeTilingContext &ctx,
    const KDA::ChunkKdaBwdFinalizeTilingData &shape)
{
    using namespace KDA::Arch22;
    auto fail = [&]() { return ge::GRAPH_FAILED; };
    auto *t = context->GetTilingData<FinalizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, t);
    *t = {};
    t->batch = shape.B;
    t->heads = shape.NV;
    t->tokens = shape.T;
    t->chunks = shape.totalChunkNum;
    const uint64_t maxBytes = static_cast<uint64_t>(INT64_MAX) - ctx.sysWorkspaceSize;
    if (static_cast<uint64_t>(t->chunks) > maxBytes / t->heads / (PARTIAL_FLOATS * sizeof(float))) return fail();
    const uint64_t work = static_cast<uint64_t>(t->chunks) * ((t->heads + 1) / HEADS_PER_WORK);
    t->usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(ctx.aicCoreNum, work));
    if (!t->usedCoreNum) return fail();
    t->isVariable = shape.isVariable;
    t->outputTokenMajor = shape.outputTokenMajor;
    t->hasQkL2Norm = ctx.qRstd != nullptr;
    t->betaMode = shape.betaMode;
    t->scale = static_cast<float>(ctx.scale);
    t->lowerBound = static_cast<float>(ctx.lowerBound);
    t->partialOffset = static_cast<uint64_t>(t->usedCoreNum) * WS_CORE_BYTES;
    const uint64_t partialBytes = static_cast<uint64_t>(t->chunks) * t->heads * PARTIAL_FLOATS * sizeof(float);
    if (partialBytes > maxBytes - t->partialOffset) return fail();
    context->SetTilingKey(t->hasQkL2Norm ? 102 : 101);
    context->SetBlockDim(t->usedCoreNum);
    context->SetScheduleMode(1);
    context->GetWorkspaceSizes(1)[0] = ctx.sysWorkspaceSize + t->partialOffset + partialBytes;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
#endif
