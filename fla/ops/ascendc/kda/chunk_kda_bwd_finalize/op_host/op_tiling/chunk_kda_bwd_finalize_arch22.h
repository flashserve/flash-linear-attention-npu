#ifndef KDA_FINALIZE_ARCH22_HOST_H
#define KDA_FINALIZE_ARCH22_HOST_H
#include "../../op_kernel/arch22/tiling.h"
#include "../../../chunk_kda_bwd/op_host/chunk_kda_bwd_c_tiling_processor.h"

namespace optiling {
inline ge::graphStatus TilingFinalizeArch22(
    gert::TilingContext *context, const ChunkKdaBwdFinalizeTilingContext &ctx)
{
    if (ctx.chunkSize != 64 || !ctx.useExp2 || ctx.stateVFirst || ctx.qRstd || ctx.kRstd) {
        OP_LOGE(context->GetNodeName(), "arch22 requires C=64, exp2, K-first state and external L2Norm backward");
        return ge::GRAPH_FAILED;
    }
    auto *t = context->GetTilingData<KDA::ChunkKdaBwdFinalizeArch22TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, t);
    *t = {};
    ChunkKdaBwdCTilingContext c{};
    c.nodeName = ctx.nodeName;
    // 显式重排两套已有接口，Phase C 的数学实现与资源布局不变。
    constexpr size_t map[] = {
        INPUT_Q, INPUT_K, INPUT_V, INPUT_V_NEW, INPUT_GK, INPUT_BETA,
        INPUT_AKK, INPUT_H, INPUT_DH, INPUT_DV_SCAN, INPUT_DQ_RAW, INPUT_D_AQK,
        INPUT_RAW_G, INPUT_A_LOG, INPUT_DT_BIAS, INPUT_CU_SEQLENS, INPUT_CHUNK_INDICES};
    gert::StorageShape canonical[KDA_C_INPUT_COUNT];
    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); ++i) {
        const auto *shape = i < KDA_C_REQUIRED_INPUT_COUNT ?
            context->GetRequiredInputShape(map[i]) : context->GetOptionalInputShape(map[i]);
        // 按 IR 输入编号查询，不能把缺省可选输入后的编号当成物理槽位。
        const auto *desc = context->GetOptionalInputDesc(map[i]);
        if (!shape || !desc) continue;
        // 子算子可能继承展平的 OriginShape；接口契约使用实际连续存储 shape。
        canonical[i].MutableOriginShape() = shape->GetStorageShape();
        canonical[i].MutableStorageShape() = shape->GetStorageShape();
        c.shapes[i] = &canonical[i];
        c.types[i] = desc->GetDataType();
    }
    c.scale = static_cast<float>(ctx.scale);
    c.chunkSize = 64;
    c.safeGate = ctx.safeGate;
    c.useGateInKernel = ctx.useGateInKernel;
    c.lowerBound = static_cast<float>(ctx.lowerBound);
    c.validateIntermediateShapes = true;
    c.aicCoreNum = ctx.aicCoreNum;
    c.systemWorkspaceSize = 0;
    ChunkKdaBwdCTilingProcessor processor(c, t->phaseC);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    // 内部 Gate 交给下方 GatePost；外部 Gate 仍需保留原 Bwd 的 reverse cumsum。
    t->phaseC.deferGatePost = ctx.useGateInKernel ? 1 : 0;
    uint64_t cursor = 0;
    auto reserve = [&cursor](uint64_t bytes) {
        const auto offset = cursor;
        cursor = (cursor + bytes + 511U) / 512U * 512U;
        return offset;
    };
    const uint64_t tokens = static_cast<uint64_t>(t->phaseC.batch) * t->phaseC.headNum * t->phaseC.seqlen;
    t->fp32Output = context->GetOutputDesc(0)->GetDataType() == ge::DT_FLOAT;
    if (context->GetOutputDesc(1)->GetDataType() != context->GetOutputDesc(0)->GetDataType() ||
        context->GetOutputDesc(3)->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "arch22 requires matching dq/dk dtypes and FP32 dbeta");
        return ge::GRAPH_FAILED;
    }
    t->safeGate = ctx.safeGate;
    t->dqOffset = reserve(t->fp32Output ? 0 : tokens * 128U * sizeof(float));
    t->dkOffset = reserve(t->fp32Output ? 0 : tokens * 128U * sizeof(float));
    t->dAkkOffset = reserve(tokens * 64U * sizeof(float));
    t->dAqkOffset = reserve(tokens * 64U * sizeof(float));
    t->phaseWorkspaceOffset = cursor;
    context->SetTilingKey(101);
    context->SetBlockDim(processor.GetBlockDim());
    context->SetScheduleMode(1);
    context->GetWorkspaceSizes(1)[0] = ctx.sysWorkspaceSize + cursor + processor.GetWorkspaceSize();
    return ge::GRAPH_SUCCESS;
}
}
#endif
