#ifndef CHUNK_KDA_BWD_STAGES_H
#define CHUNK_KDA_BWD_STAGES_H

#include "../../../chunk_kda_bwd_prepare/op_host/op_api/chunk_kda_bwd_prepare.h"
#include "../../../chunk_kda_bwd_finalize/op_host/op_api/chunk_kda_bwd_finalize.h"
#include "../../../../gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_host/op_api/chunk_gated_delta_rule_bwd_dhu.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/cast.h"
#include "opdev/tensor_view_utils.h"

namespace {
// 仅创建视图。Dhu 使用带 batch 维的 BNSD，KDA 的 varlen 入口使用 NTD。
const aclTensor *KdaBwdBatchView(const aclTensor *tensor, bool packed, aclOpExecutor *executor)
{
    if (!packed) return tensor;
    op::Shape shape;
    shape.AppendDim(1);
    const auto &source = tensor->GetViewShape();
    for (size_t i = 0; i < source.GetDimNum(); ++i) shape.AppendDim(source.GetDim(i));
    return l0op::Reshape(tensor, shape, executor);
}

bool AppendKdaBwdStages(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk, const aclTensor *aqk, const aclTensor *akk,
    const aclTensor *w, const aclTensor *qg, const aclTensor *kg,
    const aclTensor *vNew, const aclTensor *h, const aclTensor *dO,
    const aclTensor *rawG, const aclTensor *aLog, const aclTensor *dtBias,
    const aclIntArray *cu, const aclIntArray *indices,
    double scale, int64_t chunkSize, bool safeGate, bool useGate, double lowerBound,
    const aclTensor *dq, const aclTensor *dk, const aclTensor *dv,
    const aclTensor *db, const aclTensor *dg, const aclTensor *dA, const aclTensor *dBias,
    aclOpExecutor *executor)
{
    const auto queryShape = q->GetViewShape();
    const auto valueShape = v->GetViewShape();
    const auto state = h->GetViewShape();
    const auto dtype = q->GetDataType();
    const auto *dAqk = executor->AllocTensor(aqk->GetViewShape(), op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    // value 梯度跟随 V 维，query 梯度跟随 K 维。
    const auto *dvLocal = executor->AllocTensor(valueShape, dtype, op::Format::FORMAT_ND);
    const auto *dqRaw = executor->AllocTensor(queryShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    const auto *dh = executor->AllocTensor(state, dtype, op::Format::FORMAT_ND);
    const auto *dvScan = executor->AllocTensor(valueShape, dtype, op::Format::FORMAT_ND);
    if (!dAqk || !dvLocal || !dqRaw || !dh || !dvScan) return false;
    const auto prepare = l0op::ChunkKdaBwdPrepare(
        aqk, vNew, dO, h, cu, indices, cu != nullptr ? "NTD" : "BNSD", scale, chunkSize, false,
        dAqk, dvLocal, dqRaw, executor);
    for (auto *output : prepare) if (!output) return false;
    const bool packed = cu != nullptr;
    const auto *q4 = KdaBwdBatchView(qg, packed, executor);
    const auto *k4 = KdaBwdBatchView(kg, packed, executor);
    const auto *w4 = KdaBwdBatchView(w, packed, executor);
    const auto *do4 = KdaBwdBatchView(dO, packed, executor);
    const auto *dv4 = KdaBwdBatchView(dvLocal, packed, executor);
    const auto *gk4 = KdaBwdBatchView(gk, packed, executor);
    const auto *dh5 = KdaBwdBatchView(dh, packed, executor);
    const auto *dvScan4 = KdaBwdBatchView(dvScan, packed, executor);
    if (!q4 || !k4 || !w4 || !do4 || !dv4 || !gk4 || !dh5 || !dvScan4) return false;
    const auto scan = l0op::ChunkGatedDeltaRuleBwdDhu(
        q4, k4, w4, do4, dv4, nullptr, gk4, nullptr, nullptr,
        cu, indices, scale, chunkSize, true, false, dh5, nullptr, dvScan4, executor);
    if (!scan[0] || !scan[2]) return false;
    if (useGate && rawG && rawG->GetDataType() != op::DataType::DT_FLOAT) {
        rawG = l0op::Cast(rawG, op::DataType::DT_FLOAT, executor);
        if (!rawG) return false;
    }
    const auto finalize = l0op::ChunkKdaBwdFinalize(
        q, k, v, gk, rawG, beta, aLog, dtBias, akk, vNew, h, dh, dvScan, dAqk, dqRaw,
        nullptr, nullptr, cu, indices, scale, lowerBound, chunkSize,
        safeGate, useGate, true, false, dq, dk, dv, db, dg, dA, dBias, executor);
    for (auto *output : finalize) if (!output) return false;
    return true;
}
} // namespace
#endif
