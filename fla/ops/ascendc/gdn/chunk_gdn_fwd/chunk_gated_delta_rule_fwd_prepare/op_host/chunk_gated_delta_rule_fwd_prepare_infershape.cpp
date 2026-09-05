/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "register/op_impl_registry.h"

namespace ops {
namespace {

constexpr int64_t IN_Q = 0;
constexpr int64_t IN_K = 1;
constexpr int64_t IN_V = 2;
constexpr int64_t IN_G = 3;
constexpr int64_t OUT_G = 0;
constexpr int64_t OUT_W = 1;
constexpr int64_t OUT_U = 2;
constexpr int64_t OUT_A = 3;
constexpr int64_t OUT_Q_HAT = 4;
constexpr int64_t OUT_K_HAT = 5;
constexpr int64_t OUT_Q_RSTD = 6;
constexpr int64_t OUT_K_RSTD = 7;
constexpr int64_t OUT_BETA = 8;

void SetShape3(gert::Shape *shape, int64_t d0, int64_t d1, int64_t d2)
{
    if (shape == nullptr) {
        return;
    }
    shape->SetDimNum(3);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
}

void SetShape4(gert::Shape *shape, int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    if (shape == nullptr) {
        return;
    }
    shape->SetDimNum(4);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
    shape->SetDim(3, d3);
}

void CopyShape(const gert::Shape *src, gert::Shape *dst)
{
    if (src == nullptr || dst == nullptr) {
        return;
    }
    dst->SetDimNum(src->GetDimNum());
    for (size_t i = 0; i < src->GetDimNum(); ++i) {
        dst->SetDim(i, src->GetDim(i));
    }
}

} // namespace

static ge::graphStatus InferShapeChunkGatedDeltaRuleFwdPrepare(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(IN_Q);
    const gert::Shape *kShape = context->GetInputShape(IN_K);
    const gert::Shape *vShape = context->GetInputShape(IN_V);
    const gert::Shape *gShape = context->GetInputShape(IN_G);
    if (qShape == nullptr || kShape == nullptr || vShape == nullptr || gShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (qShape->GetDimNum() != 4 || vShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const int64_t B = qShape->GetDim(0);
    const int64_t HK = qShape->GetDim(1);
    const int64_t T = qShape->GetDim(2);
    const int64_t K = qShape->GetDim(3);
    const int64_t HV = vShape->GetDim(1);
    const int64_t V = vShape->GetDim(3);

    int64_t chunkSize = 64;
    if (context->GetAttrs() != nullptr &&
        context->GetAttrs()->GetAttrPointer<int64_t>(0) != nullptr) {
        chunkSize = *context->GetAttrs()->GetAttrPointer<int64_t>(0);
    }

    CopyShape(gShape, context->GetOutputShape(OUT_G));
    SetShape4(context->GetOutputShape(OUT_W), B, HV, T, K);
    CopyShape(vShape, context->GetOutputShape(OUT_U));
    SetShape4(context->GetOutputShape(OUT_A), B, HV, T, chunkSize);
    CopyShape(qShape, context->GetOutputShape(OUT_Q_HAT));
    CopyShape(kShape, context->GetOutputShape(OUT_K_HAT));
    SetShape3(context->GetOutputShape(OUT_Q_RSTD), B, HK, T);
    SetShape3(context->GetOutputShape(OUT_K_RSTD), B, HK, T);
    CopyShape(gShape, context->GetOutputShape(OUT_BETA));
    (void)V;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkGatedDeltaRuleFwdPrepare(gert::InferDataTypeContext *context)
{
    const ge::DataType qDtype = context->GetInputDataType(IN_Q);
    const ge::DataType kDtype = context->GetInputDataType(IN_K);
    const ge::DataType vDtype = context->GetInputDataType(IN_V);
    context->SetOutputDataType(OUT_G, ge::DT_FLOAT);
    context->SetOutputDataType(OUT_W, kDtype);
    context->SetOutputDataType(OUT_U, vDtype);
    context->SetOutputDataType(OUT_A, kDtype);
    context->SetOutputDataType(OUT_Q_HAT, qDtype);
    context->SetOutputDataType(OUT_K_HAT, kDtype);
    context->SetOutputDataType(OUT_Q_RSTD, ge::DT_FLOAT);
    context->SetOutputDataType(OUT_K_RSTD, ge::DT_FLOAT);
    context->SetOutputDataType(OUT_BETA, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRuleFwdPrepare)
    .InferShape(InferShapeChunkGatedDeltaRuleFwdPrepare)
    .InferDataType(InferDataTypeChunkGatedDeltaRuleFwdPrepare);

} // namespace ops
