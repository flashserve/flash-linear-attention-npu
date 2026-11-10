/*!
 * \file fused_recurrent_rwkv8_infershape.cpp
 * \brief
 */
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace gert;
namespace ops {

const size_t Q_INDEX = 0;
const size_t V_INDEX = 3;
const size_t OUT_INDEX = 0;
const size_t S_INDEX = 1;
const size_t SA_INDEX = 2;

const size_t ATTR_SCALE_INDEX = 0;
const size_t ATTR_OUTPUT_CHUNK_STATE_INDEX = 1;
const size_t ATTR_OUTPUT_SA_INDEX = 2;
const size_t ATTR_REVERSE_INDEX = 3;
const size_t ATTR_CHUNK_LEN_INDEX = 4;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;
const size_t DIM_4 = 4;

const size_t IO_DIM_NUM = 4;
const size_t S_DIM_NUM = 5;
const int64_t DEFAULT_CHUNK_LEN = 16;   // 对齐官方 wkv7_cuda.cu backward 的 chunk 重建粒度

static bool GetBoolAttr(InferShapeContext *context, size_t index)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    const bool *ptr = attrs->GetAttrPointer<bool>(index);
    return ptr != nullptr ? *ptr : false;
}

static int64_t GetChunkLenAttr(InferShapeContext *context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return DEFAULT_CHUNK_LEN;
    }
    const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_LEN_INDEX);
    return (ptr != nullptr && *ptr > 0) ? *ptr : DEFAULT_CHUNK_LEN;
}

static ge::graphStatus InferShapeFusedRecurrentRwkv8(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedRecurrentRwkv8", "inference context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto shapeQ = context->GetInputShape(Q_INDEX);              // q (B,H,T,K)
    auto shapeV = context->GetInputShape(V_INDEX);              // v (B,H,T,V)
    auto shapeOut = context->GetOutputShape(OUT_INDEX);         // o (B,H,T,V)
    auto shapeS = context->GetOutputShape(S_INDEX);             // s (B,H,T//chunk_len,K,V)
    auto shapeSa = context->GetOutputShape(SA_INDEX);           // sa (B,H,T,V)
    if (shapeQ == nullptr || shapeV == nullptr || shapeOut == nullptr ||
        shapeS == nullptr || shapeSa == nullptr) {
        OP_LOGE(opName, "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }

    shapeOut->SetDimNum(IO_DIM_NUM);
    for (size_t i = 0; i < IO_DIM_NUM; i++) {
        shapeOut->SetDim(i, shapeV->GetDim(i));   // o 形状同 v
    }

    // 训练预埋输出：attr 关闭时给零尺寸占位（kernel 侧 flag 跳过写出）
    if (GetBoolAttr(context, ATTR_OUTPUT_CHUNK_STATE_INDEX)) {
        const int64_t chunkLen = GetChunkLenAttr(context);
        shapeS->SetDimNum(S_DIM_NUM);
        shapeS->SetDim(DIM_0, shapeQ->GetDim(DIM_0));               // B
        shapeS->SetDim(DIM_1, shapeQ->GetDim(DIM_1));               // H
        shapeS->SetDim(DIM_2, shapeQ->GetDim(DIM_2) / chunkLen);    // T//chunk_len（floor）
        shapeS->SetDim(DIM_3, shapeQ->GetDim(DIM_3));               // K
        shapeS->SetDim(DIM_4, shapeV->GetDim(DIM_3));               // V
    } else {
        shapeS->SetDimNum(1);
        shapeS->SetDim(DIM_0, 0);
    }

    if (GetBoolAttr(context, ATTR_OUTPUT_SA_INDEX)) {
        shapeSa->SetDimNum(IO_DIM_NUM);
        for (size_t i = 0; i < IO_DIM_NUM; i++) {
            shapeSa->SetDim(i, shapeV->GetDim(i));   // sa 与 o 同 shape
        }
    } else {
        shapeSa->SetDimNum(1);
        shapeSa->SetDim(DIM_0, 0);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedRecurrentRwkv8(gert::InferDataTypeContext *context)
{
    // o 跟随 q 的 dtype（fp16/bf16/fp32）；s/sa 恒 fp32（state 不降精度）
    context->SetOutputDataType(OUT_INDEX, context->GetInputDataType(Q_INDEX));
    context->SetOutputDataType(S_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(SA_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedRecurrentRwkv8)
    .InferShape(InferShapeFusedRecurrentRwkv8)
    .InferDataType(InferDataTypeFusedRecurrentRwkv8);
} // namespace ops
