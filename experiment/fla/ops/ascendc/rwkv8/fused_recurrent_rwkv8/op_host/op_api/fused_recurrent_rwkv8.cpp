/*!
 * \file fused_recurrent_rwkv8.cpp
 * \brief L0 launcher for FusedRecurrentRwkv8 (WKV7 fused recurrent forward).
 */
#include "fused_recurrent_rwkv8.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedRecurrentRwkv8);

const aclTensor *FusedRecurrentRwkv8(const aclTensor *q, const aclTensor *w, const aclTensor *k, const aclTensor *v,
                                     const aclTensor *z, const aclTensor *b, const aclTensor *initialState,
                                     float scale, bool outputChunkState, bool outputSa, bool reverse,
                                     int64_t chunkLen, aclTensor *sOut, aclTensor *saOut,
                                     aclOpExecutor *executor)
{
    L0_DFX(FusedRecurrentRwkv8, q, w, k, v, z, b, initialState, scale, outputChunkState, outputSa, reverse,
           chunkLen, sOut, saOut);

    // out dtype 跟随 q（fp16/bf16/fp32）；sOut/saOut 由 L2 恒分配传入
    auto out = executor->AllocTensor(q->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out AllocTensor failed."),
             return nullptr);

    // infershape
    auto ret = INFER_SHAPE(FusedRecurrentRwkv8,
                           OP_INPUT(q, w, k, v, z, b, initialState),
                           OP_OUTPUT(out, sOut, saOut),
                           OP_ATTR(scale, outputChunkState, outputSa, reverse, chunkLen));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "FusedRecurrentRwkv8 InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedRecurrentRwkv8,
                                      OP_INPUT(q, w, k, v, z, b, initialState),
                                      OP_OUTPUT(out, sOut, saOut),
                                      OP_ATTR(scale, outputChunkState, outputSa, reverse, chunkLen));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "FusedRecurrentRwkv8 ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return out;
}
} // namespace l0op
