#ifndef PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_FUSED_RECURRENT_RWKV8
#define PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_FUSED_RECURRENT_RWKV8

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor *FusedRecurrentRwkv8(const aclTensor *q, const aclTensor *w, const aclTensor *k, const aclTensor *v,
                                     const aclTensor *z, const aclTensor *b, const aclTensor *initialState,
                                     float scale, bool outputChunkState, bool outputSa, bool reverse,
                                     int64_t chunkLen, aclTensor *sOut, aclTensor *saOut,
                                     aclOpExecutor *executor);
}

#endif // PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_FUSED_RECURRENT_RWKV8
