// -----------------------------------------------------------------------------------------------------------
// aclnnChunkKdaFwdFused — chunk32 大融合 KDA forward(issue #829)
// kernel 契约:
//   q/k/v/g/beta:bf16,[H,T,K] 头连续(3 维),K=V=128,T%128==0
//   a_log:fp32 [H] 原始值(核内 exp);dt_bias:fp32 [H*K]
//   o:bf16 [H,T,128]
//   initial_state(可空):fp32 [H,K,V] ND(state_v_first=false 形态)
//   final_state(可空):fp32 [H,K,V],非空即请求末组状态写回
//   varlen 由 aclnnChunkKdaFwdV2 入口支持,布局为 TND 或 NTD
// -----------------------------------------------------------------------------------------------------------
#ifndef OP_API_INC_ACLNN_CHUNK_KDA_FWD_FUSED_H
#define OP_API_INC_ACLNN_CHUNK_KDA_FWD_FUSED_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkKdaFwdFusedGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLog,
    const aclTensor *dtBias,
    const aclTensor *initialState,
    double scale,
    double lowerBound,
    const aclTensor *o,
    const aclTensor *finalState,
    const aclIntArray *cuSeqlens,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaFwdFused(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
