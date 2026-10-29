// -----------------------------------------------------------------------------------------------------------
// aclnnChunkKdaFwdFused — chunk32 大融合 KDA forward(issue #829)
// kernel 契约:
//   q/k:bf16 [H,T,K];v/g:[HV,T,K/V];beta:[HV,T];HV 为 H 的正整数倍
//   g/beta 必须同为 bf16 或 fp32;K=V=128,T>0(kernel 处理非 128 对齐尾块)
//   a_log:fp32 [HV] 原始值(核内 exp);dt_bias:fp32 [HV*K]
//   o:bf16 [HV,T,V]
//   initial_state/final_state(可空):fp32 [HV,K,V](dense)或 [nSeq*HV,K,V](varlen)
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
