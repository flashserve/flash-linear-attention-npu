// -----------------------------------------------------------------------------------------------------------
// chunk_kda_fwd_fused l0op 声明(chunk32 大融合,issue #829)
// -----------------------------------------------------------------------------------------------------------
#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_FWD_FUSED_OP_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_FWD_FUSED_OP_H

#include "opdev/op_executor.h"

namespace l0op {
// initialState 可空:fp32 [H,K,V];finalState 可空:非空即请求写出 fp32 [H,K,V]。
// cuSeqlens 可空:非空即 varlen 单启动(任务 = seq*HV + head 拍平,
// 序列边界经 attr 进 tiling,kernel 逐任务解析)。
// epsilon:q/k L2 归一化的 epsilon,须有限且大于 0。
const aclTensor *KdaChunkForwardFused(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g,
    const aclTensor *beta, const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *initialState,
    double scale, double lowerBound, const aclTensor *o, const aclTensor *finalState,
    const aclIntArray *cuSeqlens, double epsilon, aclOpExecutor *executor,
    const char *layout = "TND");
} // namespace l0op

#endif
