#ifndef OP_API_ACLNN_FUSED_RECURRENT_RWKV8_H
#define OP_API_ACLNN_FUSED_RECURRENT_RWKV8_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFusedRecurrentRwkv8 的第一段接口，根据具体的计算流程，计算workspace大小。
 *
 * 算子功能：RWKV-v8 (WKV7) fused recurrent 前向递推，逐 token 更新 (B,H,K,V) 状态账本（接口朝向）并输出 o（K/V 可不同）。
 *   sa    = state @ z_t
 *   state = state * decay_t[None,:] + sa[:,None] * b_t[None,:] + v_t[:,None] * k_t[None,:]
 *   o_t   = state @ (q_t * scale)        decay = exp(-exp(w))
 *
 * @param [in] q: 数据类型支持：float16/bfloat16/float32。shape (B,H,T,K)。
 * @param [in] w: 数据类型支持：同 q。shape (B,H,T,K)，log 域衰减系数。
 * @param [in] k: 数据类型支持：同 q。shape (B,H,T,K)。
 * @param [in] v: 数据类型支持：同 q。shape (B,H,T,V)。
 * @param [in] z: 数据类型支持：同 q。shape (B,H,T,K)。
 * @param [in] b: 数据类型支持：同 q。shape (B,H,T,K)。
 * @param [in] initialState: 数据类型支持：float32。shape (B,H,K,V)，可空（空表示零初态）。
 *   朝向与 sOut 一致（= 内核账本 Sᵀ 原样）。
 * @param [in] scale: 数据类型支持：float32。q 的缩放系数。
 * @param [in] reverse: true 表示 T 维倒序递推（对齐 fla reverse）。
 * @param [in] outputChunkState: true 表示产出 chunk 快照 s（训练预埋，对齐官方 CUDA s_）。
 * @param [in] outputSa: true 表示产出每 token 的 sa（训练预埋，对齐官方 CUDA sa_）。
 * @param [in] chunkLen: s 快照间隔，必须 >= 1，默认 16（对齐官方 wkv7_cuda.cu backward 的
 *   chunk 重建粒度；非 16 值与官方 backward 不兼容）。
 * @param [out] out: 数据类型支持：同 q。shape (B,H,T,V)，必填。
 * @param [out] sOut: 数据类型支持：float32。shape (B,H,T//chunkLen,K,V)，outputChunkState=true 时必填。
 *   布局为官方 CUDA 转置口径（快照 [j][i] = S[i][j]）。
 * @param [out] saOut: 数据类型支持：float32。shape (B,H,T,V)，outputSa=true 时必填。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnFusedRecurrentRwkv8GetWorkspaceSize(
    const aclTensor *q, const aclTensor *w, const aclTensor *k, const aclTensor *v, const aclTensor *z,
    const aclTensor *b, const aclTensor *initialState, float scale, bool reverse, bool outputChunkState,
    bool outputSa, int64_t chunkLen, aclTensor *out, aclTensor *sOut, aclTensor *saOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFusedRecurrentRwkv8 的第二段接口，执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnFusedRecurrentRwkv8(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_FUSED_RECURRENT_RWKV8_H
