// -----------------------------------------------------------------------------------------------------------
// chunk_kda_fwd_fused l0op 实现(chunk32 大融合,issue #829)
// -----------------------------------------------------------------------------------------------------------

#include "chunk_kda_fwd_fused.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

// l0op::Exp 在 CANN libopapi.so 有导出符号,仓库 vendor 头未含 exp.h,前置声明之。
namespace l0op {
const aclTensor *Exp(const aclTensor *x, aclOpExecutor *executor);
}

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaFwdFused);

const aclTensor *KdaChunkForwardFused(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g,
    const aclTensor *beta, const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *initialState,
    double scale, double lowerBound, const aclTensor *o, const aclTensor *finalState,
    const aclIntArray *cuSeqlens, double epsilon, aclOpExecutor *executor,
    const char *layout)
{
    L0_DFX(KdaChunkForwardFused, q, k, v, g, beta, aLog, dtBias, initialState, scale, lowerBound, o, finalState);

    // kernel 契约:GM 里放 exp(A_log);本层对外是原始 A_log,
    // 在这里补 device 侧逐点 exp,保证 aclnn 入口与 dispatch 路由行为一致。
    // KDA_FD32_SKIP_EXP=1 时跳过核外 exp,输入必须是已 exp 的 A_log。
    const aclTensor *aLogExp = aLog;
    if (getenv("KDA_FD32_SKIP_EXP") == nullptr || getenv("KDA_FD32_SKIP_EXP")[0] != '1') {
        aLogExp = l0op::Exp(aLog, executor);
        if (aLogExp == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "l0op::Exp on aLog failed.");
            return nullptr;
        }
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaFwdFused,
        OP_INPUT(q, k, v, g, beta, aLogExp, dtBias, initialState),
        OP_OUTPUT(o, finalState),
        OP_ATTR(scale, static_cast<float>(lowerBound), layout, cuSeqlens, static_cast<float>(epsilon)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaFwdFused failed.");
        return nullptr;
    }
    return o;
}

} // namespace l0op
